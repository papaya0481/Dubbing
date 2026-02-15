import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import textgrid

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算生成语音与原语音的 word-level 对齐 MSE 并绘制分布图")
    parser.add_argument("--manifest", type=str, required=True, help="生成脚本输出的 generation_manifest.jsonl")
    parser.add_argument("--generated-aligned-dir", type=str, required=True, help="生成音频对应的 TextGrid 目录")
    parser.add_argument("--output-dir", type=str, default="/home/ruixin/Dubbing/mel_convert/distribution/results")
    parser.add_argument("--tier", type=str, default="words")
    parser.add_argument("--mode", choices=["boundary", "center", "duration"], default="boundary")
    parser.add_argument("--ignore-empty", action="store_true", default=True)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_tier(tg: textgrid.TextGrid, tier_name: str):
    try:
        return tg.getFirst(tier_name)
    except Exception:
        for candidate in tg.tiers:
            if tier_name.lower() in candidate.name.lower():
                return candidate
    raise ValueError(f"TextGrid 中找不到 tier: {tier_name}")


def read_word_intervals(path: Path, tier_name: str, ignore_empty: bool = True) -> list[tuple[float, float, str]]:
    tg = textgrid.TextGrid.fromFile(str(path))
    tier = get_tier(tg, tier_name)
    intervals: list[tuple[float, float, str]] = []
    for iv in tier:
        mark = str(iv.mark).strip()
        if ignore_empty and not mark:
            continue
        intervals.append((float(iv.minTime), float(iv.maxTime), mark))
    if not intervals:
        raise ValueError(f"无有效 intervals: {path}")
    return intervals


def normalize_mark(mark: str) -> str:
    return "".join(ch for ch in mark.lower().strip() if ch.isalnum() or ch == "'")


def align_intervals(
    ref_intervals: list[tuple[float, float, str]],
    gen_intervals: list[tuple[float, float, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_marks = [normalize_mark(x[2]) for x in ref_intervals]
    gen_marks = [normalize_mark(x[2]) for x in gen_intervals]

    n = min(len(ref_marks), len(gen_marks))
    if n == 0:
        raise ValueError("空对齐序列")

    ref_cut = ref_intervals[:n]
    gen_cut = gen_intervals[:n]

    ref_s = np.array([x[0] for x in ref_cut], dtype=np.float64)
    ref_e = np.array([x[1] for x in ref_cut], dtype=np.float64)
    gen_s = np.array([x[0] for x in gen_cut], dtype=np.float64)
    gen_e = np.array([x[1] for x in gen_cut], dtype=np.float64)
    return ref_s, ref_e, gen_s, gen_e


def compute_mse(
    ref_intervals: list[tuple[float, float, str]],
    gen_intervals: list[tuple[float, float, str]],
    mode: str,
) -> float:
    ref_s, ref_e, gen_s, gen_e = align_intervals(ref_intervals, gen_intervals)

    if mode == "boundary":
        ref_vec = np.stack([ref_s, ref_e], axis=1).reshape(-1)
        gen_vec = np.stack([gen_s, gen_e], axis=1).reshape(-1)
    elif mode == "center":
        ref_vec = (ref_s + ref_e) / 2.0
        gen_vec = (gen_s + gen_e) / 2.0
    else:  # duration
        ref_vec = ref_e - ref_s
        gen_vec = gen_e - gen_s

    return float(np.mean((ref_vec - gen_vec) ** 2))


def build_generated_textgrid_index(generated_aligned_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for tg_path in generated_aligned_dir.rglob("*.TextGrid"):
        stem = tg_path.stem
        if stem not in index:
            index[stem] = []
        index[stem].append(tg_path)
    return index


def infer_generated_textgrid_path(
    generated_aligned_dir: Path,
    output_wav: str,
    tg_index: dict[str, list[Path]] | None = None,
) -> Path:
    wav_path = Path(output_wav)
    wav_stem = wav_path.stem

    direct_path = generated_aligned_dir / f"{wav_stem}.TextGrid"
    if direct_path.exists():
        return direct_path

    parent_named_path = generated_aligned_dir / wav_path.parent.name / f"{wav_stem}.TextGrid"
    if parent_named_path.exists():
        return parent_named_path

    if tg_index and wav_stem in tg_index and tg_index[wav_stem]:
        return tg_index[wav_stem][0]

    return direct_path


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    generated_aligned_dir = Path(args.generated_aligned_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_manifest(manifest_path)
    valid_status = {"ok", "skipped_existing"}
    ok_records = [r for r in records if r.get("status") in valid_status]
    tg_index = build_generated_textgrid_index(generated_aligned_dir)

    print(
        f"[Info] manifest_records={len(records)} usable_records={len(ok_records)} "
        f"generated_textgrids={sum(len(v) for v in tg_index.values())}"
    )

    mse_list: list[float] = []
    details: list[dict[str, Any]] = []

    for rec in ok_records:
        ref_tg = Path(rec["source_textgrid"])
        gen_tg = infer_generated_textgrid_path(generated_aligned_dir, rec["output_wav"])
        if not ref_tg.exists() or not gen_tg.exists():
            continue

        try:
            ref_intervals = read_word_intervals(ref_tg, args.tier, ignore_empty=args.ignore_empty)
            gen_intervals = read_word_intervals(gen_tg, args.tier, ignore_empty=args.ignore_empty)
            mse = compute_mse(ref_intervals, gen_intervals, mode=args.mode)
            mse_list.append(mse)
            details.append(
                {
                    "sample_id": rec.get("sample_id"),
                    "clip_filename": rec.get("clip_filename"),
                    "repeat_idx": rec.get("repeat_idx"),
                    "output_wav": rec.get("output_wav"),
                    "ref_textgrid": str(ref_tg),
                    "gen_textgrid": str(gen_tg),
                    "mse": mse,
                    "mode": args.mode,
                }
            )
        except Exception:
            continue

    if not mse_list:
        raise RuntimeError("没有可用样本用于 MSE 计算，请检查 manifest 和生成对齐目录")

    mse_arr = np.array(mse_list, dtype=np.float64)
    stats = {
        "count": int(len(mse_arr)),
        "mean": float(np.mean(mse_arr)),
        "std": float(np.std(mse_arr)),
        "min": float(np.min(mse_arr)),
        "p25": float(np.percentile(mse_arr, 25)),
        "p50": float(np.percentile(mse_arr, 50)),
        "p75": float(np.percentile(mse_arr, 75)),
        "max": float(np.max(mse_arr)),
        "mode": args.mode,
    }

    details_path = output_dir / "word_alignment_mse_details.json"
    stats_path = output_dir / "word_alignment_mse_stats.json"
    fig_path = output_dir / "word_alignment_mse_distribution.png"
    dist_path = output_dir / "word_alignment_mse_integer_distribution.json"

    with details_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 计算整数区间分布
    max_mse = int(np.ceil(mse_arr.max()))
    bins = list(range(0, max_mse + 2))  # 0 to max+1
    hist, bin_edges = np.histogram(mse_arr, bins=bins)
    total_count = len(mse_arr)
    percentages = [(count / total_count * 100) for count in hist]
    cumulative_percentages = [sum(percentages[:i+1]) for i in range(len(percentages))]
    distribution = {
        "bins": bins[:-1],  # 区间起始
        "counts": hist.tolist(),
        "percentages": percentages,
        "cumulative_percentages": cumulative_percentages,
        "bin_edges": bin_edges.tolist(),
        "mode": args.mode,
        "total_count": total_count,
    }
    with dist_path.open("w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)

    print(f"MSE Integer Distribution ({args.mode}) - Cumulative Percentages:")
    for i, (count, pct, cum_pct) in enumerate(zip(hist, percentages, cumulative_percentages)):
        start = bins[i]
        end = bins[i+1]
        print(f"  [0, {end}): {cum_pct:.1f}%")

    if plt is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(mse_arr, bins=40, alpha=0.85, color="steelblue", edgecolor="white")
        plt.axvline(stats["mean"], color="red", linestyle="--", linewidth=1.5, label=f"mean={stats['mean']:.6f}")
        plt.axvline(stats["p50"], color="orange", linestyle=":", linewidth=1.5, label=f"median={stats['p50']:.6f}")
        plt.title(f"Word-level Alignment MSE Distribution ({args.mode})")
        plt.xlabel("MSE")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
    else:
        print("[Warn] matplotlib 不可用，跳过分布图绘制")

    print(f"Done. count={stats['count']} mean={stats['mean']:.6f} p50={stats['p50']:.6f}")
    print(f"Stats: {stats_path}")
    print(f"Details: {details_path}")
    print(f"Distribution: {dist_path}")
    if plt is not None:
        print(f"Figure: {fig_path}")

if __name__ == "__main__":
    main()
