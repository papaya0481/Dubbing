import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import textgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按文件名中的 r1/r2 配对，计算对齐 MSE 并记录 metadata")
    parser.add_argument("--audio-dir", type=str, required=True, help="音频目录，文件名需包含 r1/r2")
    parser.add_argument("--aligned-dir", type=str, required=True, help="对应 TextGrid 目录")
    parser.add_argument("--tier", type=str, default="words")
    parser.add_argument("--mode", choices=["boundary", "center", "duration"], default="boundary")
    parser.add_argument("--ignore-empty", action="store_true", default=True)
    return parser.parse_args()


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
    n = min(len(ref_intervals), len(gen_intervals))
    if n == 0:
        raise ValueError("空对齐序列")

    ref_cut = ref_intervals[:n]
    gen_cut = gen_intervals[:n]

    ref_marks = [normalize_mark(x[2]) for x in ref_cut]
    gen_marks = [normalize_mark(x[2]) for x in gen_cut]
    mismatch = sum(1 for a, b in zip(ref_marks, gen_marks) if a and b and a != b)
    if mismatch > n * 0.6:
        raise ValueError(f"参考与生成词序列差异过大: mismatch={mismatch}/{n}")

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
    else:
        ref_vec = ref_e - ref_s
        gen_vec = gen_e - gen_s

    return float(np.mean((ref_vec - gen_vec) ** 2))


def build_textgrid_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for tg_path in root.rglob("*.TextGrid"):
        index[tg_path.stem] = tg_path
    return index


def split_repeat_stem(stem: str) -> tuple[str, str | None]:
    match = re.match(r"^(.*?)(?:[_-]?r([12]))$", stem, flags=re.IGNORECASE)
    if match is None:
        return stem, None
    base = match.group(1)
    tag = f"r{match.group(2)}"
    return base, tag


def build_r1_r2_pairs(audio_dir: Path) -> list[dict[str, Path]]:
    buckets: dict[str, dict[str, Path]] = {}
    for wav_path in sorted(audio_dir.rglob("*.wav")):
        base, tag = split_repeat_stem(wav_path.stem)
        if tag not in {"r1", "r2"}:
            continue
        if base not in buckets:
            buckets[base] = {}
        buckets[base][tag] = wav_path

    pairs: list[dict[str, Path]] = []
    for base, item in buckets.items():
        if "r1" in item and "r2" in item:
            pairs.append({"pair_key": base, "r1_wav": item["r1"], "r2_wav": item["r2"]})
    return pairs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    aligned_dir = Path(args.aligned_dir)
    metadata_out = audio_dir / "generation_metadata.csv"
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    tg_index = build_textgrid_index(aligned_dir)
    pairs = build_r1_r2_pairs(audio_dir)
    print(f"[Info] audio_pairs(r1+r2)={len(pairs)}")

    records: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        r1_wav = pair["r1_wav"]
        r2_wav = pair["r2_wav"]
        pair_key = str(pair["pair_key"])

        r1_tg = tg_index.get(r1_wav.stem)
        r2_tg = tg_index.get(r2_wav.stem)

        rec: dict[str, Any] = {
            "pair_key": pair_key,
            "r1_wav": str(r1_wav),
            "r2_wav": str(r2_wav),
            "r1_textgrid": str(r1_tg) if r1_tg is not None else "",
            "r2_textgrid": str(r2_tg) if r2_tg is not None else "",
            "mse_mode": args.mode,
            "mse": "",
            "status": "ok",
            "error": "",
        }

        if r1_tg is None or r2_tg is None:
            rec["status"] = "missing_textgrid"
            rec["error"] = "r1_or_r2_textgrid_not_found"
            records.append(rec)
            continue

        try:
            r1_intervals = read_word_intervals(r1_tg, args.tier, ignore_empty=args.ignore_empty)
            r2_intervals = read_word_intervals(r2_tg, args.tier, ignore_empty=args.ignore_empty)
            mse = compute_mse(r1_intervals, r2_intervals, mode=args.mode)
            rec["mse"] = f"{mse:.10f}"
        except Exception as exc:
            rec["status"] = "mse_error"
            rec["error"] = str(exc)

        records.append(rec)
        if idx % 100 == 0:
            print(f"[Progress] {idx}/{len(pairs)}")

    fieldnames = [
        "pair_key",
        "r1_wav",
        "r2_wav",
        "r1_textgrid",
        "r2_textgrid",
        "mse_mode",
        "mse",
        "status",
        "error",
    ]
    write_csv(metadata_out, records, fieldnames)
    print(f"Done. pairs={len(pairs)} records={len(records)} mode={args.mode}")
    print(f"Metadata CSV (updated): {metadata_out}")


if __name__ == "__main__":
    main()
