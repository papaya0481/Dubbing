import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import textgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 word-level 对齐 MSE 过滤样本，并复制通过样本")
    parser.add_argument("--metadata-csv", type=str, required=True, help="生成脚本输出的 generation_metadata.csv")
    parser.add_argument("--generated-aligned-dir", type=str, required=True, help="生成音频 MFA 对齐 TextGrid 目录")
    parser.add_argument("--reference-aligned-dir", type=str, required=True, help="参考音频 MFA 对齐 TextGrid 目录")
    parser.add_argument("--output-dir", type=str, required=True, help="过滤后输出目录")
    parser.add_argument("--mse-threshold", type=float, required=True, help="保留样本的 MSE 阈值，保留条件: mse <= threshold")
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


def build_textgrid_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for tg_path in root.rglob("*.TextGrid"):
        index.setdefault(tg_path.stem, []).append(tg_path)
    return index


def infer_generated_textgrid_path(generated_aligned_dir: Path, output_wav: str, tg_index: dict[str, list[Path]]) -> Path:
    wav_path = Path(output_wav)
    wav_stem = wav_path.stem

    direct_path = generated_aligned_dir / f"{wav_stem}.TextGrid"
    if direct_path.exists():
        return direct_path

    nested_path = generated_aligned_dir / wav_path.parent.name / f"{wav_stem}.TextGrid"
    if nested_path.exists():
        return nested_path

    if wav_stem in tg_index and tg_index[wav_stem]:
        return tg_index[wav_stem][0]

    return direct_path


def infer_reference_textgrid_path(reference_aligned_dir: Path, sample_key: str, tg_index: dict[str, list[Path]]) -> Path:
    direct_path = reference_aligned_dir / f"{sample_key}.TextGrid"
    if direct_path.exists():
        return direct_path

    if sample_key in tg_index and tg_index[sample_key]:
        return tg_index[sample_key][0]

    return direct_path


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_metadata_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv)
    generated_aligned_dir = Path(args.generated_aligned_dir)
    reference_aligned_dir = Path(args.reference_aligned_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metadata_csv(metadata_csv)
    gen_tg_index = build_textgrid_index(generated_aligned_dir)
    ref_tg_index = build_textgrid_index(reference_aligned_dir)

    passed_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    copy_wav_dir = output_dir / "wavs"
    copy_txt_dir = output_dir / "txts"
    copy_gen_tg_dir = output_dir / "textgrids" / "generated"
    copy_ref_tg_dir = output_dir / "textgrids" / "reference"

    process_rows = [r for r in rows if r.get("status") in {"ok", "skipped_existing"}]
    print(f"[Info] metadata_rows={len(rows)} process_rows={len(process_rows)}")

    for idx, row in enumerate(process_rows, start=1):
        sample_key = str(row.get("sample_key", "")).strip()
        output_wav = str(row.get("output_wav", "")).strip()
        output_txt = str(row.get("output_txt", "")).strip()

        rec = dict(row)
        rec["mse"] = ""
        rec["mse_mode"] = args.mode
        rec["gen_textgrid"] = ""
        rec["ref_textgrid"] = ""
        rec["filter_status"] = "rejected"
        rec["filter_reason"] = ""

        if not sample_key or not output_wav:
            rec["filter_reason"] = "missing_sample_key_or_output_wav"
            rejected_rows.append(rec)
            continue

        gen_tg = infer_generated_textgrid_path(generated_aligned_dir, output_wav, gen_tg_index)
        ref_tg = infer_reference_textgrid_path(reference_aligned_dir, sample_key, ref_tg_index)

        rec["gen_textgrid"] = str(gen_tg)
        rec["ref_textgrid"] = str(ref_tg)

        if not gen_tg.exists() or not ref_tg.exists():
            rec["filter_reason"] = "missing_textgrid"
            rejected_rows.append(rec)
            continue

        try:
            ref_intervals = read_word_intervals(ref_tg, args.tier, ignore_empty=args.ignore_empty)
            gen_intervals = read_word_intervals(gen_tg, args.tier, ignore_empty=args.ignore_empty)
            mse = compute_mse(ref_intervals, gen_intervals, mode=args.mode)
            rec["mse"] = f"{mse:.10f}"
        except Exception as exc:
            rec["filter_reason"] = f"mse_error: {exc}"
            rejected_rows.append(rec)
            continue

        if float(rec["mse"]) > args.mse_threshold:
            rec["filter_reason"] = "mse_above_threshold"
            rejected_rows.append(rec)
            continue

        wav_path = Path(output_wav)
        txt_path = Path(output_txt) if output_txt else wav_path.with_suffix(".txt")

        if not wav_path.exists():
            rec["filter_reason"] = "missing_output_wav"
            rejected_rows.append(rec)
            continue

        rec["filter_status"] = "passed"
        rec["filter_reason"] = ""
        passed_rows.append(rec)

        safe_copy(wav_path, copy_wav_dir / wav_path.name)
        if txt_path.exists():
            safe_copy(txt_path, copy_txt_dir / txt_path.name)
        safe_copy(gen_tg, copy_gen_tg_dir / gen_tg.name)
        safe_copy(ref_tg, copy_ref_tg_dir / ref_tg.name)

        if idx % 100 == 0:
            print(f"[Progress] {idx}/{len(process_rows)}")

    extra_fields = ["mse", "mse_mode", "gen_textgrid", "ref_textgrid", "filter_status", "filter_reason"]
    fieldnames = list(rows[0].keys()) + [f for f in extra_fields if f not in rows[0].keys()] if rows else extra_fields

    passed_csv = output_dir / "filtered_passed_metadata.csv"
    rejected_csv = output_dir / "filtered_rejected_metadata.csv"

    write_csv(passed_csv, passed_rows, fieldnames)
    write_csv(rejected_csv, rejected_rows, fieldnames)

    print(
        f"Done. total={len(process_rows)} passed={len(passed_rows)} rejected={len(rejected_rows)} "
        f"threshold={args.mse_threshold} mode={args.mode}"
    )
    print(f"Passed CSV: {passed_csv}")
    print(f"Rejected CSV: {rejected_csv}")
    print(f"Copied wavs: {copy_wav_dir}")


if __name__ == "__main__":
    main()
