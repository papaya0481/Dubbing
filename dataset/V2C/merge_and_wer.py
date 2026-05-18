#!/usr/bin/env python3
"""
merge_and_wer.py — 整合多个 v2c 格式数据集目录，运行 Whisper ASR 计算 WER，输出合并目录。

目录结构（每个源目录）:
  <src>/
    metadata.csv          # 包含 Movie, Speaker, Utterance, Audio_Filename, ...
    audios/ost/*.wav
    videos/*.mp4
    meta_files/*.csv      # 按电影分割的子 CSV

用法示例:
  python merge_and_wer.py \\
      --src /data2/ruixin/datasets/v2c_origin /data2/ruixin/downloads/v2c_part2/output_origin \\
      --dst /data2/ruixin/datasets/v2c_merged \\
      [--model large] \\
      [--language en] \\
      [--skip-asr]        # 跳过 ASR，仅整合文件
      [--no-save-src]     # 不把 WER 写回源目录
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import wer as jiwer_wer


# ─── Text / WER Utilities ──────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def calculate_wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref)
    h = normalize_text(hyp)
    if not r:
        return 1.0 if h else 0.0
    return float(jiwer_wer(r, h))


# ─── Whisper Utilities ────────────────────────────────────────────────────────

def load_whisper_model(model_name: str):
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "需要 openai-whisper：pip install -U openai-whisper"
        ) from exc
    print(f"[INFO] 加载 Whisper 模型: {model_name}")
    return whisper.load_model(model_name, download_root=os.getenv("WHISPER_CACHE_DIR", None))


def transcribe(model, audio_path: str, language: str | None = None) -> str:
    try:
        kwargs: dict = {}
        if language:
            kwargs["language"] = language
        result = model.transcribe(str(audio_path), **kwargs)
        return (result.get("text") or "").strip()
    except Exception as e:
        print(f"[WARN] 转录失败 {audio_path}: {e}")
        return ""


# ─── WER Distribution Report ─────────────────────────────────────────────────

def print_wer_distribution_to_txt(wer_values: list[float], dst: Path) -> None:
    arr = np.array(wer_values)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")]
    labels = [
        "[0.0, 0.1)", "[0.1, 0.2)", "[0.2, 0.3)", "[0.3, 0.4)",
        "[0.4, 0.5)", "[0.5, 0.6)", "[0.6, 0.7)", "[0.7, 0.8)",
        "[0.8, 0.9)", "[0.9, 1.0)", "[1.0,  +∞)",
    ]
    counts, _ = np.histogram(arr, bins=bins)
    total = len(arr)

    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("                      WER 分布")
    lines.append("=" * 62)
    lines.append(f"  {'区间':>12}  {'数量':>7}  {'占比':>6}")
    lines.append("-" * 62)
    for label, count in zip(labels, counts):
        pct = count / total * 100 if total > 0 else 0.0
        lines.append(f"  {label:>12}  {count:>7}  {pct:>5.1f}%")
    lines.append("-" * 62)
    lines.append(f"  均值 WER : {arr.mean():.4f}")
    lines.append(f"  中位 WER : {float(np.median(arr)):.4f}")
    lines.append(f"  样本总数 : {total}")
    lines.append("=" * 62)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = dst / "wer_distribution.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print(f"[INFO] WER 分布文本已保存: {report_path}")


# ─── File Copying ────────────────────────────────────────────────────────────

def copy_file(src_path: Path, dst_path: Path) -> None:
    if dst_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="整合多个 v2c 数据集目录并用 Whisper 计算 WER"
    )
    parser.add_argument("--src", nargs="+", required=True,
                        help="源目录列表（至少一个）")
    parser.add_argument("--dst", required=True,
                        help="目标输出目录")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper 模型名称 (默认: large)")
    parser.add_argument("--language", default="en",
                        help="ASR 语言 (默认: en)")
    parser.add_argument("--skip-asr", action="store_true",
                        help="跳过 ASR 推理，仅整合文件结构")
    parser.add_argument("--no-save-src", action="store_true",
                        help="不把 WER 等新增列写回源目录的 metadata.csv")
    args = parser.parse_args()

    src_dirs = [Path(p) for p in args.src]
    dst = Path(args.dst)

    # ── 验证源目录 ──────────────────────────────────────────────────────────────
    for src in src_dirs:
        if not (src / "metadata.csv").exists():
            print(f"[ERROR] 找不到 {src}/metadata.csv")
            sys.exit(1)

    # ── Step 1: 加载所有 metadata ───────────────────────────────────────────────
    print("\n[Step 1] 加载 metadata ...")
    all_dfs: list[pd.DataFrame] = []
    for src in src_dirs:
        df = pd.read_csv(src / "metadata.csv")
        df["_src_dir"] = str(src)
        all_dfs.append(df)
        print(f"  {len(df):>5} 行  ←  {src}/metadata.csv")

    merged_df = pd.concat(all_dfs, ignore_index=True)
    before = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["Audio_Filename"], keep="first")
    after = len(merged_df)
    print(f"  合并后: {before} 行 → 去重后: {after} 行 (删除 {before - after} 重复)")

    # ── Step 2: ASR + WER ──────────────────────────────────────────────────────
    if not args.skip_asr:
        print("\n[Step 2] ASR 推理 + WER 计算 ...")
        model = load_whisper_model(args.model)

        # 支持断点续跑：如果已有 WER 列则跳过对应样本
        already_done: set[str] = set()
        if "WER" in merged_df.columns and "ASR_Text" in merged_df.columns:
            done_mask = merged_df["WER"].notna()
            already_done = set(merged_df.loc[done_mask, "Audio_Filename"].tolist())
            if already_done:
                print(f"  [续跑] 跳过已有 WER 的 {len(already_done)} 条样本")
        else:
            merged_df["ASR_Text"] = pd.NA
            merged_df["WER"] = pd.NA

        for idx, row in tqdm(
            merged_df.iterrows(), total=len(merged_df), desc="  ASR"
        ):
            fname = row["Audio_Filename"]
            if not isinstance(fname, str) or not fname.strip():
                merged_df.at[idx, "ASR_Text"] = ""
                merged_df.at[idx, "WER"] = 1.0
                continue
            fname = fname.strip()
            if fname in already_done:
                continue

            src_dir = Path(row["_src_dir"])

            # 定位音频文件
            audio_path = src_dir / "audios" / "ost" / fname
            if not audio_path.exists():
                rel = str(row.get("Audio_Path", ""))
                if rel:
                    audio_path = src_dir / rel
            if not audio_path.exists():
                merged_df.at[idx, "ASR_Text"] = ""
                merged_df.at[idx, "WER"] = 1.0
                continue

            hyp = transcribe(model, str(audio_path), args.language)
            ref = str(row.get("Utterance", ""))
            w = round(calculate_wer(ref, hyp), 4)

            merged_df.at[idx, "ASR_Text"] = hyp
            merged_df.at[idx, "WER"] = w

    # ── Step 3: 计算 Word_Count 列 ───────────────────────────────────────────
    print("\n[Step 3] 计算 Word_Count 列 ...")
    merged_df["Word_Count"] = merged_df["Utterance"].apply(
        lambda u: len(str(u or "").strip().split())
    )

    # ── Step 4: 整合媒体文件到 dst ─────────────────────────────────────────────
    print(f"\n[Step 4] 复制媒体文件到 {dst} ...")
    dst_audio_dir = dst / "audios" / "ost"
    dst_video_dir = dst / "videos"
    dst_meta_dir  = dst / "meta_files"
    for d in [dst_audio_dir, dst_video_dir, dst_meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="  媒体"):
        src_dir = Path(row["_src_dir"])

        # 音频
        audio_fname = str(row.get("Audio_Filename", "")).strip()
        if audio_fname:
            src_audio = src_dir / "audios" / "ost" / audio_fname
            if src_audio.exists():
                copy_file(src_audio, dst_audio_dir / audio_fname)

        # 视频
        clip_fname = str(row.get("Clip_Filename", "")).strip()
        if clip_fname:
            src_video = src_dir / "videos" / clip_fname
            if src_video.exists():
                copy_file(src_video, dst_video_dir / clip_fname)

    # ── Step 5: 写合并 metadata.csv ────────────────────────────────────────────
    print(f"\n[Step 5] 写入合并 metadata.csv ...")
    out_df = merged_df.drop(columns=["_src_dir"], errors="ignore").copy()
    out_df = out_df.drop(columns=["Keep"], errors="ignore")
    # 更新路径为相对于 dst
    out_df["Audio_Path"] = out_df["Audio_Filename"].apply(
        lambda f: f"audios/ost/{f}" if str(f).strip() else ""
    )
    out_df["Clip_Path"] = out_df["Clip_Filename"].apply(
        lambda f: f"videos/{f}" if str(f).strip() else ""
    )
    metadata_out = dst / "metadata.csv"
    out_df.to_csv(metadata_out, index=False)
    print(f"  保存: {metadata_out}  ({len(out_df)} 行)")

    # ── Step 6: 写 per-movie meta_files CSV ────────────────────────────────────
    if "Movie" in out_df.columns:
        print(f"\n[Step 6] 写入 per-movie meta_files ...")
        for movie, group in out_df.groupby("Movie"):
            out_path = dst_meta_dir / f"{movie}.csv"
            group.to_csv(out_path, index=False)
        n_movies = out_df["Movie"].nunique()
        print(f"  已写入 {n_movies} 个电影 CSV → {dst_meta_dir}")

    # ── Step 7: 写回源目录 metadata ────────────────────────────────────────────
    if not args.skip_asr and not args.no_save_src:
        print("\n[Step 7] 将 ASR_Text / WER / Word_Count 写回源目录 metadata.csv ...")
        extra_cols = [c for c in ["ASR_Text", "WER", "Word_Count"] if c in merged_df.columns]
        wer_index = merged_df.set_index("Audio_Filename")[extra_cols]

        for src in src_dirs:
            src_df = pd.read_csv(src / "metadata.csv")
            src_df = src_df.drop(columns=["Keep"], errors="ignore")
            for col in extra_cols:
                src_df[col] = src_df["Audio_Filename"].map(wer_index[col])
            src_df.to_csv(src / "metadata.csv", index=False)
            print(f"  已更新: {src}/metadata.csv")

    # ── Step 8: WER 分布报告 ───────────────────────────────────────────────────
    if "WER" in merged_df.columns:
        wer_vals = merged_df["WER"].dropna().astype(float).tolist()
        if wer_vals:
            print_wer_distribution_to_txt(wer_vals, dst)

    # ── 汇总统计 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("           筛选统计")
    print("=" * 40)
    for src in src_dirs:
        n = int((merged_df["_src_dir"] == str(src)).sum()) if "_src_dir" in merged_df.columns else "?"
        print(f"  {src.name:30s}: {n} 行")
    print(f"  {'总样本数':17s}: {len(merged_df)}")
    if "Word_Count" in merged_df.columns and len(merged_df) > 0:
        avg_words = float(merged_df["Word_Count"].astype(float).mean())
        print(f"  {'平均词数':17s}: {avg_words:.2f}")
    print("=" * 40)
    print(f"\n[完成] 输出目录: {dst}")


if __name__ == "__main__":
    main()
