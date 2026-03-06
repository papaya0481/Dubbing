"""
gen_semantic_stretch.py
=======================
读取已生成数据集的 generation_metadata.csv，逐条：
  1. 用原始文本、语音 prompt、情绪向量调用 IndexTTS2Semantic.infer_with_semantic_warp()
  2. 以 <csv_parent>/aligned/{sample_key}_r{repeat_idx}.TextGrid 作为 target_textgrid
     驱动语义 latent 时序拉伸
  3. 输出最终 wav 到 --output-dir

所有必需输入（text、source_prompt_wav、emo_text、TextGrid）缺一不可，缺少则直接报错。

用法示例：
    python mel_convert/generate/gen_semantic_stretch.py \\
        --metadata-csv /data2/ruixin/datasets/MELD_gen_pairs_semanti/sent_emo/generation_metadata.csv \\
        --output-dir   /data2/ruixin/datasets/MELD_gen_pairs_semanti/semantic_stretch \\
        --model-dir    /data2/ruixin/index-tts2/checkpoints \\
        --index-tts-root /home/ruixin/Dubbing/dubbing
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import sys
import traceback
import unicodedata
from pathlib import Path
from typing import Any

import torch
import tgt as _tgt

# ---------------------------------------------------------------------------
# Emotion helpers（与 gen.py 保持一致）
# ---------------------------------------------------------------------------

EMOTION_DIMENSIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

_EMOTION_ALIASES: dict[str, str] = {
    "joy":      "happy",
    "anger":    "angry",
    "sadness":  "sad",
    "fear":     "afraid",
    "fearful":  "afraid",
    "disgust":  "disgusted",
    "surprise": "surprised",
    "neutral":  "calm",
}


def get_emotion_vector(emotion: str) -> list[float]:
    emotion = emotion.lower().strip()
    emotion = _EMOTION_ALIASES.get(emotion, emotion)
    if emotion not in EMOTION_DIMENSIONS:
        emotion = "calm"
    vec = [0.0] * len(EMOTION_DIMENSIONS)
    vec[EMOTION_DIMENSIONS.index(emotion)] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Text sanitisation（与 gen.py 保持一致）
# ---------------------------------------------------------------------------

_UNICODE_TO_ASCII: dict[int, str] = {
    0x2018: "'", 0x2019: "'", 0x02BC: "'",
    0x201C: '"', 0x201D: '"',
    0x2013: "-", 0x2014: "-", 0x2012: "-",
    0x00A0: " ", 0x202F: " ", 0x3000: " ",
    0x0091: "'", 0x0092: "'", 0x0093: '"', 0x0094: '"',
    0x0096: "-", 0x0097: "-",
}
_ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x00AD}


def sanitize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in _ZERO_WIDTH:
            continue
        replacement = _UNICODE_TO_ASCII.get(cp)
        chars.append(replacement if replacement is not None else ch)
    text = "".join(chars)
    return text.encode("ascii", errors="ignore").decode("ascii")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def sanitize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k).strip(): v for k, v in row.items() if k is not None}


REQUIRED_FIELDS = ["sample_key", "repeat_idx", "text", "source_prompt_wav", "emo_text"]


def load_rows(metadata_csv: Path) -> list[dict[str, Any]]:
    """载入 generation_metadata.csv，不做字段校验，由调用方逐条处理。"""
    rows: list[dict[str, Any]] = []
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            rows.append(sanitize_row_keys(raw_row))
    return rows


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 generation_metadata.csv 逐条进行语义 latent 拉伸再生成"
    )
    parser.add_argument(
        "--metadata-csv", type=str, required=True,
        help="generation_metadata.csv 路径，其同级目录下必须有 aligned/ 子目录"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出 wav 存放目录"
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="IndexTTS2 checkpoints 目录"
    )
    parser.add_argument(
        "--index-tts-root", type=str, required=True,
        help="包含 indextts/ 包的根目录（通常为 .../dubbing）"
    )
    parser.add_argument("--is-fp16", action="store_true", default=False)
    parser.add_argument(
        "--diffusion-steps", type=int, default=25,
        help="CFM 扩散步数（默认 25）"
    )
    parser.add_argument(
        "--inference-cfg-rate", type=float, default=0.7,
        help="CFM classifier-free guidance 比例（默认 0.7）"
    )
    parser.add_argument(
        "--tier-name", type=str, default="phones",
        choices=["phones", "words"],
        help="TextGrid 使用的 tier（默认 phones）"
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata_csv 不存在: {metadata_csv}")

    aligned_dir = metadata_csv.parent / "aligned"
    if not aligned_dir.is_dir():
        raise FileNotFoundError(
            f"aligned 目录不存在: {aligned_dir}\n"
            f"（期望在 metadata_csv 的同级目录下）"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] reading metadata CSV: {metadata_csv}")
    rows = load_rows(metadata_csv)
    total = len(rows)
    print(f"[Load] total rows: {total}")

    if args.dry_run:
        print("[DRY RUN] 仅校验输入，跳过模型推理")
        # 提前检查每行的 TextGrid 和 prompt 是否存在
        missing_tg: list[str] = []
        missing_prompt: list[str] = []
        for row in rows:
            sample_key = str(row["sample_key"]).strip()
            repeat_idx = str(row["repeat_idx"]).strip()
            tg_path = aligned_dir / f"{sample_key}_r{repeat_idx}.TextGrid"
            if not tg_path.exists():
                missing_tg.append(str(tg_path))
            prompt_path = Path(str(row["source_prompt_wav"]).strip())
            if not prompt_path.exists():
                missing_prompt.append(str(prompt_path))
        if missing_tg:
            print(f"[DRY RUN] 缺少 TextGrid：{len(missing_tg)} 条，前5条：{missing_tg[:5]}")
        if missing_prompt:
            print(f"[DRY RUN] 缺少 prompt：{len(missing_prompt)} 条，前5条：{missing_prompt[:5]}")
        print("[DRY RUN] done")
        return

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------
    sys.path.insert(0, args.index_tts_root)

    infer_semantic_mod = importlib.import_module("indextts.infer_semantic")
    IndexTTS2Semantic  = getattr(infer_semantic_mod, "IndexTTS2Semantic")

    print("[Init] loading IndexTTS2Semantic …")
    tts = IndexTTS2Semantic(
        model_dir=args.model_dir,
        cfg_path=str(Path(args.model_dir) / "config.yaml"),
        is_fp16=args.is_fp16,
        use_cuda_kernel=False,
        mfa_aligner=None,
    )
    
    mfa_module  = importlib.import_module("modules.mfa_alinger")
    MFAAligner  = getattr(mfa_module, "MFAAligner")
    print("[Init] loading MFAAligner …")
    mfa_aligner = MFAAligner(beam=20, retry_beam=200)
    
    tts.mfa_aligner = mfa_aligner

    # ------------------------------------------------------------------
    # 输出 metadata CSV
    # ------------------------------------------------------------------
    fieldnames = [
        "sample_key", "repeat_idx",
        "text", "emo_text",
        "source_prompt_wav",
        "target_textgrid",
        "output_wav",
        "seg_lens", "wav_secs",
        "status", "error",
    ]
    out_metadata_path = output_dir / "generation_metadata.csv"
    current = 0

    with out_metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            current += 1
            sample_key = str(row["sample_key"]).strip()
            repeat_idx = str(row["repeat_idx"]).strip()
            text       = sanitize_text(str(row["text"]).strip())
            emo_text   = str(row["emo_text"]).strip()
            source_prompt_wav = Path(str(row["source_prompt_wav"]).strip())

            # ---- 必需输入校验，缺少则跳过 ----
            missing = [fld for fld in REQUIRED_FIELDS if not str(row.get(fld, "")).strip()]
            if missing:
                warn = (
                    f"[{current}/{total}] {sample_key}_r{repeat_idx} -> skipped_missing_fields "
                    f"{missing}"
                )
                print(warn, file=sys.stderr)
                writer.writerow({
                    "sample_key": sample_key, "repeat_idx": repeat_idx,
                    "text": text, "emo_text": emo_text,
                    "source_prompt_wav": str(row.get("source_prompt_wav", "")),
                    "target_textgrid": "",
                    "output_wav": "",
                    "seg_lens": "", "wav_secs": "",
                    "status": "skipped_missing_fields", "error": warn,
                })
                f.flush()
                continue

            if not text:
                raise ValueError(f"[{sample_key}_r{repeat_idx}] text 为空")
            if not emo_text:
                raise ValueError(f"[{sample_key}_r{repeat_idx}] emo_text 为空")
            if not source_prompt_wav.exists():
                raise FileNotFoundError(
                    f"[{sample_key}_r{repeat_idx}] source_prompt_wav 不存在: {source_prompt_wav}"
                )

            tg_path = aligned_dir / f"{sample_key}_r{repeat_idx}.TextGrid"
            if not tg_path.exists():
                print(f"[{current}/{total}] {sample_key}_r{repeat_idx} -> skipped_missing_textgrid", flush=True)
                writer.writerow({
                    "sample_key": sample_key, "repeat_idx": repeat_idx,
                    "text": text, "emo_text": emo_text,
                    "source_prompt_wav": str(source_prompt_wav),
                    "target_textgrid": str(tg_path),
                    "output_wav": "",
                    "seg_lens": "", "wav_secs": "",
                    "status": "skipped_missing_textgrid", "error": str(tg_path),
                })
                current += 1
                continue

            out_wav = output_dir / f"{sample_key}_r{repeat_idx}.wav"

            # ---- 跳过已完成 ----
            if out_wav.exists():
                print(f"[{current}/{total}] {sample_key}_r{repeat_idx} -> skipped_existing")
                writer.writerow({
                    "sample_key": sample_key, "repeat_idx": repeat_idx,
                    "text": text, "emo_text": emo_text,
                    "source_prompt_wav": str(source_prompt_wav),
                    "target_textgrid": str(tg_path),
                    "output_wav": str(out_wav),
                    "seg_lens": "", "wav_secs": "",
                    "status": "skipped_existing", "error": "",
                })
                f.flush()
                continue

            # ---- 构建情绪向量：emo_text 中可能有 "|" 分隔多段 ----
            emo_labels  = emo_text.split("|")
            emo_vectors = [get_emotion_vector(e) for e in emo_labels]

            # ---- 推理 ----
            seg_lens_str = ""
            wav_secs_str = ""
            status       = "ok"
            error_msg    = ""

            # 预先读取 TextGrid 目标时长，供后续差值检查
            _tg_obj = _tgt.read_textgrid(str(tg_path))
            tg_target_secs: float = float(_tg_obj.end_time)

            try:
                result = tts.infer_with_semantic_warp(
                    spk_audio_prompt=str(source_prompt_wav),
                    text=text,
                    output_path=str(out_wav),
                    target_textgrid=str(tg_path),
                    tier_name=args.tier_name,
                    diffusion_steps=args.diffusion_steps,
                    inference_cfg_rate=args.inference_cfg_rate,
                    emo_vector=emo_vectors,
                )
                # result = (output_path, seg_lens, wav_length_final)
                _, seg_ls, wav_secs_raw = result
                seg_lens_str = str(seg_ls)
                wav_secs_str = str(wav_secs_raw)

                # ---- 时长差值检查 ----
                dur_diff = abs(float(wav_secs_raw) - tg_target_secs)
                if dur_diff > 0.02:
                    warn_msg = (
                        f"[WARN] 时长偏差过大，丢弃样本："
                        f"{sample_key}_r{repeat_idx}  "
                        f"gen={float(wav_secs_raw):.4f}s  "
                        f"target={tg_target_secs:.4f}s  "
                        f"diff={dur_diff:.4f}s"
                    )
                    print(warn_msg, file=sys.stderr)
                    if out_wav.exists():
                        out_wav.unlink()
                    status    = "discarded"
                    error_msg = warn_msg
                    seg_lens_str = ""
                    wav_secs_str = ""
            except Exception as exc:
                traceback.print_exc()
                status    = "failed"
                error_msg = str(exc)
                err_str   = str(exc)
                is_cuda_fatal = (
                    "CUDA" in err_str or "cuda" in err_str
                    or "AcceleratorError" in type(exc).__name__
                )
                if is_cuda_fatal:
                    writer.writerow({
                        "sample_key": sample_key, "repeat_idx": repeat_idx,
                        "text": text, "emo_text": emo_text,
                        "source_prompt_wav": str(source_prompt_wav),
                        "target_textgrid": str(tg_path),
                        "output_wav": str(out_wav),
                        "seg_lens": "", "wav_secs": "",
                        "status": "failed", "error": error_msg,
                    })
                    f.flush()
                    print(f"[FATAL] CUDA context corrupted, exiting. Re-run to resume.")
                    sys.exit(1)

            writer.writerow({
                "sample_key": sample_key, "repeat_idx": repeat_idx,
                "text": text, "emo_text": emo_text,
                "source_prompt_wav": str(source_prompt_wav),
                "target_textgrid": str(tg_path),
                "output_wav": str(out_wav),
                "seg_lens": seg_lens_str, "wav_secs": wav_secs_str,
                "status": status, "error": error_msg,
            })
            f.flush()
            print(f"[{current}/{total}] {sample_key}_r{repeat_idx} -> {status}")

    print(f"Done. metadata saved to: {out_metadata_path}")


if __name__ == "__main__":
    main()
