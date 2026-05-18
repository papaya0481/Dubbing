"""
fix_metadata_and_gen_transcripts.py

两件事：
1. 修复 metadata.csv 中 Utterance 列的转义字符问题：
   原始 MELD CSV 以 Windows-1252 编码存储右单引号（\x92），
   pandas 默认 UTF-8 读取后变为 U+0092 控制字符（显示为消失的撇号）。
   本脚本将其统一替换为普通 ASCII 撇号 '，并覆盖保存 metadata.csv。

2. 为 audios/ost/ 目录下每个 .wav 文件生成同名 .txt 文件，
   内容为对应的修复后的 Utterance。
"""

import os
import argparse
import pandas as pd


# Windows-1252 特殊引号字符映射到普通 ASCII 等价字符
_WIN1252_MAP = {
    "\x80": "",      # EURO SIGN (不常见于对话)
    "\x82": ",",     # SINGLE LOW-9 QUOTATION MARK
    "\x84": ",,",    # DOUBLE LOW-9 QUOTATION MARK
    "\x85": "...",   # HORIZONTAL ELLIPSIS
    "\x86": "+",     # DAGGER
    "\x87": "++",    # DOUBLE DAGGER
    "\x88": "^",     # MODIFIER LETTER CIRCUMFLEX ACCENT
    "\x89": "%%",    # PER MILLE SIGN
    "\x8b": "<",     # SINGLE LEFT-POINTING ANGLE QUOTATION
    "\x91": "'",     # LEFT SINGLE QUOTATION MARK  → '
    "\x92": "'",     # RIGHT SINGLE QUOTATION MARK → '
    "\x93": '"',     # LEFT DOUBLE QUOTATION MARK  → "
    "\x94": '"',     # RIGHT DOUBLE QUOTATION MARK → "
    "\x95": "*",     # BULLET
    "\x96": "-",     # EN DASH
    "\x97": "--",    # EM DASH
    "\x99": "TM",    # TRADE MARK SIGN
    "\x9b": ">",     # SINGLE RIGHT-POINTING ANGLE QUOTATION
    # Unicode 版本（以防已经被某些步骤转换）
    "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
    "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK
    "\u2013": "-",   # EN DASH
    "\u2014": "--",  # EM DASH
    "\u2026": "...", # HORIZONTAL ELLIPSIS
    "\u0092": "'",   # U+0092 PRIVATE USE TWO（UTF-8 读 \xc2\x92 的结果）
}


def fix_escape_chars(text: str) -> str:
    """替换所有 Windows-1252 特殊字符为 ASCII 等价字符。"""
    if not isinstance(text, str):
        return text
    for src, dst in _WIN1252_MAP.items():
        text = text.replace(src, dst)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="修复 metadata.csv 转义字符并生成音频对应 transcript txt 文件"
    )
    parser.add_argument(
        "--metadata",
        default="/data2/ruixin/datasets/MELD_raw/metadata.csv",
        help="metadata.csv 路径",
    )
    parser.add_argument(
        "--audio-dir",
        default="/data2/ruixin/datasets/MELD_raw/audios/ost",
        help="音频文件所在目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要写入的内容，不实际写文件",
    )
    args = parser.parse_args()

    # ── 1. 读取并修复 metadata.csv ──────────────────────────────────────────
    print(f"读取 metadata.csv: {args.metadata}")
    df = pd.read_csv(args.metadata)

    before_count = df["Utterance"].str.contains("\x92|\u0092", regex=True).sum()
    print(f"  修复前含问题字符的行数: {before_count}")

    df["Utterance"] = df["Utterance"].apply(fix_escape_chars)
    if "ASR_Text" in df.columns:
        df["ASR_Text"] = df["ASR_Text"].apply(fix_escape_chars)

    after_count = df["Utterance"].str.contains("\x92|\u0092", regex=True).sum()
    print(f"  修复后含问题字符的行数: {after_count}")

    if not args.dry_run:
        df.to_csv(args.metadata, index=False)
        print(f"  已覆盖保存: {args.metadata}")
    else:
        print("  [dry-run] 跳过保存 metadata.csv")

    # ── 2. 构建 Audio_Filename → Utterance 查找表 ──────────────────────────
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        fname = str(row.get("Audio_Filename", "")).strip()
        utterance = str(row.get("Utterance", "")).strip()
        if fname:
            lookup[fname] = utterance

    # ── 3. 为每个 .wav 生成同名 .txt ──────────────────────────────────────
    print(f"\n扫描音频目录: {args.audio_dir}")
    wav_files = sorted(
        f for f in os.listdir(args.audio_dir) if f.lower().endswith(".wav")
    )
    print(f"  找到 WAV 文件: {len(wav_files)} 个")

    written = 0
    skipped_no_transcript = 0
    skipped_existing = 0

    for wav_name in wav_files:
        txt_name = os.path.splitext(wav_name)[0] + ".txt"
        txt_path = os.path.join(args.audio_dir, txt_name)

        utterance = lookup.get(wav_name)
        if utterance is None:
            skipped_no_transcript += 1
            print(f"  [warn] 找不到 transcript: {wav_name}")
            continue

        if os.path.exists(txt_path) and not args.dry_run:
            # 已存在则覆盖（确保内容是修复后的版本）
            skipped_existing += 1

        if args.dry_run:
            print(f"  [dry-run] {txt_name}: {utterance[:60]}")
        else:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(utterance)
            written += 1

    print(f"\n完成！")
    if not args.dry_run:
        print(f"  写入 txt 文件: {written} 个")
        print(f"    其中覆盖已有: {skipped_existing} 个")
    print(f"  无 transcript 跳过: {skipped_no_transcript} 个")


if __name__ == "__main__":
    main()
