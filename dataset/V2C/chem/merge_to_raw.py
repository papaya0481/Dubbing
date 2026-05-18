"""Merge CHEM clipped data into MELD-like raw dataset layout.

Directory layout expected under --input-videos-dir:
  <video_id>/
    metadata.csv
    clipped/
      *.mp4
      *.wav
    clip_wer.csv or wer.csv (optional)

Output layout under --output-dir:
  output_dir/
    videos/
    audios/ost/
    metadata.csv
        wer_stats.txt
"""

from __future__ import annotations

import argparse
import shutil
import re
import importlib
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
from detect_face import RetinaFace, detect_faces_in_video


MELD_LIKE_COLUMNS = [
    "Sample_ID",
    "Split",
    "Dialogue_ID",
    "Utterance_ID",
    "Audio_Filename",
    "Audio_Path",
    "Video_Filename",
    "Video_Path",
    "Utterance",
    "Speaker",
    "Emotion",
    "ASR_Text",
    "WER",
]

_detector = None


def init_worker(gpu_id):
    """Initialize detector once per worker."""
    global _detector
    _detector = RetinaFace(gpu_id=gpu_id)


def check_video_faces(video_path):
    """Check if video has faces in all frames (smoothed)."""
    _, _, _, all_detected_smoothed = detect_faces_in_video(video_path, _detector)
    return str(video_path), all_detected_smoothed


class WeTextUtteranceNormalizer:
    """Normalize utterance with WeTextProcessing while preserving contractions."""

    def __init__(self) -> None:
        self._tn_zh = None
        self._tn_en = None
        try:
            zh_module = importlib.import_module("tn.chinese.normalizer")
            en_module = importlib.import_module("tn.english.normalizer")
            ChineseNormalizer = getattr(zh_module, "Normalizer")
            EnglishNormalizer = getattr(en_module, "Normalizer")

            self._tn_zh = ChineseNormalizer()
            self._tn_en = EnglishNormalizer()
            print("[INFO] WeTextProcessing normalizer initialized.")
        except Exception as exc:
            print(f"[WARN] WeTextProcessing unavailable, keep original utterance. reason={exc}")

        # Keep common English contractions unchanged, e.g. there's / don't / it's.
        self._contraction_pattern = re.compile(r"\b\w+'(?:s|re|ve|ll|d|m|t)\b", re.IGNORECASE)

    @staticmethod
    def _index_to_letters(index: int) -> str:
        """Convert index to alphabetic ID: 0 -> a, 25 -> z, 26 -> aa."""
        chars = []
        i = index
        while True:
            chars.append(chr(ord("a") + (i % 26)))
            i = i // 26 - 1
            if i < 0:
                break
        return "".join(reversed(chars))

    def _protect_contractions(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace contractions with placeholders so only those tokens bypass normalization."""
        placeholder_map: dict[str, str] = {}

        def repl(match: re.Match[str]) -> str:
            key = f"zzkeep{self._index_to_letters(len(placeholder_map))}zz"
            placeholder_map[key] = match.group(0)
            return key

        protected = self._contraction_pattern.sub(repl, text)
        return protected, placeholder_map

    @staticmethod
    def _restore_placeholders(text: str, placeholder_map: dict[str, str]) -> str:
        restored = text
        for key, original in placeholder_map.items():
            restored = re.sub(re.escape(key), original, restored, flags=re.IGNORECASE)
        return restored

    def _detect_lang(self, text: str) -> str:
        if not text:
            return "en"
        zh_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        total_chars = len(text)
        if total_chars == 0:
            return "en"
        return "zh" if zh_count / total_chars > 0.3 else "en"

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        text = text.strip()
        if not text:
            return text

        # Protect contractions, but still normalize the rest of the sentence.
        text_to_normalize, placeholder_map = self._protect_contractions(text)

        if self._tn_zh is None and self._tn_en is None:
            return text

        lang = self._detect_lang(text_to_normalize)
        try:
            if lang == "zh" and self._tn_zh is not None:
                out = self._tn_zh.normalize(text_to_normalize)
                if not out:
                    return text
                return self._restore_placeholders(out, placeholder_map)
            if lang == "en" and self._tn_en is not None:
                out = self._tn_en.normalize(text_to_normalize)
                if not out:
                    return text
                return self._restore_placeholders(out, placeholder_map)
            return text
        except Exception:
            return text


class UtteranceFilter:
    """Filter out utterances matching blocked patterns."""

    def should_filter(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        return ("+" in text) or ("-" in text)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if df.empty:
        return []
    return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in df.to_dict(orient="records")]


def write_csv_rows(csv_path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    ensure_dir(csv_path.parent)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns].fillna("")
    df.to_csv(csv_path, index=False, encoding="utf-8")


def write_wer_stats_txt(output_txt_path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(output_txt_path.parent)

    if not rows:
        output_txt_path.write_text("No rows available.\n", encoding="utf-8")
        return

    df = pd.DataFrame(rows)
    if "WER" not in df.columns:
        output_txt_path.write_text("No WER column found.\n", encoding="utf-8")
        return

    raw = df["WER"].astype(str).str.strip()
    raw = raw.mask(raw == "", pd.NA)
    is_percent = raw.str.endswith("%", na=False)
    numeric = pd.to_numeric(raw.str.rstrip("%"), errors="coerce")
    numeric = numeric.where(~is_percent, numeric / 100.0)

    valid = numeric.dropna()
    total_rows = len(df)
    valid_count = len(valid)
    invalid_count = total_rows - valid_count

    bin_edges = [i / 10 for i in range(11)]
    bin_labels = [f"[{i/10:.1f}, {(i+1)/10:.1f})" for i in range(9)] + ["[0.9, 1.0]"]
    in_range = valid[(valid >= 0.0) & (valid <= 1.0)]
    hist_counts = (
        pd.cut(in_range, bins=bin_edges, labels=bin_labels, include_lowest=True, right=True)
        .value_counts(sort=False)
        .reindex(bin_labels, fill_value=0)
    )
    below_zero = int((valid < 0.0).sum())
    above_one = int((valid > 1.0).sum())

    max_count = int(hist_counts.max()) if len(hist_counts) > 0 else 0
    bar_width = 40

    lines = [
        "WER Histogram Report (bin=0.1)",
        f"Total rows      : {total_rows}",
        f"Valid WER       : {valid_count}",
        f"Invalid WER     : {invalid_count}",
        f"WER < 0         : {below_zero}",
        f"WER > 1         : {above_one}",
        "",
    ]

    for label, count in hist_counts.items():
        scaled = int(round((int(count) / max_count) * bar_width)) if max_count > 0 else 0
        bar = "#" * scaled
        lines.append(f"{label:12} | {bar:<40} ({int(count)})")

    output_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_first_existing(path_candidates: list[Path]) -> Path | None:
    for p in path_candidates:
        if p.exists() and p.is_file():
            return p
    return None


def get_first_non_empty(row: dict[str, str], candidates: list[str]) -> str:
    lower_map = {k.lower(): k for k in row.keys()}
    for c in candidates:
        key = lower_map.get(c.lower())
        if key is None:
            continue
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def build_wer_map(wer_csv_path: Path | None) -> dict[str, dict[str, str]]:
    if wer_csv_path is None:
        return {}

    rows = read_csv_rows(wer_csv_path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        clip_name = get_first_non_empty(
            row,
            [
                "Clip_Name",
                "clip_name",
                "Video_Filename",
                "video_filename",
                "Audio_Filename",
                "audio_filename",
                "filename",
                "file_name",
                "name",
            ],
        )
        asr_text = get_first_non_empty(
            row,
            [
                "ASR_Text",
                "asr_text",
                "Transcription",
                "transcription",
                "Recognized_Text",
                "recognized_text",
                "Hypothesis",
                "hypothesis",
                "text",
            ],
        )
        wer_value = get_first_non_empty(row, ["WER", "wer"])

        if not clip_name:
            continue

        key_name = Path(clip_name).name
        key_stem = Path(key_name).stem
        payload = {"ASR_Text": asr_text, "WER": wer_value}
        result[key_name] = payload
        result[key_stem] = payload

    return result


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if not dst.exists():
        shutil.copy2(src, dst)
    return True


def unique_filename(name: str, used_names: set[str]) -> str:
    candidate = name
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 1
    while candidate in used_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(candidate)
    return candidate


def merge_to_raw(input_videos_dir: Path, output_dir: Path, split_name: str, gpu_id: int = 0, workers: int = 4) -> None:
    videos_dir = output_dir / "videos"
    audios_dir = output_dir / "audios" / "ost"
    ensure_dir(videos_dir)
    ensure_dir(audios_dir)

    sample_id = 1
    all_rows: list[dict[str, str]] = []
    no_normed_rows: list[dict[str, str]] = []
    normalizer = WeTextUtteranceNormalizer()
    utterance_filter = UtteranceFilter()
    filtered_count = 0
    face_filtered_count = 0

    used_video_names: set[str] = set()
    used_audio_names: set[str] = set()

    video_folders = sorted([p for p in input_videos_dir.iterdir() if p.is_dir()])
    print(f"Found {len(video_folders)} video folders under: {input_videos_dir}")

    # Collect all video paths first
    all_video_paths = []
    for folder in video_folders:
        clipped_dir = folder / "clipped"
        if clipped_dir.exists() and clipped_dir.is_dir():
            all_video_paths.extend(sorted(clipped_dir.glob("*.mp4")))

    # Run face detection in parallel
    print(f"Running face detection on {len(all_video_paths)} videos with {workers} workers...")
    face_results = {}
    with Pool(processes=workers, initializer=init_worker, initargs=(gpu_id,)) as pool:
        for video_path, has_faces in tqdm(pool.imap(check_video_faces, all_video_paths), total=len(all_video_paths), desc="Face detection"):
            face_results[video_path] = has_faces

    for folder in video_folders:
        clipped_dir = folder / "clipped"
        metadata_csv = folder / "metadata.csv"

        if not clipped_dir.exists() or not clipped_dir.is_dir() or not metadata_csv.exists():
            print(f"[SKIP] Missing clipped dir or metadata.csv: {folder}")
            continue

        wer_csv = pick_first_existing(
            [
                folder / "clip_wer.csv",
                folder / "wer.csv",
                clipped_dir / "clip_wer.csv",
                clipped_dir / "wer.csv",
            ]
        )
        wer_map = build_wer_map(wer_csv)

        try:
            metadata_rows = read_csv_rows(metadata_csv)
        except Exception as exc:
            print(f"[WARN] Failed to read {metadata_csv}: {exc}")
            continue

        for utt_idx, row in enumerate(metadata_rows):
            clip_name = get_first_non_empty(
                row,
                ["Clip_Name", "clip_name", "Video_Filename", "video_filename", "Video_Name", "video_name"],
            )
            if not clip_name:
                continue

            clip_filename = Path(clip_name).name
            if not clip_filename.lower().endswith(".mp4"):
                clip_filename = f"{Path(clip_filename).stem}.mp4"

            wav_filename = f"{Path(clip_filename).stem}.wav"
            src_video = clipped_dir / clip_filename
            src_audio = clipped_dir / wav_filename

            # Face detection filter: skip videos with at least one frame without face
            if src_video.exists() and str(src_video) in face_results:
                if not face_results[str(src_video)]:
                    face_filtered_count += 1
                    continue

            # Prevent collisions across folders while preserving source names when possible.
            out_video_filename = unique_filename(clip_filename, used_video_names)
            out_audio_filename = unique_filename(wav_filename, used_audio_names)
            out_video_path = videos_dir / out_video_filename
            out_audio_path = audios_dir / out_audio_filename

            video_ok = safe_copy(src_video, out_video_path)
            audio_ok = safe_copy(src_audio, out_audio_path)

            if not video_ok:
                print(f"[WARN] Missing video file: {src_video}")
            if not audio_ok:
                print(f"[WARN] Missing audio file: {src_audio}")

            wer_payload = wer_map.get(clip_filename) or wer_map.get(Path(clip_filename).stem) or {}
            utterance = get_first_non_empty(row, ["Utterance", "utterance", "Text", "text"])
            if utterance_filter.should_filter(utterance):
                filtered_count += 1
                continue
            utterance_normed = normalizer.normalize(utterance)

            merged = {
                "Sample_ID": str(sample_id),
                "Split": split_name,
                "Dialogue_ID": folder.name,
                "Utterance_ID": str(utt_idx),
                "Audio_Filename": out_audio_filename if audio_ok else "",
                "Audio_Path": f"audios/ost/{out_audio_filename}" if audio_ok else "",
                "Video_Filename": out_video_filename if video_ok else "",
                "Video_Path": f"videos/{out_video_filename}" if video_ok else "",
                "Utterance": utterance_normed,
                "Speaker": "None",
                "Emotion": "None",
                "ASR_Text": wer_payload.get("ASR_Text", ""),
                "WER": wer_payload.get("WER", ""),
            }

            merged_no_normed = dict(merged)
            merged_no_normed["Utterance"] = utterance

            all_rows.append(merged)
            no_normed_rows.append(merged_no_normed)
            sample_id += 1

    metadata_no_normed_out = output_dir / "no_normed.csv"
    write_csv_rows(metadata_no_normed_out, no_normed_rows, MELD_LIKE_COLUMNS)

    metadata_out = output_dir / "metadata.csv"
    write_csv_rows(metadata_out, all_rows, MELD_LIKE_COLUMNS)
    wer_stats_out = output_dir / "wer_stats.txt"
    write_wer_stats_txt(wer_stats_out, all_rows)

    print("=" * 60)
    print("Merge complete")
    print(f"Rows written: {len(all_rows)}")
    print(f"Rows filtered by +/- rule: {filtered_count}")
    print(f"Rows filtered by face detection: {face_filtered_count}")
    print(f"Output metadata: {metadata_out}")
    print(f"Output no_normed: {metadata_no_normed_out}")
    print(f"WER stats txt: {wer_stats_out}")
    print(f"Videos dir: {videos_dir}")
    print(f"Audios dir: {audios_dir}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CHEM processed clips into MELD-like raw data structure."
    )
    parser.add_argument(
        "--input-dir",
        default="/data2/ruixin/downloads/chem_raw_processed2/videos",
        help="Directory containing per-video folders with clipped files and metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Will create videos/, audios/ost/, and metadata.csv",
    )
    parser.add_argument(
        "--split",
        default="chem",
        help="Value used for Split column in metadata.csv (default: chem)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for face detection (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for face detection (default: 4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_to_raw(
        input_videos_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        split_name=args.split,
        gpu_id=args.gpu_id,
        workers=args.workers,
    )