import argparse
import os
import subprocess
import pandas as pd
import ast
import re
from difflib import SequenceMatcher
from tqdm import tqdm
from jiwer import wer

def infer_split_name(samples_csv_path: str) -> str:
    name = os.path.basename(samples_csv_path).lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return "dev"


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_episode_map(map_csv_path: str, map_root: str) -> pd.DataFrame:
    df = pd.read_csv(map_csv_path)

    if not map_root:
        raise ValueError("map_root is required to resolve relative Filepath values.")

    def to_abs(p: str) -> str:
        p = str(p)
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(map_root, p))

    df["Filepath"] = df["Filepath"].apply(to_abs)
    return df


def find_episode_filepath(ep_map_df: pd.DataFrame, season: int, episode: int) -> str | None:
    row = ep_map_df[(ep_map_df["Season"] == season) & (ep_map_df["Episode"] == episode)]
    if row.empty:
        return None
    return row.iloc[0]["Filepath"]


def normalize_timecode(t: str) -> str:
    s = str(t).strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 2:
        # MM:SS.xxx -> 00:MM:SS.xxx
        s = f"00:{s}"
    elif len(parts) == 1:
        # SS.xxx -> 00:00:SS.xxx
        s = f"00:00:{s}"
    return s


def cut_clip(input_path: str, start_time: str, end_time: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", normalize_timecode(start_time),
        "-to", normalize_timecode(end_time),
        "-i", input_path,
        "-c:v", "copy",       # 关键：视频流直接复制，画质绝对无损
        "-c:a", "aac",        # 音频转为 AAC，保证 MP4 兼容性
        "-b:a", "192k",       # 音频码率，192k 听感很好
        "-y",
        output_path,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def extract_ost_audio(input_path: str, start_time: str, end_time: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", normalize_timecode(start_time),
        "-to", normalize_timecode(end_time),
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def parse_list_field(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [s]


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def calculate_wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref)
    h = normalize_text(hyp)
    # if not r:
    #     return 1.0 if h else 0.0
    return wer(r, h)


def load_whisper_model(model_name: str):
    try:
        import whisper  # type: ignore
    except Exception as exc:
        raise RuntimeError("ASR检查需要安装openai-whisper：pip install -U openai-whisper") from exc
    return whisper.load_model(model_name)


def transcribe_audio_whisper(model, audio_path: str, language: str | None = None) -> str:
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = model.transcribe(audio_path, **kwargs)
    return (result.get("text") or "").strip()


def merge_consecutive_emotions(emotions, utterances):
    if not emotions:
        return emotions, utterances

    merged_emotions = []
    merged_utterances = [] if utterances is not None else None

    for idx, emo in enumerate(emotions):
        utt = ""
        if utterances is not None and idx < len(utterances):
            utt = str(utterances[idx])

        if merged_emotions and emo == merged_emotions[-1]:
            if merged_utterances is not None and merged_utterances:
                if utt:
                    prev = str(merged_utterances[-1])
                    merged_utterances[-1] = (prev + " " + utt).strip()
            continue

        merged_emotions.append(emo)
        if merged_utterances is not None:
            merged_utterances.append(utt)

    return merged_emotions, merged_utterances


def process_samples(samples_csv_path: str, episode_map_csv_path: str, output_dir: str, map_root: str,
                    asr_check: bool = False, asr_model: str = "base", asr_threshold: float = 0.5,
                    asr_language: str | None = None, asr_keep_bad: bool = False,
                    test_mode: bool = False) -> None:
    samples_df = pd.read_csv(samples_csv_path)
    ep_map_df = build_episode_map(episode_map_csv_path, map_root=map_root)

    split_name = infer_split_name(samples_csv_path)
    
    # Test mode: skip if not dev
    if test_mode and split_name != "dev":
        print(f"Test mode: skipping {split_name} split")
        return
    
    # New structure: no split subdirectories
    clips_dir = os.path.join(output_dir, "videos")
    ost_dir = os.path.join(output_dir, "audios", "ost")
    ensure_dir(clips_dir)
    ensure_dir(ost_dir)

    output_rows = []
    success = 0
    skipped = 0
    discarded = 0
    asr_warnings = []

    whisper_model = load_whisper_model(asr_model) if asr_check else None

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Processing samples"):
        season = row.get("Season")
        episode = row.get("Episode")
        start_time = row.get("Start_Time")
        end_time = row.get("End_Time")

        sample_id = idx + 1
        clip_filename = f"{split_name}_sample_{sample_id}.mp4"
        clip_path = os.path.join(clips_dir, clip_filename)
        ost_filename = f"{split_name}_sample_{sample_id}.wav"
        ost_path = os.path.join(ost_dir, ost_filename)

        clip_ok = False
        ost_ok = False
        input_filepath = None
        if pd.notna(season) and pd.notna(episode):
            input_filepath = find_episode_filepath(ep_map_df, int(season), int(episode))

        if input_filepath and os.path.exists(str(input_filepath)):
            if pd.notna(start_time) and pd.notna(end_time):
                clip_ok = cut_clip(str(input_filepath), start_time, end_time, clip_path)
                ost_ok = extract_ost_audio(str(input_filepath), start_time, end_time, ost_path)

        if clip_ok:
            success += 1
        else:
            skipped += 1

        out_row = row.to_dict()
        out_row["Split"] = split_name  # Add split information
        emotions = parse_list_field(out_row.get("Emotions"))
        utterances = parse_list_field(out_row.get("Utterances"))
        merged_utterances = None
        if emotions:
            merged_emotions, merged_utterances = merge_consecutive_emotions(emotions, utterances)
            out_row["Emotions"] = merged_emotions
            if merged_utterances is not None:
                out_row["Utterances"] = merged_utterances
            out_row["Length"] = len(merged_emotions)
        out_row["Sample_ID"] = sample_id
        out_row["Clip_Filename"] = clip_filename if clip_ok else ""
        out_row["Clip_Path"] = os.path.relpath(clip_path, output_dir) if clip_ok else ""
        out_row["ost_path"] = os.path.relpath(ost_path, output_dir) if ost_ok else ""

        asr_bad = False
        if asr_check and ost_ok and merged_utterances is not None:
            ref_text = " ".join([str(u) for u in merged_utterances if str(u).strip()])
            if ref_text.strip():
                try:
                    asr_text = transcribe_audio_whisper(whisper_model, ost_path, asr_language)
                    wer = calculate_wer(ref_text, asr_text)
                    if wer > asr_threshold:
                        asr_bad = True
                        asr_warnings.append({
                            "Sample_ID": sample_id,
                            "WER": wer,
                            "ASR_Text": asr_text,
                            "Ref_Text": ref_text,
                            "ost_path": os.path.relpath(ost_path, output_dir),
                        })
                except Exception as exc:
                    asr_bad = True
                    asr_warnings.append({
                        "Sample_ID": sample_id,
                        "WER": -1.0,
                        "ASR_Text": "",
                        "Ref_Text": ref_text,
                        "ost_path": os.path.relpath(ost_path, output_dir),
                        "Error": str(exc),
                    })

        if asr_bad and not asr_keep_bad:
            discarded += 1
            if clip_ok and os.path.exists(clip_path):
                os.remove(clip_path)
            if ost_ok and os.path.exists(ost_path):
                os.remove(ost_path)
            continue

        output_rows.append(out_row)

    out_df = pd.DataFrame(output_rows)
    # Reorder columns: put Sample_ID and Split first
    if not out_df.empty:
        cols = ['Sample_ID', 'Split'] + [c for c in out_df.columns if c not in ['Sample_ID', 'Split']]
        out_df = out_df[cols]

    # Output to metadata.csv (append mode if exists)
    output_csv_path = os.path.join(output_dir, "metadata.csv")
    
    # Append or write new
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        out_df = pd.concat([existing_df, out_df], ignore_index=True)
    
    out_df.to_csv(output_csv_path, index=False)

    print(f"Split: {split_name}")
    print(f"Clips output dir: {clips_dir}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Clips created: {success}, skipped: {skipped}")
    if asr_check:
        print(f"ASR丢弃样本数: {discarded}")

    if asr_check:
        if asr_warnings:
            print(f"ASR检查发现问题样本数: {len(asr_warnings)}")
            for item in asr_warnings:
                if "Error" in item:
                    print(f"[WARN] Sample_ID={item['Sample_ID']} ASR错误: {item['Error']}")
                else:
                    print(f"[WARN] Sample_ID={item['Sample_ID']} WER={item['WER']:.3f} (Req < {asr_threshold})")
                    print(f"       Ref: {item['Ref_Text']}")
                    print(f"       Hyp: {item['ASR_Text']}")
        else:
            print("ASR检查未发现问题样本。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut MELD samples into video clips using ffmpeg and Friends episode map."
    )
    parser.add_argument("--episode-map", required=True, help="Path to friends_episode_map.csv.")
    parser.add_argument("--output-dir", required=True, help="Output directory; clips go to dir, csv to dir/metadata.csv.")
    parser.add_argument("--map-root", required=True, help="Root directory for relative Filepath in episode map.")
    parser.add_argument("--samples-dir", default=".", help="Directory containing dev.csv, train.csv, test.csv (default: current directory).")
    
    # ASR Args
    parser.add_argument("--asr-check", action="store_true", help="Enable ASR check for OST audio.")
    parser.add_argument("--asr-model", default="base", help="Whisper model name for ASR.")
    parser.add_argument("--asr-threshold", type=float, default=0.5, help="WER threshold (lower is better, default 0.5).")
    parser.add_argument("--asr-language", default="en", help="ASR language code (e.g., en, zh).")
    parser.add_argument("--asr-keep-bad", action="store_true", help="Keep samples that fail ASR check.")
    parser.add_argument("--test-mode", action="store_true", help="Test mode: only process dev split.")

    args = parser.parse_args()
    
    # Find sample CSV files
    splits = ["dev", "train", "test"]
    samples_files = []
    for split in splits:
        csv_path = os.path.join(args.samples_dir, f"{split}.csv")
        if os.path.exists(csv_path):
            samples_files.append(csv_path)
        else:
            print(f"Warning: {split}.csv not found in {args.samples_dir}")
    
    if not samples_files:
        print("Error: No sample CSV files found (dev.csv, train.csv, test.csv)")
        exit(1)
    
    # Clear metadata.csv if it exists (for fresh start)
    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        print(f"Removing existing {metadata_path}")
        os.remove(metadata_path)
    
    # Process each split
    for samples_csv in samples_files:
        print(f"\n{'='*60}")
        print(f"Processing: {samples_csv}")
        print(f"{'='*60}")
        process_samples(
            samples_csv,
            args.episode_map,
            args.output_dir,
            map_root=args.map_root,
            asr_check=args.asr_check,
            asr_model=args.asr_model,
            asr_threshold=args.asr_threshold,
            asr_language=args.asr_language,
            asr_keep_bad=args.asr_keep_bad,
            test_mode=args.test_mode
        )