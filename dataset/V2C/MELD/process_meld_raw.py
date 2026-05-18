import argparse
import os
import subprocess
import pandas as pd
import re
import shutil
from tqdm import tqdm
from jiwer import wer
import glob

# ==========================================
# Helpers (Adapted from extract_clips.py)
# ==========================================

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def calculate_wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref)
    h = normalize_text(hyp)
    # Avoid division by zero if reference is empty
    if not r:
        return 1.0 if h else 0.0
    return wer(r, h)

def load_whisper_model(model_name: str):
    try:
        import whisper  # type: ignore
    except Exception as exc:
        raise RuntimeError("ASR requires openai-whisper: pip install -U openai-whisper") from exc
    return whisper.load_model(model_name)

def transcribe_audio_whisper(model, audio_path: str, language: str | None = None) -> str:
    kwargs = {}
    if language:
        kwargs["language"] = language
    try:
        result = model.transcribe(audio_path, **kwargs)
        return (result.get("text") or "").strip()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

def extract_audio(input_path: str, output_path: str) -> bool:
    """Extract audio from video using ffmpeg, converting to 16k mono PCM wav."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running ffmpeg for {input_path}: {e}")
        return False

# ==========================================
# Main Processing Logic
# ==========================================

def load_annotations(csv_path: str) -> dict:
    """Load MELD annotations into a dict: (Dialogue_ID, Utterance_ID) -> fields."""
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        did = row.get("Dialogue_ID")
        uid = row.get("Utterance_ID")
        text = row.get("Utterance")
        speaker = row.get("Speaker")
        emotion = row.get("Emotion", row.get("emotion"))
        if pd.notna(did) and pd.notna(uid):
            lookup[(int(did), int(uid))] = {
                "Utterance": "" if pd.isna(text) else str(text),
                "Speaker": "" if pd.isna(speaker) else str(speaker),
                "emotion": "" if pd.isna(emotion) else str(emotion),
            }
    return lookup

def parse_filename(filename: str):
    """Parse diaX_uttY.mp4"""
    # Standard format: diaX_uttY.mp4
    match = re.search(r"dia(\d+)_utt(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def process_split(
    split_name: str,
    video_dir: str,
    csv_path: str,
    output_base_dir: str,
    model,
    language: str = "en",
    sample_id_start: int = 1
):
    print(f"\nProcessing SPLIT: {split_name}")
    print(f"  Videos: {video_dir}")
    print(f"  CSV: {csv_path}")

    if not os.path.exists(video_dir):
        print(f"  [Error] Video directory not found: {video_dir}")
        return [], 0

    if not os.path.exists(csv_path):
        print(f"  [Error] Transcript CSV not found: {csv_path}")
        return [], 0

    # Load annotations
    annotations = load_annotations(csv_path)

    # Prepare flat output dirs shared across splits
    audio_dir = os.path.join(output_base_dir, "audios", "ost")
    video_out_dir = os.path.join(output_base_dir, "videos")
    ensure_dir(audio_dir)
    ensure_dir(video_out_dir)

    # Find videos
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    stats_total = 0
    stats_kept = 0
    stats_failed_extraction = 0
    stats_missing_transcript = 0

    records = []
    sample_id = sample_id_start

    for vid_path in tqdm(video_files, desc=f"{split_name}"):
        stats_total += 1
        src_filename = os.path.basename(vid_path)
        did, uid = parse_filename(src_filename)

        if did is None or uid is None:
            continue

        # Get Reference Text
        ref_entry = annotations.get((did, uid))
        if ref_entry is None:
            stats_missing_transcript += 1
            continue
        ref_text = ref_entry["Utterance"]
        speaker = ref_entry["Speaker"]
        emotion = ref_entry["emotion"]

        # File base name shared by audio and video outputs
        base_name = f"{split_name}_dia{did}_utt{uid}"
        wav_filename = base_name + ".wav"
        clip_filename = base_name + ".mp4"
        wav_path = os.path.join(audio_dir, wav_filename)
        clip_path = os.path.join(video_out_dir, clip_filename)

        # Extract Audio only when target wav does not already exist.
        audio_ok = False
        if os.path.exists(wav_path):
            audio_ok = True
        else:
            audio_ok = extract_audio(vid_path, wav_path)

        if not audio_ok:
            stats_failed_extraction += 1
            continue

        # Copy Video only when target mp4 does not already exist.
        clip_ok = False
        if os.path.exists(clip_path):
            clip_ok = True
        else:
            try:
                shutil.copy2(vid_path, clip_path)
                clip_ok = True
            except Exception as e:
                print(f"  [Warn] Failed to copy video {vid_path}: {e}")

        # ASR & WER
        hyp_text = transcribe_audio_whisper(model, wav_path, language=language)
        error_rate = calculate_wer(ref_text, hyp_text)
        stats_kept += 1

        records.append({
            "Sample_ID": sample_id,
            "Split": split_name,
            "Dialogue_ID": did,
            "Utterance_ID": uid,
            "Audio_Filename": wav_filename,
            "Audio_Path": os.path.relpath(wav_path, output_base_dir),
            "Video_Filename": clip_filename if clip_ok else "",
            "Video_Path": os.path.relpath(clip_path, output_base_dir) if clip_ok else "",
            "Utterance": ref_text,
            "Speaker": speaker,
            "Emotion": emotion,
            "ASR_Text": hyp_text,
            "WER": round(error_rate, 4),
        })
        sample_id += 1

    print(f"  Total Videos Found: {len(video_files)}")
    print(f"  Processed: {stats_total}")
    print(f"  Kept: {stats_kept}")
    print(f"  Missing Transcript: {stats_missing_transcript}")
    print(f"  Failed Extraction: {stats_failed_extraction}")

    return records, sample_id

def main():
    parser = argparse.ArgumentParser(description="Process MELD Raw Videos: Extract Audio -> ASR -> Filter by WER")
    parser.add_argument("--meld-raw", default="/data2/ruixin/downloads/MELD-RAW/MELD.Raw", help="Path to MELD.Raw root")
    parser.add_argument("--output-dir", required=True, help="Target directory for output")
    parser.add_argument("--csv-dir", default="./MELD", help="Directory containing *_sent_emo.csv files")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language for ASR")

    args = parser.parse_args()
    
    # Setup MELD paths based on known structure
    # dev -> dev/dev_splits_complete
    # train -> train/train_splits
    # test -> test/output_repeated_splits_test
    
    structure = {
        "dev": {
            "vid_sub": "dev/dev_splits_complete",
            "csv": "dev_sent_emo.csv"
        },
        "train": {
            "vid_sub": "train/train_splits",
            "csv": "train_sent_emo.csv"
        },
        "test": {
            "vid_sub": "test/output_repeated_splits_test",
            "csv": "test_sent_emo.csv"
        }
    }
    
    print("Loading Whisper Model...")
    model = load_whisper_model(args.model)
    
    all_records = []
    next_sample_id = 1

    for split, info in structure.items():
        vid_dir = os.path.join(args.meld_raw, info["vid_sub"])
        csv_path = os.path.join(args.csv_dir, info["csv"])

        split_records, next_sample_id = process_split(
            split,
            vid_dir,
            csv_path,
            args.output_dir,
            model,
            args.language,
            sample_id_start=next_sample_id,
        )
        all_records.extend(split_records)

    # Save Metadata (always write metadata.csv even if empty)
    ensure_dir(args.output_dir)
    priority_cols = ["Sample_ID", "Split", "Dialogue_ID", "Utterance_ID",
                     "Audio_Filename", "Audio_Path", "Video_Filename", "Video_Path",
                     "Utterance", "Speaker", "Emotion", "ASR_Text", "WER"]

    if all_records:
        df = pd.DataFrame(all_records)
        ordered_cols = priority_cols + [c for c in df.columns if c not in priority_cols]
        df = df[ordered_cols]
    else:
        df = pd.DataFrame(columns=priority_cols)

    meta_path = os.path.join(args.output_dir, "metadata.csv")
    df.to_csv(meta_path, index=False)
    print(f"\nMetadata saved to: {meta_path}")

    # summary
    print(f"Total Samples: {len(df)}")
    if len(df) > 0:
        print(f"Mean WER: {df['WER'].mean():.4f}")
    else:
        print("Mean WER: N/A (no samples)")

if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import pandas as pd
import re
import shutil
from tqdm import tqdm
from jiwer import wer
import glob

# ==========================================
# Helpers (Adapted from extract_clips.py)
# ==========================================

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def calculate_wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref)
    h = normalize_text(hyp)
    # Avoid division by zero if reference is empty
    if not r:
        return 1.0 if h else 0.0
    return wer(r, h)

def load_whisper_model(model_name: str):
    try:
        import whisper  # type: ignore
    except Exception as exc:
        raise RuntimeError("ASR requires openai-whisper: pip install -U openai-whisper") from exc
    return whisper.load_model(model_name)

def transcribe_audio_whisper(model, audio_path: str, language: str | None = None) -> str:
    kwargs = {}
    if language:
        kwargs["language"] = language
    try:
        result = model.transcribe(audio_path, **kwargs)
        return (result.get("text") or "").strip()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

def extract_audio(input_path: str, output_path: str) -> bool:
    """Extract audio from video using ffmpeg, converting to 16k mono PCM wav."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running ffmpeg for {input_path}: {e}")
        return False

# ==========================================
# Main Processing Logic
# ==========================================

def load_transcripts(csv_path: str) -> dict:
    """Load transcripts from MELD CSV into a dict: (Dialogue_ID, Utterance_ID) -> Text"""
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        did = row.get("Dialogue_ID")
        uid = row.get("Utterance_ID")
        text = row.get("Utterance")
        if pd.notna(did) and pd.notna(uid):
            lookup[(int(did), int(uid))] = str(text)
    return lookup

def parse_filename(filename: str):
    """Parse diaX_uttY.mp4"""
    # Standard format: diaX_uttY.mp4
    match = re.search(r"dia(\d+)_utt(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def process_split(
    split_name: str,
    video_dir: str,
    csv_path: str,
    output_base_dir: str,
    model,
    language: str = "en",
    sample_id_start: int = 1
):
    print(f"\nProcessing SPLIT: {split_name}")
    print(f"  Videos: {video_dir}")
    print(f"  CSV: {csv_path}")

    if not os.path.exists(video_dir):
        print(f"  [Error] Video directory not found: {video_dir}")
        return [], 0

    if not os.path.exists(csv_path):
        print(f"  [Error] Transcript CSV not found: {csv_path}")
        return [], 0

    # Load transcripts
    transcripts = load_transcripts(csv_path)

    # Prepare flat output dirs shared across splits
    audio_dir = os.path.join(output_base_dir, "audios", "ost")
    video_out_dir = os.path.join(output_base_dir, "videos")
    ensure_dir(audio_dir)
    ensure_dir(video_out_dir)

    # Find videos
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    stats_total = 0
    stats_kept = 0
    stats_failed_extraction = 0
    stats_missing_transcript = 0

    records = []
    sample_id = sample_id_start

    for vid_path in tqdm(video_files, desc=f"{split_name}"):
        stats_total += 1
        src_filename = os.path.basename(vid_path)
        did, uid = parse_filename(src_filename)

        if did is None or uid is None:
            continue

        # Get Reference Text
        ref_text = transcripts.get((did, uid))
        if ref_text is None:
            stats_missing_transcript += 1
            continue

        # File base name shared by audio and video outputs
        base_name = f"{split_name}_dia{did}_utt{uid}"
        wav_filename = base_name + ".wav"
        clip_filename = base_name + ".mp4"
        wav_path = os.path.join(audio_dir, wav_filename)
        clip_path = os.path.join(video_out_dir, clip_filename)

        # Extract Audio
        if not extract_audio(vid_path, wav_path):
            stats_failed_extraction += 1
            continue

        # Copy Video
        clip_ok = False
        try:
            shutil.copy2(vid_path, clip_path)
            clip_ok = True
        except Exception as e:
            print(f"  [Warn] Failed to copy video {vid_path}: {e}")

        # ASR & WER
        hyp_text = transcribe_audio_whisper(model, wav_path, language=language)
        error_rate = calculate_wer(ref_text, hyp_text)
        stats_kept += 1

        records.append({
            "Sample_ID": sample_id,
            "Split": split_name,
            "Dialogue_ID": did,
            "Utterance_ID": uid,
            "Audio_Filename": wav_filename,
            "Audio_Path": os.path.relpath(wav_path, output_base_dir),
            "Video_Filename": clip_filename if clip_ok else "",
            "Video_Path": os.path.relpath(clip_path, output_base_dir) if clip_ok else "",
            "Utterance": ref_text,
            "ASR_Text": hyp_text,
            "WER": round(error_rate, 4),
        })
        sample_id += 1

    print(f"  Total Videos Found: {len(video_files)}")
    print(f"  Processed: {stats_total}")
    print(f"  Kept: {stats_kept}")
    print(f"  Missing Transcript: {stats_missing_transcript}")
    print(f"  Failed Extraction: {stats_failed_extraction}")

    return records, sample_id

def main():
    parser = argparse.ArgumentParser(description="Process MELD Raw Videos: Extract Audio -> ASR -> Filter by WER")
    parser.add_argument("--meld-raw", default="/data2/ruixin/downloads/MELD-RAW/MELD.Raw", help="Path to MELD.Raw root")
    parser.add_argument("--output-dir", required=True, help="Target directory for output")
    parser.add_argument("--csv-dir", default="./MELD", help="Directory containing *_sent_emo.csv files")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language for ASR")

    args = parser.parse_args()
    
    # Setup MELD paths based on known structure
    # dev -> dev/dev_splits_complete
    # train -> train/train_splits
    # test -> test/output_repeated_splits_test
    
    structure = {
        "dev": {
            "vid_sub": "dev/dev_splits_complete",
            "csv": "dev_sent_emo.csv"
        },
        "train": {
            "vid_sub": "train/train_splits",
            "csv": "train_sent_emo.csv"
        },
        "test": {
            "vid_sub": "test/output_repeated_splits_test",
            "csv": "test_sent_emo.csv"
        }
    }
    
    print("Loading Whisper Model...")
    model = load_whisper_model(args.model)
    
    all_records = []
    next_sample_id = 1

    for split, info in structure.items():
        vid_dir = os.path.join(args.meld_raw, info["vid_sub"])
        csv_path = os.path.join(args.csv_dir, info["csv"])

        split_records, next_sample_id = process_split(
            split,
            vid_dir,
            csv_path,
            args.output_dir,
            model,
            args.language,
            sample_id_start=next_sample_id,
        )
        all_records.extend(split_records)

    # Save Metadata
    if all_records:
        df = pd.DataFrame(all_records)
        # Put key columns first
        priority_cols = ["Sample_ID", "Split", "Dialogue_ID", "Utterance_ID",
                         "Audio_Filename", "Audio_Path", "Video_Filename", "Video_Path",
                         "Utterance", "ASR_Text", "WER"]
        ordered_cols = priority_cols + [c for c in df.columns if c not in priority_cols]
        df = df[ordered_cols]
        meta_path = os.path.join(args.output_dir, "metadata.csv")
        df.to_csv(meta_path, index=False)
        print(f"\nMetadata saved to: {meta_path}")

        # summary
        print(f"Total Samples: {len(df)}")
        print(f"Mean WER: {df['WER'].mean():.4f}")

if __name__ == "__main__":
    main()
