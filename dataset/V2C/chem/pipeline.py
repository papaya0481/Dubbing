import argparse
import os
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_path,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def transcribe_audio_whisper(model, audio_path: str, language: str | None = None) -> str:
    """Transcribe audio using Whisper model."""
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = model.transcribe(audio_path, **kwargs)
    return (result.get("text") or "").strip()


def load_whisper_model(model_name: str):
    """Load Whisper model."""
    try:
        import whisper  # type: ignore
    except Exception as exc:
        raise RuntimeError("需要安装openai-whisper：pip install -U openai-whisper") from exc
    return whisper.load_model(model_name, download_root="/data2/ruixin/.cache/whisper", device="cuda")


def collect_video_sources(input_path: Path) -> list[dict[str, str | Path]]:
    """Collect videos from either nested cut folders or a flat videos directory."""
    flat_videos_dir = input_path / "videos"
    if flat_videos_dir.is_dir():
        flat_video_files = sorted(flat_videos_dir.glob("*.mp4"))
        if flat_video_files:
            return [
                {
                    "video_id": video_file.stem,
                    "cut_num": "",
                    "base_name": video_file.stem,
                    "source_path": video_file,
                }
                for video_file in flat_video_files
            ]

    nested_sources = []
    for video_folder in sorted([path for path in input_path.iterdir() if path.is_dir()]):
        video_id = video_folder.name
        for video_file in sorted(video_folder.glob("cut-*.mp4")):
            cut_parts = video_file.stem.split("-", maxsplit=1)
            cut_num = cut_parts[1] if len(cut_parts) > 1 else ""
            base_name = f"{video_id}_cut{cut_num}" if cut_num else f"{video_id}_{video_file.stem}"
            nested_sources.append({
                "video_id": video_id,
                "cut_num": cut_num,
                "base_name": base_name,
                "source_path": video_file,
            })

    if nested_sources:
        return nested_sources

    root_video_files = sorted(input_path.glob("*.mp4"))
    return [
        {
            "video_id": video_file.stem,
            "cut_num": "",
            "base_name": video_file.stem,
            "source_path": video_file,
        }
        for video_file in root_video_files
    ]


def process_videos(intervals_dir: str, output_dir: str, whisper_model_name: str = "large",
                   language: str | None = "en") -> None:
    process_videos_new(
        intervals_dir=intervals_dir,
        output_dir=output_dir,
        whisper_model_name=whisper_model_name,
        language=language,
    )
    
def process_videos_new(intervals_dir: str, output_dir: str, whisper_model_name: str = "large",
                   language: str | None = "en") -> None:
    """
    Process videos in intervals directory:
    1. Extract audio from each video clip
    2. Transcribe audio using Whisper
    3. Create metadata CSV
    """
    intervals_path = Path(intervals_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    audios_dir = output_path / "audios" / "ost"
    videos_dir = output_path / "videos"
    ensure_dir(str(audios_dir))
    ensure_dir(str(videos_dir))
    
    # Load Whisper model
    print(f"Loading Whisper model: {whisper_model_name}")
    whisper_model = load_whisper_model(whisper_model_name)
    
    # Collect all video files
    video_sources = collect_video_sources(intervals_path)
    if not video_sources:
        raise FileNotFoundError(
            f"No mp4 files found in {intervals_dir}. Supported layouts: <root>/videos/*.mp4 or <root>/<video_id>/cut-*.mp4"
        )
    
    metadata_rows = []
    sample_id = 1
    
    print(f"Found {len(video_sources)} video files")
    
    for video_source in tqdm(video_sources, desc="Processing videos"):
        video_file = Path(video_source["source_path"])
        video_id = str(video_source["video_id"])
        cut_num = str(video_source["cut_num"])
        base_name = str(video_source["base_name"])

        audio_filename = f"{base_name}.wav"
        txt_filename = f"{base_name}.txt"
        audio_path = audios_dir / audio_filename
        txt_path = audios_dir / txt_filename

        # Copy video to videos folder with new name
        video_filename = f"{base_name}.mp4"
        video_dest = videos_dir / video_filename

        # Extract audio
        audio_ok = extract_audio_from_video(str(video_file), str(audio_path))

        # Copy video file
        video_ok = False
        if audio_ok:
            try:
                shutil.copy2(str(video_file), str(video_dest))
                video_ok = True
            except Exception as e:
                print(f"Error copying video {video_file}: {e}")

        # Transcribe audio
        transcription = ""
        if audio_ok:
            try:
                transcription = transcribe_audio_whisper(whisper_model, str(audio_path), language)
                # Save transcription to txt file
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(transcription)
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")

        # Add to metadata
        metadata_rows.append({
            "Sample_ID": sample_id,
            "Video_ID": video_id,
            "Cut_Number": cut_num,
            "Video_Path": f"videos/{video_filename}" if video_ok else "",
            "Audio_Path": f"audios/ost/{audio_filename}" if audio_ok else "",
            "Transcription_Path": f"audios/ost/{txt_filename}" if audio_ok else "",
            "Transcription": transcription,
            "Original_Path": str(video_file.relative_to(intervals_path)),
        })

        sample_id += 1
    
    # Create metadata CSV
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_csv_path = output_path / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total samples: {len(metadata_rows)}")
    print(f"Output directory: {output_dir}")
    print(f"Audios directory: {audios_dir}")
    print(f"Videos directory: {videos_dir}")
    print(f"Metadata CSV: {metadata_csv_path}")
    print(f"{'='*60}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process chem video clips: extract audio, transcribe with Whisper, create metadata."
    )
    parser.add_argument(
        "--intervals-dir",
        default="/data2/ruixin/downloads/chem_raw_no_processed",
        help="Path to input videos. Supports <root>/videos/*.mp4 and <root>/<video_id>/cut-*.mp4"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--whisper-model",
        default="large",
        help="Whisper model name (default: large)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for Whisper (default: en)"
    )
    
    args = parser.parse_args()
    
    process_videos_new(
        intervals_dir=args.intervals_dir,
        output_dir=args.output_dir,
        whisper_model_name=args.whisper_model,
        language=args.language
    )
