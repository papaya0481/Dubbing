import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
    7: "others",
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_srt_file(path: str) -> Dict[str, Tuple[str, str, str]]:
    entries: Dict[str, Tuple[str, str, str]] = {}
    if not os.path.exists(path):
        return entries

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n\r") for line in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Expect index line
        if not line.isdigit():
            i += 1
            continue

        idx = int(line)
        idx_key = f"{idx:04d}"

        # Time line
        i += 1
        if i >= len(lines):
            break

        time_line = lines[i].strip()
        if "-->" in time_line:
            start_time, end_time = [t.strip() for t in time_line.split("-->")]
        else:
            start_time, end_time = "", ""

        i += 1

        # Collect text lines until blank
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        text = " ".join(text_lines).strip()
        entries[idx_key] = (text, start_time, end_time)

    return entries


def build_rows(movie_speaker_map: dict, emotions_map: dict, srt_dir: str) -> List[Tuple[str, str, str, str, str, int, str, str]]:
    rows = []
    srt_cache: Dict[str, Dict[str, Tuple[str, str, str]]] = {}

    for movie, speakers in movie_speaker_map.items():
        srt_path = os.path.join(srt_dir, f"{movie}.srt")
        if movie not in srt_cache:
            srt_cache[movie] = parse_srt_file(srt_path)
        srt_map = srt_cache[movie]

        for speaker, indices in speakers.items():
            for idx_str in indices:
                idx_key = str(idx_str).zfill(4)
                text, start_time, end_time = srt_map.get(idx_key, ("", "", ""))

                emotion_key = f"{movie}@{speaker}_00_{idx_key}_00"
                emotion_id = emotions_map.get(emotion_key, 7)
                emotion_label = EMOTION_LABELS.get(emotion_id, "others")

                rows.append((movie, speaker, text, idx_key, emotion_label, emotion_id, start_time, end_time))

    return rows


def write_csv(rows: List[Tuple[str, str, str, str, str, int, str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["movie", "speaker", "utterance", "srt_index", "emotion", "emotion_id", "start_time", "end_time"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build CSV from V2C speaker ids, SRT, and emotions.")
    parser.add_argument(
        "--movie-speaker-json",
        default="/home/ruixin/dataset/V2C/V2C/DataConstruction/movie_speaker_id.json",
        help="Path to movie_speaker_id.json",
    )
    parser.add_argument(
        "--emotions-json",
        default="/home/ruixin/dataset/V2C/V2C/DataConstruction/emotions.json",
        help="Path to emotions.json",
    )
    parser.add_argument(
        "--srt-dir",
        default="/home/ruixin/dataset/V2C/V2C/DataConstruction/SRT",
        help="Directory containing SRT files",
    )
    parser.add_argument(
        "--output",
        default="/home/ruixin/dataset/V2C/V2C/DataConstruction/movie_speaker_emotion.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    movie_speaker_map = load_json(args.movie_speaker_json)
    emotions_map = load_json(args.emotions_json)

    rows = build_rows(movie_speaker_map, emotions_map, args.srt_dir)
    write_csv(rows, args.output)

    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
