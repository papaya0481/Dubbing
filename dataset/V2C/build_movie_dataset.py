import argparse
import os
import re
import subprocess
import warnings
import logging
import ast
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- 日志设置 ---
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = self.COLORS.get(levelname, "")
        record.levelname = f"{color}{levelname}{self.RESET}"
        msg = super().format(record)
        record.levelname = levelname
        return msg

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("movie_dataset")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = TqdmLoggingHandler()
    handler.setFormatter(ColorFormatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

# --- 辅助函数 ---
def parse_time_vectorized(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    dt = pd.to_datetime(s, format="%H:%M:%S.%f", errors="coerce")
    seconds = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second + dt.dt.microsecond / 1e6
    return seconds.fillna(0.0)

def parse_srt_time(time_str: str) -> float:
    """解析SRT时间格式 HH:MM:SS,mmm 为秒数"""
    time_str = time_str.strip().replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return 0.0

def seconds_to_timecode(seconds: float) -> str:
    """将秒数转换为时间码格式"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"

def normalize_timecode(t: str) -> str:
    s = str(t).strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 2:
        s = f"00:{s}"
    elif len(parts) == 1:
        s = f"00:00:{s}"
    return s

def clean_speaker_prefix(utterance: str) -> str:
    if utterance is None: return ""
    s = str(utterance).strip()
    return re.sub(r"^\s*[^:]{1,20}:\s+", "", s)

def normalize_text(text: str) -> str:
    if text is None: return ""
    try:
        return str(text).encode("latin1", errors="strict").decode("cp1252")
    except Exception:
        return str(text)

def contains_others(emotions: Any) -> bool:
    if emotions is None: return False
    if isinstance(emotions, list):
        return any(str(e).strip().lower() == "others" for e in emotions)
    if isinstance(emotions, str):
        s = emotions.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return any(str(e).strip().lower() == "others" for e in parsed)
            except Exception:
                pass
        return re.search(r"\bothers\b", s, flags=re.IGNORECASE) is not None
    return False

def cut_clip(input_path: str, start_time: str, end_time: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", normalize_timecode(start_time),
        "-to", normalize_timecode(end_time),
        "-i", input_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-y",
        output_path,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0

def extract_audio(input_path: str, start_time: str, end_time: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", normalize_timecode(start_time),
        "-to", normalize_timecode(end_time),
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y",
        output_path,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0

def load_movie_map(map_csv: str, video_root: str, logger) -> Dict[str, Dict[str, Any]]:
    """
    返回: {movie: {"video_path": str, "time_offset": float, "checked": bool}}
    """
    df = pd.read_csv(map_csv)
    movie_map = {}
    
    has_checked = 'checked' in df.columns
    has_offset = 'time_offset' in df.columns
    
    if not has_checked:
        logger.warning(f"CSV {map_csv} 缺少 'checked' 列")
    if not has_offset:
        logger.warning(f"CSV {map_csv} 缺少 'time_offset' 列")
    
    for _, row in df.iterrows():
        movie = str(row.get("movie") or "").strip()
        filename = str(row.get("filename") or "").strip()
        
        if not movie or not filename: 
            continue
        
        # Check checked status
        is_checked = False
        if has_checked:
            val = row.get('checked', False)
            try:
                if str(val).lower() == 'true':
                    is_checked = True
            except:
                is_checked = False
        
        if not is_checked:
            logger.warning(f"Skipping {movie}: checked != True")
            continue
        
        # Get time offset
        offset = 0.0
        if has_offset:
            offset = row.get('time_offset', 0.0)
            if pd.isna(offset):
                logger.warning(f"Skipping {movie}: time_offset is NaN")
                continue
            offset = float(offset)
        
        path = filename if os.path.isabs(filename) else os.path.normpath(os.path.join(video_root, filename))
        
        if not os.path.exists(path):
            logger.warning(f"Skipping {movie}: Video file not found at {path}")
            continue
        
        movie_map[movie] = {
            "video_path": path,
            "time_offset": offset,
            "checked": is_checked
        }
    
    return movie_map

# --- 核心构建逻辑 ---

def build_sample(records: List[Dict[str, Any]], start_idx: int, length: int) -> Dict[str, Any]:
    rows = [records[start_idx + k] for k in range(length)]
    first, last = rows[0], rows[-1]
    
    return {
        "Movie": first["movie"],
        "Speaker": first["speaker"],
        "Emotions": [r["emotion"] for r in rows],
        "Utterances": [r["utterance_clean"] for r in rows],
        "Start_Time": first["start_time"],
        "End_Time": last["end_time"],
        "Length": length,
        "Srt_Indices": [r["srt_index"] for r in rows],
        # 用于后续去重和校验
        "_temp_srt_indices": tuple(r["srt_index"] for r in rows) 
    }

class MovieDatasetBuilder:
    def __init__(self, movie_map_csv: str, utterances_csv: str, video_root: str, output_root: str, max_gap: float = 10.0, max_words: int = 50, test_mode: bool = False, prefer_longer: bool = False, use_offset: bool = True):
        self.logger = setup_logger()
        self.movie_map = load_movie_map(movie_map_csv, video_root, self.logger)
        self.utterances_csv = utterances_csv
        self.output_root = output_root
        self.max_gap = max_gap
        self.max_words = max_words
        self.test_mode = test_mode
        self.prefer_longer = prefer_longer
        self.use_offset = use_offset
        self.logger.info("Movie map loaded: %d entries", len(self.movie_map))

    def _is_sequence_valid(self, records, start_idx, length):
        """
        严格检查序列连续性：
        1. 必须是同一电影、同一Speaker
        2. SRT Index 必须严格连续 (prev + 1 == curr)
        3. 时间间隔必须符合要求
        """
        prev = records[start_idx]
        for k in range(1, length):
            curr = records[start_idx + k]
            
            # Speaker / Movie Check
            if curr["movie"] != prev["movie"] or curr["speaker"] != prev["speaker"]:
                return False
            
            # SRT Continuity Check (关键：防止把 others 过滤后的断层拼起来)
            if curr["srt_index"] != prev["srt_index"] + 1:
                return False
                
            # Time Gap Check
            if (curr["start_sec"] - prev["end_sec"]) > self.max_gap:
                return False
            
            prev = curr
        return True

    def _has_emotion_change(self, records, start_idx, length):
        """
        底线：序列中必须包含至少一次情感变化。
        [Neu, Neu] -> False
        [Neu, Joy] -> True
        [Neu, Neu, Joy] -> True
        [Neu, Joy, Joy] -> True
        """
        emotions = set(records[start_idx + k]["emotion"] for k in range(length))
        return len(emotions) > 1

    def _check_word_count(self, records, start_idx, length):
        total = sum(len(str(records[start_idx + k]["utterance_clean"]).split()) for k in range(length))
        return total <= self.max_words

    def extract_samples(self) -> pd.DataFrame:
        self.logger.info("Reading utterances...")
        df = pd.read_csv(self.utterances_csv)
        
        # 1. 最开始就过滤 'others'
        self.logger.info(f"Original rows: {len(df)}")
        df = df[~df["emotion"].apply(contains_others)]
        self.logger.info(f"Rows after filtering 'others': {len(df)}")

        # 2. 清洗与时间解析
        df["utterance_clean"] = df["utterance"].apply(clean_speaker_prefix).apply(normalize_text)
        df["start_sec"] = parse_time_vectorized(df["start_time"])
        df["end_sec"] = parse_time_vectorized(df["end_time"])

        # 3. 排序 (这对后续的 srt_index 连续性检查至关重要)
        df = df.sort_values(by=["movie", "srt_index"]).reset_index(drop=True)
        records = df.to_dict("records")
        n = len(records)
        
        valid_samples = []

        self.logger.info(f"Scanning {n} rows with Sliding Window strategy (Maximize Quantity)...")
        
        # 核心滑动窗口循环
        for i in tqdm(range(n - 1), desc="Scanning"):
            
            # --- 策略：在位置 i，同时检查长度 2 和长度 3 ---
            found_length_3 = False
            
            # 1. 检查长度 3: [i, i+1, i+2] (优先检查，当 prefer_longer 启用时)
            if i < n - 2:
                if self._is_sequence_valid(records, i, 3):
                    if self._has_emotion_change(records, i, 3):
                        # 只有长度3才检查字数限制 (长度2通常不需要限制，或者你可以选择加上)
                        if self._check_word_count(records, i, 3):
                            valid_samples.append(build_sample(records, i, 3))
                            found_length_3 = True
                        # 注意：如果长度3字数超标，我们不在这里做 trim 变成长度2
                        # 因为滑动窗口到了 i+1 时，会自动捕获 [i+1, i+2]，那里就是天然的 trim 结果
                        # 这样避免了产生重复数据。
            
            # 2. 检查长度 2: [i, i+1] (如果 prefer_longer 启用且找到了 length 3，则跳过)
            if i < n - 1:
                if self._is_sequence_valid(records, i, 2):
                    if self._has_emotion_change(records, i, 2):
                        valid_samples.append(build_sample(records, i, 2))

        result_df = pd.DataFrame(valid_samples)
        
        if not result_df.empty:
            # 去重：滑动窗口可能会产生子集重复（虽然逻辑上 [A,B] 和 [A,B,C] 不算重复，
            # 但为了保证数据纯净，我们确保 Srt_Indices 是唯一的）
            # 对于 list/tuple 列去重
            before = len(result_df)
            result_df = result_df.loc[result_df.astype(str).drop_duplicates(subset=["_temp_srt_indices"]).index]
            result_df = result_df.drop(columns=["_temp_srt_indices"])
            
            self.logger.info(f"Extracted {len(result_df)} samples (from potential {before} candidates).")
        else:
            self.logger.warning("No valid samples found!")

        return result_df

    def ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def save_outputs(self, samples_df: pd.DataFrame) -> None:
        if samples_df.empty: return

        videos_root = os.path.join(self.output_root, "videos")
        audios_root = os.path.join(self.output_root, "audios", "ost")
        meta_root = os.path.join(self.output_root, "meta_files")
        self.ensure_dir(videos_root)
        self.ensure_dir(audios_root)
        self.ensure_dir(meta_root)

        all_meta_rows = []
        
        # Filter movies with valid video paths first
        movies_with_videos = []
        for movie, movie_df in samples_df.groupby("Movie"):
            movie_info = self.movie_map.get(movie)
            if movie_info and movie_info.get("video_path"):
                movies_with_videos.append((movie, movie_df))
            else:
                self.logger.warning(f"Video missing or not verified: {movie}")
        
        # Apply test mode limit
        if self.test_mode and len(movies_with_videos) > 4:
            self.logger.info(f"Test mode: processing first 4 movies (out of {len(movies_with_videos)} available)")
            movies_with_videos = movies_with_videos[:4]

        for movie, movie_df in tqdm(movies_with_videos, desc="Processing Movies"):
            movie_info = self.movie_map.get(movie)
            video_path = movie_info["video_path"]
            time_offset = movie_info["time_offset"] if self.use_offset else 0.0
            
            if time_offset != 0.0:
                self.logger.info(f"\n处理 {movie}... (应用时间偏移: {time_offset:.2f}s)")
            else:
                self.logger.info(f"\n处理 {movie}...")

            output_rows = []
            sample_id = 0
            
            for _, row in tqdm(movie_df.iterrows(), total=len(movie_df), desc=f"Clips: {movie}", leave=False):
                sample_id += 1
                # 文件名格式：{movie}_sample_{id}_{srt_range}.mp4
                srt_range = f"{row['Srt_Indices'][0]}-{row['Srt_Indices'][-1]}"
                clip_name = f"{movie}_sample_{sample_id}_{srt_range}.mp4"
                audio_name = f"{movie}_sample_{sample_id}_{srt_range}.wav"
                txt_name = f"{movie}_sample_{sample_id}_{srt_range}.txt"
                
                # 直接放在 videos 和 audios/ost 文件夹下，不再创建子文件夹
                clip_path = os.path.join(videos_root, clip_name)
                audio_path = os.path.join(audios_root, audio_name)
                txt_path = os.path.join(audios_root, txt_name)

                clip_ok = False
                audio_ok = False
                
                # 初始化时间相关变量
                start_time_str = str(row.get("Start_Time", ""))
                end_time_str = str(row.get("End_Time", ""))
                start_time_adjusted = start_time_str
                end_time_adjusted = end_time_str
                
                if pd.notna(row.get("Start_Time")) and pd.notna(row.get("End_Time")):
                    # 应用时间偏移
                    start_time_str = str(row["Start_Time"])
                    end_time_str = str(row["End_Time"])
                    
                    start_sec = parse_srt_time(start_time_str)
                    end_sec = parse_srt_time(end_time_str)
                    
                    # 加上偏移
                    start_sec_adjusted = max(0, start_sec + time_offset)
                    end_sec_adjusted = max(0, end_sec + time_offset)
                    
                    start_time_adjusted = seconds_to_timecode(start_sec_adjusted)
                    end_time_adjusted = seconds_to_timecode(end_sec_adjusted)
                    
                    clip_ok = cut_clip(video_path, start_time_adjusted, end_time_adjusted, clip_path)
                    audio_ok = extract_audio(video_path, start_time_adjusted, end_time_adjusted, audio_path)

                # 保存文本
                txt_ok = False
                try:
                    # Utterances 是一个列表，将其合并为一个字符串
                    utterances = row.get("Utterances", [])
                    if isinstance(utterances, list):
                        text_content = " ".join(str(u) for u in utterances)
                    else:
                        text_content = str(utterances)
                    
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    txt_ok = True
                except Exception as e:
                    self.logger.warning(f"保存文本失败: {e}")

                out_row = row.to_dict()
                out_row["Start_Time_Original"] = start_time_str
                out_row["End_Time_Original"] = end_time_str
                out_row["Start_Time_Adjusted"] = start_time_adjusted
                out_row["End_Time_Adjusted"] = end_time_adjusted
                out_row["Time_Offset"] = time_offset
                out_row["Clip_Filename"] = clip_name if clip_ok else ""
                out_row["Clip_Path"] = os.path.relpath(clip_path, self.output_root) if clip_ok else ""
                out_row["Audio_Filename"] = audio_name if audio_ok else ""
                out_row["Audio_Path"] = os.path.relpath(audio_path, self.output_root) if audio_ok else ""
                output_rows.append(out_row)

            if output_rows:
                mini_df = pd.DataFrame(output_rows)
                mini_df.to_csv(os.path.join(meta_root, f"{movie}.csv"), index=False)
                all_meta_rows.append(mini_df)

        if all_meta_rows:
            full_meta = pd.concat(all_meta_rows, ignore_index=True)
            full_meta.to_csv(os.path.join(self.output_root, "metadata.csv"), index=False)
            self.logger.info("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-map", required=True)
    parser.add_argument("--utterances", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-gap", type=float, default=10.0)
    parser.add_argument("--max-words", type=int, default=50)
    parser.add_argument("--test", action="store_true", help="Test mode: process only first 4 movies")
    parser.add_argument("--prefer-longer", action="store_true", help="If length 3 sample is valid, skip length 2 at the same position")
    parser.add_argument("--no-offset", action="store_true", help="不使用任何时间偏移，直接使用原始时间")
    args = parser.parse_args()

    builder = MovieDatasetBuilder(
        movie_map_csv=args.movie_map,
        utterances_csv=args.utterances,
        video_root=args.video_root,
        output_root=args.output_root,
        max_gap=args.max_gap,
        max_words=args.max_words,
        test_mode=args.test,
        prefer_longer=args.prefer_longer,
        use_offset=not args.no_offset,
    )
    samples = builder.extract_samples()
    builder.save_outputs(samples)

if __name__ == "__main__":
    main()
