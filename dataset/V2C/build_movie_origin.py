import argparse
import os
import re
import subprocess
import warnings
import logging
from typing import List, Dict, Any, Tuple, Optional

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

# --- SRT解析 ---
def parse_srt_time(time_str: str) -> float:
    """解析SRT时间格式 HH:MM:SS,mmm 为秒数"""
    time_str = time_str.strip().replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return 0.0

# --- 辅助函数 ---
def normalize_timecode(t: str) -> str:
    s = str(t).strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 2:
        s = f"00:{s}"
    elif len(parts) == 1:
        s = f"00:00:{s}"
    return s

def seconds_to_timecode(seconds: float) -> str:
    """将秒数转换为时间码格式"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"

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

class MovieDatasetBuilder:
    def __init__(self, movie_map_csv: str, utterances_csv: str, 
                 video_root: str, output_root: str,
                 test_mode: bool = False, use_offset: bool = True,
                 target_movie: str = None):
        self.logger = setup_logger()
        self.movie_map_csv = movie_map_csv
        self.utterances_csv = utterances_csv
        self.video_root = video_root
        self.output_root = output_root
        self.test_mode = test_mode
        self.use_offset = use_offset
        self.target_movie = target_movie
        
        if target_movie:
            self.logger.info(f"只处理指定电影: {target_movie}")
        
        # 加载movie_map
        self.movie_map_df = pd.read_csv(movie_map_csv)
        self.logger.info(f"已加载 {len(self.movie_map_df)} 部电影")
    
    def build_dataset(self):
        """基于movie_speaker_emotion.csv和时间偏移构建数据集"""
        self.logger.info("\n开始构建数据集...")
        
        # 读取utterances CSV
        self.logger.info(f"读取 {self.utterances_csv}...")
        utterances_df = pd.read_csv(self.utterances_csv)
        self.logger.info(f"共 {len(utterances_df)} 条字幕记录")
        
        videos_root = os.path.join(self.output_root, "videos")
        audios_root = os.path.join(self.output_root, "audios", "ost")
        meta_root = os.path.join(self.output_root, "meta_files")
        os.makedirs(videos_root, exist_ok=True)
        os.makedirs(audios_root, exist_ok=True)
        os.makedirs(meta_root, exist_ok=True)
        
        all_meta_rows = []
        
        # 验证电影并获取偏移
        
        # 预先筛选有效的电影
        valid_movies_data = {} # movie -> (offset, video_path)
        
        # 检查CSV列
        has_checked = 'checked' in self.movie_map_df.columns
        if not has_checked:
             self.logger.warning(f"CSV {self.movie_map_csv} 缺少 'checked' 列")
        
        skipped_count = 0
        
        for _, row in self.movie_map_df.iterrows():
            movie = row['movie']
            
            # Check target movie filter
            if self.target_movie and movie != self.target_movie:
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
                # If specifically filtered by target_movie, warn loudly
                if self.target_movie and movie == self.target_movie:
                     self.logger.error(f"Target movie {movie} is not checked/verified in CSV! Skipping.")
                elif not self.target_movie:
                     pass
                skipped_count += 1
                continue
                
            # Check time_offset
            offset = row.get('time_offset', 0.0)
            if pd.isna(offset):
                self.logger.warning(f"Skipping {movie}: time_offset is NaN")
                skipped_count += 1
                continue
                
            filename = row.get('filename', '')
            if pd.isna(filename) or not filename:
                 self.logger.warning(f"Skipping {movie}: No filename")
                 skipped_count += 1
                 continue
                 
            video_path = os.path.join(self.video_root, filename) if not os.path.isabs(filename) else filename
            if not os.path.exists(video_path):
                self.logger.warning(f"Skipping {movie}: Video file not found at {video_path}")
                skipped_count += 1
                continue

            # Valid
            valid_movies_data[movie] = (float(offset), video_path)
            
        self.logger.info(f"Valid movies to process: {len(valid_movies_data)}")
        if self.target_movie and len(valid_movies_data) == 0:
             self.logger.error("No valid movies found for target!")
             return

        # Filter movies in utterances
        movies_to_process = []
        for movie in utterances_df['movie'].unique():
            if movie in valid_movies_data:
                movies_to_process.append(movie)
        
        if self.test_mode and len(movies_to_process) > 4:
            self.logger.info(f"测试模式: 只处理前4部电影 (共{len(movies_to_process)}部)")
            movies_to_process = movies_to_process[:4]
            
        # Process loop
        for movie in tqdm(movies_to_process, desc="处理电影"):
            time_offset, video_path = valid_movies_data[movie]
            
            if not self.use_offset:
                time_offset = 0.0
            
            self.logger.info(f"\n处理 {movie}...")
            if time_offset != 0.0:
                self.logger.info(f"应用时间偏移: {time_offset:.2f}s")
            
            # 获取该电影的所有字幕
            movie_utterances = utterances_df[utterances_df['movie'] == movie]
            
            output_rows = []
            sample_id = 0
            
            for _, row in tqdm(movie_utterances.iterrows(), total=len(movie_utterances), 
                             desc=f"裁剪片段: {movie}", leave=False):
                sample_id += 1
                
                # 解析时间（格式："00:19:46,602"）
                start_time_str = str(row['start_time']).strip().strip('"')
                end_time_str = str(row['end_time']).strip().strip('"')
                
                start_sec = parse_srt_time(start_time_str)
                end_sec = parse_srt_time(end_time_str)
                
                # 应用时间偏移
                start_sec_adjusted = start_sec + time_offset
                end_sec_adjusted = end_sec + time_offset
                
                # 确保时间不为负
                start_sec_adjusted = max(0, start_sec_adjusted)
                end_sec_adjusted = max(0, end_sec_adjusted)
                
                start_time = seconds_to_timecode(start_sec_adjusted)
                end_time = seconds_to_timecode(end_sec_adjusted)
                
                # 文件名
                speaker = row['speaker']
                srt_idx = row['srt_index']
                clip_name = f"{movie}_{speaker}_{sample_id}_{srt_idx}.mp4"
                audio_name = f"{movie}_{speaker}_{sample_id}_{srt_idx}.wav"
                txt_name = f"{movie}_{speaker}_{sample_id}_{srt_idx}.txt"
                
                clip_path = os.path.join(videos_root, clip_name)
                audio_path = os.path.join(audios_root, audio_name)
                txt_path = os.path.join(audios_root, txt_name)
                
                # 裁剪视频和音频
                clip_ok = cut_clip(video_path, start_time, end_time, clip_path)
                audio_ok = extract_audio(video_path, start_time, end_time, audio_path)
                
                # 保存文本
                txt_ok = False
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(str(row['utterance']))
                    txt_ok = True
                except Exception as e:
                    self.logger.warning(f"保存文本失败: {e}")
                
                output_rows.append({
                    "Movie": movie,
                    "Speaker": speaker,
                    "Emotion": row['emotion'],
                    "Emotion_ID": row['emotion_id'],
                    "Srt_Index": srt_idx,
                    "Utterance": row['utterance'],
                    "Start_Time_Original": start_time_str,
                    "End_Time_Original": end_time_str,
                    "Start_Time_Adjusted": start_time,
                    "End_Time_Adjusted": end_time,
                    "Time_Offset": time_offset,
                    "Clip_Filename": clip_name if clip_ok else "",
                    "Clip_Path": f"videos/{clip_name}" if clip_ok else "",
                    "Audio_Filename": audio_name if audio_ok else "",
                    "Audio_Path": f"audios/ost/{audio_name}" if audio_ok else "",
                })
            
            if output_rows:
                movie_df = pd.DataFrame(output_rows)
                movie_df.to_csv(os.path.join(meta_root, f"{movie}.csv"), index=False)
                all_meta_rows.append(movie_df)
        
        if all_meta_rows:
            full_meta = pd.concat(all_meta_rows, ignore_index=True)
            full_meta.to_csv(os.path.join(self.output_root, "metadata.csv"), index=False)
            self.logger.info(f"\n完成! 共处理 {len(full_meta)} 个片段")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-map", required=True, help="movie_video_map.csv路径")
    parser.add_argument("--utterances", required=True, help="movie_speaker_emotion.csv路径")
    parser.add_argument("--video-root", required=True, help="用户视频根目录")
    parser.add_argument("--output-root", required=True, help="输出目录")
    parser.add_argument("--test", action="store_true", help="测试模式:只处理前4部电影")
    parser.add_argument("--no-offset", action="store_true", help="不使用任何时间偏移，直接使用原始时间")
    parser.add_argument("--movie", type=str, default=None, help="只处理指定的电影（电影名称）")
    args = parser.parse_args()
    
    builder = MovieDatasetBuilder(
        movie_map_csv=args.movie_map,
        utterances_csv=args.utterances,
        video_root=args.video_root,
        output_root=args.output_root,
        test_mode=args.test,
        use_offset=not args.no_offset,
        target_movie=args.movie,
    )
    
    builder.build_dataset()

if __name__ == "__main__":
    main()
