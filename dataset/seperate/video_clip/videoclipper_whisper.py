#!/usr/bin/env python3
"""
VideoClipper with Whisper Large V3
使用 Whisper 替代 Qwen3-ASR，避免 cuDNN 问题
"""

import os
import sys
import time
import torch
import librosa
import argparse
import traceback
import multiprocessing as mp
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from queue import Empty
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
from tqdm import tqdm
import whisper
from whisper.utils import get_writer

# 导入原有的工具函数
sys.path.insert(0, os.path.dirname(__file__))
from utils.argparse_tools import ArgumentParser
from utils.trans_utils import write_state, load_state


@dataclass
class VideoTask:
    """视频处理任务"""
    video_file: str
    stage: int
    sd_switch: str
    base_output_dir: str
    lang: str
    gpu_id: Optional[int] = None


def whisper_result_to_srt(result: dict) -> str:
    """将 Whisper 结果转换为 SRT 格式"""
    srt_lines = []
    for i, segment in enumerate(result['segments'], 1):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()

        # 转换为 SRT 时间格式
        start_time = format_timestamp(start)
        end_time = format_timestamp(end)

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")

    return "\n".join(srt_lines)


def format_timestamp(seconds: float) -> str:
    """格式化时间戳为 SRT 格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def whisper_result_to_sentence_info(result: dict) -> List[dict]:
    """将 Whisper 结果转换为 sentence_info 格式（兼容原有的 video_clip 函数）"""
    sentence_info = []
    for segment in result['segments']:
        sentence_info.append({
            'start': int(segment['start'] * 1000),  # 转换为毫秒
            'end': int(segment['end'] * 1000),
            'text': segment['text'].strip(),
            'spk': 'unknown'  # Whisper 不支持说话人分离
        })
    return sentence_info


class WhisperVideoClipper:
    """使用 Whisper 的视频剪辑器"""

    def __init__(self, model, lang: str = 'en'):
        self.model = model
        self.lang = lang

    def transcribe_audio(self, audio_file: str, language: str = None) -> Tuple[str, dict]:
        """使用 Whisper 转录音频"""
        # Whisper 语言代码映射
        lang_map = {
            'zh': 'zh',
            'en': 'en',
            'ja': 'ja',
            'ko': 'ko',
        }
        whisper_lang = lang_map.get(language or self.lang, 'en')

        # 转录
        result = self.model.transcribe(
            audio_file,
            language=whisper_lang,
            task='transcribe',
            verbose=False,
            word_timestamps=True  # 启用词级时间戳以获得更精确的对齐
        )

        # 转换为 SRT
        srt_content = whisper_result_to_srt(result)

        # 构建状态
        state = {
            'sentences': whisper_result_to_sentence_info(result),
            'recog_res_raw': result['text'],
            'language': result['language']
        }

        return srt_content, state

    def video_recog(self, video_filename: str, output_dir: str = None) -> Tuple[str, dict]:
        """从视频提取音频并转录"""
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename)
            base_name, _ = os.path.splitext(base_name)
            audio_file = os.path.join(output_dir, base_name + '.wav')
        else:
            base_name, _ = os.path.splitext(video_filename)
            audio_file = base_name + '.wav'

        # 提取音频
        with VideoFileClip(video_filename) as video:
            if video.audio is None:
                raise ValueError(f"No audio in video: {video_filename}")
            video.audio.write_audiofile(audio_file, codec='pcm_s16le', logger=None)
            video.close()
            del video

        # 转录
        srt_content, state = self.transcribe_audio(audio_file)
        state['video_filename'] = video_filename

        # 清理临时音频文件
        if os.path.exists(audio_file):
            os.remove(audio_file)

        return srt_content, state

    def video_clip(self, state: dict, output_dir: str = None):
        """根据时间戳剪辑视频"""
        sentences = state['sentences']
        video_file = state['video_file']
        vocal_file = state.get('vocal_file')
        raw_audio_file = state.get('raw_audio_file')
        instrumental_file = state.get('instrumental_file')

        ts = []
        for sentence in sentences:
            start_time = sentence['start'] / 1000.0  # 转换为秒
            end_time = sentence['end'] / 1000.0
            speaker_id = sentence.get('spk', 'unknown')
            ts.append([start_time, end_time, speaker_id])

        if not ts:
            return "No valid periods found"

        # 创建输出目录
        clipped_folder = os.path.join(output_dir, 'clipped')
        vocal_folder = os.path.join(output_dir, 'vocals')
        instrumental_folder = os.path.join(output_dir, 'instrumental')
        os.makedirs(clipped_folder, exist_ok=True)
        os.makedirs(vocal_folder, exist_ok=True)
        os.makedirs(instrumental_folder, exist_ok=True)

        # 剪辑每个片段
        for i, (start, end, speaker_id) in enumerate(ts):
            base_name = os.path.basename(video_file)
            video_name_without_ext, _ = os.path.splitext(base_name)

            # 生成文件名
            start_hours = int(start // 3600)
            start_minutes = int((start % 3600) // 60)
            start_seconds = int(start % 60)
            start_milliseconds = int((start - int(start)) * 100)

            if speaker_id != 'unknown':
                clip_filename = f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_{start_milliseconds:02}_spk{speaker_id}"
            else:
                clip_filename = f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_{start_milliseconds:02}"

            # 文件路径
            clip_filepath = os.path.join(clipped_folder, clip_filename)
            video_clip = clip_filepath + '.mp4'
            audio_clip = clip_filepath + '.wav'
            clip_srt_file = clip_filepath + '.srt'

            # 跳过已存在的文件
            if os.path.exists(video_clip) and os.path.exists(audio_clip):
                continue

            # 剪辑视频
            try:
                with VideoFileClip(video_file) as video:
                    end = min(end, video.duration)
                    sub = video.subclipped(start, end)
                    sub.write_videofile(video_clip, audio_codec="aac", logger=None)
                    if sub.audio:
                        sub.audio.write_audiofile(audio_clip, codec='pcm_s16le', logger=None)
                    sub.close()
                    del sub

                # 如果有 vocal 文件，也剪辑
                if vocal_file and os.path.exists(vocal_file):
                    vocal_clip = os.path.join(vocal_folder, clip_filename) + '.wav'
                    with AudioFileClip(vocal_file) as vocal:
                        end = min(end, vocal.duration)
                        sub_vo = vocal.subclipped(start, end)
                        sub_vo.write_audiofile(vocal_clip, codec='pcm_s16le', logger=None)
                        sub_vo.close()
                        del sub_vo

                # 如果有 instrumental 文件，也剪辑
                if instrumental_file and os.path.exists(instrumental_file):
                    instrumental_clip = os.path.join(instrumental_folder, clip_filename) + '.wav'
                    with AudioFileClip(instrumental_file) as instrumental:
                        end = min(end, instrumental.duration)
                        sub_in = instrumental.subclipped(start, end)
                        sub_in.write_audiofile(instrumental_clip, codec='pcm_s16le', logger=None)
                        sub_in.close()
                        del sub_in

                # 生成该片段的 SRT
                segment_srt = f"1\n{format_timestamp(0)} --> {format_timestamp(end - start)}\n{sentences[i]['text']}\n"
                with open(clip_srt_file, 'w', encoding='utf-8') as f:
                    f.write(segment_srt)

            except Exception as e:
                print(f"[WARNING] Failed to clip segment {i}: {e}")
                continue

        return f"{len(ts)} clips created"


def init_whisper_model(model_name: str = "large-v3", device: str = "cuda"):
    """初始化 Whisper 模型"""
    print(f"Loading Whisper model: {model_name} on {device}")

    # if "cuda" in device:
    #     # 配置 PyTorch
    #     torch.backends.cudnn.enabled = False
    #     torch.backends.cudnn.benchmark = False  # Whisper 可以使用 benchmark

    model = whisper.load_model(model_name, device=device, download_root="/data2/ruixin/.cache/whisper")
    print(f"Whisper model loaded successfully")
    return model


# 全局变量，用于存储工作进程的模型
_worker_model = None


def _init_worker_process(gpu_id: int, model_name: str, device: str):
    """初始化工作进程"""
    global _worker_model
    try:
        if device == 'cuda':
            # 设置环境变量

            # 验证 CUDA
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA not available in worker {os.getpid()}")

            torch.cuda.init()
            torch.cuda.set_device(gpu_id)
            actural_gup_id = f'cuda:{torch.cuda.current_device()}'

            # 预热
            dummy = torch.zeros(1, device=actural_gup_id)
            del dummy
            torch.cuda.synchronize()

            print(f"[Worker-{os.getpid()}] CUDA initialized on GPU {gpu_id}")

        # 加载 Whisper 模型
        _worker_model = init_whisper_model(model_name, actural_gup_id if device == 'cuda' else 'cpu')
        print(f"[Worker-{os.getpid()}] Whisper model loaded")

    except Exception as e:
        print(f"[Worker-{os.getpid()}] Failed to initialize: {e}")
        traceback.print_exc()
        raise


def process_single_video(task: VideoTask, device: str) -> Tuple[bool, str, Optional[str], Optional[Dict]]:
    """处理单个视频文件"""
    global _worker_model

    timing = {}
    start_time = time.time()

    try:
        video_file = task.video_file
        stage = task.stage
        base_output_dir = task.base_output_dir
        lang = task.lang

        # 准备文件路径
        t0 = time.time()
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        parent_dir = os.path.dirname(video_file)
        vocal_file = os.path.join(parent_dir, "vocals", f"{video_name}.wav")
        instrumental_file = os.path.join(parent_dir, "instrumental", f"{video_name}.wav")
        raw_audio_file = os.path.join(parent_dir, f"{video_name}.wav")

        if not os.path.exists(vocal_file):
            vocal_file = None
        if not os.path.exists(instrumental_file):
            instrumental_file = None
        if not os.path.exists(raw_audio_file):
            raw_audio_file = None

        parent_dir_name = os.path.basename(parent_dir)
        output_dir = os.path.join(base_output_dir, parent_dir_name, video_name)
        os.makedirs(output_dir, exist_ok=True)
        timing['setup'] = time.time() - t0

        # 创建 clipper
        clipper = WhisperVideoClipper(_worker_model, lang)

        # 执行处理
        if stage == 1:
            # 使用 vocal 文件或视频音频
            if vocal_file:
                t1 = time.time()
                srt_content, state = clipper.transcribe_audio(vocal_file, lang)
                timing['transcribe'] = time.time() - t1
                state['audio_filename'] = vocal_file
            else:
                t1 = time.time()
                srt_content, state = clipper.video_recog(video_file, output_dir)
                timing['video_recog'] = time.time() - t1

            # 保存结果
            t2 = time.time()
            total_srt_file = os.path.join(output_dir, 'total.srt')
            with open(total_srt_file, 'w', encoding='utf-8') as fout:
                fout.write(srt_content)
            write_state(output_dir, state)
            timing['save'] = time.time() - t2

        elif stage == 2:
            t1 = time.time()
            state = load_state(output_dir)
            state['video_file'] = video_file
            state['raw_audio_file'] = raw_audio_file
            state['vocal_file'] = vocal_file
            state['instrumental_file'] = instrumental_file
            timing['load_state'] = time.time() - t1

            t2 = time.time()
            clipper.video_clip(state, output_dir=output_dir)
            timing['clip'] = time.time() - t2

        timing['total'] = time.time() - start_time
        return True, video_file, None, timing

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        timing['total'] = time.time() - start_time
        return False, task.video_file, error_msg, timing


def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                   model_name: str, device: str, stage: int, lang: str):
    """工作进程函数"""
    try:
        # 初始化模型（仅 stage 1 需要）
        if stage == 1:
            _init_worker_process(gpu_id, model_name, device)

        # 处理任务
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break

                success, video_file, error, timing = process_single_video(task, device)
                result_queue.put((success, video_file, error, timing))

            except Empty:
                continue
            except Exception as e:
                error_msg = f"Worker error: {e}\n{traceback.format_exc()}"
                result_queue.put((False, "unknown", error_msg, None))

    except Exception as e:
        print(f"[Worker-{os.getpid()}] Fatal error: {e}")
        traceback.print_exc()
    finally:
        result_queue.put(None)  # 工作进程结束信号


def count_srt_entries(srt_path: str) -> int:
    """统计 SRT 文件条目数"""
    for enc in ['utf-8', 'gbk', 'gb2312']:
        try:
            with open(srt_path, 'r', encoding=enc) as f:
                return sum(1 for line in f if line.strip())
        except UnicodeDecodeError:
            continue
    return 0


def find_all_videos(folder: str, base_output_dir: str, stage: int, skip_processed: bool) -> List[str]:
    """查找所有待处理的视频文件"""
    all_videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.mp4'):
                file_path = os.path.join(root, file)
                if skip_processed and base_output_dir:
                    parent_dir_name = os.path.basename(root)
                    video_name = os.path.splitext(file)[0]
                    output_subdir = os.path.join(base_output_dir, parent_dir_name, video_name)
                    total_srt = os.path.join(output_subdir, 'total.srt')
                    clipped_dir = os.path.join(output_subdir, 'clipped')

                    if stage == 1 and os.path.exists(total_srt):
                        if count_srt_entries(total_srt) != 0:
                            continue
                    elif stage == 2:
                        if os.path.exists(total_srt) and os.path.isdir(clipped_dir):
                            expected_count = count_srt_entries(total_srt)
                            actual_count = len(os.listdir(clipped_dir))
                            if actual_count > 0 and actual_count >= expected_count - 30:
                                continue
                all_videos.append(file_path)
    return all_videos


def get_parser():
    parser = ArgumentParser(description="VideoClipper with Whisper",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True,
                       help="Stage: 1=Transcribe, 2=Clip")
    parser.add_argument("--file", type=str, required=True,
                       help="Input video file or folder")
    parser.add_argument("--sd_switch", type=str, default="no", choices=["no", "yes"],
                       help="Speaker diarization (not supported by Whisper)")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--skip_processed", action="store_true",
                       help="Skip already processed videos")
    parser.add_argument("--lang", type=str, default="en",
                       help="Language: en, zh, ja, ko, etc.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"],
                       help="Device to use")
    parser.add_argument("--model", type=str, default="large-v3",
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size")
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)

    file_or_folder = args.file
    stage = args.stage
    sd_switch = args.sd_switch
    output_dir = args.output_dir
    skip_processed = args.skip_processed
    lang = args.lang
    device = args.device
    model_name = args.model

    if device == 'gpu':
        device = 'cuda'

    # 确定使用的设备数量
    use_cuda = device == 'cuda' and torch.cuda.is_available()
    num_workers = torch.cuda.device_count() if use_cuda else 1

    print(f"{'='*60}")
    print(f"VideoClipper with Whisper {model_name}")
    print(f"Stage: {stage}, Language: {lang}, Device: {device}")
    if use_cuda:
        print(f"Using {num_workers} GPU(s)")
    else:
        print(f"Using CPU")
    print(f"{'='*60}\n")

    # 单文件处理
    if not os.path.isdir(file_or_folder):
        if stage == 1:
            _init_worker_process(0, model_name, device)
        task = VideoTask(file_or_folder, stage, sd_switch, output_dir, lang, 0)
        success, video, error, timing = process_single_video(task, device)

        if success:
            print(f"\n✅ Success: {video}")
            if timing:
                print(f"   Timing: {timing}")
        else:
            print(f"\n❌ Failed: {video}")
            print(f"Error: {error}")
        return

    # 查找所有视频
    all_videos = find_all_videos(file_or_folder, output_dir, stage, skip_processed)
    print(f"Found {len(all_videos)} videos to process\n")

    if not all_videos:
        print("No videos to process")
        return

    # 创建任务列表
    tasks = [VideoTask(video, stage, sd_switch, output_dir, lang) for video in all_videos]

    # 创建任务队列和结果队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 将任务放入队列
    for task in tasks:
        task_queue.put(task)

    # 添加结束信号
    for _ in range(num_workers):
        task_queue.put(None)

    # 启动工作进程
    processes = []
    for gpu_id in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, task_queue, result_queue, model_name, device, stage, lang)
        )
        p.start()
        processes.append(p)
        print(f"Started worker process {p.pid} on GPU {gpu_id}")

    print()

    # 收集结果并显示进度
    results = []
    timings = []
    finished_workers = 0
    with tqdm(total=len(tasks), desc="Processing videos", unit="video") as pbar:
        while finished_workers < num_workers:
            try:
                result = result_queue.get(timeout=1)
                if result is None:
                    finished_workers += 1
                else:
                    success, video_file, error, timing = result
                    results.append((success, video_file, error))
                    if timing:
                        timings.append(timing)

                    # 更新进度条
                    if timings:
                        avg_time = sum(t.get('total', 0) for t in timings) / len(timings)
                        pbar.set_postfix({'avg_time': f'{avg_time:.1f}s'})

                    pbar.update(1)
            except Empty:
                continue
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user. Terminating workers...")
                for p in processes:
                    p.terminate()
                for p in processes:
                    p.join(timeout=5)
                sys.exit(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 统计结果
    success_count = sum(1 for success, _, _ in results if success)
    failed_count = len(results) - success_count

    print(f"\n{'='*60}")
    print(f"Processing Complete")
    print(f"Total: {len(results)}, Success: {success_count}, Failed: {failed_count}")

    # 显示性能统计
    if timings:
        avg_total = sum(t.get('total', 0) for t in timings) / len(timings)
        print(f"\nPerformance Statistics:")
        print(f"  Average total time: {avg_total:.2f}s")

        # 显示各阶段平均时间
        if stage == 1:
            timing_keys = ['setup', 'transcribe', 'video_recog', 'save']
            for key in timing_keys:
                values = [t.get(key, 0) for t in timings if key in t]
                if values:
                    avg_val = sum(values) / len(values)
                    pct = (avg_val / avg_total * 100) if avg_total > 0 else 0
                    print(f"  Average {key}: {avg_val:.2f}s ({pct:.1f}%)")

    print(f"{'='*60}\n")

    # 显示失败的任务
    if failed_count > 0:
        print("Failed videos:")
        for success, video, error in results:
            if not success:
                print(f"\n❌ {video}")
                if error:
                    print(f"Error: {error}")


if __name__ == '__main__':
    # 清理主进程的 CUDA 环境变量
    torch.cuda.empty_cache()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']

    mp.set_start_method('spawn', force=True)
    main()
