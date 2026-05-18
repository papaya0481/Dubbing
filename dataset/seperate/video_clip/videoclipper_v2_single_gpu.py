#!/usr/bin/env python3
"""
VideoClipper V2 - 单 GPU 顺序处理版本
用于排查多进程 cuDNN 问题的临时方案
"""

import os
import sys

# 强制使用单个 GPU
if len(sys.argv) > 1 and '--gpu' in sys.argv:
    gpu_idx = sys.argv.index('--gpu')
    if gpu_idx + 1 < len(sys.argv):
        gpu_id = sys.argv[gpu_idx + 1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"Using GPU {gpu_id}")
        # 移除这两个参数
        sys.argv.pop(gpu_idx + 1)
        sys.argv.pop(gpu_idx)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("Using GPU 0 (default)")

import torch
import librosa
import argparse
import traceback
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
from tqdm import tqdm

# 导入原有的工具函数
sys.path.insert(0, os.path.dirname(__file__))
from utils.subtitle_utils import generate_srt, generate_srt_clip, process_asr_to_sentence_info
from utils.argparse_tools import ArgumentParser
from utils.trans_utils import write_state, load_state, convert_pcm_to_float
from funasr import AutoModel


@dataclass
class VideoTask:
    video_file: str
    stage: int
    sd_switch: str
    base_output_dir: str
    lang: str


# 导入 VideoClipper 类（从 videoclipper_v2.py）
class VideoClipper:
    """视频剪辑器"""
    def __init__(self, model, lang: str = 'zh'):
        self.model = model
        self.lang = lang

    def recog(self, audio_input, audio_file, sd_switch='yes', state=None):
        """语音识别"""
        if state is None:
            state = {}
        sr, data = audio_input
        data = convert_pcm_to_float(data)

        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        if len(data.shape) == 2:
            data = data.mean(axis=1)
        state['audio_input'] = (sr, data)

        if self.lang == 'en':
            rec_result = self.model.transcribe(
                audio=audio_file,
                language="English",
                return_time_stamps=True,
            )
            sentence_info, recog_res_raw = process_asr_to_sentence_info(rec_result[0])
            res_srt = generate_srt(sentence_info)
            state['sentences'] = sentence_info
            state['recog_res_raw'] = recog_res_raw
        else:
            if sd_switch == 'yes':
                rec_result = self.model.generate(
                    data,
                    return_spk_res=True,
                    sentence_timestamp=True,
                    return_raw_text=True,
                    is_final=True,
                    pred_timestamp=False,
                    en_post_proc=False,
                    cache={},
                    merge_vad=True,
                    merge_length_s=30
                )
            else:
                rec_result = self.model.generate(
                    data,
                    return_spk_res=False,
                    sentence_timestamp=True,
                    return_raw_text=True,
                    is_final=True,
                    pred_timestamp=False,
                    en_post_proc=False,
                    cache={},
                    merge_vad=True,
                    merge_length_s=30
                )
            res_srt = generate_srt(rec_result[0]['sentence_info'])
            state['recog_res_raw'] = rec_result[0]['raw_text']
            state['timestamp'] = rec_result[0]['timestamp']
            state['sentences'] = rec_result[0]['sentence_info']

        del data
        return res_srt, state

    def video_recog(self, video_filename, sd_switch='yes', output_dir=None):
        """视频语音识别"""
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename)
            base_name, _ = os.path.splitext(base_name)
            audio_file = os.path.join(output_dir, base_name + '.wav')
        else:
            base_name, _ = os.path.splitext(video_filename)
            audio_file = base_name + '.wav'

        with VideoFileClip(video_filename) as video:
            if video.audio is None:
                raise ValueError(f"No audio in video: {video_filename}")
            video.audio.write_audiofile(audio_file, codec='pcm_s16le', logger=None)
            video.close()
            del video

        wav, sr = librosa.load(audio_file, sr=16000)
        results = self.recog((sr, wav), audio_file, sd_switch, {'video_filename': video_filename})
        if os.path.exists(audio_file):
            os.remove(audio_file)
        return results

    def video_clip(self, state, output_dir=None):
        """视频剪辑"""
        # ... (与 videoclipper_v2.py 相同的实现)
        pass  # 这里省略，因为主要测试 stage 1


def init_model(lang: str, device: str):
    """初始化模型"""
    print(f"Initializing model for {lang} on {device}...")

    # 配置 cuDNN
    if device == 'cuda':
        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

    if lang == 'en':
        from qwen_asr import Qwen3ASRModel
        asr_path = "Qwen/Qwen3-ASR-1.7B"
        aligner_path = "Qwen/Qwen3-ForcedAligner-0.6B"
        model = Qwen3ASRModel.from_pretrained(
            asr_path,
            device_map='cuda:0' if device == 'cuda' else 'cpu',
            dtype=torch.bfloat16,
            max_inference_batch_size=1,
            max_new_tokens=16000,
            forced_aligner=aligner_path,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map='cuda:0' if device == 'cuda' else 'cpu',
            ),
        )
    else:
        model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            device='cuda:0' if device == 'cuda' else 'cpu'
        )

    print("Model loaded successfully")
    return model


def process_video(task: VideoTask, model) -> Tuple[bool, str, Optional[str]]:
    """处理单个视频"""
    try:
        video_file = task.video_file
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        parent_dir = os.path.dirname(video_file)
        parent_dir_name = os.path.basename(parent_dir)
        output_dir = os.path.join(task.base_output_dir, parent_dir_name, video_name)
        os.makedirs(output_dir, exist_ok=True)

        clipper = VideoClipper(model, task.lang)

        if task.stage == 1:
            res_srt, state = clipper.video_recog(video_file, task.sd_switch, output_dir)
            total_srt_file = os.path.join(output_dir, 'total.srt')
            with open(total_srt_file, 'w') as fout:
                fout.write(res_srt)
            write_state(output_dir, state)

        return True, video_file, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return False, task.video_file, error_msg


def main():
    parser = ArgumentParser(description="VideoClipper V2 - Single GPU")
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--sd_switch", type=str, default="yes", choices=["no", "yes"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print("=" * 60)
    print("VideoClipper V2 - Single GPU Mode")
    print(f"Stage: {args.stage}, Language: {args.lang}")
    print(f"Device: {args.device}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)

    # 初始化模型
    model = init_model(args.lang, args.device)

    # 查找视频文件
    if os.path.isfile(args.file):
        videos = [args.file]
    else:
        videos = []
        for root, _, files in os.walk(args.file):
            for file in files:
                if file.lower().endswith('.mp4'):
                    videos.append(os.path.join(root, file))

    print(f"\nFound {len(videos)} videos\n")

    # 顺序处理
    results = []
    for video in tqdm(videos, desc="Processing"):
        task = VideoTask(video, args.stage, args.sd_switch, args.output_dir, args.lang)
        success, video_file, error = process_video(task, model)
        results.append((success, video_file, error))

    # 统计结果
    success_count = sum(1 for s, _, _ in results if s)
    print(f"\n{'='*60}")
    print(f"Complete: {success_count}/{len(results)} succeeded")
    print(f"{'='*60}")

    # 显示失败
    for success, video, error in results:
        if not success:
            print(f"\n❌ {video}")
            print(f"   {error[:200]}")


if __name__ == '__main__':
    main()
