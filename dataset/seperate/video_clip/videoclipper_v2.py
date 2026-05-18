import os
import sys
import time
import wave
import tempfile
import re
import csv
import shutil
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
from utils.subtitle_utils import generate_srt, generate_srt_clip, process_asr_to_sentence_info, normalize_sentence_info
from utils.argparse_tools import ArgumentParser
from utils.trans_utils import write_state, load_state, convert_pcm_to_float
from funasr import AutoModel


@dataclass
class VideoTask:
    """视频处理任务"""
    video_file: str
    stage: int
    sd_switch: str
    base_output_dir: str
    lang: str
    gpu_id: Optional[int] = None
    force_restart: bool = False
    text_normalize: bool = False


TERMINAL_PUNCTUATION = {"。", "！", "？", ".", "!", "?"}
COMMA_PUNCTUATION = {"，", ","}
MIN_TOKENS_FOR_COMMA_CUT = 3


def _text_to_string(text) -> str:
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        return " ".join(str(item) for item in text).strip()
    return str(text).strip()


def _normalize_text_for_wer(text: str) -> str:
    """Normalize text before WER calculation."""
    text = (text or "").lower().strip()
    if not text:
        return ""
    # Keep alnum, CJK and spaces; strip most punctuation for robust comparison.
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_for_wer(text: str) -> list[str]:
    """Tokenize by words if spaces exist, otherwise fallback to CJK char-level tokens."""
    normalized = _normalize_text_for_wer(text)
    if not normalized:
        return []
    if " " in normalized:
        return [tok for tok in normalized.split(" ") if tok]
    return [ch for ch in normalized if not ch.isspace()]


def _levenshtein_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    if not ref_tokens:
        return len(hyp_tokens)
    if not hyp_tokens:
        return len(ref_tokens)

    prev_row = list(range(len(hyp_tokens) + 1))
    for i, ref_tok in enumerate(ref_tokens, start=1):
        curr_row = [i]
        for j, hyp_tok in enumerate(hyp_tokens, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr_row.append(min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def compute_wer(reference_text: str, hypothesis_text: str) -> Optional[float]:
    """Compute WER (or char-level CER-like fallback for no-space text)."""
    ref_tokens = _tokenize_for_wer(reference_text)
    hyp_tokens = _tokenize_for_wer(hypothesis_text)

    if not ref_tokens:
        return None
    distance = _levenshtein_distance(ref_tokens, hyp_tokens)
    return distance / len(ref_tokens)


def _silence_hf_pad_token_warning(model) -> None:
    """Set pad_token_id explicitly for HF-style generation models to avoid warning spam."""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    def _fix_component(component) -> None:
        if component is None:
            return

        eos_id = None
        pad_id = None

        generation_config = getattr(component, "generation_config", None)
        config = getattr(component, "config", None)

        if generation_config is not None:
            eos_id = getattr(generation_config, "eos_token_id", None)
            pad_id = getattr(generation_config, "pad_token_id", None)

        if eos_id is None and config is not None:
            eos_id = getattr(config, "eos_token_id", None)
        if pad_id is None and config is not None:
            pad_id = getattr(config, "pad_token_id", None)

        if pad_id is None and eos_id is not None:
            if generation_config is not None:
                generation_config.pad_token_id = eos_id
            if config is not None:
                config.pad_token_id = eos_id

    # Cover common wrappers where the actual HF model may be nested.
    for attr_name in ["model", "asr_model", "forced_aligner", "aligner", "_model"]:
        _fix_component(getattr(model, attr_name, None))

    # Also try the object itself in case it directly exposes HF fields.
    _fix_component(model)


def write_wer_csv(output_root: str, rows: List[Dict]) -> str:
    """Write per-video WER summary to output_root/wer.csv."""
    csv_path = os.path.join(output_root, 'wer.csv')
    os.makedirs(output_root, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'Parent_Folder',
                'Video_Name',
                'Video_Output_Dir',
                'Average_WER',
                'Scored_Clips',
                'Total_Clips',
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _split_text_by_terminal(text: str) -> list[str]:
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s*", text) if part.strip()]
    return parts or [text.strip()]


def _is_terminal_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    return token[-1] in TERMINAL_PUNCTUATION or token in TERMINAL_PUNCTUATION


def _is_comma_token(token: str) -> bool:
    """Check if token ends with comma."""
    token = token.strip()
    if not token:
        return False
    return token[-1] in COMMA_PUNCTUATION or token in COMMA_PUNCTUATION


def _is_decimal_period(tokens: list[str], idx: int) -> bool:
    """Check if period/fullstop is decimal, including cross-token form like '4. 37'."""
    token = tokens[idx].strip()
    if not token or '.' not in token:
        return False

    # Same-token decimal, e.g. "3.14"
    if re.search(r'\d\.\d', token):
        return True

    prev_token = tokens[idx - 1].strip() if idx > 0 else ""
    next_token = tokens[idx + 1].strip() if idx + 1 < len(tokens) else ""

    # Cross-token decimal, e.g. "4." + "37" or "4" + ".37"
    left_is_digit = bool(re.search(r'\d\.$', token)) or token == "." and bool(re.search(r'\d$', prev_token))
    right_is_digit = bool(re.match(r'^\d', next_token)) or bool(re.match(r'^\.\d', token))
    return left_is_digit and right_is_digit


def _is_numeric_comma_separator(tokens: list[str], idx: int) -> bool:
    """Check if comma is used as a numeric separator, e.g. 1,000 or 1 , 000."""
    token = tokens[idx].strip()
    if not token or ',' not in token and '，' not in token:
        return False

    prev_token = tokens[idx - 1].strip() if idx > 0 else ""
    next_token = tokens[idx + 1].strip() if idx + 1 < len(tokens) else ""

    # Cross-token separator: "1" + "," + "000"
    if token in {",", "，"}:
        return bool(re.search(r'\d$', prev_token)) and bool(re.match(r'^\d', next_token))

    # Same-token left-bound separator: "1," + "000"
    if re.search(r'\d[，,]$', token) and re.match(r'^\d', next_token):
        return True

    # Same-token right-bound separator: "1" + ",000"
    if re.match(r'^[，,]\d', token) and re.search(r'\d$', prev_token):
        return True

    return False


def _build_aligned_tokens(sent: dict, timestamps: list) -> Optional[list[str]]:
    """Return tokens aligned 1:1 with timestamps, or None when alignment is unavailable."""
    text = sent.get("text")
    raw_text = sent.get("raw_text")

    if isinstance(text, list) and len(text) == len(timestamps):
        return [str(item) for item in text]

    if isinstance(text, str):
        text_tokens = text.split()
        if len(text_tokens) == len(timestamps):
            return text_tokens

    if isinstance(raw_text, str):
        raw_tokens = raw_text.split()
        if len(raw_tokens) == len(timestamps):
            return raw_tokens

    return None


def split_sentences_by_terminal_punctuation(sentences: list[dict]) -> list[dict]:
    """Split sentences by terminal punctuation using word-level timestamp alignment."""
    split_sentences: list[dict] = []
    for sent in sentences:
        timestamps = sent.get("timestamp") or []

        if not timestamps:
            continue

        aligned_tokens = _build_aligned_tokens(sent, timestamps)
        if not aligned_tokens:
            split_sentences.append(sent)
            continue

        chunk_start = 0
        created_chunks = 0
        for idx, token in enumerate(aligned_tokens):
            # Check for must-cut punctuation (terminal marks)
            should_cut = False
            if _is_terminal_token(token):
                # Don't split if period is a decimal point (digits on both sides)
                if not _is_decimal_period(aligned_tokens, idx):
                    should_cut = True
            elif _is_comma_token(token):
                # Check if token count from chunk_start is > MIN_TOKENS_FOR_COMMA_CUT
                tokens_since_chunk_start = idx - chunk_start + 1
                if tokens_since_chunk_start > MIN_TOKENS_FOR_COMMA_CUT and not _is_numeric_comma_separator(aligned_tokens, idx):
                    should_cut = True
            
            if not should_cut:
                continue

            chunk_tokens = aligned_tokens[chunk_start: idx + 1]
            chunk_ts = timestamps[chunk_start: idx + 1]
            if not chunk_tokens or not chunk_ts:
                chunk_start = idx + 1
                continue

            if isinstance(sent.get("text"), list):
                chunk_text = chunk_tokens
            else:
                chunk_text = " ".join(chunk_tokens).strip()

            chunk_sent = {
                "text": chunk_text,
                "start": chunk_ts[0][0],
                "end": chunk_ts[-1][1],
                "timestamp": chunk_ts,
                "raw_text": " ".join(chunk_tokens).strip(),
            }
            if "spk" in sent:
                chunk_sent["spk"] = sent["spk"]
            split_sentences.append(chunk_sent)
            created_chunks += 1
            chunk_start = idx + 1

        if chunk_start < len(aligned_tokens):
            tail_tokens = aligned_tokens[chunk_start:]
            tail_ts = timestamps[chunk_start:]
            if tail_tokens and tail_ts:
                if isinstance(sent.get("text"), list):
                    tail_text = tail_tokens
                else:
                    tail_text = " ".join(tail_tokens).strip()

                tail_sent = {
                    "text": tail_text,
                    "start": tail_ts[0][0],
                    "end": tail_ts[-1][1],
                    "timestamp": tail_ts,
                    "raw_text": " ".join(tail_tokens).strip(),
                }
                if "spk" in sent:
                    tail_sent["spk"] = sent["spk"]
                split_sentences.append(tail_sent)
                created_chunks += 1

        if created_chunks == 0:
            split_sentences.append(sent)

    return split_sentences


def init_worker_model(gpu_id: int, lang: str, device: str):
    """在工作进程中初始化模型（CUDA 环境应该已经在 _init_worker_process 中设置好）"""
    if device == 'cuda':
        actual_device = 'cuda:0'
        print(f"[Worker-{os.getpid()}] Loading model on cuda:0 (physical GPU {gpu_id})")

        # 确保 cuDNN 正确初始化
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = False  # 避免初始化问题
        # torch.backends.cudnn.deterministic = True

    else:
        actual_device = 'cpu'
        print(f"[Worker-{os.getpid()}] Loading model on CPU")

    if lang == 'zh':
        model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            device=actual_device
        )
    elif lang == 'en':
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "[ERROR] qwen_asr package not found. Install with:\n"
                "pip install -U qwen-asr[vllm]\n"
                "pip install -U flash-attn --no-build-isolation"
            )
        asr_path = "Qwen/Qwen3-ASR-1.7B"
        aligner_path = "Qwen/Qwen3-ForcedAligner-0.6B"

        # 在加载模型前，再次确认 CUDA 状态
        if device == 'cuda':
            print(f"[Worker-{os.getpid()}] Pre-model check: CUDA available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
            # 强制同步，确保 CUDA 完全初始化
            torch.cuda.synchronize()

        model = Qwen3ASRModel.from_pretrained(
            asr_path,
            device_map=actual_device,
            dtype=torch.bfloat16,
            max_inference_batch_size=1,
            max_new_tokens=16000,
            forced_aligner=aligner_path,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=actual_device,
            ),
        )
        _silence_hf_pad_token_warning(model)

        # 模型加载后，测试一次推理
        if device == 'cuda':
            print(f"[Worker-{os.getpid()}] Testing model inference...")
            try:
                # qwen_asr 的 transcribe 在这里要求音频文件路径，不能直接传 numpy.ndarray
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    test_wav_path = tmp_wav.name

                try:
                    with wave.open(test_wav_path, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)  # int16
                        wav_file.setframerate(16000)
                        wav_file.writeframes(b"\x00\x00" * 16000)  # 1 秒静音

                    _ = model.transcribe(
                        audio=test_wav_path,
                        language="English",
                        return_time_stamps=True,
                    )
                finally:
                    if os.path.exists(test_wav_path):
                        os.remove(test_wav_path)
                print(f"[Worker-{os.getpid()}] Model test inference successful")
            except Exception as e:
                print(f"[Worker-{os.getpid()}] Model test inference failed: {e}")
                raise

    else:
        raise ValueError(f"Unsupported language: {lang}")

    return model


# 全局变量，用于存储工作进程的模型
_worker_model = None


def _init_worker_process(gpu_id: int, lang: str, device: str):
    """初始化工作进程"""
    global _worker_model
    try:
        # 在 spawn 后立即设置 CUDA 环境（必须在导入任何 CUDA 相关模块之前）
        if device == 'cuda':
            # 设置环境变量
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # 禁用 TF32，可能有助于稳定性
            os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

            # 验证 CUDA 可用性
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA not available in worker process {os.getpid()}")

            # 初始化 CUDA
            torch.cuda.init()
            device_count = torch.cuda.device_count()
            if device_count != 1:
                raise RuntimeError(
                    f"Worker {os.getpid()} should see exactly 1 GPU, but sees {device_count}. "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
                )

            torch.cuda.set_device(0)

            # 配置 cuDNN
            # torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.allow_tf32 = False

            # 预热 CUDA 上下文，避免触发依赖 cuDNN 的算子初始化失败
            print(f"[Worker-{os.getpid()}] Warming up CUDA context...")
            dummy = torch.randn(256, 256, device='cuda:0')
            dummy2 = dummy @ dummy.T
            _ = dummy2.mean().item()
            del dummy, dummy2
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            print(f"[Worker-{os.getpid()}] CUDA initialized: Physical GPU {gpu_id} -> Logical cuda:0")
            print(f"[Worker-{os.getpid()}] GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"[Worker-{os.getpid()}] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"[Worker-{os.getpid()}] cuDNN version: {torch.backends.cudnn.version()}")
            print(f"[Worker-{os.getpid()}] cuDNN enabled: {torch.backends.cudnn.enabled}")

        # 加载模型
        _worker_model = init_worker_model(gpu_id, lang, device)
        print(f"[Worker-{os.getpid()}] Model loaded successfully")

    except Exception as e:
        print(f"[Worker-{os.getpid()}] Failed to initialize: {e}")
        traceback.print_exc()
        raise


class VideoClipper:
    """视频剪辑器"""
    def __init__(self, model, lang: str = 'zh', text_normalize: bool = False):
        self.model = model
        self.lang = lang
        self.text_normalize = text_normalize

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
            # Stage 1: normalize text while keeping timestamp aligned (only if enabled)
            if self.text_normalize:
                normalize_sentence_info(sentence_info)
            # Generate SRT for total.srt (no additional normalization needed in generate_srt)
            res_srt = generate_srt(sentence_info, normalize=False)
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
            # Stage 1: normalize text while keeping timestamp aligned (only if enabled)
            if self.text_normalize:
                normalize_sentence_info(rec_result[0]['sentence_info'])
            # Generate SRT for total.srt (no additional normalization needed in generate_srt)
            res_srt = generate_srt(rec_result[0]['sentence_info'], normalize=False)
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
        sentences = split_sentences_by_terminal_punctuation(state['sentences'])
        video_file = state['video_file']
        vocal_file = state['vocal_file']
        raw_audio_file = state['raw_audio_file']
        instrumental_file = state['instrumental_file']

        ts = []
        for sentence in sentences:
            start_time = sentence['start'] / 1000.0
            end_time = sentence['end'] / 1000.0
            speaker_id = sentence.get('spk', 'unknown')
            ts.append([start_time, end_time, speaker_id])

        srt_index = 1
        time_acc_ost = 0.0
        metadata_rows = []
        metadata_csv_file = os.path.join(output_dir, 'metadata.csv')

        if len(ts):
            for i, (start, end, speaker_id) in enumerate(ts):
                clipped_folder = os.path.join(output_dir, 'clipped')
                vocal_folder = os.path.join(output_dir, 'vocals')
                instrumental_folder = os.path.join(output_dir, 'instrumental')
                os.makedirs(clipped_folder, exist_ok=True)
                os.makedirs(vocal_folder, exist_ok=True)
                os.makedirs(instrumental_folder, exist_ok=True)

                srt_clip, subs, srt_index = generate_srt_clip(
                    sentences,
                    start,
                    end,
                    begin_index=srt_index - 1,
                    time_acc_ost=time_acc_ost,
                    normalize=False,
                )
                if not subs:
                    continue

                base_name = os.path.basename(video_file)
                video_name_without_ext, _ = os.path.splitext(base_name)
                start_hours = int(subs[0][0][0] // 3600)
                start_minutes = int((subs[0][0][0] % 3600) // 60)
                start_seconds = int(subs[0][0][0] % 60)
                start_milliseconds = int((subs[0][0][0] - int(subs[0][0][0])) * 100)

                if speaker_id != 'unknown':
                    clip_filename = f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_{start_milliseconds:02}_spk{speaker_id}"
                else:
                    clip_filename = f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_{start_milliseconds:02}"

                clip_filepath = os.path.join(clipped_folder, clip_filename)
                video_clip = clip_filepath + '.mp4'
                audio_clip = clip_filepath + '.wav'
                clip_srt_file = clip_filepath + '.srt'
                vocal_clip = os.path.join(vocal_folder, clip_filename) + '.wav'
                instrumental_clip = os.path.join(instrumental_folder, clip_filename) + '.wav'
                utterance = " ".join(sub[1] for sub in subs).strip()

                if not (os.path.exists(video_clip) and os.path.exists(audio_clip) and os.path.exists(clip_srt_file)):
                    if vocal_file is None:
                        with VideoFileClip(video_file) as video:
                            end = min(end, video.duration)
                            sub = video.subclipped(start, end)
                            sub.write_videofile(video_clip, audio_codec="aac", logger=None)
                            sub.audio.write_audiofile(audio_clip, codec='pcm_s16le', logger=None)
                            sub.close()
                            del sub
                    else:
                        with VideoFileClip(video_file) as video:
                            end = min(end, video.duration)
                            sub_v = video.subclipped(start, end)
                            sub_v.write_videofile(video_clip, audio_codec="aac", logger=None)
                            sub_v.close()
                            del sub_v
                        with AudioFileClip(vocal_file) as vocal:
                            end = min(end, vocal.duration)
                            sub_vo = vocal.subclipped(start, end)
                            sub_vo.write_audiofile(vocal_clip, codec='pcm_s16le', logger=None)
                            sub_vo.close()
                            del sub_vo
                        if instrumental_file:
                            with AudioFileClip(instrumental_file) as instrumental:
                                end = min(end, instrumental.duration)
                                sub_in = instrumental.subclipped(start, end)
                                sub_in.write_audiofile(instrumental_clip, codec='pcm_s16le', logger=None)
                                sub_in.close()
                                del sub_in
                        if raw_audio_file:
                            with AudioFileClip(raw_audio_file) as audio:
                                end = min(end, audio.duration)
                                sub_a = audio.subclipped(start, end)
                                sub_a.write_audiofile(audio_clip, codec='pcm_s16le', logger=None)
                                sub_a.close()
                                del sub_a

                    with open(clip_srt_file, 'w') as fout:
                        fout.write(srt_clip)

                metadata_rows.append({
                    'Clip_Name': os.path.basename(video_clip),
                    'Utterance': utterance,
                    'Duration_Seconds': round(max(0.0, end - start), 3),
                })

                time_acc_ost += (end - start)

            message = f"{len(ts)} clips created"
        else:
            message = "No valid periods found"

        with open(metadata_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['Clip_Name', 'Utterance', 'Duration_Seconds'])
            writer.writeheader()
            writer.writerows(metadata_rows)

        return message


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


def process_single_video(task: VideoTask, device: str) -> Tuple[bool, str, Optional[str], Optional[Dict]]:
    """处理单个视频文件（在工作进程中调用）

    Returns:
        (success, video_file, error_message, timing_info)
    """
    global _worker_model

    timing = {}
    start_time = time.time()

    try:
        video_file = task.video_file
        stage = task.stage
        sd_switch = task.sd_switch
        base_output_dir = task.base_output_dir
        lang = task.lang
        force_restart = task.force_restart

        # Stage 1 and Stage 2 WER both rely on the same ASR backend.
        model = _worker_model if stage in (1, 2) else None

        # 准备文件路径
        t0 = time.time()
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        parent_dir = os.path.dirname(video_file)
        vocal_file = os.path.join(parent_dir, "vocals", f"{video_name}.wav")
        instrumental_file = os.path.join(parent_dir, "instrumental", f"{video_name}.wav")
        raw_audio_file = os.path.join(parent_dir, f"{video_name}.wav")

        if not os.path.exists(vocal_file):
            print(f"[Worker-{os.getpid()}] Warning: Vocal file not found for {video_file}")
            vocal_file = None
        if not os.path.exists(instrumental_file):
            instrumental_file = None
        if not os.path.exists(raw_audio_file):
            raw_audio_file = None

        parent_dir_name = os.path.basename(parent_dir)
        output_dir = os.path.join(base_output_dir, parent_dir_name, video_name)
        os.makedirs(output_dir, exist_ok=True)
        timing['setup'] = time.time() - t0

        # 创建 VideoClipper 实例
        clipper = VideoClipper(model, lang, task.text_normalize)

        # 执行处理
        if stage == 1:
            if vocal_file:
                t1 = time.time()
                wav, sr = librosa.load(vocal_file, sr=16000)
                timing['audio_load'] = time.time() - t1

                t2 = time.time()
                res_srt, state = clipper.recog((sr, wav), vocal_file, sd_switch, {'audio_filename': vocal_file})
                timing['asr'] = time.time() - t2
            else:
                # 分解 video_recog 的各个步骤
                t1 = time.time()
                # 提取音频
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    _, base_name = os.path.split(video_file)
                    base_name, _ = os.path.splitext(base_name)
                    audio_file = os.path.join(output_dir, base_name + '.wav')
                else:
                    base_name, _ = os.path.splitext(video_file)
                    audio_file = base_name + '.wav'

                with VideoFileClip(video_file) as video:
                    if video.audio is None:
                        raise ValueError(f"No audio in video: {video_file}")
                    video.audio.write_audiofile(audio_file, codec='pcm_s16le', logger=None)
                    video.close()
                    del video
                timing['audio_extract'] = time.time() - t1

                t2 = time.time()
                wav, sr = librosa.load(audio_file, sr=16000)
                timing['audio_load'] = time.time() - t2

                t3 = time.time()
                res_srt, state = clipper.recog((sr, wav), audio_file, sd_switch, {'video_filename': video_file})
                timing['asr'] = time.time() - t3

                if os.path.exists(audio_file):
                    os.remove(audio_file)

                timing['video_recog'] = timing['audio_extract'] + timing['audio_load'] + timing['asr']

            t3 = time.time()
            total_srt_file = os.path.join(output_dir, 'total.srt')
            with open(total_srt_file, 'w') as fout:
                fout.write(res_srt)
            write_state(output_dir, state)
            timing['save'] = time.time() - t3

        elif stage == 2:
            if force_restart:
                print(f"[Worker-{os.getpid()}] Force restart enabled - clearing previous outputs for {video_file}")
                clipped_folder = os.path.join(output_dir, 'clipped')
                if os.path.isdir(clipped_folder):
                    shutil.rmtree(clipped_folder)
                metadata_csv_file = os.path.join(output_dir, 'metadata.csv')
                if os.path.isfile(metadata_csv_file):
                    os.remove(metadata_csv_file)

            t1 = time.time()
            state = load_state(output_dir)
            state['video_file'] = video_file
            state['raw_audio_file'] = raw_audio_file
            state['vocal_file'] = vocal_file
            state['instrumental_file'] = instrumental_file
            timing['load_state'] = time.time() - t1

            t2 = time.time()
            message = clipper.video_clip(state, output_dir=output_dir)
            print(f"[Worker-{os.getpid()}] {message} for {video_file}")
            timing['clip'] = time.time() - t2

            t3 = time.time()
            metadata_csv_path = os.path.join(output_dir, 'metadata.csv')
            clip_wer_rows = []
            scored_wers: List[float] = []

            if model is None:
                raise RuntimeError("ASR model is not initialized for stage 2 WER calculation")

            if os.path.isfile(metadata_csv_path):
                with open(metadata_csv_path, 'r', encoding='utf-8') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        clip_name = (row.get('Clip_Name') or '').strip()
                        reference_text = (row.get('Utterance') or '').strip()
                        if not clip_name:
                            continue

                        clip_stem, _ = os.path.splitext(clip_name)
                        clip_audio = os.path.join(output_dir, 'clipped', clip_stem + '.wav')
                        if not os.path.isfile(clip_audio):
                            clip_audio = os.path.join(output_dir, 'vocals', clip_stem + '.wav')
                        if not os.path.isfile(clip_audio):
                            continue

                        wav, sr = librosa.load(clip_audio, sr=16000)
                        _, clip_state = clipper.recog((sr, wav), clip_audio, 'no', {'audio_filename': clip_audio})
                        hypothesis_text = _text_to_string(clip_state.get('recog_res_raw', ''))

                        wer = compute_wer(reference_text, hypothesis_text)
                        clip_wer_rows.append({
                            'Clip_Name': clip_name,
                            'Reference': reference_text,
                            'Hypothesis': hypothesis_text,
                            'WER': '' if wer is None else f"{wer:.6f}",
                        })
                        if wer is not None:
                            scored_wers.append(wer)

            clip_wer_csv = os.path.join(output_dir, 'clip_wer.csv')
            with open(clip_wer_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['Clip_Name', 'Reference', 'Hypothesis', 'WER'])
                writer.writeheader()
                writer.writerows(clip_wer_rows)

            avg_wer = (sum(scored_wers) / len(scored_wers)) if scored_wers else None
            timing['wer'] = time.time() - t3
            timing['wer_summary'] = {
                'parent_folder': parent_dir_name,
                'video_name': video_name,
                'video_output_dir': output_dir,
                'average_wer': avg_wer,
                'scored_clips': len(scored_wers),
                'total_clips': len(clip_wer_rows),
            }

        timing['total'] = time.time() - start_time
        return True, video_file, None, timing

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        timing['total'] = time.time() - start_time
        return False, task.video_file, error_msg, timing


def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                   lang: str, device: str, stage: int):
    """工作进程函数"""
    try:
        # 初始化模型（stage 1 ASR；stage 2 为 WER 评估重新加载）
        if stage == 1 or stage == 2:
            _init_worker_process(gpu_id, lang, device)

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


def get_parser():
    parser = ArgumentParser(description="Video Clipper V2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True, help="Stage: 1=ASR & VAD, 2=Clip")
    parser.add_argument("--file", type=str, required=True, help="Input video file or folder")
    parser.add_argument("--sd_switch", type=str, default="yes", choices=["no", "yes"], help="Enable speaker diarization")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--skip_processed", action="store_true", help="Skip already processed videos")
    parser.add_argument("--force_restart", action="store_true", help="Stage 2 only: delete clipped outputs and restart splitting")
    parser.add_argument("--lang", type=str, default="zh", help="Language: zh or en")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"], help="Device to use")
    parser.add_argument("--text_normalize", action="store_true", help="Enable text normalization (default: disabled)")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of videos to process (for testing)")
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)

    file_or_folder = args.file
    stage = args.stage
    sd_switch = args.sd_switch
    output_dir = args.output_dir
    skip_processed = args.skip_processed
    force_restart = args.force_restart
    lang = args.lang
    device = args.device
    text_normalize = args.text_normalize
    num_samples = args.num_samples

    if device == 'gpu':
        device = 'cuda'

    # 确定使用的设备数量
    use_cuda = device == 'cuda' and torch.cuda.is_available()
    num_workers = torch.cuda.device_count() if use_cuda else 1

    print(f"{'='*60}")
    print(f"Video Clipper V2")
    print(f"Stage: {stage}, Language: {lang}, Device: {device}")
    print(f"Text Normalization: {'Enabled' if text_normalize else 'Disabled'}")
    if use_cuda:
        print(f"Using {num_workers} GPU(s)")
    else:
        print(f"Using CPU")
    print(f"{'='*60}\n")

    # 单文件处理
    if not os.path.isdir(file_or_folder):
        if stage == 1:
            _init_worker_process(0, lang, device)
        elif stage == 2:
            # Stage 2 单文件模式下独立加载 ASR，用于 clip WER 计算
            _init_worker_process(0, lang, device)
        task = VideoTask(file_or_folder, stage, sd_switch, output_dir, lang, 0, force_restart, text_normalize)
        success, video, error, timing = process_single_video(task, device)

        if success:
            print(f"\n✅ Success: {video}")
            if timing:
                print(f"   Timing: {timing}")
            if stage == 2 and timing and timing.get('wer_summary'):
                summary = timing['wer_summary']
                csv_path = write_wer_csv(output_dir, [{
                    'Parent_Folder': summary['parent_folder'],
                    'Video_Name': summary['video_name'],
                    'Video_Output_Dir': summary['video_output_dir'],
                    'Average_WER': '' if summary['average_wer'] is None else f"{summary['average_wer']:.6f}",
                    'Scored_Clips': summary['scored_clips'],
                    'Total_Clips': summary['total_clips'],
                }])
                print(f"   WER summary saved to: {csv_path}")
        else:
            print(f"\n❌ Failed: {video}")
            print(f"Error: {error}")
        return

    # 查找所有视频
    all_videos = find_all_videos(file_or_folder, output_dir, stage, skip_processed)
    print(f"Found {len(all_videos)} videos to process")
    
    # 限制样本数量（用于测试）
    if num_samples is not None:
        all_videos = all_videos[:num_samples]
        print(f"Limited to {num_samples} samples for testing")
    print()

    if not all_videos:
        print("No videos to process")
        return

    # 创建任务列表
    tasks = [VideoTask(video, stage, sd_switch, output_dir, lang, None, force_restart, text_normalize) for video in all_videos]

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
            args=(gpu_id, task_queue, result_queue, lang, device, stage)
        )
        p.start()
        processes.append(p)
        print(f"Started worker process {p.pid} on GPU {gpu_id}")

    print()

    # 收集结果并显示进度
    results = []
    timings = []
    finished_workers = 0
    with tqdm(total=len(tasks), desc="Processing", unit="video") as pbar:
        while finished_workers < num_workers:
            try:
                result = result_queue.get(timeout=1)
                if result is None:
                    finished_workers += 1
                else:
                    success, video_file, error, timing = result
                    results.append((success, video_file, error, timing))
                    if timing:
                        timings.append(timing)

                    # 更新进度条描述，显示平均处理时间
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
    success_count = sum(1 for success, _, _, _ in results if success)
    failed_count = len(results) - success_count

    print(f"\n{'='*60}")
    print(f"Processing Complete")
    print(f"Total: {len(results)}, Success: {success_count}, Failed: {failed_count}")

    if stage == 2:
        wer_rows = []
        for success, _, _, timing in results:
            if not success or not timing:
                continue
            summary = timing.get('wer_summary')
            if not summary:
                continue
            wer_rows.append({
                'Parent_Folder': summary['parent_folder'],
                'Video_Name': summary['video_name'],
                'Video_Output_Dir': summary['video_output_dir'],
                'Average_WER': '' if summary['average_wer'] is None else f"{summary['average_wer']:.6f}",
                'Scored_Clips': summary['scored_clips'],
                'Total_Clips': summary['total_clips'],
            })

        if wer_rows:
            csv_path = write_wer_csv(output_dir, wer_rows)
            print(f"WER summary saved to: {csv_path}")

    # 显示性能统计
    if timings:
        avg_total = sum(t.get('total', 0) for t in timings) / len(timings)
        print(f"\nPerformance Statistics:")
        print(f"  Average total time: {avg_total:.2f}s")

        # 显示各阶段平均时间
        if stage == 1:
            timing_keys = ['audio_extract', 'audio_load', 'asr', 'save']
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
        for success, video, error, _ in results:
            if not success:
                print(f"\n❌ {video}")
                if error:
                    # 只显示错误的前几行，避免输出过长
                    error_lines = error.split('\n')[:10]
                    print(f"   {chr(10).join(error_lines)}")
                    if len(error.split('\n')) > 10:
                        print(f"   ... (truncated)")


if __name__ == '__main__':
    # 清理主进程的 CUDA 环境变量，让子进程自己设置
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']

    mp.set_start_method('spawn', force=True)
    main()
