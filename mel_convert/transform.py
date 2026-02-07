import tgt
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod

# 尝试导入 bigvgan，如果没有则提示
try:
    import bigvgan
except ImportError:
    raise ImportError("请安装 BigVGAN: pip install bigvgan einops")

# ==========================================
# 1. 基类定义 (保持不变)
# ==========================================
class BaseAudioTransformer(ABC):
    """
    音频变换器抽象基类
    """
    
    def __init__(self, 
                 min_gap_threshold: float = 0.001, 
                 max_stretch: float = 8.0,
                 verbose: bool = True):
        self.min_gap_threshold = min_gap_threshold
        self.max_stretch = max_stretch
        self.verbose = verbose
    
    @abstractmethod
    def load_audio(self, path: str): pass
    
    @abstractmethod
    def get_duration(self, audio) -> float: pass
    
    @abstractmethod
    def get_sample_rate(self, audio) -> int: pass
    
    @abstractmethod
    def extract_segment(self, audio, start: float, end: float): pass
    
    @abstractmethod
    def create_silence(self, duration: float, sample_rate: int): pass
    
    @abstractmethod
    def time_stretch(self, audio_segment, stretch_factor: float, is_noise: bool = False): pass
    
    @abstractmethod
    def concatenate_segments(self, segments: List, sample_rate: int): pass
    
    @abstractmethod
    def save_audio(self, audio, output_path: str, sample_rate: int = None): pass
    
    def get_real_words(self, tier: tgt.IntervalTier) -> List[tgt.Interval]:
        return [i for i in tier if i.text not in ['', 'sp', 'sil', '<eps>']]

    def transform(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the transform method.")


# ==========================================
# 2. NVIDIA BigVGAN Transformer 实现
# ==========================================
class BigVGANTransformer(BaseAudioTransformer):
    """
    基于 NVIDIA BigVGAN V2 的变换器
    流程: Wav -> Log-Mel -> Interpolate (Resize) -> BigVGAN -> Wav
    特点: 
    1. 在 Mel 频谱上进行线性插值，完美保持音调。
    2. BigVGAN V2 具有极强的鲁棒性，能修复插值带来的频谱模糊。
    3. 能够处理未见过的说话人 (Zero-shot)。
    """
    
    def __init__(self,
                 model_id: str = "nvidia/bigvgan_v2_22khz_80band_256x",
                 min_gap_threshold: float = 0.001,
                 device: str = "cuda", 
                 verbose: bool = True):
        super().__init__(min_gap_threshold, 8.0, verbose)
        self.device = device
        
        if verbose: print(f"正在加载 BigVGAN 模型: {model_id} ...")
        
        # 1. 加载 BigVGAN 模型
        # use_cuda_kernel=False 兼容性更好，如果报错可以尝试 True
        self.model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False).to(self.device)
        self.model.remove_weight_norm()
        self.model.eval()
        
        # 2. 从模型配置中获取 Mel 参数
        # BigVGAN 对输入 Mel 的参数极度敏感，必须完全匹配
        self.sample_rate = self.model.h.sampling_rate # 通常是 22050
        self.n_fft = self.model.h.n_fft               # 1024
        self.hop_length = self.model.h.hop_size       # 256
        self.win_length = self.model.h.win_size       # 1024
        self.n_mels = self.model.h.num_mels           # 80
        self.fmin = self.model.h.fmin
        # fmax 为 None 时自动设为采样率的一半 (librosa/torchaudio 的默认行为)
        self.fmax = self.model.h.fmax if self.model.h.fmax is not None else self.sample_rate / 2.0
        
        if verbose:
            print(f"✅ BigVGAN 加载成功")
            print(f"   Sample Rate: {self.sample_rate} Hz")
            print(f"   Mel Bands: {self.n_mels}")
            print(f"   fmin: {self.fmin} Hz, fmax: {self.fmax} Hz")
            print(f"   n_fft: {self.n_fft}, win_length: {self.win_length}")
            print(f"   Hop Length: {self.hop_length}")

        # 3. 初始化配套的 Mel 提取器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=2.0,
            normalized=False,
            center=True # BigVGAN 官方实现使用 center padding
        ).to(self.device)

    def load_audio(self, path: str):
        """加载音频 -> 重采样 -> 返回 (wav_tensor, sr)"""
        wav, sr = torchaudio.load(path)
        wav = wav.to(self.device)
        
        # 必须重采样到 BigVGAN 的采样率 (24kHz)
        if sr != self.sample_rate:
            if self.verbose: 
                # print(f"Resampling {sr} -> {self.sample_rate}")
                pass
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            wav = resampler(wav)
            
        # 确保是 (Batch, Time)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            
        return wav, self.sample_rate

    def get_duration(self, audio) -> float:
        # audio: (wav, sr)
        return audio[0].shape[-1] / audio[1]

    def get_sample_rate(self, audio) -> int:
        return audio[1]

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        波形转 Log-Mel 频谱 (符合 BigVGAN 输入要求)
        官方流程: STFT -> Magnitude (power=2) -> Sqrt -> Mel Filterbank -> Log
        """
        # 1. Extract Mel (power=2.0 已在 mel_transform 中设置)
        # center=True 时,PyTorch 会自动处理 padding
        mel = self.mel_transform(wav)  # 输出: magnitude^2
        
        # 2. 取平方根获得线性幅度谱
        mel = torch.sqrt(mel + 1e-9)
        
        # 3. Logarithm (BigVGAN 使用 dynamic_range_compression)
        # 与官方 spectral_normalize_torch 等价
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return log_mel

    def extract_segment(self, audio, start: float, end: float):
        """
        提取波形片段并直接转换为 Log-Mel
        Returns: Log-Mel Tensor (1, n_mels, time)
        """
        wav, sr = audio
        if end <= start: return None
        
        start_idx = int(round(start * sr))
        end_idx = int(round(end * sr))
        
        start_idx = max(0, start_idx)
        end_idx = min(wav.shape[-1], end_idx)
        
        if start_idx >= end_idx: return None
        
        wav_seg = wav[:, start_idx:end_idx]
        
        # 立即转为 Mel
        with torch.no_grad():
            mel_seg = self.wav_to_mel(wav_seg)
            
        return mel_seg

    def create_silence(self, duration: float, sample_rate: int):
        """生成静音的 Mel 频谱"""
        if duration <= self.min_gap_threshold: return None
        
        # 为了获得准确的静音数值（Log Mel 下不是0，而是负数），我们生成静音波形跑一遍流程
        num_samples = int(duration * sample_rate)
        silent_wav = torch.zeros((1, num_samples), device=self.device)
        
        with torch.no_grad():
            silent_mel = self.wav_to_mel(silent_wav)
            
        return silent_mel

    def time_stretch(self, audio_segment, stretch_factor: float, is_noise: bool = False):
        """
        在 Mel 频谱上进行线性插值 (Linear Interpolation)
        """
        if audio_segment is None: return None
        
        current_frames = audio_segment.shape[-1]
        target_frames = int(current_frames * stretch_factor)
        
        return self.time_stretch_exact(audio_segment, target_frames)
        
    def time_stretch_exact(self, mel_segment, target_frames: int):
        """精确拉伸到指定帧数"""
        if mel_segment is None or target_frames <= 0:
            return torch.zeros((1, self.n_mels, 0), device=self.device)
            
        # 线性插值 (Image Resize)
        # Input: (Batch, Channels, Length)
        mel_out = F.interpolate(
            mel_segment,
            size=target_frames,
            mode='linear',
            align_corners=False
        )
        return mel_out

    def concatenate_segments(self, segments: List, sample_rate: int):
        # 这里的 segments 都是 Mel
        valid = [s for s in segments if s is not None and s.shape[-1] > 0]
        if not valid: return None
        return torch.cat(valid, dim=2)

    def save_audio(self, audio_mel, output_path: str, sample_rate: int = None):
        """
        Vocoding: Log-Mel -> BigVGAN -> Waveform -> File
        """
        if self.verbose: print(f"Generating waveform using BigVGAN...")
        
        with torch.no_grad():
            # BigVGAN 推理
            wav_gen = self.model(audio_mel)
            
        wav_cpu = wav_gen.cpu().squeeze(0) # (Channels, Time)
        
        # 归一化防止爆音
        max_val = wav_cpu.abs().max()
        if max_val > 0.99:
            wav_cpu = wav_cpu / max_val * 0.99
            
        torchaudio.save(output_path, wav_cpu, self.sample_rate)

    def transform(self, source_audio_path, source_textgrid_path, target_textgrid_path, target_audio_path, output_path, tier_name="words"):
        try:
            if self.verbose: print(f"Starting Processing (BigVGAN Paradigm)...")
            
            # 1. 加载源音频
            wav_src_tuple = self.load_audio(source_audio_path) # (wav, 24000)
            
            # 2. 获取目标参考时长
            wav_tgt_raw, sr_tgt_raw = torchaudio.load(target_audio_path)
            target_duration_sec = wav_tgt_raw.shape[-1] / sr_tgt_raw
            
            # 计算总帧数 (Canvas Width)
            # Frame = Sec * SR / Hop
            total_target_frames = int(target_duration_sec * self.sample_rate / self.hop_length)
            
            if self.verbose:
                print(f"Target Duration: {target_duration_sec:.3f}s")
                print(f"Target Frames: {total_target_frames} (hop={self.hop_length})")

            # 3. 加载 TextGrid
            tg_src = tgt.io.read_textgrid(source_textgrid_path)
            tg_tgt = tgt.io.read_textgrid(target_textgrid_path)
            words_src = self.get_real_words(tg_src.get_tier_by_name(tier_name))
            words_tgt = self.get_real_words(tg_tgt.get_tier_by_name(tier_name))
            
            if len(words_src) != len(words_tgt):
                print(f"Error: Word mismatch! {len(words_src)} vs {len(words_tgt)}")
                return False

            # 4. 创建 Mel 画布 (Canvas)
            # 获取静音的 Log-Mel 值作为背景底色
            silence_ref = self.create_silence(0.05, self.sample_rate)
            # 取静音片段的平均值或最小值作为填充值
            silence_val = silence_ref.mean() 
            
            # Shape: (1, 100, Total_Frames)
            final_mel = torch.ones((1, self.n_mels, total_target_frames), device=self.device) * silence_val
            
            last_end_tgt = 0.0
            last_end_src = 0.0
            
            # 辅助函数：秒 -> 帧
            def sec2frame(sec): return int(sec * self.sample_rate / self.hop_length)
            
            # 5. 逐词拼贴
            for i in range(len(words_tgt)):
                w_src = words_src[i]
                w_tgt = words_tgt[i]
                
                # === A. 填补间隙 (Gap) ===
                gap_start = sec2frame(last_end_tgt)
                gap_end = sec2frame(w_tgt.start_time)
                gap_len = gap_end - gap_start
                
                if gap_len > 0:
                    gap_src_dur = w_src.start_time - last_end_src
                    # 只有当源音频也有足够长的间隙时，才提取底噪
                    if gap_src_dur > self.min_gap_threshold:
                        mel_seg = self.extract_segment(wav_src_tuple, last_end_src, w_src.start_time)
                        if mel_seg is not None:
                            processed = self.time_stretch_exact(mel_seg, gap_len)
                            # 贴到画布 (注意边界检查)
                            valid = min(gap_len, final_mel.shape[-1] - gap_start)
                            if valid > 0:
                                final_mel[..., gap_start : gap_start + valid] = processed[..., :valid]
                                
                # === B. 填补单词 (Word) ===
                word_start = sec2frame(w_tgt.start_time)
                word_end = sec2frame(w_tgt.end_time)
                word_len = word_end - word_start
                
                if word_len > 0:
                    mel_seg = self.extract_segment(wav_src_tuple, w_src.start_time, w_src.end_time)
                    if mel_seg is not None:
                        # 核心：在 Mel 上做线性拉伸，保持音调不变
                        processed = self.time_stretch_exact(mel_seg, word_len)
                        valid = min(word_len, final_mel.shape[-1] - word_start)
                        if valid > 0:
                            final_mel[..., word_start : word_start + valid] = processed[..., :valid]
                            
                last_end_tgt = w_tgt.end_time
                last_end_src = w_src.end_time
            
            # 6. 尾部处理
            tail_start = sec2frame(last_end_tgt)
            tail_len = total_target_frames - tail_start
            
            if tail_len > 0:
                src_total_dur = self.get_duration(wav_src_tuple)
                tail_src_dur = src_total_dur - last_end_src
                if tail_src_dur > self.min_gap_threshold:
                    mel_seg = self.extract_segment(wav_src_tuple, last_end_src, src_total_dur)
                    if mel_seg is not None:
                        processed = self.time_stretch_exact(mel_seg, tail_len)
                        valid = min(tail_len, final_mel.shape[-1] - tail_start)
                        if valid > 0:
                            final_mel[..., tail_start : tail_start + valid] = processed[..., :valid]

            # 7. 生成并保存
            self.save_audio(final_mel, output_path)
            
            if self.verbose: print(f"✅ 完成！文件已保存: {output_path}")
            return True

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


# ==========================================
# 3. 运行示例
# ==========================================
if __name__ == "__main__":
    # 检查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 实例化 BigVGAN 变换器
    # 首次运行会自动下载模型 (约 200-300MB)
    transformer = BigVGANTransformer(
        device=device,
        verbose=True
    )
    
    # 执行变换
    transformer.transform(
        source_audio_path="mel_convert/test/test_short1_1.wav",
        source_textgrid_path="mel_convert/test/aligned/test_short1_1.TextGrid",
        target_textgrid_path="mel_convert/test/aligned/test_short1_2.TextGrid",
        target_audio_path="mel_convert/test/test_short1_2.wav",
        output_path="output_bigvgan.wav",
        tier_name="phones" # 建议使用 phones 层级，控制更细腻
    )