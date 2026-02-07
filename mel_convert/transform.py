import tgt
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod
from meldataset import get_mel_spectrogram

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
    基于 NVIDIA BigVGAN 的变换器。
    使用外部引入的 get_mel_spectrogram 确保特征提取 100% 准确。
    """
    
    def __init__(self,
                 model_id: str = "nvidia/bigvgan_v2_22khz_80band_256x",
                 min_gap_threshold: float = 0.001,
                 device: str = "cuda", 
                 verbose: bool = True):
        super().__init__(min_gap_threshold, 8.0, verbose)
        self.device = device
        
        if verbose: print(f"加载 BigVGAN 模型: {model_id} ...")
        
        # 1. 加载模型
        try:
            self.model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False).to(self.device)
        except:
            # 兼容性加载
            print("尝试使用备用方式加载...")
            self.model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False).to(self.device)

        self.model.remove_weight_norm()
        self.model.eval()
        
        # 2. 获取参数 (h)
        # 这些参数将直接传给 get_mel_spectrogram
        self.h = self.model.h
        self.sample_rate = self.h.sampling_rate
        
        # 3. 检查 fmax (防止为 null 导致的潜在问题)
        # 虽然 get_mel_spectrogram 可能内部处理了，但为了 canvas 计算，我们需要确保它是个数字
        if not hasattr(self.h, 'fmax') or self.h.fmax is None:
            self.h.fmax = self.sample_rate / 2.0
            
        if verbose:
            print(f"✅ Ready. SR: {self.sample_rate}, Hop: {self.h.hop_size}, Mels: {self.h.num_mels}")

    def load_audio(self, path: str):
        """加载音频 -> 重采样 (使用 librosa 以与 BigVGAN 官方示例一致)"""
        # 使用 librosa.load 自动重采样到目标采样率
        # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        
        # 转换为 PyTorch tensor: wav is FloatTensor with shape [B(1), T_time]
        wav = torch.FloatTensor(wav).unsqueeze(0)
        wav = wav.to(self.device)
            
        return wav, self.sample_rate

    def get_duration(self, audio) -> float:
        return audio[0].shape[-1] / audio[1]

    def get_sample_rate(self, audio) -> int:
        return audio[1]

    def extract_segment(self, audio, start: float, end: float):
        wav, sr = audio
        if end <= start: return None
        
        # 计算采样点
        start_idx = max(0, int(round(start * sr)))
        end_idx = min(wav.shape[-1], int(round(end * sr)))
        
        if start_idx >= end_idx: return None
        
        # 提取波形 (1, T)
        wav_seg = wav[:, start_idx:end_idx]
        
        # === 核心改动：直接调用外部函数 ===
        # 假设 get_mel_spectrogram(wav, h) -> returns (B, n_mels, T)
        with torch.no_grad():
            mel_seg = get_mel_spectrogram(wav_seg, self.h)
            
        return mel_seg

    def create_silence(self, duration: float, sample_rate: int):
        if duration <= self.min_gap_threshold: return None
        
        num_samples = int(duration * sample_rate)
        # 使用微弱噪音代替纯0，避免 Log-Mel 出现 -inf
        silent_wav = torch.randn((1, num_samples), device=self.device) * 1e-6
        
        with torch.no_grad():
            silent_mel = get_mel_spectrogram(silent_wav, self.h)
            
        return silent_mel

    def time_stretch(self, audio_segment, stretch_factor: float, is_noise: bool = False):
        # 这里的 audio_segment 是 Mel
        if audio_segment is None: return None
        current_frames = audio_segment.shape[-1]
        target_frames = int(current_frames * stretch_factor)
        return self.time_stretch_exact(audio_segment, target_frames)
        
    def time_stretch_exact(self, mel_segment, target_frames: int):
        """
        使用双三次插值 (Bicubic) 进行 Mel 拉伸，保持最佳纹理
        """
        if mel_segment is None or target_frames <= 0:
            return torch.zeros((1, self.h.num_mels, 0), device=self.device)
            
        # F.interpolate 'bicubic' 需要 4D 输入 (B, C, H, W)
        # Mel 是 (B, n_mels, Time)，我们将其视为 (B, 1, n_mels, Time)
        mel_img = mel_segment.unsqueeze(1)
        
        mel_out = F.interpolate(
            mel_img,
            size=(self.h.num_mels, target_frames), # 高度不变，只变宽度(时间)
            mode='bicubic',
            align_corners=False
        )
        
        return mel_out.squeeze(1)

    # 移除 concatenate_segments，因为我们在 transform 里直接操作 Canvas

    def paste_with_crossfade(self, canvas, segment, start_frame):
        """Mel 域的简单拼接"""
        if segment is None: return
        
        seg_len = segment.shape[-1]
        canvas_len = canvas.shape[-1]
        
        # 边界检查
        if start_frame >= canvas_len: return
        
        # 计算有效长度
        valid_len = min(seg_len, canvas_len - start_frame)
        if valid_len <= 0: return
        
        # 直接覆盖 (Mel 域直接覆盖通常比时域硬拼接好很多)
        # 如果需要更平滑，可以在这里实现 Mel 值的加权平均，但 Bicubic 插值通常已经足够平滑
        canvas[..., start_frame : start_frame + valid_len] = segment[..., :valid_len]

    def save_audio(self, audio_mel, output_path: str, sample_rate: int = None):
        if self.verbose: print(f"Vocoding...")
        
        with torch.no_grad():
            # BigVGAN 生成
            wav_gen = self.model(audio_mel)
            
        wav_cpu = wav_gen.cpu().squeeze(0)
        
        # 软削波 (Soft Clipping) - 模拟电子管过载，比硬截断更悦耳
        wav_cpu = torch.tanh(wav_cpu)
        
        # 输出时长信息
        if self.verbose: 
            duration_sec = wav_cpu.shape[-1] / self.sample_rate
            print(f"生成音频时长: {duration_sec:.4f} 秒")
        
        torchaudio.save(output_path, wav_cpu, self.sample_rate)

    def transform(self, source_audio_path, source_textgrid_path, target_textgrid_path, target_audio_path, output_path, tier_name="words"):
        try:
            if self.verbose: print(f"Start Processing...")
            
            # 1. 加载
            wav_src_tuple = self.load_audio(source_audio_path)
            
            
            # 2. 计算目标帧数
            wav_tgt_raw, sr_tgt_raw = torchaudio.load(target_audio_path)
            target_duration_sec = wav_tgt_raw.shape[-1] / sr_tgt_raw
            total_target_frames = int(target_duration_sec * self.sample_rate / self.h.hop_size)
            
            # 计算目标音频时长
            if self.verbose:
                print(f"目标音频时长: {target_duration_sec:.4f} 秒, 目标 Mel 帧数: {total_target_frames}")
            
            # 3. 加载 TextGrid
            tg_src = tgt.io.read_textgrid(source_textgrid_path)
            tg_tgt = tgt.io.read_textgrid(target_textgrid_path)
            words_src = self.get_real_words(tg_src.get_tier_by_name(tier_name))
            words_tgt = self.get_real_words(tg_tgt.get_tier_by_name(tier_name))
            
            if len(words_src) != len(words_tgt): return False

            # 4. 创建画布 (使用静音底噪填充)
            silence_ref = self.create_silence(0.1, self.sample_rate)
            silence_val = silence_ref.mean()
            final_mel = torch.ones((1, self.h.num_mels, total_target_frames), device=self.device) * silence_val
            
            last_end_tgt = 0.0
            last_end_src = 0.0
            
            # 秒 -> 帧 转换器
            def sec2frame(sec): return int(sec * self.sample_rate / self.h.hop_size)
            
            # 5. 循环处理
            for i in range(len(words_tgt)):
                w_src = words_src[i]
                w_tgt = words_tgt[i]
                
                # A. 间隙
                gap_start = sec2frame(last_end_tgt)
                gap_end = sec2frame(w_tgt.start_time)
                gap_len = gap_end - gap_start
                
                if gap_len > 0:
                    gap_src_dur = w_src.start_time - last_end_src
                    if gap_src_dur > self.min_gap_threshold:
                        mel_seg = self.extract_segment(wav_src_tuple, last_end_src, w_src.start_time)
                        if mel_seg is not None:
                            processed = self.time_stretch_exact(mel_seg, gap_len)
                            self.paste_with_crossfade(final_mel, processed, gap_start)

                # B. 单词
                word_start = sec2frame(w_tgt.start_time)
                word_end = sec2frame(w_tgt.end_time)
                word_len = word_end - word_start
                
                if word_len > 0:
                    mel_seg = self.extract_segment(wav_src_tuple, w_src.start_time, w_src.end_time)
                    if mel_seg is not None:
                        processed = self.time_stretch_exact(mel_seg, word_len)
                        self.paste_with_crossfade(final_mel, processed, word_start)
                        
                last_end_tgt = w_tgt.end_time
                last_end_src = w_src.end_time
            
            # 6. 尾部
            tail_start = sec2frame(last_end_tgt)
            tail_len = total_target_frames - tail_start
            if tail_len > 0:
                src_dur = self.get_duration(wav_src_tuple)
                if src_dur - last_end_src > self.min_gap_threshold:
                    mel_seg = self.extract_segment(wav_src_tuple, last_end_src, src_dur)
                    if mel_seg is not None:
                        processed = self.time_stretch_exact(mel_seg, tail_len)
                        self.paste_with_crossfade(final_mel, processed, tail_start)

            # 7. 保存
            self.save_audio(final_mel, output_path)
            
            if self.verbose: print(f"✅ Saved: {output_path}")
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
        source_audio_path="mel_convert/test/test_long1_1.wav",
        source_textgrid_path="mel_convert/test/aligned/test_long1_1.TextGrid",
        target_textgrid_path="mel_convert/test/aligned/test_long1_2.TextGrid",
        target_audio_path="mel_convert/test/test_long1_2.wav",
        output_path="output_bigvgan_long.wav",
        tier_name="words" # 建议使用 phones 层级，控制更细腻
    )