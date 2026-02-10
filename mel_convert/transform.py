import tgt
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import PchipInterpolator
from pathlib import Path
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from meldataset import get_mel_spectrogram

try:
    import bigvgan
except ImportError:
    raise ImportError("请安装 BigVGAN: pip install bigvgan einops")

# ==========================================
# 1. 基类定义
# ==========================================
class BaseAudioTransformer(ABC):
    def __init__(self, min_gap_threshold: float = 0.001, max_stretch: float = 8.0, verbose: bool = True):
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
    def save_audio(self, audio, output_path: str, sample_rate: int = None): pass
    
    def get_real_words(self, tier: tgt.IntervalTier) -> List[tgt.Interval]:
        return [i for i in tier if i.text not in ['', 'sp', 'sil', '<eps>']]

    def transform(self, *args, **kwargs):
        raise NotImplementedError

# ==========================================
# 2. 全局光滑扭曲 Transformer
# ==========================================
class GlobalWarpTransformer(BaseAudioTransformer):
    """
    基于全局时间扭曲 (Global Time Warping) 的变换器。
    不切片，不拼接，而是通过计算一条光滑的时间映射曲线，
    一次性将源 Mel 频谱“流变”为目标 Mel 频谱。
    """
    
    def __init__(self,
                 model_id: str = "nvidia/bigvgan_v2_22khz_80band_256x",
                 device: str = "cuda", 
                 verbose: bool = True):
        super().__init__(0.001, 8.0, verbose)
        self.device = device
        
        if verbose: print(f"加载 BigVGAN: {model_id} ...")
        
        try:
            self.model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False).to(self.device)
        except:
            print("Fallback load...")
            self.model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False).to(self.device)

        self.model.remove_weight_norm()
        self.model.eval()
        
        self.h = self.model.h
        self.sample_rate = self.h.sampling_rate
        if not hasattr(self.h, 'fmax') or self.h.fmax is None:
            self.h.fmax = self.sample_rate / 2.0
            
        if verbose: print(f"✅ Ready. SR: {self.sample_rate}, Hop: {self.h.hop_size}")

    def load_audio(self, path: str):
        wav, sr = torchaudio.load(path)
        wav = wav.to(self.device)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            wav = resampler(wav)
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        return wav, self.sample_rate

    def get_duration(self, audio) -> float:
        return audio[0].shape[-1] / audio[1]
    
    def get_sample_rate(self, audio) -> int:
        return audio[1]

    def create_silence(self, duration: float, sample_rate: int):
        num_samples = int(duration * sample_rate)
        silent_wav = torch.randn((1, num_samples), device=self.device) * 1e-6
        with torch.no_grad():
            silent_mel = get_mel_spectrogram(silent_wav, self.h)
        return silent_mel

    def save_audio(self, audio_mel, output_path: str, sample_rate: int = None):
        if self.verbose: print(f"Vocoding...")
        with torch.no_grad():
            wav_gen = self.model(audio_mel)
        wav_cpu = wav_gen.cpu().squeeze(0)
        wav_cpu = torch.tanh(wav_cpu) # Soft clip
        torchaudio.save(output_path, wav_cpu, self.sample_rate)

    def calculate_warping_path(self, src_anchors, tgt_anchors, total_tgt_frames):
        """
        核心算法：计算光滑的时间映射路径
        src_anchors: 源音频的关键时间点 (秒)
        tgt_anchors: 目标音频的关键时间点 (秒)
        """
        # 1. 转换为帧索引
        def sec2frame(sec): return sec * self.sample_rate / self.h.hop_size
        
        src_frames = np.array([sec2frame(t) for t in src_anchors])
        tgt_frames = np.array([sec2frame(t) for t in tgt_anchors])
        
        # 2. 使用 Pchip 插值 (Piecewise Cubic Hermite Interpolating Polynomial)
        # 为什么要用 Pchip 而不是 CubicSpline？
        # 因为 Pchip 保证单调性 (Monotonicity)。时间是不能倒流的。
        # 普通样条曲线可能会产生震荡，导致时间“倒退”，这是不允许的。
        interpolator = PchipInterpolator(tgt_frames, src_frames)
        
        # 3. 生成所有目标帧对应的源帧索引
        grid_tgt = np.arange(total_tgt_frames)
        grid_src = interpolator(grid_tgt)
        
        return torch.from_numpy(grid_src).float().to(self.device)

    def warp_mel(self, source_mel, warping_path):
        """
        使用 grid_sample 进行非均匀拉伸
        source_mel: (1, n_mels, src_len)
        warping_path: (tgt_len, ) 每个目标帧对应的源帧位置
        """
        B, C, src_len = source_mel.shape
        tgt_len = warping_path.shape[0]
        
        # grid_sample 需要归一化到 [-1, 1] 的坐标
        # -1 代表 index 0, 1 代表 index src_len-1
        
        # 归一化 X 坐标 (时间轴)
        # map: 0 -> -1, src_len-1 -> 1
        # formula: 2 * (x / (src_len - 1)) - 1
        norm_x = 2 * (warping_path / (src_len - 1)) - 1
        
        # 扩展维度以适配 grid_sample (N, H, W, 2)
        # 我们把 Mel 看作高度为 1 的图像，宽度为 Time
        # 其实可以直接处理 2D，这里为了物理意义清晰，我们只扭曲时间轴
        
        # grid shape: (1, 1, tgt_len, 2)
        # 最后一个维度 2 代表 (x, y)。我们只变 x，y 保持不变
        
        # 构建 Y 坐标 (频率轴)
        # 频率轴不需要扭曲，所以是线性的 -1 到 1
        # 但 grid_sample 的 grid 是输出图像的坐标网格。
        # 对于 Mel 谱，我们希望保留所有频率信息，所以可以用一种 hack：
        # 将 Mel 视为 (1, C, T) 的 1D 序列处理，或者 (C, T) 的 2D 图片
        
        # 更简单的做法：把 Mel 视为 Batch=1, Channels=n_mels, Height=1, Width=src_len
        # 这样我们只需要在 Width 方向采样
        
        source_mel_4d = source_mel.unsqueeze(2) # (1, n_mels, 1, src_len)
        
        # 构建 Grid
        # x: norm_x (1, 1, tgt_len)
        # y: 0 (代表中心)
        grid_x = norm_x.view(1, 1, tgt_len, 1).expand(1, 1, -1, 1) # (1, 1, tgt_len, 1)
        grid_y = torch.zeros_like(grid_x) # (1, 1, tgt_len, 1)
        
        # grid: (1, 1, tgt_len, 2)
        grid = torch.cat([grid_x, grid_y], dim=-1)
        
        # 采样
        # mode='bicubic': 双三次插值，保证光滑
        # padding_mode='border': 防止边缘黑边
        warped_mel = F.grid_sample(source_mel_4d, grid, mode='bicubic', padding_mode='border', align_corners=True)
        
        return warped_mel.squeeze(2) # (1, n_mels, tgt_len)


    def transform(self, source_audio_path, source_textgrid_path, target_textgrid_path, target_audio_path, output_path, tier_name="words"):
        try:
            if self.verbose: print(f"Processing Global Warp...")
            
            # 1. 加载源音频并转 Mel
            wav_src, sr = self.load_audio(source_audio_path)
            with torch.no_grad():
                mel_src = get_mel_spectrogram(wav_src, self.h)
            
            # 2. 准备 TextGrid
            tg_src = tgt.io.read_textgrid(source_textgrid_path)
            tg_tgt = tgt.io.read_textgrid(target_textgrid_path)
            words_src = self.get_real_words(tg_src.get_tier_by_name(tier_name))
            words_tgt = self.get_real_words(tg_tgt.get_tier_by_name(tier_name))
            
            if len(words_src) != len(words_tgt): return False
            
            # 3. 构建关键点 (Anchors)
            # 我们需要建立 Source Time <-> Target Time 的映射关系
            # 起点 (0, 0)
            src_anchors = [0.0]
            tgt_anchors = [0.0]
            
            for i in range(len(words_tgt)):
                w_src = words_src[i]
                w_tgt = words_tgt[i]
                
                # 添加单词的起点和终点作为“桩”
                # 需要保证 Target Time 是严格单调递增的
                
                # --- Start Point ---
                t_start = w_tgt.start_time
                s_start = w_src.start_time
                
                # 检查与上一个点是否重复或逆序
                if t_start <= tgt_anchors[-1]:
                    # 如果时间完全重叠，且源时间也重叠，那就是完全同一个点（例如单词紧挨着），跳过添加
                    if abs(t_start - tgt_anchors[-1]) < 1e-6 and abs(s_start - src_anchors[-1]) < 1e-6:
                        pass
                    else:
                        # 如果时间重叠但源时间不同（跳跃），或者微小逆序，强制稍微平移一点
                        # 0.1ms 的平移对听感无影响，但能满足严格单调递增条件
                        t_start = tgt_anchors[-1] + 1e-4
                        tgt_anchors.append(t_start)
                        src_anchors.append(s_start)
                else:
                    tgt_anchors.append(t_start)
                    src_anchors.append(s_start)

                # --- End Point ---
                t_end = w_tgt.end_time
                s_end = w_src.end_time
                
                if t_end <= tgt_anchors[-1]:
                    if abs(t_end - tgt_anchors[-1]) < 1e-6 and abs(s_end - src_anchors[-1]) < 1e-6:
                        pass
                    else:
                        t_end = tgt_anchors[-1] + 1e-4
                        tgt_anchors.append(t_end)
                        src_anchors.append(s_end)
                else:
                    tgt_anchors.append(t_end)
                    src_anchors.append(s_end)
            
            # 加上尾部终点
            # 获取目标总时长
            wav_tgt_raw, sr_tgt_raw = torchaudio.load(target_audio_path)
            tgt_duration = wav_tgt_raw.shape[-1] / sr_tgt_raw
            src_duration = wav_src.shape[-1] / sr
            
            # 如果最后一个单词结束时间小于总时长，把终点也加上
            # 同样需要保证严格单调递增
            if tgt_anchors[-1] < tgt_duration - 1e-4: # 留一点余量，防止浮点误差导致极其接近的点
                tgt_anchors.append(tgt_duration)
                src_anchors.append(src_duration)
            elif tgt_anchors[-1] >= tgt_duration:
                # 如果由于前面的 shift 导致最后一点已经超过了实际时长，
                # 其实也没关系，interpolate 会覆盖整个 range。
                # 但为了逻辑严谨，我们不再添加回退的点。
                pass
            
            # 4. 计算光滑路径
            # 目标总帧数
            total_target_frames = int(tgt_duration * self.sample_rate / self.h.hop_size)
            
            warping_path = self.calculate_warping_path(src_anchors, tgt_anchors, total_target_frames)
            
            # 5. 执行全局扭曲
            # 这一步包含了你想要的所有特性：
            # - 速率自动平滑 (Pchip 插值特性)
            # - 无拼接断点 (全局采样)
            # - 音调不变 (Mel 域操作)
            final_mel = self.warp_mel(mel_src, warping_path)
            
            # 6. 生成保存
            self.save_audio(final_mel, output_path)
            
            if self.verbose: print(f"✅ Saved: {output_path}")
            return True

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer = GlobalWarpTransformer(device=device, verbose=True)
    
    transformer.transform(
        source_audio_path="mel_convert/test/test_long1_1.wav",
        source_textgrid_path="mel_convert/test/aligned/test_long1_1.TextGrid",
        target_textgrid_path="mel_convert/test/aligned/test_long1_2.TextGrid",
        target_audio_path="mel_convert/test/test_long1_2.wav",
        output_path="output_.wav",
        tier_name="words"
    )
