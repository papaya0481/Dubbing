import tgt
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import PchipInterpolator
from pathlib import Path
from typing import Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from types import SimpleNamespace

import json
from typing import Dict

try:
    from .meldataset import get_mel_spectrogram
except Exception:
    from meldataset import get_mel_spectrogram

try:
    import bigvgan
except ImportError:
    bigvgan = None
    
from logger import get_logger
logger = get_logger("dubbing.mel_transform")

# ==========================================
# 1. 基类定义
# ==========================================
class BaseAudioTransformer(ABC):
    def __init__(self, min_gap_threshold: float = 0.001, max_stretch: float = 8.0, verbose: bool = True) -> None:
        """初始化音频变换基类。/ Initialize base audio transformer settings."""
        self.min_gap_threshold = min_gap_threshold
        self.max_stretch = max_stretch
        self.verbose = verbose
    
    @abstractmethod
    def load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """加载音频并返回波形与采样率。/ Load audio and return waveform with sample rate."""
        pass

    @abstractmethod
    def get_duration(self, audio: Tuple[torch.Tensor, int]) -> float:
        """获取音频时长（秒）。/ Get audio duration in seconds."""
        pass

    @abstractmethod
    def get_sample_rate(self, audio: Tuple[torch.Tensor, int]) -> int:
        """获取音频采样率。/ Get sample rate from audio container."""
        pass

    @abstractmethod
    def save_audio(self, audio: torch.Tensor, output_path: str, sample_rate: Optional[int] = None) -> None:
        """保存音频结果。/ Save transformed audio to disk."""
        pass
    
    def get_real_words(self, tier: tgt.IntervalTier) -> List[tgt.Interval]:
        """提取有效词区间。/ Extract non-silence word intervals from a tier."""
        return [i for i in tier if i.text not in ['', 'sp', 'sil', '<eps>']]

    def transform(self, *args: Any, **kwargs: Any) -> bool:
        """执行变换流程。/ Run the transformation pipeline."""
        raise NotImplementedError


class GlobalWarpTransformer(BaseAudioTransformer):
    """
    基于全局时间扭曲 (Global Time Warping) 的变换器。
    不切片，不拼接，而是通过计算一条光滑的时间映射曲线，
    一次性将源 Mel 频谱“流变”为目标 Mel 频谱。
    """
    
    def __init__(self,
                 use_vocoder: bool = True,
                 model_id: str = "nvidia/bigvgan_v2_22khz_80band_256x",
                 vocoder_model: Optional[Any] = None,
                 device: str = "cuda", 
                 phoneme_mapping_path: Optional[dict] = "dubbing/modules/english_us_arpa_300.json",
                 verbose: bool = True) -> None:
        """
        初始化全局时间扭曲变换器。/ Initialize global time-warp transformer.
            use_vocoder (bool): 是否使用声码器将 Mel 转回音频。/ Whether to use vocoder to convert mel back to audio.
            vocoder_model (Optional[Any]): 可选的自定义声码器。提供后将强覆盖并忽略 model_id。
        Args:
            model_id (str): BigVGAN pretrained model ID.
            vocoder_model (Optional[Any]): Vocoder model instance.
        """
        super().__init__()
        self.use_vocoder = use_vocoder
        self.device = device
        self.model_id = model_id
        self.model = None
        self.phoneme_mapping = self._load_phoneme_mapping(phoneme_mapping_path)

        def _default_hparams_from_model_id(mid: str) -> SimpleNamespace:
            from transformers import AutoConfig
            try:
                config = AutoConfig.from_pretrained(mid)
                return SimpleNamespace(
                    sampling_rate=config.sampling_rate,
                    n_fft=config.n_fft,
                    num_mels=config.num_mels,
                    hop_size=config.hop_size,
                    win_size=config.win_size,
                    fmin=config.fmin,
                    fmax=config.fmax if hasattr(config, 'fmax') else config.sampling_rate / 2.0,
                )
            except Exception:
                raise ValueError(f"Failed to load config for model_id '{mid}'.")  

        if not self.use_vocoder:
            self.h = _default_hparams_from_model_id(model_id)
            self.sample_rate = self.h.sampling_rate
            if self.verbose:
                logger.warning(f"Vocoder disabled. Use model_id params only. SR: {self.sample_rate}, Hop: {self.h.hop_size}")
            return

        if bigvgan is None:
            raise ImportError("请安装 BigVGAN: pip install bigvgan einops")

        if vocoder_model is not None:
            self.model = vocoder_model.to(self.device) if hasattr(vocoder_model, "to") else vocoder_model
        else:
            self.model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False).to(self.device)

        if hasattr(self.model, "remove_weight_norm"):
            self.model.remove_weight_norm()
        self.model.eval()

        self.h = self.model.h
        self.sample_rate = self.h.sampling_rate
        if not hasattr(self.h, 'fmax') or self.h.fmax is None:
            self.h.fmax = self.sample_rate / 2.0

        if self.verbose:
            print(f"✅ Ready. SR: {self.sample_rate}, Hop: {self.h.hop_size}")

    def load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """读取并重采样音频。/ Load audio and resample to model sample rate."""
        wav, sr = torchaudio.load(path)
        wav = wav.to(self.device)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            wav = resampler(wav)
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        return wav, self.sample_rate

    def get_duration(self, audio: Tuple[torch.Tensor, int]) -> float:
        """返回音频时长（秒）。/ Return duration in seconds."""
        return audio[0].shape[-1] / audio[1]
    
    def get_sample_rate(self, audio: Tuple[torch.Tensor, int]) -> int:
        """返回采样率。/ Return sample rate."""
        return audio[1]

    def create_silence(self, duration: float, sample_rate: int) -> torch.Tensor:
        """生成静音 Mel。/ Create a near-silent mel segment."""
        num_samples = int(duration * sample_rate)
        silent_wav = torch.randn((1, num_samples), device=self.device) * 1e-6
        with torch.no_grad():
            silent_mel = get_mel_spectrogram(silent_wav, self.h)
        return silent_mel

    def save_audio(self, audio_mel: torch.Tensor, output_path: str, sample_rate: Optional[int] = None) -> None:
        """将 Mel 通过声码器保存为音频。/ Vocode mel and save waveform to file."""
        if not self.use_vocoder or self.model is None:
            raise RuntimeError("Vocoder is disabled (use_vocoder=False); cannot save waveform from mel.")
        with torch.no_grad():
            wav_gen = self.model(audio_mel)
        wav_cpu = wav_gen.cpu().squeeze(0)
        wav_cpu = torch.tanh(wav_cpu) # Soft clip
        torchaudio.save(output_path, wav_cpu, self.sample_rate)

    def calculate_warping_path(self, src_anchors: List[float], tgt_anchors: List[float], total_tgt_frames: int) -> torch.Tensor:
        """计算时间映射路径。/ Compute smooth source-time mapping for target frames."""
        def sec2frame(sec: float) -> float:
            return sec * self.sample_rate / self.h.hop_size
        
        src_frames = np.array([sec2frame(t) for t in src_anchors])
        tgt_frames = np.array([sec2frame(t) for t in tgt_anchors])
        
        # PCHIP keeps monotonicity and avoids time-reversal artifacts.
        interpolator = PchipInterpolator(tgt_frames, src_frames)
        
        grid_tgt = np.arange(total_tgt_frames)
        grid_src = interpolator(grid_tgt)
        
        return torch.from_numpy(grid_src).float().to(self.device)

    def warp_mel(self, source_mel: torch.Tensor, warping_path: torch.Tensor) -> torch.Tensor:
        """
        使用 grid_sample 沿时间轴扭曲 Mel。/ Warp mel on time axis with grid_sample.

        Args:
            source_mel: 输入 Mel，形状为 (1, n_mels, src_len)。/ Input mel of shape (1, n_mels, src_len).
            warping_path: 目标帧对应源帧索引，形状为 (tgt_len,)。/ Source-frame index per target frame, shape (tgt_len,).

        Returns:
            torch.Tensor: 扭曲后的 Mel，形状为 (1, n_mels, tgt_len)。/ Warped mel of shape (1, n_mels, tgt_len).
        """
        B, C, src_len = source_mel.shape
        tgt_len = warping_path.shape[0]

        norm_x = 2 * (warping_path / (src_len - 1)) - 1

        source_mel_4d = source_mel.unsqueeze(2) # (1, n_mels, 1, src_len)

        grid_x = norm_x.view(1, 1, tgt_len, 1).expand(1, 1, -1, 1) # (1, 1, tgt_len, 1)
        grid_y = torch.zeros_like(grid_x) # (1, 1, tgt_len, 1)

        grid = torch.cat([grid_x, grid_y], dim=-1)

        warped_mel = F.grid_sample(source_mel_4d, grid, mode='bicubic', padding_mode='border', align_corners=True)

        return warped_mel.squeeze(2) # (1, n_mels, tgt_len)
    
    @staticmethod
    def _load_phoneme_mapping(path: str) -> Dict[str, int]:
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"phoneme mapping json not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        mapping = payload.get("phone_mapping", payload)
        if not isinstance(mapping, dict):
            raise ValueError(f"invalid phoneme mapping format: {p}")
        return {str(k): int(v) for k, v in mapping.items()}
    
    def _reverse_phoneme_mapping(self, phoneme_sequence: torch.Tensor) -> np.ndarray:
        id_to_phone = {v: k for k, v in self.phoneme_mapping.items()}
        return np.array([id_to_phone.get(int(pid), "<unk>") for pid in phoneme_sequence])

    def length_regulate_phoneme_ids(
        self,
        source_phone_tier: tgt.IntervalTier,
        source_total_frames: int,
        warping_path: torch.Tensor,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Build frame-level phoneme IDs on warped timeline.

        1) Convert source phone intervals to source frame IDs.
        2) Sample source frame IDs by warping path to get target frame IDs.
        """
        src_ids = torch.full((source_total_frames,), int(pad_id), dtype=torch.long)

        def sec2frame(sec: float) -> int:
            return int(round(sec * self.sample_rate / self.h.hop_size))

        for interval in source_phone_tier:
            token = (interval.text or "").strip()
            if token == "":
                continue

            token_norm = token if token in self.phoneme_mapping else token.upper()
            token_id = int(self.phoneme_mapping.get(token_norm, pad_id))

            s = max(0, min(source_total_frames, sec2frame(interval.start_time)))
            e = max(s, min(source_total_frames, sec2frame(interval.end_time)))
            if e > s:
                src_ids[s:e] = token_id

        src_idx = torch.round(warping_path).long()
        src_idx = torch.clamp(src_idx, min=0, max=max(0, source_total_frames - 1))
        tgt_ids = src_ids[src_idx]
        return tgt_ids

    def _append_monotonic_anchor(
        self,
        src_anchors: List[float],
        tgt_anchors: List[float],
        src_time: float,
        tgt_time: float,
        eps: float = 1e-4,
        dup_tol: float = 1e-6,
    ) -> None:
        """追加严格单调关键点。/ Append anchor while keeping target time strictly increasing."""
        if tgt_time <= tgt_anchors[-1]:
            is_same_point = abs(tgt_time - tgt_anchors[-1]) < dup_tol and abs(src_time - src_anchors[-1]) < dup_tol
            if is_same_point:
                return
            tgt_time = tgt_anchors[-1] + eps

        tgt_anchors.append(tgt_time)
        src_anchors.append(src_time)

    def build_anchors(
        self,
        words_src: List[tgt.Interval],
        words_tgt: List[tgt.Interval],
        src_duration: float,
        tgt_duration: float,
        eps: float = 1e-4,
    ) -> Tuple[List[float], List[float]]:
        """构建单调关键点。/ Build monotonic source/target anchors from aligned words."""
        src_anchors = [0.0]
        tgt_anchors = [0.0]

        for w_src, w_tgt in zip(words_src, words_tgt):
            self._append_monotonic_anchor(src_anchors, tgt_anchors, w_src.start_time, w_tgt.start_time, eps=eps)
            self._append_monotonic_anchor(src_anchors, tgt_anchors, w_src.end_time, w_tgt.end_time, eps=eps)

        if tgt_anchors[-1] < tgt_duration - eps:
            tgt_anchors.append(tgt_duration)
            src_anchors.append(src_duration)

        return src_anchors, tgt_anchors


    def transform(
        self,
        source_audio_path: str,
        source_textgrid_path: str,
        target_textgrid_path: str,
        target_audio_path: str,
        output_path: str,
        tier_name: str = "words",
    ) -> bool:
        """
        执行全局扭曲并输出音频。/ Run global warp transform and save output audio.

        Args:
            source_audio_path: 源音频路径。/ Path to source audio.
            source_textgrid_path: 源对齐 TextGrid 路径。/ Path to source TextGrid.
            target_textgrid_path: 目标对齐 TextGrid 路径。/ Path to target TextGrid.
            target_audio_path: 目标音频路径（用于总时长）。/ Path to target audio used for target duration.
            output_path: 输出音频路径。/ Output audio path.
            tier_name: 使用的 TextGrid 层名。/ Tier name used in TextGrid.

        Returns:
            bool: 成功返回 True，词数不匹配返回 False。/ True on success, False when word counts mismatch.
        """
        if self.verbose: print(f"Processing Global Warp...")

        wav_src, sr = self.load_audio(source_audio_path)
        with torch.no_grad():
            mel_src = get_mel_spectrogram(wav_src, self.h)

        tg_src = tgt.io.read_textgrid(source_textgrid_path)
        tg_tgt = tgt.io.read_textgrid(target_textgrid_path)
        words_src = self.get_real_words(tg_src.get_tier_by_name(tier_name))
        words_tgt = self.get_real_words(tg_tgt.get_tier_by_name(tier_name))

        if len(words_src) != len(words_tgt):
            print(f"❌ Word count mismatch: {len(words_src)} in source vs {len(words_tgt)} in target. Double check.")

        wav_tgt_raw, sr_tgt_raw = torchaudio.load(target_audio_path)
        tgt_duration = wav_tgt_raw.shape[-1] / sr_tgt_raw
        src_duration = wav_src.shape[-1] / sr

        src_anchors, tgt_anchors = self.build_anchors(words_src, words_tgt, src_duration, tgt_duration)

        total_target_frames = int(tgt_duration * self.sample_rate / self.h.hop_size)
        warping_path = self.calculate_warping_path(src_anchors, tgt_anchors, total_target_frames)
        final_mel = self.warp_mel(mel_src, warping_path)
        self.save_audio(final_mel, output_path)

        if self.verbose: print(f"✅ Saved: {output_path}")
        return True
    
    
    def transform_mel_with_path(
        self,
        source_mel: torch.Tensor,
        source_textgrid: Union[str, Path, tgt.TextGrid],
        target_textgrid: Union[str, Path, tgt.TextGrid],
        tier_name: str = "phones",
        pad_id: int = 0,
    ):
        """
        Warp mel using source/target TextGrid alignment.

        Args:
            source_mel (torch.Tensor): Input mel of shape (1, n_mels, src_len).
            source_textgrid (str | Path | tgt.TextGrid): Source TextGrid path or object.
            target_textgrid (str | Path | tgt.TextGrid): Target TextGrid path or object.
            pad_id (int): Fallback phone id for unknown/blank phones.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
                - (warped mel, frame-level phoneme ids).
        """
        def _as_textgrid(x: Union[str, Path, tgt.TextGrid]) -> tgt.TextGrid:
            if isinstance(x, tgt.TextGrid):
                return x
            return tgt.io.read_textgrid(str(x))

        def _duration_from_tier(tier: tgt.IntervalTier) -> float:
            if len(tier) == 0:
                return 0.0
            return max(iv.end_time for iv in tier)

        tg_src = _as_textgrid(source_textgrid)
        tg_tgt = _as_textgrid(target_textgrid)

        tier_src = tg_src.get_tier_by_name(tier_name)
        tier_tgt = tg_tgt.get_tier_by_name(tier_name)

        words_src = self.get_real_words(tier_src)
        words_tgt = self.get_real_words(tier_tgt)

        src_duration = _duration_from_tier(tier_src)
        tgt_duration = _duration_from_tier(tier_tgt)

        src_anchors, tgt_anchors = self.build_anchors(words_src, words_tgt, src_duration, tgt_duration)

        total_target_frames = max(1, int(tgt_duration * self.sample_rate / self.h.hop_size))
        warping_path = self.calculate_warping_path(src_anchors, tgt_anchors, total_target_frames)
        final_mel = self.warp_mel(source_mel, warping_path)

        phone_tier = tg_src.get_tier_by_name(tier_name)
        phoneme_ids = self.length_regulate_phoneme_ids(
            source_phone_tier=phone_tier,
            source_total_frames=source_mel.shape[-1],
            warping_path=warping_path,
            pad_id=pad_id,
        )
        return final_mel, phoneme_ids
    