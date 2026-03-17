import csv
import os
import re
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

import pandas as pd
import tgt
import torch
import torchaudio
from torch.utils.data import Dataset

warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
    module=r"torchaudio\._backend\.ffmpeg",
)
warnings.filterwarnings(
    "ignore",
    message=r".*implementation will be changed to use torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
    module=r"torchaudio\._backend\.utils",
)

from modules.mel_strech.meldataset import get_mel_spectrogram
from modules.mel_strech.mel_transform import GlobalWarpTransformer

from data_provider.collate_funcs import collate_cfm_phase1, collate_cfm_index_phase1  # noqa: F401
from data_provider.utils import CFMIndexCacheBuilder, CFMIndexLipsInferCacheBuilder  # noqa: F401

from logger import get_logger

logger = get_logger("dubbing.data_loader")


@dataclass
class PairSample:
    pair_key: str
    r1_audio: Path
    r2_audio: Path
    r1_tg: Path
    r2_tg: Path


def _pair_key_and_role(stem: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"^(.*?)(?:[_-]?r([12]))$", stem, flags=re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), f"r{m.group(2)}"


class Dataset_CFM_Phase1(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        split_ratio: float = 0.95,
        seed: int = 2026,
        filter_enabled: bool = True,
        mse_threshold: float = 0.08,
        tier_name: str = "phones",
        phoneme_map_path: str = "dubbing/modules/english_us_arpa_300.json",
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.ost_dirs = self._discover_ost_dirs(self.root_dir)
        self.split = split
        self.tier_name = tier_name
        self.filter_enabled = filter_enabled
        self.mse_threshold = mse_threshold
        self.device = torch.device("cpu")

        self._warper = GlobalWarpTransformer(
            use_vocoder=False,
            device=str(self.device),
            verbose=False,
        )
        self.mel_h = SimpleNamespace(**vars(self._warper.h))
        self.sample_rate = int(self.mel_h.sampling_rate)
        self.pad_phoneme_id = int(self._warper.phoneme_mapping.get("<eps>", 0))

        all_pairs = self._discover_pairs()
        all_pairs = self._apply_filter(all_pairs)
        self.samples = self._split_samples(all_pairs, split, split_ratio, seed)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} in {root_dir}")

    @staticmethod
    def _discover_ost_dirs(root_dir: Path) -> List[Path]:
        ost_dirs = sorted([p for p in root_dir.rglob("ost") if p.is_dir()])
        if not ost_dirs:
            raise FileNotFoundError(f"No 'ost' directories found under: {root_dir}")
        return ost_dirs

    def _resolve_aligned_dir(self, ost_dir: Path) -> Optional[Path]:
        candidates = [
            ost_dir.parent / "aligned",
            self.root_dir / "aligned",
        ]
        for aligned_dir in candidates:
            if aligned_dir.exists():
                return aligned_dir
        return None

    def _discover_pairs(self) -> List[PairSample]:
        samples: List[PairSample] = []

        for ost_dir in self.ost_dirs:
            aligned_dir = self._resolve_aligned_dir(ost_dir)
            if aligned_dir is None:
                logger.warning(f"Skip ost dir without aligned dir: {ost_dir}")
                continue

            pair_map: Dict[str, Dict[str, Path]] = {}
            for wav_path in ost_dir.rglob("*.wav"):
                key, role = _pair_key_and_role(wav_path.stem)
                if key is None or role is None:
                    continue
                if key not in pair_map:
                    pair_map[key] = {}
                pair_map[key][role] = wav_path

            for key, pair in pair_map.items():
                if "r1" not in pair or "r2" not in pair:
                    continue
                r1_wav = pair["r1"]
                r2_wav = pair["r2"]

                rel_r1 = r1_wav.relative_to(ost_dir)
                rel_r2 = r2_wav.relative_to(ost_dir)

                r1_tg = (aligned_dir / rel_r1).with_suffix(".TextGrid")
                r2_tg = (aligned_dir / rel_r2).with_suffix(".TextGrid")
                if not r1_tg.exists() or not r2_tg.exists():
                    continue

                samples.append(PairSample(key, r1_wav, r2_wav, r1_tg, r2_tg))

        samples.sort(key=lambda x: x.pair_key)
        return samples


    def _load_filter_table(self) -> Optional[pd.DataFrame]:
        if not self.filter_enabled:
            return None

        frames: List[pd.DataFrame] = []
        for ost_dir in self.ost_dirs:
            aligned_dir = self._resolve_aligned_dir(ost_dir)
            if aligned_dir is None:
                continue
            csv_path = aligned_dir / "alignment_analysis.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if "phone_duration_deviation" in df.columns:
                frames.append(df)

        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def _apply_filter(self, samples: List[PairSample]) -> List[PairSample]:
        df = self._load_filter_table()
        if df is None:
            return samples

        if "pair_key" in df.columns and "phone_duration_deviation" in df.columns:
            allowed_keys = set(
                df.loc[df["phone_duration_deviation"] <= self.mse_threshold, "pair_key"]
                .astype(str)
                .tolist()
            )
            before = len(samples)
            filtered = [s for s in samples if s.pair_key in allowed_keys]
            after = len(filtered)
            logger.info(
                f"Filter (phone_duration_deviation <= {self.mse_threshold}): "
                f"{before} -> {after} samples "
                f"(removed {before - after}, kept {after / before * 100:.1f}%)"
            )
            return filtered

        return samples

    @staticmethod
    def _split_samples(samples: List[PairSample], split: str, split_ratio: float, seed: int) -> List[PairSample]:
        if split not in {"train", "val", "test"}:
            return samples

        n_total = len(samples)
        if n_total == 0:
            return samples

        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=rng).tolist()
        samples = [samples[i] for i in perm]

        train_count = int(n_total * split_ratio)
        train_count = max(1, min(train_count, n_total))

        remaining = n_total - train_count
        val_test_count = remaining // 2

        if val_test_count == 0:
            if n_total >= 3:
                val_test_count = 1
            else:
                if split == "train":
                    return samples
                return samples[-1:]

        train_end = n_total - 2 * val_test_count
        val_end = train_end + val_test_count

        if split == "train":
            return samples[:train_end]
        if split == "val":
            return samples[train_end:val_end]
        return samples[val_end:]

    def _load_wav(self, wav_path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(wav_path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def _wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mel = get_mel_spectrogram(wav, self.mel_h)
        return mel

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _extract_text(tg_path: Path, tier_name: str) -> str:
        """Read word-level text from a TextGrid tier."""
        try:
            tg = tgt.io.read_textgrid(str(tg_path))
            tier = tg.get_tier_by_name(tier_name)
            words = [iv.text for iv in tier if iv.text.strip() not in ("", "sp", "sil", "<eps>")]
            return " ".join(words)
        except Exception:
            return ""

    def __getitem__(self, index: int):
        sample = self.samples[index]

        r1_wav = self._load_wav(sample.r1_audio)
        r2_wav = self._load_wav(sample.r2_audio)

        r1_mel = self._wav_to_mel(r1_wav)
        r2_mel = self._wav_to_mel(r2_wav)

        stretched_r1_mel, phoneme_ids = self._warper.transform_mel_with_path(
            source_mel=r1_mel,
            source_textgrid=sample.r1_tg,
            target_textgrid=sample.r2_tg,
            tier_name=self.tier_name,
        )

        target_frames = min(stretched_r1_mel.shape[-1], r2_mel.shape[-1])
        stretched_r1_mel = stretched_r1_mel[:, :, :target_frames]
        r2_mel = r2_mel[:, :, :target_frames]
        phoneme_ids = phoneme_ids[:target_frames]

        # Normalize x0 and x1 with the SAME scale derived from x0,
        # so the model learns in a stable space and we can invert at inference.
        cond_mel = stretched_r1_mel.squeeze(0)   # [n_mels, T]
        x1 = r2_mel.squeeze(0)                     # [n_mels, T]
        x_mean = cond_mel.mean()
        x_std  = cond_mel.std().clamp(min=1e-5)
        cond_mel = (cond_mel - x_mean) / x_std
        x1 = (x1 - x_mean) / x_std

        mse = float(torch.mean((cond_mel - x1) ** 2).item())

        text_r1 = self._extract_text(sample.r1_tg, "words")

        # 每个样本字段含义：
        # pair_key: 配对样本唯一标识（同一语句的 r1/r2）
        # cond_mel: 条件 mel（由 r1 拉伸到 r2 时序后并按 r1 统计量归一化）[n_mels, T]
        # x1: 训练目标 mel（r2 mel，使用 cond_mel 的均值/方差归一化）[n_mels, T]
        # phoneme_ids: 与时间帧对齐的音素 id 序列 [T]
        # x_len: 有效帧长 T（long）
        # x_mean/x_std: 归一化统计量（推理反归一化会用到）
        # mse: cond_mel 与 x1 的逐帧均方误差（样本质量参考）
        # text_r1: r1 的 words 层文本（可用于日志/可视化）
        return {
            "pair_key": sample.pair_key,
            "cond_mel": cond_mel,
            "x1": x1,
            "phoneme_ids": phoneme_ids,
            "x_len": torch.tensor(target_frames, dtype=torch.long),
            "x_mean": x_mean,
            "x_std": x_std,
            "mse": mse,
            "text_r1": text_r1,
        }
        
class Dataset_CFM_Phase1_StretchEntireMel(Dataset_CFM_Phase1):
    """Variant that stretches source mel to target length via bicubic interpolation
    instead of TextGrid-guided warping. Phoneme IDs are read directly from the
    target TextGrid. Everything else (normalization, filtering, splits) is identical.
    """

    def __getitem__(self, index: int):
        import torch.nn.functional as F

        sample = self.samples[index]

        r1_wav = self._load_wav(sample.r1_audio)
        r2_wav = self._load_wav(sample.r2_audio)

        r1_mel = self._wav_to_mel(r1_wav)  # (1, n_mels, T_src)
        r2_mel = self._wav_to_mel(r2_wav)  # (1, n_mels, T_tgt)

        target_frames = r2_mel.shape[-1]

        # Bicubic resize along the time axis only.
        # F.interpolate with mode='bicubic' requires 4-D input.
        cond_mel_raw = F.interpolate(
            r1_mel.unsqueeze(0),                          # (1, 1, n_mels, T_src)
            size=(r1_mel.shape[1], target_frames),
            mode='bicubic',
            align_corners=True,
        ).squeeze(0)                                       # (1, n_mels, target_frames)

        # Phoneme IDs directly from target TextGrid.
        tg_tgt = tgt.io.read_textgrid(str(sample.r2_tg))
        phone_tier_tgt = tg_tgt.get_tier_by_name(self.tier_name)
        phoneme_ids = self._warper.length_regulate_phoneme_ids(
            target_phone_tier=phone_tier_tgt,
            total_tgt_frames=target_frames,
            pad_id=self.pad_phoneme_id,
        )

        cond_mel = cond_mel_raw.squeeze(0)  # (n_mels, target_frames)
        x1       = r2_mel.squeeze(0)        # (n_mels, target_frames)

        x_mean = cond_mel.mean()
        x_std  = cond_mel.std().clamp(min=1e-5)
        cond_mel = (cond_mel - x_mean) / x_std
        x1       = (x1       - x_mean) / x_std

        mse = float(torch.mean((cond_mel - x1) ** 2).item())

        text_r1 = self._extract_text(sample.r1_tg, "words")

        # 每个样本字段含义与 Dataset_CFM_Phase1 保持一致；
        # 区别仅在 cond_mel 的构造方式：这里是对 r1_mel 做整段双三次插值到目标长度。
        return {
            "pair_key": sample.pair_key,
            "cond_mel": cond_mel,
            "x1": x1,
            "phoneme_ids": phoneme_ids,
            "x_len": torch.tensor(target_frames, dtype=torch.long),
            "x_mean": x_mean,
            "x_std": x_std,
            "mse": mse,
            "text_r1": text_r1,
        }


# =============================================================================
# CFMIndexCacheBuilder  –  see data_provider/utils.py
# =============================================================================
# (imported above; re-exported so existing code that does
#  ``from data_provider.data_loader import CFMIndexCacheBuilder`` keeps working)


# =============================================================================
# Dataset_CFM_Index_Phase1
# =============================================================================


class Dataset_CFM_Index_Phase1(Dataset):
    """Dataset for training IndexTTS2-CFM (Phase 1) with condition caching.

    Reads a ``metadata.csv`` file with columns:
      - ``prompt_audio_path`` : reference speaker audio
      - ``out_pt``            : pre-computed S_infer .pt file  [1?, T_codes, 1024]
      - ``out_wav``           : ground-truth target audio
      - ``gen_error``         : non-empty rows are skipped

    On first instantiation (or when cache files are missing) a
    ``CFMIndexCacheBuilder`` is created to compute and persist per-sample
    conditions.  Subsequent runs load directly from cache.

    Each ``__getitem__`` returns CPU tensors:
      - ``stem``        : str
      - ``ref_mel``     : [num_mels, T_ref]
      - ``style``       : [192]
      - ``prompt_cond`` : [T_ref, 512]
      - ``infer_cond``  : [T_gen, 512]
      - ``x1_mel``      : [num_mels, T_gen]
    """

    def __init__(
        self,
        csv_path: str,
        mel_h,
        preprocess,
        sr_ref_16k: int = 16000,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 2026,
        max_ref_sec: float = 15.0,
        max_gen_sec: float = 10.0,
        max_code_len: int = 500,
        cache_dir: Optional[str] = None,
        cache_batch_size: int = 16,
    ):
        super().__init__()
        self.csv_path         = Path(csv_path)
        self.data_root        = self.csv_path.parent
        self.split            = split
        self.mel_h            = mel_h
        self.preprocess       = preprocess
        self.sr_22k           = int(mel_h.sampling_rate)
        self.sr_ref_16k       = int(sr_ref_16k)
        self.max_ref_sec      = max_ref_sec
        self.max_gen_sec      = max_gen_sec
        self.max_code_len     = max_code_len
        self.cache_batch_size = cache_batch_size
        self.cache_dir        = (
            Path(cache_dir) if cache_dir
            else self.data_root / "cfm_index_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        all_samples = self._load_csv()
        self.samples = self._split_samples(all_samples, split, split_ratio, seed)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples for split={split!r} in {csv_path}"
            )
        logger.info(
            f"Dataset_CFM_Index_Phase1 [{split}]: {len(self.samples)} samples"
        )
        self._ensure_cache()

    # ------------------------------------------------------------------
    # CSV loading / splitting
    # ------------------------------------------------------------------

    def _load_csv(self) -> List[Dict]:
        samples: List[Dict] = []
        seen_stems: set = set()
        with open(self.csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("gen_error", "").strip():
                    continue
                out_pt  = row.get("out_pt",  "").strip()
                out_wav = row.get("out_wav", "").strip()
                prompt  = row.get("prompt_audio_path", "").strip()
                if not (out_pt and out_wav and prompt):
                    continue
                if out_pt and not Path(out_pt).is_absolute():
                    out_pt = str((self.data_root / out_pt).resolve())
                if out_wav and not Path(out_wav).is_absolute():
                    out_wav = str((self.data_root / out_wav).resolve())
                if not Path(out_pt).exists() or not Path(out_wav).exists():
                    continue
                stem = Path(out_pt).stem
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                samples.append({
                    "stem":              stem,
                    "prompt_audio_path": prompt,
                    "out_pt":            out_pt,
                    "out_wav":           out_wav,
                })
        return samples

    @staticmethod
    def _split_samples(
        samples: List[Dict], split: str, split_ratio: float, seed: int
    ) -> List[Dict]:
        n = len(samples)
        if n == 0:
            return samples
        rng     = torch.Generator().manual_seed(seed)
        perm    = torch.randperm(n, generator=rng).tolist()
        samples = [samples[i] for i in perm]
        train_n   = max(1, min(int(n * split_ratio), n))
        remaining = n - train_n
        vt_n      = remaining // 2
        if vt_n == 0:
            vt_n = 1 if n >= 3 else 0
        train_end = n - 2 * vt_n
        val_end   = train_end + vt_n
        if split == "train":
            return samples[:train_end]
        if split == "val":
            return samples[train_end:val_end]
        return samples[val_end:]

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _ensure_cache(self):
        missing = [
            s for s in self.samples
            if not (self.cache_dir / f"{s['stem']}.pt").exists()
        ]
        if not missing:
            logger.info(
                f"[Cache] All {len(self.samples)} samples already cached"
            )
            return
        logger.info(
            f"[Cache] Building {len(missing)}/{len(self.samples)} missing "
            f"entries → {self.cache_dir}"
        )
        CFMIndexCacheBuilder(
            preprocess=self.preprocess,
            mel_h=self.mel_h,
            cache_dir=self.cache_dir,
            sr_ref_16k=self.sr_ref_16k,
            max_ref_sec=self.max_ref_sec,
            max_gen_sec=self.max_gen_sec,
            max_code_len=self.max_code_len,
            batch_size=self.cache_batch_size,
        ).build(missing)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        data = torch.load(
            self.cache_dir / f"{item['stem']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        # 缓存样本字段含义：
        # ref_mel: 参考音频 mel [num_mels, T_ref]
        # style: 说话人/风格向量 [192]
        # prompt_cond: 参考端语义条件 [T_ref, 512]
        # infer_cond: 目标端语义条件 [T_gen, 512]
        # x1_mel: 目标音频 mel（训练目标）[num_mels, T_gen]
        # stem: 样本唯一 stem（由当前 metadata 行补充）
        data["stem"] = item["stem"]
        return data


class Dataset_CFM_Index_Phase1_ForLipsFeat(Dataset):
    """针对 lips 特征训练的独立 Dataset。

    设计目标
    --------
    1. 保持与 ``Dataset_CFM_Index_Phase1`` 相近的使用方式：
       - 仍使用 ``metadata.csv``
       - 仍在首次实例化时构建/读取缓存
    2. 缓存构建阶段直接完成 lips 相关增强：
       - 加载 ``source_textgrid``（MELD_semantic/audios/aligned）
       - 加载 ``lips_textgrid``（MELD_predict_results/textgrid）
       - 以 source/target TextGrid 对 ``S_infer`` 做 semantic stretch
       - stretch 前对 TextGrid 音素统一做 ARPA→VFA→ARPAbet 规范化
       - 用 stretch 后的 ``S_infer`` 重新计算 ``infer_cond``
       - 用新的 ``infer_cond`` 生成并缓存 ``x1_mel``
       - 加载并返回 ``lips_hidden_states``（MELD_predict_results/hidden_states）

    样本规则
    --------
    只保留交集样本：metadata 与 source_textgrid / lips_textgrid /
    lips_hidden_states 同时存在的样本才会进入 ``self.samples``。

    返回字段
    --------
    与 ``Dataset_CFM_Index_Phase1`` 相比，新增：
        - ``lips_hidden_states``
        - ``lips_textgrid``
        - ``source_textgrid``
    """

    def __init__(
        self,
        flow_dataset_path: str,
        mel_h,
        preprocess,
        sr_ref_16k: int = 16000,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 2026,
        max_ref_sec: float = 15.0,
        max_gen_sec: float = 10.0,
        max_code_len: int = 500,
        cache_dir: Optional[str] = None,
        cache_batch_size: int = 16,
        tier_name: str = "phones",
        warp_type: str = "cond",  # "semantic" or "cond" — matches SemanticTransformer.input_type
        grid_sample_mode: str = "bilinear",  # for semantic stretch; "nearest" or "bilinear"
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.flow_dataset_path = Path(flow_dataset_path)
        self.tier_name = tier_name
        self.split = split
        self.mel_h = mel_h
        self.preprocess = preprocess
        self.sr_22k = int(mel_h.sampling_rate)
        self.sr_ref_16k = int(sr_ref_16k)
        self.max_ref_sec = max_ref_sec
        self.max_gen_sec = max_gen_sec
        self.max_code_len = max_code_len
        self.cache_batch_size = cache_batch_size
        self.warp_type = warp_type
        self.grid_sample_mode = grid_sample_mode

        self.semantic_root = self.flow_dataset_path / "semantic"
        self.predict_root = self.flow_dataset_path / "predict_results"

        self.source_tg_dir = self.semantic_root / "audios" / "aligned"
        self.lips_tg_dir = self.predict_root / "textgrids"
        self.lips_hs_dir = self.predict_root / "hidden_states"
        self.csv_path_auto = self.semantic_root / "metadata.csv"
        self.csv_path = self.csv_path_auto
        self.data_root = self.csv_path.parent
        self.cache_dir = (
            Path(cache_dir) if cache_dir
            else self.data_root / f"cfm_index_lipsfeat_cache_{self.warp_type[:4]}_{self.grid_sample_mode[:4]}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.csv_path_auto.exists():
            raise FileNotFoundError(f"metadata.csv not found: {self.csv_path_auto}")
        if not self.source_tg_dir.exists():
            raise FileNotFoundError(f"source textgrid dir not found: {self.source_tg_dir}")
        if not self.lips_tg_dir.exists():
            raise FileNotFoundError(f"lips textgrid dir not found: {self.lips_tg_dir}")
        if not self.lips_hs_dir.exists():
            raise FileNotFoundError(f"lips hidden_states dir not found: {self.lips_hs_dir}")

        all_samples = self._load_csv()
        self.samples = self._split_samples(all_samples, split, split_ratio, seed)
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid intersected samples for split={split!r} in {self.csv_path_auto}"
            )

        logger.info(
            f"Dataset_CFM_Index_Phase1_ForLipsFeat [{split}] before cache: {len(self.samples)} samples"
        )
        self._ensure_cache()

        # Some samples may still fail during cache building (e.g. malformed
        # alignment files). Keep only samples with cache files to ensure
        # dataloader stability.
        before = len(self.samples)
        self.samples = [
            s for s in self.samples
            if (self.cache_dir / f"{s['stem']}.pt").exists()
        ]
        if len(self.samples) != before:
            logger.warning(
                f"[LipsFeat] Drop samples without lips cache: {before} -> {len(self.samples)}"
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid intersected samples for split={split!r} in {self.csv_path_auto}"
            )
        logger.info(
            f"Dataset_CFM_Index_Phase1_ForLipsFeat [{split}]: {len(self.samples)} samples"
        )

    @staticmethod
    def _split_samples(
        samples: List[Dict], split: str, split_ratio: float, seed: int
    ) -> List[Dict]:
        n = len(samples)
        if n == 0:
            return samples
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=rng).tolist()
        samples = [samples[i] for i in perm]
        train_n = max(1, min(int(n * split_ratio), n))
        remaining = n - train_n
        vt_n = remaining // 2
        if vt_n == 0:
            vt_n = 1 if n >= 3 else 0
        train_end = n - 2 * vt_n
        val_end = train_end + vt_n
        if split == "train":
            return samples[:train_end]
        if split == "val":
            return samples[train_end:val_end]
        return samples[val_end:]

    def _ensure_cache(self):
        missing = [
            s for s in self.samples
            if not (self.cache_dir / f"{s['stem']}.pt").exists()
        ]
        if not missing:
            logger.info(
                f"[LipsCache] All {len(self.samples)} samples already cached"
            )
            return
        logger.info(
            f"[LipsCache] Building {len(missing)}/{len(self.samples)} missing entries -> {self.cache_dir}"
        )
        CFMIndexLipsInferCacheBuilder(
            preprocess=self.preprocess,
            mel_h=self.mel_h,
            cache_dir=self.cache_dir,
            sr_ref_16k=self.sr_ref_16k,
            max_ref_sec=self.max_ref_sec,
            max_gen_sec=self.max_gen_sec,
            max_code_len=self.max_code_len,
            batch_size=self.cache_batch_size,
            tier_name=self.tier_name,
            warp_type=self.warp_type,
            grid_sample_mode=self.grid_sample_mode,
        ).build(missing)

    def _load_csv(self) -> List[Dict]:
        samples: List[Dict] = []
        seen_stems: set = set()
        with open(self.csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("gen_error", "").strip():
                    continue
                out_pt = row.get("out_pt", "").strip()
                out_wav = row.get("out_wav", "").strip()
                prompt = row.get("prompt_audio_path", "").strip()
                if not (out_pt and out_wav and prompt):
                    continue
                if out_pt and not Path(out_pt).is_absolute():
                    out_pt = str((self.data_root / out_pt).resolve())
                if out_wav and not Path(out_wav).is_absolute():
                    out_wav = str((self.data_root / out_wav).resolve())
                if prompt and not Path(prompt).is_absolute():
                    prompt = str((self.data_root / prompt).resolve())
                if not Path(out_pt).exists() or not Path(out_wav).exists() or not Path(prompt).exists():
                    continue
                stem = Path(out_pt).stem
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                samples.append({
                    "stem": stem,
                    "prompt_audio_path": prompt,
                    "out_pt": out_pt,
                    "out_wav": out_wav,
                })

        merged = []
        for s in samples:
            stem = s["stem"]

            source_tg = None
            lips_tg = None
            lips_hs = None

            for ext in (".TextGrid", ".textgrid"):
                p = self.source_tg_dir / f"{stem}{ext}"
                if p.exists():
                    source_tg = p
                    break
            for ext in (".TextGrid", ".textgrid"):
                p = self.lips_tg_dir / f"{stem}{ext}"
                if p.exists():
                    lips_tg = p
                    break
            for ext in (".pt", ".pth", ".npy"):
                p = self.lips_hs_dir / f"{stem}{ext}"
                if p.exists():
                    lips_hs = p
                    break

            if source_tg is None or lips_tg is None or lips_hs is None:
                continue

            s2 = dict(s)
            s2["source_textgrid"] = str(source_tg)
            s2["lips_textgrid"] = str(lips_tg)
            s2["lips_hidden_states"] = str(lips_hs)
            merged.append(s2)

        logger.info(f"[LipsFeat] Intersection filter: {len(samples)} -> {len(merged)} samples")
        return merged

    def __len__(self) -> int:
        return len(self.samples)

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]

        cached = torch.load(
            self.cache_dir / f"{item['stem']}.pt",
            map_location="cpu",
            weights_only=True,
        )

        hs_path = Path(item["lips_hidden_states"])
        if hs_path.suffix.lower() == ".npy":
            import numpy as np

            lips_hs = torch.from_numpy(np.load(str(hs_path)))
        else:
            lips_hs = torch.load(str(hs_path), map_location="cpu", weights_only=False)
        if isinstance(lips_hs, dict):
            for k in ("hidden_states", "lips_hidden_states", "features", "x"):
                if k in lips_hs:
                    lips_hs = lips_hs[k]
                    break
        if not torch.is_tensor(lips_hs):
            lips_hs = torch.as_tensor(lips_hs)
        if lips_hs.dim() == 3 and lips_hs.size(0) == 1:
            lips_hs = lips_hs.squeeze(0)
        lips_hs = lips_hs.float().cpu()

        # 每个样本字段含义：
        # stem/out_wav: 样本标识与目标 wav 路径（调试/导出用）
        # ref_mel/style/prompt_cond/infer_cond/x1_mel: 与 Index Phase1 一致的缓存条件与目标
        # lips_hidden_states: 嘴型模型特征（通常为 [T_lips, C]）
        # lips_textgrid/source_textgrid: lips 与 source 对齐文件路径（便于复核时序映射）
        return {
            "stem": item["stem"],
            "out_wav": item["out_wav"],
            "ref_mel": cached["ref_mel"],
            "style": cached["style"],
            "prompt_cond": cached["prompt_cond"],
            "infer_cond": cached["infer_cond"],
            "x1_mel": cached["x1_mel"],
            "lips_hidden_states": lips_hs,
            "lips_textgrid": item["lips_textgrid"],
            "source_textgrid": item["source_textgrid"],
        }
        

# =============================================================================
# Dataset_LipsCFM_Phase2
# =============================================================================
"""Dataset_LipsCFM_Phase2 docstring

每一项：
- `stem`: 样本唯一标识（通常来自 wav/pt 文件名）
- `bf_stretched_mel`: 由 原始semantic token直接输出得来的wav，作为调试使用

Flow Part:
- `ref_mel`: 参考音频的 mel（通常是 22kHz）[num_mels, T_ref]
- `style`: 说话人/风格向量 [192]
- `prompt_cond`: 参考端语义条件 [T_ref, 512]
- `infer_cond`: 目标端语义条件 [T_gen, 512]. 
    这里的 infer_cond 应该是没有拉伸的，即直接从原始semantic token生成的条件
    ，后续会在模型里做动态拉伸。
- `x1_mel`: 目标音频的 mel（训练目标）[num_mels, T_gen]. 
    这里需要直接使用 raw dataset 中的原始音频
- `source_textgrid`: 原始对齐文件路径（便于复核时序映射）

Lips Net Part: 
    需要的信息有:
    - lips_roi
    - extracted_sampels
- `img_sequences`: 嘴型图像序列(通常为 [T_img, C, H, W])
- `frame_phoneme_label`: 每帧的phoneme id 标签 (字符串 )
- `frame_phoneme_label_id`: 每帧的phoneme id 标签（数值 id)
- `phoneme_label_id`: 音素序列标签(数值 id)
- `phoneme_label_id_with_sil`: 音素序列标签（包含 sil/sp 等停顿符）
"""
class Dataset_LipsCFM_Phase2(Dataset):
    """Dataset for training LipsCFM Phase 2 with condition caching.
    
    Reads from these following sources:
    - `raw_dataset_path` (str): 
        The root directory of the raw dataset, 
        which contains the original audio files and their corresponding TextGrid alignments.
        
        This should contain the following structure:
        ```
        raw_dataset_path/
            audios/
                aligned/
                ost/
                vocals/
                ins/
            videos/
            metadata.csv
        ```
        
    - `flow_dataset_path` (str):
        The root directory of the flow dataset,
        which contains the semantic tokens, and the metadata.csv that links them.
        
        Includes a ``metadata.csv`` file with columns:
        - ``prompt_audio_path`` : reference speaker audio
        - ``out_pt``            : pre-computed S_infer .pt file  [1?, T_codes, 1024]
        - ``out_wav``           : ground-truth target audio
        - ``gen_error``         : non-empty rows are skipped
        
    - `lips_dataset_path` (str):
        The root directory of the lips dataset,
        which contains the data that lips training has used.

        With following structrue:
        ```
        lips/
            lips_roi/
            *.csv
        ```
        A csv file in `lips/` should contain columns:
        - `sample_id`
    
    
    """
    def __init__(
        self,
        flow_dataset_path: str,
        mel_h,
        preprocess,
        sr_ref_16k: int = 16000,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 2026,
        max_ref_sec: float = 15.0,
        max_gen_sec: float = 10.0,
        max_code_len: int = 500,
        cache_dir: Optional[str] = None,
        cache_batch_size: int = 16,
        tier_name: str = "phones",
    ):
        super().__init__()
        self.flow_dataset_path = Path(flow_dataset_path)
        self.tier_name = tier_name