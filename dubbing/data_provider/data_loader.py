import os
import re
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

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

    def _build_warping_path(self, r1_tg: Path, r2_tg: Path, r1_wav: torch.Tensor, r2_wav: torch.Tensor) -> torch.Tensor:
        tg_src = tgt.io.read_textgrid(str(r1_tg))
        tg_tgt = tgt.io.read_textgrid(str(r2_tg))

        words_src = self._warper.get_real_words(tg_src.get_tier_by_name(self.tier_name))
        words_tgt = self._warper.get_real_words(tg_tgt.get_tier_by_name(self.tier_name))

        src_duration = r1_wav.shape[-1] / self.sample_rate
        tgt_duration = r2_wav.shape[-1] / self.sample_rate

        if len(words_src) > 0 and len(words_src) == len(words_tgt):
            src_anchors, tgt_anchors = self._warper.build_anchors(words_src, words_tgt, src_duration, tgt_duration)
        else:
            src_anchors, tgt_anchors = [0.0, src_duration], [0.0, tgt_duration]

        total_target_frames = max(1, int(tgt_duration * self.sample_rate / self.mel_h.hop_size))

        warping_path = self._warper.calculate_warping_path(src_anchors, tgt_anchors, total_target_frames)
        return warping_path

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
        x0 = stretched_r1_mel.squeeze(0)   # [n_mels, T]
        x1 = r2_mel.squeeze(0)             # [n_mels, T]
        x_mean = x0.mean()
        x_std  = x0.std().clamp(min=1e-5)
        x0 = (x0 - x_mean) / x_std
        x1 = (x1 - x_mean) / x_std

        mse = float(torch.mean((x0 - x1) ** 2).item())

        text_r1 = self._extract_text(sample.r1_tg, "words")

        return {
            "pair_key": sample.pair_key,
            "x0": x0,
            "x1": x1,
            "phoneme_ids": phoneme_ids,
            "x_len": torch.tensor(target_frames, dtype=torch.long),
            "x_mean": x_mean,
            "x_std": x_std,
            "mse": mse,
            "text_r1": text_r1,
        }


def collate_cfm_phase1(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.stack([item["x_len"] for item in batch], dim=0)
    max_len = int(lengths.max().item())
    n_mels = batch[0]["x0"].shape[0]

    x0 = torch.zeros(len(batch), n_mels, max_len, dtype=batch[0]["x0"].dtype)
    x1 = torch.zeros(len(batch), n_mels, max_len, dtype=batch[0]["x1"].dtype)
    phoneme_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    x_mean = torch.stack([item["x_mean"] for item in batch], dim=0)  # [B]
    x_std  = torch.stack([item["x_std"]  for item in batch], dim=0)  # [B]

    pair_keys = []
    mse_list = []
    text_r1_list = []
    text_r2_list = []
    for i, item in enumerate(batch):
        t = int(item["x_len"].item())
        x0[i, :, :t] = item["x0"][:, :t]
        x1[i, :, :t] = item["x1"][:, :t]
        phoneme_ids[i, :t] = item["phoneme_ids"][:t]
        pair_keys.append(item["pair_key"])
        mse_list.append(item["mse"])
        text_r1_list.append(item["text_r1"])

    return {
        "pair_key": pair_keys,
        "x0": x0,
        "x1": x1,
        "phoneme_ids": phoneme_ids,
        "x_lens": lengths,
        "x_mean": x_mean,
        "x_std": x_std,
        "mse": mse_list,
        "text_r1": text_r1_list,
    }