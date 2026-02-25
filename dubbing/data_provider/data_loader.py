import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tgt
import torch
import torchaudio
from torch.utils.data import Dataset

from dubbing.modules.mel_strech.meldataset import get_mel_spectrogram
from dubbing.modules.mel_strech.mel_transform import GlobalWarpTransformer


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
        sample_rate: int = 22050,
        n_fft: int = 1024,
        num_mels: int = 80,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: int = 0,
        fmax: Optional[int] = None,
        tier_name: str = "words",
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.ost_dir = self.root_dir / "ost"
        self.aligned_dir = self.root_dir / "aligned"
        self.split = split
        self.tier_name = tier_name
        self.filter_enabled = filter_enabled
        self.mse_threshold = mse_threshold
        self.sample_rate = sample_rate
        self.device = torch.device("cpu")

        self.mel_h = SimpleNamespace(
            n_fft=n_fft,
            num_mels=num_mels,
            sampling_rate=sample_rate,
            hop_size=hop_size,
            win_size=win_size,
            fmin=fmin,
            fmax=(sample_rate // 2 if fmax is None else fmax),
        )

        self._warper = GlobalWarpTransformer.__new__(GlobalWarpTransformer)
        self._warper.sample_rate = sample_rate
        self._warper.h = self.mel_h
        self._warper.device = self.device

        all_pairs = self._discover_pairs()
        all_pairs = self._apply_filter(all_pairs)
        self.samples = self._split_samples(all_pairs, split, split_ratio, seed)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} in {root_dir}")

    def _discover_pairs(self) -> List[PairSample]:
        if not self.ost_dir.exists():
            raise FileNotFoundError(f"ost folder not found: {self.ost_dir}")
        if not self.aligned_dir.exists():
            raise FileNotFoundError(f"aligned folder not found: {self.aligned_dir}")

        pair_map: Dict[str, Dict[str, Path]] = {}
        for wav_path in self.ost_dir.rglob("*.wav"):
            key, role = _pair_key_and_role(wav_path.stem)
            if key is None or role is None:
                continue
            if key not in pair_map:
                pair_map[key] = {}
            pair_map[key][role] = wav_path

        samples: List[PairSample] = []
        for key, pair in pair_map.items():
            if "r1" not in pair or "r2" not in pair:
                continue
            r1_wav = pair["r1"]
            r2_wav = pair["r2"]

            rel_r1 = r1_wav.relative_to(self.ost_dir)
            rel_r2 = r2_wav.relative_to(self.ost_dir)

            r1_tg = (self.aligned_dir / rel_r1).with_suffix(".TextGrid")
            r2_tg = (self.aligned_dir / rel_r2).with_suffix(".TextGrid")
            if not r1_tg.exists() or not r2_tg.exists():
                continue

            samples.append(PairSample(key, r1_wav, r2_wav, r1_tg, r2_tg))

        samples.sort(key=lambda x: x.pair_key)
        return samples

    def _load_filter_table(self) -> Optional[pd.DataFrame]:
        if not self.filter_enabled:
            return None

        candidates = [
            self.ost_dir / "generated_metada.csv",
            self.ost_dir / "generation_metadata.csv",
            self.ost_dir / "generated_metadata.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            return None

        df = pd.read_csv(csv_path)
        if "mse" not in df.columns:
            return None
        return df

    def _apply_filter(self, samples: List[PairSample]) -> List[PairSample]:
        df = self._load_filter_table()
        if df is None:
            return samples

        allowed_keys = set()
        if "pair_key" in df.columns:
            allowed_keys = set(df.loc[df["mse"] <= self.mse_threshold, "pair_key"].astype(str).tolist())
            return [s for s in samples if s.pair_key in allowed_keys]

        if "r1" in df.columns and "r2" in df.columns:
            valid = df[df["mse"] <= self.mse_threshold]
            valid_pairs = set(zip(valid["r1"].astype(str), valid["r2"].astype(str)))
            return [
                s
                for s in samples
                if (s.r1_audio.name in {p[0] for p in valid_pairs}) and (s.r2_audio.name in {p[1] for p in valid_pairs})
            ]

        return samples

    @staticmethod
    def _split_samples(samples: List[PairSample], split: str, split_ratio: float, seed: int) -> List[PairSample]:
        if split not in {"train", "test"}:
            return samples

        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(samples), generator=rng).tolist()
        samples = [samples[i] for i in perm]

        cut = max(1, int(len(samples) * split_ratio))
        if split == "train":
            return samples[:cut]
        return samples[cut:] if cut < len(samples) else samples[-1:]

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

    def _stretch_r1_to_r2(self, r1_mel: torch.Tensor, r1_tg: Path, r2_tg: Path, r1_wav: torch.Tensor, r2_wav: torch.Tensor) -> torch.Tensor:
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
        stretched_mel = self._warper.warp_mel(r1_mel, warping_path)
        return stretched_mel

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        r1_wav = self._load_wav(sample.r1_audio)
        r2_wav = self._load_wav(sample.r2_audio)

        r1_mel = self._wav_to_mel(r1_wav)
        r2_mel = self._wav_to_mel(r2_wav)

        stretched_r1_mel = self._stretch_r1_to_r2(r1_mel, sample.r1_tg, sample.r2_tg, r1_wav, r2_wav)

        target_frames = min(stretched_r1_mel.shape[-1], r2_mel.shape[-1])
        stretched_r1_mel = stretched_r1_mel[:, :, :target_frames]
        r2_mel = r2_mel[:, :, :target_frames]

        return {
            "pair_key": sample.pair_key,
            "x0": stretched_r1_mel.squeeze(0),
            "x1": r2_mel.squeeze(0),
            "x_len": torch.tensor(target_frames, dtype=torch.long),
        }


def collate_cfm_phase1(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.stack([item["x_len"] for item in batch], dim=0)
    max_len = int(lengths.max().item())
    n_mels = batch[0]["x0"].shape[0]

    x0 = torch.zeros(len(batch), n_mels, max_len, dtype=batch[0]["x0"].dtype)
    x1 = torch.zeros(len(batch), n_mels, max_len, dtype=batch[0]["x1"].dtype)

    pair_keys = []
    for i, item in enumerate(batch):
        t = int(item["x_len"].item())
        x0[i, :, :t] = item["x0"][:, :t]
        x1[i, :, :t] = item["x1"][:, :t]
        pair_keys.append(item["pair_key"])

    return {
        "pair_key": pair_keys,
        "x0": x0,
        "x1": x1,
        "x_lens": lengths,
    }