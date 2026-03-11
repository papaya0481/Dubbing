import csv
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


def collate_cfm_phase1(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.stack([item["x_len"] for item in batch], dim=0)
    max_len = int(lengths.max().item())
    n_mels = batch[0]["cond_mel"].shape[0]

    cond_mel = torch.zeros(len(batch), n_mels, max_len, dtype=batch[0]["cond_mel"].dtype)
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
        cond_mel[i, :, :t] = item["cond_mel"][:, :t]
        x1[i, :, :t] = item["x1"][:, :t]
        phoneme_ids[i, :t] = item["phoneme_ids"][:t]
        pair_keys.append(item["pair_key"])
        mse_list.append(item["mse"])
        text_r1_list.append(item["text_r1"])

    return {
        "pair_key": pair_keys,
        "cond_mel": cond_mel,
        "x1": x1,
        "phoneme_ids": phoneme_ids,
        "x_lens": lengths,
        "x_mean": x_mean,
        "x_std": x_std,
        "mse": mse_list,
        "text_r1": text_r1_list,
    }


# =============================================================================
# Dataset_CFM_Index_Phase1
# =============================================================================

def _mel_spectrogram_cpu(
    wav: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sr: int = 22050,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax=None,
) -> torch.Tensor:
    """Compute log-mel spectrogram on CPU.

    Matches the behaviour of ``indextts.s2mel.modules.audio.mel_spectrogram``:
    librosa mel filter bank, reflect-padded STFT, log(clamp(x, min=1e-5)).

    Args:
        wav: [1, T] or [T] float32 waveform at *sr* Hz.
    Returns:
        [num_mels, T_mel] float32 log-mel spectrogram.
    """
    import librosa

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # [1, T]

    # Mel filter bank derived by librosa (matches indextts exactly)
    mel_fb = torch.from_numpy(
        librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    ).float()  # [num_mels, n_fft//2+1]

    window = torch.hann_window(win_size)

    # Reflect-pad to match indextts (centre=False behaviour)
    pad = (n_fft - hop_size) // 2
    wav_p = torch.nn.functional.pad(
        wav.unsqueeze(1), (pad, pad), mode="reflect"
    ).squeeze(1)  # [1, T+2*pad]

    spec = torch.stft(
        wav_p,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )  # [1, n_fft//2+1, T_mel]

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)  # magnitude [1, n_fft//2+1, T_mel]

    mel = torch.matmul(mel_fb, spec)               # [1, num_mels, T_mel]
    mel = torch.log(torch.clamp(mel, min=1e-5))    # log-mel
    return mel.squeeze(0)                          # [num_mels, T_mel]


class Dataset_CFM_Index_Phase1(Dataset):
    """Dataset for training IndexTTS2-CFM (Phase 1).

    Reads a ``metadata.csv`` file with the following columns:
      - ``prompt_audio_path`` : reference speaker audio (absolute or relative to csv dir)
      - ``out_pt``            : path to pre-computed S_infer .pt file  [1?, T_codes, 1024]
      - ``out_wav``           : path to target audio (ground truth for training)
      - ``gen_error``         : if non-empty the row is skipped

    Each ``__getitem__`` returns a dict with CPU tensors:
      - ``stem``              : sample identifier string
      - ``s_infer``           : [T_codes, 1024] GPT semantic features
      - ``s_infer_len``       : scalar LongTensor
      - ``ref_audio_22k``     : [T_22k]  reference audio at 22050 Hz
      - ``ref_audio_22k_len`` : scalar LongTensor
      - ``ref_audio_16k``     : [T_16k]  reference audio at 16000 Hz
      - ``ref_audio_16k_len`` : scalar LongTensor
      - ``x1_mel``            : [80, T_gen] log-mel spectrogram of target audio
      - ``x1_len``            : scalar LongTensor
    """

    MEL_SR    = 22050
    MEL_SR16  = 16000
    MEL_N_FFT = 1024
    MEL_WIN   = 1024
    MEL_HOP   = 256
    MEL_MELS  = 80
    MEL_FMIN  = 0.0

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 2026,
        max_ref_sec: float = 15.0,
        max_gen_sec: float = 10.0,
        max_code_len: int = 500,
    ):
        super().__init__()
        self.csv_path    = Path(csv_path)
        self.data_root   = self.csv_path.parent
        self.split       = split
        self.max_ref_sec = max_ref_sec
        self.max_gen_sec = max_gen_sec
        self.max_code_len = max_code_len

        all_samples = self._load_csv()
        self.samples = self._split_samples(all_samples, split, split_ratio, seed)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples for split={split!r} in {csv_path}"
            )
        logger.info(
            f"Dataset_CFM_Index_Phase1 [{split}]: {len(self.samples)} samples"
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_csv(self) -> List[Dict]:
        samples: List[Dict] = []
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
                # Resolve relative paths
                if out_pt and not Path(out_pt).is_absolute():
                    out_pt = str((self.data_root / out_pt).resolve())
                if out_wav and not Path(out_wav).is_absolute():
                    out_wav = str((self.data_root / out_wav).resolve())
                if not Path(out_pt).exists() or not Path(out_wav).exists():
                    continue
                samples.append({
                    "stem":               Path(out_pt).stem,
                    "prompt_audio_path":  prompt,
                    "out_pt":             out_pt,
                    "out_wav":            out_wav,
                })
        return samples

    @staticmethod
    def _split_samples(
        samples: List[Dict], split: str, split_ratio: float, seed: int
    ) -> List[Dict]:
        n = len(samples)
        if n == 0:
            return samples

        rng   = torch.Generator().manual_seed(seed)
        perm  = torch.randperm(n, generator=rng).tolist()
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
        return samples[val_end:]  # test

    def _load_audio(self, path: str, target_sr: int, max_sec: float) -> torch.Tensor:
        """Load and resample audio; returns [T] float32 (mono, truncated)."""
        wav, sr = torchaudio.load(str(path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        max_len = int(max_sec * target_sr)
        return wav[:max_len].float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]

        # ---- S_infer (pre-computed GPT semantic features) ---------------
        s_infer = torch.load(item["out_pt"], map_location="cpu")
        if s_infer.dim() == 3 and s_infer.size(0) == 1:
            s_infer = s_infer.squeeze(0)     # [T_codes, 1024]
        s_infer = s_infer.float()
        if s_infer.size(0) > self.max_code_len:
            s_infer = s_infer[: self.max_code_len, :]

        # ---- Reference audio --------------------------------------------
        ref_22k = self._load_audio(item["prompt_audio_path"], self.MEL_SR,   self.max_ref_sec)
        ref_16k = self._load_audio(item["prompt_audio_path"], self.MEL_SR16, self.max_ref_sec)

        # ---- Target mel (x1) -------------------------------------------
        x1_wav = self._load_audio(item["out_wav"], self.MEL_SR, self.max_gen_sec)
        x1_mel = _mel_spectrogram_cpu(
            x1_wav,
            n_fft=self.MEL_N_FFT, num_mels=self.MEL_MELS,
            sr=self.MEL_SR, hop_size=self.MEL_HOP, win_size=self.MEL_WIN,
            fmin=self.MEL_FMIN,
        )  # [80, T_gen]

        return {
            "stem":               item["stem"],
            "s_infer":            s_infer,
            "s_infer_len":        torch.tensor(s_infer.size(0),  dtype=torch.long),
            "ref_audio_22k":      ref_22k,
            "ref_audio_22k_len":  torch.tensor(ref_22k.size(0),  dtype=torch.long),
            "ref_audio_16k":      ref_16k,
            "ref_audio_16k_len":  torch.tensor(ref_16k.size(0),  dtype=torch.long),
            "x1_mel":             x1_mel,
            "x1_len":             torch.tensor(x1_mel.size(-1),  dtype=torch.long),
        }


def collate_cfm_index_phase1(batch: List[Dict]) -> Dict:
    """Pad and collate a batch from Dataset_CFM_Index_Phase1."""
    B = len(batch)

    max_codes  = max(item["s_infer_len"].item()       for item in batch)
    max_r22    = max(item["ref_audio_22k_len"].item()  for item in batch)
    max_r16    = max(item["ref_audio_16k_len"].item()  for item in batch)
    max_gen    = max(item["x1_len"].item()             for item in batch)
    n_mels     = batch[0]["x1_mel"].size(0)

    s_infer       = torch.zeros(B, max_codes, 1024)
    ref_audio_22k = torch.zeros(B, max_r22)
    ref_audio_16k = torch.zeros(B, max_r16)
    x1_mel        = torch.zeros(B, n_mels, max_gen)

    s_infer_lens      = torch.zeros(B, dtype=torch.long)
    ref_audio_22k_lens = torch.zeros(B, dtype=torch.long)
    ref_audio_16k_lens = torch.zeros(B, dtype=torch.long)
    x1_lens            = torch.zeros(B, dtype=torch.long)
    stems: List[str]   = []

    for i, item in enumerate(batch):
        tc  = item["s_infer_len"].item()
        t22 = item["ref_audio_22k_len"].item()
        t16 = item["ref_audio_16k_len"].item()
        tg  = item["x1_len"].item()

        s_infer[i, :tc, :]      = item["s_infer"][:tc, :]
        ref_audio_22k[i, :t22]  = item["ref_audio_22k"][:t22]
        ref_audio_16k[i, :t16]  = item["ref_audio_16k"][:t16]
        x1_mel[i, :, :tg]       = item["x1_mel"][:, :tg]

        s_infer_lens[i]       = tc
        ref_audio_22k_lens[i] = t22
        ref_audio_16k_lens[i] = t16
        x1_lens[i]            = tg
        stems.append(item["stem"])

    return {
        "stems":              stems,
        "s_infer":            s_infer,
        "s_infer_lens":       s_infer_lens,
        "ref_audio_22k":      ref_audio_22k,
        "ref_audio_22k_lens": ref_audio_22k_lens,
        "ref_audio_16k":      ref_audio_16k,
        "ref_audio_16k_lens": ref_audio_16k_lens,
        "x1_mel":             x1_mel,
        "x1_lens":            x1_lens,
    }