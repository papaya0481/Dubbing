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


class Dataset_CFM_Index_Phase1(Dataset):
    """Dataset for training IndexTTS2-CFM (Phase 1) with condition caching.

    Reads a ``metadata.csv`` file with columns:
      - ``prompt_audio_path`` : reference speaker audio
      - ``out_pt``            : pre-computed S_infer .pt file  [1?, T_codes, 1024]
      - ``out_wav``           : ground-truth target audio
      - ``gen_error``         : non-empty rows are skipped

    On first instantiation (or when cache files are missing), the dataset
    loads IndexTTS2 frozen sub-models (w2v-bert, semantic_codec, CAMPPlus,
    length_regulator) and pre-computes per-sample CFM conditions, saving
    each as ``<cache_dir>/<stem>.pt``.  Subsequent runs skip this step and
    load directly from cache.

    Each ``__getitem__`` returns CPU tensors:
      - ``stem``        : str
      - ``ref_mel``     : [num_mels, T_ref]   reference mel (prompt)
      - ``style``       : [192]               CAMPPlus speaker embedding
      - ``prompt_cond`` : [T_ref, 512]        length_regulator output for S_ref
      - ``infer_cond``  : [T_gen, 512]        length_regulator output for S_infer
      - ``x1_mel``      : [num_mels, T_gen]   target log-mel spectrogram
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
    ):
        super().__init__()
        self.csv_path     = Path(csv_path)
        self.data_root    = self.csv_path.parent
        self.split        = split
        self.mel_h        = mel_h
        self.preprocess   = preprocess
        self.sr_22k       = int(mel_h.sampling_rate)
        self.sr_ref_16k   = int(sr_ref_16k)
        self.max_ref_sec  = max_ref_sec
        self.max_gen_sec  = max_gen_sec
        self.max_code_len = max_code_len
        self.cache_dir    = (
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
    # CSV loading / splitting (unchanged)
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
                if out_pt and not Path(out_pt).is_absolute():
                    out_pt = str((self.data_root / out_pt).resolve())
                if out_wav and not Path(out_wav).is_absolute():
                    out_wav = str((self.data_root / out_wav).resolve())
                if not Path(out_pt).exists() or not Path(out_wav).exists():
                    continue
                samples.append({
                    "stem":              Path(out_pt).stem,
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
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self, stem: str) -> Path:
        return self.cache_dir / f"{stem}.pt"

    def _ensure_cache(self):
        missing = [
            s for s in self.samples
            if not self._cache_path(s["stem"]).exists()
        ]
        if not missing:
            logger.info(
                f"[Cache] All {len(self.samples)} samples already cached "
                f"in {self.cache_dir}"
            )
            return
        logger.info(
            f"[Cache] Building cache for {len(missing)}/{len(self.samples)} "
            f"samples → {self.cache_dir}"
        )
        self._build_cache(missing)

    def _build_cache(self, samples: List[Dict]):
        """Load IndexTTS2 frozen models, compute conditions for *samples*,
        save each as a .pt file, then release models."""
        # Ensure index-tts2 is importable
        _proj_root  = Path(__file__).resolve().parents[2]
        _index_root = _proj_root / "index-tts2"
        for _p in [str(_index_root), str(_proj_root)]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from omegaconf import OmegaConf
        from transformers import SeamlessM4TFeatureExtractor
        from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        from huggingface_hub import hf_hub_download
        import safetensors

        pre    = self.preprocess
        mdl    = Path(pre.model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Cache] Using device: {device}")

        idx_cfg = OmegaConf.load(str(mdl / "config.yaml"))

        logger.info("[Cache] Loading SeamlessM4TFeatureExtractor …")
        feat_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        logger.info("[Cache] Loading w2v-bert-2.0 …")
        sem_model, sem_mean, sem_std = build_semantic_model(str(mdl / pre.w2v_stat))
        sem_model = sem_model.to(device).eval()
        sem_mean  = sem_mean.to(device)
        sem_std   = sem_std.to(device)

        logger.info("[Cache] Loading semantic_codec …")
        codec = build_semantic_codec(idx_cfg.semantic_codec)
        codec_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
        )
        safetensors.torch.load_model(codec, codec_ckpt)
        sem_codec = codec.to(device).eval()

        logger.info("[Cache] Loading CAMPPlus …")
        campplus_ckpt = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
        campplus = campplus.to(device).eval()

        logger.info("[Cache] Loading length_regulator from s2mel.pth …")
        s2mel_model = MyModel(idx_cfg.s2mel, use_gpt_latent=False)
        s2mel_model, _, _, _ = load_checkpoint2(
            s2mel_model, None, str(mdl / pre.s2mel_checkpoint),
            load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        len_reg = s2mel_model.models["length_regulator"].to(device).eval()

        frozen = dict(
            feat_extractor=feat_extractor,
            sem_model=sem_model, sem_mean=sem_mean, sem_std=sem_std,
            sem_codec=sem_codec, campplus=campplus, len_reg=len_reg,
            device=device,
        )

        for item in tqdm(samples, desc="Building CFM condition cache"):
            try:
                self._cache_one(item, frozen)
            except Exception as e:
                logger.warning(f"[Cache] Failed for {item['stem']}: {e}")

        # Release GPU memory
        del frozen, sem_model, sem_codec, campplus, len_reg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[Cache] Done. Frozen models released.")

    @torch.no_grad()
    def _cache_one(self, item: Dict, frozen: Dict):
        """Compute and save CFM conditions for one sample."""
        device = frozen["device"]

        # ---- Load audio -------------------------------------------------
        def _load(path, sr, max_sec):
            wav, orig_sr = torchaudio.load(str(path))
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)
            if orig_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_sr, sr)
            return wav[: int(max_sec * sr)].float()

        ref_22k = _load(item["prompt_audio_path"], self.sr_22k,    self.max_ref_sec)
        ref_16k = _load(item["prompt_audio_path"], self.sr_ref_16k, self.max_ref_sec)
        x1_wav  = _load(item["out_wav"],           self.sr_22k,    self.max_gen_sec)

        # ---- ref_mel [num_mels, T_ref] ----------------------------------
        ref_mel = get_mel_spectrogram(
            ref_22k.unsqueeze(0).to(device), self.mel_h
        ).squeeze(0).cpu()   # [num_mels, T_ref]
        T_ref = ref_mel.size(-1)

        # ---- x1_mel [num_mels, T_gen] -----------------------------------
        x1_mel = get_mel_spectrogram(
            x1_wav.unsqueeze(0).to(device), self.mel_h
        ).squeeze(0).cpu()   # [num_mels, T_gen]
        T_gen = x1_mel.size(-1)

        # ---- style (CAMPPlus) [192] -------------------------------------
        fbank = torchaudio.compliance.kaldi.fbank(
            ref_16k.unsqueeze(0).to(device),
            num_mel_bins=80, dither=0, sample_frequency=self.sr_ref_16k,
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        style = frozen["campplus"](fbank.unsqueeze(0)).squeeze(0).cpu()  # [192]

        # ---- S_ref: w2v-bert → semantic_codec --------------------------
        inputs  = frozen["feat_extractor"](
            ref_16k.numpy(), sampling_rate=self.sr_ref_16k, return_tensors="pt"
        )
        feat_in = inputs["input_features"].to(device)
        attn_m  = inputs["attention_mask"].to(device)
        out     = frozen["sem_model"](
            input_features=feat_in, attention_mask=attn_m, output_hidden_states=True
        )
        spk_emb = (out.hidden_states[17] - frozen["sem_mean"]) / frozen["sem_std"]
        _, S_ref = frozen["sem_codec"].quantize(spk_emb)

        # ---- prompt_cond [T_ref, 512] -----------------------------------
        prompt_cond = frozen["len_reg"](
            S_ref,
            ylens=torch.LongTensor([T_ref]).to(device),
            n_quantizers=3, f0=None,
        )[0].squeeze(0).cpu()   # [T_ref, 512]

        # ---- S_infer: load .pt ------------------------------------------
        s_infer = torch.load(item["out_pt"], map_location="cpu")
        if s_infer.dim() == 3 and s_infer.size(0) == 1:
            s_infer = s_infer.squeeze(0)
        s_infer = s_infer.float()
        if s_infer.size(0) > self.max_code_len:
            s_infer = s_infer[: self.max_code_len, :]

        # ---- infer_cond [T_gen, 512] (target length from x1_mel) --------
        infer_cond = frozen["len_reg"](
            s_infer.unsqueeze(0).to(device),
            ylens=torch.LongTensor([T_gen]).to(device),
            n_quantizers=3, f0=None,
        )[0].squeeze(0).cpu()   # [T_gen, 512]

        # ---- Save cache -------------------------------------------------
        torch.save(
            {
                "ref_mel":     ref_mel,      # [num_mels, T_ref]
                "style":       style,        # [192]
                "prompt_cond": prompt_cond,  # [T_ref, 512]
                "infer_cond":  infer_cond,   # [T_gen, 512]
                "x1_mel":      x1_mel,       # [num_mels, T_gen]
            },
            self._cache_path(item["stem"]),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        item  = self.samples[index]
        data  = torch.load(
            self._cache_path(item["stem"]), map_location="cpu", weights_only=True
        )
        data["stem"] = item["stem"]
        return data


def collate_cfm_index_phase1(batch: List[Dict]) -> Dict:
    """Collate pre-computed CFM conditions from Dataset_CFM_Index_Phase1.

    Builds padded batch tensors ready for CFM.forward / CFM.inference:
      - x1_full    [B, num_mels, T_max]   ref_mel ++ x1_mel, padded
      - cond       [B, T_max, 512]        prompt_cond ++ infer_cond, padded
      - ref_mels   [B, num_mels, T_ref_max]  reference mel for CFM.inference
      - style      [B, 192]
      - x_lens     [B]  T_ref + T_gen per sample
      - prompt_lens[B]  T_ref per sample
    """
    B        = len(batch)
    num_mels = batch[0]["ref_mel"].size(0)

    T_refs   = [item["ref_mel"].size(-1)   for item in batch]
    T_gens   = [item["x1_mel"].size(-1)    for item in batch]
    T_totals = [r + g for r, g in zip(T_refs, T_gens)]

    T_ref_max   = max(T_refs)
    T_total_max = max(T_totals)

    x1_full  = torch.zeros(B, num_mels, T_total_max)
    cond     = torch.zeros(B, T_total_max, 512)
    ref_mels = torch.zeros(B, num_mels, T_ref_max)

    for i, item in enumerate(batch):
        T_r = T_refs[i]
        T_g = T_gens[i]
        x1_full[i, :, :T_r]       = item["ref_mel"]
        x1_full[i, :, T_r:T_r+T_g] = item["x1_mel"]
        cond[i, :T_r, :]          = item["prompt_cond"]
        cond[i, T_r:T_r+T_g, :]  = item["infer_cond"]
        ref_mels[i, :, :T_r]      = item["ref_mel"]

    style       = torch.stack([item["style"] for item in batch], dim=0)  # [B, 192]
    x_lens      = torch.tensor(T_totals, dtype=torch.long)
    prompt_lens = torch.tensor(T_refs,   dtype=torch.long)
    stems       = [item["stem"] for item in batch]

    return {
        "stems":       stems,
        "x1_full":     x1_full,
        "cond":        cond,
        "ref_mels":    ref_mels,
        "style":       style,
        "x_lens":      x_lens,
        "prompt_lens": prompt_lens,
    }