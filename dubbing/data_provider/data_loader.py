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
# CFMIndexCacheBuilder  –  standalone batched cache builder
# =============================================================================


class CFMIndexCacheBuilder:
    """Build per-sample ``.pt`` cache files for Dataset_CFM_Index_Phase1.

    Heavy frozen-model inference (w2v-bert, CAMPPlus) is **batched** for speed.
    The length_regulator is run per-sample to avoid variable-length padding
    artefacts in the interpolated output.

    Speedup over the old per-sample approach comes primarily from batching the
    w2v-bert encoder (the dominant bottleneck) and CAMPPlus.

    Cache format (``<cache_dir>/<stem>.pt``):
      - ``ref_mel``     : [num_mels, T_ref]   reference mel spectrogram
      - ``style``       : [192]               CAMPPlus speaker embedding
      - ``prompt_cond`` : [T_ref, 512]        length_regulator(S_ref)
      - ``infer_cond``  : [T_gen, 512]        length_regulator(S_infer)
      - ``x1_mel``      : [num_mels, T_gen]   target mel spectrogram
    """

    def __init__(
        self,
        preprocess,
        mel_h,
        cache_dir,
        sr_ref_16k: int = 16000,
        max_ref_sec: float = 15.0,
        max_gen_sec: float = 10.0,
        max_code_len: int = 500,
        batch_size: int = 16,
    ):
        self.preprocess   = preprocess
        self.mel_h        = mel_h
        self.cache_dir    = Path(cache_dir)
        self.sr_22k       = int(mel_h.sampling_rate)
        self.sr_ref_16k   = int(sr_ref_16k)
        self.max_ref_sec  = max_ref_sec
        self.max_gen_sec  = max_gen_sec
        self.max_code_len = max_code_len
        self.batch_size   = batch_size
        self._models: Optional[Dict] = None
        self.device: Optional[torch.device] = None

    def cache_path(self, stem: str) -> Path:
        return self.cache_dir / f"{stem}.pt"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, samples: List[Dict]) -> None:
        """Build cache for *samples* (those not yet cached)."""
        self._ensure_importable()
        self._load_models()
        try:
            batches = [
                samples[i: i + self.batch_size]
                for i in range(0, len(samples), self.batch_size)
            ]
            for batch_items in tqdm(batches, desc="Building CFM condition cache"):
                try:
                    self._process_batch(batch_items)
                except Exception as e:
                    logger.warning(
                        f"[Cache] Batch failed ({e!r}), falling back to per-sample"
                    )
                    for item in batch_items:
                        try:
                            self._process_batch([item])
                        except Exception as e2:
                            logger.warning(
                                f"[Cache] Skipping {item['stem']}: {e2!r}"
                            )
        finally:
            self._release_models()

    # ------------------------------------------------------------------
    # Model loading / release
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_importable():
        _proj  = Path(__file__).resolve().parents[2]
        _index = _proj / "index-tts2"
        for _p in [str(_index), str(_proj)]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

    def _load_models(self):
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
        logger.info(f"[Cache] device={device}")

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

        logger.info("[Cache] Loading length_regulator …")
        s2mel_model = MyModel(idx_cfg.s2mel, use_gpt_latent=False)
        s2mel_model, _, _, _ = load_checkpoint2(
            s2mel_model, None, str(mdl / pre.s2mel_checkpoint),
            load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        len_reg = s2mel_model.models["length_regulator"].to(device).eval()

        self.device  = device
        self._models = dict(
            feat_extractor=feat_extractor,
            sem_model=sem_model, sem_mean=sem_mean, sem_std=sem_std,
            sem_codec=sem_codec, campplus=campplus, len_reg=len_reg,
        )
        logger.info("[Cache] Frozen models ready.")

    def _release_models(self):
        if self._models is None:
            return
        del self._models
        self._models = None
        self.device  = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[Cache] Frozen models released.")

    # ------------------------------------------------------------------
    # Audio loading helper
    # ------------------------------------------------------------------

    def _load_audio(self, path: str, sr: int, max_sec: float) -> torch.Tensor:
        wav, orig_sr = torchaudio.load(str(path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        return wav[: int(max_sec * sr)].float()

    # ------------------------------------------------------------------
    # Core: batched processing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _process_batch(self, items: List[Dict]) -> None:
        dev = self.device
        m   = self._models
        B   = len(items)

        # ── 1. Load audio ────────────────────────────────────────────────
        ref_22k = [self._load_audio(it["prompt_audio_path"], self.sr_22k,     self.max_ref_sec) for it in items]
        ref_16k = [self._load_audio(it["prompt_audio_path"], self.sr_ref_16k, self.max_ref_sec) for it in items]
        x1_wavs = [self._load_audio(it["out_wav"],           self.sr_22k,     self.max_gen_sec) for it in items]

        # ── 2. Mel (per-sample; fast) ─────────────────────────────────
        ref_mels = [
            get_mel_spectrogram(w.unsqueeze(0).to(dev), self.mel_h).squeeze(0).cpu()
            for w in ref_22k
        ]
        x1_mels = [
            get_mel_spectrogram(w.unsqueeze(0).to(dev), self.mel_h).squeeze(0).cpu()
            for w in x1_wavs
        ]
        T_refs = [rm.size(-1) for rm in ref_mels]
        T_gens = [xm.size(-1) for xm in x1_mels]

        # ── 3. Batched fbank → CAMPPlus ──────────────────────────────
        fbanks = []
        for w16 in ref_16k:
            fb = torchaudio.compliance.kaldi.fbank(
                w16.unsqueeze(0).to(dev),
                num_mel_bins=80, dither=0, sample_frequency=self.sr_ref_16k,
            )
            fbanks.append(fb - fb.mean(0, keepdim=True))
        max_fb = max(f.size(0) for f in fbanks)
        fb_pad = torch.zeros(B, max_fb, 80, device=dev)
        for i, fb in enumerate(fbanks):
            fb_pad[i, :fb.size(0)] = fb
        styles = m["campplus"](fb_pad).cpu()   # [B, 192]

        # ── 4. Batched w2v-bert encoder ──────────────────────────────
        inputs  = m["feat_extractor"](
            [w.numpy() for w in ref_16k],
            sampling_rate=self.sr_ref_16k,
            return_tensors="pt",
            padding=True,
        )
        feat_in = inputs["input_features"].to(dev)
        attn_m  = inputs["attention_mask"].to(dev)
        out     = m["sem_model"](
            input_features=feat_in, attention_mask=attn_m, output_hidden_states=True
        )
        spk_embs = (out.hidden_states[17] - m["sem_mean"]) / m["sem_std"]
        # attention_mask frames ≡ hidden-state frames for w2v-bert-2.0
        T_feats  = attn_m.sum(dim=1).tolist()

        # Batched semantic_codec quantize; padding positions are discarded per sample
        _, S_refs_padded = m["sem_codec"].quantize(spk_embs)   # [B, T_feat_max, 1024]

        # ── 5. Per-sample len_reg → prompt_cond ───────────────────────
        prompt_conds = []
        for i in range(B):
            T_f     = int(T_feats[i])
            s_ref_i = S_refs_padded[i:i+1, :T_f, :]   # [1, T_feat_i, 1024]
            pc = m["len_reg"](
                s_ref_i,
                ylens=torch.LongTensor([T_refs[i]]).to(dev),
                n_quantizers=3, f0=None,
            )[0].squeeze(0).cpu()                       # [T_ref_i, 512]
            prompt_conds.append(pc)

        # ── 6. Per-sample S_infer → len_reg → infer_cond ───────────────
        infer_conds = []
        for i, item in enumerate(items):
            s = torch.load(item["out_pt"], map_location="cpu")
            if s.dim() == 3 and s.size(0) == 1:
                s = s.squeeze(0)
            s = s.float()
            if s.size(0) > self.max_code_len:
                s = s[: self.max_code_len]
            ic = m["len_reg"](
                s.unsqueeze(0).to(dev),
                ylens=torch.LongTensor([T_gens[i]]).to(dev),
                n_quantizers=3, f0=None,
            )[0].squeeze(0).cpu()                       # [T_gen_i, 512]
            infer_conds.append(ic)

        # ── 7. Save per-sample cache ─────────────────────────────────
        for i, item in enumerate(items):
            torch.save(
                {
                    "ref_mel":     ref_mels[i],
                    "style":       styles[i],
                    "prompt_cond": prompt_conds[i],
                    "infer_cond":  infer_conds[i],
                    "x1_mel":      x1_mels[i],
                },
                self.cache_path(item["stem"]),
            )


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
        data["stem"] = item["stem"]
        return data


def collate_cfm_index_phase1(batch: List[Dict]) -> Dict:
    """Collate pre-computed CFM conditions from Dataset_CFM_Index_Phase1.

    ``prompt_cond`` and ``infer_cond`` are returned as **separate** padded
    tensors.  The Exp is responsible for assembling the full condition tensor
    (e.g. via ``_assemble_cond``) before calling CFM.forward / CFM.inference.

    Keys returned
    -------------
    x1_full     [B, num_mels, T_max]       ref_mel ++ x1_mel, zero-padded
    ref_mels    [B, num_mels, T_ref_max]   reference mel for CFM.inference
    prompt_cond [B, T_ref_max, 512]        length_regulator(S_ref), zero-padded
    infer_cond  [B, T_gen_max, 512]        length_regulator(S_infer), zero-padded
    style       [B, 192]
    x_lens      [B]   T_ref + T_gen per sample  (total CFM sequence length)
    prompt_lens [B]   T_ref per sample
    infer_lens  [B]   T_gen per sample
    """
    B        = len(batch)
    num_mels = batch[0]["ref_mel"].size(0)

    T_refs   = [item["ref_mel"].size(-1) for item in batch]
    T_gens   = [item["x1_mel"].size(-1)  for item in batch]
    T_totals = [r + g for r, g in zip(T_refs, T_gens)]

    T_ref_max   = max(T_refs)
    T_gen_max   = max(T_gens)
    T_total_max = max(T_totals)

    x1_full     = torch.zeros(B, num_mels, T_total_max)
    ref_mels    = torch.zeros(B, num_mels, T_ref_max)
    prompt_cond = torch.zeros(B, T_ref_max, 512)
    infer_cond  = torch.zeros(B, T_gen_max, 512)

    for i, item in enumerate(batch):
        T_r = T_refs[i]
        T_g = T_gens[i]
        x1_full[i, :, :T_r]        = item["ref_mel"]
        x1_full[i, :, T_r:T_r+T_g] = item["x1_mel"]
        ref_mels[i, :, :T_r]       = item["ref_mel"]
        prompt_cond[i, :T_r, :]    = item["prompt_cond"]
        infer_cond[i, :T_g, :]     = item["infer_cond"]

    style       = torch.stack([item["style"] for item in batch])   # [B, 192]
    x_lens      = torch.tensor(T_totals, dtype=torch.long)
    prompt_lens = torch.tensor(T_refs,   dtype=torch.long)
    infer_lens  = torch.tensor(T_gens,   dtype=torch.long)
    stems       = [item["stem"] for item in batch]

    return {
        "stems":       stems,
        "x1_full":     x1_full,
        "ref_mels":    ref_mels,
        "prompt_cond": prompt_cond,
        "infer_cond":  infer_cond,
        "style":       style,
        "x_lens":      x_lens,
        "prompt_lens": prompt_lens,
        "infer_lens":  infer_lens,
    }