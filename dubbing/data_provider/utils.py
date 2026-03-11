"""dubbing/data_provider/utils.py

Standalone batched cache builder for Dataset_CFM_Index_Phase1.

Moved here from data_loader.py to keep the dataset module focused on
dataset / dataloader concerns only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from tqdm.auto import tqdm

from modules.mel_strech.meldataset import get_mel_spectrogram
from logger import get_logger

logger = get_logger("dubbing.data_provider.utils")


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
