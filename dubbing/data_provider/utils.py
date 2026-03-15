"""dubbing/data_provider/utils.py

Standalone batched cache builder for Dataset_CFM_Index_Phase1.

Moved here from data_loader.py to keep the dataset module focused on
dataset / dataloader concerns only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from tqdm.auto import tqdm
import tgt

from modules.mel_strech.meldataset import get_mel_spectrogram
from modules.semantic_stretch.semantic_transform import SemanticTransformer
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
        mel_ratio: float = 1.72265625,  # 1/50 * 22050/256, aligned with infer_v2
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
        self.mel_ratio    = mel_ratio
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

    def _lr_ylens_from_semantic(self, semantic: torch.Tensor, dev: torch.device) -> torch.LongTensor:
        """Build LR target length from semantic time dim via fixed mel_ratio."""
        if semantic.dim() == 3:
            t_sem = int(semantic.size(1))
        elif semantic.dim() == 2:
            t_sem = int(semantic.size(0))
        else:
            raise ValueError(f"Unexpected semantic shape for length regulator: {tuple(semantic.shape)}")
        t_mel = max(1, int(t_sem * self.mel_ratio))
        return torch.LongTensor([t_mel]).to(dev)

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
                ylens=self._lr_ylens_from_semantic(s_ref_i, dev),
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
                ylens=self._lr_ylens_from_semantic(s.unsqueeze(0), dev),
                n_quantizers=3, f0=None,
            )[0].squeeze(0).cpu()                       # [T_gen_i, 512]
            infer_conds.append(ic)

        # ── 7. Save per-sample cache ─────────────────────────────────
        for i, item in enumerate(items):
            t_ref = min(T_refs[i], int(prompt_conds[i].size(0)))
            t_gen = min(T_gens[i], int(infer_conds[i].size(0)))
            torch.save(
                {
                    "ref_mel":     ref_mels[i][:, :t_ref],
                    "style":       styles[i],
                    "prompt_cond": prompt_conds[i][:t_ref],
                    "infer_cond":  infer_conds[i][:t_gen],
                    "x1_mel":      x1_mels[i][:, :t_gen],
                },
                self.cache_path(item["stem"]),
            )


class CFMIndexLipsInferCacheBuilder(CFMIndexCacheBuilder):
    """Build independent lipsfeat cache with stretched semantics.

    Cache format matches ``Dataset_CFM_Index_Phase1`` but the generated target
    branch differs:
      - ``ref_mel``     : mel from prompt audio
      - ``style``       : speaker embedding from prompt audio
      - ``prompt_cond`` : length_regulator(S_ref)
      - ``infer_cond``  : length_regulator(stretched S_infer)
      - ``x1_mel``      : CFM-generated mel from the stretched infer condition
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
        tier_name: str = "phones",
        diffusion_steps: int = 10,
        inference_cfg_rate: float = 0.7,
        batch_size: int = 8,
        warp_type: str = "cond",  # "semantic" or "cond"
    ):
        super().__init__(
            preprocess=preprocess,
            mel_h=mel_h,
            cache_dir=cache_dir,
            sr_ref_16k=sr_ref_16k,
            max_ref_sec=max_ref_sec,
            max_gen_sec=max_gen_sec,
            max_code_len=max_code_len,
            batch_size=batch_size,
        )
        self.tier_name = tier_name
        self.diffusion_steps = int(diffusion_steps)
        self.inference_cfg_rate = float(inference_cfg_rate)
        self.warp_type = warp_type
        self._cfm = None
        self._phoneme_vocab = None
        self.semantic_transformer = None

    def _load_runtime(self):
        super()._load_models()
        from omegaconf import OmegaConf
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2

        model_dir = Path(self.preprocess.model_dir)
        cfg = OmegaConf.load(str(model_dir / "config.yaml"))
        s2mel_model = MyModel(cfg.s2mel, use_gpt_latent=False)
        s2mel_model, _, _, _ = load_checkpoint2(
            s2mel_model,
            None,
            str(model_dir / self.preprocess.s2mel_checkpoint),
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self._cfm = s2mel_model.models["cfm"].to(self.device).eval()
        try:
            self._cfm.estimator.setup_caches(
                max_batch_size=max(1, int(self.batch_size)),
                max_seq_length=8192,
            )
        except Exception:
            pass

        try:
            from lips.data.phoneme_vocab import PhonemeVocab
        except Exception:
            _proj = Path(__file__).resolve().parents[2]
            if str(_proj) not in sys.path:
                sys.path.insert(0, str(_proj))
            from lips.data.phoneme_vocab import PhonemeVocab
        self._phoneme_vocab = PhonemeVocab()
        self.semantic_transformer = SemanticTransformer(device=self.device, verbose=False, input_type=self.warp_type)

    def _release_runtime(self):
        self._cfm = None
        self._phoneme_vocab = None
        self.semantic_transformer = None
        super()._release_models()

    @torch.no_grad()
    def _process_batch(self, items: List[Dict]) -> None:
        dev = self.device
        m = self._models
        B = len(items)

        ref_22k = [self._load_audio(it["prompt_audio_path"], self.sr_22k, self.max_ref_sec) for it in items]
        ref_16k = [self._load_audio(it["prompt_audio_path"], self.sr_ref_16k, self.max_ref_sec) for it in items]
        ref_mels = [
            get_mel_spectrogram(w.unsqueeze(0).to(dev), self.mel_h).squeeze(0).cpu()
            for w in ref_22k
        ]
        T_refs = [rm.size(-1) for rm in ref_mels]

        fbanks = []
        for w16 in ref_16k:
            fb = torchaudio.compliance.kaldi.fbank(
                w16.unsqueeze(0).to(dev),
                num_mel_bins=80,
                dither=0,
                sample_frequency=self.sr_ref_16k,
            )
            fbanks.append(fb - fb.mean(0, keepdim=True))
        max_fb = max(f.size(0) for f in fbanks)
        fb_pad = torch.zeros(B, max_fb, 80, device=dev)
        for i, fb in enumerate(fbanks):
            fb_pad[i, :fb.size(0)] = fb
        styles = m["campplus"](fb_pad).cpu()

        inputs = m["feat_extractor"](
            [w.numpy() for w in ref_16k],
            sampling_rate=self.sr_ref_16k,
            return_tensors="pt",
            padding=True,
        )
        feat_in = inputs["input_features"].to(dev)
        attn_m = inputs["attention_mask"].to(dev)
        out = m["sem_model"](
            input_features=feat_in,
            attention_mask=attn_m,
            output_hidden_states=True,
        )
        spk_embs = (out.hidden_states[17] - m["sem_mean"]) / m["sem_std"]
        T_feats = attn_m.sum(dim=1).tolist()
        _, S_refs_padded = m["sem_codec"].quantize(spk_embs)

        prompt_conds = []
        for i in range(B):
            T_f = int(T_feats[i])
            s_ref_i = S_refs_padded[i:i + 1, :T_f, :]
            prompt_cond = m["len_reg"](
                s_ref_i,
                ylens=self._lr_ylens_from_semantic(s_ref_i, dev),
                n_quantizers=3,
                f0=None,
            )[0].squeeze(0).cpu()
            prompt_conds.append(prompt_cond)

        infer_conds = []
        for i, item in enumerate(items):
            s_infer = torch.load(item["out_pt"], map_location="cpu", weights_only=False)
            if s_infer.dim() == 2:
                s_infer = s_infer.unsqueeze(0)
            elif s_infer.dim() == 3 and s_infer.size(0) != 1:
                s_infer = s_infer[:1]
            s_infer = s_infer.float()
            if s_infer.size(1) > self.max_code_len:
                s_infer = s_infer[:, : self.max_code_len, :]

            src_tg = tgt.io.read_textgrid(str(item["source_textgrid"]))
            tgt_tg = tgt.io.read_textgrid(str(item["lips_textgrid"]))
            for tg_obj in (src_tg, tgt_tg):
                try:
                    tier = tg_obj.get_tier_by_name(self.tier_name)
                except Exception:
                    continue
                for iv in tier:
                    txt = (iv.text or "").strip()
                    if txt == "":
                        continue
                    vfa_id = self._phoneme_vocab.arpabet_to_vfa_id(txt)
                    iv.text = self._phoneme_vocab.vfa_id_to_arpabet(vfa_id)

            if self.warp_type == "cond":
                # Run LR first to get cond, then warp in cond (mel) space
                infer_cond_pre = m["len_reg"](
                    s_infer.to(dev),
                    ylens=self._lr_ylens_from_semantic(s_infer, dev),
                    n_quantizers=3,
                    f0=None,
                )[0]  # (1, T_gen, 512)
                infer_cond_warped, _ = self.semantic_transformer.transform(
                    x=infer_cond_pre,
                    source_textgrid=src_tg,
                    target_textgrid=tgt_tg,
                    tier_name=self.tier_name,
                )
                infer_cond = infer_cond_warped.squeeze(0).cpu()
            else:
                # Warp in semantic space first, then run LR
                s_warped, _ = self.semantic_transformer.transform(
                    x=s_infer,
                    source_textgrid=src_tg,
                    target_textgrid=tgt_tg,
                    tier_name=self.tier_name,
                )
                infer_cond = m["len_reg"](
                    s_warped.to(dev),
                    ylens=self._lr_ylens_from_semantic(s_warped, dev),
                    n_quantizers=3,
                    f0=None,
                )[0].squeeze(0).cpu()
            infer_conds.append(infer_cond)

        # 统一按真实 cond 长度构建 batch，避免 mel 长度与 LR 输出长度存在 ±1 误差导致写入失败。
        T_refs = [min(T_refs[i], int(prompt_conds[i].size(0))) for i in range(B)]
        T_gens = [int(infer_conds[i].size(0)) for i in range(B)]

        # Keep pre-processing batched, but force CFM inference to single-sample mode.
        # This avoids cross-sample coupling inside CFM inference for variable prompt lengths.
        vc_target_list = []
        for i in range(B):
            t_ref = T_refs[i]
            t_gen = T_gens[i]
            total_len = t_ref + t_gen

            cond_i = torch.cat(
                [prompt_conds[i][:t_ref], infer_conds[i][:t_gen]],
                dim=0,
            ).unsqueeze(0)  # (1, T_total, 512)
            ref_mel_i = ref_mels[i][:, :t_ref].unsqueeze(0)  # (1, 80, T_ref)
            style_i = styles[i].unsqueeze(0)  # (1, 192)

            vc_target_i = self._cfm.inference(
                cond_i.to(dev),
                torch.LongTensor([total_len]).to(dev),
                ref_mel_i.to(dev),
                style_i.to(dev),
                None,
                self.diffusion_steps,
                inference_cfg_rate=self.inference_cfg_rate,
            ).cpu().squeeze(0)  # (80, T_total)
            vc_target_list.append(vc_target_i)

        for i, item in enumerate(items):
            t_ref = T_refs[i]
            t_gen = T_gens[i]
            x1_mel = vc_target_list[i][:, t_ref:t_ref + t_gen]
            torch.save(
                {
                    "ref_mel": ref_mels[i][:, :t_ref],
                    "style": styles[i],
                    "prompt_cond": prompt_conds[i][:t_ref],
                    "infer_cond": infer_conds[i][:t_gen],
                    "x1_mel": x1_mel,
                    "_lips_cache_version": 4,
                    "_x1_from_stretched_infer": True,
                },
                self.cache_path(item["stem"]),
            )

    def build(self, samples: List[Dict]) -> None:
        self._ensure_importable()
        self._load_runtime()
        try:
            missing = []
            failed = []
            for s in samples:
                cp = self.cache_path(s["stem"])
                if not cp.exists():
                    missing.append(s)
                    continue
                try:
                    cached = torch.load(cp, map_location="cpu", weights_only=False)
                    if (
                        not bool(cached.get("_x1_from_stretched_infer", False))
                        or int(cached.get("_lips_cache_version", 0)) < 4
                    ):
                        missing.append(s)
                except Exception:
                    missing.append(s)

            if not missing:
                logger.info(f"[LipsCache] All {len(samples)} samples already cached")
                return

            logger.info(
                f"[LipsCache] Building {len(missing)}/{len(samples)} entries -> {self.cache_dir}"
            )
            batches = [
                missing[i: i + self.batch_size]
                for i in range(0, len(missing), self.batch_size)
            ]
            for batch_items in tqdm(batches, desc="Building CFM lips infer cache"):
                try:
                    self._process_batch(batch_items)
                except Exception as e:
                    logger.warning(f"[LipsCache] Batch failed ({e!r}), fallback to per-sample")
                    if os.environ.get("DUBBING_LIPS_CACHE_DEBUG_RERAISE", "0") == "1":
                        logger.exception("[LipsCache] Batch traceback (debug reraised)")
                        raise
                    for item in batch_items:
                        try:
                            self._process_batch([item])
                        except Exception as e2:
                            failed.append(item.get("stem", "<unknown>"))
                            logger.warning(
                                f"[LipsCache] Skipping {item.get('stem', '<unknown>')}: {e2!r}"
                            )

            if failed:
                logger.warning(
                    f"[LipsCache] {len(failed)} samples failed to build and will be unavailable"
                )
        finally:
            self._release_runtime()
