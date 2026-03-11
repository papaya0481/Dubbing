"""Exp_CFM_Index_Phase1_TrainExpand

Training & testing experiment for the IndexTTS2-style CFM (Phase 1).

Key differences from Exp_CFM_Phase1_TrainExpand:
  - Model:  dubbing.modules.cfm_index.CFM  (DiT-based, s2mel architecture)
  - Data:   Dataset_CFM_Index_Phase1 (loads from metadata.csv with S_infer .pt)
  - Frozen: w2v-bert-2.0, semantic_codec (RepCodec), CAMPPlus, length_regulator
            are loaded once and used to build per-batch CFM conditions

Condition building (per batch, under torch.no_grad()):
  For each sample i:
    1. ref_audio_16k  → w2v-bert → semantic_codec.quantize → S_ref
    2. S_ref          → length_regulator(ylens=T_ref_mel)  → prompt_cond [1, T_ref, 512]
    3. ref_audio_22k  → mel_spectrogram                    → ref_mel  [1, 80, T_ref]
    4. ref_audio_16k  → CAMPPlus fbank                     → style    [1, 192]
    5. s_infer        → length_regulator(ylens=T_gen)      → infer_cond [1, T_gen, 512]
    6. cat([prompt_cond, infer_cond], dim=1)               → cond  [1, T_total, 512]
    7. cat([ref_mel,   x1_mel_i],    dim=2)                → x1    [1, 80, T_total]

  Batched output (after padding):
    x1        [B, 80, T_max]   – concat of ref mel + target mel, padded
    cond      [B, T_max, 512]  – cat_condition, padded
    style     [B, 192]
    x_lens    [B]              – total (ref + gen) frame count per sample
    prompt_lens [B]            – ref frame count per sample (= ref_mel.size(-1))

  CFM.forward(x1, x_lens, prompt_lens, cond, style) → (loss, _)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchaudio
from tqdm.auto import tqdm

from exp.basic import Exp_Basic
from data_provider.data_factory import data_provider
from logger import get_logger

logger = get_logger("dubbing.exp.cfm_index")

# ---------------------------------------------------------------------------
# Ensure index-tts2 is importable (mirrors test_cfm_index.py setup)
# ---------------------------------------------------------------------------
_PROJ_ROOT  = Path(__file__).resolve().parents[3]   # /home/ruixin/Dubbing
_INDEX_ROOT = _PROJ_ROOT / "index-tts2"

for _p in [str(_INDEX_ROOT), str(_PROJ_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Mel spectrogram helper (matches indextts.s2mel.modules.audio.mel_spectrogram)
# ---------------------------------------------------------------------------

_mel_basis: dict  = {}
_hann_window: dict = {}

def _mel_spectrogram(
    wav: torch.Tensor,
    n_fft: int = 1024,
    num_mels: int = 80,
    sr: int = 22050,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax=None,
    center: bool = False,
) -> torch.Tensor:
    """GPU/CPU log-mel spectrogram identical to indextts audio.mel_spectrogram."""
    import librosa

    global _mel_basis, _hann_window  # noqa: PLW0603

    cache_key = f"{sr}_{fmax}_{wav.device}"
    if cache_key not in _mel_basis:
        mel_fb = torch.from_numpy(
            librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        ).float().to(wav.device)
        _mel_basis[cache_key]  = mel_fb
        _hann_window[f"{sr}_{wav.device}"] = torch.hann_window(win_size).to(wav.device)

    pad = (n_fft - hop_size) // 2
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav_p = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            wav_p, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
            window=_hann_window[f"{sr}_{wav.device}"],
            center=center, normalized=False, onesided=True, return_complex=True,
        )
    )  # [..., n_fft//2+1, T, 2]
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)   # [..., n_fft//2+1, T]
    spec = torch.matmul(_mel_basis[cache_key], spec)  # [..., num_mels, T]
    return torch.log(torch.clamp(spec, min=1e-5))


class Exp_CFM_Index_Phase1_TrainExpand(Exp_Basic):
    """Full train / test experiment for IndexTTS2-CFM fine-tuning."""

    def __init__(self, args):
        self.best_ckpt_path = None
        self._bigvgan = None
        super().__init__(args)
        self._load_frozen_models()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self):
        from modules.cfm_index.flow_matching import CFM

        model = CFM(self.args.model)
        if self.args.system.use_multi_gpu and self.args.system.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.system.device_ids)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"CFM (cfm_index) parameters: {num_params:,}")
        # Pre-allocate KV-cache; use model directly if wrapped in DataParallel
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        core.estimator.setup_caches(max_batch_size=8, max_seq_length=8192)
        return model

    # ------------------------------------------------------------------
    # Frozen sub-models (w2v-bert, codec, campplus, length_regulator)
    # ------------------------------------------------------------------

    def _load_frozen_models(self):
        """Load IndexTTS2 sub-models needed to build CFM conditions.

        All models are frozen (eval mode, no-grad). They are loaded from
        ``args.preprocess.model_dir``.
        """
        from omegaconf import OmegaConf
        from transformers import SeamlessM4TFeatureExtractor
        from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        from huggingface_hub import hf_hub_download
        import safetensors

        pre  = self.args.preprocess
        mdl  = Path(pre.model_dir)
        dev  = self.device

        # Load IndexTTS2 config to construct length_regulator
        idx_cfg = OmegaConf.load(str(mdl / "config.yaml"))

        logger.info("[FrozenModels] Loading SeamlessM4TFeatureExtractor …")
        self._feat_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        logger.info("[FrozenModels] Loading w2v-bert-2.0 semantic model …")
        stat_path = str(mdl / pre.w2v_stat)
        self._sem_model, self._sem_mean, self._sem_std = build_semantic_model(stat_path)
        self._sem_model = self._sem_model.to(dev).eval()
        self._sem_mean  = self._sem_mean.to(dev)
        self._sem_std   = self._sem_std.to(dev)

        logger.info("[FrozenModels] Loading semantic_codec (RepCodec) …")
        codec = build_semantic_codec(idx_cfg.semantic_codec)
        ckpt_path = hf_hub_download(
            "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
        )
        safetensors.torch.load_model(codec, ckpt_path)
        self._sem_codec = codec.to(dev).eval()

        logger.info("[FrozenModels] Loading CAMPPlus …")
        campplus_ckpt = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
        self._campplus = campplus.to(dev).eval()

        logger.info("[FrozenModels] Loading length_regulator from s2mel.pth …")
        s2mel_path = str(mdl / pre.s2mel_checkpoint)
        s2mel_model = MyModel(idx_cfg.s2mel, use_gpt_latent=False)
        s2mel_model, _, _, _ = load_checkpoint2(
            s2mel_model, None, s2mel_path,
            load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        self._len_reg = s2mel_model.models["length_regulator"].to(dev).eval()

        self._mel_ratio = float(pre.mel_ratio)   # ≈ 1.7227
        logger.info("[FrozenModels] All frozen sub-models loaded.")

    # ------------------------------------------------------------------
    # Per-sample condition builder (no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_sem_emb(self, input_features, attention_mask) -> torch.Tensor:
        out = self._sem_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[17]   # [B, T, 1024]
        return (feat - self._sem_mean) / self._sem_std

    @torch.no_grad()
    def _build_conditions(self, batch: Dict) -> tuple:
        """Build CFM conditions for a full batch.

        For each sample a reference mel + semantic condition are computed from
        the reference audio, then concatenated with the semantic condition of
        the target (s_infer) to form cat_condition.

        Returns
        -------
        x1_full   : [B, 80, T_max_total]   ref_mel ++ target_mel (padded)
        cond      : [B, T_max_total, 512]  cat_condition          (padded)
        style     : [B, 192]               CAMPPlus embeddings
        x_lens    : [B]  LongTensor        total (ref+gen) frame counts
        prompt_lens: [B] LongTensor        ref frame counts
        """
        dev = self.device
        B   = len(batch["stems"])

        s_infer      = batch["s_infer"].to(dev)           # [B, T_codes, 1024]
        s_infer_lens = batch["s_infer_lens"]               # [B]
        ref_22k      = batch["ref_audio_22k"]              # [B, T_22k]  CPU
        ref_22k_lens = batch["ref_audio_22k_lens"]         # [B]
        ref_16k      = batch["ref_audio_16k"]              # [B, T_16k]  CPU
        ref_16k_lens = batch["ref_audio_16k_lens"]         # [B]
        x1_mel       = batch["x1_mel"].to(dev)             # [B, 80, T_gen]
        x1_lens      = batch["x1_lens"]                    # [B]

        x1_full_list  : List[torch.Tensor] = []   # each [80, T_total_i]
        cond_list     : List[torch.Tensor] = []   # each [T_total_i, 512]
        style_list    : List[torch.Tensor] = []   # each [1, 192]
        x_lens_list   : List[int]          = []
        prompt_lens_list: List[int]        = []

        for i in range(B):
            tc  = s_infer_lens[i].item()
            t22 = ref_22k_lens[i].item()
            t16 = ref_16k_lens[i].item()
            tg  = x1_lens[i].item()

            # ---- ref_mel [1, 80, T_ref] ---------------------------------
            wav_22 = ref_22k[i, :t22].unsqueeze(0).to(dev)
            ref_mel = _mel_spectrogram(wav_22)        # [1, 80, T_ref]
            T_ref   = ref_mel.size(-1)

            # ---- style (CAMPPlus) [1, 192] ------------------------------
            wav_16 = ref_16k[i, :t16].unsqueeze(0).to(dev)
            fbank  = torchaudio.compliance.kaldi.fbank(
                wav_16, num_mel_bins=80, dither=0, sample_frequency=16000
            )
            fbank  = fbank - fbank.mean(dim=0, keepdim=True)
            style_i = self._campplus(fbank.unsqueeze(0))   # [1, 192]
            style_list.append(style_i)

            # ---- w2v-bert → semantic_codec → S_ref ----------------------
            inputs = self._feat_extractor(
                ref_16k[i, :t16].numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            )
            feat_in = inputs["input_features"].to(dev)
            attn_m  = inputs["attention_mask"].to(dev)
            spk_emb = self._get_sem_emb(feat_in, attn_m)     # [1, T_feat, 1024]
            _, S_ref = self._sem_codec.quantize(spk_emb)

            # ---- prompt_cond [1, T_ref, 512] ----------------------------
            ylens_ref  = torch.LongTensor([T_ref]).to(dev)
            prompt_cond = self._len_reg(
                S_ref, ylens=ylens_ref, n_quantizers=3, f0=None
            )[0]  # [1, T_ref, 512]

            # ---- infer_cond [1, T_gen, 512] (use actual target length) --
            s_inf_i    = s_infer[i, :tc, :].unsqueeze(0)    # [1, T_codes, 1024]
            ylens_gen  = torch.LongTensor([tg]).to(dev)
            infer_cond = self._len_reg(
                s_inf_i, ylens=ylens_gen, n_quantizers=3, f0=None
            )[0]  # [1, T_gen, 512]

            # ---- concatenate -------------------------------------------
            cat_cond = torch.cat([prompt_cond, infer_cond], dim=1)   # [1, T_total, 512]
            T_total  = T_ref + tg

            # ---- x1 full (ref_mel ++ target_mel) [80, T_total] ----------
            x1_gen_i = x1_mel[i, :, :tg]                  # [80, T_gen]
            x1_full_i = torch.cat(
                [ref_mel.squeeze(0), x1_gen_i], dim=-1
            )  # [80, T_total]

            x1_full_list.append(x1_full_i)
            cond_list.append(cat_cond.squeeze(0))   # [T_total, 512]
            x_lens_list.append(T_total)
            prompt_lens_list.append(T_ref)

        # ---- Pad to batch max lengths -----------------------------------
        T_max  = max(x_lens_list)
        n_mels = x1_full_list[0].size(0)

        x1_full = torch.zeros(B, n_mels, T_max, device=dev)
        cond    = torch.zeros(B, T_max, 512, device=dev)
        style   = torch.cat(style_list, dim=0)              # [B, 192]

        for i in range(B):
            T = x_lens_list[i]
            x1_full[i, :, :T] = x1_full_list[i][:, :T]
            cond[i, :T, :]    = cond_list[i][:T, :]

        x_lens      = torch.tensor(x_lens_list,    dtype=torch.long, device=dev)
        prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device=dev)

        return x1_full, cond, style, x_lens, prompt_lens

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_data(self, flag: str):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # ------------------------------------------------------------------
    # One epoch (train or eval)
    # ------------------------------------------------------------------

    def _run_one_epoch(self, loader, train: bool, stage: str) -> float:
        total_loss = 0.0
        count      = 0

        core = (
            self.model.module
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )

        if train:
            self.model.train()
        else:
            self.model.eval()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            progress = tqdm(loader, desc=stage, leave=False, dynamic_ncols=True)
            for batch in progress:
                # Build CFM conditions (frozen models, always no_grad)
                try:
                    x1, cond, style, x_lens, prompt_lens = self._build_conditions(batch)
                except Exception as e:
                    logger.warning(f"[{stage}] condition build failed: {e}")
                    continue

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with (torch.enable_grad() if train else torch.no_grad()):
                    loss, _ = core(
                        x1=x1,
                        x_lens=x_lens,
                        prompt_lens=prompt_lens,
                        cond=cond,
                        style=style,
                    )

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.training.max_grad_norm
                    )
                    self.optimizer.step()

                total_loss += float(loss.item())
                count      += 1
                avg = total_loss / count
                progress.set_postfix(
                    loss=f"{float(loss.item()):.6f}", avg=f"{avg:.6f}"
                )

        return total_loss / max(count, 1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, setting: str):
        train_data, train_loader = self._get_data("train")
        val_data,   val_loader   = self._get_data("val")
        test_data,  test_loader  = self._get_data("test")

        os.makedirs(self.args.system.checkpoints, exist_ok=True)
        ckpt_dir = os.path.join(self.args.system.checkpoints, setting)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
        self._save_args(ckpt_dir)

        t = self.args.training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=t.learning_rate, weight_decay=t.weight_decay,
        )
        self.scheduler = self._build_scheduler(self.optimizer)

        best_val          = float("inf")
        stale_epochs      = 0
        early_stop_pat    = t.early_stop_patience
        self.epoch_logs   = []

        logger.info(
            f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}"
        )

        for epoch in range(1, t.epochs + 1):
            t0 = time.time()
            train_loss = self._run_one_epoch(
                train_loader, train=True,  stage=f"Train {epoch}/{t.epochs}"
            )
            val_loss = self._run_one_epoch(
                val_loader,   train=False, stage=f"Val   {epoch}/{t.epochs}"
            )

            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            lr_tag = (
                f" LR={cur_lr:.2e}→{new_lr:.2e}"
                if new_lr != cur_lr else f" LR={cur_lr:.2e}"
            )
            logger.info(
                f"Epoch {epoch}/{t.epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} |"
                f"{lr_tag} | {time.time()-t0:.1f}s"
            )
            self.epoch_logs.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "lr": cur_lr,
                "time": round(time.time() - t0, 1),
            })

            if val_loss < best_val:
                best_val     = val_loss
                stale_epochs = 0
                torch.save(
                    {"model": self.model.state_dict(), "args": self.args},
                    self.best_ckpt_path,
                )
                logger.info(f"  → best checkpoint saved: {self.best_ckpt_path}")
            else:
                stale_epochs += 1

            if stale_epochs >= early_stop_pat:
                logger.warning(
                    f"Early stop at epoch {epoch} "
                    f"(no improvement for {early_stop_pat} epochs)"
                )
                break

            if epoch % 5 == 0 and os.path.exists(self.best_ckpt_path):
                logger.info(f"Periodic test at epoch {epoch} …")
                self.test(setting, epoch=epoch)
                self._infer_and_save(
                    loader=train_loader,
                    output_dir=os.path.join(ckpt_dir, f"train_outputs@train_{epoch}"),
                    stage_name="TrainInfer",
                    max_batches=t.train_infer_max_batches,
                )

        return self.model

    # ------------------------------------------------------------------
    # Test / inference
    # ------------------------------------------------------------------

    def test(self, setting: str, test: int = 0, epoch: int = 0) -> float:
        _, test_loader = self._get_data("test")

        ckpt_dir       = os.path.join(self.args.system.checkpoints, setting)
        best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
        if os.path.exists(best_ckpt_path):
            state = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"], strict=False)
            logger.info(f"Loaded checkpoint: {best_ckpt_path}")

        test_loss = self._run_one_epoch(test_loader, train=False, stage="Test")
        logger.info(f"Test loss: {test_loss:.6f}")

        if hasattr(self, "epoch_logs") and self.epoch_logs:
            self._save_training_log(ckpt_dir, self.epoch_logs)

        output_dir = os.path.join(ckpt_dir, f"test_outputs@test{test}_{epoch}")
        self._infer_and_save(
            loader=test_loader,
            output_dir=output_dir,
            stage_name="Infer+Save",
            max_batches=self.args.training.test_infer_max_batches,
        )
        logger.info(f"Test outputs saved to: {output_dir}")
        return test_loss

    # ------------------------------------------------------------------
    # Inference + audio saving
    # ------------------------------------------------------------------

    def _get_bigvgan(self):
        if self._bigvgan is None:
            from indextts.s2mel.modules.bigvgan import bigvgan as _bigvgan_mod
            from omegaconf import OmegaConf

            mdl_dir = Path(self.args.preprocess.model_dir)
            idx_cfg = OmegaConf.load(str(mdl_dir / "config.yaml"))
            bvg_name = idx_cfg.vocoder.name   # nvidia/bigvgan_v2_22khz_80band_256x

            logger.info(f"[BigVGAN] Loading {bvg_name} …")
            m = _bigvgan_mod.BigVGAN.from_pretrained(bvg_name, use_cuda_kernel=False)
            m = m.to(self.device)
            m.remove_weight_norm()
            m.eval()
            self._bigvgan = m
            logger.info("[BigVGAN] Loaded.")
        return self._bigvgan

    def _infer_and_save(
        self,
        loader,
        output_dir: str,
        stage_name: str,
        max_batches: int = 8,
    ):
        os.makedirs(output_dir, exist_ok=True)
        vocoder = self._get_bigvgan()

        core = (
            self.model.module
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )
        core.eval()

        tr = self.args.training
        progress = tqdm(loader, desc=stage_name, dynamic_ncols=True)
        with torch.no_grad():
            for j, batch in enumerate(progress):
                if j >= max_batches:
                    break

                try:
                    x1, cond, style, x_lens, prompt_lens = self._build_conditions(batch)
                except Exception as e:
                    logger.warning(f"[{stage_name}] condition build failed: {e}")
                    continue

                stems = batch["stems"]
                B = x1.size(0)

                # Build ref_mel per sample for passing to CFM.inference
                ref_mels: List[torch.Tensor] = []
                for i in range(B):
                    t22 = batch["ref_audio_22k_lens"][i].item()
                    w22 = batch["ref_audio_22k"][i, :t22].unsqueeze(0).to(self.device)
                    ref_mels.append(_mel_spectrogram(w22))   # [1, 80, T_ref_i]

                for i in range(B):
                    ref_mel_i  = ref_mels[i]               # [1, 80, T_ref_i]
                    T_ref_i    = ref_mel_i.size(-1)
                    T_total_i  = int(x_lens[i].item())
                    cond_i     = cond[i:i+1, :T_total_i, :]  # [1, T_total, 512]
                    xl_i       = x_lens[i:i+1]
                    style_i    = style[i:i+1]                 # [1, 192]

                    pred_full = core.inference(
                        mu=cond_i,
                        x_lens=xl_i,
                        prompt=ref_mel_i,
                        style=style_i,
                        f0=None,
                        n_timesteps=tr.inference_steps,
                        inference_cfg_rate=tr.inference_cfg_rate,
                    )  # [1, 80, T_total]

                    pred_mel = pred_full[:, :, T_ref_i:]  # [1, 80, T_gen]
                    x1_gt    = x1[i:i+1, :, T_ref_i: T_total_i]  # [1, 80, T_gen]

                    # Save predicted and ground-truth audio
                    safe_key = str(stems[i]).replace("/", "_").replace(" ", "_")
                    for tag, mel_t in [("pred", pred_mel), ("ref", x1_gt)]:
                        wav = vocoder(mel_t.float()).squeeze(1)  # [1, T_wav]
                        wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu().to(torch.int16)
                        out_path = os.path.join(output_dir, f"{safe_key}_{tag}.wav")
                        torchaudio.save(out_path, wav, 22050)

        logger.info(f"[{stage_name}] {min(j+1, max_batches)} batches saved to {output_dir}")
