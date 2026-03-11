"""Exp_CFM_Index_Phase1_TrainExpand

Training & testing experiment for the IndexTTS2-style CFM (Phase 1).

Condition building is fully delegated to Dataset_CFM_Index_Phase1 (with
file-level caching).  The Exp class only trains / evaluates the CFM model
using pre-computed batch tensors supplied by the DataLoader:

  batch keys (from collate_cfm_index_phase1):
    x1_full     [B, num_mels, T_max]   ref_mel ++ target_mel (padded)
    cond        [B, T_max, 512]        prompt_cond ++ infer_cond (padded)
    ref_mels    [B, num_mels, T_ref_max]  reference mel for inference
    style       [B, 192]
    x_lens      [B]  total frame counts (ref + gen)
    prompt_lens [B]  ref frame counts
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from tqdm.auto import tqdm

from exp.basic import Exp_Basic
from data_provider.data_factory import data_provider
from logger import get_logger


def _assemble_cond(
    prompt_cond: torch.Tensor,
    infer_cond: torch.Tensor,
    prompt_lens: torch.Tensor,
    infer_lens: torch.Tensor,
) -> torch.Tensor:
    """Assemble full condition tensor from separate prompt / infer parts.

    Parameters
    ----------
    prompt_cond : [B, T_ref_max, 512]  (zero-padded)
    infer_cond  : [B, T_gen_max, 512]  (zero-padded)
    prompt_lens : [B]  actual T_ref per sample
    infer_lens  : [B]  actual T_gen per sample

    Returns
    -------
    cond : [B, max(T_ref + T_gen), 512]  zero-padded, on same device as inputs
    """
    B      = prompt_cond.size(0)
    device = prompt_cond.device
    T_max  = int((prompt_lens + infer_lens).max())
    cond   = torch.zeros(B, T_max, 512, device=device, dtype=prompt_cond.dtype)
    for i in range(B):
        T_r = int(prompt_lens[i])
        T_g = int(infer_lens[i])
        cond[i, :T_r, :]      = prompt_cond[i, :T_r, :]
        cond[i, T_r:T_r+T_g, :] = infer_cond[i, :T_g, :]
    return cond

logger = get_logger("dubbing.exp.cfm_index")

# ---------------------------------------------------------------------------
# Ensure index-tts2 is importable (mirrors test_cfm_index.py setup)
# ---------------------------------------------------------------------------
_PROJ_ROOT  = Path(__file__).resolve().parents[3]   # /home/ruixin/Dubbing
_INDEX_ROOT = _PROJ_ROOT / "index-tts2"

for _p in [str(_INDEX_ROOT), str(_PROJ_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Exp_CFM_Index_Phase1_TrainExpand(Exp_Basic):
    """Full train / test experiment for IndexTTS2-CFM fine-tuning."""

    def __init__(self, args):
        self.best_ckpt_path = None
        self._bigvgan = None
        super().__init__(args)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self):
        from modules.cfm_index.flow_matching import CFM

        model = CFM(self.args.model)

        # Load pre-trained IndexTTS2 s2mel weights (CFM sub-module only)
        s2mel_path = Path(self.args.preprocess.model_dir) / self.args.preprocess.s2mel_checkpoint
        if s2mel_path.exists():
            logger.info(f"[CFM] Loading pre-trained weights from {s2mel_path} …")
            state = torch.load(str(s2mel_path), map_location="cpu", weights_only=False)
            cfm_state = state["net"]["cfm"]
            missing, unexpected = model.load_state_dict(cfm_state, strict=False)
            if missing:
                logger.warning(
                    f"[CFM] Missing keys ({len(missing)}): "
                    f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
                )
            if unexpected:
                logger.warning(
                    f"[CFM] Unexpected keys ({len(unexpected)}): "
                    f"{unexpected[:5]}{'…' if len(unexpected) > 5 else ''}"
                )
            logger.info(f"[CFM] Pre-trained weights loaded (strict=False).")
        else:
            logger.warning(f"[CFM] s2mel checkpoint not found at {s2mel_path}, training from scratch.")

        if self.args.system.use_multi_gpu and self.args.system.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.system.device_ids)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"CFM (cfm_index) parameters: {num_params:,}")
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        core.estimator.setup_caches(max_batch_size=8, max_seq_length=8192)
        return model

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
                dev = self.device
                x1          = batch["x1_full"].to(dev)
                style       = batch["style"].to(dev)
                x_lens      = batch["x_lens"].to(dev)
                prompt_lens = batch["prompt_lens"].to(dev)
                infer_lens  = batch["infer_lens"].to(dev)
                cond        = _assemble_cond(
                    batch["prompt_cond"].to(dev),
                    batch["infer_cond"].to(dev),
                    prompt_lens, infer_lens,
                )

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

                dev         = self.device
                x1          = batch["x1_full"].to(dev)              # [B, num_mels, T_max]
                style       = batch["style"].to(dev)                # [B, 192]
                x_lens      = batch["x_lens"].to(dev)               # [B]
                prompt_lens = batch["prompt_lens"].to(dev)          # [B]
                infer_lens  = batch["infer_lens"].to(dev)           # [B]
                ref_mels    = batch["ref_mels"].to(dev)             # [B, num_mels, T_ref_max]
                cond        = _assemble_cond(
                    batch["prompt_cond"].to(dev),
                    batch["infer_cond"].to(dev),
                    prompt_lens, infer_lens,
                )                                                    # [B, T_max, 512]
                stems       = batch["stems"]
                B = x1.size(0)

                for i in range(B):
                    T_ref_i   = int(prompt_lens[i].item())
                    T_total_i = int(x_lens[i].item())
                    ref_mel_i = ref_mels[i:i+1, :, :T_ref_i]         # [1, num_mels, T_ref]
                    cond_i    = cond[i:i+1, :T_total_i, :]            # [1, T_total, 512]
                    style_i   = style[i:i+1]                          # [1, 192]

                    pred_full = core.inference(
                        cond=cond_i,
                        x_lens=x_lens[i:i+1],
                        prompt=ref_mel_i,
                        style=style_i,
                        f0=None,
                        n_timesteps=tr.inference_steps,
                        inference_cfg_rate=tr.inference_cfg_rate,
                    )  # [1, num_mels, T_total]

                    pred_mel = pred_full[:, :, T_ref_i:]               # [1, num_mels, T_gen]
                    x1_gt    = x1[i:i+1, :, T_ref_i:T_total_i]       # [1, num_mels, T_gen]

                    safe_key = str(stems[i]).replace("/", "_").replace(" ", "_")
                    for tag, mel_t in [("pred", pred_mel), ("ref", x1_gt)]:
                        wav = vocoder(mel_t.float()).squeeze(1)         # [1, T_wav]
                        wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu().to(torch.int16)
                        torchaudio.save(
                            os.path.join(output_dir, f"{safe_key}_{tag}.wav"), wav, 22050
                        )
