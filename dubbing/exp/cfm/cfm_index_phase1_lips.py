"""Exp_CFM_Index_Phase1_Lips

Training & testing experiment for the IndexTTS2-style CFM with cross-attention
to lips features (Phase 1).

This experiment uses CrossAttnCFM model which performs cross-attention between
infer_cond (Q) and lips_feat (K/V) before feeding into the DiT estimator.

Pretrained weights from IndexTTS2 s2mel checkpoint are loaded for existing
modules only. New modules (lips_cross_attn) are initialized with Gaussian
random initialization (except zero-initialized linear layers).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from accelerate import Accelerator
from tqdm.auto import tqdm

from exp.basic import Exp_Basic
from data_provider.data_factory import data_provider
from logger import get_logger


logger = get_logger("dubbing.exp.cfm_index_lips")

# ---------------------------------------------------------------------------
# Ensure index-tts2 is importable
# ---------------------------------------------------------------------------
_PROJ_ROOT  = Path(__file__).resolve().parents[3]
_INDEX_ROOT = _PROJ_ROOT / "index-tts2"

for _p in [str(_INDEX_ROOT), str(_PROJ_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Exp_CFM_Index_Phase1_Lips(Exp_Basic):
    """Full train / test experiment for IndexTTS2-CFM with lips cross-attention."""

    def __init__(self, args):
        self.best_ckpt_path = None
        self._bigvgan = None
        self._model_prepared = False
        self.accelerator = Accelerator()
        super().__init__(args)

    def _acquire_device(self):
        logger.info(f"[Accelerate] device: {self.accelerator.device}")
        return self.accelerator.device

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self):
        from modules.cfm_index.flow_matching import CrossAttnCFM

        model = CrossAttnCFM(self.args.model)

        # Load pre-trained IndexTTS2 s2mel weights (existing modules only)
        s2mel_path = Path(self.args.preprocess.model_dir) / self.args.preprocess.s2mel_checkpoint
        if s2mel_path.exists():
            logger.info(f"[CrossAttnCFM] Loading pre-trained weights from {s2mel_path} …")
            state = torch.load(str(s2mel_path), map_location="cpu", weights_only=False)
            cfm_state = state["net"]["cfm"]
            missing, unexpected = model.load_state_dict(cfm_state, strict=False)
            if missing:
                logger.info(
                    f"[CrossAttnCFM] Missing keys ({len(missing)}): "
                    f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
                )
                logger.info("[CrossAttnCFM] New modules will use Gaussian random init.")
            if unexpected:
                logger.warning(
                    f"[CrossAttnCFM] Unexpected keys ({len(unexpected)}): "
                    f"{unexpected[:5]}{'…' if len(unexpected) > 5 else ''}"
                )
            logger.info(f"[CrossAttnCFM] Pre-trained weights loaded (strict=False).")
        else:
            logger.warning(
                f"[CrossAttnCFM] s2mel checkpoint not found at {s2mel_path}, "
                "training from scratch."
            )

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"CrossAttnCFM parameters: {num_params:,}")
        model.estimator.setup_caches(max_batch_size=8, max_seq_length=8192)
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

        core = self.accelerator.unwrap_model(self.model)

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
                prompt_cond = batch["prompt_cond"].to(dev)
                infer_cond  = batch["infer_cond"].to(dev)
                lips_feat   = batch["lips_hidden_states"].to(dev)
                lips_lens   = batch["lips_lens"].to(dev)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with (torch.enable_grad() if train else torch.no_grad()):
                    loss, _ = core(
                        x1=x1,
                        x_lens=x_lens,
                        prompt_lens=prompt_lens,
                        prompt_cond=prompt_cond,
                        infer_cond=infer_cond,
                        lips_feat=lips_feat,
                        lips_lens=lips_lens,
                        style=style,
                    )

                if train:
                    self.accelerator.backward(loss)
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
        (
            self.model, self.optimizer, self.scheduler, train_loader, val_loader, test_loader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler,
            train_loader, val_loader, test_loader,
        )
        self._model_prepared = True

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
                    {"model": self.accelerator.unwrap_model(self.model).state_dict(), "args": self.args},
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

        if not self._model_prepared:
            self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
            self._model_prepared = True

        ckpt_dir       = os.path.join(self.args.system.checkpoints, setting)
        best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
        if os.path.exists(best_ckpt_path):
            state = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            self.accelerator.unwrap_model(self.model).load_state_dict(state["model"], strict=False)
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
            bvg_name = idx_cfg.vocoder.name

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

        core = self.accelerator.unwrap_model(self.model)
        core.eval()

        tr = self.args.training
        progress = tqdm(loader, desc=stage_name, dynamic_ncols=True)
        with torch.no_grad():
            for j, batch in enumerate(progress):
                if j >= max_batches:
                    break

                dev         = self.device
                x1          = batch["x1_full"].to(dev)
                style       = batch["style"].to(dev)
                x_lens      = batch["x_lens"].to(dev)
                prompt_lens = batch["prompt_lens"].to(dev)
                infer_lens  = batch["infer_lens"].to(dev)
                ref_mels    = batch["ref_mels"].to(dev)
                prompt_cond = batch["prompt_cond"].to(dev)
                infer_cond  = batch["infer_cond"].to(dev)
                stems       = batch["stems"]
                B = x1.size(0)

                for i in range(B):
                    T_ref_i   = int(prompt_lens[i].item())
                    T_total_i = int(x_lens[i].item())
                    ref_mel_i = ref_mels[i:i+1, :, :T_ref_i]

                    # Assemble cond for inference
                    T_g = int(infer_lens[i].item())
                    cond_i = torch.zeros(1, T_total_i, 512, device=dev, dtype=prompt_cond.dtype)
                    cond_i[0, :T_ref_i] = prompt_cond[i, :T_ref_i]
                    cond_i[0, T_ref_i:T_ref_i+T_g] = infer_cond[i, :T_g]

                    style_i = style[i:i+1]

                    pred_full = core.inference(
                        cond=cond_i,
                        x_lens=x_lens[i:i+1],
                        prompt=ref_mel_i,
                        style=style_i,
                        f0=None,
                        n_timesteps=tr.inference_steps,
                        inference_cfg_rate=tr.inference_cfg_rate,
                    )

                    pred_mel = pred_full[:, :, T_ref_i:]
                    x1_gt    = x1[i:i+1, :, T_ref_i:T_total_i]

                    safe_key = str(stems[i]).replace("/", "_").replace(" ", "_")
                    for tag, mel_t in [("pred", pred_mel), ("ref", x1_gt)]:
                        wav = vocoder(mel_t.float()).squeeze(1)
                        wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu().to(torch.int16)
                        torchaudio.save(
                            os.path.join(output_dir, f"{safe_key}_{tag}.wav"), wav, 22050
                        )
