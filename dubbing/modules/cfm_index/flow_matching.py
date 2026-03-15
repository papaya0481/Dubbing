"""Conditional Flow Matching (CFM) for IndexTTS2 S2Mel.

Ported and refactored from indextts/s2mel/modules/flow_matching.py.
All imports from indextts.* removed; uses local .DiT module.
Weight keys are preserved for full checkpoint compatibility.

Inference entry-point: CFM.inference(...)
Training entry-point:  CFM.forward(...)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .DiT import DiT, IndexDiTConfig
from .cross_attn import LipsTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CFMConfig:
    """Hyper-parameters for the flow-matching sampler.

    Defaults follow index-tts2/checkpoints/config.yaml (s2mel block).
    """
    reg_loss_type: str = "l1"          # "l1" | "l2"
    dit_type: str = "DiT"              # only "DiT" is supported
    DiT: Any = field(default_factory=IndexDiTConfig)

    @classmethod
    def from_args(cls, args: Any) -> "CFMConfig":
        """Build from an OmegaConf / munch config object (e.g. cfg.s2mel)."""
        return cls(
            reg_loss_type=getattr(args, "reg_loss_type", "l1"),
            dit_type=getattr(args, "dit_type", "DiT"),
            DiT=IndexDiTConfig.from_args(args),
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BASECFM(nn.Module, ABC):
    """Abstract CFM base class.  Subclasses must implement ``forward``."""

    sigma_min: float = 1e-6

    def __init__(self, args: Any):
        super().__init__()
        cfg = args if isinstance(args, CFMConfig) else CFMConfig.from_args(args)
        self.criterion = nn.MSELoss() if cfg.reg_loss_type == "l2" else nn.L1Loss()
        self.zero_prompt_speech_token: bool = getattr(
            cfg.DiT.DiT if isinstance(cfg.DiT, IndexDiTConfig) else cfg.DiT,
            "zero_prompt_speech_token",
            False,
        )
        self.estimator: nn.Module = None  # set by subclass

    @torch.inference_mode()
    def inference(
        self,
        cond: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        f0,
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.7,
    ) -> torch.Tensor:
        """Run ODE forward from noise to mel-spectrogram.

        Args:
            cond:         Concatenated semantic condition [B, T_ref+T_infer, 512].
            x_lens:       Total mel lengths [B].
            prompt:       Reference mel [B, 80, T_ref].
            style:        Global speaker embedding [B, 192].
            f0:           Unused (kept for API compatibility).
            n_timesteps:  Number of Euler integration steps.
            temperature:  Noise scale.
            inference_cfg_rate: Classifier-free guidance strength.
        Returns:
            Generated mel [B, 80, T].
        """
        B, T = cond.size(0), cond.size(1)
        z = torch.randn([B, self.estimator.in_channels, T], device=cond.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=cond.device)
        return self._solve_euler(z, x_lens, prompt, cond, style, t_span, inference_cfg_rate)

    def _solve_euler(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        cond: torch.Tensor,
        style: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg_rate: float,
    ) -> torch.Tensor:
        """Fixed-step Euler ODE solver with classifier-free guidance."""
        t = t_span[0]
        prompt_len = prompt.size(-1)
        B = x.size(0)

        # Build prompt tensor: reference occupies the first prompt_len frames.
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0.0
        if self.zero_prompt_speech_token:
            cond = cond.clone()
            cond[..., :prompt_len] = 0.0

        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Batched CFG: conditional + unconditional in one forward pass.
                # For conditional branch: cfg_mask=True (keep condition)
                # For unconditional branch: cfg_mask=False (drop condition)
                cfg_mask_cond = torch.ones(B, device=x.device, dtype=torch.bool)    # [B]
                cfg_mask_uncond = torch.zeros(B, device=x.device, dtype=torch.bool)  # [B]
                cfg_mask_combined = torch.cat([cfg_mask_cond, cfg_mask_uncond], dim=0)  # [2B]

                dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0),
                    x_lens,
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style)], dim=0),
                    torch.cat([cond, torch.zeros_like(cond)], dim=0),
                    cfg_mask=cfg_mask_combined,
                )
                dphi_cond, dphi_uncond = dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_cond - inference_cfg_rate * dphi_uncond
            else:
                cfg_mask = torch.ones(B, device=x.device, dtype=torch.bool)  # keep all conditions
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, cond, cfg_mask=cfg_mask)

            x = x + dt * dphi_dt
            t = t + dt
            x[..., :prompt_len] = 0.0

        return x

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

class CFM(BASECFM):
    """Conditional Flow Matching model used by IndexTTS2 S2Mel.

    Accepts either a raw OmegaConf/munch config (``cfg.s2mel``) for
    backward-compatible instantiation, or a ``CFMConfig`` dataclass.
    """

    def __init__(self, args: Any):
        super().__init__(args)
        cfg = args if isinstance(args, CFMConfig) else CFMConfig.from_args(args)
        if cfg.dit_type != "DiT":
            raise NotImplementedError(f"Unknown dit_type: {cfg.dit_type!r}")
        self.estimator = DiT(cfg.DiT)
        self.cond_drop_rate = getattr(self.estimator, 'class_dropout_prob', 0.1)  # for CFG training

    def forward(
        self,
        x1: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        cond: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass (conditional flow matching loss).

        Args:
            x1:          Ground-truth mel [B, 80, T].
            x_lens:      Valid frame lengths [B].
            prompt_lens: Prompt (reference) frame lengths [B].
            cond:        Semantic condition [B, T, 512].
            style:       Speaker embedding [B, 192].
        Returns:
            (loss, estimator_output + sigma_correction)
        """
        b, _, T = x1.shape
        t = torch.rand([b, 1, 1], device=cond.device, dtype=x1.dtype)
        z = torch.randn_like(x1)

        xt = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # 向量化构建 prompt mask，替代逐样本循环
        # prompt_mask: [B, T]，prompt 区域为 True
        frame_idx = torch.arange(T, device=x1.device)               # [T]
        prompt_mask = frame_idx.unsqueeze(0) < prompt_lens.unsqueeze(1)  # [B, T]
        prompt_mask_mel = prompt_mask.unsqueeze(1)                   # [B, 1, T]
        # valid region: frames in [prompt_lens, x_lens)
        valid_mask = (~prompt_mask) & (frame_idx.unsqueeze(0) < x_lens.unsqueeze(1))  # [B, T]

        prompt = x1 * prompt_mask_mel                                # [B, 80, T]
        xt = xt.masked_fill(prompt_mask_mel, 0.0)
        if self.zero_prompt_speech_token:
            cond = cond.clone()
            cond = cond.masked_fill(prompt_mask.unsqueeze(2), 0.0)  # [B, T, 512]

        # Generate cfg_mask: [B] bool tensor
        # True = keep condition, False = drop condition (for CFG training)
        # Similar to dubbing/modules/cfm/flow_matching.py line 145
        if self.training:
            # Get class_dropout_prob from estimator
            cfg_mask = torch.rand(b, device=x1.device) > self.cond_drop_rate  # [B] bool
        else:
            cfg_mask = torch.ones(b, device=x1.device, dtype=torch.bool)  # inference: keep all

        estimator_out = self.estimator(
            xt, prompt, x_lens,
            t.squeeze(1).squeeze(1), style, cond,
            cfg_mask=cfg_mask,
        )
        valid_mask_mel = valid_mask.unsqueeze(1).expand_as(estimator_out)  # [B, 80, T]
        loss = self.criterion(estimator_out[valid_mask_mel], u[valid_mask_mel])

        return loss, estimator_out + (1 - self.sigma_min) * z


class CrossAttnCFM(CFM):
    """CFM variant with cross-attention to lips_feat before conditioning DiT.

    Uses LipsTransformer (multi-layer transformer with cross-attention) to fuse
    infer_cond and lips_feat. Output is the residual-connected semantic input,
    before concatenating with prompt_cond and passing into the DiT estimator.

    Supports multi-conditional Classifier-Free Guidance with three mutually exclusive dropout modes:
    1. Drop infer_cond (query) in LipsTransformer
    2. Drop lips_feat (context) in LipsTransformer
    3. Drop all conditions in DiT (joint dropout)
    """

    def __init__(self, args: Any):
        super().__init__(args)   # builds self.estimator = DiT(cfg.DiT)
        cfg = args if isinstance(args, CFMConfig) else CFMConfig.from_args(args)
        if cfg.dit_type != "DiT":
            raise NotImplementedError(f"Unknown dit_type: {cfg.dit_type!r}")
        content_dim = cfg.DiT.DiT.content_dim
        n_head      = cfg.DiT.DiT.num_heads
        self.lips_cross_attn = LipsTransformer(
            dim=content_dim, n_head=n_head, context_dim=content_dim
        )

        # Multi-conditional CFG rates (mutually exclusive)
        self.conds_cfg_rate = 0.1   # probability of dropping infer_cond (query)
        self.lips_cfg_rate = 0.1    # probability of dropping lips_feat (context)
        self.dit_cfg_rate = getattr(self.estimator, 'class_dropout_prob', 0.1)  # joint dropout in DiT
    
    def forward(
        self,
        x1: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        prompt_cond: torch.Tensor,
        infer_cond: torch.Tensor,
        lips_feat: torch.Tensor,
        lips_lens: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass (conditional flow matching loss).
        This module will firstly do cross attention between infer_cond and lips_feat.
        Q is from infer_cond, K and V are from lips_feat.
        Then the output of cross attention will be concatenated with prompt_cond on the Time dimension (T=T_prompt+T_infer) and fed into the DiT estimator as the condition.

        Note that a residual connection is added between the output of cross attention and infer_cond.

        Multi-conditional CFG: Three mutually exclusive dropout modes:
        1. Drop infer_cond (conds_cfg_mask=False)
        2. Drop lips_feat (lips_cfg_mask=False)
        3. Drop all conditions in DiT (dit_cfg_mask=False)

        Args:
            x1:          Ground-truth mel [B, 80, T].
            x_lens:      Valid frame lengths [B].
            prompt_lens: Prompt (reference) frame lengths [B].
            prompt_cond: Prompt (reference) semantic condition [B, T_prompt, 512].
            infer_cond: Inference semantic condition [B, T_infer, 512].
            lips_feat: Lip reading features [B, T_lips, 512].
            style:       Speaker embedding [B, 192].
        Returns:
            (loss, estimator_output + sigma_correction)
        """
        B = x1.size(0)

        # ------------------------------------------------------------------
        # Generate mutually exclusive CFG masks (only during training)
        # ------------------------------------------------------------------
        if self.training:
            # Total dropout probability
            total_cfg_rate = self.conds_cfg_rate + self.lips_cfg_rate + self.dit_cfg_rate
            rand_vals = torch.rand(B, device=x1.device)  # [B]

            # Mutually exclusive assignment:
            # [0, conds_cfg_rate) -> drop infer_cond
            # [conds_cfg_rate, conds_cfg_rate + lips_cfg_rate) -> drop lips_feat
            # [conds_cfg_rate + lips_cfg_rate, total_cfg_rate) -> drop all in DiT
            # [total_cfg_rate, 1.0) -> keep all

            conds_cfg_mask = rand_vals >= self.conds_cfg_rate  # False if in [0, conds_cfg_rate)
            lips_cfg_mask = (rand_vals < self.conds_cfg_rate) | (rand_vals >= self.conds_cfg_rate + self.lips_cfg_rate)
            dit_cfg_mask = rand_vals >= (self.conds_cfg_rate + self.lips_cfg_rate)
        else:
            # Inference: keep all conditions
            conds_cfg_mask = torch.ones(B, device=x1.device, dtype=torch.bool)
            lips_cfg_mask = torch.ones(B, device=x1.device, dtype=torch.bool)
            dit_cfg_mask = torch.ones(B, device=x1.device, dtype=torch.bool)

        # ------------------------------------------------------------------
        # 1. Cross-attend: Q from infer_cond, K/V from lips_feat (+residual)
        #    with CFG masking applied inside LipsTransformer
        # ------------------------------------------------------------------
        infer_lens = x_lens - prompt_lens  # [B]
        attended_infer = self.lips_cross_attn(
            infer_cond, lips_feat,
            query_lens=infer_lens,
            context_lens=lips_lens,
            conds_cfg_mask=conds_cfg_mask,
            lips_cfg_mask=lips_cfg_mask,
        )  # [B, T_infer, 512]

        # ------------------------------------------------------------------
        # 2. Assemble full cond: per-sample [prompt || attended_infer] with
        #    variable lengths, zero-padding to T_mel = x1.size(2)
        # ------------------------------------------------------------------
        T_mel  = x1.size(2)
        C      = infer_cond.size(-1)
        cond = torch.zeros(B, T_mel, C, device=x1.device, dtype=infer_cond.dtype)
        for i in range(B):
            T_r = int(prompt_lens[i])
            T_g = int(infer_lens[i])
            cond[i, :T_r]        = prompt_cond[i, :T_r]
            cond[i, T_r:T_r+T_g] = attended_infer[i, :T_g]

        # ------------------------------------------------------------------
        # 3. Flow-matching forward (same logic as CFM.forward)
        # ------------------------------------------------------------------
        b, _, T = x1.shape
        t = torch.rand([b, 1, 1], device=cond.device, dtype=x1.dtype)
        z = torch.randn_like(x1)

        xt = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u  = x1 - (1 - self.sigma_min) * z

        frame_idx = torch.arange(T, device=x1.device)
        prompt_mask = frame_idx.unsqueeze(0) < prompt_lens.unsqueeze(1)   # [B, T]
        prompt_mask_mel = prompt_mask.unsqueeze(1)                             # [B, 1, T]
        valid_mask = (~prompt_mask) & (frame_idx.unsqueeze(0) < x_lens.unsqueeze(1))

        prompt = x1 * prompt_mask_mel
        xt = xt.masked_fill(prompt_mask_mel, 0.0)
        if self.zero_prompt_speech_token:
            cond = cond.masked_fill(prompt_mask.unsqueeze(2), 0.0)

        # Pass dit_cfg_mask to DiT for joint condition dropout
        estimator_out = self.estimator(
            xt, prompt, x_lens,
            t.squeeze(1).squeeze(1), style, cond,
            cfg_mask=dit_cfg_mask,
        )
        valid_mask_mel = valid_mask.unsqueeze(1).expand_as(estimator_out)
        loss = self.criterion(estimator_out[valid_mask_mel], u[valid_mask_mel])

        return loss, estimator_out + (1 - self.sigma_min) * z

    @torch.inference_mode()
    def inference(
        self,
        prompt_cond: torch.Tensor,
        infer_cond: torch.Tensor,
        lips_feat: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        lips_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        f0,
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.7,
    ) -> torch.Tensor:
        """Run ODE forward from noise to mel-spectrogram with multi-conditional CFG.

        Args:
            prompt_cond:  Prompt semantic condition [B, T_ref, 512].
            infer_cond:   Inference semantic condition [B, T_infer, 512].
            lips_feat:    Lip reading features [B, T_lips, 512].
            x_lens:       Total mel lengths [B].
            prompt_lens:  Prompt (reference) frame lengths [B].
            lips_lens:    Lip feature lengths [B].
            prompt:       Reference mel [B, 80, T_ref].
            style:        Global speaker embedding [B, 192].
            f0:           Unused (kept for API compatibility).
            n_timesteps:  Number of Euler integration steps.
            temperature:  Noise scale.
            inference_cfg_rate: Classifier-free guidance strength.
        Returns:
            Generated mel [B, 80, T].
        """
        B = prompt_cond.size(0)
        T = int(x_lens.max())
        z = torch.randn([B, self.estimator.in_channels, T], device=prompt_cond.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=prompt_cond.device)
        return self._solve_euler_multi_cond(
            z, x_lens, prompt_lens, prompt, prompt_cond, infer_cond,
            lips_feat, lips_lens, style, t_span, inference_cfg_rate
        )

    def _solve_euler_multi_cond(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        prompt: torch.Tensor,
        prompt_cond: torch.Tensor,
        infer_cond: torch.Tensor,
        lips_feat: torch.Tensor,
        lips_lens: torch.Tensor,
        style: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg_rate: float,
    ) -> torch.Tensor:
        """Fixed-step Euler ODE solver with multi-conditional classifier-free guidance.

        Multi-conditional CFG uses 3 branches (conds always exist):
        1. Full conditional (all conditions enabled)
        2. Drop lips_feat (lips_cfg_mask=False)
        3. Drop all conditions in DiT (dit_cfg_mask=False)

        Final prediction: pred = (1 + cfg_rate) * pred_full - cfg_rate/2 * (pred_no_lips + pred_no_dit)
        """
        t = t_span[0]
        prompt_len = prompt.size(-1)
        B = x.size(0)

        # Build prompt tensor: reference occupies the first prompt_len frames.
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0.0

        infer_lens = x_lens - prompt_lens  # [B]

        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]

            if inference_cfg_rate > 0:
                # Multi-conditional CFG: 3 branches (conds always present)
                # Branch 1: full conditional (all True)
                # Branch 2: drop lips_feat (lips_cfg_mask=False)
                # Branch 3: drop all in DiT (dit_cfg_mask=False)

                # Replicate inputs 3 times
                x_in = torch.cat([x, x, x], dim=0)  # [3B, C, T]
                prompt_x_in = torch.cat([prompt_x, prompt_x, prompt_x], dim=0)
                t_in = torch.cat([t.unsqueeze(0)] * 3, dim=0)
                style_in = torch.cat([style, style, style], dim=0)

                # Build cfg masks for each branch
                ones_B = torch.ones(B, device=x.device, dtype=torch.bool)
                zeros_B = torch.zeros(B, device=x.device, dtype=torch.bool)

                # Branch 1: conds=True, lips=True, dit=True
                # Branch 2: conds=True, lips=False, dit=True
                # Branch 3: conds=True, lips=True, dit=False
                conds_cfg_mask = torch.cat([ones_B, ones_B, ones_B], dim=0)  # [3B] - always True
                lips_cfg_mask = torch.cat([ones_B, zeros_B, ones_B], dim=0)   # [3B]
                dit_cfg_mask = torch.cat([ones_B, ones_B, zeros_B], dim=0)    # [3B]

                # Replicate infer_cond, lips_feat, lips_lens
                infer_cond_in = torch.cat([infer_cond] * 3, dim=0)
                lips_feat_in = torch.cat([lips_feat] * 3, dim=0)
                infer_lens_in = torch.cat([infer_lens] * 3, dim=0)
                lips_lens_in = torch.cat([lips_lens] * 3, dim=0)
                prompt_cond_in = torch.cat([prompt_cond] * 3, dim=0)

                # Cross-attention with CFG masking
                attended_infer = self.lips_cross_attn(
                    infer_cond_in, lips_feat_in,
                    query_lens=infer_lens_in,
                    context_lens=lips_lens_in,
                    conds_cfg_mask=conds_cfg_mask,
                    lips_cfg_mask=lips_cfg_mask,
                )  # [3B, T_infer, 512]

                # Assemble full cond for all 3 branches
                T_mel = x.size(2)
                C = infer_cond.size(-1)
                cond_in = torch.zeros(3 * B, T_mel, C, device=x.device, dtype=infer_cond.dtype)
                for i in range(3 * B):
                    T_r = int(prompt_lens[i % B])
                    T_g = int(infer_lens[i % B])
                    cond_in[i, :T_r] = prompt_cond_in[i, :T_r]
                    cond_in[i, T_r:T_r+T_g] = attended_infer[i, :T_g]

                if self.zero_prompt_speech_token:
                    cond_in = cond_in.clone()
                    frame_idx = torch.arange(T_mel, device=x.device)
                    prompt_lens_in = torch.cat([prompt_lens] * 3, dim=0)
                    prompt_mask = frame_idx.unsqueeze(0) < prompt_lens_in.unsqueeze(1)
                    cond_in = cond_in.masked_fill(prompt_mask.unsqueeze(2), 0.0)

                # Forward through DiT with dit_cfg_mask
                x_lens_in = torch.cat([x_lens] * 3, dim=0)
                dphi_dt = self.estimator(
                    x_in, prompt_x_in, x_lens_in,
                    t_in, style_in, cond_in,
                    cfg_mask=dit_cfg_mask,
                )

                # Split into 3 branches
                dphi_full, dphi_no_lips, dphi_no_dit = dphi_dt.chunk(3, dim=0)

                # Multi-conditional CFG formula (2 unconditional branches)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_full - (inference_cfg_rate / 2.0) * (
                    dphi_no_lips + dphi_no_dit
                )
            else:
                # No CFG: keep all conditions
                ones_B = torch.ones(B, device=x.device, dtype=torch.bool)

                attended_infer = self.lips_cross_attn(
                    infer_cond, lips_feat,
                    query_lens=infer_lens,
                    context_lens=lips_lens,
                    conds_cfg_mask=ones_B,
                    lips_cfg_mask=ones_B,
                )

                T_mel = x.size(2)
                C = infer_cond.size(-1)
                cond = torch.zeros(B, T_mel, C, device=x.device, dtype=infer_cond.dtype)
                for i in range(B):
                    T_r = int(prompt_lens[i])
                    T_g = int(infer_lens[i])
                    cond[i, :T_r] = prompt_cond[i, :T_r]
                    cond[i, T_r:T_r+T_g] = attended_infer[i, :T_g]

                if self.zero_prompt_speech_token:
                    cond = cond.clone()
                    frame_idx = torch.arange(T_mel, device=x.device)
                    prompt_mask = frame_idx.unsqueeze(0) < prompt_lens.unsqueeze(1)
                    cond = cond.masked_fill(prompt_mask.unsqueeze(2), 0.0)

                dphi_dt = self.estimator(
                    x, prompt_x, x_lens,
                    t.unsqueeze(0), style, cond,
                    cfg_mask=ones_B,
                )

            x = x + dt * dphi_dt
            t = t + dt
            x[..., :prompt_len] = 0.0

        return x
    