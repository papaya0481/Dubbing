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
        mu: torch.Tensor,
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
            mu:           Concatenated semantic condition [B, T_ref+T_infer, 512].
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
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.estimator.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self._solve_euler(z, x_lens, prompt, mu, style, t_span, inference_cfg_rate)

    def _solve_euler(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg_rate: float,
    ) -> torch.Tensor:
        """Fixed-step Euler ODE solver with classifier-free guidance."""
        t = t_span[0]
        prompt_len = prompt.size(-1)

        # Build prompt tensor: reference occupies the first prompt_len frames.
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0.0
        if self.zero_prompt_speech_token:
            mu = mu.clone()
            mu[..., :prompt_len] = 0.0

        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Batched CFG: conditional + unconditional in one forward pass.
                dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0),
                    x_lens,
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style)], dim=0),
                    torch.cat([mu, torch.zeros_like(mu)], dim=0),
                )
                dphi_cond, dphi_uncond = dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_cond - inference_cfg_rate * dphi_uncond
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)

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

        estimator_out = self.estimator(
            xt, prompt, x_lens,
            t.squeeze(1).squeeze(1), style, cond, prompt_lens,
        )
        valid_mask_mel = valid_mask.unsqueeze(1).expand_as(estimator_out)  # [B, 80, T]
        loss = self.criterion(estimator_out[valid_mask_mel], u[valid_mask_mel])

        return loss, estimator_out + (1 - self.sigma_min) * z
