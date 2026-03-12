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

        # Build prompt tensor: reference occupies the first prompt_len frames.
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0.0
        if self.zero_prompt_speech_token:
            cond = cond.clone()
            cond[..., :prompt_len] = 0.0

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
                    torch.cat([cond, torch.zeros_like(cond)], dim=0),
                )
                dphi_cond, dphi_uncond = dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_cond - inference_cfg_rate * dphi_uncond
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, cond)

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
            t.squeeze(1).squeeze(1), style, cond,
        )
        valid_mask_mel = valid_mask.unsqueeze(1).expand_as(estimator_out)  # [B, 80, T]
        loss = self.criterion(estimator_out[valid_mask_mel], u[valid_mask_mel])

        return loss, estimator_out + (1 - self.sigma_min) * z


class LipsCrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention: Q from infer_cond, K/V from lips_feat.

    Uses the Attention module from .transformer (supports cross-attention via
    ``is_cross_attention=True``) with RMSNorm pre-normalisation and a
    residual connection: output = query + attn(norm(query), norm(context)).
    """

    def __init__(self, dim: int = 512, n_head: int = 8, context_dim: int = 512):
        super().__init__()
        from .transformer import (
            ModelArgs,
            Attention as _TransAttn,
            RMSNorm,
            precompute_freqs_cis,
        )
        self._precompute_freqs = precompute_freqs_cis
        self._head_dim = dim // n_head
        cfg = ModelArgs(
            dim=dim,
            n_head=n_head,
            n_local_heads=n_head,
            head_dim=dim // n_head,
            context_dim=context_dim,
            has_cross_attention=True,
        )
        self.norm_q  = RMSNorm(dim)
        self.norm_kv = RMSNorm(context_dim)
        self.attn    = _TransAttn(cfg, is_cross_attention=True)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_lens: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            query:        [B, T_q, dim]       (infer_cond)
            context:      [B, T_c, ctxt_dim]  (lips_feat)
            query_lens:   [B] valid lengths for Q (None = all valid)
            context_lens: [B] valid lengths for K/V (None = all valid)
        Returns:
            [B, T_q, dim] with residual added
        """
        T_q = query.size(1)
        T_c = context.size(1)
        max_len = max(T_q, T_c)
        freqs = self._precompute_freqs(
            max_len, self._head_dim, dtype=query.dtype
        ).to(query.device)

        # Build boolean attn_mask [B, 1, T_q, T_c]: True = attend, False = ignore.
        # Masks out padding in both Q and K/V sequences.
        if query_lens is not None or context_lens is not None:
            idx_q = torch.arange(T_q, device=query.device)   # [T_q]
            idx_c = torch.arange(T_c, device=query.device)   # [T_c]
            q_mask = (idx_q.unsqueeze(0) < query_lens.unsqueeze(1)
                      if query_lens is not None
                      else torch.ones(query.size(0), T_q, dtype=torch.bool, device=query.device))
            k_mask = (idx_c.unsqueeze(0) < context_lens.unsqueeze(1)
                      if context_lens is not None
                      else torch.ones(query.size(0), T_c, dtype=torch.bool, device=query.device))
            # [B, T_q, T_c] → [B, 1, T_q, T_c]
            attn_mask = (q_mask.unsqueeze(2) & k_mask.unsqueeze(1)).unsqueeze(1)
        else:
            attn_mask = None

        attn_out = self.attn(
            x=self.norm_q(query),
            freqs_cis=freqs[:T_q],
            mask=attn_mask,
            input_pos=None,
            context=self.norm_kv(context),
            context_freqs_cis=freqs[:T_c],
        )
        out = query + attn_out
        return out


class CrossAttnCFM(CFM):
    """CFM variant with cross-attention to lips_feat before conditioning DiT.

    Adds a LipsCrossAttentionLayer that fuses infer_cond and lips_feat via
    cross-attention (Q from infer_cond, K/V from lips_feat) with a residual
    connection, before concatenating with prompt_cond and passing the
    assembled condition into the DiT estimator.
    """

    def __init__(self, args: Any):
        super().__init__(args)   # builds self.estimator = DiT(cfg.DiT)
        cfg = args if isinstance(args, CFMConfig) else CFMConfig.from_args(args)
        if cfg.dit_type != "DiT":
            raise NotImplementedError(f"Unknown dit_type: {cfg.dit_type!r}")
        content_dim = cfg.DiT.DiT.content_dim
        n_head      = cfg.DiT.DiT.num_heads
        self.lips_cross_attn = LipsCrossAttentionLayer(
            dim=content_dim, n_head=n_head, context_dim=content_dim
        )
    
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
        # ------------------------------------------------------------------
        # 1. Cross-attend: Q from infer_cond, K/V from lips_feat (+residual)
        # ------------------------------------------------------------------
        infer_lens = x_lens - prompt_lens  # [B]
        attended_infer = self.lips_cross_attn(
            infer_cond, lips_feat,
            query_lens=infer_lens,
            context_lens=lips_lens,
        )  # [B, T_infer, 512]

        # ------------------------------------------------------------------
        # 2. Assemble full cond: per-sample [prompt || attended_infer] with
        #    variable lengths, zero-padding to T_mel = x1.size(2)
        # ------------------------------------------------------------------
        B      = x1.size(0)
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

        estimator_out = self.estimator(
            xt, prompt, x_lens,
            t.squeeze(1).squeeze(1), style, cond,
        )
        valid_mask_mel = valid_mask.unsqueeze(1).expand_as(estimator_out)
        loss = self.criterion(estimator_out[valid_mask_mel], u[valid_mask_mel])

        return loss, estimator_out + (1 - self.sigma_min) * z
    