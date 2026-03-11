"""IndexTTS2 Diffusion Transformer (DiT) estimator for S2Mel.

Ported from indextts/s2mel/modules/diffusion_transformer.py.
All references to indextts.* removed; local modules are used instead.
Weight keys are identical to the original to preserve checkpoint compatibility.

Default hyper-parameters (from checkpoints/config.yaml):
    DiT:
        hidden_dim: 512,  num_heads: 8,  depth: 13
        in_channels: 80,  content_dim: 512
        style_condition: true,  final_layer_type: 'wavenet'
        long_skip_connection: true,  uvit_skip_connection: true
        is_causal: false,  time_as_token: false,  style_as_token: false
        class_dropout_prob: 0.1
    wavenet:
        hidden_dim: 512,  num_layers: 8,  kernel_size: 5
        dilation_rate: 1,  p_dropout: 0.2,  style_condition: true
    style_encoder:
        dim: 192
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .transformer import ModelArgs, Transformer, precompute_freqs_cis
from .wavenet import WN


# ---------------------------------------------------------------------------
# Sequence-mask helper  (mirrors indextts.s2mel.modules.commons.sequence_mask)
# ---------------------------------------------------------------------------

def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DiTConfig:
    hidden_dim: int = 512
    num_heads: int = 8
    depth: int = 13
    in_channels: int = 80
    content_dim: int = 512
    content_codebook_size: int = 1024
    content_type: str = "discrete"   # always uses cond_projection in practice
    style_condition: bool = True
    final_layer_type: str = "wavenet"
    long_skip_connection: bool = True
    uvit_skip_connection: bool = True
    is_causal: bool = False
    time_as_token: bool = False
    style_as_token: bool = False
    class_dropout_prob: float = 0.1
    block_size: int = 8192


@dataclass
class WaveNetConfig:
    hidden_dim: int = 512
    num_layers: int = 8
    kernel_size: int = 5
    dilation_rate: int = 1
    p_dropout: float = 0.2
    style_condition: bool = True


@dataclass
class StyleEncoderConfig:
    dim: int = 192


@dataclass
class IndexDiTConfig:
    """Top-level config that groups DiT, WaveNet and style-encoder settings."""
    DiT: DiTConfig = field(default_factory=DiTConfig)
    wavenet: WaveNetConfig = field(default_factory=WaveNetConfig)
    style_encoder: StyleEncoderConfig = field(default_factory=StyleEncoderConfig)

    @classmethod
    def from_args(cls, args: Any) -> "IndexDiTConfig":
        """Build from an OmegaConf / munch config object (e.g. cfg.s2mel)."""
        d = args.DiT
        dit = DiTConfig(
            hidden_dim=d.hidden_dim,
            num_heads=d.num_heads,
            depth=d.depth,
            in_channels=d.in_channels,
            content_dim=d.content_dim,
            content_codebook_size=d.content_codebook_size,
            content_type=getattr(d, "content_type", "discrete"),
            style_condition=getattr(d, "style_condition", True),
            final_layer_type=getattr(d, "final_layer_type", "wavenet"),
            long_skip_connection=getattr(d, "long_skip_connection", True),
            uvit_skip_connection=getattr(d, "uvit_skip_connection", True),
            is_causal=getattr(d, "is_causal", False),
            time_as_token=getattr(d, "time_as_token", False),
            style_as_token=getattr(d, "style_as_token", False),
            class_dropout_prob=getattr(d, "class_dropout_prob", 0.1),
            block_size=getattr(d, "block_size", 8192),
        )
        w = getattr(args, "wavenet", None)
        wn = WaveNetConfig(
            hidden_dim=getattr(w, "hidden_dim", 512) if w else 512,
            num_layers=getattr(w, "num_layers", 8) if w else 8,
            kernel_size=getattr(w, "kernel_size", 5) if w else 5,
            dilation_rate=getattr(w, "dilation_rate", 1) if w else 1,
            p_dropout=getattr(w, "p_dropout", 0.2) if w else 0.2,
            style_condition=getattr(w, "style_condition", True) if w else True,
        )
        se = getattr(args, "style_encoder", None)
        style = StyleEncoderConfig(dim=getattr(se, "dim", 192) if se else 192)
        return cls(DiT=dit, wavenet=wn, style_encoder=style)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → dense embedding via a 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)
        self.scale = 1000

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = self.scale * t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class StyleEmbedder(nn.Module):
    """Projects a style vector into the model's hidden dimension."""

    def __init__(self, input_size: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.embedding_table = nn.Embedding(int(dropout_prob > 0), hidden_size)
        self.style_in = weight_norm(nn.Linear(input_size, hidden_size, bias=True))
        self.input_size = input_size
        self.dropout_prob = dropout_prob

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            return labels  # token-drop path (not used in inference)
        return self.style_in(labels)


class FinalLayer(nn.Module):
    """AdaLN-zero final projection (used inside the WaveNet decoder path)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# DiT  (main estimator)
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """Diffusion Transformer estimator for IndexTTS2 S2Mel.

    Accepts the full s2mel config object (OmegaConf / munch) *or* an
    ``IndexDiTConfig`` instance.  Passing the raw config object preserves
    backwards-compatibility with ``MyModel`` / ``CFM`` from the original code.
    """

    def __init__(self, args: Any):
        super().__init__()

        # ---- normalise config ----
        if isinstance(args, IndexDiTConfig):
            cfg = args
        else:
            cfg = IndexDiTConfig.from_args(args)

        d = cfg.DiT
        wn = cfg.wavenet
        se = cfg.style_encoder

        # ---- flags (must match original attribute names for weight compat) ----
        self.time_as_token = d.time_as_token
        self.style_as_token = d.style_as_token
        self.uvit_skip_connection = d.uvit_skip_connection

        # ---- Transformer backbone ----
        model_args = ModelArgs(
            block_size=d.block_size,
            n_layer=d.depth,
            n_head=d.num_heads,
            dim=d.hidden_dim,
            head_dim=d.hidden_dim // d.num_heads,
            vocab_size=1024,
            uvit_skip_connection=d.uvit_skip_connection,
            time_as_token=d.time_as_token,
        )
        self.transformer = Transformer(model_args)

        self.in_channels = d.in_channels
        self.out_channels = d.in_channels
        self.num_heads = d.num_heads

        self.x_embedder = weight_norm(nn.Linear(d.in_channels, d.hidden_dim, bias=True))

        self.content_type = d.content_type
        self.content_codebook_size = d.content_codebook_size
        self.content_dim = d.content_dim
        self.cond_embedder = nn.Embedding(d.content_codebook_size, d.hidden_dim)
        self.cond_projection = nn.Linear(d.content_dim, d.hidden_dim, bias=True)

        self.is_causal = d.is_causal
        self.t_embedder = TimestepEmbedder(d.hidden_dim)

        input_pos = torch.arange(16384)
        self.register_buffer("input_pos", input_pos)

        # ---- final decoder ----
        self.final_layer_type = d.final_layer_type
        if self.final_layer_type == "wavenet":
            self.t_embedder2 = TimestepEmbedder(wn.hidden_dim)
            self.conv1 = nn.Linear(d.hidden_dim, wn.hidden_dim)
            self.conv2 = nn.Conv1d(wn.hidden_dim, d.in_channels, 1)
            self.wavenet = WN(
                hidden_channels=wn.hidden_dim,
                kernel_size=wn.kernel_size,
                dilation_rate=wn.dilation_rate,
                n_layers=wn.num_layers,
                gin_channels=wn.hidden_dim,
                p_dropout=wn.p_dropout,
                causal=False,
            )
            self.final_layer = FinalLayer(wn.hidden_dim, 1, wn.hidden_dim)
            self.res_projection = nn.Linear(d.hidden_dim, wn.hidden_dim)
            self.wavenet_style_condition = wn.style_condition
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(d.hidden_dim, d.hidden_dim),
                nn.SiLU(),
                nn.Linear(d.hidden_dim, d.in_channels),
            )

        self.transformer_style_condition = d.style_condition
        self.class_dropout_prob = d.class_dropout_prob
        self.content_mask_embedder = nn.Embedding(1, d.hidden_dim)
        self.long_skip_connection = d.long_skip_connection
        self.skip_linear = nn.Linear(d.hidden_dim + d.in_channels, d.hidden_dim)

        # Input fusion: [x, prompt_x, cond] + optional style
        cond_x_in_dim = (
            d.hidden_dim + d.in_channels * 2
            + se.dim * int(d.style_condition) * int(not d.style_as_token)
        )
        self.cond_x_merge_linear = nn.Linear(cond_x_in_dim, d.hidden_dim)

        if d.style_as_token:
            self.style_in = nn.Linear(se.dim, d.hidden_dim)

    def setup_caches(self, max_batch_size: int, max_seq_length: int):
        self.transformer.setup_caches(max_batch_size, max_seq_length, use_kv_cache=False)

    def forward(
        self,
        x: torch.Tensor,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: torch.Tensor,
        mask_content: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:        Noisy mel [B, 80, T].
            prompt_x: Reference mel (zero-padded) [B, 80, T].
            x_lens:   Valid lengths [B].
            t:        Timestep [B].
            style:    CAMPPlus speaker embedding [B, 192].
            cond:     GPT semantic tokens projected to content_dim [B, T, 512].
        Returns:
            Predicted velocity field [B, 80, T].
        """
        class_dropout = (self.training and torch.rand(1).item() < self.class_dropout_prob) or mask_content

        B, _, T = x.size()
        t1 = self.t_embedder(t)                              # [B, D]
        cond = self.cond_projection(cond)                    # [B, T, D]

        x_t = x.transpose(1, 2)                              # [B, T, 80]
        prompt_t = prompt_x.transpose(1, 2)                  # [B, T, 80]

        x_in = torch.cat([x_t, prompt_t, cond], dim=-1)     # [B, T, 80+80+D]

        if self.transformer_style_condition and not self.style_as_token:
            x_in = torch.cat([x_in, style[:, None, :].expand(-1, T, -1)], dim=-1)   #[B, T, 80+80+D+style_dim]

        if class_dropout:
            x_in[..., self.in_channels:] = 0.0

        x_in = self.cond_x_merge_linear(x_in)               # [B, T, D]

        if self.style_as_token:
            style_tok = self.style_in(style)
            if class_dropout:
                style_tok = torch.zeros_like(style_tok)
            x_in = torch.cat([style_tok.unsqueeze(1), x_in], dim=1)

        if self.time_as_token:
            x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)

        eff_lens = x_lens + int(self.style_as_token) + int(self.time_as_token)
        x_mask = sequence_mask(eff_lens).to(x.device).unsqueeze(1)   # [B, 1, T']
        input_pos = self.input_pos[: x_in.size(1)]
        x_mask_exp = x_mask[:, None, :].expand(-1, 1, x_in.size(1), -1) if not self.is_causal else None

        x_res = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_exp)

        if self.time_as_token:
            x_res = x_res[:, 1:]
        if self.style_as_token:
            x_res = x_res[:, 1:]

        if self.long_skip_connection:
            x_res = self.skip_linear(torch.cat([x_res, x_t], dim=-1))

        if self.final_layer_type == "wavenet":
            xw = self.conv1(x_res).transpose(1, 2)           # [B, D, T]
            t2 = self.t_embedder2(t)
            xw = (
                self.wavenet(xw, x_mask, g=t2.unsqueeze(2)).transpose(1, 2)
                + self.res_projection(x_res)
            )
            xw = self.final_layer(xw, t1).transpose(1, 2)   # [B, D, T]
            return self.conv2(xw)                             # [B, 80, T]
        else:
            return self.final_mlp(x_res).transpose(1, 2)     # [B, 80, T]
