import torch
import torch.nn as nn
import math
from typing import Any
from .attn import (
    AdaLayerNormZero_Final,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
    sequence_mask,
)


class InputEmbedding(nn.Module):
    """
    Early fusion layer for DiT inputs.

    This module concatenates current state `x`, condition feature `cond`, prior feature `mu`,
    and optional speaker embedding `spks` on the channel dimension, then projects to `dim`.
    
    Here, `cond` is decided by phoneme embedding and lips embedding.
    `mu` is the stretched prior mel-spectrogram feature.
    """

    def __init__(
        self,
        mel_dim: int,
        cond_dim: int,
        mu_dim: int,
        dim: int,
        spk_dim: int | None = None,
    ):
        """
        Args:
            mel_dim (int): Number of mel channels.
            cond_dim (int): Number of condition channels.
            mu_dim (int): Number of prior feature channels.
            dim (int): Hidden size of the DiT backbone.
            spk_dim (int | None): Speaker embedding dimension. If None, speaker condition is disabled.
        """
        super().__init__()
        # Whether to append speaker embedding during fusion.
        self.spk_dim = spk_dim
        # Total concatenated channels: x + cond + mu + optional speaker embedding.
        in_dim = mel_dim + cond_dim + mu_dim + (spk_dim or 0)
        # Project concatenated features to DiT hidden dimension.
        self.proj = nn.Linear(in_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Current state tensor with shape [B, T, mel_dim].
            cond (torch.Tensor): Condition tensor with shape [B, T, mel_dim].
            mu (torch.Tensor): Prior tensor with shape [B, T, mu_dim].
            spks (torch.Tensor | None): Optional speaker embedding with shape [B, spk_dim].

        Returns:
            torch.Tensor: Fused hidden states with shape [B, T, dim].
        """
        # Base branches: current state + condition + prior.
        feats = [x, cond, mu]
        # Optional speaker branch: expand to time dimension before concatenation.
        if self.spk_dim is not None and spks is not None:
            # [B, spk_dim] -> [B, T, spk_dim]
            spk_feat = spks[:, None, :].expand(-1, x.size(1), -1)
            feats.append(spk_feat)
        # [B, T, *] -> [B, T, dim]
        return self.proj(torch.cat(feats, dim=-1))


class RotaryEmbedding(nn.Module):
    """
    RoPE frequency generator.

    Generates rotary positional embedding frequencies for a given sequence length.
    """

    def __init__(self, dim_head: int, theta: float = 10000.0):
        """
        Args:
            dim_head (int): Head dimension used by attention.
            theta (float): RoPE base.
        """
        super().__init__()
        self.dim_head = dim_head
        self.theta = theta

    def forward_from_seq_len(self, seq_len: int, device: torch.device | None = None) -> tuple[torch.Tensor, None]:
        """
        Build rotary embedding tuple from sequence length.

        Args:
            seq_len (int): Sequence length T.
            device (torch.device | None): Target device.

        Returns:
            tuple[torch.Tensor, None]:
                - freqs: [T, dim_head]
                - xpos_scale: None (not used in current implementation)
        """
        freqs = precompute_freqs_cis(dim=self.dim_head, end=seq_len, theta=self.theta)
        if device is not None:
            freqs = freqs.to(device)
        return freqs, None


def add_optional_chunk_mask(
    x: torch.Tensor,
    padding_mask: torch.Tensor,
    use_dynamic_chunk: bool = False,
    use_dynamic_left_chunk: bool = False,
    decoding_chunk_size: int = 0,
    static_chunk_size: int = 0,
    num_decoding_left_chunks: int = -1,
) -> torch.Tensor:
    """
    Build attention mask from padding mask.

    Notes:
        The chunk-related arguments are kept for API compatibility,
        but the current implementation only applies valid-position masking.

    Args:
        x (torch.Tensor): Unused placeholder for API compatibility.
        padding_mask (torch.Tensor): Boolean mask [B, T], True for valid positions.
        use_dynamic_chunk (bool): Reserved argument.
        use_dynamic_left_chunk (bool): Reserved argument.
        decoding_chunk_size (int): Reserved argument.
        static_chunk_size (int): Reserved argument.
        num_decoding_left_chunks (int): Reserved argument.

    Returns:
        torch.Tensor: Attention mask with shape [B, T, T].
    """
    # Reserved arguments are intentionally unused in this minimal implementation.
    del x, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size, static_chunk_size, num_decoding_left_chunks
    # B: batch size, T: sequence length.
    batch, seq_len = padding_mask.shape
    # Valid positions on key axis.
    attn_mask = padding_mask[:, None, :].expand(batch, seq_len, seq_len)
    # Combine with valid positions on query axis.
    attn_mask = attn_mask & padding_mask[:, :, None].expand(batch, seq_len, seq_len)
    return attn_mask


class LipSyncDiT(nn.Module):
    """
    DiT backbone for lip-sync style generation with explicit constructor arguments.

    Pipeline:
        1) Time-step embedding.
        2) Input feature fusion.
        3) Multi-layer DiT blocks with RoPE.
        4) Final AdaLN modulation and mel projection.
    """

    def __init__(
        self,
        args: Any = None,
        *,
        dim=None,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        cond_dim=None,
        mu_dim=None,
        long_skip_connection=False,
        
        # New added arguments for lip-sync specific settings
        phoneme_vocab_size=72,
        lip_dim=512,
        
        # other arguments for API compatibility, not used in current implementation
        spk_dim=None,
        out_channels=None,
        static_chunk_size=50,
        num_decoding_left_chunks=2,
    ):
        """
        Args:
            dim (int): Hidden size of DiT blocks.
            depth (int): Number of DiT blocks.
            heads (int): Number of attention heads.
            dim_head (int): Dimension per attention head.
            dropout (float): Dropout used in attention/FFN.
            ff_mult (int): Expansion factor of FFN hidden layer.
            mel_dim (int): Mel channel dimension.
            cond_dim (int | None): Condition feature dimension. Defaults to `dim`.
            mu_dim (int | None): Prior feature dimension. Defaults to `mel_dim`.
            
            phoneme_vocab_size (int): Vocabulary size for phoneme embedding. Defaults to 8194.
            lip_dim (int): Dimension for lip embedding. Defaults to 512.
            
            long_skip_connection (bool): Whether to enable long skip connection.
            spk_dim (int | None): Speaker embedding dimension.
            out_channels (int | None): Reserved output channels argument.
            static_chunk_size (int): Reserved streaming chunk size.
            num_decoding_left_chunks (int): Reserved left chunk count for decoding.
        """
        super().__init__()

        if args is not None:
            dim = getattr(args, "hidden_dim", getattr(args, "dim", dim))
            depth = getattr(args, "depth", depth)
            heads = getattr(args, "num_heads", getattr(args, "heads", heads))
            dim_head = getattr(args, "dim_head", None)
            if dim_head is None and dim is not None and heads is not None:
                dim_head = dim // heads
            dropout = getattr(args, "dropout", dropout)
            ff_mult = getattr(args, "ff_mult", ff_mult)
            mel_dim = getattr(args, "in_channels", getattr(args, "mel_dim", mel_dim))
            cond_dim = getattr(args, "cond_dim", cond_dim)
            mu_dim = getattr(args, "mu_dim", mu_dim)
            long_skip_connection = getattr(args, "long_skip_connection", long_skip_connection)
            phoneme_vocab_size = getattr(args, "phoneme_vocab_size", phoneme_vocab_size)
            lip_dim = getattr(args, "lip_dim", lip_dim)
            spk_dim = getattr(args, "spk_dim", spk_dim)
            out_channels = getattr(args, "out_channels", out_channels)
            static_chunk_size = getattr(args, "static_chunk_size", static_chunk_size)
            num_decoding_left_chunks = getattr(args, "num_decoding_left_chunks", num_decoding_left_chunks)

        if dim is None:
            raise ValueError("`dim` is required for LipSyncDiT initialization")
        if dim_head is None:
            dim_head = dim // heads

        # -------------------------------------------------------
        # 1. Time and input condition encoders
        # -------------------------------------------------------
        
        # Phoneme embedding (kept for compatibility; you may remove it when external cond is ready).
        self.phoneme_embed = nn.Embedding(phoneme_vocab_size, dim)
        # Lip embedding projection (kept for compatibility; you may remove it when external cond is ready).
        self.lip_proj = nn.Linear(lip_dim, dim)

        # Time-step embedding: t [B] -> [B, dim]
        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        if cond_dim is None:
            cond_dim = dim

        self.mel_dim = mel_dim
        self.mu_dim = mu_dim
        self.cond_dim = cond_dim

        # Input fusion: x/cond/mu/(spk) -> [B, T, dim]
        self.input_embed = InputEmbedding(mel_dim, cond_dim, mu_dim, dim, spk_dim)

        # RoPE generator from sequence length.
        self.rotary_embed = RotaryEmbedding(dim_head)

        # Core structure metadata.
        self.dim = dim
        self.depth = depth

        # -------------------------------------------------------
        # 2. Transformer Backbone
        # -------------------------------------------------------

        # Stack of DiT blocks.
        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        # Optional long skip connection.
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        # -------------------------------------------------------
        # 3. 输出层
        # -------------------------------------------------------

        # Final modulation + projection to mel channels.
        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        # Reserved runtime/streaming settings.
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        """
        
        
        Args:
            x (torch.Tensor): 
                Noisy input state, shape [B, mel_dim, T].
            mask (torch.Tensor): 
                Length tensor [B] or boolean mask [B, T].
            mu (torch.Tensor): P
                Prior feature tensor, shape [B, mu_dim, T]. Here `mu` is the stretched prior mel-spectrogram feature.
            t (torch.Tensor): Diffusion timestep tensor [B] or scalar.
            spks (torch.Tensor | None): Optional speaker embedding [B, spk_dim].
            cond (torch.Tensor | None): Optional condition tensor [B, cond_dim, T]. If None, use `mu`.
            streaming (bool): Whether to use the streaming mask branch.

        Returns:
            torch.Tensor: Output mel prediction with shape [B, mel_dim, T].
        """
        # 1) Convert to Transformer-friendly layout: [B, C, T] -> [B, T, C].
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)

        if x.size(-1) != self.mel_dim:
            raise ValueError(f"Expected x channel dim {self.mel_dim}, but got {x.size(-1)}")
        if mu.size(-1) != self.mu_dim:
            raise ValueError(f"Expected mu channel dim {self.mu_dim}, but got {mu.size(-1)}")

        # Use mu branch when cond is not provided.
        if cond is None:
            if self.cond_dim != self.mu_dim:
                raise ValueError(
                    f"cond is required when cond_dim ({self.cond_dim}) != mu_dim ({self.mu_dim})"
                )
            cond = mu
        else:
            cond = cond.transpose(1, 2)

        if cond.size(-1) != self.cond_dim:
            raise ValueError(f"Expected cond channel dim {self.cond_dim}, but got {cond.size(-1)}")

        # 2) Prepare timestep tensor.
        batch, seq_len = x.shape[0], x.shape[1]
        # Expand scalar timestep for batch inference.
        if t.ndim == 0:
            t = t.repeat(batch)

        # 3) Encode time and fuse conditional features.
        t = self.time_embed(t)
        # x/cond/mu/(spk) -> [B, T, dim]
        x = self.input_embed(x, cond, mu, spks)

        # 4) Build RoPE for current sequence length.
        rope = self.rotary_embed.forward_from_seq_len(seq_len, device=x.device)

        # Cache residual for optional long skip.
        if self.long_skip_connection is not None:
            residual = x

        # 5) Normalize mask format.
        if mask.dim() == 1:
            # Length vector -> boolean validity mask.
            mask = sequence_mask(mask, max_len=seq_len).to(x.device)
        else:
            mask = mask.to(x.device).bool()

        # Build attention mask for DiT blocks.
        if streaming is True:
            attn_mask = add_optional_chunk_mask(
                x,
                mask.bool(),
                False,
                False,
                0,
                self.static_chunk_size,
                self.num_decoding_left_chunks,
            ).unsqueeze(dim=1)
        else:
            attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, 0, -1).unsqueeze(dim=1)

        # 6) Transformer backbone forward.
        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope)

        # 7) Apply optional long skip fusion.
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # 8) Output head.
        # [B, T, dim] -> [B, T, mel_dim] -> [B, mel_dim, T]
        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output
