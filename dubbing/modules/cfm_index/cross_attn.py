"""Multi-layer transformer with cross-attention for lips conditioning.

Q comes from query (e.g. infer_cond), K/V come from context (e.g. lips_feat).
Each layer applies: self-attention → cross-attention → FFN.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .transformer import (
    ModelArgs,
    TransformerBlock,
    RMSNorm,
    precompute_freqs_cis,
)


class LipsTransformer(nn.Module):
    """Stack of TransformerBlocks with cross-attention.

    Args:
        dim:               Query/output feature dimension.
        context_dim:       Context (K/V source) feature dimension.
        n_layer:           Number of transformer layers.
        n_head:            Number of attention heads for both self- and cross-attention.
        intermediate_size: FFN hidden size. Defaults to ~8/3 * dim (SwiGLU convention).
        norm_eps:          RMSNorm epsilon.
    """

    def __init__(
        self,
        dim: int = 512,
        context_dim: int = 512,
        n_layer: int = 4,
        n_head: int = 8,
        intermediate_size: Optional[int] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        assert dim % n_head == 0, f"dim ({dim}) must be divisible by n_head ({n_head})"
        cfg = ModelArgs(
            dim=dim,
            n_head=n_head,
            n_local_heads=n_head,
            head_dim=dim // n_head,
            context_dim=context_dim,
            has_cross_attention=True,
            n_layer=n_layer,
            intermediate_size=intermediate_size,
            norm_eps=norm_eps,
        )
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(n_layer))
        self.norm = RMSNorm(dim, eps=norm_eps)
        self._head_dim = dim // n_head

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        query_lens: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            query:        [B, T_q, dim]         query sequence (e.g. infer_cond)
            context:      [B, T_c, context_dim] context sequence (e.g. lips_feat)
            query_lens:   [B] valid lengths for query (None = all valid)
            context_lens: [B] valid lengths for context (None = all valid)

        Returns:
            [B, T_q, dim] transformed query with post-norm applied.
        """
        B, T_q, _ = query.shape
        T_c = context.shape[1]

        freqs = precompute_freqs_cis(
            max(T_q, T_c), self._head_dim, dtype=query.dtype
        ).to(query.device)

        # Self-attention mask: bidirectional, masking out query padding.
        if query_lens is not None:
            idx_q = torch.arange(T_q, device=query.device)
            q_valid = idx_q.unsqueeze(0) < query_lens.unsqueeze(1)  # [B, T_q]
            # [B, 1, T_q, T_q]
            self_attn_mask = (q_valid.unsqueeze(2) & q_valid.unsqueeze(1)).unsqueeze(1)
        else:
            self_attn_mask = None

        # Cross-attention mask: [B, 1, T_q, T_c].
        if query_lens is not None or context_lens is not None:
            idx_q = torch.arange(T_q, device=query.device)
            idx_c = torch.arange(T_c, device=query.device)
            q_valid = (
                idx_q.unsqueeze(0) < query_lens.unsqueeze(1)
                if query_lens is not None
                else torch.ones(B, T_q, dtype=torch.bool, device=query.device)
            )
            k_valid = (
                idx_c.unsqueeze(0) < context_lens.unsqueeze(1)
                if context_lens is not None
                else torch.ones(B, T_c, dtype=torch.bool, device=query.device)
            )
            cross_attn_mask = (q_valid.unsqueeze(2) & k_valid.unsqueeze(1)).unsqueeze(1)
        else:
            cross_attn_mask = None

        x = query
        for layer in self.layers:
            x = layer(
                x=x,
                c=None,
                input_pos=None,
                freqs_cis=freqs[:T_q],
                mask=self_attn_mask,
                context=context,
                context_freqs_cis=freqs[:T_c],
                cross_attention_mask=cross_attn_mask,
            )

        return query + self.norm(x)
