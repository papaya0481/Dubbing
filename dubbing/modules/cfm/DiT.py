import torch
import torch.nn as nn
import math
from .attn import (
    AdaLayerNormZero_Final,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
    sequence_mask,
)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, mu_dim: int, dim: int, spk_dim: int | None = None):
        super().__init__()
        self.spk_dim = spk_dim
        in_dim = mel_dim + mel_dim + mu_dim + (spk_dim or 0)
        self.proj = nn.Linear(in_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = [x, cond, mu]
        if self.spk_dim is not None and spks is not None:
            spk_feat = spks[:, None, :].expand(-1, x.size(1), -1)
            feats.append(spk_feat)
        return self.proj(torch.cat(feats, dim=-1))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim_head: int, theta: float = 10000.0):
        super().__init__()
        self.dim_head = dim_head
        self.theta = theta

    def forward_from_seq_len(self, seq_len: int, device: torch.device | None = None) -> tuple[torch.Tensor, None]:
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
    del x, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size, static_chunk_size, num_decoding_left_chunks
    batch, seq_len = padding_mask.shape
    attn_mask = padding_mask[:, None, :].expand(batch, seq_len, seq_len)
    attn_mask = attn_mask & padding_mask[:, :, None].expand(batch, seq_len, seq_len)
    return attn_mask


class LipSyncDiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        mu_dim=None,
        long_skip_connection=False,
        spk_dim=None,
        out_channels=None,
        static_chunk_size=50,
        num_decoding_left_chunks=2,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        if cond is None:
            cond = mu
        else:
            cond = cond.transpose(1, 2)

        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks)

        rope = self.rotary_embed.forward_from_seq_len(seq_len, device=x.device)

        if self.long_skip_connection is not None:
            residual = x

        if mask.dim() == 1:
            mask = sequence_mask(mask, max_len=seq_len).to(x.device)
        else:
            mask = mask.to(x.device).bool()

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

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output
