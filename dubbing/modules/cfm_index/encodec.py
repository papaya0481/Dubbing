"""Convolutional layers with optional weight-norm / spectral-norm normalisation.

Trimmed from Meta's encodec source: only the parts used by WN are kept.
Compatible with indextts/s2mel/modules/encodec.py weight keys.
"""

import math
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zero",
    value: float = 0.0,
) -> torch.Tensor:
    """F.pad wrapper that handles reflect padding on short inputs gracefully."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]) -> torch.Tensor:
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    if norm == "weight_norm":
        return weight_norm(module)
    if norm == "spectral_norm":
        return spectral_norm(module)
    return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    """Return the companion normalisation module (Identity when not needed)."""
    if norm in ("layer_norm", "time_group_norm"):
        # These norms are not used by WN; raise if accidentally requested.
        raise NotImplementedError(f"Norm '{norm}' is not supported in cfm_index.")
    return nn.Identity()


# ---------------------------------------------------------------------------
# Normalised Conv1d wrapper  (preserves weight keys from original encodec.py)
# ---------------------------------------------------------------------------

class NormConv1d(nn.Module):
    """Conv1d + optional weight/spectral-norm parametrisation."""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# SConv1d  (asymmetric / causal padding Conv1d)
# ---------------------------------------------------------------------------

class SConv1d(nn.Module):
    """Conv1d with built-in asymmetric / causal padding and optional norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "reflect",
        **kwargs,
    ):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                f"SConv1d: stride > 1 and dilation > 1 "
                f"(kernel_size={kernel_size}, stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels, out_channels, kernel_size, stride,
            dilation=dilation, groups=groups, bias=bias,
            causal=causal, norm=norm, norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)
