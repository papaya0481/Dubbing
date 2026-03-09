"""WaveNet residual stack used as the final decoder in IndexTTS2's S2Mel DiT.

Ported from indextts/s2mel/modules/wavenet.py.
External dependencies on indextts are removed; uses local .encodec.SConv1d.
Weight keys are unchanged to preserve checkpoint compatibility.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from .encodec import SConv1d


# ---------------------------------------------------------------------------
# Utility: fused tanh-sigmoid gate (JIT-scripted for speed)
# ---------------------------------------------------------------------------

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor,
) -> torch.Tensor:
    n_ch = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_ch, :])
    s_act = torch.sigmoid(in_act[:, n_ch:, :])
    return t_act * s_act


# ---------------------------------------------------------------------------
# LayerNorm over channels (for 1-D conv layout [B, C, T])
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# ---------------------------------------------------------------------------
# WaveNet residual stack
# ---------------------------------------------------------------------------

class WN(nn.Module):
    """Dilated WaveNet stack with gated activations and optional global conditioning.

    Args:
        hidden_channels: Feature dimension.
        kernel_size: Convolution kernel size (must be odd).
        dilation_rate: Base dilation (dilation at layer i = dilation_rate ** i).
        n_layers: Number of residual layers.
        gin_channels: Dimension of the global conditioning vector (0 = disabled).
        p_dropout: Dropout probability.
        causal: If True, use causal convolutions.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = SConv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1, norm="weight_norm"
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = SConv1d(
                hidden_channels, 2 * hidden_channels, kernel_size,
                dilation=dilation, padding=padding, norm="weight_norm", causal=causal,
            )
            self.in_layers.append(in_layer)

            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            res_skip_layer = SConv1d(
                hidden_channels, res_skip_channels, 1, norm="weight_norm", causal=causal,
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_ch = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = (
                g[:, i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels, :]
                if g is not None
                else torch.zeros_like(x_in)
            )
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_ch)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            nn.utils.remove_weight_norm(layer)
