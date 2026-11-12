"""Nested prefix layers (Matryoshka / MatFormer style) for MatSched-Net."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLinear(nn.Module):
    """Nested prefix linear layer: uses [:dout, :din] weight slice."""

    def __init__(self, din_max: int, dout_max: int, bias: bool = True) -> None:
        super().__init__()
        self.din_max = din_max
        self.dout_max = dout_max
        self.weight = nn.Parameter(torch.empty(dout_max, din_max))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dout_max))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.din_max
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, din: int, dout: int) -> torch.Tensor:
        w = self.weight[:dout, :din]
        b = self.bias[:dout] if self.bias is not None else None
        return F.linear(x[..., :din], w, b)


class NRMSNorm(nn.Module):
    """RMSNorm applied strictly over the first `d` channels with sliced affine weight."""

    def __init__(self, dmax: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dmax = dmax
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dmax))

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        xd = x[..., :d]
        rms = torch.rsqrt(xd.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xd * rms * self.weight[:d]


class NAttn(nn.Module):
    """Nested multi-head attention with fixed head_dim=16 and always-allowed null key/value."""

    def __init__(self, dmax: int = 128, head_dim: int = 16) -> None:
        super().__init__()
        self.dmax = dmax
        self.head_dim = head_dim
        self.q = NLinear(dmax, dmax)
        self.k = NLinear(dmax, dmax)
        self.v = NLinear(dmax, dmax)
        self.o = NLinear(dmax, dmax)
        self.null_k = nn.Parameter(torch.zeros(dmax))
        self.null_v = nn.Parameter(torch.zeros(dmax))

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        # x: (..., L, d) -> (..., num_heads, L, head_dim)
        new_shape = (*x.shape[:-1], num_heads, self.head_dim)
        return x.view(new_shape).transpose(-3, -2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., num_heads, L, head_dim) -> (..., L, d)
        x = x.transpose(-3, -2).contiguous()
        return x.view(*x.shape[:-2], -1)

    def forward(
        self,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: torch.Tensor,
        d: int,
        null: bool = True,
    ) -> torch.Tensor:
        num_heads = d // self.head_dim
        if num_heads * self.head_dim != d:
            raise ValueError(f"d={d} must be divisible by head_dim={self.head_dim}")

        q = self.q(xq, d, d)
        k = self.k(xkv, d, d)
        v = self.v(xkv, d, d)

        if null:
            nk = self.null_k[:d].view(*([1] * (k.ndim - 2)), 1, d).expand(*k.shape[:-2], 1, d)
            nv = self.null_v[:d].view(*([1] * (v.ndim - 2)), 1, d).expand(*v.shape[:-2], 1, d)
            k = torch.cat([k, nk], dim=-2)
            v = torch.cat([v, nv], dim=-2)
            true_col = torch.ones((*mask.shape[:-1], 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask, true_col], dim=-1)

        qh = self._split_heads(q, num_heads)
        kh = self._split_heads(k, num_heads)
        vh = self._split_heads(v, num_heads)

        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask.unsqueeze(-3))
        return self.o(self._merge_heads(out), d, d)


class NFFN(nn.Module):
    """Nested feed-forward network with hidden dimension `mult * d`."""

    def __init__(self, dmax: int = 128, mult: int = 4) -> None:
        super().__init__()
        self.mult = mult
        self.fc1 = NLinear(dmax, mult * dmax)
        self.fc2 = NLinear(mult * dmax, dmax)

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        hid = self.mult * d
        return self.fc2(F.gelu(self.fc1(x, d, hid)), hid, d)


class NTriLinear(nn.Module):
    """Prefix-safe projection of (core, mach, pair) to (B, J, M, dout).

    Avoids concatenating [core, mach, pair] along the last dim, which would shift
    channel offsets between levels d=32, 64, 128 and break Matryoshka prefix sharing.
    """

    def __init__(self, dmax: int, dout_max: int) -> None:
        super().__init__()
        self.p_core = NLinear(dmax, dout_max, bias=True)
        self.p_mach = NLinear(dmax, dout_max, bias=False)
        self.p_pair = NLinear(dmax, dout_max, bias=False)

    def forward(
        self,
        core: torch.Tensor,
        mach: torch.Tensor,
        pair: torch.Tensor,
        d: int,
        dout: int,
    ) -> torch.Tensor:
        hc = self.p_core(core, d, dout)[:, :, None, :]
        hm = self.p_mach(mach, d, dout)[:, None, :, :]
        hp = self.p_pair(pair, d, dout)
        return hc + hm + hp
