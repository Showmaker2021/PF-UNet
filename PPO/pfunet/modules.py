from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

Obs = namedtuple('Obs', 'xy feats raw mask_pair const head_idx has_head Ja S alive_m', defaults=(None,))


def build_pf_obs(state, eps: float = 1e-6) -> Obs:
    """
    Fully vectorized Observation Builder converting ReSched 8-tuple state into Priority-Field observation.
    Args:
        state: 8-tuple (min_dur, dur, o_at, m_at, dep_o, dep_m, act_mask, rev_pos)
        eps: small float to prevent division by zero
    Returns:
        Obs namedtuple
    """
    (min_dur, dur, o_at, m_at, dep_o, dep_m, act_mask, rev_pos) = state
    B, N, M = dur.shape

    # (1) Job slots = operation heads, sorted by operation index ascending; remainder padded
    head = act_mask.any(-1)  # [B, N]
    J = int(head.sum(1).max().item())  # max active jobs in batch
    order = torch.argsort((~head).to(torch.int8), dim=1, stable=True)[:, :J]  # [B, J]
    has_head = head.gather(1, order)  # [B, J]
    safe = order  # valid operation indices, including padded slots

    # (2) Remaining ops of job = row of head in dependency_o (D2)
    G = dep_o.gather(1, order[..., None].expand(B, J, N)) & has_head[..., None]  # [B, J, N] bool
    Gf = G.to(dur.dtype)
    W = (Gf * min_dur[:, None, :]).sum(-1)  # [B, J] total remaining min_duration (including head)
    n = Gf.sum(-1)  # [B, J] remaining op count

    # (3) Head operation data
    R = o_at.gather(1, safe)  # [B, J] job ready time
    p_min = min_dur.gather(1, safe)  # [B, J]
    p_hd = dur.gather(1, safe[..., None].expand(B, J, M))  # [B, J, M]
    ok_hd = act_mask.gather(1, safe[..., None].expand(B, J, M)) & has_head[..., None]  # [B, J, M]
    nM_hd = ok_hd.sum(-1).clamp_min(1).to(dur.dtype)  # [B, J] compatible machine count

    # (4) Normalization bounds (ignore disconnected/dead machines whose m_at is unshifted)
    alive_m = dep_m.any(1)  # [B, M] bool
    has_alive_m = alive_m.any(1, keepdim=True)
    alive_m = torch.where(has_alive_m, alive_m, torch.ones_like(alive_m))

    A = m_at  # [B, M]
    t0 = A.masked_fill(~alive_m, float('inf')).min(1).values  # [B]
    Rt = torch.maximum(R, t0[:, None])
    LBj = (Rt + W).masked_fill(~has_head, float('-inf')).max(1).values
    LB = torch.maximum(LBj, A.masked_fill(~alive_m, float('-inf')).max(1).values)
    S = (LB - t0).clamp_min(eps)  # [B]

    # (5) Layout [B, M, J]
    P = p_hd.permute(0, 2, 1)  # [B, M, J]
    OK = ok_hd.permute(0, 2, 1)  # [B, M, J] bool
    ST = torch.maximum(R[:, None, :], A[:, :, None])  # [B, M, J]
    S3 = S[:, None, None]
    Ma = alive_m.sum(1, keepdim=True).to(dur.dtype).clamp_min(1)  # [B, 1]
    dur_ref = dur.masked_fill(~dep_m, 0.0).amax(dim=(1, 2)).clamp_min(eps)[:, None, None]
    x = ((ST - t0[:, None, None]) / S3).clamp(0.0, 1.0)
    y = (W / S[:, None])[:, None, :].expand(B, M, J).clamp(0.0, 1.0)
    wait_gap = (ST - A[:, :, None]).clamp_min(0.0)
    wait = (wait_gap / (wait_gap + P.clamp_min(eps))).clamp(0.0, 1.0)
    prog = (n / n.max(1, keepdim=True).values.clamp_min(1))[:, None, :].expand(B, M, J)  # D3
    flex = (nM_hd / Ma)[:, None, :].expand(B, M, J).clamp(0.0, 1.0)
    p_norm = (P / dur_ref).clamp(0.0, 1.0)
    p_fast = (p_min.clamp_min(eps)[:, None, :] / P.clamp_min(eps)).clamp(0.0, 1.0) * OK.to(P.dtype)

    xy = torch.stack([x, y], -1)  # [B, M, J, 2]
    feats = torch.stack([torch.ones_like(x), p_norm, flex, wait, prog], -1)  # [B, M, J, 5] channels 1..5
    raw = torch.stack([p_norm, p_fast, wait, x, y, flex, prog], -1)  # [B, M, J, 7]

    # (6) Machine constant channels: c6 (machine idle time), c7 (future load relative to active avg)
    nM_all = dep_m.sum(-1).clamp_min(1).to(dur.dtype)  # [B, N]
    L = ((dur / nM_all[..., None]) * (~head)[..., None]).sum(1)  # [B, M] load of future ops
    c6 = ((A - t0[:, None]) / S[:, None]).clamp_min(0.0) * alive_m.to(A.dtype)
    c7 = (Ma * L / (L.sum(1, keepdim=True) + eps)) * alive_m.to(A.dtype)
    const = torch.stack([c6, c7], -1)  # [B, M, 2]

    Ja = has_head.sum(1).clamp_min(1).to(dur.dtype)  # [B]
    return Obs(xy, feats, raw, OK, const, order, has_head, Ja, S, alive_m)


def rasterize(xy, vals, mask, H: int, W: int):
    """
    Bilinear splatting of discrete points onto continuous canvas.
    Args:
        xy: [B, M, J, 2] in [0, 1]
        vals: [B, M, J, C]
        mask: [B, M, J] bool
        H, W: canvas dimensions
    Returns:
        canvas: [B*M, C, H, W]
    """
    B, M, J, C = vals.shape
    gx, gy = xy[..., 0] * (W - 1), xy[..., 1] * (H - 1)
    x0 = gx.floor().clamp(0, W - 1)
    y0 = gy.floor().clamp(0, H - 1)
    fx, fy = gx - x0, gy - y0
    x0, y0 = x0.long(), y0.long()
    x1, y1 = (x0 + 1).clamp(max=W - 1), (y0 + 1).clamp(max=H - 1)
    
    canvas = vals.new_zeros(B * M, H * W, C)
    for xi, yi, w in ((x0, y0, (1 - fx) * (1 - fy)),
                      (x1, y0, fx * (1 - fy)),
                      (x0, y1, (1 - fx) * fy),
                      (x1, y1, fx * fy)):
        idx = (yi * W + xi).reshape(B * M, J)
        contrib = (vals * (w * mask)[..., None]).reshape(B * M, J, C)
        canvas.scatter_add_(1, idx[..., None].expand(-1, -1, C), contrib)
        
    return canvas.view(B * M, H, W, C).permute(0, 3, 1, 2)  # [B*M, C, H, W]


def rasterize_obs(obs: Obs, H: int, W: int):
    """
    Rasterizes observation into 8-channel canvas:
      channels 0..4: bilinear splatted feats (divided by J_a)
      channels 5..6: machine constants c6, c7 broadcast
      channel 7: context channel (machine-average of channel 0)
    """
    B, M, J = obs.mask_pair.shape
    c = rasterize(obs.xy, obs.feats, obs.mask_pair, H, W)  # [B*M, 5, H, W]
    c = c / obs.Ja.repeat_interleave(M)[:, None, None, None]  # divide by J_a
    const = obs.const.reshape(B * M, 2, 1, 1).expand(-1, -1, H, W)  # channels 6, 7
    if obs.alive_m is not None:
        ma = obs.alive_m.sum(1).clamp_min(1).to(c.dtype)[:, None, None, None, None]  # [B, 1, 1, 1, 1]
        ctx = (c[:, :1].view(B, M, 1, H, W) * obs.alive_m.to(c.dtype).view(B, M, 1, 1, 1)).sum(1, keepdim=True) / ma
    else:
        ctx = c[:, :1].view(B, M, 1, H, W).mean(1, keepdim=True)  # channel 8: mean across machines of ch 0
    ctx = ctx.expand(B, M, 1, H, W).reshape(B * M, 1, H, W)
    return torch.cat([c, const, ctx], 1)  # [B*M, 8, H, W]


def readout(maps, xy, B: int, M: int):
    """
    Bilinear grid sampling readout from feature maps.
    Args:
        maps: [B*M, C, h, w]
        xy: [B, M, J, 2] in [0, 1]
    Returns:
        [B, M, J, C]
    """
    J = xy.shape[2]
    grid = xy.reshape(B * M, J, 1, 2) * 2 - 1  # normalize to [-1, 1], order (x, y) = (width, height)
    out = F.grid_sample(maps, grid, mode='bilinear', padding_mode='border', align_corners=True)  # [B*M, C, J, 1]
    return out.squeeze(-1).permute(0, 2, 1).reshape(B, M, J, -1)  # [B, M, J, C]


class ConvBlock(nn.Module):
    """Conv3x3 + GroupNorm + GELU"""
    def __init__(self, c_in: int, c_out: int, groups: int = 4):
        super().__init__()
        g = groups
        while g > 1 and (c_out % g != 0):
            g //= 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(g, c_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    """
    2D Multi-scale U-Net with skip connections.
    """
    def __init__(self, c_in: int = 8, chs=(32, 64, 128, 256), blocks: int = 2, groups: int = 8):
        super().__init__()
        L = len(chs)
        self.enc = nn.ModuleList([
            nn.Sequential(*[
                ConvBlock(c_in if (i == 0 and b == 0) else (chs[i] if b > 0 else chs[i-1]), chs[i], groups)
                for b in range(blocks)
            ]) for i in range(L)
        ])
        self.down = nn.ModuleList([
            nn.Conv2d(chs[i], chs[i], 3, stride=2, padding=1) for i in range(L - 1)
        ])
        self.up = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(chs[i + 1], chs[i], 3, padding=1)
            ) for i in range(L - 1)
        ])
        self.dec = nn.ModuleList([
            nn.Sequential(*[
                ConvBlock(2 * chs[i] if b == 0 else chs[i], chs[i], groups)
                for b in range(blocks)
            ]) for i in range(L - 1)
        ])

    def encode(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.down[i](x)
        return skips, x  # x: bottleneck [B*M, 256, H/8, W/8]

    def decode(self, h, skips):
        outs = []
        for i in reversed(range(len(self.dec))):
            h = self.up[i](h)
            h = self.dec[i](torch.cat([h, skips[i]], 1))
            outs.append(h)
        return outs[::-1]  # [d0 (full, 32), d1 (1/2, 64), ...]


class MachineAttention(nn.Module):
    """
    Permutation-invariant self-attention across machines without positional encodings.
    """
    def __init__(self, dim: int = 256, heads: int = 4, layers: int = 2, ff: int = 512):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, ff, batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, g, alive_m=None):
        pad_mask = (~alive_m) if alive_m is not None else None
        return self.enc(g, src_key_padding_mask=pad_mask)  # [B, M, dim]


class MLP(nn.Module):
    """Multi-Layer Perceptron with GELU activations."""
    def __init__(self, dims, act=nn.GELU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
