"""MatSched-Net: Hierarchical Matryoshka Policy & Critic Network for FJSP."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nested import NAttn, NFFN, NLinear, NRMSNorm, NTriLinear

RING_W = {"core": 128, "mid": 64, "out": 32}
LEVEL_D = {1: 32, 2: 64, 3: 128}


def chan_mask(width: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Binary channel mask (d,) keeping only the first min(width, d) channels."""
    return (torch.arange(d, device=device) < min(width, d)).to(dtype)


class DeepSetsTail(nn.Module):
    """Compress tail operations (rank > K) into one vector w_j per job via sum + max."""

    def __init__(self, f_in: int = 3, dmax: int = 128) -> None:
        super().__init__()
        self.f_in = f_in
        self.phi1 = NLinear(f_in, dmax)
        self.phi2 = NLinear(dmax, dmax)
        self.proj_sum = NLinear(dmax, dmax, bias=True)
        self.proj_max = NLinear(dmax, dmax, bias=False)

    def forward(self, tail: Dict[str, torch.Tensor], num_jobs: int, d: int) -> torch.Tensor:
        feat = tail["feat"]  # (B, N, f_in)
        mask = tail["mask"]  # (B, N)
        job = tail["job"]    # (B, N)
        B, _, _ = feat.shape

        h = F.gelu(self.phi1(feat, self.f_in, d))
        h = self.phi2(h, d, d)
        mask_f = mask.unsqueeze(-1).to(h.dtype)
        h_masked = h * mask_f

        job_exp = job.unsqueeze(-1).expand(-1, -1, d)
        sum_pool = torch.zeros((B, num_jobs, d), dtype=h.dtype, device=h.device)
        sum_pool.scatter_add_(1, job_exp, h_masked)

        has_tail = torch.zeros((B, num_jobs), dtype=torch.long, device=h.device)
        has_tail.scatter_add_(1, job, mask.long())
        has_tail_mask = (has_tail > 0).unsqueeze(-1).to(h.dtype)
        mean_pool = sum_pool / has_tail.clamp_min(1).unsqueeze(-1).to(h.dtype)

        neg_inf = torch.tensor(-1e9, dtype=h.dtype, device=h.device)
        h_for_max = torch.where(mask.unsqueeze(-1), h, neg_inf)
        max_pool = torch.full((B, num_jobs, d), -1e9, dtype=h.dtype, device=h.device)
        max_pool.scatter_reduce_(1, job_exp, h_for_max, reduce="amax", include_self=True)
        max_pool = max_pool * has_tail_mask

        return self.proj_sum(mean_pool, d, d) + self.proj_max(max_pool, d, d)


class PairMLP(nn.Module):
    """Prefix-safe residual MLP over (core_j, mach_i, pair_ji)."""

    def __init__(self, dmax: int = 128) -> None:
        super().__init__()
        self.tri = NTriLinear(dmax, dmax)
        self.out = NLinear(dmax, dmax)

    def forward(self, core: torch.Tensor, mach: torch.Tensor, pair: torch.Tensor, d: int) -> torch.Tensor:
        return self.out(F.gelu(self.tri(core, mach, pair, d, d)), d, d)


class PairPolicyHead(nn.Module):
    """Prefix-safe scoring head over (core_j, mach_i, pair_ji) -> scalar logit."""

    def __init__(self, dmax: int = 128) -> None:
        super().__init__()
        self.tri = NTriLinear(dmax, dmax)
        self.out = NLinear(dmax, 1)

    def forward(self, core: torch.Tensor, mach: torch.Tensor, pair: torch.Tensor, d: int) -> torch.Tensor:
        return self.out(F.gelu(self.tri(core, mach, pair, d, d)), d, 1).squeeze(-1)


class MatSchedBlock(nn.Module):
    """One hierarchical bidirectional layer of MatSched-Net."""

    def __init__(
        self,
        dmax: int = 128,
        head_dim: int = 16,
        ff_mult: int = 2,
        use_m2o: bool = True,
        use_mid_out: bool = True,
        use_cc: bool = True,
        use_pair_ffn: bool = False,
        use_out_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.use_m2o = use_m2o
        self.use_mid_out = use_mid_out
        self.use_cc = use_cc
        self.use_pair_ffn = use_pair_ffn
        self.use_out_ffn = use_out_ffn
        self.mid_dmax = min(dmax, RING_W["mid"])
        self.out_dmax = min(dmax, RING_W["out"])

        # 1. Inward flow (mid ring operates strictly within RING_W['mid']=64)
        if use_mid_out:
            self.norm_mid_in = NRMSNorm(self.mid_dmax)
            self.norm_kv_mid = NRMSNorm(self.mid_dmax)
            self.att_mid = NAttn(self.mid_dmax, head_dim)
            self.norm_core_in = NRMSNorm(dmax)
            self.norm_kv_in = NRMSNorm(dmax)
            self.att_in = NAttn(dmax, head_dim)
            self.norm_out_proj = NRMSNorm(self.out_dmax)
            self.proj_out = NLinear(self.out_dmax, self.out_dmax)

        # 2. Bidirectional core
        self.norm_core_p = NRMSNorm(dmax)
        self.norm_mach_p = NRMSNorm(dmax)
        self.norm_pair_p = NRMSNorm(dmax)
        self.pair_mlp = PairMLP(dmax)

        self.norm_core_c2p = NRMSNorm(dmax)
        self.norm_pair_c2p = NRMSNorm(dmax)
        self.att_c2p = NAttn(dmax, head_dim)

        if use_m2o:
            self.norm_mach_m2p = NRMSNorm(dmax)
            self.norm_pair_m2p = NRMSNorm(dmax)
            self.att_m2p = NAttn(dmax, head_dim)

        if use_cc:
            self.norm_core_cc = NRMSNorm(dmax)
            self.att_cc = NAttn(dmax, head_dim)

        # 3. FFN per stream
        self.norm_core_ff = NRMSNorm(dmax)
        self.ffn_core = NFFN(dmax, ff_mult)
        self.norm_mach_ff = NRMSNorm(dmax)
        self.ffn_mach = NFFN(dmax, ff_mult)
        if use_pair_ffn:
            self.norm_pair_ff = NRMSNorm(dmax)
            self.ffn_pair = NFFN(dmax, ff_mult)

        if use_mid_out:
            self.norm_mid_ff = NRMSNorm(self.mid_dmax)
            self.ffn_mid = NFFN(self.mid_dmax, ff_mult)
            if use_out_ffn:
                self.norm_out_ff = NRMSNorm(self.out_dmax)
                self.ffn_out = NFFN(self.out_dmax, ff_mult)

    def forward(
        self,
        core: torch.Tensor,
        mach: torch.Tensor,
        pair: torch.Tensor,
        mid: Optional[torch.Tensor],
        outer: Optional[torch.Tensor],
        jv: torch.Tensor,
        el: torch.Tensor,
        mid_valid: torch.Tensor,
        cm: Dict[str, torch.Tensor],
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        jv_f = jv.unsqueeze(-1).to(core.dtype)
        el_f = el.unsqueeze(-1).to(pair.dtype)
        mv_f = mid_valid.unsqueeze(-1).to(core.dtype)
        d_mid = min(d, self.mid_dmax)
        d_out = min(d, self.out_dmax)

        # 1. Inward flow
        if self.use_mid_out and outer is not None:
            if mid is not None:
                kv = torch.cat([mid[..., :d_mid], outer[:, :, None, :d_mid], core[:, :, None, :d_mid]], dim=2)
                kv_valid = torch.cat([mid_valid, jv.unsqueeze(-1), jv.unsqueeze(-1)], dim=-1)
                job_local_mask = mid_valid.unsqueeze(-1) & kv_valid.unsqueeze(-2)  # (B, J, K, K+2)
                mid_upd = self.att_mid(
                    self.norm_mid_in(mid, d_mid),
                    self.norm_kv_mid(kv, d_mid),
                    mask=job_local_mask,
                    d=d_mid,
                    null=True,
                ) * mv_f
                if d_mid < d:
                    mid_upd = F.pad(mid_upd, (0, d - d_mid))
                mid = mid + mid_upd

                kv2 = torch.cat([mid, outer[:, :, None, :]], dim=2)
                m2 = (jv.unsqueeze(-1) & torch.cat([mid_valid, jv.unsqueeze(-1)], dim=-1)).unsqueeze(-2)
                core = core + self.att_in(
                    self.norm_core_in(core, d).unsqueeze(2),
                    self.norm_kv_in(kv2, d),
                    mask=m2,
                    d=d,
                    null=True,
                ).squeeze(2) * jv_f
            else:
                out_upd = self.proj_out(self.norm_out_proj(outer, d_out), d_out, d_out) * jv_f
                if d_out < d:
                    out_upd = F.pad(out_upd, (0, d - d_out))
                core = core + out_upd

        # 2. Bidirectional core
        pair = pair + self.pair_mlp(
            self.norm_core_p(core, d),
            self.norm_mach_p(mach, d),
            self.norm_pair_p(pair, d),
            d,
        ) * el_f

        core = core + self.att_c2p(
            self.norm_core_c2p(core, d).unsqueeze(2),
            self.norm_pair_c2p(pair, d),
            mask=el.unsqueeze(2),
            d=d,
            null=True,
        ).squeeze(2) * jv_f

        if self.use_m2o:
            q_m2p = self.norm_mach_m2p(mach, d).unsqueeze(2)
            kv_m2p = torch.cat([self.norm_pair_m2p(pair, d).transpose(1, 2), q_m2p], dim=2)
            m_self = torch.ones((el.shape[0], el.shape[2], 1), dtype=torch.bool, device=el.device)
            m_m2p = torch.cat([el.transpose(1, 2), m_self], dim=-1).unsqueeze(2)
            mach = mach + self.att_m2p(
                q_m2p,
                kv_m2p,
                mask=m_m2p,
                d=d,
                null=False,
            ).squeeze(2)

        if self.use_cc:
            cc_mask = jv.unsqueeze(-1) & jv.unsqueeze(-2)
            core = core + self.att_cc(
                self.norm_core_cc(core, d),
                self.norm_core_cc(core, d),
                mask=cc_mask,
                d=d,
                null=True,
            ) * jv_f

        # 3. FFN + strict ring channel masking
        core = (core + self.ffn_core(self.norm_core_ff(core, d), d)) * cm["core"] * jv_f
        mach = (mach + self.ffn_mach(self.norm_mach_ff(mach, d), d)) * cm["core"]
        if self.use_pair_ffn:
            pair = (pair + self.ffn_pair(self.norm_pair_ff(pair, d), d)) * cm["core"] * el_f
        else:
            pair = pair * cm["core"] * el_f

        if self.use_mid_out and outer is not None:
            if self.use_out_ffn:
                out_ff = self.ffn_out(self.norm_out_ff(outer, d_out), d_out)
                if d_out < d:
                    out_ff = F.pad(out_ff, (0, d - d_out))
                outer = (outer + out_ff) * cm["out"] * jv_f
            else:
                outer = outer * cm["out"] * jv_f
            if mid is not None:
                mid_ff = self.ffn_mid(self.norm_mid_ff(mid, d_mid), d_mid)
                if d_mid < d:
                    mid_ff = F.pad(mid_ff, (0, d - d_mid))
                mid = (mid + mid_ff) * cm["mid"] * mv_f

        return core, mach, pair, mid, outer


class MatSchedNet(nn.Module):
    """MatSched-Net with nested prefix levels 1 (32), 2 (64), 3 (128)."""

    def __init__(
        self,
        dmax: int = 128,
        num_layers: int = 3,
        K: int = 2,
        head_dim: int = 16,
        ff_mult: int = 2,
        use_m2o: bool = True,
        use_mid_out: bool = True,
        use_cc: bool = True,
        use_pair_ffn: bool = False,
        use_out_ffn: bool = False,
        shared_head: bool = False,
        fixed_width: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dmax = dmax
        self.num_layers = num_layers
        self.K = K
        self.ff_mult = ff_mult
        self.use_mid_out = use_mid_out
        self.shared_head = shared_head
        self.fixed_width = fixed_width

        self.embed_core = NLinear(4, dmax)
        self.embed_mach = NLinear(4, dmax)
        self.embed_pair = NLinear(6, dmax)
        if use_mid_out:
            self.embed_mid = NLinear(3, min(dmax, RING_W["mid"]))
            self.deepsets = DeepSetsTail(3, min(dmax, RING_W["out"]))

        self.layers = nn.ModuleList(
            [
                MatSchedBlock(
                    dmax=dmax,
                    head_dim=head_dim,
                    ff_mult=ff_mult,
                    use_m2o=use_m2o,
                    use_mid_out=use_mid_out,
                    use_cc=use_cc,
                    use_pair_ffn=use_pair_ffn,
                    use_out_ffn=use_out_ffn,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_core_out = NRMSNorm(dmax)
        self.norm_mach_out = NRMSNorm(dmax)
        self.norm_pair_out = NRMSNorm(dmax)

        if shared_head:
            self.head = PairPolicyHead(dmax)
        else:
            self.heads = nn.ModuleDict(
                {
                    "1": PairPolicyHead(32 if fixed_width is None else fixed_width),
                    "2": PairPolicyHead(64 if fixed_width is None else fixed_width),
                    "3": PairPolicyHead(dmax),
                }
            )

        # Critic only used by PPO at level 3
        self.critic_c = NLinear(dmax, dmax, bias=True)
        self.critic_m = NLinear(dmax, dmax, bias=False)
        self.critic_out = NLinear(dmax, 1)

    def _level_dim(self, level: int) -> int:
        if self.fixed_width is not None:
            return self.fixed_width
        return LEVEL_D[level]

    def forward(
        self, inp: Dict[str, object], level: int = 3
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        d = self._level_dim(level)
        core_f = inp["core_f"]
        mach_f = inp["mach_f"]
        pair_f = inp["pair_f"]
        mid_f = inp["mid_f"]
        mid_valid = inp["mid_valid"]
        elig = inp["elig"]
        jvalid = inp["jvalid"]

        B, J, M = elig.shape
        device, dtype = core_f.device, core_f.dtype
        cm = {r: chan_mask(RING_W[r], d, device, dtype) for r in RING_W}

        jv_f = jvalid.unsqueeze(-1).to(dtype)
        el_f = elig.unsqueeze(-1).to(dtype)

        core = self.embed_core(core_f, 4, d) * cm["core"] * jv_f
        mach = self.embed_mach(mach_f, 4, d) * cm["core"]
        pair = self.embed_pair(pair_f, 6, d) * cm["core"] * el_f

        if self.use_mid_out:
            d_out = min(d, RING_W["out"])
            d_mid = min(d, RING_W["mid"])
            outer = self.deepsets(inp["tail"], J, d_out)
            if d_out < d:
                outer = F.pad(outer, (0, d - d_out))
            outer = outer * cm["out"] * jv_f
            if level >= 2:
                mid = self.embed_mid(mid_f, 3, d_mid)
                if d_mid < d:
                    mid = F.pad(mid, (0, d - d_mid))
                mid = mid * cm["mid"] * mid_valid.unsqueeze(-1).to(dtype)
            else:
                mid = None
        else:
            outer = None
            mid = None

        for lyr in self.layers:
            core, mach, pair, mid, outer = lyr(
                core=core,
                mach=mach,
                pair=pair,
                mid=mid,
                outer=outer,
                jv=jvalid,
                el=elig,
                mid_valid=mid_valid,
                cm=cm,
                d=d,
            )

        c_out = self.norm_core_out(core, d) * jv_f
        m_out = self.norm_mach_out(mach, d)
        p_out = self.norm_pair_out(pair, d) * el_f

        head_mod = self.head if self.shared_head else self.heads[str(level)]
        score = head_mod(c_out, m_out, p_out, d)  # (B, J, M)
        score = score.masked_fill(~elig, float("-inf"))
        logp = F.log_softmax(score.flatten(1), dim=-1)  # (B, J*M)

        value: Optional[torch.Tensor] = None
        if level == 3 or self.fixed_width is not None:
            jv_cnt = jv_f.sum(dim=1).clamp_min(1.0)
            core_pooled = (c_out * jv_f).sum(dim=1) / jv_cnt
            mach_pooled = m_out.mean(dim=1)
            vc = self.critic_c(core_pooled, d, d) + self.critic_m(mach_pooled, d, d)
            value = self.critic_out(F.gelu(vc), d, 1).squeeze(-1)

        return logp, value
