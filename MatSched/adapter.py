"""State adapter transforming FJSPEnv.state into hierarchical job/machine/pair tensors."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _gather_rows(x: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """Gather rows along dim=1 safely when invalid indices are -1."""
    safe_idx = row_idx.clamp_min(0)
    if x.ndim == 2:
        return x.gather(1, safe_idx)
    if x.ndim == 3:
        return x.gather(1, safe_idx[..., None].expand(-1, -1, x.shape[-1]))
    raise ValueError(f"Unsupported tensor rank for _gather_rows: {x.ndim}")


@torch.no_grad()
def build_inputs(env, K: int = 2, device: torch.device | str | None = None) -> Dict[str, object]:
    """Convert env.state into hierarchical tensors for MatSchedNet.

    Verified against PPO/SchedulingEnvironment.py:
      - env.state.duration: (B, N_rem, M), 0 means incompatible
      - env.state.o_AT: (B, N_rem), pruned available time (do NOT use available_time_o)
      - env.state.m_AT: (B, M)
      - env.state.action_mask: (B, N_rem, M)
      - env.state.job_idx: (B, N_rem)
      - env.problem.duration: (B, N_init, M)
      - env.problem.job_idx: (B, N_init)
    """
    st = env.state
    if device is None:
        device = torch.device("cpu")

    dur = torch.as_tensor(st.duration, dtype=torch.float32, device=device)
    o_at = torch.as_tensor(st.o_AT, dtype=torch.float32, device=device)
    m_at = torch.as_tensor(st.m_AT, dtype=torch.float32, device=device)
    amask = torch.as_tensor(st.action_mask, dtype=torch.bool, device=device)
    job = torch.as_tensor(st.job_idx, dtype=torch.long, device=device)

    prob_dur = torch.as_tensor(env.problem.duration, dtype=torch.float32, device=device)
    scaler = prob_dur.amax(dim=(1, 2), keepdim=False).unsqueeze(-1).clamp_min(1.0)  # (B, 1)

    B, N, M = dur.shape
    J = int(np.max(env.problem.job_idx)) + 1

    # Compute min duration over eligible machines (dur > 0)
    inf_val = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    mind = torch.where(dur > 0, dur, inf_val).amin(dim=-1)  # (B, N)

    # Identify head operation (rank == 0 among remaining operations per job)
    is_head = amask.any(dim=-1)  # (B, N)
    oh = F.one_hot(job, num_classes=J).to(torch.long)  # (B, N, J)
    rank = (oh.cumsum(dim=1) * oh).sum(dim=-1) - 1  # (B, N)
    if not torch.equal(is_head, rank == 0):
        raise AssertionError(
            "Operation ordering assumption violated: is_head != (rank == 0). "
            "Remaining operations in a job must be ordered by precedence."
        )

    ar = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    hr_buf = torch.full((B, J + 1), -1, dtype=torch.long, device=device)
    hr_idx = torch.where(is_head, job, J)
    hr_val = torch.where(is_head, ar, -1)
    hr = hr_buf.scatter_(1, hr_idx, hr_val)[:, :J]  # (B, J)
    jvalid = hr >= 0  # (B, J)

    # Mid ring: rank 1..K per job
    is_mid = (rank >= 1) & (rank <= K)
    mr_buf = torch.full((B, J * K + 1), -1, dtype=torch.long, device=device)
    mr_idx = torch.where(is_mid, job * K + (rank - 1), J * K)
    mr_val = torch.where(is_mid, ar, -1)
    mr = mr_buf.scatter_(1, mr_idx, mr_val)[:, : J * K].view(B, J, K)
    mid_valid = mr >= 0  # (B, J, K)

    # Outer ring: tail operations (rank > K)
    is_tail = rank > K  # (B, N)

    # Head features, strictly zeroed for finished jobs
    jv_f = jvalid.float()
    d_head = _gather_rows(dur, hr) * jv_f[..., None]  # (B, J, M)
    elig = (d_head > 0) & jvalid[..., None]  # (B, J, M)
    o_head = _gather_rows(o_at, hr) * jv_f  # (B, J)
    mind_head = _gather_rows(mind, hr) * jv_f  # (B, J)

    # Single reference horizon over active head operations and active machines (matching ReSched)
    active_o = torch.where(jvalid, o_head, inf_val).amin(dim=1, keepdim=True)
    active_mach = (dur > 0).any(dim=1)  # (B, M)
    active_m = torch.where(active_mach, m_at, inf_val).amin(dim=1, keepdim=True)
    t_ref = torch.minimum(active_o, active_m)  # (B, 1)
    t_ref = torch.where(torch.isfinite(t_ref), t_ref, torch.zeros_like(t_ref))

    start = torch.maximum(m_at[:, None, :], o_head[..., None])  # (B, J, M)
    scaler_3d = scaler[..., None]  # (B, 1, 1)
    t_ref_3d = t_ref[..., None]  # (B, 1, 1)

    # Bounded relative machine efficiency ratio: mind_head / d_head in (0, 1] (no OOD spikes)
    # Future contention lookahead: immediate next op (rank == 1) duration on the same machine
    if K >= 1:
        d_next = _gather_rows(dur, mr[:, :, 0]) * mid_valid[:, :, 0:1].float()  # (B, J, M)
    else:
        d_next = torch.zeros_like(d_head)
    pair_f = torch.stack(
        [
            d_head / scaler_3d,
            mind_head[..., None] / d_head.clamp_min(1e-6),
            (start - t_ref_3d) / scaler_3d,
            (start + d_head - t_ref_3d) / scaler_3d,
            (m_at[:, None, :] - o_head[..., None]) / scaler_3d,
            d_next / scaler_3d,
        ],
        dim=-1,
    ) * elig[..., None].float()  # (B, J, M, 6)

    n_el = (dur > 0).sum(dim=-1).clamp_min(1).float()  # (B, N)
    rem_elig = (dur > 0) & (rank >= 1)[..., None]  # (B, N, M)
    fl = rem_elig.float() * dur / n_el[..., None]  # (B, N, M)
    fl_bn = (rem_elig & (n_el[..., None] <= 2)).float() * dur / n_el[..., None]  # bottleneck load
    mach_f = torch.stack(
        [
            (m_at - t_ref) / scaler,
            elig.sum(dim=1).float() / float(J),
            fl.sum(dim=1) / (scaler * float(J)),
            fl_bn.sum(dim=1) / (scaler * float(J)),
        ],
        dim=-1,
    )  # (B, M, 4)

    cnt = torch.zeros((B, J), dtype=torch.float32, device=device).scatter_add_(
        1, job, torch.ones((B, N), dtype=torch.float32, device=device)
    )
    work = torch.zeros((B, J), dtype=torch.float32, device=device).scatter_add_(1, job, mind)
    max_cnt = cnt.amax(dim=1, keepdim=True).clamp_min(1.0)

    core_f = (
        torch.stack(
            [
                (o_head - t_ref) / scaler,
                mind_head / scaler,
                cnt / max_cnt,
                torch.log1p(work / scaler),
            ],
            dim=-1,
        )
        * jv_f[..., None]
    )  # (B, J, 4)

    mr_flat = mr.view(B, J * K)
    mind_mid = _gather_rows(mind, mr_flat).view(B, J, K)
    nel_mid = _gather_rows(n_el, mr_flat).view(B, J, K)
    rank_mid = _gather_rows(rank.float(), mr_flat).view(B, J, K)
    mid_f = (
        torch.stack(
            [
                mind_mid / scaler_3d,
                nel_mid / float(M),
                rank_mid / float(max(K, 1)),
            ],
            dim=-1,
        )
        * mid_valid[..., None].float()
    )  # (B, J, K, 3)

    tail_feat = (
        torch.stack(
            [
                mind / scaler,
                n_el / float(M),
                rank.float() / max_cnt,
            ],
            dim=-1,
        )
        * is_tail[..., None].float()
    )  # (B, N, 3)

    return {
        "core_f": core_f,
        "mach_f": mach_f,
        "pair_f": pair_f,
        "mid_f": mid_f,
        "mid_valid": mid_valid,
        "elig": elig,
        "jvalid": jvalid,
        "tail": {
            "feat": tail_feat,
            "mask": is_tail,
            "job": job,
        },
        "head_row": hr,
        "scaler": scaler.squeeze(-1),
    }


def to_env_action(
    sel_job: torch.Tensor, sel_mach: torch.Tensor, head_row: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert selected (job, machine) indices into FJSPEnv action tuple (seq_idx, machine_idx)."""
    seq = head_row.gather(1, sel_job.unsqueeze(1)).squeeze(1)
    return seq.detach().cpu().numpy(), sel_mach.detach().cpu().numpy()


def select_rows(inp: Dict[str, object], idx: torch.Tensor) -> Dict[str, object]:
    """Slice batch dimension by index tensor `idx`."""
    tail = inp["tail"]
    return {
        "core_f": inp["core_f"][idx],
        "mach_f": inp["mach_f"][idx],
        "pair_f": inp["pair_f"][idx],
        "mid_f": inp["mid_f"][idx],
        "mid_valid": inp["mid_valid"][idx],
        "elig": inp["elig"][idx],
        "jvalid": inp["jvalid"][idx],
        "tail": {
            "feat": tail["feat"][idx],
            "mask": tail["mask"][idx],
            "job": tail["job"][idx],
        },
        "head_row": inp["head_row"][idx],
        "scaler": inp["scaler"][idx],
    }


def move_inputs_to(inp: Dict[str, object], device: torch.device | str) -> Dict[str, object]:
    """Move an input dict (typically on CPU) to `device`."""
    tail = inp["tail"]
    return {
        "core_f": inp["core_f"].to(device),
        "mach_f": inp["mach_f"].to(device),
        "pair_f": inp["pair_f"].to(device),
        "mid_f": inp["mid_f"].to(device),
        "mid_valid": inp["mid_valid"].to(device),
        "elig": inp["elig"].to(device),
        "jvalid": inp["jvalid"].to(device),
        "tail": {
            "feat": tail["feat"].to(device),
            "mask": tail["mask"].to(device),
            "job": tail["job"].to(device),
        },
        "head_row": inp["head_row"].to(device),
        "scaler": inp["scaler"].to(device),
    }


def collate_inputs(items: Sequence[Dict[str, object]], device: torch.device | str | None = None) -> Dict[str, object]:
    """Collate single-instance or batched input dicts, padding variable-length tail tensors."""
    keys_fixed = ("core_f", "mach_f", "pair_f", "mid_f", "mid_valid", "elig", "jvalid", "head_row", "scaler")
    out: Dict[str, object] = {}
    for k in keys_fixed:
        tensors = [it[k] for it in items]
        cat_t = torch.cat(tensors, dim=0)
        out[k] = cat_t.to(device) if device is not None else cat_t

    max_n = max(it["tail"]["feat"].shape[1] for it in items)
    feats, masks, jobs = [], [], []
    for it in items:
        t = it["tail"]
        n = t["feat"].shape[1]
        if n < max_n:
            pad_n = max_n - n
            feats.append(F.pad(t["feat"], (0, 0, 0, pad_n), value=0.0))
            masks.append(F.pad(t["mask"], (0, pad_n), value=False))
            jobs.append(F.pad(t["job"], (0, pad_n), value=0))
        else:
            feats.append(t["feat"])
            masks.append(t["mask"])
            jobs.append(t["job"])

    cat_feat = torch.cat(feats, dim=0)
    cat_mask = torch.cat(masks, dim=0)
    cat_job = torch.cat(jobs, dim=0)
    if device is not None:
        cat_feat = cat_feat.to(device)
        cat_mask = cat_mask.to(device)
        cat_job = cat_job.to(device)

    out["tail"] = {
        "feat": cat_feat,
        "mask": cat_mask,
        "job": cat_job,
    }
    return out
