"""Adaptive early-exit router for MatSched-Net."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from adapter import build_inputs, select_rows, to_env_action


def _compute_accept_mask(
    logp: torch.Tensor,
    p: torch.Tensor,
    elig_flat: torch.Tensor,
    threshold: float,
    metric: str,
) -> torch.Tensor:
    """Compute boolean early-exit acceptance mask at level < 3.

    Metrics:
      - 'margin': accept if p_top1 - p_top2 >= threshold (higher threshold = stricter).
      - 'norm_entropy': accept if H(p) / log(max(|A_valid|, 2)) <= threshold (lower threshold = stricter).
      - 'top1': accept if p_top1 >= threshold (higher threshold = stricter).
    """
    if metric == "margin":
        k2 = min(2, p.shape[-1])
        top2 = p.topk(k2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1] if k2 == 2 else top2[:, 0]
        return margin >= threshold

    if metric == "norm_entropy":
        lp = logp.float()
        pf = p.float()
        safe_lp = torch.where(elig_flat, lp, torch.zeros_like(lp))
        safe_p = torch.where(elig_flat, pf, torch.zeros_like(pf))
        ent = -(safe_p * safe_lp).sum(dim=-1)
        n_valid = elig_flat.sum(dim=-1).clamp_min(2).float()
        norm_ent = ent / torch.log(n_valid)
        # If only 1 valid action exists, entropy is 0 -> accept immediately
        one_valid = elig_flat.sum(dim=-1) <= 1
        return (norm_ent <= threshold) | one_valid

    if metric == "top1":
        return p.amax(dim=-1) >= threshold

    raise ValueError(f"Unknown router metric: {metric}")


@torch.no_grad()
def act_adaptive_from_inputs(
    model,
    inp: Dict[str, object],
    tau: Tuple[float, float] = (0.45, 0.45),
    greedy: bool = True,
    metric: str = "norm_entropy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run adaptive early-exit on prebuilt `inp` dict.

    Returns:
        chosen_flat: (B,) index in [0, J*M)
        chosen_logp: (B,) log-probability from the level that accepted the sample
        level_used: (B,) int tensor in {1, 2, 3}
    """
    elig = inp["elig"]
    B = elig.shape[0]
    device = elig.device

    chosen = torch.full((B,), -1, dtype=torch.long, device=device)
    chosen_logp = torch.zeros((B,), dtype=torch.float32, device=device)
    level_used = torch.zeros((B,), dtype=torch.long, device=device)
    todo = torch.arange(B, dtype=torch.long, device=device)

    for lv in (1, 2, 3):
        sub_inp = select_rows(inp, todo) if len(todo) < B else inp
        logp, _ = model(sub_inp, level=lv)
        p = logp.exp()

        if lv < 3:
            sub_elig_flat = sub_inp["elig"].flatten(1)
            accept = _compute_accept_mask(logp, p, sub_elig_flat, tau[lv - 1], metric)
        else:
            accept = torch.ones((len(todo),), dtype=torch.bool, device=device)

        if accept.any():
            if greedy:
                pick = p[accept].argmax(dim=-1)
            else:
                pick = Categorical(probs=p[accept]).sample()
            acc_idx = todo[accept]
            chosen[acc_idx] = pick
            chosen_logp[acc_idx] = logp[accept].gather(1, pick.unsqueeze(1)).squeeze(1).float()
            level_used[acc_idx] = lv

        todo = todo[~accept]
        if len(todo) == 0:
            break

    return chosen, chosen_logp, level_used


@torch.no_grad()
def act_adaptive(
    model,
    env,
    tau: Tuple[float, float] = (0.45, 0.45),
    greedy: bool = True,
    K: int = 2,
    metric: str = "norm_entropy",
    device: Optional[torch.device | str] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], torch.Tensor]:
    """Build inputs from `env` and execute adaptive early-exit routing."""
    if device is None:
        device = next(model.parameters()).device
    inp = build_inputs(env, K=K, device=device)
    chosen, _, level_used = act_adaptive_from_inputs(model, inp, tau=tau, greedy=greedy, metric=metric)
    M = inp["elig"].shape[2]
    sel_job = chosen // M
    sel_mach = chosen % M
    return to_env_action(sel_job, sel_mach, inp["head_row"]), level_used
