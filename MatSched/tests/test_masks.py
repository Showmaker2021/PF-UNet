"""Test 3 (M2): Verify probability masks, zero NaNs with finished jobs/idle machines, zero leakage, and job permutation equivariance."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

MATSCHED_DIR = Path(__file__).resolve().parents[1]
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from adapter import build_inputs, to_env_action
from configs.matsched_fjsp import make_fjsp_env
from model import MatSchedNet


def _advance_until_partial_jobs_and_idle_machine() -> tuple:
    """Advance an episode until >=1 job is finished and construct/find a state with an idle machine."""
    env = make_fjsp_env(num_jobs=6, num_machines=4, sd1=True)
    env.set_seed(11)
    env.generate_data(batch_size=3)
    env.reset_state()

    done = False
    found_finished_job = False
    while not done:
        inp = build_inputs(env, K=2)
        if not inp["jvalid"].all():
            found_finished_job = True
            break
        # Pick actions favoring job 0 so job 0 finishes early
        elig = inp["elig"]
        sel_jobs, sel_machs = [], []
        for b in range(elig.shape[0]):
            coords = torch.nonzero(elig[b], as_tuple=False)
            sel_jobs.append(coords[0, 0])
            sel_machs.append(coords[0, 1])
        _, _, done = env.step(to_env_action(torch.stack(sel_jobs), torch.stack(sel_machs), inp["head_row"]))

    assert found_finished_job, "Expected at least one finished job before episode end"
    return env, build_inputs(env, K=2)


def test_masks_nan_leakage_and_equivariance() -> None:
    torch.manual_seed(123)
    _, inp = _advance_until_partial_jobs_and_idle_machine()

    # Force machine 0 to have zero eligible head jobs to test M2O NaN safety explicitly
    inp_idle = copy.deepcopy(inp)
    # Keep at least one eligible machine per sample
    inp_idle["elig"][:, :, 0] = False
    inp_idle["pair_f"][:, :, 0, :] = 0.0
    # Make sure every batch element still has at least one eligible pair
    assert inp_idle["elig"].flatten(1).any(dim=1).all()

    model = MatSchedNet(dmax=128, num_layers=3, K=2)
    model.eval()

    for lv in (1, 2, 3):
        with torch.no_grad():
            logp, val = model(inp_idle, level=lv)
        probs = logp.exp()
        assert torch.isfinite(probs).all(), f"NaN/Inf in probs at level {lv}"
        if lv == 3:
            assert torch.isfinite(val).all(), "NaN/Inf in critic value"

        elig_flat = inp_idle["elig"].flatten(1)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), f"Prob sum != 1 at level {lv}"
        assert (probs[~elig_flat] == 0.0).all(), f"Non-zero prob on ineligible action at level {lv}"

    # Zero leakage from finished job slots (~jvalid)
    inp_noise = copy.deepcopy(inp_idle)
    finished = ~inp_noise["jvalid"]
    assert finished.any()
    inp_noise["core_f"][finished] = torch.randn_like(inp_noise["core_f"][finished]) * 100.0
    inp_noise["pair_f"][finished] = torch.randn_like(inp_noise["pair_f"][finished]) * 100.0
    inp_noise["mid_f"][finished] = torch.randn_like(inp_noise["mid_f"][finished]) * 100.0

    with torch.no_grad():
        lp_a, val_a = model(inp_idle, level=3)
        lp_b, val_b = model(inp_noise, level=3)
    assert torch.allclose(lp_a, lp_b, atol=1e-5), "Finished job features leaked into policy!"
    assert torch.allclose(val_a, val_b, atol=1e-5), "Finished job features leaked into critic!"

    # Job permutation equivariance
    B, J, M = inp["elig"].shape
    perm = torch.randperm(J)
    inv_perm = torch.argsort(perm)

    inp_perm = copy.deepcopy(inp)
    inp_perm["core_f"] = inp["core_f"][:, perm]
    inp_perm["pair_f"] = inp["pair_f"][:, perm]
    inp_perm["mid_f"] = inp["mid_f"][:, perm]
    inp_perm["mid_valid"] = inp["mid_valid"][:, perm]
    inp_perm["elig"] = inp["elig"][:, perm]
    inp_perm["jvalid"] = inp["jvalid"][:, perm]
    inp_perm["head_row"] = inp["head_row"][:, perm]
    inp_perm["tail"]["job"] = inv_perm[inp["tail"]["job"]]

    for lv in (1, 2, 3):
        with torch.no_grad():
            lp_orig, v_orig = model(inp, level=lv)
            lp_perm, v_perm = model(inp_perm, level=lv)
        p_orig = lp_orig.exp().view(B, J, M)
        p_perm = lp_perm.exp().view(B, J, M)
        diff = (p_orig[:, perm, :] - p_perm).abs().max().item()
        assert diff < 1e-5, f"Permutation equivariance failed at level {lv}: max diff={diff}"
        if lv == 3:
            assert (v_orig - v_perm).abs().max().item() < 1e-5, "Critic not permutation invariant!"

    print("  [PASS] Masks, NaN safety with idle machine/finished jobs, zero leakage, permutation equivariance")


if __name__ == "__main__":
    print("Running test_masks...")
    test_masks_nan_leakage_and_equivariance()
    print("ALL test_masks PASSED")
