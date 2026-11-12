"""Test 2 (M2): Verify Matryoshka nested prefix invariance for levels 1 (d=32) and 2 (d=64)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

MATSCHED_DIR = Path(__file__).resolve().parents[1]
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from adapter import build_inputs
from configs.matsched_fjsp import make_fjsp_env
from model import MatSchedNet


def _perturb_beyond_prefix(model: MatSchedNet, d_keep: int, level_keep: int) -> None:
    """Randomize all parameters outside the first `d_keep` channels (and higher-level heads/critic)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Heads not used by level_keep
            if name.startswith("heads."):
                lv_str = name.split(".")[1]
                if int(lv_str) != level_keep:
                    param.add_(torch.randn_like(param) * 5.0)
                    continue
                # For the kept head, output dim of final linear is 1, input dim is d_keep
                if "tri." in name and param.ndim == 2:
                    if param.shape[0] > d_keep:
                        param[d_keep:, :].add_(torch.randn_like(param[d_keep:, :]) * 5.0)
                    if param.shape[1] > d_keep:
                        param[:, d_keep:].add_(torch.randn_like(param[:, d_keep:]) * 5.0)
                elif "tri." in name and param.ndim == 1 and param.shape[0] > d_keep:
                    param[d_keep:].add_(torch.randn_like(param[d_keep:]) * 5.0)
                elif "out.weight" in name and param.shape[1] > d_keep:
                    param[:, d_keep:].add_(torch.randn_like(param[:, d_keep:]) * 5.0)
                continue

            if name.startswith("critic_"):
                param.add_(torch.randn_like(param) * 5.0)
                continue

            # Inside NFFN: fc1 has shape (ff_mult*d_ring, d_ring), fc2 has shape (d_ring, ff_mult*d_ring)
            if ".fc1." in name:
                hid_keep = model.ff_mult * d_keep
                if param.shape[0] > hid_keep:
                    if param.ndim == 2:
                        param[hid_keep:, :].add_(torch.randn_like(param[hid_keep:, :]) * 5.0)
                    else:
                        param[hid_keep:].add_(torch.randn_like(param[hid_keep:]) * 5.0)
                if param.ndim == 2 and param.shape[1] > d_keep:
                    k0 = min(hid_keep, param.shape[0])
                    param[:k0, d_keep:].add_(torch.randn_like(param[:k0, d_keep:]) * 5.0)
                continue
            if ".fc2." in name:
                hid_keep = model.ff_mult * d_keep
                if param.ndim == 2:
                    if param.shape[0] > d_keep:
                        param[d_keep:, :].add_(torch.randn_like(param[d_keep:, :]) * 5.0)
                    if param.shape[1] > hid_keep:
                        k0 = min(d_keep, param.shape[0])
                        param[:k0, hid_keep:].add_(torch.randn_like(param[:k0, hid_keep:]) * 5.0)
                elif param.shape[0] > d_keep:
                    param[d_keep:].add_(torch.randn_like(param[d_keep:]) * 5.0)
                continue

            # Shared head final linear out.weight: (1, dmax)
            if name == "head.out.weight":
                param[:, d_keep:].add_(torch.randn_like(param[:, d_keep:]) * 5.0)
                continue
            if name == "head.out.bias":
                continue

            # Input embeddings: (dmax, f_in)
            if name.startswith("embed_") or "deepsets.phi1." in name:
                if param.ndim == 2:
                    param[d_keep:, :].add_(torch.randn_like(param[d_keep:, :]) * 5.0)
                else:
                    param[d_keep:].add_(torch.randn_like(param[d_keep:]) * 5.0)
                continue

            # Standard dmax x dmax linear, norm, or null_k/v
            if param.ndim == 1 and param.shape[0] > d_keep:
                param[d_keep:].add_(torch.randn_like(param[d_keep:]) * 5.0)
            elif param.ndim == 2:
                if param.shape[0] > d_keep:
                    param[d_keep:, :].add_(torch.randn_like(param[d_keep:, :]) * 5.0)
                if param.shape[1] > d_keep:
                    param[:d_keep, d_keep:].add_(torch.randn_like(param[:d_keep, d_keep:]) * 5.0)


def test_nested_prefix() -> None:
    torch.manual_seed(7)
    env = make_fjsp_env(num_jobs=6, num_machines=4, sd1=True)
    env.set_seed(7)
    env.generate_data(batch_size=4)
    env.reset_state()
    inp = build_inputs(env, K=2)

    elig_flat = inp["elig"].flatten(1)
    for shared_head in (False, True):
        model = MatSchedNet(dmax=128, num_layers=3, K=2, shared_head=shared_head)
        model.eval()

        # Check level 1 (d=32)
        with torch.no_grad():
            lp1_before, v1 = model(inp, level=1)
        assert v1 is None
        m1 = copy.deepcopy(model)
        _perturb_beyond_prefix(m1, d_keep=32, level_keep=1)
        with torch.no_grad():
            lp1_after, _ = m1(inp, level=1)
        assert torch.equal(lp1_before, lp1_after), f"Level 1 mismatch (shared_head={shared_head})"
        diff1 = (lp1_before[elig_flat] - lp1_after[elig_flat]).abs().max().item()
        assert diff1 == 0.0, f"Level 1 prefix invariance failed (shared_head={shared_head}): max diff={diff1}"

        # Check level 2 (d=64)
        with torch.no_grad():
            lp2_before, v2 = model(inp, level=2)
        assert v2 is None
        m2 = copy.deepcopy(model)
        _perturb_beyond_prefix(m2, d_keep=64, level_keep=2)
        with torch.no_grad():
            lp2_after, _ = m2(inp, level=2)
        assert torch.equal(lp2_before, lp2_after), f"Level 2 mismatch (shared_head={shared_head})"
        diff2 = (lp2_before[elig_flat] - lp2_after[elig_flat]).abs().max().item()
        assert diff2 == 0.0, f"Level 2 prefix invariance failed (shared_head={shared_head}): max diff={diff2}"

        print(f"  [PASS] Nested prefix exact invariance (shared_head={shared_head}): diff1={diff1}, diff2={diff2}")


if __name__ == "__main__":
    print("Running test_nested_prefix...")
    test_nested_prefix()
    print("ALL test_nested_prefix PASSED")
