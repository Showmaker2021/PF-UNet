"""Test 5 (M2/M5): Verify KL distillation from frozen pi3 onto pi1 and pi2 decreases monotonically."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

MATSCHED_DIR = Path(__file__).resolve().parents[1]
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from adapter import build_inputs, move_inputs_to
from configs.matsched_fjsp import make_fjsp_env
from model import MatSchedNet
from train_ppo import masked_kl_div


def test_kd_monotonic() -> None:
    torch.manual_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_fjsp_env(num_jobs=8, num_machines=4, sd1=True)
    env.set_seed(99)
    env.generate_data(batch_size=16)
    env.reset_state()
    inp = move_inputs_to(build_inputs(env, K=2, device="cpu"), device)
    elig_flat = inp["elig"].flatten(1)

    model = MatSchedNet(dmax=128, num_layers=3, K=2).to(device)

    # Freeze everything needed for teacher, and only update level 1 & 2 heads + lower prefix or full student params
    # Note: here we freeze pi3 teacher distribution before optimization and optimize model params for lv1/lv2 KD
    # (specifically heads['1'], heads['2'] + prefix weights while checking teacher snapshot).
    model.eval()
    with torch.no_grad():
        teacher_logp, _ = model(inp, level=3)
        teacher_logp = teacher_logp.detach()

    # Freeze weights beyond 64 channels and heads['3'] + critic to simulate frozen pi3 upper capacity,
    # or train heads['1'] and heads['2'] + shared prefix to test KL minimization.
    opt = torch.optim.SGD(
        list(model.heads["1"].parameters()) + list(model.heads["2"].parameters()),
        lr=0.2,
        momentum=0.5,
    )

    kl1_hist, kl2_hist = [], []
    model.train()
    for step in range(30):
        opt.zero_grad(set_to_none=True)
        lp1, _ = model(inp, level=1)
        lp2, _ = model(inp, level=2)
        kl1 = masked_kl_div(teacher_logp, lp1, elig_flat)
        kl2 = masked_kl_div(teacher_logp, lp2, elig_flat)
        (kl1 + kl2).backward()
        opt.step()
        kl1_hist.append(float(kl1.item()))
        kl2_hist.append(float(kl2.item()))

    print(f"  KL1: {kl1_hist[0]:.4f} -> {kl1_hist[-1]:.4f}")
    print(f"  KL2: {kl2_hist[0]:.4f} -> {kl2_hist[-1]:.4f}")
    for i in range(1, len(kl1_hist)):
        assert kl1_hist[i] <= kl1_hist[i - 1] + 1e-6, f"KL1 increased at step {i}: {kl1_hist[i-1]} -> {kl1_hist[i]}"
        assert kl2_hist[i] <= kl2_hist[i - 1] + 1e-6, f"KL2 increased at step {i}: {kl2_hist[i-1]} -> {kl2_hist[i]}"
    assert kl1_hist[-1] < kl1_hist[0] * 0.5 and kl2_hist[-1] < kl2_hist[0] * 0.5
    print("  [PASS] KL1 and KL2 decreased monotonically")


if __name__ == "__main__":
    print("Running test_kd...")
    test_kd_monotonic()
    print("ALL test_kd PASSED")
