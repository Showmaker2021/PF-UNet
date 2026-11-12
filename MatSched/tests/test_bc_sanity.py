"""Test 4 (M3 / Gate G-1): Behavior cloning sanity check on MWKR dispatching rule (>95% accuracy)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MATSCHED_DIR = Path(__file__).resolve().parents[1]
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from adapter import build_inputs, move_inputs_to, to_env_action
from configs.matsched_fjsp import make_fjsp_env
from model import MatSchedNet
from rollout import _concat_padded_inputs, pad_tail_to


def mwkr_expert_action(inp: dict) -> torch.Tensor:
    """MWKR (Most Work Remaining) expert over active (job, machine) pairs.

    Selects the active job with maximum remaining min-work (`core_f[..., 3]`),
    then the eligible machine with smallest processing time (`pair_f[..., 0]`).
    """
    elig = inp["elig"]  # (B, J, M)
    B, J, M = elig.shape
    work_rem = inp["core_f"][..., 3].clone()  # (B, J)
    work_rem = work_rem.masked_fill(~inp["jvalid"], -1e9)
    best_job = work_rem.argmax(dim=-1)  # (B,)

    dur_job = inp["pair_f"][torch.arange(B, device=elig.device), best_job, :, 0]  # (B, M)
    elig_job = elig[torch.arange(B, device=elig.device), best_job, :]
    dur_job = dur_job.masked_fill(~elig_job, 1e9)
    best_mach = dur_job.argmin(dim=-1)
    return best_job * M + best_mach


def test_bc_sanity() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_fjsp_env(num_jobs=10, num_machines=5, sd1=True)
    env.set_seed(42)
    env.generate_data(batch_size=16)
    env.reset_state()
    n_init = env.problem.duration.shape[1]

    inps, targets = [], []
    done = False
    while not done:
        inp = build_inputs(env, K=2, device="cpu")
        act_flat = mwkr_expert_action(inp)
        M = inp["elig"].shape[2]
        env_act = to_env_action(act_flat // M, act_flat % M, inp["head_row"])
        _, _, done = env.step(env_act)
        inps.append(pad_tail_to(inp, n_init))
        targets.append(act_flat)

    all_inp = _concat_padded_inputs(inps)
    all_tgt = torch.cat(targets, dim=0)
    n_samples = all_tgt.shape[0]
    print(f"  Collected {n_samples} MWKR expert steps on SD1 10x5")

    model = MatSchedNet(dmax=128, num_layers=3, K=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)

    model.train()
    mb_size = 128
    for epoch in range(1, 101):
        perm = torch.randperm(n_samples)
        correct = 0
        total_loss = 0.0
        for s in range(0, n_samples, mb_size):
            idx = perm[s : s + mb_size]
            from adapter import select_rows

            mb_inp = move_inputs_to(select_rows(all_inp, idx), device)
            mb_tgt = all_tgt[idx].to(device)

            opt.zero_grad(set_to_none=True)
            logp, _ = model(mb_inp, level=3)
            loss = F.nll_loss(logp, mb_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            correct += int((logp.argmax(dim=-1) == mb_tgt).sum().item())
            total_loss += float(loss.item()) * len(idx)

        sched.step()
        acc = correct / n_samples
        if epoch % 10 == 0 or epoch == 1 or acc >= 0.97:
            print(f"  Epoch {epoch:3d} | loss={total_loss/n_samples:.4f} | acc={acc*100:.2f}%")
        if acc >= 0.97 and epoch >= 20:
            break

    # Final deterministic eval pass
    model.eval()
    final_correct = 0
    with torch.no_grad():
        from adapter import select_rows

        for s in range(0, n_samples, mb_size):
            idx = torch.arange(s, min(s + mb_size, n_samples))
            mb_inp = move_inputs_to(select_rows(all_inp, idx), device)
            mb_tgt = all_tgt[idx].to(device)
            logp, _ = model(mb_inp, level=3)
            final_correct += int((logp.argmax(dim=-1) == mb_tgt).sum().item())
    final_acc = final_correct / n_samples
    print(f"  Final BC accuracy: {final_acc*100:.2f}%")
    assert final_acc > 0.95, f"Gate G-1 FAILED: BC accuracy {final_acc*100:.2f}% <= 95%"
    print("  [PASS] Gate G-1 passed (MWKR BC accuracy > 95%)")


if __name__ == "__main__":
    print("Running test_bc_sanity (Gate G-1)...")
    test_bc_sanity()
    print("ALL test_bc_sanity PASSED")
