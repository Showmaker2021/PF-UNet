"""Test 1 (M1): Verify FJSPEnv contract, build_inputs, to_env_action, ordering assumption, and uneven job lengths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

MATSCHED_DIR = Path(__file__).resolve().parents[1]
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from adapter import build_inputs, to_env_action
from configs.matsched_fjsp import ensure_ppo_on_path, load_fjsp_dataset_and_baseline, make_fjsp_env


def _run_random_episode(env, K: int = 2) -> None:
    state = env.reset_state()
    B = env.batch_size
    done = False
    step = 0
    while not done:
        assert state.o_AT.shape == state.duration.shape[:2], (
            f"Expected o_AT shape {state.duration.shape[:2]}, got {state.o_AT.shape}"
        )
        inp = build_inputs(env, K=K)

        # Verify min_duration matches masked numpy min
        np_min = np.ma.masked_array(state.duration, mask=~state.o2m_mask).min(-1).filled(0.0)
        head_row = inp["head_row"].numpy()
        jvalid = inp["jvalid"].numpy()
        for b in range(B):
            for j in range(head_row.shape[1]):
                if jvalid[b, j]:
                    r = head_row[b, j]
                    assert np.isclose(
                        inp["core_f"][b, j, 1].item() * inp["scaler"][b].item(),
                        np_min[b, r],
                        atol=1e-4,
                    )

        elig = inp["elig"]
        # Each job has at most 1 head row
        is_head = torch.as_tensor(state.action_mask).any(-1)
        assert int(is_head.sum().item()) == int(inp["jvalid"].sum().item())

        # Randomly pick one eligible (job, machine) per batch item
        sel_jobs, sel_machs = [], []
        for b in range(B):
            coords = torch.nonzero(elig[b], as_tuple=False)
            assert len(coords) > 0, f"No eligible action at step {step} batch {b}"
            pick = coords[torch.randint(len(coords), (1,)).item()]
            sel_jobs.append(pick[0])
            sel_machs.append(pick[1])
        sel_job_t = torch.stack(sel_jobs)
        sel_mach_t = torch.stack(sel_machs)

        seq_np, mach_np = to_env_action(sel_job_t, sel_mach_t, inp["head_row"])
        for b in range(B):
            assert state.action_mask[b, seq_np[b], mach_np[b]], "Adapter produced invalid env action!"

        state, _, done = env.step((seq_np, mach_np))
        step += 1

    # Check makespan finite & positive
    mk = state.finish_time.max(axis=1).max(axis=1)
    assert np.all(mk > 0) and np.all(np.isfinite(mk))


def test_adapter_sd1_and_uneven_jobs() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # 1. SD1 10x5 (jobs have differing op counts sampled in [0.8M, 1.2M], so jobs finish at different steps)
    env = make_fjsp_env(num_jobs=10, num_machines=5, sd1=True)
    env.set_seed(42)
    env.generate_data(batch_size=8)
    _run_random_episode(env, K=2)
    print("  [PASS] SD1 10x5 random rollout via adapter")

    # 2. General FJSP with heterogeneous job lengths
    ensure_ppo_on_path()
    from SchedulingEnvironment import FJSPEnv

    env2 = FJSPEnv(
        generator_params={
            "num_jobs": 8,
            "num_machines": 4,
            "min_operation": 2,
            "max_operation": 7,
            "min_processing_time": 1,
            "max_processing_time": 50,
            "min_eligible_ma_per_op": 1,
            "max_eligible_ma_per_op": 4,
        },
        sd1=False,
    )
    env2.set_seed(123)
    env2.generate_data(batch_size=6)
    _run_random_episode(env2, K=2)
    print("  [PASS] Heterogeneous job lengths rollout via adapter")

    # 3. Verify operation ordering assumption on Brandimarte and Hurink_rdata
    for bpath in ("data/FJSP/TNNLS/BenchData/Brandimarte", "data/FJSP/TNNLS/BenchData/Hurink_rdata"):
        ds, _ = load_fjsp_dataset_and_baseline(bpath)
        env_b = FJSPEnv()
        for inst_num, j_arr, o_arr, d_arr, dep_arr, conn_arr in zip(*ds):
            env_b.load_data(inst_num, j_arr, o_arr, d_arr, dep_arr, conn_arr)
            env_b.reset_state()
            build_inputs(env_b, K=2)
        print(f"  [PASS] Ordering assumption verified on {bpath}")


if __name__ == "__main__":
    print("Running test_adapter...")
    test_adapter_sd1_and_uneven_jobs()
    print("ALL test_adapter PASSED")
