"""Diagnostic check for remaining technical risks: dataset ordering, latency breakdown, and 30x10 scaling."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

MATSCHED_DIR = Path(__file__).resolve().parent
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from adapter import build_inputs, to_env_action
from configs.matsched_fjsp import REPO_ROOT, ensure_ppo_on_path, load_fjsp_dataset_and_baseline, make_fjsp_env
from evaluate import evaluate_resched_baseline
from model import MatSchedNet


def check_all_datasets_ordering() -> None:
    ensure_ppo_on_path()
    from SchedulingEnvironment import FJSPEnv

    paths = [
        "data/FJSP/TNNLS/BenchData/Brandimarte",
        "data/FJSP/TNNLS/BenchData/Hurink_rdata",
        "data/FJSP/TNNLS/BenchData/Hurink_edata",
        "data/FJSP/TNNLS/BenchData/Hurink_vdata",
        "data/FJSP/TNNLS/SD1/10x5",
        "data/FJSP/TNNLS/SD1/30x10",
        "data/FJSP/TNNLS/SD1/40x10",
        "data/FJSP/TNNLS/SD2/10x5+mix",
        "data/FJSP/TNNLS/SD2/30x10+mix",
    ]
    env = FJSPEnv()
    for p in paths:
        full = REPO_ROOT / p
        if not full.exists():
            print(f"  [SKIP] {p} (not found)")
            continue
        ds, _ = load_fjsp_dataset_and_baseline(p)
        n_inst = sum(ds[0])
        for inst_num, j_arr, o_arr, d_arr, dep_arr, conn_arr in zip(*ds):
            env.load_data(inst_num, j_arr, o_arr, d_arr, dep_arr, conn_arr)
            env.reset_state()
            build_inputs(env, K=2)
        print(f"  [PASS] {p}: {n_inst} instances pass ordering assertion")


@torch.no_grad()
def profile_breakdown(dpath: str, batch_size: int = 100) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatSchedNet().to(device).eval()
    env = make_fjsp_env()
    ds, _ = load_fjsp_dataset_and_baseline(dpath)

    # Warmup
    inst_num, j_arr, o_arr, d_arr, dep_arr, conn_arr = [x[0] for x in ds]
    bs = min(batch_size, inst_num)
    for lv in (1, 2, 3):
        env.load_data(bs, j_arr[:bs], o_arr[:bs], d_arr[:bs], dep_arr[:bs], conn_arr[:bs])
        env.reset_state()
        done = False
        while not done:
            inp = build_inputs(env, K=2, device=device)
            logp, _ = model(inp, level=lv)
            pick = logp.argmax(dim=-1)
            M = inp["elig"].shape[2]
            _, _, done = env.step(to_env_action(pick // M, pick % M, inp["head_row"]))

    print(f"\n--- Latency breakdown on {dpath} (bs={bs}) ---")
    for lv in (1, 2, 3):
        t_adapt, t_model, t_env = 0.0, 0.0, 0.0
        env.load_data(bs, j_arr[:bs], o_arr[:bs], d_arr[:bs], dep_arr[:bs], conn_arr[:bs])
        env.reset_state()
        done = False
        while not done:
            t0 = time.perf_counter()
            inp = build_inputs(env, K=2, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            logp, _ = model(inp, level=lv)
            pick = logp.argmax(dim=-1)
            M = inp["elig"].shape[2]
            act = to_env_action(pick // M, pick % M, inp["head_row"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            _, _, done = env.step(act)
            t3 = time.perf_counter()
            t_adapt += t1 - t0
            t_model += t2 - t1
            t_env += t3 - t2
        tot = t_adapt + t_model + t_env
        print(
            f"  Level {lv}: total={tot*1000/bs:.2f} ms/inst | "
            f"adapter={t_adapt*1000/bs:.2f}ms ({t_adapt/tot*100:.0f}%) | "
            f"model={t_model*1000/bs:.2f}ms ({t_model/tot*100:.0f}%) | "
            f"env={t_env*1000/bs:.2f}ms ({t_env/tot*100:.0f}%)"
        )

    res = evaluate_resched_baseline("ckpt/PPO/SD1-10x05.pth", [dpath], batch_size=bs, device=device)
    print(f"  ReSched baseline total: {res[0]['sec_per_inst']*1000:.2f} ms/inst (includes env+adapter+model)")


if __name__ == "__main__":
    print("=== 1. Dataset ordering assumption check ===")
    check_all_datasets_ordering()
    print("\n=== 2. Latency breakdown (10x5 & 30x10) ===")
    profile_breakdown("data/FJSP/TNNLS/SD1/10x5", batch_size=100)
    if (REPO_ROOT / "data/FJSP/TNNLS/SD1/30x10").exists():
        profile_breakdown("data/FJSP/TNNLS/SD1/30x10", batch_size=50)
