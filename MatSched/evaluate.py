"""Evaluation script for MatSched-Net (fixed levels, adaptive router, Pareto) and ReSched baseline (M0)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

MATSCHED_DIR = Path(__file__).resolve().parent
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from adapter import build_inputs, to_env_action
from configs.matsched_fjsp import REPO_ROOT, ensure_ppo_on_path, load_fjsp_dataset_and_baseline, make_fjsp_env
from model import MatSchedNet
from router import act_adaptive


@torch.no_grad()
def evaluate_matsched(
    model: MatSchedNet,
    env,
    dataset,
    baseline: Optional[np.ndarray],
    batch_size: int = 100,
    level: Optional[int] = 3,
    tau: Optional[Tuple[float, float]] = None,
    greedy: bool = True,
    K: int = 2,
    metric: str = "norm_entropy",
    device: Optional[torch.device | str] = None,
    measure_flops: bool = False,
) -> Dict[str, object]:
    """Evaluate MatSchedNet at fixed `level` or adaptive router (`tau != None`)."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.eval()

    all_scores: List[np.ndarray] = []
    level_counts = torch.zeros(4, dtype=torch.long)  # indices 1,2,3
    total_time = 0.0
    total_inst = 0
    total_flops = 0

    for inst_num, j_arr, o_arr, d_arr, dep_arr, conn_arr in zip(*dataset):
        ep = 0
        while ep < inst_num:
            bs = min(batch_size, inst_num - ep)
            env.load_data(
                bs,
                j_arr[ep : ep + bs],
                o_arr[ep : ep + bs],
                d_arr[ep : ep + bs],
                dep_arr[ep : ep + bs],
                conn_arr[ep : ep + bs],
            )
            env.reset_state()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            flop_ctx = None
            if measure_flops and total_inst == 0:
                from torch.utils.flop_counter import FlopCounterMode
                flop_ctx = FlopCounterMode(display=False)
                flop_ctx.__enter__()

            done = False
            while not done:
                if tau is not None:
                    action, lv_used = act_adaptive(
                        model, env, tau=tau, greedy=greedy, K=K, metric=metric, device=device
                    )
                    for lv in (1, 2, 3):
                        level_counts[lv] += int((lv_used == lv).sum().item())
                else:
                    inp = build_inputs(env, K=K, device=device)
                    logp, _ = model(inp, level=level)
                    p = logp.exp()
                    pick = p.argmax(dim=-1) if greedy else torch.distributions.Categorical(probs=p).sample()
                    M = inp["elig"].shape[2]
                    action = to_env_action(pick // M, pick % M, inp["head_row"])
                    level_counts[level] += bs

                state, _, done = env.step(action)

            if flop_ctx is not None:
                flop_ctx.__exit__(None, None, None)
                total_flops = flop_ctx.get_total_flops() // bs

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += time.perf_counter() - t0
            total_inst += bs
            all_scores.append(state.finish_time.max(axis=1).max(axis=1))
            ep += bs

    scores = np.concatenate(all_scores, axis=0)
    gap = float(np.mean((scores / baseline - 1.0) * 100.0)) if baseline is not None else None
    tot_steps = int(level_counts[1:].sum().item())
    lv_frac = {lv: (float(level_counts[lv].item()) / tot_steps if tot_steps else 0.0) for lv in (1, 2, 3)}
    return {
        "makespan": float(scores.mean()),
        "gap": gap,
        "sec_per_inst": total_time / max(total_inst, 1),
        "level_frac": lv_frac,
        "flops_per_inst": total_flops if measure_flops else None,
    }


@torch.no_grad()
def evaluate_resched_baseline(
    ckpt_path: str,
    dataset_paths: List[str],
    batch_size: int = 200,
    device: Optional[torch.device | str] = None,
    measure_flops: bool = False,
) -> List[Dict[str, object]]:
    """M0 helper: evaluate ReSched PPO checkpoint fairly without modifying PPO/."""
    ensure_ppo_on_path()
    from SchedulingEnvironment import FJSPEnv
    from SchedulingModel import Model as ReSchedModel, build_model_input

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    model_params = {
        "action_space": 2,
        "embedding_dim": 128,
        "block_num": 2,
        "head_num": 8,
        "qkv_dim": 16,
        "ff_hidden_dim": 512,
    }
    model = ReSchedModel(**model_params).to(device)
    ckpt = torch.load(REPO_ROOT / ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    model.set_decode_type("greedy")

    env = FJSPEnv()
    results = []
    for dpath in dataset_paths:
        dataset, baseline = load_fjsp_dataset_and_baseline(dpath)
        all_scores = []
        total_time = 0.0
        total_inst = 0
        flops_per_inst = None

        for inst_num, j_arr, o_arr, d_arr, dep_arr, conn_arr in zip(*dataset):
            ep = 0
            while ep < inst_num:
                bs = min(batch_size, inst_num - ep)
                env.load_data(
                    bs,
                    j_arr[ep : ep + bs],
                    o_arr[ep : ep + bs],
                    d_arr[ep : ep + bs],
                    dep_arr[ep : ep + bs],
                    conn_arr[ep : ep + bs],
                )
                state = env.reset_state()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()

                flop_ctx = None
                if measure_flops and flops_per_inst is None:
                    from torch.utils.flop_counter import FlopCounterMode
                    flop_ctx = FlopCounterMode(display=False)
                    flop_ctx.__enter__()

                done = False
                while not done:
                    m_in, _ = build_model_input(state, env)
                    m_in = [t.to(device) if isinstance(t, torch.Tensor) else t for t in m_in]
                    action, _, _, _ = model(m_in)
                    state, _, done = env.step(action)

                if flop_ctx is not None:
                    flop_ctx.__exit__(None, None, None)
                    flops_per_inst = flop_ctx.get_total_flops() // bs

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                total_time += time.perf_counter() - t0
                total_inst += bs
                all_scores.append(state.finish_time.max(axis=1).max(axis=1))
                ep += bs

        scores = np.concatenate(all_scores, axis=0)
        gap = float(np.mean((scores / baseline - 1.0) * 100.0)) if baseline is not None else None
        results.append(
            {
                "dataset": dpath,
                "makespan": float(scores.mean()),
                "gap": gap,
                "sec_per_inst": total_time / max(total_inst, 1),
                "flops_per_inst": flops_per_inst,
            }
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["resched_m0", "matsched", "pareto"], default="resched_m0")
    p.add_argument("--ckpt", type=str, default="ckpt/PPO/SD1-10x05.pth")
    p.add_argument("--datasets", nargs="+", default=["data/FJSP/TNNLS/SD1/10x5", "data/FJSP/TNNLS/SD1/15x10"])
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--router_metric", choices=["norm_entropy", "margin", "top1"], default="norm_entropy")
    p.add_argument("--flops", action="store_true")
    args = p.parse_args()

    if args.mode == "resched_m0":
        rows = evaluate_resched_baseline(args.ckpt, args.datasets, batch_size=args.batch_size, measure_flops=args.flops)
        for r in rows:
            gap_s = f"{r['gap']:.3f}%" if r["gap"] is not None else "N/A"
            fl_s = f"{r['flops_per_inst']/1e6:.2f}M" if r["flops_per_inst"] is not None else "N/A"
            print(
                f"{r['dataset']} | mk={r['makespan']:.3f} | gap={gap_s} | "
                f"ms/inst={r['sec_per_inst']*1000:.3f} | FLOPs/inst={fl_s}"
            )
    else:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        cfg_m = ckpt["cfg"]["model"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MatSchedNet(**cfg_m).to(device)
        model.load_state_dict(ckpt["model"])
        env = make_fjsp_env()
        for dpath in args.datasets:
            ds, bl = load_fjsp_dataset_and_baseline(dpath)
            if args.mode == "matsched":
                r = evaluate_matsched(
                    model, env, ds, bl, batch_size=args.batch_size, level=args.level, device=device, measure_flops=args.flops
                )
                print(f"{dpath} [L{args.level}] | mk={r['makespan']:.3f} | gap={r['gap']:.3f}% | ms/inst={r['sec_per_inst']*1000:.3f}")
            else:
                grid = (
                    [(0.15, 0.25), (0.30, 0.40), (0.45, 0.55), (0.60, 0.70), (0.75, 0.85)]
                    if args.router_metric == "norm_entropy"
                    else [(0.02, 0.05), (0.05, 0.10), (0.08, 0.15), (0.12, 0.25), (0.20, 0.35)]
                )
                for t1, t2 in grid:
                    r = evaluate_matsched(
                        model, env, ds, bl, batch_size=args.batch_size, tau=(t1, t2), metric=args.router_metric, device=device
                    )
                    print(f"{dpath} [{args.router_metric}] tau=({t1},{t2}) | gap={r['gap']:.3f}% | ms/inst={r['sec_per_inst']*1000:.3f} | lv={r['level_frac']}")
