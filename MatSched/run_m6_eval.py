"""Comprehensive M5/M6 evaluation: Gate G1 (nested vs independent) and Gate G2 (Pareto sweep)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch

MATSCHED_DIR = Path(__file__).resolve().parent
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from configs.matsched_fjsp import REPO_ROOT, load_fjsp_dataset_and_baseline, make_fjsp_env
from evaluate import evaluate_matsched, evaluate_resched_baseline
from model import MatSchedNet


def load_matsched_ckpt(ckpt_path: str, device: torch.device) -> MatSchedNet:
    ckpt = torch.load(REPO_ROOT / ckpt_path, map_location=device)
    model = MatSchedNet(**ckpt["cfg"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="MatSched/checkpoints/m5_v2_10x5_kd/best.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_fjsp_env()

    m5_path = args.ckpt
    if not (REPO_ROOT / m5_path).exists():
        m5_path = "MatSched/checkpoints/m5_mini_10x5_kd/best.pt"

    print(f"Evaluating checkpoint: {m5_path}")
    print("=== 1. Gate G1: Nested pi1/pi2/pi3 vs Independent d=32 / d=64 & ReSched ===")
    datasets_g1 = [
        "data/FJSP/TNNLS/SD1/10x5",
        "data/FJSP/TNNLS/SD1/15x10",
        "data/FJSP/TNNLS/SD2/10x5+mix",
        "data/FJSP/TNNLS/BenchData/Brandimarte",
    ]
    m5_model = load_matsched_ckpt(m5_path, device)

    d32_ckpt = REPO_ROOT / "MatSched/checkpoints/indep_d32_10x5/best.pt"
    d64_ckpt = REPO_ROOT / "MatSched/checkpoints/indep_d64_10x5/best.pt"
    m_d32 = load_matsched_ckpt("MatSched/checkpoints/indep_d32_10x5/best.pt", device) if d32_ckpt.exists() else None
    m_d64 = load_matsched_ckpt("MatSched/checkpoints/indep_d64_10x5/best.pt", device) if d64_ckpt.exists() else None

    resched_10 = evaluate_resched_baseline("ckpt/PPO/SD1-10x05.pth", datasets_g1, batch_size=100, device=device)
    resched_map = {r["dataset"]: r for r in resched_10}

    for dpath in datasets_g1:
        ds, bl = load_fjsp_dataset_and_baseline(dpath)
        print(f"\nDataset: {dpath}")
        rr = resched_map[dpath]
        print(f"  ReSched (SD1-10x05.pth) | mk={rr['makespan']:.3f} | gap={rr['gap']:.3f}% | ms/inst={rr['sec_per_inst']*1000:.2f}")
        for lv in (1, 2, 3):
            r = evaluate_matsched(m5_model, env, ds, bl, batch_size=100, level=lv, device=device)
            print(f"  MatSched Nested L{lv}     | mk={r['makespan']:.3f} | gap={r['gap']:.3f}% | ms/inst={r['sec_per_inst']*1000:.2f}")
        if m_d32 is not None:
            r32 = evaluate_matsched(m_d32, env, ds, bl, batch_size=100, level=3, device=device)
            print(f"  Indep fixed d=32        | mk={r32['makespan']:.3f} | gap={r32['gap']:.3f}% | ms/inst={r32['sec_per_inst']*1000:.2f}")
        if m_d64 is not None:
            r64 = evaluate_matsched(m_d64, env, ds, bl, batch_size=100, level=3, device=device)
            print(f"  Indep fixed d=64        | mk={r64['makespan']:.3f} | gap={r64['gap']:.3f}% | ms/inst={r64['sec_per_inst']*1000:.2f}")

    print("\n=== 2. Gate G2: Router Pareto Sweep (norm_entropy & margin) ===")
    ent_grid: List[Tuple[float, float]] = [(0.15, 0.25), (0.30, 0.45), (0.50, 0.65), (0.75, 0.85)]
    margin_grid: List[Tuple[float, float]] = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20)]

    for dpath, bs in (("data/FJSP/TNNLS/SD1/10x5", 100), ("data/FJSP/TNNLS/SD1/15x10", 50), ("data/FJSP/TNNLS/SD1/30x10", 25)):
        ds, bl = load_fjsp_dataset_and_baseline(dpath)
        rr = evaluate_resched_baseline("ckpt/PPO/SD1-10x05.pth", [dpath], batch_size=bs, device=device)[0]
        print(f"\nPareto on {dpath} (bs={bs}) | ReSched(10x05): gap={rr['gap']:.3f}%, ms/inst={rr['sec_per_inst']*1000:.2f}")
        r_l1 = evaluate_matsched(m5_model, env, ds, bl, batch_size=bs, level=1, device=device)
        print(f"  [L1 fixed]              | gap={r_l1['gap']:.3f}% | ms/inst={r_l1['sec_per_inst']*1000:.2f} | L1=100.0% L2=  0.0% L3=  0.0%")
        for metric, grid in (("norm_entropy", ent_grid), ("margin", margin_grid)):
            for t1, t2 in grid:
                r = evaluate_matsched(m5_model, env, ds, bl, batch_size=bs, tau=(t1, t2), metric=metric, device=device)
                lf = r["level_frac"]
                print(
                    f"  {metric:12s} ({t1:.2f},{t2:.2f}) | gap={r['gap']:.3f}% | ms/inst={r['sec_per_inst']*1000:.2f} | "
                    f"L1={lf[1]*100:5.1f}% L2={lf[2]*100:5.1f}% L3={lf[3]*100:5.1f}%"
                )
        r_l3 = evaluate_matsched(m5_model, env, ds, bl, batch_size=bs, level=3, device=device)
        print(f"  [L3 fixed]              | gap={r_l3['gap']:.3f}% | ms/inst={r_l3['sec_per_inst']*1000:.2f} | L1=  0.0% L2=  0.0% L3=100.0%")


if __name__ == "__main__":
    main()
