"""Profile MatSched-Net parameter count, FLOPs, and CUDA inference latency across levels 1, 2, 3."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

MATSCHED_DIR = Path(__file__).resolve().parent
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from configs.matsched_fjsp import load_fjsp_dataset_and_baseline, make_fjsp_env
from evaluate import evaluate_matsched
from model import MatSchedNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatSchedNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MatSchedNet total params (all 3 levels + critic): {n_params:,}")

    env = make_fjsp_env()
    ds, bl = load_fjsp_dataset_and_baseline("data/FJSP/TNNLS/SD1/10x5")

    # Warmup all 3 levels
    for lv in (1, 2, 3):
        evaluate_matsched(model, env, ds, bl, batch_size=100, level=lv, device=device, measure_flops=False)

    for lv in (1, 2, 3):
        r_time = evaluate_matsched(model, env, ds, bl, batch_size=100, level=lv, device=device, measure_flops=False)
        r_flops = evaluate_matsched(model, env, ds, bl, batch_size=100, level=lv, device=device, measure_flops=True)
        print(
            f"SD1/10x5 Level {lv} | ms/inst={r_time['sec_per_inst']*1000:.3f} | "
            f"FLOPs/inst={r_flops['flops_per_inst']/1e6:.2f}M"
        )


if __name__ == "__main__":
    main()
