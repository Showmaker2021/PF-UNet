"""Sequential Batch Training Script for PF-UNet / ReSched
Runs all sizes sequentially to populate Table 1.
"""

import subprocess
import sys
import argparse

SIZES = ["10x5", "20x5", "15x10", "20x10"]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Train all FJSP sizes sequentially")
    parser.add_argument("--algo", type=str, choices=["reinforce", "ppo"], default="reinforce",
                        help="RL algorithm (default: reinforce)")
    parser.add_argument("--model", type=str, choices=["pfunet", "resched"], default="pfunet",
                        help="Model architecture (default: pfunet)")
    parser.add_argument("--dataset", type=str, choices=["SD1", "SD2"], default="SD1",
                        help="Dataset distribution (default: SD1)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (default: 2000)")
    parser.add_argument("--small-budget", action="store_true",
                        help="Quick test run on all sizes (80 epochs for reinforce)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, mps, cuda, cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("        BATCH SEQUENTIAL TRAINING SUITE FOR TABLE 1")
    print(f"Algorithm: {args.algo.upper()} | Model: {args.model.upper()} | Dataset: {args.dataset}")
    print(f"Sizes to train: {', '.join(SIZES)}")
    print("=" * 70)

    for idx, size in enumerate(SIZES, 1):
        print(f"\n[{idx}/{len(SIZES)}] Starting training for size {size}...")
        cmd = [
            sys.executable, "train.py",
            "--algo", args.algo,
            "--model", args.model,
            "--dataset", args.dataset,
            "--size", size,
            "--device", args.device,
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.small_budget:
            cmd.append("--small-budget")

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Error occurred while training size {size}! Exiting.")
            sys.exit(ret.returncode)
        print(f"Successfully finished size {size}!\n")

    print("=" * 70)
    print("ALL 4 SIZES COMPLETED SUCCESSFULLY!")
    print("All checkpoints and evaluation logs are saved in result/FJSP/")
    print("=" * 70)


if __name__ == "__main__":
    main()
