"""Sequential Batch Training Script for PF-UNet / ReSched
Runs all sizes for SD1 and/or SD2 sequentially to populate Table 1.
"""

import subprocess
import sys
import argparse
import time

SIZES = ["10x5", "20x5", "15x10", "20x10"]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Train all FJSP sizes sequentially")
    parser.add_argument("--algo", type=str, choices=["reinforce", "ppo"], default="reinforce",
                        help="RL algorithm (default: reinforce)")
    parser.add_argument("--model", type=str, choices=["pfunet", "resched"], default="pfunet",
                        help="Model architecture (default: pfunet)")
    parser.add_argument("--model-size", type=str, choices=["small", "base", "tiny"], default="small",
                        help="PF-UNet model variant: small (~930K params, matched 1:1 with ReSched), base (~3.6M params), or tiny (~230K params) (default: small)")
    parser.add_argument("--dataset", type=str, choices=["SD1", "SD2", "all"], default="SD1",
                        help="Dataset to train: SD1 (4 sizes), SD2 (4 sizes), or all (8 sizes total) (default: SD1)")
    parser.add_argument("--sizes", type=str, nargs="+", default=None,
                        help="Subset of sizes to run (default: 10x5 20x5 15x10 20x10)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (default: 400 for PPO, 2000 for REINFORCE)")
    parser.add_argument("--small-budget", action="store_true",
                        help="Quick test run on all sizes (80 epochs for reinforce)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, mps, cuda, cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_to_run = ["SD1", "SD2"] if args.dataset.lower() == "all" else [args.dataset.upper()]
    target_sizes = args.sizes if args.sizes else SIZES
    total_runs = len(datasets_to_run) * len(target_sizes)

    print("=" * 75)
    print("        BATCH SEQUENTIAL TRAINING SUITE FOR TABLE 1")
    print(f"Algorithm       : {args.algo.upper()}")
    model_desc = f"{args.model.upper()} ({args.model_size.upper()})" if args.model.lower() == "pfunet" else args.model.upper()
    print(f"Model           : {model_desc}")
    print(f"Datasets        : {', '.join(datasets_to_run)}")
    print(f"Sizes per ds    : {', '.join(target_sizes)}")
    print(f"Total Runs      : {total_runs} models to train")
    print("=" * 75)

    run_count = 0
    start_time_all = time.time()

    for ds in datasets_to_run:
        for size in target_sizes:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] Starting training: Dataset={ds}, Size={size}...")
            start_t = time.time()

            cmd = [
                sys.executable, "train.py",
                "--algo", args.algo,
                "--model", args.model,
                "--model-size", args.model_size,
                "--dataset", ds,
                "--size", size,
                "--device", args.device,
            ]
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.small_budget:
                cmd.append("--small-budget")

            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"[ERROR] Failed during Dataset={ds}, Size={size}! Return code: {ret.returncode}")
                sys.exit(ret.returncode)

            elapsed = time.time() - start_t
            print(f"[DONE] Completed {ds} {size} in {elapsed / 60:.1f} minutes!\n")

    total_elapsed = time.time() - start_time_all
    print("=" * 75)
    print(f"ALL {total_runs} MODELS COMPLETED SUCCESSFULLY in {total_elapsed / 3600:.2f} hours!")
    print("All checkpoints and evaluation logs are saved in result/FJSP/")
    print("=" * 75)


if __name__ == "__main__":
    main()
