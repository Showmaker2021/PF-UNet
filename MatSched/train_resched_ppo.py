"""Train original ReSched baseline with PPO for an exact number of epochs.

Preserves PPO/ and REINFORCE/ completely untouched.
Enables exact Equal-Budget (e.g. 400 epochs) comparison on JSSP/FJSP/FFSP.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train ReSched PPO baseline under equal budget")
    parser.add_argument("--problem", choices=["jssp", "fjsp", "ffsp"], default="jssp")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs (default 400)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_device_num", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ppo_dir = repo_root / "PPO"

    orig_cwd = os.getcwd()
    os.chdir(ppo_dir)
    sys.path.insert(0, str(ppo_dir))

    try:
        from SchedulingRunner import Runner

        cfg = importlib.import_module(f"configs.{args.problem}")
        logger_params = dict(cfg.logger_params)
        env_params = dict(cfg.env_params)
        model_params = dict(cfg.model_params)
        optimizer_params = dict(cfg.optimizer_params)
        runner_params = dict(cfg.runner_params)

        runner_params["test_only"] = False
        runner_params["training"]["epochs"] = args.epochs
        runner_params["seed"] = args.seed
        runner_params["cuda_device_num"] = args.cuda_device_num
        logger_params["save_file"] = False

        run_name = args.run_name or f"{args.problem.upper()}-10x10-PPO-{args.epochs}"
        logger_params["run_name"] = run_name

        if args.problem == "jssp":
            runner_params["test"]["dataset_path"] = [
                "../data/JSSP/L2D/BenchDataNmpy/tai15x15.npy",
                "../data/JSSP/L2D/BenchDataNmpy/tai20x15.npy",
                "../data/JSSP/L2D/BenchDataNmpy/tai20x20.npy",
                "../data/JSSP/L2D/BenchDataNmpy/tai30x15.npy",
                "../data/JSSP/L2D/BenchDataNmpy/tai30x20.npy",
                "../data/JSSP/L2D/BenchDataNmpy/tai50x15.npy",
                "../data/JSSP/L2D/BenchDataNmpy/tai50x20.npy",
                "../data/JSSP/L2D/BenchDataNmpy/tai100x20.npy",
            ]
            runner_params["test"]["inference_type"] = ["greedy"]

        runner = Runner(
            logger_params=logger_params,
            env_params=env_params,
            model_params=model_params,
            optimizer_params=optimizer_params,
            runner_params=runner_params,
        )
        runner.run()

    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
