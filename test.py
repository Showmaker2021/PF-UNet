"""Turnkey Benchmark Evaluation Script for PF-UNet / ReSched
Evaluates trained checkpoints across standard FJSP benchmarks.
"""

import os
import sys
import copy
import argparse
import importlib
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PF-UNet / ReSched Checkpoint")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained .pth model weights")
    parser.add_argument("--algo", type=str, choices=["reinforce", "ppo"], default="reinforce",
                        help="RL algorithm framework (default: reinforce)")
    parser.add_argument("--model", type=str, choices=["pfunet", "resched"], default="pfunet",
                        help="Model architecture (default: pfunet)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Execution device: auto (detects cuda -> mps -> cpu), cuda, mps, cpu")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Evaluation batch size (default: 200)")
    parser.add_argument("--inference-type", type=str, default="greedy",
                        choices=["greedy", "sampling", "aug_sample"],
                        help="Inference decoding mode (default: greedy)")
    parser.add_argument("--sample-times", type=int, default=100,
                        help="Number of samples if inference_type is sampling (default: 100)")
    return parser.parse_args()


def detect_device(preferred):
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path_abs = os.path.abspath(args.model_path)

    algo_folder = "REINFORCE" if args.algo.lower() == "reinforce" else "PPO"
    algo_dir = os.path.join(root_dir, algo_folder)

    # Change working directory so relative paths in ReSched configs resolve properly
    os.chdir(algo_dir)
    sys.path.insert(0, algo_dir)

    cfg = importlib.import_module("configs.fjsp")
    from SchedulingRunner import Runner

    logger_params = copy.deepcopy(cfg.logger_params)
    env_params = copy.deepcopy(cfg.env_params)
    optimizer_params = copy.deepcopy(cfg.optimizer_params)
    runner_params = copy.deepcopy(cfg.runner_params)

    # 1. Model Selection
    if args.model.lower() == "pfunet":
        model_params = copy.deepcopy(cfg.PFUNET_PARAMS)
        if args.algo.lower() == "reinforce":
            model_params["rl_algo"] = "reinforce"
        else:
            model_params["rl_algo"] = "ppo"
    else:
        model_params = copy.deepcopy(cfg.model_params)

    # 2. Configure Test Mode
    runner_params["test_only"] = True
    runner_params["checkpoint"] = None
    runner_params["model_path"] = model_path_abs

    # 3. Test params
    runner_params["test"]["inference_type"] = [args.inference_type]
    runner_params["test"]["batch_size"] = args.batch_size
    runner_params["test"]["sample_times"] = args.sample_times

    # 4. Device Configuration
    dev_str = detect_device(args.device)
    runner_params["use_cuda"] = (dev_str in ("cuda", "mps"))

    print("=" * 70)
    print("                     EVALUATION CONFIGURATION")
    print("=" * 70)
    print(f"Algorithm       : {args.algo.upper()}")
    print(f"Model           : {args.model.upper()}")
    print(f"Model Weights   : {runner_params['model_path']}")
    print(f"Device          : {dev_str.upper()}")
    print(f"Inference Type  : {args.inference_type}")
    print(f"Batch Size      : {args.batch_size}")
    print("=" * 70)

    runner = Runner(
        logger_params=logger_params,
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        runner_params=runner_params,
    )
    runner.run()


if __name__ == "__main__":
    main()
