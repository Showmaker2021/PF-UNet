"""Turnkey Benchmark Evaluation Script for PF-UNet / ReSched
Evaluates trained checkpoints across standard FJSP benchmarks and synthetic datasets.
"""

import os
import sys
import copy
import argparse
import importlib
import torch


BENCHMARK_DATASETS = [
    '../data/FJSP/TNNLS/BenchData/Brandimarte',
    '../data/FJSP/TNNLS/BenchData/Hurink_rdata',
    '../data/FJSP/TNNLS/BenchData/Hurink_edata',
    '../data/FJSP/TNNLS/BenchData/Hurink_vdata',
]

SD1_DATASETS = [
    '../data/FJSP/TNNLS/SD1/10x5',
    '../data/FJSP/TNNLS/SD1/15x10',
    '../data/FJSP/TNNLS/SD1/20x5',
    '../data/FJSP/TNNLS/SD1/20x10',
    '../data/FJSP/TNNLS/SD1/30x10',
    '../data/FJSP/TNNLS/SD1/40x10',
]

SD2_DATASETS = [
    '../data/FJSP/TNNLS/SD2/10x5+mix',
    '../data/FJSP/TNNLS/SD2/15x10+mix',
    '../data/FJSP/TNNLS/SD2/20x5+mix',
    '../data/FJSP/TNNLS/SD2/20x10+mix',
    '../data/FJSP/TNNLS/SD2/30x10+mix',
    '../data/FJSP/TNNLS/SD2/40x10+mix',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PF-UNet / ReSched Checkpoint")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained .pth model weights")
    parser.add_argument("--algo", type=str, choices=["reinforce", "ppo"], default="reinforce",
                        help="RL algorithm framework (default: reinforce)")
    parser.add_argument("--model", type=str, choices=["pfunet", "resched"], default="pfunet",
                        help="Model architecture: pfunet or resched (default: pfunet)")
    parser.add_argument("--suite", type=str, default="all",
                        choices=["all", "benchmark", "sd1", "sd2", "custom"],
                        help="Dataset suite to evaluate: all (16 datasets), benchmark (4 public benchmarks), sd1 (6 datasets), sd2 (6 datasets), custom")
    parser.add_argument("--size", type=str, default=None,
                        choices=["10x5", "15x10", "20x5", "20x10", "30x10", "40x10"],
                        help="Optional specific problem size to filter in sd1 or sd2 suite")
    parser.add_argument("--dataset-path", type=str, nargs="+", default=None,
                        help="Explicit path(s) to dataset folder(s) (overrides --suite)")
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


def resolve_datasets(args):
    if args.dataset_path is not None:
        return args.dataset_path

    if args.suite == "benchmark":
        return BENCHMARK_DATASETS
    elif args.suite == "sd1":
        if args.size:
            return [f"../data/FJSP/TNNLS/SD1/{args.size}"]
        return SD1_DATASETS
    elif args.suite == "sd2":
        if args.size:
            return [f"../data/FJSP/TNNLS/SD2/{args.size}+mix"]
        return SD2_DATASETS
    elif args.suite == "all":
        return BENCHMARK_DATASETS + SD1_DATASETS + SD2_DATASETS
    else:
        raise ValueError(f"Unknown suite '{args.suite}'. Pass --dataset-path for custom paths.")


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

    # 3. Configure Test Datasets explicitly
    target_datasets = resolve_datasets(args)
    runner_params["test"]["dataset_path"] = target_datasets
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
    print(f"Suite           : {args.suite}")
    print(f"Num Datasets    : {len(target_datasets)}")
    print("Target Datasets :")
    for d in target_datasets:
        print(f"  - {d}")
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
