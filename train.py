"""Turnkey Training Script for PF-UNet / ReSched
Supports Apple Silicon (M4 / MPS), NVIDIA CUDA, and CPU.
"""

import os
import sys
import copy
import argparse
import importlib
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train PF-UNet / ReSched on FJSP")
    parser.add_argument("--size", type=str, default="10x5",
                        choices=["10x5", "15x10", "20x5", "20x10"],
                        help="FJSP problem size to train on: 10x5, 15x10, 20x5, 20x10 (default: 10x5)")
    parser.add_argument("--dataset", type=str, default="SD1", choices=["SD1", "SD2"],
                        help="Training dataset distribution: SD1 (standard) or SD2 (mixed) (default: SD1)")
    parser.add_argument("--algo", type=str, choices=["reinforce", "ppo"], default="reinforce",
                        help="RL algorithm to run (default: reinforce)")
    parser.add_argument("--model", type=str, choices=["pfunet", "resched"], default="pfunet",
                        help="Model architecture (default: pfunet)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: 2000 for full, or 80 for small budget)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size (default: auto -> 50 for small, 24 for larger sizes as in paper)")
    parser.add_argument("--mini-batch-size", type=int, default=50,
                        help="PPO mini batch size (default: 50)")
    parser.add_argument("--k-epochs", type=int, default=None,
                        help="PPO K update epochs (default: 4 from config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--small-budget", action="store_true",
                        help="Quick test run (REINFORCE: 80 epochs, PPO: 400 epochs)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Execution device: auto (detects cuda -> mps -> cpu), cuda, mps, cpu")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom experiment run name for logging")
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

    # 2. Problem Size & Dataset Configuration (Jobs x Machines)
    jobs_str, mach_str = args.size.split("x")
    num_jobs, num_machines = int(jobs_str), int(mach_str)
    env_params["generate_param"]["num_jobs"] = num_jobs
    env_params["generate_param"]["num_machines"] = num_machines
    is_sd1 = (args.dataset.upper() == "SD1")
    env_params["generate_param"]["SD1"] = is_sd1

    # Match exact generator specifications from paper:
    # SD1: duration in [1, 20], ops in [0.8n, 1.2n]
    # SD2: duration in [1, 99], ops in [1, n]
    if is_sd1:
        env_params["generate_param"]["max_processing_time"] = 20
        val_data_path = f"../data/FJSP/TNNLS/data_train_vali/SD1/{num_jobs}x{num_machines}"
    else:
        env_params["generate_param"]["min_processing_time"] = 1
        env_params["generate_param"]["max_processing_time"] = 99
        # In SD2 (Wang et al., 2024b / TNNLS), each job has num_machines operations with mixed machine flexibility,
        # perfectly matching the pre-generated validation and test data (*+mix)
        env_params["generate_param"].pop("min_operation", None)
        env_params["generate_param"].pop("max_operation", None)
        val_data_path = f"../data/FJSP/TNNLS/data_train_vali/SD2/{num_jobs}x{num_machines}+mix"

    if os.path.exists(os.path.join(algo_dir, val_data_path)):
        runner_params["validation"]["dataset_path"] = val_data_path

    # 3. Configure Training Mode
    runner_params["test_only"] = False
    runner_params["checkpoint"] = None
    runner_params["model_path"] = None
    runner_params["seed"] = args.seed

    # 4. Epochs, Batch Size & Episodes (Auto-matched to ReSched Appendix B.2)
    # "Due to resource constraints, for larger sized problem, 
    # the number of training instances per epoch is reduced to 500 and the batch size to 24."
    is_large_size = (num_jobs >= 15 or num_machines >= 10)
    effective_batch_size = args.batch_size if args.batch_size is not None else (24 if is_large_size else 50)
    effective_episodes = 500 if is_large_size else 1000

    if args.epochs is not None:
        runner_params["training"]["epochs"] = args.epochs
    elif args.small_budget:
        runner_params["training"]["epochs"] = 80 if args.algo.lower() == "reinforce" else 400

    if args.algo.lower() == "reinforce":
        runner_params["training"]["batch_size"] = effective_batch_size
        runner_params["training"]["episode"] = effective_episodes
    else:
        runner_params["training"]["sample_batch_size"] = effective_batch_size
        mini_bs = args.mini_batch_size if args.mini_batch_size is not None else effective_batch_size
        runner_params["training"]["mini_batch_size"] = min(effective_batch_size, mini_bs)
        if args.k_epochs is not None:
            runner_params["training"]["K_epochs"] = args.k_epochs

    # 5. Device Configuration
    dev_str = detect_device(args.device)
    runner_params["use_cuda"] = (dev_str in ("cuda", "mps"))

    # 6. Test Suite to run immediately after training
    # Match the datasets evaluated in Table 1
    test_paths = [
        f"../data/FJSP/TNNLS/SD1/{num_jobs}x{num_machines}",
        "../data/FJSP/TNNLS/BenchData/Brandimarte",
        "../data/FJSP/TNNLS/BenchData/Hurink_rdata",
        "../data/FJSP/TNNLS/BenchData/Hurink_edata",
        "../data/FJSP/TNNLS/BenchData/Hurink_vdata",
    ]
    runner_params["test"]["dataset_path"] = [p for p in test_paths if os.path.exists(os.path.join(algo_dir, p))]

    # 7. Logger Run Name
    default_run_name = f"{args.dataset.upper()}-{num_jobs}x{num_machines:02d}-{args.model.upper()}-{args.algo.upper()}"
    logger_params["run_name"] = args.run_name if args.run_name else default_run_name
    logger_params["save_file"] = True
    logger_params["folder_path"] = "../result"

    print("=" * 70)
    print("                     TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Algorithm       : {args.algo.upper()}")
    print(f"Model           : {args.model.upper()}")
    print(f"Dataset         : {args.dataset.upper()}")
    print(f"Problem Size    : {num_jobs} Jobs x {num_machines} Machines")
    print(f"Device          : {dev_str.upper()}")
    print(f"Total Epochs    : {runner_params['training']['epochs']}")
    print(f"Batch Size      : {effective_batch_size}")
    if args.algo.lower() == "reinforce":
        print(f"Episodes/Epoch  : {effective_episodes}")
    else:
        print(f"PPO Mini-Batch  : {runner_params['training']['mini_batch_size']}")
        print(f"PPO K-Epochs    : {runner_params['training']['K_epochs']}")
    print(f"Random Seed     : {runner_params['seed']}")
    print(f"Log Folder      : {logger_params['folder_path']}/{logger_params['run_name']}")
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
