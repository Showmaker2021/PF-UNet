"""FJSP configuration and dataset/env helpers reusing ReSched infrastructure."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PPO_DIR = REPO_ROOT / "PPO"


def _load_ppo_fjsp_module():
    """Load PPO/configs/fjsp.py without colliding with MatSched/configs/."""
    fjsp_path = PPO_DIR / "configs" / "fjsp.py"
    spec = importlib.util.spec_from_file_location("_ppo_configs_fjsp", str(fjsp_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def ensure_ppo_on_path() -> None:
    """Add PPO directory to sys.path to reuse FJSPEnv and SD/benchmark loaders."""
    ppo_str = str(PPO_DIR)
    if ppo_str not in sys.path:
        sys.path.append(ppo_str)


def make_fjsp_env(num_jobs: int = 10, num_machines: int = 5, sd1: bool = True, max_processing_time: int = 20):
    """Instantiate FJSPEnv without importing SchedulingModel."""
    ensure_ppo_on_path()
    from SchedulingEnvironment import FJSPEnv

    gen_params = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "max_processing_time": max_processing_time,
    }
    return FJSPEnv(generator_params=gen_params, sd1=sd1)


def load_fjsp_dataset_and_baseline(rel_data_path: str) -> tuple[Any, Any]:
    """Load dataset and OR/UB baseline using PPO/configs/fjsp.py loaders.

    PPO/configs/fjsp.py uses relative paths like '../data/...' assuming cwd == PPO/.
    We temporarily chdir to PPO_DIR to invoke them safely without editing PPO/.
    """
    ppo_fjsp = _load_ppo_fjsp_module()
    load_data_from_SD = ppo_fjsp.load_data_from_SD
    get_benchmark_name = ppo_fjsp.get_benchmark_name
    load_benchmark_solution = ppo_fjsp.load_benchmark_solution
    load_or_solution_for_FJSP = ppo_fjsp.load_or_solution_for_FJSP

    if not rel_data_path.startswith("..") and not os.path.isabs(rel_data_path):
        # Convert repo-root-relative path (e.g., 'data/FJSP/...') to PPO-relative path
        ppo_rel = os.path.relpath(REPO_ROOT / rel_data_path, PPO_DIR)
    else:
        ppo_rel = rel_data_path

    prev_cwd = os.getcwd()
    try:
        os.chdir(PPO_DIR)
        dataset = load_data_from_SD(ppo_rel)
        if "BenchData" in ppo_rel:
            bname = get_benchmark_name(ppo_rel)
            baseline = load_benchmark_solution(bname) if bname else None
        elif "data_train_vali" in ppo_rel:
            baseline = None
        else:
            p = Path(ppo_rel)
            source, data_name = p.parent.name, p.name
            or_path = f"../data/FJSP/TNNLS/or_solution/{source}/solution_{data_name}.npy"
            baseline = load_or_solution_for_FJSP(or_path)
    finally:
        os.chdir(prev_cwd)

    return dataset, baseline


DEFAULT_CONFIG = {
    "env": {
        "num_jobs": 10,
        "num_machines": 5,
        "sd1": True,
        "max_processing_time": 20,
    },
    "model": {
        "dmax": 128,
        "num_layers": 3,
        "K": 2,
        "head_dim": 16,
        "ff_mult": 2,
        "use_m2o": True,
        "use_mid_out": True,
        "use_cc": True,
        "use_pair_ffn": False,
        "use_out_ffn": False,
        "shared_head": False,
        "fixed_width": None,
    },
    "ppo": {
        "gamma": 1.0,
        "gae_lambda": 0.98,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "entropy_coef": 0.01,
        "k_epochs": 4,
        "lr": 6e-4,
        "lr_min": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip": 1.0,
        "episodes_per_iter": 32,
        "rollout_batch_size": 32,
        "minibatch_size": 1024,
        "use_bf16": True,
    },
    "kd": {
        "enabled": False,
        "mode": "kl",             # 'kl', 'pg', or 'both' (Ablation A6)
        "beta1": 0.015,
        "beta2": 0.02,
        "start_iter": 100,
        "dagger_weight": 0.10,
        "dagger_episodes": 8,
        "router_metric": "norm_entropy",
        "tau_range": (0.20, 0.70),  # (0.20, 0.70) for norm_entropy; (0.02, 0.25) for margin
    },
}
