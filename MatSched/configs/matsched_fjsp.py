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


def make_jssp_env(num_jobs: int = 10, num_machines: int = 10, min_processing_time: int = 1, max_processing_time: int = 99):
    """Instantiate JSSPEnv without importing SchedulingModel."""
    ensure_ppo_on_path()
    from SchedulingEnvironment import JSSPEnv

    gen_params = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "min_processing_time": min_processing_time,
        "max_processing_time": max_processing_time,
    }
    return JSSPEnv(generator_params=gen_params)


def make_ffsp_env(num_jobs: int = 20, num_machines: int = 12, num_stages: int = 3, min_processing_time: int = 2, max_processing_time: int = 9):
    """Instantiate FFSPEnv without importing SchedulingModel."""
    ensure_ppo_on_path()
    from SchedulingEnvironment import FFSPEnv

    gen_params = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "num_stages": num_stages,
        "min_processing_time": min_processing_time,
        "max_processing_time": max_processing_time,
        "resource_flexibility": True,
        "machine_cnt_list": [4, 4, 4],
        "machine_flexibility": False,
    }
    return FFSPEnv(generator_params=gen_params)


def _load_reinforce_ffsp_module():
    """Load REINFORCE/configs/ffsp.py safely."""
    rf_ffsp_path = REPO_ROOT / "REINFORCE" / "configs" / "ffsp.py"
    spec = importlib.util.spec_from_file_location("_rf_configs_ffsp", str(rf_ffsp_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_ffsp_dataset_and_baseline(rel_data_path: str, instance_num: int = 1000) -> tuple[Any, Any]:
    """Load FFSP test dataset (.pt from Matnet) using REINFORCE/configs/ffsp.py loader."""
    rf_ffsp = _load_reinforce_ffsp_module()
    load_data_from_Matnet_files = rf_ffsp.load_data_from_Matnet_files

    abs_path = (REPO_ROOT / rel_data_path).resolve() if not os.path.isabs(rel_data_path) else Path(rel_data_path)
    dataset = load_data_from_Matnet_files(str(abs_path), instance_num=instance_num)
    # Matnet test instances do not have pre-stored optimal baseline (reported makespan directly)
    return dataset, None


def _load_reinforce_jssp_module():
    """Load REINFORCE/configs/jssp.py safely."""
    rf_jssp_path = REPO_ROOT / "REINFORCE" / "configs" / "jssp.py"
    spec = importlib.util.spec_from_file_location("_rf_configs_jssp", str(rf_jssp_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_jssp_dataset_and_baseline(rel_data_path: str) -> tuple[Any, Any]:
    """Load JSSP dataset (npy) and UB baseline using REINFORCE/configs/jssp.py loaders."""
    rf_jssp = _load_reinforce_jssp_module()
    load_data_from_L2D = rf_jssp.load_data_from_L2D
    load_benchmark_solution_for_JSSP = rf_jssp.load_benchmark_solution_for_JSSP

    abs_path = (REPO_ROOT / rel_data_path).resolve() if not os.path.isabs(rel_data_path) else Path(rel_data_path)
    dataset = load_data_from_L2D(str(abs_path))

    # Identify benchmark name (e.g., tai15x15, dmu20x15)
    stem = abs_path.stem
    baseline = load_benchmark_solution_for_JSSP(stem)
    return dataset, baseline


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
