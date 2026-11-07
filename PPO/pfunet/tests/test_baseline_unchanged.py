import sys
import os
from pathlib import Path
import numpy as np
import torch

ppo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)
os.chdir(ppo_dir)

from utils import set_seed, load_dataset
from SchedulingEnvironment import FJSPEnv
from SchedulingRunner import Runner
from configs.fjsp import (
    env_params,
    model_params,
    optimizer_params,
    runner_params,
    logger_params,
)

def test_baseline_unchanged():
    print("Testing baseline unchanged regression on 20 fixed instances...")
    set_seed(114514)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load recorded baseline
    baseline_file = Path(__file__).parent / "baseline_20_makespans.npy"
    assert baseline_file.exists(), f"Baseline file {baseline_file} not found!"
    baseline_makespans = np.load(baseline_file)
    
    # 2. Instantiate model via Runner to test full hook integration
    # Ensure model_type is 'resched'
    cfg_model_params = dict(model_params)
    assert cfg_model_params.get('model_type', 'resched') == 'resched', "Default model_type must be resched!"
    
    cfg_runner_params = dict(runner_params)
    cfg_runner_params['test_only'] = True
    model_path = 'ckpt/PPO/SD1-10x05.pth'
    if not os.path.exists(model_path):
        model_path = '../ckpt/PPO/SD1-10x05.pth'
    cfg_runner_params['model_path'] = model_path
    
    # Load dataset
    data_path = 'data/FJSP/TNNLS/SD1/10x5'
    if not os.path.exists(data_path):
        data_path = '../data/FJSP/TNNLS/SD1/10x5'
    dataset = load_dataset(data_path)
    job_idx = dataset[1][0][:20]
    op_idx = dataset[2][0][:20]
    dur = dataset[3][0][:20]
    dep = dataset[4][0][:20]
    conn = dataset[5][0][:20]
    
    # Load model via Runner
    runner = Runner(
        logger_params=logger_params,
        env_params=env_params,
        model_params=cfg_model_params,
        optimizer_params=optimizer_params,
        runner_params=cfg_runner_params,
    )
    
    model = runner.model.to(device)
    model.eval()
    model.set_decode_type('greedy')
    
    # Run greedy inference
    env = FJSPEnv(generator_params=None)
    env.load_data(20, job_idx, op_idx, dur, dep, conn)
    state = env.reset_state()
    done = False
    
    from SchedulingModel import build_model_input
    with torch.no_grad():
        while not done:
            state_input, _ = build_model_input(state, env)
            state_input = [s.to(device) if isinstance(s, torch.Tensor) else s for s in state_input]
            action, _, _, _ = model(state_input)
            state, _, done = env.step(action)
            
    current_makespans = state.finish_time.max(axis=1).max(axis=1)
    print("Baseline makespans:", baseline_makespans)
    print("Current makespans :", current_makespans)
    
    assert (current_makespans == baseline_makespans).all(), \
        f"Regression detected! Current != Baseline: diff = {np.abs(current_makespans - baseline_makespans)}"
        
    print("\n[SUCCESS] Baseline is 100% bitwise unchanged after hooks!")

if __name__ == '__main__':
    test_baseline_unchanged()
