import sys
import os
from pathlib import Path
import numpy as np
import torch

# Ensure PPO folder is at the beginning of path
ppo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)

from utils import set_seed, load_dataset
from SchedulingEnvironment import FJSPEnv
from SchedulingModel import Model, build_model_input

def record_baseline():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(114514)

    # 1. Load 20 fixed instances from SD1 10x5 test dataset
    data_path = 'data/FJSP/TNNLS/SD1/10x5'
    if not os.path.exists(data_path):
        data_path = '../data/FJSP/TNNLS/SD1/10x5'
    dataset = load_dataset(data_path)
    # dataset structure: [instance_nums, job_idxs, op_idxs, durations, dependencies, connections]
    job_idx = dataset[1][0][:20]
    op_idx = dataset[2][0][:20]
    dur = dataset[3][0][:20]
    dep = dataset[4][0][:20]
    conn = dataset[5][0][:20]
    
    # 2. Load model checkpoint
    model_path = 'ckpt/PPO/SD1-10x05.pth'
    if not os.path.exists(model_path):
        model_path = '../ckpt/PPO/SD1-10x05.pth'
    
    model_params = {
        'action_space': 2,
        'embedding_dim': 128,
        'block_num': 2,
        'head_num': 8,
        'qkv_dim': 16,
        'ff_hidden_dim': 512,
    }
    model = Model(**model_params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.set_decode_type('greedy')

    # 3. Run greedy inference on the 20 instances
    env = FJSPEnv(generator_params=None)
    env.load_data(20, job_idx, op_idx, dur, dep, conn)
    state = env.reset_state()
    done = False
    with torch.no_grad():
        while not done:
            state_input, _ = build_model_input(state, env)
            state_input = [s.to(device) if isinstance(s, torch.Tensor) else s for s in state_input]
            action, _, _, _ = model(state_input)
            state, _, done = env.step(action)

    makespans = state.finish_time.max(axis=1).max(axis=1)
    print("Baseline 20 makespans:", makespans)

    save_path = Path(__file__).parent / "baseline_20_makespans.npy"
    np.save(save_path, makespans)
    print(f"Saved baseline 20 makespans to {save_path}")

if __name__ == '__main__':
    record_baseline()
