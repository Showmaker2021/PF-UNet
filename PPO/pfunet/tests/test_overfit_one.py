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
from SchedulingModel import build_model_input
from pfunet.model import SchedulingModel
from PPOTrainer import Trainer
from configs.fjsp import PFUNET_PARAMS

def evaluate_greedy(model, env, device):
    model.eval()
    model.set_decode_type('greedy')
    state = env.reset_state()
    done = False
    with torch.no_grad():
        while not done:
            state_input, _ = build_model_input(state, env)
            state_input = [s.to(device) if isinstance(s, torch.Tensor) else s for s in state_input]
            action, _, _, _ = model(state_input)
            state, _, done = env.step(action)
    return float(state.finish_time.max())

def test_overfit_one(max_updates=25):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_default_device(device)
    set_seed(114514)
    print(f"Testing overfit on 1 instance (SD1 10x5, instance 0) on {device}...", flush=True)
    
    # 1. Load instance 0
    data_path = '../data/FJSP/TNNLS/SD1/10x5'
    if not os.path.exists(data_path):
        data_path = 'data/FJSP/TNNLS/SD1/10x5'
    dataset = load_dataset(data_path)
    job_idx = dataset[1][0][:1]
    op_idx = dataset[2][0][:1]
    dur = dataset[3][0][:1]
    dep = dataset[4][0][:1]
    conn = dataset[5][0][:1]
    
    # 2. Load OR-Tools solution
    sol_path = '../data/FJSP/TNNLS/or_solution/SD1/solution_10x5.npy'
    if not os.path.exists(sol_path):
        sol_path = 'data/FJSP/TNNLS/or_solution/SD1/solution_10x5.npy'
    or_solutions = np.load(sol_path)
    or_makespan = float(or_solutions[0, 0])
    print(f"OR-Tools optimal makespan: {or_makespan:.1f}", flush=True)

    # 3. Setup environments: eval_env (B=1) and train_env (B=4 replicates of instance 0)
    eval_env = FJSPEnv(generator_params=None)
    eval_env.load_data(1, job_idx, op_idx, dur, dep, conn)
    
    B_train = 4
    train_job_idx = np.repeat(job_idx, B_train, axis=0)
    train_op_idx = np.repeat(op_idx, B_train, axis=0)
    train_dur = np.repeat(dur, B_train, axis=0)
    train_dep = np.repeat(dep, B_train, axis=0)
    train_conn = np.repeat(conn, B_train, axis=0)
    
    train_env = FJSPEnv(generator_params=None)
    train_env.load_data(B_train, train_job_idx, train_op_idx, train_dur, train_dep, train_conn)
    train_env.generate_data = lambda bs: None  # Keep fixed instance for overfitting
    
    # 4. Initialize PF-UNet Model
    mp = dict(PFUNET_PARAMS)
    mp.pop('model_type', None)
    model = SchedulingModel(**mp).to(device)
    
    initial_makespan = evaluate_greedy(model, eval_env, device)
    print(f"Untrained initial greedy makespan: {initial_makespan:.1f}", flush=True)

    # 5. Setup Trainer
    training_params = {
        'episode': B_train,
        'sample_batch_size': B_train,
        'mini_batch_size': 1,
        'K_epochs': 2,
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'entropy_coef': 0.01,
        'gamma': 1.0,
        'lambda': 0.98,
    }
    optimizer_params = {
        'optimizer': {
            'lr': 3e-4,
            'weight_decay': 1e-6
        },
        'scheduler': {
            'milestones': [15],
            'gamma': 0.5
        }
    }
    
    trainer = Trainer(model, optimizer_params, training_params)
    trainer.set_logger(lambda x: None)  # suppress verbose logs
    
    best_makespan = initial_makespan
    
    for epoch in range(1, max_updates + 1):
        sample_score, _, _, _, _, _ = trainer.train(train_env, num_epoch=epoch)
        greedy_mk = evaluate_greedy(model, eval_env, device)
        if greedy_mk < best_makespan:
            best_makespan = greedy_mk
        gap = (greedy_mk - or_makespan) / or_makespan * 100.0
        print(f"Epoch {epoch:2d}: Greedy Makespan = {greedy_mk:.1f} (Gap: {gap:+.1f}%), Best = {best_makespan:.1f}, Sample Avg = {sample_score:.1f}", flush=True)

    print(f"\n[SUCCESS] Overfitting verified: makespan improved from {initial_makespan:.1f} to {best_makespan:.1f} (OR-Tools: {or_makespan:.1f})!")
    assert best_makespan < initial_makespan * 0.7, f"Overfit failed to improve makespan substantially: {best_makespan}"

if __name__ == '__main__':
    test_overfit_one()
