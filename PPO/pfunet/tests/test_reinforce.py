import sys
import os
import torch
import numpy as np

# Setup paths
ppo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_dir = os.path.dirname(ppo_dir)
rf_dir = os.path.join(root_dir, 'REINFORCE')

for p in [ppo_dir, rf_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(rf_dir)

from utils import set_seed, load_dataset
from SchedulingEnvironment import FJSPEnv
from SchedulingModel import build_model_input
from pfunet.model import SchedulingModel
from REINFORCETrainer import Trainer
from configs.fjsp import PFUNET_PARAMS

def test_reinforce_pipeline():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_default_device(device)
    set_seed(114514)
    print(f"Testing PF-UNet REINFORCE pipeline on {device}...", flush=True)

    # 1. Verify model interface in REINFORCE mode
    mp = dict(PFUNET_PARAMS)
    mp.pop('model_type', None)
    model = SchedulingModel(**mp).to(device)
    assert model.critic is None, "Critic must be None in REINFORCE mode!"
    print("[PASS] SchedulingModel initialized without Critic for REINFORCE.", flush=True)

    # 2. Test forward pass returning 2-tuple (action, prob)
    env = FJSPEnv(generator_params={'num_jobs': 4, 'num_machines': 3, 'max_processing_time': 20}, sd1=True)
    env.generate_data(2)
    state = env.reset_state()
    inputs, _ = build_model_input(state, env)
    
    model.set_decode_type('sampling')
    out = model(inputs)
    assert len(out) == 2, f"Expected 2-tuple output in REINFORCE mode, got {len(out)}"
    act, prob = out
    assert prob.shape == (2,), f"Expected prob shape (2,), got {prob.shape}"
    print("[PASS] Model forward returns exactly 2-tuple: (action, prob).", flush=True)

    # 3. Test backward pass and gradient flow
    loss = - prob.log().sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient!"
    print("[PASS] Gradients flow cleanly to all model parameters.", flush=True)

    # 4. Run short multi-epoch training with REINFORCE Trainer
    train_env = FJSPEnv(generator_params={'num_jobs': 10, 'num_machines': 5, 'max_processing_time': 20}, sd1=True)
    train_params = {
        'epochs': 5,
        'episode': 4,
        'batch_size': 2,
        'discount_factor': 0.99
    }
    opt_params = {
        'optimizer': {'lr': 1e-4, 'weight_decay': 1e-6},
        'scheduler': {'milestones': [10], 'gamma': 0.5}
    }
    
    trainer = Trainer(model, opt_params, train_params)
    trainer.set_logger(lambda x: None)
    
    print("\nRunning 5 epochs of REINFORCE training on GPU...", flush=True)
    for ep in range(1, 6):
        score, loss_val, _ = trainer.train(train_env)
        print(f"Epoch {ep}: Sample Score = {score:.2f}, REINFORCE Loss = {loss_val:.4f}", flush=True)

    print("\n[SUCCESS] PF-UNet REINFORCE pipeline verified 100% end-to-end!")

if __name__ == '__main__':
    test_reinforce_pipeline()
