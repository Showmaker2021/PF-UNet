import sys
import os
import torch
import numpy as np

ppo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)

from SchedulingEnvironment import FJSPEnv
from SchedulingModel import build_model_input
from pfunet.model import SchedulingModel

def test_policy_interface():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Testing PF-UNet policy interface on device: {device}...")
    
    B = 4
    env = FJSPEnv(generator_params={'num_jobs': 10, 'num_machines': 5, 'max_processing_time': 20}, sd1=True)
    env.generate_data(B)
    state = env.reset_state()
    inputs, _ = build_model_input(state, env)
    inputs = [s.to(device) if isinstance(s, torch.Tensor) else s for s in inputs]
    act_mask = inputs[6]
    _, N, M = inputs[1].shape
    
    model = SchedulingModel().to(device)
    
    # 1. Test greedy decoding
    model.set_decode_type('greedy')
    out = model(inputs)
    assert len(out) == 4, f"Expected 4 outputs, got {len(out)}"
    act, prob_sel, value, prob_dist = out
    
    assert len(act) == 2, "Action should be a list/tuple of [seq_id, machine_id]"
    seq_id, machine_id = act
    assert seq_id.shape == (B,) and machine_id.shape == (B,), f"Action shapes invalid: {seq_id.shape}, {machine_id.shape}"
    assert prob_sel.shape == (B,), f"prob_sel shape invalid: {prob_sel.shape}"
    assert value.shape == (B,), f"value shape invalid: {value.shape}"
    assert prob_dist.shape == (B, N, M), f"prob_dist shape invalid: {prob_dist.shape}"
    
    # Probabilities sum to 1
    prob_sums = prob_dist.flatten(1).sum(-1)
    assert torch.allclose(prob_sums, torch.ones(B, device=device), atol=1e-5), \
        f"Probabilities do not sum to 1! Sums: {prob_sums}"
        
    # Infeasible entries are 0
    infeasible_mask = ~act_mask
    assert (prob_dist[infeasible_mask] == 0.0).all(), "Non-zero probability assigned to infeasible actions!"
    print("[PASS] Greedy decode: output shapes and probability constraints verified.")

    # 2. Test sampling decoding
    model.set_decode_type('sampling')
    out_samp = model(inputs)
    act_samp, prob_sel_samp, value_samp, prob_dist_samp = out_samp
    assert torch.allclose(prob_dist_samp.flatten(1).sum(-1), torch.ones(B, device=device), atol=1e-5)
    assert (prob_dist_samp[infeasible_mask] == 0.0).all()
    print("[PASS] Sampling decode: output shapes and probability constraints verified.")

    # 3. Test teacher_forcing decoding
    model.set_decode_type('teacher_forcing')
    out_tf = model(inputs, action=act)
    act_tf, prob_sel_tf, value_tf, prob_dist_tf = out_tf
    assert torch.equal(act_tf[0], seq_id) and torch.equal(act_tf[1], machine_id)
    assert torch.allclose(prob_sel_tf, prob_sel, atol=1e-5)
    print("[PASS] Teacher forcing decode: verified match with provided action.")

    # 4. Backward pass: check gradient flow through all parameters
    loss = value_tf.sum() - prob_sel_tf.clamp_min(1e-8).log().sum()
    loss.backward()
    
    param_count = 0
    grad_none_count = 0
    nan_grad_count = 0
    for name, param in model.named_parameters():
        param_count += 1
        if param.grad is None:
            grad_none_count += 1
            print(f"Warning: Param {name} has no grad!")
        elif torch.isnan(param.grad).any():
            nan_grad_count += 1
            print(f"Error: Param {name} has NaN grad!")
            
    assert grad_none_count == 0, f"{grad_none_count} parameters received no gradient!"
    assert nan_grad_count == 0, f"{nan_grad_count} parameters received NaN gradients!"
    print(f"[PASS] Backward pass: clean gradients flowed to all {param_count} parameters.")

    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"[INFO] Peak GPU VRAM allocated: {mem_mb:.2f} MB")
        
    print("\n[SUCCESS] All policy interface tests passed!")

if __name__ == '__main__':
    test_policy_interface()
