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
from pfunet.modules import build_pf_obs, Obs

def test_invariance():
    print("Testing permutation equivariance and scale invariance on PF-UNet core...")
    torch.manual_seed(114514)
    model = SchedulingModel().eval()
    
    env = FJSPEnv(generator_params={'num_jobs': 6, 'num_machines': 4, 'max_processing_time': 20}, sd1=True)
    env.generate_data(1)
    state = env.reset_state()
    inputs, _ = build_model_input(state, env)
    
    with torch.no_grad():
        obs = build_pf_obs(inputs)
        orig_logits, orig_val = model.core(obs)  # [1, M, J], [1]
        B, M, J = orig_logits.shape
        
        # 1. Job permutation equivariance
        # Permute job indices 0..J-1
        p_J = torch.randperm(J)
        obs_perm_J = Obs(
            xy=obs.xy[:, :, p_J, :],
            feats=obs.feats[:, :, p_J, :],
            raw=obs.raw[:, :, p_J, :],
            mask_pair=obs.mask_pair[:, :, p_J],
            const=obs.const,
            head_idx=obs.head_idx[:, p_J],
            has_head=obs.has_head[:, p_J],
            Ja=obs.Ja,
            S=obs.S
        )
        perm_J_logits, perm_J_val = model.core(obs_perm_J)
        expected_J_logits = orig_logits[:, :, p_J]
        
        # Check masked valid elements
        valid_mask_J = obs_perm_J.mask_pair
        assert torch.allclose(perm_J_logits[valid_mask_J], expected_J_logits[valid_mask_J], atol=1e-4), \
            f"Job permutation equivariance failed! Max diff: {(perm_J_logits[valid_mask_J] - expected_J_logits[valid_mask_J]).abs().max()}"
        assert torch.allclose(perm_J_val, orig_val, atol=1e-4), "Critic value changed after job permutation!"
        print("[PASS] Job permutation equivariance holds (logits permute accordingly, value invariant).")

        # 2. Machine permutation equivariance
        p_M = torch.randperm(M)
        obs_perm_M = Obs(
            xy=obs.xy[:, p_M, :, :],
            feats=obs.feats[:, p_M, :, :],
            raw=obs.raw[:, p_M, :, :],
            mask_pair=obs.mask_pair[:, p_M, :],
            const=obs.const[:, p_M, :],
            head_idx=obs.head_idx,
            has_head=obs.has_head,
            Ja=obs.Ja,
            S=obs.S
        )
        perm_M_logits, perm_M_val = model.core(obs_perm_M)
        expected_M_logits = orig_logits[:, p_M, :]
        valid_mask_M = obs_perm_M.mask_pair
        
        assert torch.allclose(perm_M_logits[valid_mask_M], expected_M_logits[valid_mask_M], atol=1e-4), \
            f"Machine permutation equivariance failed! Max diff: {(perm_M_logits[valid_mask_M] - expected_M_logits[valid_mask_M]).abs().max()}"
        assert torch.allclose(perm_M_val, orig_val, atol=1e-4), "Critic value changed after machine permutation!"
        print("[PASS] Machine permutation equivariance holds (logits permute accordingly, value invariant).")

        # 3. Scale invariance: scale min_dur, dur, o_at, m_at by c > 0
        c = 3.5
        (min_dur, dur, o_at, m_at, dep_o, dep_m, act_mask, rev_pos) = inputs
        scaled_inputs = (
            min_dur * c,
            dur * c,
            o_at * c,
            m_at * c,
            dep_o,
            dep_m,
            act_mask,
            rev_pos
        )
        scaled_obs = build_pf_obs(scaled_inputs)
        scaled_logits, scaled_val = model.core(scaled_obs)
        valid_mask_scale = obs.mask_pair
        
        assert torch.allclose(scaled_logits[valid_mask_scale], orig_logits[valid_mask_scale], atol=1e-4), \
            f"Scale invariance failed! Max diff: {(scaled_logits[valid_mask_scale] - orig_logits[valid_mask_scale]).abs().max()}"
        assert torch.allclose(scaled_val, orig_val, atol=1e-4), "Critic value changed after scaling!"
        print(f"[PASS] Scale invariance holds for scale factor c={c}.")

    print("\n[SUCCESS] All invariance unit tests passed!")

if __name__ == '__main__':
    test_invariance()
