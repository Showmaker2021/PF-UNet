import sys
import os
import torch
import numpy as np

ppo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)

from SchedulingEnvironment import FJSPEnv
from SchedulingModel import build_model_input

def test_dependency_direction():
    print("Testing dependency_o direction on reachable states from random rollouts...")
    rng = np.random.default_rng(114514)
    
    # Test on both SD1 and standard FJSP generator, various sizes
    scenarios = [
        ({'num_jobs': 6, 'num_machines': 4, 'max_processing_time': 20}, True),
        ({'num_jobs': 10, 'num_machines': 5, 'max_processing_time': 20}, True),
        ({'num_jobs': 8, 'num_machines': 4, 'max_processing_time': 20}, False),
    ]
    
    total_states_checked = 0
    
    for gen_params, is_sd1 in scenarios:
        for episode in range(50):
            env = FJSPEnv(generator_params=gen_params, sd1=is_sd1)
            env.generate_data(batch_size=4)
            state = env.reset_state()
            done = False
            
            while not done:
                inputs, _ = build_model_input(state, env)
                dep_o = inputs[4] # [B, N, N] bool
                act_mask = inputs[6] # [B, N, M] bool
                
                B, N, _ = act_mask.shape
                head = act_mask.any(-1) # [B, N] bool
                
                # Verify D2 against true job_idx from state
                job_idx = torch.tensor(state.job_idx) # [B, N]
                same_job = job_idx.unsqueeze(2) == job_idx.unsqueeze(1) # [B, N, N] bool
                
                # Check condition: for each head operation, is dep_o[b, head_idx, :] == same_job[b, head_idx, :]?
                # Notice: dep_o might be symmetric or directed. Let's inspect.
                match = (dep_o[head] == same_job[head]).all().item()
                if not match:
                    # Let's inspect differences
                    diff_cnt = (dep_o[head] != same_job[head]).sum().item()
                    print(f"Mismatch found: {diff_cnt} differences!")
                    # Check transpose
                    dep_o_T = dep_o.transpose(1, 2)
                    match_T = (dep_o_T[head] == same_job[head]).all().item()
                    print(f"Match with transpose? {match_T}")
                    print(f"dep_o[head] sample:\n{dep_o[head][:2]}")
                    print(f"same_job[head] sample:\n{same_job[head][:2]}")
                    sys.exit(1)
                    
                total_states_checked += B
                
                # Step random valid action
                # Choose random valid action for each batch element
                actions_seq = []
                actions_m = []
                for b in range(B):
                    valid = np.argwhere(state.action_mask[b])
                    choice = valid[rng.choice(len(valid))]
                    actions_seq.append(choice[0])
                    actions_m.append(choice[1])
                    
                state, _, done = env.step((np.array(actions_seq), np.array(actions_m)))
                
    print(f"[SUCCESS] Checked {total_states_checked} batch states: dep_o[head] == same_job[head] holds 100%!")

if __name__ == '__main__':
    test_dependency_direction()
