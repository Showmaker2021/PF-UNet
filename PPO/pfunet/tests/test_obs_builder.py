import sys
import os
import torch
import numpy as np

ppo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)

from SchedulingEnvironment import FJSPEnv
from SchedulingModel import build_model_input
from pfunet.ref_sim import RefSim, to_ref, ref_obs
from pfunet.modules import build_pf_obs

def test_obs_builder(num_states_to_check=200):
    print(f"Testing build_pf_obs against ref_obs on {num_states_to_check} random states...")
    rng = np.random.default_rng(114514)
    
    scenarios = [
        ({'num_jobs': 6, 'num_machines': 4, 'max_processing_time': 20}, True),
        ({'num_jobs': 10, 'num_machines': 5, 'max_processing_time': 20}, True),
        ({'num_jobs': 15, 'num_machines': 8, 'max_processing_time': 20}, False),
    ]
    
    checked = 0
    while checked < num_states_to_check:
        gen_params, is_sd1 = scenarios[checked % len(scenarios)]
        env = FJSPEnv(generator_params=gen_params, sd1=is_sd1)
        env.generate_data(1)
        state = env.reset_state()
        dur, jobs = to_ref(env)
        sim = RefSim(dur, jobs)
        
        done = False
        while not done and checked < num_states_to_check:
            # 1. Vectorized observation
            inputs, _ = build_model_input(state, env)
            obs = build_pf_obs(inputs)
            
            # 2. Reference observation
            job_order = list(state.job_idx[0, obs.head_idx[0].numpy()])
            ref = ref_obs(sim, job_order=job_order)
            
            # 3. Assertions
            # Coordinates in [0, 1]
            assert (obs.xy >= 0).all() and (obs.xy <= 1).all(), "Coordinates out of bounds [0, 1]!"
            
            # Number of heads matches alive jobs from state.job_idx
            n_alive_expected = len(np.unique(state.job_idx[0]))
            assert obs.has_head.sum(1).item() == n_alive_expected, \
                f"Alive job count mismatch: {obs.has_head.sum(1).item()} vs {n_alive_expected}"
            
            # Numeric parity against reference
            obs_xy = obs.xy[0].cpu().numpy()
            obs_feats = obs.feats[0].cpu().numpy()
            obs_raw = obs.raw[0].cpu().numpy()
            obs_const = obs.const[0].cpu().numpy()
            obs_ok = obs.mask_pair[0].cpu().numpy()
            
            assert np.allclose(obs_xy, ref.xy, atol=1e-5), f"xy mismatch at state {checked+1}"
            assert np.allclose(obs_feats, ref.feats, atol=1e-5), f"feats mismatch at state {checked+1}"
            assert np.allclose(obs_raw, ref.raw, atol=1e-5), f"raw mismatch at state {checked+1}"
            assert np.allclose(obs_const, ref.const, atol=1e-5), f"const mismatch at state {checked+1}"
            assert (obs_ok == ref.mask_pair).all(), f"mask_pair mismatch at state {checked+1}"
            
            checked += 1
            if checked % 50 == 0 or checked == num_states_to_check:
                print(f"Checked and verified {checked}/{num_states_to_check} states.")
                
            # Random valid step
            mask = state.action_mask[0]
            valid = np.argwhere(mask)
            seq, mach = valid[rng.choice(len(valid))]
            j = state.job_idx[0, seq]
            state, _, done = env.step(([seq], [mach]))
            sim.step(j, mach)
            if done:
                last_j = env.solution.job_idx[0][-1]
                last_m = env.solution.machine_idx[0][-1]
                sim.step(last_j, last_m)
                
    print(f"\n[SUCCESS] All {checked} states passed exact observation parity checks!")

if __name__ == '__main__':
    test_obs_builder(200)
