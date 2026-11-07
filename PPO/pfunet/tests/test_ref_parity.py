import sys
import os
from pathlib import Path
import numpy as np

# Ensure PPO folder is at the beginning of path
ppo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)

from SchedulingEnvironment import FJSPEnv
from pfunet.ref_sim import RefSim, to_ref

def run_parity_tests(num_trials=1000):
    print(f"Starting RefSim parity test across {num_trials} random trials...")
    
    # Test across multiple problem dimensions: (jobs, machines)
    configs = [
        (4, 3),
        (6, 4),
        (8, 5),
        (10, 5),
    ]
    
    rng = np.random.default_rng(114514)
    passed = 0
    
    for trial in range(num_trials):
        num_jobs, num_machines = configs[trial % len(configs)]
        env = FJSPEnv(
            generator_params={'num_jobs': num_jobs, 'num_machines': num_machines, 'max_processing_time': 20},
            sd1=True
        )
        env.generate_data(1)
        state = env.reset_state()
        dur, jobs = to_ref(env)
        sim = RefSim(dur, jobs)
        
        done = False
        while not done:
            mask = state.action_mask[0]
            valid_pairs = np.argwhere(mask)
            assert len(valid_pairs) > 0, "No valid action found before done!"
            
            choice_idx = rng.choice(len(valid_pairs))
            seq, mach = valid_pairs[choice_idx]
            job_id = state.job_idx[0, seq]
            
            state, reward, done = env.step(([seq], [mach]))
            sim.step(job_id, mach)
            
            if done:
                # FJSPEnv automatically executed the greedy last step
                last_j = env.solution.job_idx[0][-1]
                last_m = env.solution.machine_idx[0][-1]
                sim.step(last_j, last_m)
        
        env_makespan = float(state.finish_time.max())
        sim_makespan = sim.makespan()
        
        diff = abs(env_makespan - sim_makespan)
        if diff >= 1e-5:
            print(f"[FAIL] Trial {trial+1}: Env makespan {env_makespan} != Sim makespan {sim_makespan}")
            sys.exit(1)
            
        if not np.allclose(state.m_AT[0], sim.A, atol=1e-5):
            print(f"[FAIL] Trial {trial+1}: Machine available time mismatch: Env {state.m_AT[0]} vs Sim {sim.A}")
            sys.exit(1)
            
        if not sim.is_done():
            print(f"[FAIL] Trial {trial+1}: Sim has uncompleted operations remaining!")
            sys.exit(1)
            
        passed += 1
        if (trial + 1) % 200 == 0 or (trial + 1) == num_trials:
            print(f"Passed {passed}/{trial+1} trials.")
            
    print(f"\n[SUCCESS] 1000/1000 trials passed exact parity check!")

if __name__ == '__main__':
    run_parity_tests(1000)
