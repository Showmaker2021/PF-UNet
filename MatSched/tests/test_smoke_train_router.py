"""Smoke test for router.act_adaptive and 1-iteration PPO + KD + DAgger training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

MATSCHED_DIR = Path(__file__).resolve().parents[1]
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from configs.matsched_fjsp import DEFAULT_CONFIG, make_fjsp_env
from model import MatSchedNet
from rollout import iter_minibatches, rollout_episodes, sample_level_policy
from router import act_adaptive, act_adaptive_from_inputs
from train_ppo import ppo_step_loss


def test_smoke_router_and_ppo_kd() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_fjsp_env(num_jobs=6, num_machines=4, sd1=True)
    env.set_seed(0)

    model = MatSchedNet(dmax=128, num_layers=3, K=2).to(device)

    # 1. Test adaptive router end-to-end on an episode (both norm_entropy and margin)
    for metric, tau in (("norm_entropy", (0.5, 0.7)), ("margin", (0.05, 0.15))):
        env.generate_data(batch_size=4)
        env.reset_state()
        done = False
        used_levels = []
        while not done:
            act, lv = act_adaptive(model, env, tau=tau, greedy=True, K=2, metric=metric, device=device)
            used_levels.append(lv)
            _, _, done = env.step(act)
        all_lv = torch.cat(used_levels)
        assert ((all_lv >= 1) & (all_lv <= 3)).all()
        print(f"  [PASS] Router ({metric}) episode, levels: {[(int((all_lv==i).sum())) for i in (1,2,3)]}")

    # 2. Test 1-iter PPO + KD + DAgger step
    cfg = {
        "env": dict(DEFAULT_CONFIG["env"]),
        "model": dict(DEFAULT_CONFIG["model"]),
        "ppo": dict(DEFAULT_CONFIG["ppo"]),
        "kd": dict(DEFAULT_CONFIG["kd"]),
    }
    cfg["kd"]["enabled"] = True
    opt = torch.optim.Adam(model.parameters(), lr=6e-4)

    buf, mk = rollout_episodes(
        model=model,
        env=env,
        num_episodes=4,
        batch_size=4,
        policy_fn=lambda inp: sample_level_policy(model, inp, level=3),
        K=2,
        device=device,
        use_bf16=True,
    )
    model.train()
    for mb in iter_minibatches(buf, minibatch_size=64, shuffle=True, device=device):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss, stats = ppo_step_loss(model, mb, cfg, train_level=3, kd_cfg=cfg["kd"])
        assert torch.isfinite(loss), f"Non-finite loss: {loss}"
        loss.backward()
        opt.step()
        break

    # DAgger rollout
    buf2, _ = rollout_episodes(
        model=model,
        env=env,
        num_episodes=4,
        batch_size=4,
        policy_fn=lambda inp: act_adaptive_from_inputs(model, inp, tau=(0.5, 0.5), greedy=False),
        K=2,
        device=device,
        use_bf16=True,
    )
    assert buf2["action"].shape[0] > 0
    print(f"  [PASS] PPO+KD+DAgger smoke step finite (mean mk={mk.mean():.2f})")


if __name__ == "__main__":
    print("Running test_smoke_train_router...")
    test_smoke_router_and_ppo_kd()
    print("ALL test_smoke_train_router PASSED")
