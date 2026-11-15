"""Synchronous batched rollout and GAE buffer for PPO and DAgger KD."""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from adapter import build_inputs, collate_inputs, move_inputs_to, select_rows, to_env_action


def sample_level_policy(model, inp: Dict[str, object], level: int = 3, greedy: bool = False):
    """Policy function for a fixed MatSched-Net level."""
    logp, value = model(inp, level=level)
    probs = logp.exp()
    if greedy:
        action_flat = probs.argmax(dim=-1)
    else:
        action_flat = Categorical(probs=probs).sample()
    action_logp = logp.gather(1, action_flat.unsqueeze(1)).squeeze(1)
    return action_flat, action_logp, value


def pad_tail_to(inp: Dict[str, object], target_n: int) -> Dict[str, object]:
    """Pad tail tensors in `inp` along N_rem to `target_n` so steps can be stacked."""
    tail = inp["tail"]
    n = tail["feat"].shape[1]
    if n == target_n:
        return inp
    pad_n = target_n - n
    out = dict(inp)
    out["tail"] = {
        "feat": F.pad(tail["feat"], (0, 0, 0, pad_n), value=0.0),
        "mask": F.pad(tail["mask"], (0, pad_n), value=False),
        "job": F.pad(tail["job"], (0, pad_n), value=0),
    }
    return out


@torch.no_grad()
def rollout_episodes(
    model,
    env,
    num_episodes: int,
    batch_size: int,
    policy_fn: Callable[[Dict[str, object]], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    K: int = 2,
    device: Optional[torch.device | str] = None,
    use_bf16: bool = True,
    algo: str = "ppo",
    gamma: float = 1.0,
) -> Tuple[Dict[str, object], np.ndarray]:
    """Run `num_episodes` generated episodes synchronously and return flattened transition dict."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.eval()

    all_inp_chunks: List[Dict[str, object]] = []
    all_actions: List[torch.Tensor] = []
    all_logps: List[torch.Tensor] = []
    all_vals: List[torch.Tensor] = []
    all_advs: List[torch.Tensor] = []
    all_rets: List[torch.Tensor] = []
    all_makespans: List[np.ndarray] = []

    amp_enabled = use_bf16 and device.type == "cuda"

    done_eps = 0
    while done_eps < num_episodes:
        bs = min(batch_size, num_episodes - done_eps)
        env.generate_data(bs)
        env.reset_state()
        n_init = env.problem.duration.shape[1]

        step_inps: List[Dict[str, object]] = []
        step_actions: List[torch.Tensor] = []
        step_logps: List[torch.Tensor] = []
        step_values: List[torch.Tensor] = []
        step_rewards: List[torch.Tensor] = []

        done = False
        while not done:
            inp_gpu = build_inputs(env, K=K, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                act_flat, logp, value = policy_fn(inp_gpu)

            M = inp_gpu["elig"].shape[2]
            sel_job = act_flat // M
            sel_mach = act_flat % M
            env_action = to_env_action(sel_job, sel_mach, inp_gpu["head_row"])

            state, reward, done = env.step(env_action)
            scaler_np = inp_gpu["scaler"].detach().cpu().numpy()
            reward_scaled = torch.as_tensor(reward / scaler_np, dtype=torch.float32)

            inp_cpu = pad_tail_to(move_inputs_to(inp_gpu, "cpu"), n_init)
            step_inps.append(inp_cpu)
            step_actions.append(act_flat.detach().cpu())
            step_logps.append(logp.detach().float().cpu())
            step_values.append(
                value.detach().float().cpu() if value is not None else torch.zeros(bs, dtype=torch.float32)
            )
            step_rewards.append(reward_scaled)

        makespans = state.finish_time.max(axis=1).max(axis=1)
        all_makespans.append(makespans)

        rewards_bt = torch.stack(step_rewards, dim=1)  # (bs, T)
        values_bt = torch.stack(step_values, dim=1)    # (bs, T)
        if algo == "reinforce":
            ret_bt = compute_discounted_returns(rewards_bt, gamma=gamma)
            adv_bt = ret_bt - ret_bt.mean(dim=0, keepdim=True)
        else:
            adv_bt, ret_bt = compute_gae(rewards_bt, values_bt, gamma=gamma, lam=0.98)

        # Flatten (bs, T) -> (bs*T,)
        for t in range(len(step_inps)):
            all_inp_chunks.append(step_inps[t])
            all_actions.append(step_actions[t])
            all_logps.append(step_logps[t])
            all_vals.append(values_bt[:, t])
            all_advs.append(adv_bt[:, t])
            all_rets.append(ret_bt[:, t])

        done_eps += bs

    merged_inp = _concat_padded_inputs(all_inp_chunks)
    flat_adv = torch.cat(all_advs, dim=0)
    if algo != "reinforce":
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

    buffer = {
        "inp": merged_inp,
        "action": torch.cat(all_actions, dim=0),
        "old_logp": torch.cat(all_logps, dim=0),
        "old_val": torch.cat(all_vals, dim=0),
        "adv": flat_adv,
        "ret": torch.cat(all_rets, dim=0),
    }
    return buffer, np.concatenate(all_makespans, axis=0)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.98,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation on (B, T) tensors."""
    B, T = rewards.shape
    next_vals = torch.cat([values[:, 1:], torch.zeros((B, 1), dtype=values.dtype)], dim=1)
    deltas = rewards + gamma * next_vals - values

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((B,), dtype=rewards.dtype)
    for t in reversed(range(T)):
        gae = deltas[:, t] + gamma * lam * gae
        advantages[:, t] = gae

    returns = advantages + values
    return advantages, returns


def compute_discounted_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns on (B, T) tensor (matching ReSched REINFORCETrainer)."""
    returns = torch.zeros_like(rewards)
    g = torch.zeros((rewards.shape[0],), dtype=rewards.dtype)
    for t in reversed(range(rewards.shape[1])):
        g = rewards[:, t] + gamma * g
        returns[:, t] = g
    return returns


def _concat_padded_inputs(chunks: List[Dict[str, object]]) -> Dict[str, object]:
    """Concatenate list of padded input dicts along batch dim 0 using collate_inputs."""
    return collate_inputs(chunks)


def iter_minibatches(
    buffer: Dict[str, object],
    minibatch_size: int,
    shuffle: bool = True,
    device: Optional[torch.device | str] = None,
) -> Iterator[Dict[str, object]]:
    """Yield shuffled minibatches moved to `device`."""
    n = buffer["action"].shape[0]
    perm = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, minibatch_size):
        idx = perm[start : start + minibatch_size]
        inp_mb = select_rows(buffer["inp"], idx)
        if device is not None:
            inp_mb = move_inputs_to(inp_mb, device)
        yield {
            "inp": inp_mb,
            "action": buffer["action"][idx].to(device) if device else buffer["action"][idx],
            "old_logp": buffer["old_logp"][idx].to(device) if device else buffer["old_logp"][idx],
            "old_val": buffer["old_val"][idx].to(device) if device else buffer["old_val"][idx],
            "adv": buffer["adv"][idx].to(device) if device else buffer["adv"][idx],
            "ret": buffer["ret"][idx].to(device) if device else buffer["ret"][idx],
        }
