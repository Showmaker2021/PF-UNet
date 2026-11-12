"""PPO training for MatSched-Net pi3 + optional KL/PG distillation & DAgger for pi1/pi2."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MATSCHED_DIR = Path(__file__).resolve().parent
if str(MATSCHED_DIR) not in sys.path:
    sys.path.insert(0, str(MATSCHED_DIR))

from configs.matsched_fjsp import DEFAULT_CONFIG, load_fjsp_dataset_and_baseline, make_fjsp_env
from evaluate import evaluate_matsched
from model import MatSchedNet
from rollout import iter_minibatches, rollout_episodes, sample_level_policy
from router import act_adaptive_from_inputs


def masked_kl_div(teacher_logp: torch.Tensor, student_logp: torch.Tensor, elig_flat: torch.Tensor) -> torch.Tensor:
    """Compute KL(teacher || student) over eligible actions only (avoids 0 * (-inf - -inf) = NaN)."""
    t_lp = teacher_logp.float()
    s_lp = student_logp.float()
    p_t = torch.where(elig_flat, t_lp.exp(), torch.zeros_like(t_lp))
    diff = torch.where(elig_flat, t_lp - s_lp, torch.zeros_like(t_lp))
    return (p_t * diff).sum(dim=-1).mean()


def masked_entropy(logp: torch.Tensor, elig_flat: torch.Tensor) -> torch.Tensor:
    """Compute entropy over eligible actions safely."""
    lp = logp.float()
    p = torch.where(elig_flat, lp.exp(), torch.zeros_like(lp))
    safe_lp = torch.where(elig_flat, lp, torch.zeros_like(lp))
    return -(p * safe_lp).sum(dim=-1).mean()


def _ppo_pg_loss(
    logp: torch.Tensor,
    action: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    new_logp = logp.gather(1, action.unsqueeze(1)).squeeze(1).float()
    ratio = torch.exp((new_logp - old_logp).clamp(-20.0, 20.0))
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
    return -torch.min(surr1, surr2).mean()


def ppo_step_loss(
    model: MatSchedNet,
    mb: Dict[str, object],
    cfg: Dict[str, object],
    train_level: int = 3,
    kd_cfg: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PPO loss (on `train_level`) + optional KD/PG distillation onto levels 1 and 2."""
    ppo_cfg = cfg["ppo"]
    inp = mb["inp"]
    elig_flat = inp["elig"].flatten(1)
    action = mb["action"]
    old_logp = mb["old_logp"]
    adv = mb["adv"]
    ret = mb["ret"]

    logp_main, value = model(inp, level=train_level)
    pg_loss = _ppo_pg_loss(logp_main, action, old_logp, adv, ppo_cfg["clip_range"])
    v_pred = value.float()
    if "old_val" in mb:
        clip_r = ppo_cfg["clip_range"]
        v_old = mb["old_val"]
        v_clip = v_old + (v_pred - v_old).clamp(-clip_r, clip_r)
        vf_loss = torch.max((v_pred - ret).pow(2), (v_clip - ret).pow(2)).mean()
    else:
        vf_loss = F.mse_loss(v_pred, ret)
    ent = masked_entropy(logp_main, elig_flat)
    total = pg_loss + ppo_cfg["vf_coef"] * vf_loss - ppo_cfg["entropy_coef"] * ent

    stats = {
        "pg": float(pg_loss.item()),
        "vf": float(vf_loss.item()),
        "ent": float(ent.item()),
        "kl1": 0.0,
        "kl2": 0.0,
    }

    if kd_cfg and kd_cfg.get("enabled", False) and train_level == 3:
        teacher = logp_main.detach()
        kd_mode = kd_cfg.get("mode", "kl")
        for lv, beta_key, stat_key in ((1, "beta1", "kl1"), (2, "beta2", "kl2")):
            beta = kd_cfg[beta_key]
            if beta > 0:
                logp_k, _ = model(inp, level=lv)
                kl_k = masked_kl_div(teacher, logp_k, elig_flat)
                if kd_mode in ("kl", "both"):
                    total = total + beta * kl_k
                if kd_mode in ("pg", "both"):
                    pg_k = _ppo_pg_loss(logp_k, action, old_logp, adv, ppo_cfg["clip_range"])
                    ent_k = masked_entropy(logp_k, elig_flat)
                    total = total + beta * (pg_k - ppo_cfg["entropy_coef"] * ent_k)
                stats[stat_key] = float(kl_k.item())

    return total, stats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    cfg = {
        "env": dict(DEFAULT_CONFIG["env"]),
        "model": dict(DEFAULT_CONFIG["model"]),
        "ppo": dict(DEFAULT_CONFIG["ppo"]),
        "kd": dict(DEFAULT_CONFIG["kd"]),
    }
    is_sd1 = args.dataset.upper() == "SD1"
    cfg["env"]["num_jobs"] = args.num_jobs
    cfg["env"]["num_machines"] = args.num_machines
    cfg["env"]["sd1"] = is_sd1
    cfg["env"]["max_processing_time"] = 20 if is_sd1 else 99
    cfg["model"]["K"] = args.K
    cfg["model"]["num_layers"] = args.num_layers
    cfg["model"]["ff_mult"] = args.ff_mult
    cfg["model"]["use_m2o"] = not args.no_m2o
    cfg["model"]["use_mid_out"] = not args.no_mid_out
    cfg["model"]["use_cc"] = not args.no_cc
    cfg["model"]["use_pair_ffn"] = args.use_pair_ffn
    cfg["model"]["use_out_ffn"] = args.use_out_ffn
    cfg["model"]["shared_head"] = args.shared_head
    cfg["model"]["fixed_width"] = args.fixed_width
    cfg["ppo"]["episodes_per_iter"] = args.episodes_per_iter
    cfg["ppo"]["rollout_batch_size"] = args.batch_size
    cfg["ppo"]["minibatch_size"] = args.minibatch_size
    cfg["ppo"]["lr"] = args.lr
    cfg["ppo"]["lr_min"] = args.lr_min
    cfg["ppo"]["use_bf16"] = not args.no_bf16
    cfg["kd"]["enabled"] = args.enable_kd
    cfg["kd"]["mode"] = args.kd_mode
    cfg["kd"]["beta1"] = args.beta1
    cfg["kd"]["beta2"] = args.beta2
    cfg["kd"]["start_iter"] = args.kd_start_iter
    cfg["kd"]["dagger_weight"] = args.dagger_weight
    cfg["kd"]["dagger_episodes"] = args.dagger_episodes
    cfg["kd"]["router_metric"] = args.router_metric
    if args.router_metric == "margin":
        cfg["kd"]["tau_range"] = (0.02, 0.25)

    env = make_fjsp_env(**cfg["env"])
    env_val = make_fjsp_env(**cfg["env"])
    model = MatSchedNet(**cfg["model"]).to(device)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint from {args.init_ckpt}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["ppo"]["lr"], weight_decay=cfg["ppo"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.iters, 1), eta_min=cfg["ppo"]["lr_min"]
    )
    amp_enabled = cfg["ppo"]["use_bf16"] and device.type == "cuda"

    val_path = args.val_path
    if val_path is None:
        val_path = (
            f"data/FJSP/TNNLS/SD1/{args.num_jobs}x{args.num_machines}"
            if is_sd1
            else f"data/FJSP/TNNLS/SD2/{args.num_jobs}x{args.num_machines}+mix"
        )
    val_data, val_ref = None, None
    if val_path:
        val_data, val_ref = load_fjsp_dataset_and_baseline(val_path)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_gap = float("inf")

    train_level = 3
    print(
        f"Training MatSchedNet on {device} | dataset={args.dataset}({args.num_jobs}x{args.num_machines}, pt<={cfg['env']['max_processing_time']}) | "
        f"val={val_path} | bf16={amp_enabled}, fixed_width={args.fixed_width}, "
        f"KD={cfg['kd']['enabled']}[{cfg['kd']['mode']}, b1={cfg['kd']['beta1']}, b2={cfg['kd']['beta2']}], "
        f"iters={args.iters}, lr={args.lr}->{args.lr_min}"
    )

    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        kd_active = cfg["kd"]["enabled"] and (args.fixed_width is None) and (it >= cfg["kd"]["start_iter"])

        # (1) On-policy PPO rollout from pi3 (or fixed_width model)
        buf, mk = rollout_episodes(
            model=model,
            env=env,
            num_episodes=cfg["ppo"]["episodes_per_iter"],
            batch_size=cfg["ppo"]["rollout_batch_size"],
            policy_fn=lambda inp: sample_level_policy(model, inp, level=train_level, greedy=False),
            K=cfg["model"]["K"],
            device=device,
            use_bf16=cfg["ppo"]["use_bf16"],
        )

        model.train()
        meter = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl1": 0.0, "kl2": 0.0, "n": 0}
        for _ in range(cfg["ppo"]["k_epochs"]):
            for mb in iter_minibatches(buf, cfg["ppo"]["minibatch_size"], shuffle=True, device=device):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    loss, stats = ppo_step_loss(
                        model, mb, cfg, train_level=train_level, kd_cfg=cfg["kd"] if kd_active else None
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["ppo"]["grad_clip"])
                opt.step()

                for k in ("pg", "vf", "ent", "kl1", "kl2"):
                    meter[k] += stats[k]
                meter["n"] += 1

        # (2) Optional DAgger rollout from router -> gentle KD anchored by dagger_weight
        dagger_kl = 0.0
        if kd_active and cfg["kd"]["dagger_episodes"] > 0:
            lo, hi = cfg["kd"]["tau_range"]
            tau = (random.uniform(lo, hi), random.uniform(lo, hi))
            metric = cfg["kd"]["router_metric"]

            def router_policy(inp_gpu):
                act, logp, _ = act_adaptive_from_inputs(model, inp_gpu, tau=tau, greedy=False, metric=metric)
                return act, logp, None

            buf_dag, _ = rollout_episodes(
                model=model,
                env=env,
                num_episodes=cfg["kd"]["dagger_episodes"],
                batch_size=cfg["ppo"]["rollout_batch_size"],
                policy_fn=router_policy,
                K=cfg["model"]["K"],
                device=device,
                use_bf16=cfg["ppo"]["use_bf16"],
            )
            model.train()
            dag_w = cfg["kd"]["dagger_weight"]
            dag_n = 0
            for mb in iter_minibatches(buf_dag, cfg["ppo"]["minibatch_size"], shuffle=True, device=device):
                opt.zero_grad(set_to_none=True)
                inp = mb["inp"]
                elig_flat = inp["elig"].flatten(1)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                        teacher, _ = model(inp, level=3)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    logp3_cur, _ = model(inp, level=3)
                    kd_loss = masked_kl_div(teacher, logp3_cur, elig_flat)
                    for lv, beta in ((1, cfg["kd"]["beta1"]), (2, cfg["kd"]["beta2"])):
                        if beta > 0:
                            logp_k, _ = model(inp, level=lv)
                            kd_loss = kd_loss + dag_w * beta * masked_kl_div(teacher, logp_k, elig_flat)
                kd_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["ppo"]["grad_clip"])
                opt.step()
                dagger_kl += float(kd_loss.item())
                dag_n += 1
            if dag_n > 0:
                dagger_kl /= dag_n

        sched.step()
        dt = time.perf_counter() - t0
        n_mb = max(meter["n"], 1)
        if it % args.log_every == 0 or it == 1:
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            cur_lr = opt.param_groups[0]["lr"]
            print(
                f"Iter {it:4d}/{args.iters} | mk={mk.mean():.2f} | lr={cur_lr:.2e} | "
                f"pg={meter['pg']/n_mb:.4f} vf={meter['vf']/n_mb:.4f} ent={meter['ent']/n_mb:.3f} | "
                f"kl1={meter['kl1']/n_mb:.4f} kl2={meter['kl2']/n_mb:.4f} dag={dagger_kl:.4f} | "
                f"dt={dt:.2f}s peak={peak_mb:.0f}MB"
            )

        if val_data is not None and (it % args.val_every == 0 or it == args.iters):
            if cfg["kd"]["enabled"] and args.fixed_width is None:
                val_res = {}
                for lv in (1, 2, 3):
                    val_res[lv] = evaluate_matsched(
                        model=model,
                        env=env_val,
                        dataset=val_data,
                        baseline=val_ref,
                        batch_size=args.val_batch_size,
                        level=lv,
                        K=cfg["model"]["K"],
                        device=device,
                    )
                g1, g2, g3 = [val_res[lv]["gap"] for lv in (1, 2, 3)]
                print(
                    f"  [VAL @ iter {it}] gap L1={g1:.2f}% L2={g2:.2f}% L3={g3:.2f}% | "
                    f"mk3={val_res[3]['makespan']:.3f}"
                )
                score_metric = g3 if g3 is not None else val_res[3]["makespan"]
            else:
                res = evaluate_matsched(
                    model=model,
                    env=env_val,
                    dataset=val_data,
                    baseline=val_ref,
                    batch_size=args.val_batch_size,
                    level=3,
                    K=cfg["model"]["K"],
                    device=device,
                )
                gap_str = f"{res['gap']:.2f}%" if res["gap"] is not None else "N/A"
                print(
                    f"  [VAL @ iter {it}] avg_mk={res['makespan']:.3f} | gap={gap_str} | "
                    f"time/inst={res['sec_per_inst']*1000:.2f}ms"
                )
                score_metric = res["gap"] if res["gap"] is not None else res["makespan"]

            if score_metric < best_gap:
                best_gap = score_metric
                torch.save({"model": model.state_dict(), "cfg": cfg, "iter": it}, save_dir / "best.pt")

    torch.save({"model": model.state_dict(), "cfg": cfg, "iter": args.iters}, save_dir / "last.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["SD1", "SD2"], default="SD1")
    p.add_argument("--num_jobs", type=int, default=10)
    p.add_argument("--num_machines", type=int, default=5)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--ff_mult", type=int, default=2)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--episodes_per_iter", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--minibatch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--enable_kd", action="store_true")
    p.add_argument("--kd_mode", choices=["kl", "pg", "both"], default="kl")
    p.add_argument("--beta1", type=float, default=0.015)
    p.add_argument("--beta2", type=float, default=0.02)
    p.add_argument("--kd_start_iter", type=int, default=100)
    p.add_argument("--dagger_weight", type=float, default=0.10)
    p.add_argument("--dagger_episodes", type=int, default=8)
    p.add_argument("--router_metric", choices=["norm_entropy", "margin", "top1"], default="norm_entropy")
    p.add_argument("--no_m2o", action="store_true")
    p.add_argument("--no_mid_out", action="store_true")
    p.add_argument("--no_cc", action="store_true")
    p.add_argument("--use_pair_ffn", action="store_true")
    p.add_argument("--use_out_ffn", action="store_true")
    p.add_argument("--shared_head", action="store_true")
    p.add_argument("--fixed_width", type=int, default=None)
    p.add_argument("--init_ckpt", type=str, default=None)
    p.add_argument("--val_path", type=str, default=None)
    p.add_argument("--val_batch_size", type=int, default=100)
    p.add_argument("--val_every", type=int, default=25)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="MatSched/checkpoints/run")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
