"""Real-World Benchmark 2 (OmniJev v3): Real MuJoCo Reacher-v5 Continuous Physics + Bimodal Obstacle & Dynamic Hazard.

Validates Axis C + v2/v3 breakthroughs inside REAL MuJoCo 3.17 rigid-body physics (Reacher-v5):
  - Real 2-link robotic arm torque actuation, inertia, velocity damping, and 45-step closed-loop rollouts.
  - Physical obstacle placed directly between arm start and target -> two physically valid torque paths:
      (1) Swing CLOCKWISE around obstacle, OR (2) Swing COUNTER-CLOCKWISE around obstacle!
  - Plus a dynamic hazard blocking ONE of the two bypass corridors per episode (encoded in obs state).
  - Evaluates IN REAL CLOSED-LOOP MUJOCO PHYSICS ROLLOUTS:
      1. Parallel L2/L1 Regression Baseline (OpenVLA-OFT style)
      2. OmniJev v3 Jev-VLA (FlowEnergyHybridLoss + RTC Asynchronous Overlap Blending)
      3. OmniJev v3 Jev-VLA + LatentJEPAWorldHead + RiskSensitiveLatentPlanner (CVaR Mental-Sim MPC)
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnijev import (
    EvidentialDirichletLoss,
    FlowEnergyHybridLoss,
    JevVLAHead,
    LatentJEPAWorldHead,
    OmniJevConfig,
    RealTimeActionChunkBlender,
    RiskSensitiveLatentPlanner,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed MuJoCo Reacher workspace geometry for reproducible bimodal obstacle benchmark:
# Arm links: L1 = 0.10, L2 = 0.11 (max reach ~0.21).
# Start qpos = (0.0, 0.0) -> fingertip starts at (+0.21, 0.0).
# Target placed near (-0.16, 0.0) -> direct path goes through (0.0, 0.0) / center obstacle,
# so arm MUST swing either UPPER (+y, q1 > 0) or LOWER (-y, q1 < 0) around the center obstacle!
START_Q = np.array([0.0, 0.0])
OBSTACLE_XY = np.array([0.04, 0.0])
OBSTACLE_R = 0.075


def _reset_bimodal_reacher(env, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    """Reset real MuJoCo Reacher-v5 with target across obstacle & random dynamic hazard side (+1 upper, -1 lower)."""
    env.reset(seed=int(rng.integers(0, 100000)))
    target_x = -0.155 + rng.uniform(-0.015, 0.015)
    target_y = rng.uniform(-0.02, 0.02)
    qpos = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), target_x, target_y])
    qvel = np.zeros(4)
    env.unwrapped.set_state(qpos, qvel)
    obs = env.unwrapped._get_obs().astype(np.float32)  # 10D real MuJoCo observation
    hazard_side = 1.0 if rng.random() < 0.5 else -1.0  # +1 = UPPER corridor blocked, -1 = LOWER corridor blocked
    return obs, np.array([target_x, target_y], dtype=np.float32), hazard_side


def _fingertip_xy(env) -> np.ndarray:
    return env.unwrapped.get_body_com("fingertip")[:2].copy().astype(np.float32)


def _expert_torque_step(env, branch_sign: float, step_idx: int) -> np.ndarray:
    """Smooth real-physics PD torque controller swinging via upper (+1) or lower (-1) corridor cleanly to target."""
    qpos = env.unwrapped.data.qpos[:2]
    qvel = env.unwrapped.data.qvel[:2]
    # Phase 1 (steps 0..13): swing wide around center obstacle via +-1.85 rad
    # Phase 2 (steps 14..35): converge sharply onto target near (-0.155, 0) -> q1 ~= +-2.98, q2 ~= +-0.45
    target_q1 = branch_sign * (1.85 if step_idx < 13 else 2.98)
    target_q2 = branch_sign * (0.40 if step_idx < 13 else 0.45)
    err = np.array([target_q1 - qpos[0], target_q2 - qpos[1]])
    torque = np.clip(1.60 * err - 0.28 * qvel, -1.0, 1.0)
    return torque.astype(np.float32)


def _collect_mujoco_demos(num_episodes: int = 260, chunk_h: int = 12, ep_len: int = 36):
    """Collect real MuJoCo closed-loop bimodal torque trajectories (50% upper, 50% lower)."""
    env = gym.make("Reacher-v5")
    rng = np.random.default_rng(77)

    obs_chunks, act_chunks, next_obs_list, safe_labels, goal_vals = [], [], [], [], []
    for ep in range(num_episodes):
        _, tgt_xy, haz_side = _reset_bimodal_reacher(env, rng)
        branch = 1.0 if (ep % 2 == 0) else -1.0

        ep_obs, ep_act, ep_tip = [], [], []
        for t in range(ep_len):
            ep_obs.append(np.concatenate([env.unwrapped._get_obs().astype(np.float32), [haz_side, t / ep_len]]))
            act = _expert_torque_step(env, branch, t)
            ep_act.append(act)
            env.step(act)
            ep_tip.append(_fingertip_xy(env))

        for s in range(0, ep_len - chunk_h + 1, 3):
            o_s = ep_obs[s]
            a_chunk = np.stack(ep_act[s: s + chunk_h], axis=0)      # [H, 2]
            o_next = ep_obs[min(s + chunk_h, ep_len - 1)]
            tips_window = np.stack(ep_tip[s: s + chunk_h], axis=0)  # [H, 2]

            obs_dists = np.linalg.norm(tips_window - OBSTACLE_XY[None, :], axis=-1)
            hit_center = bool((obs_dists < OBSTACLE_R).any())
            in_corridor = (tips_window[:, 0] > -0.08) & (tips_window[:, 0] < 0.14)
            hit_hazard = bool(((tips_window[:, 1] * haz_side > 0.075) & in_corridor).any())
            is_safe = int((not hit_center) and (not hit_hazard))

            # Continuous goal-progress value in [0, 1] (rewards moving closer to target!)
            dist_start = float(np.linalg.norm(tips_window[0] - tgt_xy))
            dist_end = float(np.linalg.norm(tips_window[-1] - tgt_xy))
            progress_val = float(np.clip(0.5 * (1.0 - dist_end / 0.38) + 0.5 * np.clip((dist_start - dist_end) / 0.20 + 0.5, 0.0, 1.0), 0.0, 1.0))

            obs_chunks.append(o_s)
            act_chunks.append(a_chunk)
            next_obs_list.append(o_next)
            safe_labels.append(is_safe)
            goal_vals.append(progress_val)

    env.close()
    return (
        torch.tensor(np.array(obs_chunks), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.array(act_chunks), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.array(safe_labels), dtype=torch.long, device=DEVICE),
        torch.tensor(np.array(goal_vals), dtype=torch.float32, device=DEVICE),
    )


def _evaluate_in_mujoco(
    policy_mode: str,
    obs_enc_l2: nn.Module,
    obs_enc_vla: nn.Module,
    l2_policy: nn.Module,
    vla: JevVLAHead,
    planner: RiskSensitiveLatentPlanner,
    num_episodes: int = 80,
    chunk_h: int = 12,
    ep_len: int = 36,
) -> Dict[str, float]:
    """Run REAL closed-loop MuJoCo Reacher-v5 rollouts with Temporal z-Persistence + RTC + CVaR MPC!"""
    env = gym.make("Reacher-v5")
    rng = np.random.default_rng(999)
    blender = RealTimeActionChunkBlender(blend_steps=3)

    n_safe = 0
    n_success = 0
    n_reached = 0
    final_dists = []
    jerks = []
    infer_times_ms = []

    with torch.no_grad():
        for _ in range(num_episodes):
            _, tgt_xy, haz_side = _reset_bimodal_reacher(env, rng)
            hit_center = False
            hit_hazard = False
            active_chunk = None
            chunk_cursor = 0
            exec_actions = []
            ep_z = torch.randn(1, vla.z_dim, device=DEVICE)

            for t in range(ep_len):
                if active_chunk is None or chunk_cursor >= 6:
                    o_vec = np.concatenate([env.unwrapped._get_obs().astype(np.float32), [haz_side, t / ep_len]])
                    o_t = torch.tensor(o_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    if policy_mode == "l2_oft_baseline":
                        s_l2 = obs_enc_l2(o_t)
                        new_chunk = l2_policy(s_l2).view(1, chunk_h, 2)
                    elif policy_mode == "vla_flow_energy_rtc":
                        s_t = obs_enc_vla(o_t)
                        ctx = s_t.unsqueeze(1).expand(-1, 4, -1)
                        new_chunk = vla(ctx, proprio_raw=o_t, z=ep_z)["action_chunk"]
                    elif policy_mode == "v3_mental_sim_cvar_mpc":
                        s_t = obs_enc_vla(o_t)
                        ctx = s_t.unsqueeze(1).expand(-1, 4, -1)
                        z_pool = torch.cat([ep_z, torch.randn(15, vla.z_dim, device=DEVICE)], dim=0)
                        cands = vla(
                            ctx.expand(16, -1, -1), proprio_raw=o_t.expand(16, -1), z=z_pool
                        )["action_chunk"].unsqueeze(0)  # [1, 16, H, 2]
                        plan = planner.plan_cvar(s_t, cands)
                        new_chunk = plan["best_chunk"]
                        ep_z = z_pool[int(plan["best_idx"].item()): int(plan["best_idx"].item()) + 1]
                    else:
                        raise ValueError(policy_mode)

                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    infer_times_ms.append((time.perf_counter() - t0) * 1000.0)

                    if active_chunk is not None and policy_mode != "l2_oft_baseline":
                        active_chunk = blender.blend(active_chunk, new_chunk, exec_offset=chunk_cursor)
                    else:
                        active_chunk = new_chunk
                    chunk_cursor = 0

                act = active_chunk[0, chunk_cursor].clamp(-1.0, 1.0).cpu().numpy()
                chunk_cursor += 1
                exec_actions.append(act)
                env.step(act)

                tip = _fingertip_xy(env)
                if float(np.linalg.norm(tip - OBSTACLE_XY)) < OBSTACLE_R:
                    hit_center = True
                if (-0.08 < tip[0] < 0.14) and (tip[1] * haz_side > 0.075):
                    hit_hazard = True

            final_d = float(np.linalg.norm(_fingertip_xy(env) - tgt_xy))
            final_dists.append(final_d)
            ep_safe = (not hit_center) and (not hit_hazard)
            reached = final_d < 0.09
            if ep_safe:
                n_safe += 1
            if reached:
                n_reached += 1
            if ep_safe and reached:
                n_success += 1

            acts_np = np.stack(exec_actions, axis=0)
            acc = acts_np[2:] - 2 * acts_np[1:-1] + acts_np[:-2]
            jerks.append(float(np.mean(acc ** 2)))

    env.close()
    avg_ms = sum(infer_times_ms) / max(len(infer_times_ms), 1)
    return {
        "closed_loop_safe_and_reached_success_pct": round(100.0 * n_success / num_episodes, 2),
        "safe_obstacle_and_hazard_free_pct": round(100.0 * n_safe / num_episodes, 2),
        "goal_reached_pct": round(100.0 * n_reached / num_episodes, 2),
        "mean_final_tip_dist_m": round(float(np.mean(final_dists)), 4),
        "mean_torque_jerk": round(float(np.mean(jerks)), 4),
        "chunk_infer_ms": round(avg_ms, 2),
        "effective_control_hz": round(1000.0 / max(avg_ms, 0.1), 1),
    }


def main():
    torch.manual_seed(42)
    chunk_h = 12
    print("Collecting real MuJoCo Reacher-v5 bimodal physics demonstrations...")
    obs_all, act_all, next_all, safe_all, goal_all = _collect_mujoco_demos(num_episodes=260, chunk_h=chunk_h)
    print(f"Collected {obs_all.size(0)} real MuJoCo physics chunks (safe fraction: {safe_all.float().mean().item():.2f})")

    obs_dim = obs_all.size(1)
    D = 96
    cfg = OmniJevConfig(
        d_model=D, n_heads=4, d_ff=192, proprio_dim=obs_dim,
        vla_action_dim=2, vla_chunk_horizon=chunk_h, vla_z_dim=8, vla_head_layers=2,
    )

    # Independent state encoders so Jev-VLA is NEVER handicapped by L2 baseline's zero-torque collapse!
    obs_enc_l2 = nn.Sequential(nn.Linear(obs_dim, D), nn.LayerNorm(D), nn.GELU(), nn.Linear(D, D)).to(DEVICE)
    obs_enc_vla = nn.Sequential(nn.Linear(obs_dim, D), nn.LayerNorm(D), nn.GELU(), nn.Linear(D, D)).to(DEVICE)
    l2_policy = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, chunk_h * 2)).to(DEVICE)
    vla = JevVLAHead(cfg).to(DEVICE)
    world = LatentJEPAWorldHead(D, action_dim=2, horizon=chunk_h, z_env_dim=8).to(DEVICE)

    feh_loss = FlowEnergyHybridLoss(gamma=1.0, diversity_weight=1.0, wta_weight=0.55, jerk_weight=0.02)
    evid_loss = EvidentialDirichletLoss(kl_weight=0.05)

    opt_l2 = torch.optim.AdamW(list(l2_policy.parameters()) + list(obs_enc_l2.parameters()), lr=2.5e-3)
    opt_vla = torch.optim.AdamW(list(vla.parameters()) + list(obs_enc_vla.parameters()), lr=2.5e-3)

    N = obs_all.size(0)
    M_prior = 6
    for _ in range(600):
        idx = torch.randint(0, N, (96,), device=DEVICE)
        o_b, a_b = obs_all[idx], act_all[idx]

        # 1) L2 OFT baseline
        p_l2 = l2_policy(obs_enc_l2(o_b)).view(-1, chunk_h, 2)
        l_l2 = F.mse_loss(p_l2, a_b)
        opt_l2.zero_grad()
        l_l2.backward()
        opt_l2.step()

        # 2) OmniJev v3 FlowEnergyHybrid Jev-VLA (end-to-end with obs_enc_vla!)
        s_v = obs_enc_vla(o_b)
        ctx = s_v.unsqueeze(1).expand(-1, 4, -1)
        B = o_b.size(0)
        preds = vla(
            ctx.unsqueeze(1).expand(B, M_prior, 4, D).reshape(B * M_prior, 4, D),
            proprio_raw=o_b.unsqueeze(1).expand(B, M_prior, obs_dim).reshape(B * M_prior, obs_dim),
        )["action_chunk"].view(B, M_prior, chunk_h, 2)
        l_vla = feh_loss(preds, a_b)["loss"]
        opt_vla.zero_grad()
        l_vla.backward()
        torch.nn.utils.clip_grad_norm_(list(vla.parameters()) + list(obs_enc_vla.parameters()), 1.0)
        opt_vla.step()

    # Train Latent-JEPA World Dynamics Head on top of trained obs_enc_vla
    opt_w = torch.optim.AdamW(world.parameters(), lr=2.5e-3)
    obs_enc_vla.eval()
    for _ in range(600):
        idx = torch.randint(0, N, (128,), device=DEVICE)
        o_b, a_b, n_b, sf_b, gv_b = obs_all[idx], act_all[idx], next_all[idx], safe_all[idx], goal_all[idx]
        stall_mask = torch.rand(128, device=DEVICE) < 0.15
        a_train = torch.where(stall_mask.view(-1, 1, 1), torch.zeros_like(a_b), a_b)
        n_train = torch.where(stall_mask.view(-1, 1), o_b, n_b)
        gv_train = torch.where(stall_mask, torch.full_like(gv_b, 0.05), gv_b)

        with torch.no_grad():
            s_b = obs_enc_vla(o_b)
            tgt_fh = F.layer_norm(obs_enc_vla(n_train), (D,))
        sim = world.imagine_futures(s_b, a_train.unsqueeze(1))
        fh = sim["future_latent"].squeeze(1)
        l_jepa = F.mse_loss(fh, tgt_fh)
        l_sf = evid_loss(world.future_safe_noul(fh)["alpha"], sf_b)["loss"]
        l_gv = F.mse_loss(sim["future_goal_value"].squeeze(1), gv_train)
        loss_w = l_jepa + 2.5 * l_sf + 3.5 * l_gv
        opt_w.zero_grad()
        loss_w.backward()
        opt_w.step()

    obs_enc_l2.eval()
    l2_policy.eval()
    vla.eval()
    world.eval()
    planner = RiskSensitiveLatentPlanner(
        world, num_env_draws=4, cvar_alpha=0.25, w_goal=3.0, w_cvar_safe=3.5, w_epistemic=0.5
    ).to(DEVICE)

    print("Evaluating all 3 policies in REAL closed-loop MuJoCo Reacher-v5 rollouts (80 episodes each)...")
    res_l2 = _evaluate_in_mujoco("l2_oft_baseline", obs_enc_l2, obs_enc_vla, l2_policy, vla, planner, num_episodes=80, chunk_h=chunk_h)
    res_vla = _evaluate_in_mujoco("vla_flow_energy_rtc", obs_enc_l2, obs_enc_vla, l2_policy, vla, planner, num_episodes=80, chunk_h=chunk_h)
    res_mpc = _evaluate_in_mujoco("v3_mental_sim_cvar_mpc", obs_enc_l2, obs_enc_vla, l2_policy, vla, planner, num_episodes=80, chunk_h=chunk_h)

    report = {
        "environment": "MuJoCo 3.17 Reacher-v5 (Real Continuous Physics + Physical Obstacle + Dynamic Hazard)",
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "l2_oft_parallel_baseline": res_l2,
        "omnijev_v3_flow_energy_rtc": res_vla,
        "omnijev_v3_latent_jepa_cvar_mpc": res_mpc,
    }
    print("\n[Real MuJoCo Closed-Loop Results]\n", json.dumps(report, indent=2))
    out_path = ROOT / "experiments" / "v3_mujoco_vla_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved MuJoCo results to: {out_path}")


if __name__ == "__main__":
    main()
