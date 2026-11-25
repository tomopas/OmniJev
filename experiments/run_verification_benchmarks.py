"""Empirical PyTorch/CUDA Verification Benchmarks for OmniJev (Axes A-E).

Runs 7 concrete quantitative experiments validating the core architectural claims:
  1. [Axis A+D] Exact O(1) Permutation Invariance of Choice & Rank (0% flip rate in 1 pass)
  2. [Axis C]   Jev-VLA Shared-z Single-Pass Action Chunking vs. L1 Regression on Bimodal Obstacle Avoidance + RTC
  3. [Axis D]   System 1.5 Latent Deliberation (K=0 vs K=2 vs K=4) on Multi-Hop Relational Synthesis (0 text tokens!)
  4. [Axis D]   Evidential + Mahalanobis OOD + Split Conformal Abstention & Cost-Aware S1->S2 Routing Pareto Frontier
  5. [Axis D]   Streaming Gated DeltaNet + SWA Exact Incremental Equivalence & Constant O(1) Memory Scaling
  6. [Axis C+D] Composable Decision Graph (Coarse Box -> High-Res Crop -> Exact Point & Dynamic Span->Choice)
  7. [Axis E]   Calibration-Preserving Mixed-Precision Quantization (INT4 Backbone + FP16 Heads) & 5-Tier Roofline Table
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnijev import (
    CDGNode,
    ComposableDecisionGraph,
    CostAwareEscalationRouter,
    EvidentialDirichletLoss,
    ModalityBatch,
    OmniJevConfig,
    OmniJevModel,
    PrototypeMahalanobisOOD,
    RealTimeActionChunkBlender,
    SplitConformalCalibrator,
    get_tier_config,
    profile_model_tier,
    simulate_mixed_precision_quantization,
)
from omnijev.system15 import compute_ece


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# EXP 1: Exact O(1) Permutation Invariance (Axis A + D)
# ===========================================================================
def exp1_permutation_invariance() -> Dict[str, float]:
    torch.manual_seed(1)
    cfg = get_tier_config("micro")
    model = OmniJevModel(cfg).to(DEVICE).eval()

    B, C = 8, 8
    batch = ModalityBatch(text_ids=torch.randint(0, cfg.vocab_size, (B, 16), device=DEVICE))
    with torch.no_grad():
        enc = model.encode_state(batch, deliberation_steps=0)
        H, q = enc["state_h"], model.query_vector(enc)
        opt_embs = model.encode_isolated_options(torch.randint(0, cfg.vocab_size, (B, C, 6), device=DEVICE))
        base_ch = model.primitives.choice(H, q, opt_embs)
        base_rk = model.primitives.rank(q, opt_embs)

        max_ch_diff = 0.0
        max_rk_diff = 0.0
        flips = 0
        trials = 50
        for _ in range(trials):
            perm = torch.randperm(C, device=DEVICE)
            inv = torch.argsort(perm)
            ch_p = model.primitives.choice(H, q, opt_embs[:, perm, :])
            rk_p = model.primitives.rank(q, opt_embs[:, perm, :])
            diff_ch = (base_ch["option_probs"] - ch_p["option_probs"][:, inv]).abs().max().item()
            diff_rk = (base_rk["utilities"] - rk_p["utilities"][:, inv]).abs().max().item()
            max_ch_diff = max(max_ch_diff, diff_ch)
            max_rk_diff = max(max_rk_diff, diff_rk)
            # Check if top winner ever flipped
            orig_winner = base_ch["choice_idx"]
            perm_winner_orig_idx = perm[ch_p["choice_idx"]]
            flips += int((orig_winner != perm_winner_orig_idx).sum().item())

    return {
        "trials": trials * B,
        "choice_max_prob_shift": max_ch_diff,
        "rank_max_utility_shift": max_rk_diff,
        "choice_flip_rate_pct": 100.0 * flips / (trials * B),
    }


# ===========================================================================
# EXP 2: Axis C — Jev-VLA Shared-z vs. Parallel L1 Regression (Bimodal Obstacle)
# ===========================================================================
def _generate_bimodal_trajectories(N: int, H: int = 16, device=DEVICE) -> tuple:
    """2D robot moves from (0, 0) to (1, 0) avoiding a circular obstacle at (0.5, 0.0) of radius 0.22.

    50% of demonstrations arc LEFT (+y), 50% arc RIGHT (-y).
    Pure L1/L2 regression averages the two modes -> y=0 -> crashes straight into obstacle!
    """
    t = torch.linspace(0.0, 1.0, H, device=device).view(1, H)
    # Random slight start/goal perturbations conditioned in observation context
    y_shift = (torch.rand(N, 1, device=device) - 0.5) * 0.04
    sign = torch.where(torch.rand(N, 1, device=device) < 0.5, 1.0, -1.0)
    amp = 0.42 + (torch.rand(N, 1, device=device) - 0.5) * 0.04

    x = t.expand(N, H)
    y = y_shift + sign * amp * torch.sin(math.pi * t)
    traj = torch.stack([x, y], dim=-1)  # [N, H, 2]

    # Observation token embedding carries y_shift (NOT which branch sign to choose — true multimodal ambiguity!)
    obs_feat = torch.cat([y_shift, torch.zeros(N, 7, device=device)], dim=-1)  # [N, 8]
    return obs_feat, traj


def _collision_and_mode_metrics(chunks: torch.Tensor, obs_center=(0.5, 0.0), obs_r: float = 0.22) -> Dict[str, float]:
    # chunks: [N, H, 2]
    dx = chunks[..., 0] - obs_center[0]
    dy = chunks[..., 1] - obs_center[1]
    min_dist = torch.sqrt(dx.pow(2) + dy.pow(2)).min(dim=-1).values
    collided = (min_dist < obs_r).float().mean().item()
    mid_y = chunks[:, chunks.size(1) // 2, 1]
    left_frac = (mid_y > obs_r).float().mean().item()
    right_frac = (mid_y < -obs_r).float().mean().item()
    endpoint_err = (chunks[:, -1, 0] - 1.0).abs().mean().item() + chunks[:, -1, 1].abs().mean().item()
    return {
        "collision_rate_pct": round(100.0 * collided, 2),
        "left_mode_pct": round(100.0 * left_frac, 2),
        "right_mode_pct": round(100.0 * right_frac, 2),
        "endpoint_l1": round(endpoint_err, 4),
    }


def exp2_vla_bimodal_obstacle_and_rtc() -> Dict[str, object]:
    torch.manual_seed(7)
    H = 16
    cfg = OmniJevConfig(
        d_model=64, n_heads=4, d_ff=128, proprio_dim=8,
        vla_action_dim=2, vla_chunk_horizon=H, vla_z_dim=8, vla_head_layers=2,
    )
    obs_proj = nn.Linear(8, cfg.d_model).to(DEVICE)

    # 1a) Baseline: Parallel L2 Regression VLA (averages bimodal left/right -> center collision!)
    l2_baseline = nn.Sequential(
        nn.Linear(cfg.d_model, 128), nn.GELU(),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, H * 2),
    ).to(DEVICE)

    # 1b) Baseline: Parallel L1 Regression VLA (OpenVLA-OFT style, collapses/clips modes!)
    l1_baseline = nn.Sequential(
        nn.Linear(cfg.d_model, 128), nn.GELU(),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, H * 2),
    ).to(DEVICE)

    # 2) OmniJev Jev-VLA: Single-Pass Shared-Random-State Multimodal Action Chunking
    from omnijev.embodied import JevVLAHead
    jev_vla = JevVLAHead(cfg).to(DEVICE)

    opt_l2 = torch.optim.AdamW(list(l2_baseline.parameters()) + list(obs_proj.parameters()), lr=3e-3)
    opt_l1 = torch.optim.AdamW(l1_baseline.parameters(), lr=3e-3)
    opt_vla = torch.optim.AdamW(jev_vla.parameters(), lr=3e-3)

    # Train all three on identical bimodal demonstrations
    for step in range(350):
        obs, target = _generate_bimodal_trajectories(128, H=H)
        ctx_tokens = obs_proj(obs).unsqueeze(1).expand(-1, 4, -1)

        # L2 baseline update
        pred_l2 = l2_baseline(ctx_tokens.mean(dim=1)).view(-1, H, 2)
        loss_l2 = F.mse_loss(pred_l2, target)
        opt_l2.zero_grad()
        loss_l2.backward()
        opt_l2.step()

        # L1 baseline update
        pred_l1 = l1_baseline(ctx_tokens.detach().mean(dim=1)).view(-1, H, 2)
        loss_l1 = F.l1_loss(pred_l1, target)
        opt_l1.zero_grad()
        loss_l1.backward()
        opt_l1.step()

        # Jev-VLA shared-z variational update
        loss_dict = jev_vla.training_loss(
            ctx_tokens.detach(), target, proprio_raw=obs, beta_kl=0.015, free_bits=0.05, jerk_weight=0.02
        )
        opt_vla.zero_grad()
        loss_dict["loss"].backward()
        opt_vla.step()

    # Evaluate prior single-pass samples (z ~ N(0, I))
    jev_vla.eval()
    l1_baseline.eval()
    l2_baseline.eval()
    with torch.no_grad():
        obs_test, _ = _generate_bimodal_trajectories(400, H=H)
        ctx_test = obs_proj(obs_test).unsqueeze(1).expand(-1, 4, -1)

        l2_preds = l2_baseline(ctx_test.mean(dim=1)).view(-1, H, 2)
        l1_preds = l1_baseline(ctx_test.mean(dim=1)).view(-1, H, 2)

        # Warmup + steady-state GPU timing
        _ = jev_vla(ctx_test, proprio_raw=obs_test, z=None)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_vla0 = time.perf_counter()
        vla_out = jev_vla(ctx_test, proprio_raw=obs_test, z=None)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        vla_ms = (time.perf_counter() - t_vla0) * 1000.0

        # Best-of-8 collision-safe reranking with zero extra backbone FLOPs!
        def obstacle_penalty(chunks):
            d = torch.sqrt((chunks[..., 0] - 0.5).pow(2) + chunks[..., 1].pow(2)).min(dim=-1).values
            return 10.0 * F.relu(0.26 - d)

        rerank_out = jev_vla.sample_best_of_k(
            ctx_test[:100], proprio_raw=obs_test[:100], num_samples=8, external_cost_fn=obstacle_penalty
        )

        # RTC Overlap Blending jerk reduction check between two consecutive chunks
        blender = RealTimeActionChunkBlender(blend_steps=4)
        c_prev = vla_out["action_chunk"][:32]
        c_next = c_prev + 0.08
        raw_join = torch.cat([c_prev[:, :6], c_next[:, :6]], dim=1)
        rtc_next = blender.blend(c_prev, c_next, exec_offset=6)
        rtc_join = torch.cat([c_prev[:, :6], rtc_next[:, :6]], dim=1)
        raw_jerk = (raw_join[:, 2:] - 2 * raw_join[:, 1:-1] + raw_join[:, :-2]).pow(2).mean().item()
        rtc_jerk = (rtc_join[:, 2:] - 2 * rtc_join[:, 1:-1] + rtc_join[:, :-2]).pow(2).mean().item()

    return {
        "l2_mean_baseline": _collision_and_mode_metrics(l2_preds),
        "l1_oft_baseline": _collision_and_mode_metrics(l1_preds),
        "jev_vla_single_pass": _collision_and_mode_metrics(vla_out["action_chunk"]),
        "jev_vla_best_of_8": _collision_and_mode_metrics(rerank_out["best_chunk"]),
        "vla_batch400_ms": round(vla_ms, 2),
        "rtc_boundary_jerk_reduction_pct": round(100.0 * (1.0 - rtc_jerk / max(raw_jerk, 1e-8)), 1),
    }


# ===========================================================================
# EXP 3: Axis D — System 1.5 Latent Deliberation (Multi-Hop Pointer Chasing)
# ===========================================================================
class _MultiHopSystem15Probe(nn.Module):
    """Unified System-1.5 probe trained with step-weighted deep supervision & TaH2 halting decider.

    Evaluates a single weight-tied model across latent depths K=0, 1, 2, 3, 4 on 6-hop graph chasing
    WITHOUT generating any autoregressive text tokens!
    """

    def __init__(self, n_nodes: int = 8, d_model: int = 96):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_emb = nn.Embedding(n_nodes, d_model)
        self.next_emb = nn.Embedding(n_nodes, d_model)
        self.query_flag = nn.Embedding(2, d_model)
        cfg = OmniJevConfig(d_model=d_model, n_heads=4, d_ff=256, n_recurrent_layers=1, max_deliberation_steps=4)
        from omnijev.backbone import System15LatentDeliberationCore
        self.core = System15LatentDeliberationCore(cfg)
        self.readout = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_nodes))

    def forward_all_depths(self, perm: torch.Tensor, start_node: torch.Tensor, max_k: int = 4) -> List[torch.Tensor]:
        B, N = perm.shape
        ids = torch.arange(N, device=perm.device).unsqueeze(0).expand(B, N)
        is_start = (ids == start_node.unsqueeze(1)).long()
        h0 = self.node_emb(ids) + self.next_emb(perm) + self.query_flag(is_start)
        delib = self.core(h0, num_steps=max_k, adaptive_halt=False)
        gather_idx = start_node.view(B, 1, 1).expand(B, 1, h0.size(-1))
        return [self.readout(hs.gather(1, gather_idx).squeeze(1)) for hs in delib["step_states"]]


def _sample_multihop_batch(B: int, N: int = 8, hops: int = 6, device=DEVICE):
    """Random permutation graph: compute intermediate & final hops f^(1..hops)(start_node)."""
    perms = torch.stack([torch.randperm(N, device=device) for _ in range(B)], dim=0)
    starts = torch.randint(0, N, (B,), device=device)
    curr = starts.clone()
    traj = [curr.clone()]
    for _ in range(hops):
        curr = perms[torch.arange(B, device=device), curr]
        traj.append(curr.clone())
    return perms, starts, traj


def exp3_system15_multihop_deliberation() -> Dict[str, float]:
    """Verify monotonic test-time latent scaling K=0 -> K=4 on 6-hop graph chasing (0 text tokens!)."""
    torch.manual_seed(11)
    N, HOPS = 8, 6
    probe = _MultiHopSystem15Probe(n_nodes=N, d_model=96).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=2.0e-3, weight_decay=1e-4)

    # Step-weighted deep supervision across K=1..4 (exact System 1.5 training recipe from Doc 04!)
    depth_weights = [0.1, 0.15, 0.25, 0.45, 1.0]
    for _ in range(1100):
        perms, starts, traj = _sample_multihop_batch(128, N=N, hops=HOPS)
        target_final = traj[HOPS]
        all_logits = probe.forward_all_depths(perms, starts, max_k=4)
        loss = sum(w * F.cross_entropy(lg, target_final) for w, lg in zip(depth_weights, all_logits))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()

    probe.eval()
    results: Dict[str, float] = {}
    with torch.no_grad():
        perms_t, starts_t, traj_t = _sample_multihop_batch(1024, N=N, hops=HOPS)
        target_t = traj_t[HOPS]
        # Warmup
        _ = probe.forward_all_depths(perms_t, starts_t, max_k=4)
        for k_steps in (0, 1, 2, 3, 4):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            lg = probe.forward_all_depths(perms_t, starts_t, max_k=k_steps)[-1]
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            acc = (lg.argmax(dim=-1) == target_t).float().mean().item() * 100.0
            results[f"K={k_steps}_acc_pct"] = round(acc, 2)
            results[f"K={k_steps}_batch1024_ms"] = round(dt_ms, 2)

        # Demonstrate Adaptive Halting savings: stop per-sample as soon as KL(p^(k) || p^(k-1)) < 1e-3
        all_lg = probe.forward_all_depths(perms_t, starts_t, max_k=4)
        probs = [F.softmax(x, dim=-1) for x in all_lg]
        halted_k = torch.full((1024,), 4, device=DEVICE, dtype=torch.long)
        halted_pred = all_lg[4].argmax(dim=-1)
        still_active = torch.ones(1024, device=DEVICE, dtype=torch.bool)
        for k in range(1, 4):
            kl = (probs[k] * (probs[k].clamp_min(1e-7).log() - probs[k - 1].clamp_min(1e-7).log())).sum(dim=-1)
            conf = probs[k].max(dim=-1).values
            can_stop = still_active & (kl < 0.02) & (conf > 0.92)
            halted_k = torch.where(can_stop, torch.full_like(halted_k, k), halted_k)
            halted_pred = torch.where(can_stop, all_lg[k].argmax(dim=-1), halted_pred)
            still_active = still_active & (~can_stop)

        results["adaptive_halt_avg_K"] = round(halted_k.float().mean().item(), 2)
        results["adaptive_halt_acc_pct"] = round((halted_pred == target_t).float().mean().item() * 100.0, 2)

    return results


# ===========================================================================
# EXP 4: Axis D — Evidential + OOI + Mahalanobis OOD + Split Conformal + S1->S2
# ===========================================================================
def exp4_evidential_conformal_and_routing() -> Dict[str, float]:
    torch.manual_seed(23)
    D, C = 64, 5
    # 5 Gaussian semantic clusters in R^D; option vectors are cluster prototypes
    prototypes = F.normalize(torch.randn(C, D, device=DEVICE), dim=-1) * 2.5

    class _EvidClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
            from omnijev.primitives import OrderInvariantChoiceHead
            self.choice = OrderInvariantChoiceHead(D, n_heads=4, set_layers=1)

        def forward(self, x, opts):
            q = self.enc(x)
            h = q.unsqueeze(1)
            out = self.choice(h, q, opts, include_ooi_sink=True)
            out["q_latent"] = q
            return out

    clf = _EvidClassifier().to(DEVICE)
    evid_loss_fn = EvidentialDirichletLoss(kl_weight=0.08)
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-3)

    def sample_batch(n: int, ooi_frac: float = 0.20, noise_std: float = 0.75):
        y = torch.randint(0, C, (n,), device=DEVICE)
        x = prototypes[y] + torch.randn(n, D, device=DEVICE) * noise_std
        # Candidate inventory has C-1 = 4 options:
        #   - In-inventory (1 - ooi_frac): includes true class + 3 distractors
        #   - Out-of-inventory (ooi_frac): drops true class -> target = OOI sink (index 4)!
        opts = torch.zeros(n, 4, D, device=DEVICE)
        targets = torch.zeros(n, dtype=torch.long, device=DEVICE)
        for i in range(n):
            true_c = int(y[i].item())
            others = [c for c in range(C) if c != true_c]
            if torch.rand(1).item() < ooi_frac:
                # Drop true class -> OOI target is index 4
                opts[i] = prototypes[others]
                targets[i] = 4
            else:
                perm_others = [others[j] for j in torch.randperm(len(others))[:3].tolist()]
                inv_c = [true_c] + perm_others
                shuf = torch.randperm(4)
                inv_shuf = [inv_c[j] for j in shuf.tolist()]
                opts[i] = prototypes[inv_shuf]
                targets[i] = inv_shuf.index(true_c)
        return x, opts, targets

    # Train Evidential Choice head with synthetic OOI negatives
    for step in range(400):
        x, opts, tgt = sample_batch(128, ooi_frac=0.22)
        out = clf(x, opts)
        loss = evid_loss_fn(out["alpha"], tgt, anneal_coef=min(1.0, step / 150.0))["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()

    clf.eval()
    ood_det = PrototypeMahalanobisOOD(D, num_prototypes=12).to(DEVICE)
    conformal = SplitConformalCalibrator(alpha=0.10, lambda_epi=0.4, lambda_ood=0.4)
    router = CostAwareEscalationRouter(error_cost=10.0, s2_cost=0.6, ooi_threshold=0.32, ood_threshold=0.65)

    with torch.no_grad():
        # Fit Mahalanobis OOD & Split Conformal on held-out in-inventory calibration split
        x_cal, opts_cal, tgt_cal = sample_batch(500, ooi_frac=0.0)
        cal_out = clf(x_cal, opts_cal)
        ood_det.fit(cal_out["q_latent"])
        cal_ood = ood_det(cal_out["q_latent"])["ood_score"]
        conformal.calibrate(cal_out["option_probs"], tgt_cal, cal_out["epistemic_vacuity"], cal_ood)

        # Test 1: In-inventory accuracy, ECE, and Finite-Sample Conformal Coverage (target >= 90%)
        x_id, opts_id, tgt_id = sample_batch(600, ooi_frac=0.0)
        id_out = clf(x_id, opts_id)
        id_ood = ood_det(id_out["q_latent"])["ood_score"]
        id_sets = conformal.predict_sets(id_out["option_probs"], id_out["epistemic_vacuity"], id_ood)
        id_acc = (id_out["choice_idx"] == tgt_id).float().mean().item() * 100.0
        id_ece = compute_ece(id_out["option_probs"], tgt_id)
        conformal_cov = id_sets["prediction_set"][torch.arange(600, device=DEVICE), tgt_id].float().mean().item() * 100.0

        # Test 2: Out-of-Inventory (true label dropped!) detection
        x_ooi, opts_ooi, _ = sample_batch(300, ooi_frac=1.0)
        ooi_out = clf(x_ooi, opts_ooi)
        ooi_recall = (ooi_out["ooi_prob"] > 0.32).float().mean().item() * 100.0
        id_false_ooi = (id_out["ooi_prob"] > 0.32).float().mean().item() * 100.0

        # Test 3: Far OOD detection via Mahalanobis + Epistemic Vacuity
        x_far = torch.randn(300, D, device=DEVICE) * 3.0 + 4.0
        far_out = clf(x_far, opts_id[:300])
        far_ood = ood_det(far_out["q_latent"])["ood_score"]
        far_caught = ((far_ood > 0.65) | (far_out["epistemic_vacuity"] > 0.45)).float().mean().item() * 100.0

        # Test 4: Combined S1 -> S2 Escalation Pareto (assuming S2 has 98% accuracy)
        rt = router.route(
            id_out["option_probs"], id_out["epistemic_vacuity"],
            ooi_prob=id_out["ooi_prob"], ood_score=id_ood,
            conformal_set_size=id_sets["set_size"], current_depth=4, max_depth=4,
        )
        exec_mask = rt["route_code"] == 0
        s1_exec_pct = exec_mask.float().mean().item() * 100.0
        s1_accepted_acc = (
            (id_out["choice_idx"][exec_mask] == tgt_id[exec_mask]).float().mean().item() * 100.0
            if exec_mask.any() else 100.0
        )
        hybrid_acc = (s1_exec_pct / 100.0) * s1_accepted_acc + (1.0 - s1_exec_pct / 100.0) * 98.0

    return {
        "in_domain_s1_acc_pct": round(id_acc, 2),
        "in_domain_ece": round(id_ece, 4),
        "conformal_coverage_target90_pct": round(conformal_cov, 2),
        "ooi_detection_recall_pct": round(ooi_recall, 2),
        "id_false_ooi_rate_pct": round(id_false_ooi, 2),
        "far_ood_caught_pct": round(far_caught, 2),
        "s1_auto_execute_pct": round(s1_exec_pct, 2),
        "s1_accepted_precision_pct": round(s1_accepted_acc, 2),
        "hybrid_s1_s2_accuracy_pct": round(hybrid_acc, 2),
    }


# ===========================================================================
# EXP 5: Axis D — Streaming Gated DeltaNet + SWA Equivalence & O(1) Memory
# ===========================================================================
def exp5_streaming_state_benchmark() -> Dict[str, object]:
    torch.manual_seed(3)
    cfg = get_tier_config("micro")
    model = OmniJevModel(cfg).to(DEVICE).eval()

    B, total_len, chunk_len = 2, 128, 16
    ids = torch.randint(0, cfg.vocab_size, (B, total_len), device=DEVICE)

    with torch.no_grad():
        # Warmup CUDA kernels for true steady-state latency
        _ = model.encode_state(ModalityBatch(text_ids=ids[:, :chunk_len]), deliberation_steps=0)
        full_enc = model.encode_state(ModalityBatch(text_ids=ids), deliberation_steps=0)

        s_state = None
        chunks = []
        chunk_times_ms = []
        state_bytes_trace = []
        for pos in range(0, total_len, chunk_len):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            c_enc = model.encode_state(
                ModalityBatch(text_ids=ids[:, pos: pos + chunk_len]),
                streaming_state=s_state,
                deliberation_steps=0,
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            chunk_times_ms.append((time.perf_counter() - t0) * 1000.0)
            s_state = c_enc["streaming_state"]
            chunks.append(c_enc["prelude_h"])
            gdn_b = sum(v.numel() * v.element_size() for v in s_state.gdn_states.values())
            swa_b = sum((k.numel() * k.element_size() + v.numel() * v.element_size()) for k, v in s_state.swa_caches.values())
            state_bytes_trace.append(gdn_b + swa_b)

        streamed = torch.cat(chunks, dim=1)
        max_err = (full_enc["prelude_h"] - streamed).abs().max().item()

    return {
        "stream_tokens": total_len,
        "chunk_size": chunk_len,
        "max_equivalence_error": round(max_err, 6),
        "first_chunk_ms": round(chunk_times_ms[0], 2),
        "last_chunk_ms": round(chunk_times_ms[-1], 2),
        "state_kb_at_chunk1": round(state_bytes_trace[0] / 1024.0, 2),
        "state_kb_at_chunk8": round(state_bytes_trace[-1] / 1024.0, 2),
        "memory_capped_o1": state_bytes_trace[-1] == state_bytes_trace[-2],
    }


# ===========================================================================
# EXP 6: Axis C + D — Composable Decision Graph (Coarse Box -> Zoom Crop -> Exact Point)
# ===========================================================================
def exp6_composable_decision_graph_zoom() -> Dict[str, object]:
    torch.manual_seed(5)
    cfg = get_tier_config("micro")
    model = OmniJevModel(cfg).to(DEVICE).eval()

    img = torch.randn(1, 3, 64, 64, device=DEVICE)
    goal = torch.randint(0, cfg.vocab_size, (1, 8), device=DEVICE)

    with torch.no_grad():
        cdg = ComposableDecisionGraph(entry_node="coarse_pass")

        def coarse_pass(ctx):
            enc = model.encode_state(ModalityBatch(text_ids=ctx["goal"], images=ctx["img"]), deliberation_steps=0)
            q = model.query_vector(enc)
            cua = model.cua(enc["state_h"], enc["spatial_map"], q)
            return {"enc": enc, "q": q, "cua": cua}

        def zoom_crop_pass(ctx):
            box = ctx["coarse_pass"]["cua"]["box"][0]  # [top, left, bottom, right] in (0, 1)
            H_im, W_im = ctx["img"].shape[-2:]
            y1 = max(0, int(box[0].item() * H_im))
            x1 = max(0, int(box[1].item() * W_im))
            y2 = min(H_im, max(y1 + 8, int(box[2].item() * H_im)))
            x2 = min(W_im, max(x1 + 8, int(box[3].item() * W_im)))
            crop = F.interpolate(ctx["img"][:, :, y1:y2, x1:x2], size=(32, 32), mode="bilinear", align_corners=False)
            enc_z = model.encode_state(ModalityBatch(text_ids=ctx["goal"], images=crop), deliberation_steps=0)
            q_z = model.query_vector(enc_z)
            pt_z = model.primitives.point(enc_z["spatial_map"], q_z)
            # Map local crop coordinate back into global normalized coordinates
            gx = (x1 + pt_z["point"][0, 0].item() * (x2 - x1)) / W_im
            gy = (y1 + pt_z["point"][0, 1].item() * (y2 - y1)) / H_im
            return {"global_point": (round(gx, 4), round(gy, 4)), "crop_px": (x1, y1, x2, y2)}

        cdg.add_node(CDGNode("coarse_pass", coarse_pass, next_fn=lambda ctx: "zoom_crop_pass"))
        cdg.add_node(CDGNode("zoom_crop_pass", zoom_crop_pass, next_fn=lambda ctx: None))

        # Warmup cuDNN kernels before steady-state timing
        _ = cdg.execute({"img": img, "goal": goal})
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = cdg.execute({"img": img, "goal": goal})

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "cdg_trace": res["_cdg_trace"],
        "refined_global_point": res["zoom_crop_pass"]["global_point"],
        "crop_box_px": res["zoom_crop_pass"]["crop_px"],
        "end_to_end_2pass_ms": round(dt_ms, 2),
    }


# ===========================================================================
# EXP 7: Axis E — Mixed-Precision Calibration Preservation & 5-Tier Roofline Table
# ===========================================================================
def exp7_quantization_and_tier_profiles() -> Dict[str, object]:
    torch.manual_seed(9)
    cfg = get_tier_config("micro")
    model = OmniJevModel(cfg).to(DEVICE).eval()

    # Compare output probability L1 distortion & epistemic drift under:
    #   (a) Naive uniform 4-bit quantization (all layers including heads)
    #   (b) OmniJev calibration-preserving mixed precision (4-bit backbone, 16-bit heads)
    q_naive = simulate_mixed_precision_quantization(model, backbone_bits=4, preserve_heads=False)
    q_mixed = simulate_mixed_precision_quantization(model, backbone_bits=4, preserve_heads=True)

    batch = ModalityBatch(
        text_ids=torch.randint(0, cfg.vocab_size, (16, 16), device=DEVICE),
        images=torch.randn(16, 3, 32, 32, device=DEVICE),
    )
    opts = torch.randint(0, cfg.vocab_size, (16, 6, 4), device=DEVICE)

    def eval_outputs(m: OmniJevModel):
        with torch.no_grad():
            e = m.encode_state(batch, deliberation_steps=0)
            q = m.query_vector(e)
            oe = m.encode_isolated_options(opts)
            ch = m.primitives.choice(e["state_h"], q, oe)
            pt = m.primitives.point(e["spatial_map"], q)
            return ch["option_probs"], ch["epistemic_vacuity"], pt["point"]

    fp_p, fp_v, fp_pt = eval_outputs(model)
    nv_p, nv_v, nv_pt = eval_outputs(q_naive)
    mx_p, mx_v, mx_pt = eval_outputs(q_mixed)

    tier_table = [
        profile_model_tier("nano", "Jetson_Orin_Nano", seq_len=128, deliberation_steps=0),
        profile_model_tier("edge", "Jetson_Orin_AGX", seq_len=256, deliberation_steps=0),
        profile_model_tier("edge", "RTX4090_FP8", seq_len=256, deliberation_steps=2),
        profile_model_tier("base", "H200_FP8", seq_len=512, deliberation_steps=0),
        profile_model_tier("base", "H200_FP8", seq_len=512, deliberation_steps=4),
        profile_model_tier("pro", "H200_FP8", seq_len=512, deliberation_steps=0),
        profile_model_tier("pro", "H200_FP8", seq_len=512, deliberation_steps=4),
        profile_model_tier("world", "H200_FP8", seq_len=512, deliberation_steps=0, include_jeveration=True),
    ]

    return {
        "naive_int4_prob_l1_drift": round((fp_p - nv_p).abs().mean().item(), 5),
        "mixed_int4_fp16head_prob_l1_drift": round((fp_p - mx_p).abs().mean().item(), 5),
        "naive_int4_point_l1_drift": round((fp_pt - nv_pt).abs().mean().item(), 5),
        "mixed_int4_fp16head_point_l1_drift": round((fp_pt - mx_pt).abs().mean().item(), 5),
        "naive_int4_vacuity_drift": round((fp_v - nv_v).abs().mean().item(), 5),
        "mixed_int4_fp16head_vacuity_drift": round((fp_v - mx_v).abs().mean().item(), 5),
        "tier_profiles": tier_table,
    }


def main():
    print(f"Running OmniJev Verification Benchmarks on device: {DEVICE}\n" + "=" * 72)
    report = {}
    for name, fn in [
        ("exp1_permutation_invariance", exp1_permutation_invariance),
        ("exp2_vla_bimodal_obstacle_and_rtc", exp2_vla_bimodal_obstacle_and_rtc),
        ("exp3_system15_multihop_deliberation", exp3_system15_multihop_deliberation),
        ("exp4_evidential_conformal_and_routing", exp4_evidential_conformal_and_routing),
        ("exp5_streaming_state_benchmark", exp5_streaming_state_benchmark),
        ("exp6_composable_decision_graph_zoom", exp6_composable_decision_graph_zoom),
        ("exp7_quantization_and_tier_profiles", exp7_quantization_and_tier_profiles),
    ]:
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        report[name] = res
        print(f"\n[{name}] ({dt:.2f}s)")
        print(json.dumps(res, indent=2))

    out_path = ROOT / "experiments" / "verification_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
