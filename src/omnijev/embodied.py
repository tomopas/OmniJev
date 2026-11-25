"""Axis C (Deep Implementation) — Embodied Control, Jev-VLA, Jev-CUA, Guardrails, & Dual-Rate Control.

Implements:
  1. JevVLAHead: Single-Pass Multimodal Action Chunking conditioned on Shared Random State z
     (solving bimodal obstacle mode-averaging!) + FAST-DCT discrete co-training + co-predicted
     safety/progress/completion + Best-of-Bz zero-backbone-overhead safe trajectory reranking.
  2. RealTimeActionChunkBlender (RTC): Latency-compensated asynchronous raised-cosine chunk blending.
  3. JevCUAHead: Single-Pass Computer-Use & Browser Agent outputting Choice(action) + action-conditioned
     Point/Box + optional DOM Choice + Scalar scroll/wait + Hotkey + Noul safety + selective S2 wakeup.
  4. ToolGuardrailHead: Zero-hallucination inline pre-execution policy verifier (Noul + Choice + Score + Span + Box).
  5. HierarchicalDualRateController: Formal Slow-Planner (S2) + Fast-Reactor (OmniJev S1/1.5) orchestrator.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OmniJevConfig
from .jeveration import AdaLNJeverationBlock
from .primitives import (
    BiaffineSpanHead,
    BoxHead,
    NoulHead,
    OrderInvariantChoiceHead,
    OrdinalScoreHead,
    PointHead,
    ScalarDistributionHead,
)


class ActionPosteriorEncoder(nn.Module):
    """Training-time trajectory mode encoder q_phi(z | a_{t:t+H}, H_t) preventing mode collapse in Jev-VLA."""

    def __init__(self, action_dim: int, horizon: int, d_model: int, z_dim: int):
        super().__init__()
        self.traj_enc = nn.Sequential(
            nn.Linear(action_dim * horizon, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.post = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * z_dim),
        )

    def forward(self, target_chunk: torch.Tensor, context_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = target_chunk.size(0)
        t_feat = self.traj_enc(target_chunk.reshape(B, -1))
        h_feat = context_h.mean(dim=1)
        mu, logvar = self.post(torch.cat([t_feat, h_feat], dim=-1)).chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 4.0)
        std = (0.5 * logvar).exp()
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar


class JevVLAHead(nn.Module):
    """Single-Pass Vision-Language-Action (VLA) Head with Shared-Random-State Multimodal Action Chunking.

    Produces in ONE forward pass:
      - Coherent H-step continuous trajectory chunk a_{t:t+H} conditioned on shared z (zero mode-averaging!)
      - Auxiliary FAST-DCT discrete action tokens (co-training representation)
      - Co-predicted physical safety (Noul), subgoal progress (Score), subgoal complete (Noul)
      - Optional Best-of-Bz trajectory reranking reusing cached backbone H_t (zero extra backbone cost!)
    """

    def __init__(self, cfg: OmniJevConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.horizon = cfg.vla_chunk_horizon
        self.action_dim = cfg.vla_action_dim
        self.z_dim = cfg.vla_z_dim

        self.step_queries = nn.Parameter(torch.randn(1, self.horizon, D) * 0.02)
        self.proprio_skip = nn.Sequential(nn.Linear(cfg.proprio_dim, D), nn.GELU(), nn.Linear(D, D))
        self.z_proj = nn.Sequential(nn.Linear(self.z_dim, D), nn.SiLU(), nn.Linear(D, D))
        self.blocks = nn.ModuleList(
            [AdaLNJeverationBlock(D, cfg.n_heads, cfg.d_ff) for _ in range(cfg.vla_head_layers)]
        )
        self.out_norm = nn.LayerNorm(D)

        # Dual action outputs: continuous chunk + FAST-DCT discrete action tokens
        self.cont_action_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, self.action_dim))
        self.fast_token_queries = nn.Parameter(torch.randn(1, cfg.vla_fast_tokens, D) * 0.02)
        self.fast_head = nn.Linear(D, cfg.vla_fast_vocab)

        # Co-predicted typed control & safety monitors
        self.traj_safety_noul = NoulHead(D)
        self.progress_score = OrdinalScoreHead(D, num_levels=cfg.max_score_levels)
        self.subgoal_done_noul = NoulHead(D)

        self.posterior = ActionPosteriorEncoder(self.action_dim, self.horizon, D, self.z_dim)

    def forward(
        self,
        context_h: torch.Tensor,
        proprio_raw: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = context_h.size(0)
        if z is None:
            z = torch.randn(B, self.z_dim, device=context_h.device, dtype=context_h.dtype)

        q = self.step_queries.expand(B, -1, -1)
        if proprio_raw is not None:
            q = q + self.proprio_skip(proprio_raw).unsqueeze(1)
        z_cond = self.z_proj(z)
        q = q + z_cond.unsqueeze(1)

        for blk in self.blocks:
            q = blk(q, context_h, z_cond)
        q_out = self.out_norm(q)                                  # [B, H, D]
        action_chunk = self.cont_action_head(q_out)               # [B, H, D_act]

        # Auxiliary FAST discrete codes from pooled trajectory representation
        q_pool = q_out.mean(dim=1)                                # [B, D]
        fast_logits = self.fast_head(self.fast_token_queries.expand(B, -1, -1) + q_pool.unsqueeze(1))

        # Co-predicted safety & progress
        safety = self.traj_safety_noul(q_pool)
        progress = self.progress_score(q_pool)
        done = self.subgoal_done_noul(q_pool)

        return {
            "action_chunk": action_chunk,                         # [B, H, D_act]
            "fast_logits": fast_logits,                           # [B, M_fast, V_fast]
            "z": z,                                               # [B, z_dim]
            "is_safe_prob": safety["noul"],                       # [B]
            "safety_epistemic_vacuity": safety["epistemic_vacuity"],
            "progress_score": progress["score"],                  # [B]
            "subgoal_complete_prob": done["noul"],                # [B]
            "traj_embedding": q_pool,                             # [B, D]
        }

    def sample_best_of_k(
        self,
        context_h: torch.Tensor,
        proprio_raw: Optional[torch.Tensor] = None,
        num_samples: int = 8,
        external_cost_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Best-of-Bz trajectory reranking reusing cached backbone H_t (zero extra backbone FLOPs!)."""
        B, L, D = context_h.shape
        h_rep = context_h.unsqueeze(1).expand(B, num_samples, L, D).reshape(B * num_samples, L, D)
        p_rep = None
        if proprio_raw is not None:
            p_rep = proprio_raw.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)

        out = self.forward(h_rep, proprio_raw=p_rep, z=None)
        chunks = out["action_chunk"].view(B, num_samples, self.horizon, self.action_dim)
        safe_p = out["is_safe_prob"].view(B, num_samples)
        prog_s = out["progress_score"].view(B, num_samples)

        utility = 2.0 * safe_p + prog_s
        if external_cost_fn is not None:
            utility = utility - external_cost_fn(chunks)

        best_idx = utility.argmax(dim=-1)                         # [B]
        batch_idx = torch.arange(B, device=context_h.device)
        return {
            "best_chunk": chunks[batch_idx, best_idx],            # [B, H, D_act]
            "all_chunks": chunks,                                 # [B, K, H, D_act]
            "utilities": utility,                                 # [B, K]
            "best_idx": best_idx,
            "best_safe_prob": safe_p[batch_idx, best_idx],
            "best_progress": prog_s[batch_idx, best_idx],
        }

    def training_loss(
        self,
        context_h: torch.Tensor,
        target_chunk: torch.Tensor,
        proprio_raw: Optional[torch.Tensor] = None,
        beta_kl: float = 0.02,
        free_bits: float = 0.1,
        jerk_weight: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """Variational shared-z trajectory training loss with jerk regularization & anti-collapse KL."""
        z_post, mu, logvar = self.posterior(target_chunk, context_h)
        out = self.forward(context_h, proprio_raw=proprio_raw, z=z_post)
        pred = out["action_chunk"]

        l1_loss = F.l1_loss(pred, target_chunk)
        # Second finite difference jerk smoothness penalty
        if self.horizon >= 3:
            acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
            jerk = acc.pow(2).mean()
        else:
            jerk = torch.tensor(0.0, device=pred.device)

        kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl = torch.clamp(kl_dim, min=free_bits).sum(dim=-1).mean()
        total = l1_loss + jerk_weight * jerk + beta_kl * kl
        return {"loss": total, "l1": l1_loss, "jerk": jerk, "kl": kl, "pred_chunk": pred}


class RealTimeActionChunkBlender:
    """Latency-compensated asynchronous Real-Time Action Chunking (RTC) overlap blender."""

    def __init__(self, blend_steps: int = 4):
        self.blend_steps = blend_steps

    def blend(
        self,
        active_chunk: torch.Tensor,
        new_chunk: torch.Tensor,
        exec_offset: int,
    ) -> torch.Tensor:
        """Seamlessly transition from remaining tail of active_chunk into new_chunk using raised-cosine weights.

        Args:
            active_chunk: [..., H, D_act] currently executing chunk
            new_chunk: [..., H, D_act] newly inferred chunk starting at step exec_offset of active_chunk
            exec_offset: number of steps elapsed in active_chunk before new_chunk takes over
        """
        H = active_chunk.size(-2)
        remaining = max(0, H - exec_offset)
        w_len = min(self.blend_steps, remaining, new_chunk.size(-2))
        if w_len <= 0:
            return new_chunk.clone()

        blended = new_chunk.clone()
        t = torch.arange(1, w_len + 1, device=new_chunk.device, dtype=new_chunk.dtype)
        weights = 0.5 * (1.0 - torch.cos(math.pi * t / (w_len + 1)))  # smooth 0 -> 1
        w = weights.view(*([1] * (new_chunk.dim() - 2)), w_len, 1)

        old_slice = active_chunk[..., exec_offset: exec_offset + w_len, :]
        new_slice = new_chunk[..., :w_len, :]
        blended[..., :w_len, :] = (1.0 - w) * old_slice + w * new_slice
        return blended


class JevCUAHead(nn.Module):
    """Single-Pass Computer-Use & Browser Agent (Jev-CUA) Head.

    Simultaneously emits in ONE forward pass:
      1. Action Primitive (Order-Invariant Choice + OOI/Abstain)
      2. Action-conditioned Primary Click Point (2D Spatial Softmax + SimCC)
      3. Region / Drag / Zoom Box (4-side DFL)
      4. Optional Indexed DOM / A11y Element Selection (Order-Invariant Choice)
      5. Continuous Scroll / Wait parameters (Scalar/Distribution)
      6. Hotkey Selector (Choice)
      7. Inline Safety & Loop Termination (3x Noul: destructive, confirm, subgoal_achieved)
      8. Selective System-2 Wakeup decision (wakes LLM only for type_text or epistemic OOD!)
    """

    ACTION_NAMES = (
        "click", "double_click", "right_click", "drag", "scroll",
        "hotkey", "type_text", "wait", "done", "abstain",
    )

    def __init__(self, cfg: OmniJevConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.num_actions = cfg.cua_num_actions
        # Orthonormal unit-norm action embeddings prevent option-collapse in cross/set attention!
        q_init, _ = torch.linalg.qr(torch.randn(D, self.num_actions))
        self.action_catalog = nn.Parameter(q_init.T.unsqueeze(0).contiguous())  # [1, num_actions, D]
        self.action_choice = OrderInvariantChoiceHead(D, cfg.n_heads, set_layers=1)

        # Action-conditioned spatial cross-coupling (zero extra backbone pass!)
        self.action_cond_proj = nn.Linear(D, D)
        self.point_head = PointHead(D, cfg.simcc_bins)
        self.box_head = BoxHead(D, cfg.box_dfl_bins)
        self.dom_choice = OrderInvariantChoiceHead(D, cfg.n_heads, set_layers=1)
        self.scroll_scalar = ScalarDistributionHead(D, n_components=2, quantiles=(0.1, 0.5, 0.9))
        self.hotkey_logits = nn.Linear(D, cfg.cua_num_hotkeys)

        # Inline safety & loop monitors
        self.is_destructive = NoulHead(D)
        self.needs_confirm = NoulHead(D)
        self.subgoal_achieved = NoulHead(D)

    def forward(
        self,
        state_h: torch.Tensor,
        spatial_map: torch.Tensor,
        q_vec: torch.Tensor,
        dom_embs: Optional[torch.Tensor] = None,
        dom_mask: Optional[torch.Tensor] = None,
        epistemic_wake_thresh: float = 0.45,
        zoom_entropy_thresh: float = 0.35,
    ) -> Dict[str, object]:
        B = state_h.size(0)
        act_opts = self.action_catalog.expand(B, -1, -1)
        act_out = self.action_choice(state_h, q_vec, act_opts, include_ooi_sink=True)

        # Expected action embedding modulates spatial query before decoding Point & Box
        act_weights = act_out["option_probs"]                              # [B, num_actions]
        act_emb = torch.einsum("bc,bcd->bd", act_weights, act_opts)        # [B, D]
        q_spatial = q_vec + self.action_cond_proj(act_emb)

        pt_out = self.point_head(spatial_map, q_spatial)
        bx_out = self.box_head(spatial_map, q_spatial)
        scroll_out = self.scroll_scalar(q_spatial)
        hotkey_probs = F.softmax(self.hotkey_logits(q_vec), dim=-1)

        dom_out = None
        if dom_embs is not None:
            dom_out = self.dom_choice(state_h, q_vec, dom_embs, option_mask=dom_mask, include_ooi_sink=True)

        destr = self.is_destructive(q_spatial)
        confirm = self.needs_confirm(q_spatial)
        achieved = self.subgoal_achieved(q_vec)

        # Determine whether System-2 generative LLM must wake up:
        #   - action == "type_text" (idx 6) or "abstain" (idx 9)
        #   - OR action epistemic vacuity > threshold
        #   - OR action OOI probability > threshold
        action_idx = act_out["choice_idx"]
        needs_text_gen = (action_idx == 6)
        explicit_abstain = (action_idx == 9)
        high_epistemic = act_out["epistemic_vacuity"] > epistemic_wake_thresh
        high_ooi = act_out["ooi_prob"] > self.cfg.ooi_threshold
        wake_system2 = needs_text_gen | explicit_abstain | high_epistemic | high_ooi

        # Coarse-to-fine zoom recommendation when spatial confidence is low
        needs_zoom_pass = pt_out["confidence"] < (1.0 - zoom_entropy_thresh)

        return {
            "action_idx": action_idx,
            "action_probs": act_out["option_probs"],
            "action_alpha": act_out["alpha"],
            "action_raw_scores": act_out["raw_scores"],
            "action_ooi_prob": act_out["ooi_prob"],
            "action_epistemic_vacuity": act_out["epistemic_vacuity"],
            "point": pt_out["point"],
            "point_peaks": pt_out["peaks"],
            "point_heatmap_logits": pt_out["heatmap_logits"],
            "point_confidence": pt_out["confidence"],
            "box": bx_out["box"],
            "box_confidence": bx_out["confidence"],
            "dom": dom_out,
            "scroll_delta_mean": scroll_out["mean"],
            "scroll_delta_std": scroll_out["std"],
            "hotkey_idx": hotkey_probs.argmax(dim=-1),
            "hotkey_probs": hotkey_probs,
            "is_destructive_prob": destr["noul"],
            "is_destructive_alpha": destr["alpha"],
            "needs_confirm_prob": confirm["noul"],
            "subgoal_achieved_prob": achieved["noul"],
            "wake_system2": wake_system2,
            "needs_zoom_pass": needs_zoom_pass,
        }


class ToolGuardrailHead(nn.Module):
    """Zero-hallucination inline pre-execution security guardrail & verifier for coding/tool agents.

    Outputs in one parallel pass:
      - is_policy_violation (Noul + epistemic vacuity)
      - risk_category (Order-Invariant Choice + OOI)
      - severity (Ordinal Score 0..4)
      - offending_span (Extractive Biaffine Span highlighting exact risky command/flag)
      - unsafe_region (Optional visual Box if screening GUI/robot image)
    """

    def __init__(self, cfg: OmniJevConfig):
        super().__init__()
        D = cfg.d_model
        self.violation_noul = NoulHead(D)
        self.risk_catalog = nn.Parameter(torch.randn(1, cfg.guardrail_num_risks, D) * 0.02)
        self.risk_choice = OrderInvariantChoiceHead(D, cfg.n_heads, set_layers=1)
        self.severity_score = OrdinalScoreHead(D, num_levels=cfg.max_score_levels)
        self.offending_span = BiaffineSpanHead(D)
        self.unsafe_box = BoxHead(D, cfg.box_dfl_bins)

    def forward(
        self,
        state_h: torch.Tensor,
        q_vec: torch.Tensor,
        spatial_map: Optional[torch.Tensor] = None,
        block_thresh: float = 0.65,
        ask_epistemic_thresh: float = 0.45,
    ) -> Dict[str, object]:
        B = state_h.size(0)
        viol = self.violation_noul(q_vec)
        risk = self.risk_choice(state_h, q_vec, self.risk_catalog.expand(B, -1, -1), include_ooi_sink=True)
        sev = self.severity_score(q_vec)
        span = self.offending_span(state_h, q_vec)
        box = self.unsafe_box(spatial_map, q_vec) if spatial_map is not None else None

        # Deterministic software branch: 0=ALLOW, 1=ESCALATE_HUMAN_OR_S2, 2=BLOCK
        decision = torch.zeros(B, device=state_h.device, dtype=torch.long)
        ask_mask = (viol["epistemic_vacuity"] > ask_epistemic_thresh) | (risk["ooi_prob"] > 0.35)
        block_mask = viol["noul"] >= block_thresh
        decision = torch.where(ask_mask, torch.ones_like(decision), decision)
        decision = torch.where(block_mask, torch.full_like(decision, 2), decision)

        return {
            "decision_code": decision,             # 0=ALLOW, 1=ESCALATE, 2=BLOCK
            "violation_prob": viol["noul"],
            "epistemic_vacuity": viol["epistemic_vacuity"],
            "risk_idx": risk["choice_idx"],
            "risk_probs": risk["option_probs"],
            "severity": sev["score"],
            "offending_span": (span["start_idx"], span["end_idx"]),
            "no_span_prob": span["no_span_prob"],
            "unsafe_box": box["box"] if box is not None else None,
        }


@dataclass
class DualRateTickResult:
    route: str                # "EXECUTE_FAST" | "HOLD_AND_INTERRUPT_S2"
    reason: str               # "in_progress" | "subgoal_complete" | "unsafe" | "epistemic_abstain" | "stall_detected"
    s2_called: bool
    tick_index: int


class HierarchicalDualRateController:
    """Slow System-2 Planner (0.1-1 Hz) + Fast OmniJev Reactor (10-100 Hz) Controller.

    Prevents JEVal-style long-horizon error accumulation by monitoring closed-loop
    progress stalls, safety guardrails, and epistemic abstention interrupts.
    """

    def __init__(
        self,
        stall_patience: int = 5,
        min_progress_delta: float = 0.02,
        safety_min_prob: float = 0.70,
        epistemic_max_vacuity: float = 0.45,
        complete_min_prob: float = 0.85,
    ):
        self.stall_patience = stall_patience
        self.min_progress_delta = min_progress_delta
        self.safety_min_prob = safety_min_prob
        self.epistemic_max_vacuity = epistemic_max_vacuity
        self.complete_min_prob = complete_min_prob
        self.reset()

    def reset(self) -> None:
        self.tick_count = 0
        self.s2_interrupts = 0
        self.fast_ticks = 0
        self.best_progress = 0.0
        self.stall_counter = 0

    def step(
        self,
        is_safe_prob: float,
        progress_score: float,
        subgoal_complete_prob: float,
        epistemic_vacuity: float,
        wake_s2_flag: bool = False,
    ) -> DualRateTickResult:
        self.tick_count += 1

        if progress_score > self.best_progress + self.min_progress_delta:
            self.best_progress = progress_score
            self.stall_counter = 0
        else:
            self.stall_counter += 1

        if is_safe_prob < self.safety_min_prob:
            self.s2_interrupts += 1
            return DualRateTickResult("HOLD_AND_INTERRUPT_S2", "unsafe", True, self.tick_count)

        if wake_s2_flag or epistemic_vacuity > self.epistemic_max_vacuity:
            self.s2_interrupts += 1
            return DualRateTickResult("HOLD_AND_INTERRUPT_S2", "epistemic_abstain", True, self.tick_count)

        if subgoal_complete_prob >= self.complete_min_prob:
            self.s2_interrupts += 1
            self.best_progress = 0.0
            self.stall_counter = 0
            return DualRateTickResult("HOLD_AND_INTERRUPT_S2", "subgoal_complete", True, self.tick_count)

        if self.stall_counter >= self.stall_patience:
            self.s2_interrupts += 1
            self.stall_counter = 0
            return DualRateTickResult("HOLD_AND_INTERRUPT_S2", "stall_detected", True, self.tick_count)

        self.fast_ticks += 1
        return DualRateTickResult("EXECUTE_FAST", "in_progress", False, self.tick_count)

    @property
    def fast_execution_ratio(self) -> float:
        return self.fast_ticks / max(self.tick_count, 1)
