"""Axis D (Deep Implementation) — Calibration, Epistemic Abstention, OOD, Conformal S1->S2 Routing, & CDGs.

Implements:
  1. EvidentialDirichletLoss & FramingInvarianceRLCDLoss (proper scoring rules + OOI negatives + framing JS).
  2. PrototypeMahalanobisOOD (latent-space spherical/Mahalanobis prototype + free-energy OOD detector).
  3. SplitConformalCalibrator (finite-sample distribution-free coverage P(y in C(x)) >= 1 - alpha).
  4. CostAwareEscalationRouter (expected-utility gating: EXECUTE vs. DELIBERATE_DEEPER vs. ESCALATE_SYSTEM2).
  5. ComposableDecisionGraph (CDG): Shared-state typed DAG engine supporting Extract-Then-Decide
     dynamic inventory building & Coarse-to-Fine visual zoom passes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialDirichletLoss(nn.Module):
    """Proper-scoring Evidential Dirichlet Loss (Expected MSE / Brier + Annealed KL toward Uniform)."""

    def __init__(self, kl_weight: float = 0.05):
        super().__init__()
        self.kl_weight = kl_weight

    @staticmethod
    def _dirichlet_kl_to_uniform(alpha_tilde: torch.Tensor) -> torch.Tensor:
        K = alpha_tilde.size(-1)
        ones = torch.ones_like(alpha_tilde)
        S_alpha = alpha_tilde.sum(dim=-1, keepdim=True)
        S_ones = torch.full_like(S_alpha, float(K))

        lnB = torch.lgamma(S_alpha) - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
        lnB_uni = torch.lgamma(S_ones) - torch.lgamma(ones).sum(dim=-1, keepdim=True)
        dg_S = torch.digamma(S_alpha)
        dg_a = torch.digamma(alpha_tilde)
        kl = (lnB - lnB_uni + ((alpha_tilde - ones) * (dg_a - dg_S)).sum(dim=-1, keepdim=True)).squeeze(-1)
        return kl

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        anneal_coef: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """alpha: [B, K] Dirichlet concentrations (>1), targets: [B] class indices in 0..K-1."""
        K = alpha.size(-1)
        y = F.one_hot(targets, num_classes=K).to(alpha.dtype)
        S = alpha.sum(dim=-1, keepdim=True)
        p = alpha / S

        # Expected squared error (Brier + Dirichlet variance penalty) -> strictly proper scoring rule!
        err = (y - p).pow(2).sum(dim=-1)
        var = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
        mse_loss = (err + var).mean()

        # Remove true-class evidence before penalizing misleading evidence toward uniform Dir(1)
        alpha_tilde = y + (1.0 - y) * alpha
        kl_loss = self._dirichlet_kl_to_uniform(alpha_tilde).mean()

        total = mse_loss + (self.kl_weight * anneal_coef) * kl_loss
        return {"loss": total, "brier_evid": mse_loss, "kl_misleading": kl_loss}


class FramingInvarianceRLCDLoss(nn.Module):
    """RLCD Calibration Loss combining Evidential Proper Scoring + Jensen-Shannon Framing Consistency."""

    def __init__(self, kl_weight: float = 0.05, framing_weight: float = 0.5):
        super().__init__()
        self.evid = EvidentialDirichletLoss(kl_weight=kl_weight)
        self.framing_weight = framing_weight

    @staticmethod
    def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(1e-7)
        q = q.clamp_min(1e-7)
        m = 0.5 * (p + q)
        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        alpha_paraphrase: Optional[torch.Tensor] = None,
        flip_map: Optional[torch.Tensor] = None,
        anneal_coef: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        base = self.evid(alpha, targets, anneal_coef=anneal_coef)
        if alpha_paraphrase is not None:
            p1 = alpha / alpha.sum(dim=-1, keepdim=True)
            p2 = alpha_paraphrase / alpha_paraphrase.sum(dim=-1, keepdim=True)
            if flip_map is not None:
                p2 = p2[:, flip_map]
            frame_js = self.js_divergence(p1, p2).mean()
        else:
            frame_js = torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype)
        total = base["loss"] + self.framing_weight * frame_js
        return {**base, "framing_js": frame_js, "loss": total}


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) over predicted confidence vs empirical accuracy."""
    conf, preds = probs.max(dim=-1)
    acc = (preds == targets).to(torch.float32)
    conf = conf.to(torch.float32)
    ece = 0.0
    n = conf.numel()
    if n == 0:
        return 0.0
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        in_bin = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        cnt = in_bin.sum().item()
        if cnt > 0:
            ece += (cnt / n) * abs(acc[in_bin].mean().item() - conf[in_bin].mean().item())
    return float(ece)


class PrototypeMahalanobisOOD(nn.Module):
    """Lightweight latent-space OOD detector using class-agnostic diagonal Mahalanobis / prototype distances."""

    def __init__(self, d_model: int, num_prototypes: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        self.register_buffer("mean", torch.zeros(d_model))
        self.register_buffer("inv_var", torch.ones(d_model))
        self.register_buffer("prototypes", torch.zeros(num_prototypes, d_model))
        self.register_buffer("dist_scale", torch.tensor(1.0))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, in_domain_embs: torch.Tensor) -> None:
        """Fit latent mean, regularized precision, and prototypes on held-out in-domain embeddings [N, D]."""
        x = in_domain_embs.to(torch.float32)
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False).clamp_min(1e-3)
        self.mean.copy_(mean)
        self.inv_var.copy_(1.0 / var)

        # Deterministic farthest/strided prototype selection (zero external clustering dependency)
        N = x.size(0)
        k = min(self.num_prototypes, N)
        idx = torch.linspace(0, N - 1, k, device=x.device).long()
        protos = F.normalize(x[idx] - mean, p=2, dim=-1)
        if k < self.num_prototypes:
            pad = protos[-1:].expand(self.num_prototypes - k, -1)
            protos = torch.cat([protos, pad], dim=0)
        self.prototypes.copy_(protos)

        raw_d = self._raw_distance(x)
        # Set scale to 95th percentile of in-domain Mahalanobis distances
        q95 = torch.quantile(raw_d, 0.95).clamp_min(1e-3)
        self.dist_scale.copy_(q95)
        self.fitted.fill_(True)

    def _raw_distance(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb.to(torch.float32)
        diff = x - self.mean.unsqueeze(0)
        maha = (diff.pow(2) * self.inv_var.unsqueeze(0)).mean(dim=-1).sqrt()  # normalized RMS Mahalanobis
        x_norm = F.normalize(diff, p=2, dim=-1)
        cos_sim = torch.matmul(x_norm, self.prototypes.T).max(dim=-1).values
        cos_dist = (1.0 - cos_sim).clamp_min(0.0)
        return 0.7 * maha + 0.3 * cos_dist

    def forward(self, emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not bool(self.fitted.item()):
            zeros = torch.zeros(emb.size(0), device=emb.device, dtype=emb.dtype)
            return {"ood_score": zeros, "raw_distance": zeros}
        raw = self._raw_distance(emb)
        # Sigmoid centered around 1.5x in-domain 95th percentile -> clean [0, 1] OOD score
        score = torch.sigmoid(3.0 * (raw / self.dist_scale.clamp_min(1e-4) - 1.35)).to(emb.dtype)
        return {"ood_score": score, "raw_distance": raw.to(emb.dtype)}


class SplitConformalCalibrator:
    """Distribution-free Split Conformal Prediction wrapper guaranteeing P(y in C(x)) >= 1 - alpha."""

    def __init__(self, alpha: float = 0.10, lambda_epi: float = 0.5, lambda_ood: float = 0.5):
        self.alpha = alpha
        self.lambda_epi = lambda_epi
        self.lambda_ood = lambda_ood
        self.q_hat: float = 1.0
        self.calibrated: bool = False

    def nonconformity_scores(
        self,
        probs: torch.Tensor,
        epistemic_vacuity: Optional[torch.Tensor] = None,
        ood_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-class nonconformity score r(x, c) = 1 - p_c + lam_e * u_epi + lam_o * s_ood."""
        r = 1.0 - probs
        if epistemic_vacuity is not None:
            r = r + self.lambda_epi * epistemic_vacuity.unsqueeze(-1)
        if ood_score is not None:
            r = r + self.lambda_ood * ood_score.unsqueeze(-1)
        return r

    def calibrate(
        self,
        cal_probs: torch.Tensor,
        cal_targets: torch.Tensor,
        cal_epistemic: Optional[torch.Tensor] = None,
        cal_ood: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute finite-sample conformal quantile q_hat on held-out calibration set."""
        scores_all = self.nonconformity_scores(cal_probs, cal_epistemic, cal_ood)
        batch_idx = torch.arange(cal_targets.size(0), device=cal_targets.device)
        true_scores = scores_all[batch_idx, cal_targets].detach().cpu()

        n = true_scores.numel()
        level = min(math.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        sorted_scores, _ = torch.sort(true_scores)
        q_idx = min(max(int(math.ceil(level * n)) - 1, 0), n - 1)
        self.q_hat = float(sorted_scores[q_idx].item())
        self.calibrated = True
        return self.q_hat

    def predict_sets(
        self,
        probs: torch.Tensor,
        epistemic_vacuity: Optional[torch.Tensor] = None,
        ood_score: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return certified prediction set mask C(x) [B, C] and set sizes |C(x)| [B]."""
        r = self.nonconformity_scores(probs, epistemic_vacuity, ood_score)
        in_set = r <= self.q_hat
        return {
            "prediction_set": in_set,                 # [B, C] bool
            "set_size": in_set.sum(dim=-1),           # [B] int: 0=OOI/OOD, 1=certified single, >1=ambiguous
            "nonconformity": r,
        }


class CostAwareEscalationRouter:
    """Optimal expected-utility System 1 -> System 1.5 -> System 2 routing gate."""

    def __init__(
        self,
        error_cost: float = 10.0,
        s2_cost: float = 0.5,
        s2_error_rate: float = 0.03,
        ooi_threshold: float = 0.35,
        epistemic_threshold: float = 0.45,
        ood_threshold: float = 0.65,
        epi_risk_weight: float = 0.5,
    ):
        self.error_cost = error_cost
        self.s2_cost = s2_cost
        self.s2_error_rate = s2_error_rate
        self.ooi_threshold = ooi_threshold
        self.epistemic_threshold = epistemic_threshold
        self.ood_threshold = ood_threshold
        self.epi_risk_weight = epi_risk_weight

    def route(
        self,
        option_probs: torch.Tensor,
        epistemic_vacuity: torch.Tensor,
        ooi_prob: Optional[torch.Tensor] = None,
        ood_score: Optional[torch.Tensor] = None,
        conformal_set_size: Optional[torch.Tensor] = None,
        current_depth: int = 0,
        max_depth: int = 6,
    ) -> Dict[str, object]:
        """Decide per sample: 0=EXECUTE_S1, 1=DELIBERATE_S15, 2=ESCALATE_S2."""
        B = option_probs.size(0)
        device = option_probs.device
        max_p = option_probs.max(dim=-1).values
        est_s1_err = (1.0 - max_p + self.epi_risk_weight * epistemic_vacuity).clamp(0.0, 1.0)

        # Expected value of escalating vs executing immediately
        expected_gain = (est_s1_err - self.s2_error_rate) * self.error_cost
        cost_triggers_s2 = expected_gain > self.s2_cost

        epi_triggers = epistemic_vacuity > self.epistemic_threshold
        ooi_triggers = (ooi_prob > self.ooi_threshold) if ooi_prob is not None else torch.zeros(B, dtype=torch.bool, device=device)
        ood_triggers = (ood_score > self.ood_threshold) if ood_score is not None else torch.zeros(B, dtype=torch.bool, device=device)

        conf_empty = (conformal_set_size == 0) if conformal_set_size is not None else torch.zeros(B, dtype=torch.bool, device=device)
        conf_ambig = (conformal_set_size > 1) if conformal_set_size is not None else torch.zeros(B, dtype=torch.bool, device=device)

        # Hard OOI / OOD / empty conformal set -> straight to System 2
        hard_s2 = ooi_triggers | ood_triggers | conf_empty
        # Borderline ambiguity / cost trigger when more latent depth is available -> try System 1.5 first!
        soft_uncertain = cost_triggers_s2 | epi_triggers | conf_ambig

        action = torch.zeros(B, dtype=torch.long, device=device)  # 0 = EXECUTE_S1
        if current_depth < max_depth:
            action = torch.where(soft_uncertain & ~hard_s2, torch.ones_like(action), action)  # 1 = DELIBERATE_S15
            action = torch.where(hard_s2, torch.full_like(action, 2), action)                 # 2 = ESCALATE_S2
        else:
            action = torch.where(soft_uncertain | hard_s2, torch.full_like(action, 2), action)

        return {
            "route_code": action,           # 0=EXECUTE, 1=DELIBERATE_S15, 2=ESCALATE_S2
            "estimated_s1_error": est_s1_err,
            "expected_escalation_gain": expected_gain,
            "hard_ood_or_ooi": hard_s2,
        }


@dataclass
class CDGNode:
    """Single typed node inside a Composable Decision Graph (CDG)."""

    name: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    next_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None


class ComposableDecisionGraph:
    """Shared-State Typed DAG Runtime for Multi-Pass Composable Decisions.

    Supports:
      - Extract-Then-Decide (Pass-1 Span/Box builds Pass-2 dynamic option inventory)
      - Coarse-to-Fine Spatial Zoom (Pass-1 Box crops high-res patch -> Pass-2 exact Point)
      - Selective System-2 Fallback Nodes without re-encoding shared state H
    """

    def __init__(self, entry_node: str, max_hops: int = 8):
        self.entry_node = entry_node
        self.max_hops = max_hops
        self.nodes: Dict[str, CDGNode] = {}

    def add_node(self, node: CDGNode) -> "ComposableDecisionGraph":
        self.nodes[node.name] = node
        return self

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the typed decision graph over shared cached state until terminal node."""
        state = dict(context)
        trace: List[str] = []
        current: Optional[str] = self.entry_node

        for _ in range(self.max_hops):
            if current is None:
                break
            if current not in self.nodes:
                raise KeyError(f"CDG node '{current}' not registered.")
            node = self.nodes[current]
            trace.append(current)
            out = node.fn(state)
            state[current] = out
            current = node.next_fn(state) if node.next_fn is not None else None

        state["_cdg_trace"] = trace
        return state
