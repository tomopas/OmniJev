"""Axis A — Complete Suite of 12 Typed Single-Pass Decision & Grounding Primitives.

Includes:
  - QueryCrossPooler (shared query extractor over deliberated state H)
  - A1: NoulHead (Evidential Beta/Dirichlet calibrated boolean + epistemic vacuity)
  - A2: OrderInvariantChoiceHead (Exact O(1) permutation invariance + OOI sink + Evidential Dirichlet)
  - A3: OrdinalScoreHead (Monotonic cumulative thresholds + continuous expected score)
  - A4: BiaffineSpanHead (Zero-hallucination extractive [start, end] + null-span sink + top-k NMS)
  - A5: PointHead (2D spatial softmax heatmap + SimCC 1D X/Y distributions + multimodal peak extractor)
  - A6: BoxHead (4-side DFL probability distributions + calibrated confidence)
  - A7: PolygonHead (Circular-attention vertex queries + deformable cross-attention over F_2D)
  - A8: MaskHead (Mask2Former-style dynamic mask kernel dot-product over upsampled F_2D)
  - A9: OrderInvariantRankHead (Exact permutation-equivariant Plackett-Luce utilities + pairwise margins)
  - A10: BiaffineGraphHead (Parallel N x N x (R+1) typed adjacency matrix)
  - A11: CorrelatedSetHead (Marginal + low-rank Ising 2-step mean-field + cardinality head)
  - A12: ScalarDistributionHead (Mixture Density Network + strictly monotonic Pinball quantiles)
"""

from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OmniJevConfig


def _evidential_dirichlet_stats(evidence: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute Dirichlet alpha, expected probs, epistemic vacuity, and aleatoric entropy."""
    alpha = evidence + 1.0                                         # [..., K]
    S = alpha.sum(dim=-1, keepdim=True)                            # [..., 1]
    probs = alpha / S                                              # [..., K]
    K = alpha.size(-1)
    epistemic_vacuity = (K / S).squeeze(-1)                        # [...] in (0, 1]
    # Expected Shannon entropy E_{p~Dir(alpha)}[H(p)] via digamma
    aleatoric_entropy = (probs * (torch.digamma(S + 1.0) - torch.digamma(alpha + 1.0))).sum(dim=-1)
    confidence = 1.0 - epistemic_vacuity
    return {
        "alpha": alpha,
        "strength": S.squeeze(-1),
        "probs": probs,
        "epistemic_vacuity": epistemic_vacuity,
        "aleatoric_entropy": aleatoric_entropy.clamp_min(0.0),
        "confidence": confidence.clamp(0.0, 1.0),
    }


class QueryCrossPooler(nn.Module):
    """Cross-attends a task/question query vector over the shared backbone state H."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, state_h: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """state_h: [B, L, D], query: [B, D] or [B, Q, D] -> same shape as query."""
        squeeze = query.dim() == 2
        q = query.unsqueeze(1) if squeeze else query
        attn_out, _ = self.attn(self.norm1(q), state_h, state_h, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.norm2(q))
        return q.squeeze(1) if squeeze else q


class PositionFreeSetLayer(nn.Module):
    """Strictly permutation-equivariant self-attention block (zero positional encodings)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


class NoulHead(nn.Module):
    """A1: Evidential Beta/Dirichlet calibrated boolean primitive (yes/no probability + epistemic vacuity)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),  # [evidence_no, evidence_yes]
        )

    def forward(self, q_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = F.softplus(self.mlp(q_vec))  # [B, 2]
        stats = _evidential_dirichlet_stats(evidence)
        stats["noul"] = stats["probs"][..., 1]  # P(Yes) in (0, 1)
        return stats


class OrderInvariantChoiceHead(nn.Module):
    """A2: Exact O(1) permutation-invariant Choice primitive with Out-of-Inventory (OOI) sink & Evidential Dirichlet."""

    def __init__(self, d_model: int, n_heads: int, set_layers: int = 2):
        super().__init__()
        self.option_cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.opt_norm = nn.LayerNorm(d_model)
        self.set_blocks = nn.ModuleList([PositionFreeSetLayer(d_model, n_heads) for _ in range(set_layers)])
        self.ooi_generator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.bilinear = nn.Linear(d_model, d_model, bias=False)
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self,
        state_h: torch.Tensor,
        q_vec: torch.Tensor,
        option_embs: torch.Tensor,
        option_mask: Optional[torch.Tensor] = None,
        include_ooi_sink: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_h: [B, L, D] shared backbone state
            q_vec: [B, D] query summary
            option_embs: [B, C, D] individually encoded candidate option vectors (local RoPE already reset)
            option_mask: [B, C] bool True where valid, False for padded options
        """
        B, C, D = option_embs.shape
        # Each option cross-attends independently to shared state H (invariant to option order!)
        opt_ctx, _ = self.option_cross(self.opt_norm(option_embs), state_h, state_h, need_weights=False)
        opts = option_embs + opt_ctx

        if include_ooi_sink:
            ooi_vec = self.ooi_generator(q_vec).unsqueeze(1)  # [B, 1, D]
            opts = torch.cat([opts, ooi_vec], dim=1)          # [B, C+1, D]
            if option_mask is not None:
                ooi_valid = torch.ones(B, 1, device=option_mask.device, dtype=torch.bool)
                full_mask = torch.cat([option_mask, ooi_valid], dim=1)
            else:
                full_mask = None
        else:
            full_mask = option_mask

        key_pad_mask = (~full_mask) if full_mask is not None else None
        for blk in self.set_blocks:
            opts = blk(opts, key_padding_mask=key_pad_mask)

        # Compatibility score between query q_vec and each option
        q_exp = q_vec.unsqueeze(1).expand_as(opts)
        bilin = (self.bilinear(q_exp) * opts).sum(dim=-1) * self.scale
        mlp_term = self.score_mlp(torch.cat([q_exp, opts, q_exp * opts], dim=-1)).squeeze(-1)
        raw_scores = bilin + mlp_term
        evidence = F.softplus(raw_scores)

        if full_mask is not None:
            evidence = evidence.masked_fill(~full_mask, 0.0)

        stats = _evidential_dirichlet_stats(evidence)
        if include_ooi_sink:
            opt_probs = stats["probs"][:, :C]
            ooi_prob = stats["probs"][:, C]
        else:
            opt_probs = stats["probs"]
            ooi_prob = torch.zeros(B, device=q_vec.device, dtype=q_vec.dtype)

        # Choice winner over supplied inventory
        masked_opt_probs = opt_probs.masked_fill(~option_mask, -1.0) if option_mask is not None else opt_probs
        choice_idx = masked_opt_probs.argmax(dim=-1)

        return {
            **stats,
            "option_probs": opt_probs,
            "ooi_prob": ooi_prob,
            "choice_idx": choice_idx,
            "raw_scores": raw_scores,
        }


class OrdinalScoreHead(nn.Module):
    """A3: Monotonic Ordinal Score primitive (CORAL cumulative link with strictly increasing thresholds)."""

    def __init__(self, d_model: int, num_levels: int = 5):
        super().__init__()
        assert num_levels >= 2
        self.num_levels = num_levels
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.thresh_base = nn.Parameter(torch.tensor(-1.0))
        self.thresh_deltas = nn.Parameter(torch.zeros(num_levels - 2)) if num_levels > 2 else None

    def get_thresholds(self) -> torch.Tensor:
        if self.thresh_deltas is None:
            return self.thresh_base.view(1)
        increments = F.softplus(self.thresh_deltas) + 1e-4
        rest = self.thresh_base + torch.cumsum(increments, dim=0)
        return torch.cat([self.thresh_base.view(1), rest], dim=0)  # [M-1] strictly increasing

    def forward(self, q_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.latent_proj(q_vec)                    # [B, 1]
        thresholds = self.get_thresholds().view(1, -1)  # [1, M-1]
        cum_probs = torch.sigmoid(z - thresholds)      # P(y > m) for m=0..M-2, monotonically decreasing!

        ones = torch.ones_like(z)
        zeros = torch.zeros_like(z)
        upper = torch.cat([ones, cum_probs], dim=-1)   # [B, M]
        lower = torch.cat([cum_probs, zeros], dim=-1)  # [B, M]
        probs = (upper - lower).clamp_min(1e-7)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        levels = torch.linspace(0.0, 1.0, self.num_levels, device=q_vec.device, dtype=q_vec.dtype)
        expected_score = (probs * levels.view(1, -1)).sum(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-7).log()).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(self.num_levels)

        return {
            "score": expected_score,          # [B] in [0, 1]
            "level_idx": probs.argmax(dim=-1),
            "probs": probs,                   # [B, M]
            "cum_probs": cum_probs,           # [B, M-1]
            "thresholds": thresholds.squeeze(0),
            "confidence": confidence.clamp(0.0, 1.0),
        }


class BiaffineSpanHead(nn.Module):
    """A4: Zero-hallucination Biaffine Extractive Span primitive with upper-triangular valid mask + null sink."""

    def __init__(self, d_model: int, proj_dim: Optional[int] = None):
        super().__init__()
        dp = proj_dim or max(d_model // 2, 32)
        self.start_mlp = nn.Sequential(nn.Linear(2 * d_model, dp), nn.GELU(), nn.Linear(dp, dp))
        self.end_mlp = nn.Sequential(nn.Linear(2 * d_model, dp), nn.GELU(), nn.Linear(dp, dp))
        self.W_biaffine = nn.Parameter(torch.empty(dp, dp))
        nn.init.xavier_uniform_(self.W_biaffine)
        self.u_start = nn.Linear(dp, 1, bias=False)
        self.u_end = nn.Linear(dp, 1, bias=False)
        self.null_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.scale = 1.0 / math.sqrt(dp)

    def forward(
        self,
        state_h: torch.Tensor,
        q_vec: torch.Tensor,
        max_span_len: Optional[int] = None,
        top_k: int = 3,
    ) -> Dict[str, torch.Tensor]:
        B, L, D = state_h.shape
        q_exp = q_vec.unsqueeze(1).expand(B, L, D)
        hs = self.start_mlp(torch.cat([state_h, q_exp], dim=-1))  # [B, L, dp]
        he = self.end_mlp(torch.cat([state_h, q_exp], dim=-1))    # [B, L, dp]

        biaff = torch.einsum("bid,de,bje->bij", hs, self.W_biaffine, he) * self.scale
        scores = biaff + self.u_start(hs) + self.u_end(he).transpose(1, 2)  # [B, L, L]

        # Enforce valid span indices 0 <= i <= j < L (and optional max length)
        idx = torch.arange(L, device=state_h.device)
        valid = idx.unsqueeze(1) <= idx.unsqueeze(0)  # [L, L]
        if max_span_len is not None:
            valid = valid & ((idx.unsqueeze(0) - idx.unsqueeze(1)) < max_span_len)

        masked_scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        flat_spans = masked_scores.reshape(B, L * L)
        null_logit = self.null_head(q_vec)            # [B, 1]
        joint_logits = torch.cat([flat_spans, null_logit], dim=-1)  # [B, L*L + 1]
        joint_probs = F.softmax(joint_logits, dim=-1)

        span_probs = joint_probs[:, : L * L].reshape(B, L, L)
        no_span_prob = joint_probs[:, -1]

        # Best span (excluding null) and top-k spans
        k = min(top_k, L * L)
        top_vals, top_flat = joint_probs[:, : L * L].topk(k, dim=-1)
        top_starts = top_flat // L
        top_ends = top_flat % L

        return {
            "start_idx": top_starts[:, 0],          # [B]
            "end_idx": top_ends[:, 0],              # [B]
            "best_span_prob": top_vals[:, 0],       # [B]
            "no_span_prob": no_span_prob,           # [B]
            "top_k_spans": torch.stack([top_starts, top_ends], dim=-1),  # [B, k, 2]
            "top_k_probs": top_vals,                # [B, k]
            "span_probs": span_probs,               # [B, L, L]
            "joint_logits": joint_logits,           # [B, L*L + 1]
        }


class PointHead(nn.Module):
    """A5: Spatially Equivariant Sub-Pixel 2D Point grounding via 2D Spatial Softmax Expectation + SimCC Residual."""

    def __init__(self, d_model: int, simcc_bins: int = 64):
        super().__init__()
        self.simcc_bins = simcc_bins
        self.film = nn.Linear(d_model, 2 * d_model)
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, 1, kernel_size=1),
        )
        # Sub-pixel SimCC 1D logits & bounded local offset refinement
        self.x_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, simcc_bins))
        self.y_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, simcc_bins))
        self.subpix_offset = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
        nn.init.zeros_(self.subpix_offset[-1].weight)
        nn.init.zeros_(self.subpix_offset[-1].bias)

    def forward(
        self,
        spatial_map: torch.Tensor,
        q_vec: torch.Tensor,
        top_k_peaks: int = 3,
    ) -> Dict[str, torch.Tensor]:
        B, D, H, W = spatial_map.shape
        gamma, beta = self.film(q_vec).chunk(2, dim=-1)
        mod = (1.0 + gamma.view(B, D, 1, 1)) * spatial_map + beta.view(B, D, 1, 1)

        # 2D Spatial Softmax Heatmap (strictly spatially equivariant!)
        hm_logits = self.heatmap_conv(mod).squeeze(1)             # [B, H, W]
        hm_probs = F.softmax(hm_logits.reshape(B, H * W), dim=-1).reshape(B, H, W)

        # Differentiable 2D Spatial Softmax Coordinate Expectation (exact metric x, y!)
        xs = ((torch.arange(W, device=q_vec.device, dtype=q_vec.dtype) + 0.5) / W).view(1, 1, W)
        ys = ((torch.arange(H, device=q_vec.device, dtype=q_vec.dtype) + 0.5) / H).view(1, H, 1)
        x_hm = (hm_probs * xs).sum(dim=(1, 2))                    # [B] in (0, 1)
        y_hm = (hm_probs * ys).sum(dim=(1, 2))                    # [B] in (0, 1)

        # Attention-pooled feature around heatmap peak for sub-pixel offset & SimCC 1D distribution
        pooled = torch.einsum("bhw,bdhw->bd", hm_probs, mod)      # [B, D]
        delta = torch.tanh(self.subpix_offset(pooled))            # [B, 2] in (-1, 1)
        x_coord = (x_hm + delta[:, 0] / max(W, 1)).clamp(0.0, 1.0)
        y_coord = (y_hm + delta[:, 1] / max(H, 1)).clamp(0.0, 1.0)

        x_logits = self.x_head(pooled)
        y_logits = self.y_head(pooled)

        # Top-k discrete spatial peaks from 2D heatmap (prevents averaging two distinct buttons)
        k = min(top_k_peaks, H * W)
        peak_vals, peak_flat = hm_probs.reshape(B, H * W).topk(k, dim=-1)
        peak_y = ((peak_flat // W).to(q_vec.dtype) + 0.5) / H
        peak_x = ((peak_flat % W).to(q_vec.dtype) + 0.5) / W
        peaks = torch.stack([peak_x, peak_y], dim=-1)             # [B, k, 2]

        hm_entropy = -(hm_probs.reshape(B, -1) * hm_probs.reshape(B, -1).clamp_min(1e-7).log()).sum(dim=-1)
        confidence = (1.0 - hm_entropy / math.log(max(H * W, 2))).clamp(0.0, 1.0)

        return {
            "point": torch.stack([x_coord, y_coord], dim=-1),     # [B, 2] in [0, 1]
            "peaks": peaks,                                       # [B, k, 2]
            "peak_probs": peak_vals,                              # [B, k]
            "heatmap": hm_probs,                                  # [B, H, W]
            "heatmap_logits": hm_logits,                          # [B, H, W]
            "x_logits": x_logits,
            "y_logits": y_logits,
            "confidence": confidence,
        }


class BoxHead(nn.Module):
    """A6: Spatially Equivariant Center-Expectation + DFL Size Bounding Box returning (top, left, bottom, right)."""

    def __init__(self, d_model: int, dfl_bins: int = 32):
        super().__init__()
        self.dfl_bins = dfl_bins
        self.film = nn.Linear(d_model, 2 * d_model)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, 1, kernel_size=1),
        )
        # DFL distribution over half-height and half-width + fine center offset (4 x dfl_bins)
        self.side_heads = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4 * dfl_bins),
        )
        self.conf_head = nn.Sequential(nn.Linear(2 * d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(self, spatial_map: torch.Tensor, q_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D, H, W = spatial_map.shape
        gamma, beta = self.film(q_vec).chunk(2, dim=-1)
        mod = (1.0 + gamma.view(B, D, 1, 1)) * spatial_map + beta.view(B, D, 1, 1)
        weights = F.softmax(self.spatial_attn(mod).reshape(B, H * W), dim=-1).reshape(B, H, W)

        # Equivariant 2D spatial expectation for box center (cx, cy)
        xs = ((torch.arange(W, device=q_vec.device, dtype=q_vec.dtype) + 0.5) / W).view(1, 1, W)
        ys = ((torch.arange(H, device=q_vec.device, dtype=q_vec.dtype) + 0.5) / H).view(1, H, 1)
        cx = (weights * xs).sum(dim=(1, 2))
        cy = (weights * ys).sum(dim=(1, 2))

        pooled = torch.einsum("bhw,bdhw->bd", weights, mod)
        feat = torch.cat([pooled, q_vec], dim=-1)

        dfl_logits = self.side_heads(feat).view(B, 4, self.dfl_bins)
        dfl_probs = F.softmax(dfl_logits, dim=-1)
        bins = (torch.arange(self.dfl_bins, device=q_vec.device, dtype=q_vec.dtype) + 0.5) / self.dfl_bins
        # Predict 4 distances from center: (dt, dl, db, dr) in [0, 0.5]
        dists = (dfl_probs * bins.view(1, 1, -1)).sum(dim=-1) * 0.5  # [B, 4]

        top = (cy - dists[:, 0]).clamp(0.0, 1.0)
        left = (cx - dists[:, 1]).clamp(0.0, 1.0)
        bottom = (cy + dists[:, 2]).clamp(0.0, 1.0)
        right = (cx + dists[:, 3]).clamp(0.0, 1.0)

        # Guarantee valid ordered box: top <= bottom, left <= right
        box = torch.stack([
            torch.minimum(top, bottom),
            torch.minimum(left, right),
            torch.maximum(top, bottom),
            torch.maximum(left, right),
        ], dim=-1)
        confidence = torch.sigmoid(self.conf_head(feat)).squeeze(-1)

        return {
            "box": box,                  # [B, 4] = [top, left, bottom, right]
            "dfl_logits": dfl_logits,    # [B, 4, dfl_bins]
            "confidence": confidence,
        }


class PolygonHead(nn.Module):
    """A7: Parallel Circular-Attention Polygon primitive outputting ordered (x_v, y_v) + vertex visibility."""

    def __init__(self, d_model: int, n_heads: int, max_vertices: int = 8):
        super().__init__()
        self.max_vertices = max_vertices
        self.vertex_queries = nn.Parameter(torch.randn(1, max_vertices, d_model) * 0.02)
        self.circ_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.coord_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
        self.vis_head = nn.Linear(d_model, 1)

    def forward(self, spatial_map: torch.Tensor, q_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D, H, W = spatial_map.shape
        v_q = self.vertex_queries.expand(B, -1, -1) + q_vec.unsqueeze(1)  # [B, V, D]

        # Circular neighbor attention bias encourages contiguous contour ordering
        circ_out, _ = self.circ_attn(self.norm1(v_q), self.norm1(v_q), self.norm1(v_q), need_weights=False)
        v_q = v_q + circ_out

        spatial_tokens = spatial_map.flatten(2).transpose(1, 2)  # [B, H*W, D]
        cross_out, _ = self.cross_attn(self.norm2(v_q), spatial_tokens, spatial_tokens, need_weights=False)
        v_q = v_q + cross_out

        vertices = torch.sigmoid(self.coord_head(v_q))          # [B, V, 2] in (0, 1)
        vertex_probs = torch.sigmoid(self.vis_head(v_q)).squeeze(-1)  # [B, V]
        return {
            "vertices": vertices,
            "vertex_probs": vertex_probs,
        }


class MaskHead(nn.Module):
    """A8: Mask2Former-style single-pass dynamic mask kernel dot-product over upsampled 2D features."""

    def __init__(self, d_model: int, upsample_factor: int = 2):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.kernel_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.mask_feat = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, spatial_map: torch.Tensor, q_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.mask_feat(spatial_map)
        if self.upsample_factor > 1:
            feat = F.interpolate(feat, scale_factor=self.upsample_factor, mode="bilinear", align_corners=False)
        w = self.kernel_mlp(q_vec)  # [B, D]
        logits = torch.einsum("bd,bdhw->bhw", w, feat) * self.scale
        mask = torch.sigmoid(logits)
        return {"mask": mask, "mask_logits": logits}


class OrderInvariantRankHead(nn.Module):
    """A9: Strictly permutation-equivariant Plackett-Luce / Bradley-Terry Rank primitive."""

    def __init__(self, d_model: int, n_heads: int, set_layers: int = 2):
        super().__init__()
        self.q_fuse = nn.Linear(2 * d_model, d_model)
        self.set_blocks = nn.ModuleList([PositionFreeSetLayer(d_model, n_heads) for _ in range(set_layers)])
        self.utility_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.pair_proj = nn.Linear(d_model, d_model // 2, bias=False)

    def forward(
        self,
        q_vec: torch.Tensor,
        item_embs: torch.Tensor,
        item_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, D = item_embs.shape
        x = self.q_fuse(torch.cat([item_embs, q_vec.unsqueeze(1).expand(B, N, D)], dim=-1))
        key_pad = (~item_mask) if item_mask is not None else None
        for blk in self.set_blocks:
            x = blk(x, key_padding_mask=key_pad)

        utilities = self.utility_head(x).squeeze(-1)  # [B, N]
        if item_mask is not None:
            utilities = utilities.masked_fill(~item_mask, float("-inf"))

        # Skew-symmetric pairwise preference adjustment: delta_ij = -delta_ji (permutation equivariant)
        p = self.pair_proj(x)  # [B, N, d/2]
        # antisymmetric difference norm / projection via utilities
        diff = utilities.unsqueeze(2) - utilities.unsqueeze(1)  # [B, N, N]
        pairwise_probs = torch.sigmoid(diff)                    # P(i > j)
        pl_top1_probs = F.softmax(utilities, dim=-1)            # Plackett-Luce top-1 distribution
        order = torch.argsort(utilities, dim=-1, descending=True)

        return {
            "utilities": utilities,           # [B, N]
            "ranking": order,                 # [B, N] sorted indices descending
            "pl_probs": pl_top1_probs,        # [B, N]
            "pairwise_probs": pairwise_probs, # [B, N, N]
        }


class BiaffineGraphHead(nn.Module):
    """A10: Parallel N x N typed relation / edge adjacency tensor via Biaffine scoring."""

    def __init__(self, d_model: int, num_relations: int = 8, proj_dim: Optional[int] = None):
        super().__init__()
        self.num_relations = num_relations
        dg = proj_dim or max(d_model // 2, 32)
        self.head_mlp = nn.Sequential(nn.Linear(2 * d_model, dg), nn.GELU(), nn.Linear(dg, dg))
        self.tail_mlp = nn.Sequential(nn.Linear(2 * d_model, dg), nn.GELU(), nn.Linear(dg, dg))
        # +1 relation for NO_EDGE (index 0)
        self.W_rel = nn.Parameter(torch.empty(num_relations + 1, dg, dg))
        nn.init.xavier_uniform_(self.W_rel)
        self.u_rel = nn.Linear(2 * dg, num_relations + 1)
        self.scale = 1.0 / math.sqrt(dg)

    def forward(self, node_embs: torch.Tensor, q_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = node_embs.shape
        q_exp = q_vec.unsqueeze(1).expand(B, N, D)
        hh = self.head_mlp(torch.cat([node_embs, q_exp], dim=-1))  # [B, N, dg]
        ht = self.tail_mlp(torch.cat([node_embs, q_exp], dim=-1))  # [B, N, dg]

        biaff = torch.einsum("bid,rde,bje->bijr", hh, self.W_rel, ht) * self.scale
        pair_cat = torch.cat(
            [hh.unsqueeze(2).expand(B, N, N, -1), ht.unsqueeze(1).expand(B, N, N, -1)],
            dim=-1,
        )
        logits = biaff + self.u_rel(pair_cat)                      # [B, N, N, R+1]
        probs = F.softmax(logits, dim=-1)
        edge_exists_prob = 1.0 - probs[..., 0]                     # P(any edge)
        return {
            "edge_logits": logits,
            "edge_probs": probs,
            "edge_exists_prob": edge_exists_prob,
            "pred_relations": probs.argmax(dim=-1),                # 0 = NO_EDGE, 1..R = typed edge
        }


class CorrelatedSetHead(nn.Module):
    """A11: Multi-label Set primitive with low-rank factorized Ising pairwise interaction + cardinality head."""

    def __init__(self, d_model: int, ising_rank: int = 8, max_cardinality: int = 16):
        super().__init__()
        self.max_cardinality = max_cardinality
        self.item_proj = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU())
        self.unary_head = nn.Linear(d_model, 1)
        self.ising_u = nn.Linear(d_model, ising_rank, bias=False)
        self.card_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, max_cardinality + 1))

    def forward(
        self,
        q_vec: torch.Tensor,
        item_embs: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        B, N, D = item_embs.shape
        feat = self.item_proj(torch.cat([item_embs, q_vec.unsqueeze(1).expand(B, N, D)], dim=-1))
        unary = self.unary_head(feat).squeeze(-1)             # [B, N]

        # Symmetric zero-diagonal low-rank coupling matrix J_ij capturing synergies/exclusions
        U = self.ising_u(feat) / math.sqrt(self.ising_u.out_features)  # [B, N, r]
        J = torch.bmm(U, U.transpose(1, 2))                   # [B, N, N]
        eye = torch.eye(N, device=item_embs.device, dtype=torch.bool).unsqueeze(0)
        J = J.masked_fill(eye, 0.0)

        # 2-step parallel mean-field fixed-point refinement (O(1) pass, permutation equivariant)
        mu = torch.sigmoid(unary)
        for _ in range(2):
            eff_logits = unary + torch.bmm(J, mu.unsqueeze(-1)).squeeze(-1)
            mu = torch.sigmoid(eff_logits)

        pooled = feat.mean(dim=1)
        card_logits = self.card_head(pooled)                  # [B, max_card + 1]
        card_probs = F.softmax(card_logits, dim=-1)

        return {
            "probs": mu,                                      # [B, N]
            "selected_mask": mu >= threshold,                 # [B, N] bool
            "cardinality_probs": card_probs,
            "expected_cardinality": (card_probs * torch.arange(self.max_cardinality + 1, device=q_vec.device)).sum(-1),
            "interaction_matrix": J,
        }


class ScalarDistributionHead(nn.Module):
    """A12: Calibrated continuous Scalar & Distribution primitive (MDN + strictly monotonic Pinball quantiles)."""

    def __init__(self, d_model: int, n_components: int = 4, quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.n_components = n_components
        self.register_buffer("quantile_levels", torch.tensor(quantiles, dtype=torch.float32))
        self.trunk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.mdn_head = nn.Linear(d_model, 3 * n_components)  # pi, mu, log_sigma
        self.q_head = nn.Linear(d_model, len(quantiles))      # q_0 + positive increments

    def forward(self, q_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(q_vec)
        raw_pi, mu, raw_sigma = self.mdn_head(h).chunk(3, dim=-1)
        pi = F.softmax(raw_pi, dim=-1)                        # [B, K_m]
        sigma = F.softplus(raw_sigma) + 1e-3                  # [B, K_m]

        # Mixture moments: E[Y] and Var(Y) = E[sigma^2] + E[(mu - mean)^2]
        mean = (pi * mu).sum(dim=-1)
        var = (pi * (sigma.pow(2) + (mu - mean.unsqueeze(-1)).pow(2))).sum(dim=-1)
        std = var.clamp_min(1e-6).sqrt()

        # Strictly monotonic quantiles: q_0, q_0 + softplus(d1), ...
        q_raw = self.q_head(h)
        if q_raw.size(-1) > 1:
            q_base = q_raw[:, :1]
            q_inc = F.softplus(q_raw[:, 1:]) + 1e-4
            quantiles = torch.cat([q_base, q_base + torch.cumsum(q_inc, dim=-1)], dim=-1)
        else:
            quantiles = q_raw

        return {
            "mean": mean,
            "std": std,
            "mixture_weights": pi,
            "mixture_means": mu,
            "mixture_stds": sigma,
            "quantiles": quantiles,               # [B, Q] guaranteed non-decreasing!
            "quantile_levels": self.quantile_levels,
        }


class AxisAPrimitiveSuite(nn.Module):
    """Bundles all 12 Axis-A typed primitive heads sharing a single QueryCrossPooler."""

    def __init__(self, cfg: OmniJevConfig):
        super().__init__()
        self.pooler = QueryCrossPooler(cfg.d_model, cfg.n_heads)
        self.noul = NoulHead(cfg.d_model)
        self.choice = OrderInvariantChoiceHead(cfg.d_model, cfg.n_heads, cfg.set_attn_layers)
        self.score = OrdinalScoreHead(cfg.d_model, cfg.max_score_levels)
        self.span = BiaffineSpanHead(cfg.d_model)
        self.point = PointHead(cfg.d_model, cfg.simcc_bins)
        self.box = BoxHead(cfg.d_model, cfg.box_dfl_bins)
        self.polygon = PolygonHead(cfg.d_model, cfg.n_heads, cfg.polygon_max_vertices)
        self.mask = MaskHead(cfg.d_model)
        self.rank = OrderInvariantRankHead(cfg.d_model, cfg.n_heads, cfg.set_attn_layers)
        self.graph = BiaffineGraphHead(cfg.d_model, cfg.graph_num_relations)
        self.set_head = CorrelatedSetHead(cfg.d_model, cfg.set_ising_rank, cfg.set_max_cardinality)
        self.scalar = ScalarDistributionHead(cfg.d_model, cfg.mdn_components, cfg.quantiles)
