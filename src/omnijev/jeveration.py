"""Axis B — Universal Cross-Modal Jeveration Engine (Single-Pass Shared-Random-State Generation).

Generalizes Jeveration (arXiv:2611.24738) to all discrete-codebook modalities:
  - B1: Images (Text-to-Image, Image-to-Image) & Deterministic Dense Prediction (z=0)
  - B2: Interactive Video / World Models (action-conditioned + AR(1) temporal z persistence)
  - B3: Audio / Speech / SFX (Parallel Level-1 semantic + cascaded micro-pass RVQ levels 2..L)
  - B4: 3D Triplane / Gaussian Splat Discrete Codes
  - B5: Structured Code / AST Slot Filling with Grammar Masking + Few-Step Masked Diffusion Refinement
"""

from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OmniJevConfig


class AdaLNJeverationBlock(nn.Module):
    """Single Jeveration decoder block with AdaLN-Zero conditioning on shared random state z."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        # 6 modulation parameters: (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        # Small non-zero init prevents dead z-gradients / posterior collapse at step 0
        self.ada_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.normal_(self.ada_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.ada_mlp[-1].bias)

    def forward(self, q: torch.Tensor, context_h: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        g1, b1, a1, g2, b2, a2 = self.ada_mlp(z_cond).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm_self(q) * (1.0 + g1) + b1
        sa_out, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + (1.0 + a1) * sa_out

        ca_out, _ = self.cross_attn(self.norm_cross(q), context_h, context_h, need_weights=False)
        q = q + ca_out

        h2 = self.norm_ffn(q) * (1.0 + g2) + b2
        q = q + (1.0 + a2) * self.ffn(h2)
        return q


class PosteriorModeEncoder(nn.Module):
    """Training-time anti-collapse variational encoder q_phi(z | c_{1:M}, H).

    Forces the shared random state z to carry global mode bits (composition, timbre, trajectory branch)
    with a free-bits KL regularization preventing posterior collapse.
    """

    def __init__(self, vocab_size: int, d_model: int, z_dim: int):
        super().__init__()
        self.code_embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * z_dim),
        )

    def forward(
        self,
        target_codes: torch.Tensor,
        context_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_sample, mu, logvar)."""
        c_pool = self.code_embed(target_codes).mean(dim=1)    # [B, D]
        h_pool = context_h.mean(dim=1)                        # [B, D]
        mu, logvar = self.encoder(torch.cat([c_pool, h_pool], dim=-1)).chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 4.0)
        std = (0.5 * logvar).exp()
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar

    @staticmethod
    def free_bits_kl(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.25) -> torch.Tensor:
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B, z_dim]
        return torch.clamp(kl_per_dim, min=free_bits).sum(dim=-1).mean()


class UniversalJeverationHead(nn.Module):
    """Core single-pass parallel discrete codebook generator conditioned on shared random state z.

    Supports:
      - Pure System 1 single-pass generation (refinement_steps=0)
      - Deterministic dense prediction (deterministic=True -> z=0)
      - Grammar / vocabulary masking (for AST / code slot filling)
      - Hybrid few-step confidence-guided masked discrete diffusion refinement (refinement_steps > 0)
        while keeping the heavy backbone state H completely cached!
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_slots: int,
        vocab_size: int,
        z_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.vocab_size = vocab_size
        self.z_dim = z_dim

        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Embedding for optional few-step masked refinement (token ID vocab_size = [MASK])
        self.draft_embed = nn.Embedding(vocab_size + 1, d_model)
        self.blocks = nn.ModuleList([AdaLNJeverationBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.code_head = nn.Linear(d_model, vocab_size)
        self.posterior = PosteriorModeEncoder(vocab_size, d_model, z_dim)

    def _decode_once(
        self,
        context_h: torch.Tensor,
        z: torch.Tensor,
        draft_codes: Optional[torch.Tensor] = None,
        extra_cond: Optional[torch.Tensor] = None,
        grammar_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = context_h.size(0)
        q = self.slot_queries.expand(B, -1, -1)
        if draft_codes is not None:
            q = q + self.draft_embed(draft_codes)
        z_cond = self.z_proj(z)
        if extra_cond is not None:
            z_cond = z_cond + extra_cond
        q = q + z_cond.unsqueeze(1)

        for blk in self.blocks:
            q = blk(q, context_h, z_cond)
        logits = self.code_head(self.out_norm(q))  # [B, M, V]
        if grammar_mask is not None:
            logits = logits.masked_fill(~grammar_mask, float("-inf"))
        return logits

    def forward(
        self,
        context_h: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        extra_cond: Optional[torch.Tensor] = None,
        grammar_mask: Optional[torch.Tensor] = None,
        refinement_steps: int = 0,
        temperature: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        B = context_h.size(0)
        if deterministic:
            z_use = torch.zeros(B, self.z_dim, device=context_h.device, dtype=context_h.dtype)
        elif z is None:
            z_use = torch.randn(B, self.z_dim, device=context_h.device, dtype=context_h.dtype)
        else:
            z_use = z

        # Pass 1: Pure single-pass parallel Jeveration
        logits = self._decode_once(context_h, z_use, draft_codes=None, extra_cond=extra_cond, grammar_mask=grammar_mask)
        probs = F.softmax(logits, dim=-1)
        if temperature > 0.0:
            scaled = F.softmax(logits / temperature, dim=-1)
            codes = torch.multinomial(scaled.reshape(-1, self.vocab_size), 1).reshape(B, self.num_slots)
            conf = probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        else:
            conf, codes = probs.max(dim=-1)

        # Optional Hybrid System 1.5 few-step confidence-guided masked refinement (backbone H stays cached!)
        if refinement_steps > 0:
            mask_token = self.vocab_size
            locked = torch.zeros_like(codes, dtype=torch.bool)
            for step in range(1, refinement_steps + 1):
                # Lock top cosine-schedule fraction of highest-confidence slots
                keep_ratio = math.sin(0.5 * math.pi * step / (refinement_steps + 1))
                k_lock = max(1, int(round(keep_ratio * self.num_slots)))
                _, top_idx = conf.topk(k_lock, dim=-1)
                locked.scatter_(1, top_idx, True)

                draft = torch.where(locked, codes, torch.full_like(codes, mask_token))
                logits = self._decode_once(
                    context_h, z_use, draft_codes=draft, extra_cond=extra_cond, grammar_mask=grammar_mask
                )
                probs = F.softmax(logits, dim=-1)
                new_conf, new_codes = probs.max(dim=-1)
                codes = torch.where(locked, codes, new_codes)
                conf = torch.where(locked, conf, new_conf)

        return {
            "logits": logits,          # [B, M, V]
            "codes": codes,            # [B, M] long
            "confidence": conf,        # [B, M]
            "z": z_use,                # [B, z_dim]
        }

    def training_loss(
        self,
        context_h: torch.Tensor,
        target_codes: torch.Tensor,
        deterministic: bool = False,
        beta_kl: float = 0.05,
        free_bits: float = 0.25,
        extra_cond: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Anti-collapse variational single-pass Jeveration loss."""
        if deterministic:
            z = torch.zeros(context_h.size(0), self.z_dim, device=context_h.device, dtype=context_h.dtype)
            kl = torch.tensor(0.0, device=context_h.device, dtype=context_h.dtype)
        else:
            z, mu, logvar = self.posterior(target_codes, context_h)
            kl = self.posterior.free_bits_kl(mu, logvar, free_bits=free_bits)

        logits = self._decode_once(context_h, z, extra_cond=extra_cond)
        recon = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_codes.reshape(-1))
        total = recon + beta_kl * kl
        return {"loss": total, "recon_ce": recon, "kl": kl, "logits": logits}


class CascadedRVQAudioJeverationHead(nn.Module):
    """B3: Single macro-pass RVQ audio generator.

    Predicts Level-1 semantic codes in parallel via UJH, then runs a tiny 2-layer
    acoustic depth refinement MLP over RVQ levels 2..L in parallel across all time steps.
    """

    def __init__(self, cfg: OmniJevConfig):
        super().__init__()
        self.T = cfg.audio_time_steps
        self.L_rvq = cfg.audio_rvq_levels
        self.V = cfg.audio_codebook_size
        self.level1_head = UniversalJeverationHead(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            num_slots=self.T,
            vocab_size=self.V,
            z_dim=cfg.jeveration_z_dim,
            n_layers=cfg.jeveration_layers,
        )
        self.l1_embed = nn.Embedding(self.V, cfg.d_model)
        self.z_proj = nn.Linear(cfg.jeveration_z_dim, cfg.d_model)
        # Parallel acoustic depth heads for RVQ levels 2..L_rvq
        self.depth_trunk = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
        )
        self.fine_heads = nn.Linear(cfg.d_model, max(self.L_rvq - 1, 1) * self.V)

    def forward(
        self,
        context_h: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out1 = self.level1_head(context_h, z=z, deterministic=False)
        l1_codes = out1["codes"]                              # [B, T]
        z_used = out1["z"]                                    # [B, z_dim]
        B = context_h.size(0)

        ctx_pool = context_h.mean(dim=1, keepdim=True).expand(B, self.T, -1)
        l1_emb = self.l1_embed(l1_codes) + self.z_proj(z_used).unsqueeze(1)
        h_fine = self.depth_trunk(torch.cat([l1_emb, ctx_pool], dim=-1))  # [B, T, D]
        fine_logits = self.fine_heads(h_fine).view(B, self.T, self.L_rvq - 1, self.V)
        fine_codes = fine_logits.argmax(dim=-1)               # [B, T, L_rvq - 1]

        all_codes = torch.cat([l1_codes.unsqueeze(-1), fine_codes], dim=-1)  # [B, T, L_rvq]
        return {
            "rvq_codes": all_codes,
            "level1_logits": out1["logits"],
            "fine_logits": fine_logits,
            "z": z_used,
        }


class CrossModalJeverationSuite(nn.Module):
    """Unified Axis-B suite attaching single-pass Jeveration across all 5 generative modalities."""

    def __init__(self, cfg: OmniJevConfig):
        super().__init__()
        self.cfg = cfg
        img_slots = cfg.image_grid_hw[0] * cfg.image_grid_hw[1]
        tri_slots = 3 * cfg.triplane_grid_hw[0] * cfg.triplane_grid_hw[1]

        # B1: Image Generation (T2I / I2I)
        self.image_gen = UniversalJeverationHead(
            cfg.d_model, cfg.n_heads, cfg.d_ff, img_slots, cfg.image_codebook_size, cfg.jeveration_z_dim, cfg.jeveration_layers
        )
        # B1b: Deterministic Dense Prediction (Depth / Normals / Layout)
        self.dense_pred = UniversalJeverationHead(
            cfg.d_model, cfg.n_heads, cfg.d_ff, img_slots, cfg.dense_codebook_size, cfg.jeveration_z_dim, max(1, cfg.jeveration_layers - 1)
        )
        # B2: Interactive Video / World Model (Action-Conditioned + AR(1) temporal z)
        self.world_action_proj = nn.Sequential(
            nn.Linear(cfg.vla_action_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.video_world = UniversalJeverationHead(
            cfg.d_model, cfg.n_heads, cfg.d_ff, img_slots * cfg.video_frames_per_chunk,
            cfg.video_codebook_size, cfg.jeveration_z_dim, cfg.jeveration_layers
        )
        # B3: Cascaded RVQ Audio / Speech / SFX
        self.audio_rvq = CascadedRVQAudioJeverationHead(cfg)
        # B4: 3D Triplane / Gaussian Splat Discrete Codes
        self.triplane_3d = UniversalJeverationHead(
            cfg.d_model, cfg.n_heads, cfg.d_ff, tri_slots, cfg.triplane_codebook_size, cfg.jeveration_z_dim, cfg.jeveration_layers
        )
        # B5: Structured Code / AST Slot Filling + Few-Step Masked Diffusion Hybrid
        self.ast_fill = UniversalJeverationHead(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.ast_max_slots, cfg.ast_codebook_size, cfg.jeveration_z_dim, cfg.jeveration_layers
        )

    def predict_world_step(
        self,
        context_h: torch.Tensor,
        action: torch.Tensor,
        prev_z: Optional[torch.Tensor] = None,
        ar_rho: float = 0.85,
    ) -> Dict[str, torch.Tensor]:
        """Interactive world-model single-pass rollout step with AR(1) flicker-free temporal z."""
        B = context_h.size(0)
        eps = torch.randn(B, self.cfg.jeveration_z_dim, device=context_h.device, dtype=context_h.dtype)
        if prev_z is None:
            z_t = eps
        else:
            z_t = ar_rho * prev_z + math.sqrt(max(1.0 - ar_rho ** 2, 1e-6)) * eps
        act_cond = self.world_action_proj(action)
        out = self.video_world(context_h, z=z_t, deterministic=False, extra_cond=act_cond)
        H_g, W_g = self.cfg.image_grid_hw
        out["frame_codes"] = out["codes"].view(B, self.cfg.video_frames_per_chunk, H_g, W_g)
        return out
