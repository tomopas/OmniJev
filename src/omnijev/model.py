"""Unified OmniJevModel tying Encoders -> Streaming Backbone -> System 1.5 Deliberation -> Axes A/B/C/D Heads."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbone import StreamingHybridBackbone, StreamingState, System15LatentDeliberationCore
from .config import OmniJevConfig
from .embodied import JevCUAHead, JevVLAHead, ToolGuardrailHead
from .encoders import ModalityBatch, UniversalMultimodalBus
from .jeveration import CrossModalJeverationSuite
from .primitives import AxisAPrimitiveSuite
from .system15 import CostAwareEscalationRouter, PrototypeMahalanobisOOD
from .v2_frontier import (
    FastWeightSkillConsolidator,
    LatentJEPAWorldHead,
    LatentMentalSimController,
    OnlineAdaptiveConformal,
    OrthogonalSubspaceAdapterBank,
    ScratchpadEquilibriumCore,
)


class OmniJevModel(nn.Module):
    """Unified System 1 / 1.5 Decision, Action, Mental-Simulation, & Cross-Modal Generation Model (v2)."""

    def __init__(self, cfg: OmniJevConfig):
        super().__init__()
        self.cfg = cfg

        # Stage 1: Universal Multimodal Input Bus
        self.bus = UniversalMultimodalBus(cfg)

        # Stage 2: Dual-Timescale Streaming Backbone (3:1 Gated DeltaNet + SWA)
        self.backbone = StreamingHybridBackbone(cfg)

        # Stage 3: System 1.5 Recurrent Latent Deliberation Cores (v1 Huginn+TaH2 & v2 Scratchpad Equilibrium)
        self.deliberation = System15LatentDeliberationCore(cfg)
        self.scratchpad_core = ScratchpadEquilibriumCore(cfg.d_model, cfg.n_heads, cfg.d_ff, num_registers=8)

        # Stage 3.5 (v2 B6): Orthogonal Multi-Task Subspace Adapters protecting calibration across Axes A-D
        self.subspaces = OrthogonalSubspaceAdapterBank(cfg.d_model)

        # Stage 4A: All 12 Axis-A Typed Primitives
        self.primitives = AxisAPrimitiveSuite(cfg)

        # Stage 4B: All 5 Axis-B Cross-Modal Jeveration Generators
        self.jeveration = CrossModalJeverationSuite(cfg)

        # Stage 4C: Embodied Robotics (Jev-VLA), Computer-Use (Jev-CUA), Guardrails, & v2 Mental-Sim MPC
        self.vla = JevVLAHead(cfg)
        self.world_jepa = LatentJEPAWorldHead(cfg.d_model, cfg.vla_action_dim, cfg.vla_chunk_horizon)
        self.mental_sim = LatentMentalSimController(self.world_jepa)
        self.cua = JevCUAHead(cfg)
        self.guardrail = ToolGuardrailHead(cfg)

        # Stage 4D: Latent OOD Detector, Cost-Aware Router, Online ACI, & Fast-Weight S2->S1 Skill Consolidator
        self.ood_detector = PrototypeMahalanobisOOD(cfg.d_model, cfg.ood_num_prototypes)
        self.router = CostAwareEscalationRouter(
            error_cost=cfg.escalation_error_cost,
            s2_cost=cfg.escalation_s2_cost,
            ooi_threshold=cfg.ooi_threshold,
        )
        self.online_aci = OnlineAdaptiveConformal(alpha_target=0.10)
        self.skill_consolidator = FastWeightSkillConsolidator(cfg.d_model)

        # Default learnable task query embedding when no explicit text query is passed
        self.default_query = nn.Parameter(torch.randn(1, cfg.d_model) * 0.02)

    def encode_state(
        self,
        batch: ModalityBatch,
        streaming_state: Optional[StreamingState] = None,
        deliberation_steps: Optional[int] = 0,
        adaptive_halt: bool = False,
    ) -> Dict[str, object]:
        """Encode multimodal inputs once, update streaming state, and run System 1 / 1.5 deliberation.

        Args:
            batch: Multimodal input container (any subset of text, image, audio, proprio, 3d, graph).
            streaming_state: Optional persistent O(1) state from previous streaming chunks.
            deliberation_steps: 0 = pure System 1; K > 0 = fixed System 1.5 depth; None = adaptive halting.
            adaptive_halt: Enable TaH2 adaptive early stopping when deliberation_steps is None.
        """
        enc = self.bus(batch)
        h_prelude, next_stream_state = self.backbone.forward_prelude(enc["tokens"], state=streaming_state)

        delib = self.deliberation(
            h_prelude,
            num_steps=deliberation_steps,
            adaptive_halt=adaptive_halt,
        )
        h_out = self.backbone.forward_coda(delib["hidden"])

        return {
            "state_h": h_out,                             # [B, L, D]
            "prelude_h": h_prelude,                       # [B, L, D]
            "spatial_map": enc["spatial_map"],            # [B, D, H_f, W_f] or None
            "proprio_raw": enc["proprio_raw"],            # [B, D_prop] or None
            "slices": enc["slices"],
            "streaming_state": next_stream_state,
            "steps_executed": delib["steps_executed"],
            "halt_probs": delib["halt_probs"],
            "kl_trace": delib["kl_trace"],
        }

    def query_vector(self, encoded: Dict[str, object], query_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-pool a task query vector against the shared encoded state H."""
        state_h: torch.Tensor = encoded["state_h"]  # type: ignore[assignment]
        B = state_h.size(0)
        q_in = self.default_query.expand(B, -1) if query_emb is None else query_emb
        return self.primitives.pooler(state_h, q_in)

    def encode_isolated_options(self, option_token_ids: torch.Tensor) -> torch.Tensor:
        """Encode candidate option token sequences [B, C, L_opt] with isolated local RoPE reset (0..L_opt).

        Guarantees exact O(1) permutation invariance because each option starts at local pos 0
        with zero cross-option positional bias.
        """
        B, C, Lo = option_token_ids.shape
        flat = option_token_ids.reshape(B * C, Lo)
        emb = self.bus.text_embed(flat)                   # [B*C, Lo, D]
        # Pure local mean pooling (or local position-invariant summary) per option lane
        return emb.mean(dim=1).view(B, C, -1)             # [B, C, D]
