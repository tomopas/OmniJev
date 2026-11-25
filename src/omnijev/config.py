"""Unified configuration and 5-tier model family presets for OmniJev (Axes A-E)."""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class OmniJevConfig:
    """Configuration for an OmniJev System 1 / 1.5 Decision-Action-Generation model."""

    name: str = "omnijev-micro"

    # Core backbone dimensions
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 384
    n_prelude_layers: int = 4      # 3:1 Gated DeltaNet : SWA ratio
    n_coda_layers: int = 2
    swa_window: int = 64
    gdn_ratio: int = 3             # 3 GDN layers per 1 SWA layer
    dropout: float = 0.0

    # System 1.5 Recurrent Latent Deliberation (Axis D)
    n_recurrent_layers: int = 2    # Weight-tied core block depth per loop
    max_deliberation_steps: int = 6
    halt_threshold: float = 0.85
    kl_stability_eps: float = 1e-3

    # Multimodal Input Bus
    vocab_size: int = 4096
    image_channels: int = 3
    image_patch_size: int = 8
    audio_mels: int = 64
    proprio_dim: int = 16
    point3d_dim: int = 6           # (x, y, z, nx, ny, nz) or RGB
    graph_feat_dim: int = 32
    max_seq_len: int = 1024

    # Axis A: Typed Primitives
    set_attn_layers: int = 2
    max_score_levels: int = 5
    simcc_bins: int = 64           # High-res 1D bins (512 in production)
    box_dfl_bins: int = 32
    polygon_max_vertices: int = 8
    graph_num_relations: int = 8
    set_ising_rank: int = 8
    set_max_cardinality: int = 16
    mdn_components: int = 4
    quantiles: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

    # Axis B: Universal Jeveration (Single-Pass Cross-Modal Generation)
    jeveration_z_dim: int = 32
    jeveration_layers: int = 2
    image_codebook_size: int = 256
    image_grid_hw: Tuple[int, int] = (8, 8)
    dense_codebook_size: int = 64
    video_codebook_size: int = 256
    video_frames_per_chunk: int = 2
    audio_codebook_size: int = 128
    audio_rvq_levels: int = 4
    audio_time_steps: int = 16
    triplane_codebook_size: int = 128
    triplane_grid_hw: Tuple[int, int] = (4, 4)
    ast_codebook_size: int = 256
    ast_max_slots: int = 16

    # Axis C: Embodied Control (Jev-VLA, Jev-CUA, Guardrails)
    vla_action_dim: int = 7        # 6-DoF delta pose + gripper
    vla_chunk_horizon: int = 16
    vla_z_dim: int = 16
    vla_head_layers: int = 2
    vla_fast_tokens: int = 12
    vla_fast_vocab: int = 128
    cua_num_actions: int = 10      # click, dbl, rclick, drag, scroll, hotkey, type, wait, done, abstain
    cua_num_hotkeys: int = 16
    guardrail_num_risks: int = 8

    # Axis D: Calibration & Escalation
    ood_num_prototypes: int = 16
    escalation_error_cost: float = 10.0
    escalation_s2_cost: float = 0.5
    ooi_threshold: float = 0.35

    # Analytical metadata for production model tiers (Axis E)
    nominal_active_params_b: float = 0.005
    backbone_quant_bits: int = 16
    head_quant_bits: int = 16


# Production 5-tier model family presets (scaled specifications for Axis E profiling)
MODEL_TIERS: Dict[str, OmniJevConfig] = {
    "micro": OmniJevConfig(
        name="omnijev-micro",
        d_model=128,
        n_heads=4,
        d_ff=384,
        n_prelude_layers=4,
        n_coda_layers=2,
        n_recurrent_layers=2,
        max_deliberation_steps=6,
        nominal_active_params_b=0.006,
    ),
    "nano": OmniJevConfig(
        name="omnijev-nano-1.5b",
        d_model=1536,
        n_heads=12,
        d_ff=4096,
        n_prelude_layers=16,
        n_coda_layers=4,
        n_recurrent_layers=4,
        swa_window=2048,
        max_deliberation_steps=2,
        simcc_bins=256,
        jeveration_z_dim=64,
        jeveration_layers=2,
        vla_chunk_horizon=16,
        nominal_active_params_b=1.5,
        backbone_quant_bits=4,
        head_quant_bits=16,
    ),
    "edge": OmniJevConfig(
        name="omnijev-edge-4b",
        d_model=2560,
        n_heads=20,
        d_ff=6912,
        n_prelude_layers=24,
        n_coda_layers=4,
        n_recurrent_layers=4,
        swa_window=2048,
        max_deliberation_steps=4,
        simcc_bins=512,
        jeveration_z_dim=128,
        jeveration_layers=4,
        vla_chunk_horizon=32,
        nominal_active_params_b=4.0,
        backbone_quant_bits=8,
        head_quant_bits=16,
    ),
    "base": OmniJevConfig(
        name="omnijev-base-12b",
        d_model=3840,
        n_heads=30,
        d_ff=10240,
        n_prelude_layers=32,
        n_coda_layers=6,
        n_recurrent_layers=6,
        swa_window=4096,
        max_deliberation_steps=6,
        simcc_bins=512,
        jeveration_z_dim=256,
        jeveration_layers=6,
        image_codebook_size=16384,
        image_grid_hw=(32, 32),
        vla_chunk_horizon=50,
        nominal_active_params_b=12.0,
        backbone_quant_bits=8,
        head_quant_bits=16,
    ),
    "pro": OmniJevConfig(
        name="omnijev-pro-32b",
        d_model=5120,
        n_heads=40,
        d_ff=14336,
        n_prelude_layers=48,
        n_coda_layers=8,
        n_recurrent_layers=8,
        swa_window=4096,
        max_deliberation_steps=8,
        simcc_bins=512,
        jeveration_z_dim=256,
        jeveration_layers=6,
        image_codebook_size=16384,
        image_grid_hw=(32, 32),
        vla_chunk_horizon=50,
        nominal_active_params_b=32.0,
        backbone_quant_bits=8,
        head_quant_bits=16,
    ),
    "world": OmniJevConfig(
        name="omnijev-world-12b",
        d_model=3840,
        n_heads=30,
        d_ff=10240,
        n_prelude_layers=28,
        n_coda_layers=4,
        n_recurrent_layers=4,
        swa_window=4096,
        max_deliberation_steps=2,
        simcc_bins=512,
        jeveration_z_dim=256,
        jeveration_layers=8,
        image_codebook_size=16384,
        image_grid_hw=(32, 32),
        video_codebook_size=16384,
        video_frames_per_chunk=4,
        nominal_active_params_b=12.0,
        backbone_quant_bits=8,
        head_quant_bits=16,
    ),
}


def get_tier_config(tier: str) -> OmniJevConfig:
    """Return a copy of a named OmniJev model tier configuration."""
    key = tier.lower()
    if key not in MODEL_TIERS:
        raise ValueError(f"Unknown OmniJev tier '{tier}'. Available: {list(MODEL_TIERS.keys())}")
    base = MODEL_TIERS[key]
    return OmniJevConfig(**base.__dict__)
