"""Axis E — Edge Deployment Profiler, Hardware Latency/FLOP Estimator, & Mixed-Precision Quantization Simulator."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import copy
import torch
import torch.nn as nn

from .config import MODEL_TIERS, OmniJevConfig, get_tier_config


@dataclass(frozen=True)
class HardwareProfile:
    """Roofline compute & memory bandwidth characteristics for cloud and edge accelerators."""

    name: str
    tflops_fp8: float
    tflops_int4: float
    bandwidth_gbps: float
    vram_gb: float
    dispatch_overhead_ms: float


HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "H200_FP8": HardwareProfile("NVIDIA H200 (FP8)", tflops_fp8=1979.0, tflops_int4=3958.0, bandwidth_gbps=4800.0, vram_gb=141.0, dispatch_overhead_ms=1.2),
    "RTX4090_FP8": HardwareProfile("NVIDIA RTX 4090 (FP8)", tflops_fp8=660.0, tflops_int4=1320.0, bandwidth_gbps=1008.0, vram_gb=24.0, dispatch_overhead_ms=1.8),
    "Jetson_Orin_AGX": HardwareProfile("Jetson Orin AGX 64GB", tflops_fp8=137.0, tflops_int4=275.0, bandwidth_gbps=204.8, vram_gb=64.0, dispatch_overhead_ms=2.2),
    "Jetson_Orin_Nano": HardwareProfile("Jetson Orin Nano / Edge NPU", tflops_fp8=20.0, tflops_int4=40.0, bandwidth_gbps=68.0, vram_gb=8.0, dispatch_overhead_ms=2.5),
    "Apple_M4_Max": HardwareProfile("Apple M4 Max (Unified)", tflops_fp8=54.0, tflops_int4=108.0, bandwidth_gbps=546.0, vram_gb=64.0, dispatch_overhead_ms=2.0),
}


def profile_model_tier(
    tier: str,
    hw_name: str,
    seq_len: int = 256,
    deliberation_steps: int = 0,
    include_jeveration: bool = False,
) -> Dict[str, float]:
    """Analytical roofline estimator for memory footprint, O(1) streaming state, FLOPs, and latency.

    Uses calibration-preserving mixed precision:
      - Backbone weights quantized to cfg.backbone_quant_bits (INT4 or FP8)
      - Decision / Evidential / SimCC / VLA heads preserved at 16-bit FP16/BF16 (<1.5% of weights)
    """
    cfg = get_tier_config(tier)
    hw = HARDWARE_PROFILES[hw_name]

    total_params = cfg.nominal_active_params_b * 1e9
    head_frac = 0.015
    backbone_params = total_params * (1.0 - head_frac)
    head_params = total_params * head_frac

    # Weight memory in GB (mixed-precision)
    weight_bytes = backbone_params * (cfg.backbone_quant_bits / 8.0) + head_params * (cfg.head_quant_bits / 8.0)
    weight_gb = weight_bytes / 1e9

    # Constant O(1) streaming state memory:
    #   - Gated DeltaNet layers: H * Dh * Dh * 4 bytes (fp32 accum) per GDN layer
    #   - SWA layers: 2 * swa_window * D * 2 bytes (bf16 KV ring buffer) per SWA layer
    n_swa = cfg.n_prelude_layers // (cfg.gdn_ratio + 1)
    n_gdn = cfg.n_prelude_layers - n_swa
    dh = cfg.d_model // cfg.n_heads
    gdn_state_bytes = n_gdn * cfg.n_heads * dh * dh * 4
    swa_state_bytes = n_swa * 2 * cfg.swa_window * cfg.d_model * 2
    streaming_state_mb = (gdn_state_bytes + swa_state_bytes) / 1e6

    # Prefill / single-pass FLOPs: ~2 * params * seq_len
    base_flops = 2.0 * total_params * seq_len
    # Recurrent System 1.5 overhead scales only by fraction of recurrent layers
    total_backbone_layers = max(cfg.n_prelude_layers + cfg.n_coda_layers, 1)
    rec_frac = cfg.n_recurrent_layers / total_backbone_layers
    delib_flops = deliberation_steps * rec_frac * base_flops
    jev_flops = (0.35 * base_flops) if include_jeveration else 0.0
    total_flops = base_flops + delib_flops + jev_flops

    # Roofline latency = max(compute_time, memory_read_time) + kernel dispatch overhead
    # Account for ~45% realistic accelerator utilization on short single-pass batches
    peak_tflops = hw.tflops_int4 if cfg.backbone_quant_bits <= 4 else hw.tflops_fp8
    eff_tflops = peak_tflops * 0.45
    compute_ms = (total_flops / (eff_tflops * 1e12)) * 1e3

    # Each recurrent loop re-reads only the small weight-tied recurrent block
    effective_weight_read_gb = weight_gb * (1.0 + deliberation_steps * rec_frac + (0.20 if include_jeveration else 0.0))
    mem_ms = (effective_weight_read_gb / (hw.bandwidth_gbps * 0.70)) * 1e3
    latency_ms = max(compute_ms, mem_ms) + hw.dispatch_overhead_ms * (1.0 + 0.25 * deliberation_steps)

    return {
        "tier": tier,
        "hardware": hw.name,
        "params_b": cfg.nominal_active_params_b,
        "backbone_bits": cfg.backbone_quant_bits,
        "head_bits": cfg.head_quant_bits,
        "weight_gb": round(weight_gb, 3),
        "streaming_state_mb": round(streaming_state_mb, 2),
        "fits_vram": (weight_gb + streaming_state_mb / 1e3 + 0.5) <= hw.vram_gb,
        "gflops": round(total_flops / 1e9, 1),
        "latency_ms": round(latency_ms, 2),
        "max_hz": round(1000.0 / max(latency_ms, 0.1), 1),
    }


def _fake_quantize_tensor(w: torch.Tensor, bits: int = 4, group_size: int = 32) -> torch.Tensor:
    """Symmetric group-wise fake weight quantization for evaluating calibration sensitivity."""
    if bits >= 16:
        return w
    orig_shape = w.shape
    flat = w.reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=w.device, dtype=w.dtype)])
    groups = flat.view(-1, group_size)
    qmax = (1 << (bits - 1)) - 1
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(groups / scale), -qmax, qmax) * scale
    if pad > 0:
        q = q.reshape(-1)[:-pad]
    return q.reshape(orig_shape)


@torch.no_grad()
def simulate_mixed_precision_quantization(
    model: nn.Module,
    backbone_bits: int = 4,
    preserve_heads: bool = True,
    head_prefixes: Tuple[str, ...] = ("primitives.", "vla.", "cua.", "guardrail.", "ood_detector."),
    group_size: int = 32,
) -> nn.Module:
    """Clone model and apply either naive uniform INT4 or calibration-preserving mixed-precision quantization."""
    q_model = copy.deepcopy(model)
    for name, param in q_model.named_parameters():
        if param.dim() < 2:
            continue
        is_head = any(name.startswith(pref) for pref in head_prefixes)
        if preserve_heads and is_head:
            continue  # Keep typed/evidential/SimCC/VLA heads in FP16/FP32!
        param.copy_(_fake_quantize_tensor(param.data, bits=backbone_bits, group_size=group_size))
    return q_model
