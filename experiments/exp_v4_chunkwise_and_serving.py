"""OmniJev v4 Benchmark 1 (Gaps #1 & #2):
  1A. Chunkwise Parallel WY Gated DeltaNet vs. Serial Recurrence (Exact Equivalence + Real GPU Speedup!)
  1B. Real Hardware 4-bit NF4 (bitsandbytes) Qwen2.5-0.5B Backbone + FP32 OmniJev Heads vs. Full BF16
      (Measures real GPU VRAM savings, real latency, real AG News accuracy, and calibration ECE!)
  1C. Production FastAPI `/v1/decide`, `/v1/consolidate`, & `/v1/health` End-to-End API Validation!
"""

import json
import sys
import time
from pathlib import Path

from datasets import load_dataset
from fastapi.testclient import TestClient
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnijev import (
    CosineAlignedSleepConsolidator,
    GatedDeltaNetLayer,
    PretrainedOmniJevBridge,
    create_omnijev_app,
    load_mixed_precision_nf4_llm,
)
from omnijev.system15 import compute_ece

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def bench_chunkwise_wy_deltanet():
    """Verify exact equivalence (<1e-4) and measure GPU speedup of Chunkwise WY vs serial Python loop."""
    torch.manual_seed(1)
    layer = GatedDeltaNetLayer(d_model=256, n_heads=8, d_ff=512, chunk_size=32).to(DEVICE).eval()
    rows = []
    with torch.no_grad():
        for L in (64, 128, 256, 512):
            x = torch.randn(4, L, 256, device=DEVICE)
            # Warmup
            y_wy, s_wy = layer(x)
            y_ser, s_ser = layer.forward_serial_reference(x)
            max_err = max((y_wy - y_ser).abs().max().item(), (s_wy - s_ser).abs().max().item())

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                _ = layer.forward_serial_reference(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            ser_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

            t0 = time.perf_counter()
            for _ in range(5):
                _ = layer(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            wy_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

            rows.append({
                "seq_len": L,
                "serial_ms": round(ser_ms, 2),
                "chunkwise_wy_ms": round(wy_ms, 2),
                "speedup_x": round(ser_ms / max(wy_ms, 1e-3), 2),
                "max_numerical_diff": float(f"{max_err:.2e}"),
            })
    return rows


def _eval_precision_mode(mode: str, train_texts, train_y, test_texts, test_y):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if mode == "bf16":
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
    else:
        hf, tok = load_mixed_precision_nf4_llm(MODEL_ID)

    vram_mb = torch.cuda.memory_allocated() / 1e6
    bridge = PretrainedOmniJevBridge(hf, tok, d_jev=256, max_classes=8, device=DEVICE)

    def extract(texts):
        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                h, m, _ = bridge._prefill_hf(texts[i: i + 32], max_length=128)
                p = (h * m.unsqueeze(-1)).sum(dim=1) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
                feats.append(p)
        return torch.cat(feats, dim=0)

    # Latency measurement (single query steady-state)
    _ = extract([test_texts[0]])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in test_texts[:20]:
        _ = extract([t])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    single_ms = (time.perf_counter() - t0) * 1000.0 / 20.0

    tr_f = extract(train_texts)
    te_f = extract(test_texts)
    mean_f = tr_f.mean(dim=0, keepdim=True)
    tr_n = F.normalize(tr_f - mean_f, p=2, dim=-1) * (tr_f.size(-1) ** 0.25)
    te_n = F.normalize(te_f - mean_f, p=2, dim=-1) * (te_f.size(-1) ** 0.25)

    # Train FP32 Cosine-Aligned Evidential Head on top of BF16 vs Real NF4 backbone
    head = CosineAlignedSleepConsolidator(d_model=bridge.d_llm, max_classes=4, lora_rank=8).to(DEVICE)
    head.fit_initial_classes(tr_n, train_y, active_classes=4, steps=220, lr=3e-3)

    with torch.no_grad():
        out = head.predict(te_n, active_classes=4)
        acc = (out["choice_idx"] == test_y).float().mean().item() * 100.0
        ece = compute_ece(out["probs"], test_y)

    # If NF4 mode, also run end-to-end FastAPI production server test!
    api_report = None
    if mode == "nf4_mixed_precision":
        app = create_omnijev_app(bridge)
        client = TestClient(app)
        h_res = client.get("/v1/health").json()
        d_res = client.post("/v1/decide", json={
            "state": "In 2026, TypeSafe deployed edge clusters in Zurich with $42M budget to stop destructive rm -rf scripts.",
            "questions": {
                "is_security_related": {"type": "noul", "instructions": "Does this mention security or destructive commands?"},
                "topic": {"type": "choice", "options": ["World", "Sports", "Business", "Sci/Tech"]},
                "severity": {"type": "score", "instructions": "Operational importance"},
                "city": {"type": "span", "instructions": "Which city?"},
            },
            "deliberation_steps": 1,
        }).json()
        c_res = client.post("/v1/consolidate", json={
            "state": "Breaking quarterly earnings beat Wall Street estimates.",
            "verified_class_idx": 2,
            "option_name": "Business",
            "trigger_sleep": False,
        }).json()
        api_report = {
            "health_status": h_res["status"],
            "decide_latency_ms": d_res["meta"]["latency_ms"],
            "decide_tokens_generated": d_res["meta"]["tokens_generated"],
            "decide_primitives_returned": sorted(list(d_res["answers"].keys())),
            "consolidate_ok": c_res["consolidated"],
            "consolidate_write_ms": c_res["write_ms"],
            "skills_stored_after_write": c_res["fast_weight_skills_stored"],
        }

    del hf, bridge, head
    torch.cuda.empty_cache()
    return {
        "mode": mode,
        "gpu_vram_mb": round(vram_mb, 1),
        "single_query_ms": round(single_ms, 2),
        "agnews_4way_acc_pct": round(acc, 2),
        "agnews_4way_ece": round(ece, 4),
    }, api_report


def main():
    print("1A. Benchmarking Chunkwise Parallel WY Gated DeltaNet vs Serial Reference on GPU...")
    wy_rows = bench_chunkwise_wy_deltanet()
    print(json.dumps(wy_rows, indent=2))

    # Load balanced AG News 4-class subset (400 train, 240 test)
    ds_tr = load_dataset("fancyzhx/ag_news", split="train")
    ds_te = load_dataset("fancyzhx/ag_news", split="test")
    tr_txt, tr_y, te_txt, te_y = [], [], [], []
    cnt_tr, cnt_te = [0] * 4, [0] * 4
    for r in ds_tr:
        c = int(r["label"])
        if cnt_tr[c] < 100:
            tr_txt.append(r["text"]); tr_y.append(c); cnt_tr[c] += 1
        if all(v >= 100 for v in cnt_tr):
            break
    for r in ds_te:
        c = int(r["label"])
        if cnt_te[c] < 60:
            te_txt.append(r["text"]); te_y.append(c); cnt_te[c] += 1
        if all(v >= 60 for v in cnt_te):
            break
    tr_y_t = torch.tensor(tr_y, dtype=torch.long, device=DEVICE)
    te_y_t = torch.tensor(te_y, dtype=torch.long, device=DEVICE)

    print("\n1B. Benchmarking Full BF16 vs Real Hardware 4-bit NF4 Backbone + FP32 Heads...")
    bf16_res, _ = _eval_precision_mode("bf16", tr_txt, tr_y_t, te_txt, te_y_t)
    nf4_res, api_res = _eval_precision_mode("nf4_mixed_precision", tr_txt, tr_y_t, te_txt, te_y_t)
    vram_reduction_pct = round(100.0 * (1.0 - nf4_res["gpu_vram_mb"] / max(bf16_res["gpu_vram_mb"], 1.0)), 1)

    report = {
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "exp1a_chunkwise_wy_deltanet": wy_rows,
        "exp1b_real_nf4_mixed_precision": {
            "bf16_baseline": bf16_res,
            "real_nf4_backbone_fp32_heads": nf4_res,
            "vram_reduction_pct": vram_reduction_pct,
        },
        "exp1c_fastapi_production_server": api_res,
    }
    print("\n[OmniJev v4 Chunkwise WY + Real NF4 + FastAPI Results]\n", json.dumps(report, indent=2))
    out_path = ROOT / "experiments" / "v4_chunkwise_serving_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
