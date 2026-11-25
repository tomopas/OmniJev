"""OmniJev v4 Benchmark 2 (Gaps #3 & #4):
  2A. Real AG News 2->4 Continual Consolidation UPGRADE using MultiLayerAttentionTap +
      CosineAlignedSleepConsolidator + Post-Sleep Temperature Scaling (fixing v3's 77% / ECE 0.121!)
  2B. Real 4-Wave 20-Intent Enterprise `PolyAI/banking77` Continual Consolidation inside the
      Closed-Loop Self-Healing Controller (Online ACI + Mahalanobis OOD + Fast-Weight Skills +
      Cosine-Aligned Sleep):
        Wave 0 (Base): 5 banking intents
        Wave 1: +5 brand-new unseen intents (10 total)
        Wave 2: +5 brand-new unseen intents (15 total)
        Wave 3: +5 brand-new unseen intents (20 total!)
      Tracks shock escalation spikes, self-healing escalation collapse, retention, and calibrated ECE!
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnijev import (
    CosineAlignedSleepConsolidator,
    FastWeightSkillConsolidator,
    MultiLayerAttentionTap,
    OnlineAdaptiveConformal,
    PrototypeMahalanobisOOD,
    SelfHealingAdaptiveController,
)
from omnijev.system15 import compute_ece

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@torch.no_grad()
def _extract_multilayer_features(hf_model, tokenizer, texts: List[str], num_layers: int = 4, max_len: int = 128) -> torch.Tensor:
    """Extract concatenated last-M-layer mean+last-token pooled representation from frozen Qwen2.5-0.5B.

    Captures richer intermediate discriminative semantics than naive last-layer mean pooling!
    """
    base = hf_model.model
    out_vecs = []
    for i in range(0, len(texts), 32):
        enc = tokenizer(
            texts[i: i + 32], return_tensors="pt", padding=True, truncation=True, max_length=max_len
        ).to(DEVICE)
        res = base(**enc, output_hidden_states=True, use_cache=False)
        hs = res.hidden_states[-num_layers:]  # last M layers
        mask = enc["attention_mask"].to(torch.float32)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        # Weighted blend of last 4 layers (emphasizing L-2 and L-1) + last non-pad token!
        weights = [0.15, 0.25, 0.30, 0.30]
        fused = sum(w * h.to(torch.float32) for w, h in zip(weights, hs))
        mean_pool = (fused * mask.unsqueeze(-1)).sum(dim=1) / denom
        last_idx = (mask.sum(dim=1).long() - 1).clamp_min(0)
        last_tok = fused[torch.arange(fused.size(0), device=DEVICE), last_idx]
        vec = 0.5 * F.normalize(mean_pool, p=2, dim=-1) + 0.5 * F.normalize(last_tok, p=2, dim=-1)
        out_vecs.append(F.normalize(vec, p=2, dim=-1) * (vec.size(-1) ** 0.25))
    return torch.cat(out_vecs, dim=0)


def run_agnews_v4_upgrade(hf_model, tokenizer) -> Dict[str, float]:
    """Upgrade v3's AG News 2->4 continual consolidation using Multi-Layer Tap + Cosine-Aligned Sleep + Temp Scaling."""
    ds_tr = load_dataset("fancyzhx/ag_news", split="train")
    ds_te = load_dataset("fancyzhx/ag_news", split="test")

    tr_b, te_b = {c: [] for c in range(4)}, {c: [] for c in range(4)}
    for r in ds_tr:
        c = int(r["label"])
        if len(tr_b[c]) < 160:
            tr_b[c].append(r["text"])
        if all(len(v) >= 160 for v in tr_b.values()):
            break
    for r in ds_te:
        c = int(r["label"])
        if len(te_b[c]) < 120:
            te_b[c].append(r["text"])
        if all(len(v) >= 120 for v in te_b.values()):
            break

    tr_f = {c: _extract_multilayer_features(hf_model, tokenizer, tr_b[c]) for c in range(4)}
    te_f = {c: _extract_multilayer_features(hf_model, tokenizer, te_b[c]) for c in range(4)}
    D = tr_f[0].size(-1)

    cons = CosineAlignedSleepConsolidator(d_model=D, max_classes=8, lora_rank=16, coreset_per_class=80).to(DEVICE)
    fw = FastWeightSkillConsolidator(d_model=D, max_classes=8, familiarity_threshold=0.45, evidence_gain_multiplier=5.0).to(DEVICE)

    # Base training ONLY on World (0) & Sports (1)
    x01 = torch.cat([tr_f[0], tr_f[1]], dim=0)
    y01 = torch.tensor([0] * 160 + [1] * 160, device=DEVICE)
    cons.fit_initial_classes(x01, y01, active_classes=2, steps=260, lr=3e-3)

    x_orig_te = torch.cat([te_f[0], te_f[1]], dim=0)
    y_orig_te = torch.tensor([0] * 120 + [1] * 120, device=DEVICE)
    x_nov_te = torch.cat([te_f[2], te_f[3]], dim=0)
    y_nov_te = torch.tensor([2] * 120 + [3] * 120, device=DEVICE)

    with torch.no_grad():
        pre_orig = cons.predict(x_orig_te, active_classes=2)
        orig_acc_before = (pre_orig["choice_idx"] == y_orig_te).float().mean().item() * 100.0
        orig_ece_before = compute_ece(pre_orig["probs"], y_orig_te)

        # Phase 1: Instant Fast-Weight consolidation of unseen Business (2) & Sci/Tech (3) (zero backprop!)
        for c in (2, 3):
            for i in range(120):
                fw.consolidate_from_system2(tr_f[c][i], verified_class_idx=c)
                cons.record_s2_sample(tr_f[c][i], target=c)
        fw_nov = cons.predict(x_nov_te, active_classes=4, fast_weights=fw)
        fw_orig = cons.predict(x_orig_te, active_classes=4, fast_weights=fw)
        fw_nov_acc = (fw_nov["choice_idx"] == y_nov_te).float().mean().item() * 100.0
        fw_orig_acc = (fw_orig["choice_idx"] == y_orig_te).float().mean().item() * 100.0

    # Phase 2: Cosine-Aligned Micro-LoRA Sleep + Complete Fast-Weight Eviction + Post-Sleep Temp Scaling!
    t0 = time.perf_counter()
    sleep_info = cons.run_sleep_wave(active_classes=4, fast_weights=fw, steps=260, lr=3e-3, kl_weight=1.0, evict_fast_weights=True)
    sleep_ms = (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        post_orig = cons.predict(x_orig_te, active_classes=4, fast_weights=None)
        post_nov = cons.predict(x_nov_te, active_classes=4, fast_weights=None)
        all_x = torch.cat([x_orig_te, x_nov_te], dim=0)
        all_y = torch.cat([y_orig_te, y_nov_te], dim=0)
        post_all = cons.predict(all_x, active_classes=4, fast_weights=None)

        orig_after = (post_orig["choice_idx"] == y_orig_te).float().mean().item() * 100.0
        nov_after = (post_nov["choice_idx"] == y_nov_te).float().mean().item() * 100.0
        all4_after = (post_all["choice_idx"] == all_y).float().mean().item() * 100.0
        all4_ece = compute_ece(post_all["probs"], all_y)

    return {
        "orig_World_Sports_before_pct": round(orig_acc_before, 2),
        "orig_ece_before": round(orig_ece_before, 4),
        "phase1_fastweight_novel_acc_pct": round(fw_nov_acc, 2),
        "phase1_fastweight_orig_acc_pct": round(fw_orig_acc, 2),
        "phase2_sleep_ms": round(sleep_ms, 1),
        "phase2_fast_weights_remaining": int(fw.num_stored.item()),
        "phase2_orig_retention_pct": round(orig_after, 2),
        "phase2_novel_permanent_acc_pct": round(nov_after, 2),
        "phase2_all4_agnews_acc_pct": round(all4_after, 2),
        "phase2_all4_calibrated_ece": round(all4_ece, 4),
        "calibrated_temp": sleep_info["calibrated_temp"],
    }


def run_banking77_4wave_self_healing(hf_model, tokenizer) -> Dict[str, object]:
    """4-Wave (5 -> 10 -> 15 -> 20 intents) Real Banking77 Closed-Loop Self-Healing ACI + FastWeight + Sleep!"""
    ds_tr = load_dataset("PolyAI/banking77", split="train")
    ds_te = load_dataset("PolyAI/banking77", split="test")

    # Select 20 well-populated enterprise banking intents cleanly remapped to 0..19
    counts = {}
    for r in ds_tr:
        counts[int(r["label"])] = counts.get(int(r["label"]), 0) + 1
    top20 = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)[:20]
    remap = {orig: idx for idx, orig in enumerate(top20)}

    tr_b = {c: [] for c in range(20)}
    te_b = {c: [] for c in range(20)}
    for r in ds_tr:
        lbl = int(r["label"])
        if lbl in remap and len(tr_b[remap[lbl]]) < 55:
            tr_b[remap[lbl]].append(r["text"])
    for r in ds_te:
        lbl = int(r["label"])
        if lbl in remap and len(te_b[remap[lbl]]) < 30:
            te_b[remap[lbl]].append(r["text"])

    tr_f = {c: _extract_multilayer_features(hf_model, tokenizer, tr_b[c], max_len=64) for c in range(20)}
    te_f = {c: _extract_multilayer_features(hf_model, tokenizer, te_b[c], max_len=64) for c in range(20)}
    D = tr_f[0].size(-1)

    cons = CosineAlignedSleepConsolidator(d_model=D, max_classes=24, lora_rank=16, coreset_per_class=35).to(DEVICE)
    fw = FastWeightSkillConsolidator(d_model=D, max_classes=24, familiarity_threshold=0.52, evidence_gain_multiplier=5.0).to(DEVICE)
    ood = PrototypeMahalanobisOOD(d_model=D, num_prototypes=24).to(DEVICE)
    aci = OnlineAdaptiveConformal(alpha_target=0.10, gamma_aci=0.04, window_size=120, lambda_epi=0.3, lambda_ood=0.3)

    # Wave 0: Base pretraining on Intents 0..4
    x0 = torch.cat([tr_f[c] for c in range(5)], dim=0)
    y0 = torch.cat([torch.full((tr_f[c].size(0),), c, dtype=torch.long, device=DEVICE) for c in range(5)])
    cons.fit_initial_classes(x0, y0, active_classes=5, steps=260, lr=3e-3)
    ood.fit(x0)
    with torch.no_grad():
        cal_out = cons.predict(x0, active_classes=5)
        aci.initialize(cal_out["probs"], y0, cal_out["epistemic_vacuity"], ood(x0)["ood_score"])

    controller = SelfHealingAdaptiveController(cons, fw, ood, aci, sleep_trigger_size=75, ood_escalate_thresh=0.55)

    def eval_seen(num_seen: int):
        xs = torch.cat([te_f[c] for c in range(num_seen)], dim=0)
        ys = torch.cat([torch.full((te_f[c].size(0),), c, dtype=torch.long, device=DEVICE) for c in range(num_seen)])
        with torch.no_grad():
            o = cons.predict(xs, active_classes=num_seen, fast_weights=None)
            acc = (o["choice_idx"] == ys).float().mean().item() * 100.0
            ece = compute_ece(o["probs"], ys)
        return round(acc, 2), round(ece, 4)

    w0_acc, w0_ece = eval_seen(5)
    wave_logs = [{
        "wave": 0,
        "total_intents": 5,
        "heldout_all_seen_acc_pct": w0_acc,
        "heldout_ece": w0_ece,
        "shock_esc_first25_pct": 0.0,
        "healed_esc_last50_pct": 0.0,
        "closed_loop_online_acc_pct": w0_acc,
    }]

    # Waves 1, 2, 3: Stream +5 brand-new unseen enterprise banking intents per wave!
    for w_idx, new_start in enumerate([5, 10, 15], start=1):
        num_active = new_start + 5
        # Build live stream mixing 75% new-wave intents + 25% existing intents
        stream_xs, stream_ys = [], []
        for c in range(new_start, num_active):
            stream_xs.append(tr_f[c])
            stream_ys.append(torch.full((tr_f[c].size(0),), c, dtype=torch.long, device=DEVICE))
        for c in range(new_start):
            k_old = min(8, tr_f[c].size(0))
            stream_xs.append(tr_f[c][:k_old])
            stream_ys.append(torch.full((k_old,), c, dtype=torch.long, device=DEVICE))

        sx = torch.cat(stream_xs, dim=0)
        sy = torch.cat(stream_ys, dim=0)
        perm = torch.randperm(sx.size(0), device=DEVICE)
        sx, sy = sx[perm], sy[perm]

        esc_flags, fin_ok = [], []
        for i in range(sx.size(0)):
            step_res = controller.step_live(sx[i: i + 1], true_label=int(sy[i].item()), active_classes=num_active)
            esc_flags.append(1 if step_res["escalated_s2"] else 0)
            fin_ok.append(1 if step_res["final_correct"] else 0)

        # Finish wave with clean end-of-wave sleep consolidation & recalibrate ACI/OOD on healed manifold!
        if cons.s2_buffer_x:
            cons.run_sleep_wave(active_classes=num_active, fast_weights=fw, steps=200, lr=3e-3, kl_weight=1.0, evict_fast_weights=True)
        with torch.no_grad():
            all_c = torch.cat([t.to(DEVICE) for t in cons.coreset_embs.values()], dim=0)
            all_yc = torch.cat([torch.full((t.size(0),), cid, dtype=torch.long, device=DEVICE) for cid, t in cons.coreset_embs.items()])
            ood.fit(all_c)
            cp = cons.predict(all_c, active_classes=num_active, fast_weights=None)
            aci.initialize(cp["probs"], all_yc, cp["epistemic_vacuity"], ood(all_c)["ood_score"])

        # Evaluate healed steady-state escalation on 50 held-out queries from all num_active intents
        te_xs = torch.cat([te_f[c][:5] for c in range(num_active)], dim=0)
        te_ys = torch.cat([torch.full((min(5, te_f[c].size(0)),), c, dtype=torch.long, device=DEVICE) for c in range(num_active)])
        p_te = torch.randperm(te_xs.size(0), device=DEVICE)[:50]
        healed_escs = 0
        for j in p_te.tolist():
            r_h = controller.step_live(te_xs[j: j + 1], true_label=int(te_ys[j].item()), active_classes=num_active)
            if r_h["escalated_s2"]:
                healed_escs += 1

        all_acc, all_ece = eval_seen(num_active)
        wave_logs.append({
            "wave": w_idx,
            "total_intents": num_active,
            "heldout_all_seen_acc_pct": all_acc,
            "heldout_ece": all_ece,
            "shock_esc_first25_pct": round(100.0 * sum(esc_flags[:25]) / 25.0, 1),
            "healed_esc_post_wave_pct": round(100.0 * healed_escs / 50.0, 1),
            "closed_loop_online_acc_pct": round(100.0 * sum(fin_ok) / len(fin_ok), 2),
        })

    return {
        "dataset": "PolyAI/banking77 (4 Continual Waves: 5 -> 10 -> 15 -> 20 Real Enterprise Intents)",
        "total_sleep_cycles": controller.sleep_cycles_run,
        "waves": wave_logs,
    }


def main():
    torch.manual_seed(42)
    print(f"Loading frozen {MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()

    print("\nRunning 2A: AG News v4 Continual Consolidation Upgrade...")
    res_ag = run_agnews_v4_upgrade(hf_model, tokenizer)
    print(json.dumps(res_ag, indent=2))

    print("\nRunning 2B: Banking77 4-Wave Closed-Loop Self-Healing ACI + FastWeight + Sleep...")
    res_b77 = run_banking77_4wave_self_healing(hf_model, tokenizer)
    print(json.dumps(res_b77, indent=2))

    report = {
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "exp2a_agnews_v4_cosine_multilayer_upgrade": res_ag,
        "exp2b_banking77_4wave_self_healing": res_b77,
    }
    out_path = ROOT / "experiments" / "v4_multiwave_selfhealing_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
