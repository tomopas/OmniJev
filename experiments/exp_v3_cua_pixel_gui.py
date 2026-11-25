"""Real-World Benchmark 3 (OmniJev v3): Rendered RGB Multi-Widget Pixel GUI + 2-Pass CDG Zoom + Safety Guardrail.

Trains and evaluates OmniJev's vision encoder + Jev-CUA head end-to-end on REAL rendered RGB desktop/web
screenshots (Pillow) with:
  - Top window bar, left sidebar, multiple colorful distractor widgets, and tiny 14x14 px target buttons
  - Distinct visual cue & instruction types:
      * click_safe_green (action=click, is_destructive=0)
      * click_primary_blue (action=click, is_destructive=0)
      * click_danger_red (action=click, is_destructive=1 -> inline Noul safety guardrail MUST block!)
      * type_search_input (action=type_text -> MUST trigger selective System-2 wakeup!)
  - Compares 1-Pass Coarse Point vs. OmniJev v3 2-Pass Composable Decision Graph Zoom (Box -> Crop -> Point)
    measuring real button click-hit rate, mean pixel error, destructive block F1, and real RTX 3060 actions/sec!
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnijev import (
    CDGNode,
    ComposableDecisionGraph,
    EvidentialDirichletLoss,
    ModalityBatch,
    OmniJevConfig,
    OmniJevModel,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

# 4 semantic UI instruction templates (encoded as token sequences)
TASK_TEMPLATES = {
    0: ("click_green_save", 0, 0),       # action_idx=0 (click), destructive=0
    1: ("click_blue_export", 0, 0),      # action_idx=0 (click), destructive=0
    2: ("click_red_delete", 0, 1),       # action_idx=0 (click), destructive=1 (GUARDRAIL!)
    3: ("type_in_search_box", 6, 0),     # action_idx=6 (type_text -> WAKE S2!), destructive=0
}


def _render_gui_sample(rng: np.random.Generator) -> Dict[str, object]:
    """Render a realistic 128x128 RGB desktop app screenshot with tiny target widgets + distractors."""
    # Background desktop canvas + chrome
    bg_tone = int(rng.integers(228, 248))
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (bg_tone, bg_tone, bg_tone + 3))
    draw = ImageDraw.Draw(img)

    # Top title bar & left navigation sidebar
    draw.rectangle([0, 0, IMG_SIZE, 14], fill=(42, 48, 60))
    draw.rectangle([0, 14, 22, IMG_SIZE], fill=(212, 218, 228))
    # Window traffic-light dots
    draw.ellipse([3, 4, 8, 9], fill=(220, 80, 80))
    draw.ellipse([11, 4, 16, 9], fill=(230, 185, 60))
    draw.ellipse([19, 4, 24, 9], fill=(80, 195, 95))

    # Draw 4 non-overlapping widgets in the main pane [26..120, 18..120]
    slots = []
    widget_specs = [
        ("green", (34, 175, 85), (14, 12)),    # Tiny 14x12 Save button
        ("blue", (45, 115, 235), (14, 12)),    # Tiny 14x12 Export button
        ("red", (225, 45, 55), (14, 12)),      # Tiny 14x12 Destructive Delete button
        ("input", (255, 255, 255), (28, 12)),  # White bordered search input box
    ]
    # Add 3 neutral gray distractor cards/buttons
    for _ in range(3):
        dx = int(rng.integers(26, 96))
        dy = int(rng.integers(18, 108))
        draw.rectangle([dx, dy, dx + 20, dy + 10], fill=(175, 182, 192), outline=(140, 146, 156))

    boxes = {}
    for w_name, color, (bw, bh) in widget_specs:
        for _ in range(40):
            x1 = int(rng.integers(26, IMG_SIZE - bw - 4))
            y1 = int(rng.integers(18, IMG_SIZE - bh - 4))
            x2, y2 = x1 + bw, y1 + bh
            overlap = any(not (x2 + 4 < ox1 or x1 - 4 > ox2 or y2 + 4 < oy1 or y1 - 4 > oy2) for ox1, oy1, ox2, oy2 in slots)
            if not overlap:
                slots.append((x1, y1, x2, y2))
                outline = (60, 130, 220) if w_name == "input" else (30, 35, 45)
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=outline, width=1)
                if w_name == "input":
                    # Little magnifying glass dot inside search box
                    draw.ellipse([x1 + 3, y1 + 3, x1 + 7, y1 + 7], outline=(100, 110, 130))
                boxes[w_name] = (x1, y1, x2, y2)
                break

    task_id = int(rng.integers(0, 4))
    key_map = {0: "green", 1: "blue", 2: "red", 3: "input"}
    tx1, ty1, tx2, ty2 = boxes[key_map[task_id]]
    _, act_idx, is_destr = TASK_TEMPLATES[task_id]

    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # [3, 128, 128]
    cx = (tx1 + tx2) / (2.0 * IMG_SIZE)
    cy = (ty1 + ty2) / (2.0 * IMG_SIZE)
    box_norm = np.array([ty1 / IMG_SIZE, tx1 / IMG_SIZE, ty2 / IMG_SIZE, tx2 / IMG_SIZE], dtype=np.float32)

    return {
        "image": arr,
        "task_id": task_id,
        "action_idx": act_idx,
        "is_destructive": is_destr,
        "point": np.array([cx, cy], dtype=np.float32),
        "box": box_norm,
        "box_px": (tx1, ty1, tx2, ty2),
    }


def _make_gui_dataset(n: int, seed: int) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    samples = [_render_gui_sample(rng) for _ in range(n)]
    # Task instruction token IDs (4 distinct deterministic prompt sequences of len 6)
    task_toks = torch.tensor(
        [[10 + s["task_id"] * 10 + j for j in range(6)] for s in samples],
        dtype=torch.long,
        device=DEVICE,
    )
    return {
        "images": torch.tensor(np.stack([s["image"] for s in samples]), dtype=torch.float32, device=DEVICE),
        "text_ids": task_toks,
        "action_idx": torch.tensor([s["action_idx"] for s in samples], dtype=torch.long, device=DEVICE),
        "is_destructive": torch.tensor([s["is_destructive"] for s in samples], dtype=torch.long, device=DEVICE),
        "point": torch.tensor(np.stack([s["point"] for s in samples]), dtype=torch.float32, device=DEVICE),
        "box": torch.tensor(np.stack([s["box"] for s in samples]), dtype=torch.float32, device=DEVICE),
        "box_px": [s["box_px"] for s in samples],
    }


def _crop_and_remap(images: torch.Tensor, boxes_norm: torch.Tensor, crop_size: int = 64, expand: float = 2.4):
    """Differentiable/fast high-res crop around coarse predicted Box for Pass-2 CDG Zoom refinement."""
    B, _, H, W = images.shape
    crops = []
    meta = []
    for i in range(B):
        t, l, b, r = boxes_norm[i].tolist()
        cy, cx = 0.5 * (t + b), 0.5 * (l + r)
        half = max(0.5 * max(b - t, r - l) * expand, 0.16)
        y1 = max(0, int((cy - half) * H))
        x1 = max(0, int((cx - half) * W))
        y2 = min(H, max(y1 + 16, int((cy + half) * H)))
        x2 = min(W, max(x1 + 16, int((cx + half) * W)))
        patch = F.interpolate(images[i: i + 1, :, y1:y2, x1:x2], size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        crops.append(patch)
        meta.append((x1, y1, x2, y2))
    return torch.cat(crops, dim=0), meta


def _gaussian_heatmap_loss(hm_logits: torch.Tensor, gt_pt: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """KL divergence between predicted 2D spatial softmax heatmap and 2D Gaussian target centered at gt_pt."""
    B, H, W = hm_logits.shape
    xs = ((torch.arange(W, device=hm_logits.device, dtype=hm_logits.dtype) + 0.5) / W).view(1, 1, W)
    ys = ((torch.arange(H, device=hm_logits.device, dtype=hm_logits.dtype) + 0.5) / H).view(1, H, 1)
    gx = gt_pt[:, 0].view(B, 1, 1)
    gy = gt_pt[:, 1].view(B, 1, 1)
    tgt = torch.exp(-((xs - gx).pow(2) + (ys - gy).pow(2)) / (2.0 * sigma * sigma))
    tgt = tgt / tgt.sum(dim=(1, 2), keepdim=True).clamp_min(1e-7)
    log_p = F.log_softmax(hm_logits.reshape(B, H * W), dim=-1)
    return F.kl_div(log_p, tgt.reshape(B, H * W), reduction="batchmean")


def main():
    torch.manual_seed(42)
    print("Rendering realistic RGB multi-widget GUI datasets (128x128, tiny 14x12 px target buttons)...")
    train_ds = _make_gui_dataset(1600, seed=10)
    test_ds = _make_gui_dataset(300, seed=2026)

    # Compact patch_size=16 -> fast 8x8=64 patch tokens while fine CoordConv spatial_map stays crisp 32x32!
    cfg = OmniJevConfig(
        d_model=128, n_heads=4, d_ff=256, n_prelude_layers=2, n_coda_layers=1,
        image_patch_size=16, simcc_bins=64, box_dfl_bins=32,
    )
    model = OmniJevModel(cfg).to(DEVICE)
    evid_loss = EvidentialDirichletLoss(kl_weight=0.04)

    params = (
        list(model.bus.parameters())
        + list(model.backbone.parameters())
        + list(model.primitives.pooler.parameters())
        + [model.default_query]
        + list(model.cua.parameters())
        + list(model.primitives.point.parameters())
    )
    opt = torch.optim.AdamW(params, lr=2.5e-3, weight_decay=1e-4)

    N = train_ds["images"].size(0)
    for step in range(480):
        idx = torch.randint(0, N, (64,), device=DEVICE)
        imgs = train_ds["images"][idx]
        tids = train_ds["text_ids"][idx]
        gt_act = train_ds["action_idx"][idx]
        gt_des = train_ds["is_destructive"][idx]
        gt_pt = train_ds["point"][idx]
        gt_bx = train_ds["box"][idx]

        # Pass 1: Global coarse CUA forward pass
        enc = model.encode_state(ModalityBatch(text_ids=tids, images=imgs), deliberation_steps=0)
        q = model.query_vector(enc)
        cua = model.cua(enc["state_h"], enc["spatial_map"], q)

        # Proper Evidential Digamma Log-Loss + non-vanishing CE shaping on raw_scores
        from omnijev.v3_consolidation import evidential_log_loss
        l_act = evidential_log_loss(cua["action_alpha"], gt_act, kl_weight=0.02) + F.cross_entropy(cua["action_raw_scores"][:, :cfg.cua_num_actions], gt_act)
        l_des = evidential_log_loss(cua["is_destructive_alpha"], gt_des, kl_weight=0.02)
        l_hm = _gaussian_heatmap_loss(cua["point_heatmap_logits"], gt_pt, sigma=0.045)
        l_pt = F.smooth_l1_loss(cua["point"], gt_pt)
        l_bx = F.smooth_l1_loss(cua["box"], gt_bx)

        # Pass 2 (CDG Zoom training): crop around jittered box -> refine local 2D Gaussian heatmap + Point!
        jitter_box = (gt_bx + torch.randn_like(gt_bx) * 0.035).clamp(0.0, 1.0)
        crops, meta = _crop_and_remap(imgs, jitter_box, crop_size=64, expand=2.4)
        local_gt = torch.zeros_like(gt_pt)
        for i, (x1, y1, x2, y2) in enumerate(meta):
            gx, gy = gt_pt[i, 0].item() * IMG_SIZE, gt_pt[i, 1].item() * IMG_SIZE
            local_gt[i, 0] = max(0.0, min(1.0, (gx - x1) / max(x2 - x1, 1)))
            local_gt[i, 1] = max(0.0, min(1.0, (gy - y1) / max(y2 - y1, 1)))

        enc_z = model.encode_state(ModalityBatch(text_ids=tids, images=crops), deliberation_steps=0)
        q_z = model.query_vector(enc_z)
        pt_z = model.primitives.point(enc_z["spatial_map"], q_z)
        l_zoom_hm = _gaussian_heatmap_loss(pt_z["heatmap_logits"], local_gt, sigma=0.06)
        l_zoom_pt = F.smooth_l1_loss(pt_z["point"], local_gt)

        loss = 2.0 * l_act + 2.0 * l_des + 1.5 * l_hm + 8.0 * l_pt + 6.0 * l_bx + 1.5 * l_zoom_hm + 8.0 * l_zoom_pt
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    # Evaluate on 300 UNSEEN rendered RGB desktop layouts!
    model.eval()
    with torch.no_grad():
        imgs_t = test_ds["images"]
        tids_t = test_ds["text_ids"]
        gt_act_t = test_ds["action_idx"]
        gt_des_t = test_ds["is_destructive"]
        gt_pt_t = test_ds["point"]
        box_px_t = test_ds["box_px"]

        # Warmup + steady-state single-screenshot latency
        _ = model.cua(
            model.encode_state(ModalityBatch(text_ids=tids_t[:1], images=imgs_t[:1]), deliberation_steps=0)["state_h"],
            model.encode_state(ModalityBatch(text_ids=tids_t[:1], images=imgs_t[:1]), deliberation_steps=0)["spatial_map"],
            model.default_query,
        )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        enc_1 = model.encode_state(ModalityBatch(text_ids=tids_t[:1], images=imgs_t[:1]), deliberation_steps=0)
        _ = model.cua(enc_1["state_h"], enc_1["spatial_map"], model.query_vector(enc_1))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        pass1_ms = (time.perf_counter() - t0) * 1000.0

        # Full test set evaluation (batched)
        enc_t = model.encode_state(ModalityBatch(text_ids=tids_t, images=imgs_t), deliberation_steps=0)
        q_t = model.query_vector(enc_t)
        cua_t = model.cua(enc_t["state_h"], enc_t["spatial_map"], q_t)

        # Pass-2 CDG Zoom using Pass-1's PREDICTED Box (strictly zero ground-truth leakage!)
        crops_t, meta_t = _crop_and_remap(imgs_t, cua_t["box"], crop_size=64, expand=2.4)
        enc_zt = model.encode_state(ModalityBatch(text_ids=tids_t, images=crops_t), deliberation_steps=0)
        q_zt = model.query_vector(enc_zt)
        pt_zt = model.primitives.point(enc_zt["spatial_map"], q_zt)["point"]

        zoom_global_pt = torch.zeros_like(cua_t["point"])
        for i, (x1, y1, x2, y2) in enumerate(meta_t):
            zoom_global_pt[i, 0] = (x1 + pt_zt[i, 0].item() * (x2 - x1)) / IMG_SIZE
            zoom_global_pt[i, 1] = (y1 + pt_zt[i, 1].item() * (y2 - y1)) / IMG_SIZE

        # Compute exact pixel click hits inside tiny 14x12 px target bounding boxes!
        p1_hits, p2_hits = 0, 0
        p1_px_errs, p2_px_errs = [], []
        for i, (bx1, by1, bx2, by2) in enumerate(box_px_t):
            gx, gy = gt_pt_t[i, 0].item() * IMG_SIZE, gt_pt_t[i, 1].item() * IMG_SIZE
            x_1, y_1 = cua_t["point"][i, 0].item() * IMG_SIZE, cua_t["point"][i, 1].item() * IMG_SIZE
            x_2, y_2 = zoom_global_pt[i, 0].item() * IMG_SIZE, zoom_global_pt[i, 1].item() * IMG_SIZE

            p1_px_errs.append(float(np.hypot(x_1 - gx, y_1 - gy)))
            p2_px_errs.append(float(np.hypot(x_2 - gx, y_2 - gy)))
            if bx1 <= x_1 <= bx2 and by1 <= y_1 <= by2:
                p1_hits += 1
            if bx1 <= x_2 <= bx2 and by1 <= y_2 <= by2:
                p2_hits += 1

        act_acc = (cua_t["action_idx"] == gt_act_t).float().mean().item() * 100.0
        # Destructive Noul guardrail precision / recall / F1
        pred_des = (cua_t["is_destructive_prob"] >= 0.5).long()
        tp = int(((pred_des == 1) & (gt_des_t == 1)).sum().item())
        fp = int(((pred_des == 1) & (gt_des_t == 0)).sum().item())
        fn = int(((pred_des == 0) & (gt_des_t == 1)).sum().item())
        des_f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1) * 100.0

        # Selective S2 wakeup accuracy on type_text vs click
        gt_needs_s2 = (gt_act_t == 6)
        s2_wake_acc = ((cua_t["action_idx"] == 6) == gt_needs_s2).float().mean().item() * 100.0

    report = {
        "benchmark": "Rendered 128x128 RGB Multi-Widget GUI (14x12 px tiny buttons + destructive red dialogs + search box)",
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "test_screenshots": 300,
        "action_choice_acc_pct": round(act_acc, 2),
        "destructive_guardrail_noul_f1_pct": round(des_f1, 2),
        "selective_s2_type_wakeup_acc_pct": round(s2_wake_acc, 2),
        "pass1_coarse_click_hit_pct": round(100.0 * p1_hits / 300.0, 2),
        "pass1_mean_pixel_error": round(float(np.mean(p1_px_errs)), 2),
        "pass2_cdg_zoom_click_hit_pct": round(100.0 * p2_hits / 300.0, 2),
        "pass2_cdg_zoom_mean_pixel_error": round(float(np.mean(p2_px_errs)), 2),
        "single_screenshot_pass1_ms": round(pass1_ms, 2),
        "effective_actions_per_sec": round(1000.0 / max(pass1_ms, 0.1), 1),
    }
    print("\n[Real Rendered Pixel GUI Jev-CUA Results]\n", json.dumps(report, indent=2))
    out_path = ROOT / "experiments" / "v3_cua_pixel_gui_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Pixel GUI results to: {out_path}")


if __name__ == "__main__":
    main()
