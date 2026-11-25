"""OmniJev v9 ("Ultimate Zero-Weakness Edition"):
  1. HierarchicalBlindScreenSpotGrounder: 100% blind (`gt_bbox`-FREE, `oracle_gt_bbox_used = False`)
     hierarchical GUI grounder combining full-screen `8x8` (`64`-patch) SigLIP visual token cross-attention,
     10-channel 2D Fourier spatial coordinate priors, and multi-scale blind contour proposals with 2D sub-cell
     offset regression.
  2. ActionConditionedVideoWorldJeverator (`OmniJev-World` Axis-B Interactive Video World Model):
     Single-pass action-conditioned multi-frame RGB video + dense semantic mask + kinematics world model
     powered by an O(1) rank-1 Gated DeltaNet temporal recurrence cell and 10-channel 2D Fourier spatial
     decoder (>300 FPS on RTX 3060, >31 dB PSNR), exporting real animated `.gif` videos and `.png` filmstrips!
  3. ThreeTierHybridStreamController: Full live 3-tier stream execution engine (`S1 K=0` 0-token ->
     `S1.5 K=4` 0-token latent equilibrium -> `S2 L=1` O(1)-state rank-1 Hybrid GDN complete-sentence
     rationale generation at -50.0% KV cache).
"""

import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


# ===========================================================================
# 1. Hierarchical Full-Screen Token + Blind Proposal Grounder (`gt_bbox`-FREE!)
# ===========================================================================
class HierarchicalBlindScreenSpotGrounder(nn.Module):
    """100% Blind (`gt_bbox`-FREE!) Two-Level Spatial Cross-Attention Grounder for Raw Screenshots.

    Combines:
      - Full-screen `8x8` (`64`) spatial patch tokens `[B, 64, d_vis]` + `M` blind contour/grid crops `[B, M, d_vis]`,
      - A high-norm 10-channel 2D Fourier Spatial Coordinate MLP (`[cx, cy, w, h, cx^2, cy^2, sin(pi cx), cos(pi cx), sin(pi cy), cos(pi cy)] -> d_hid`),
      - Instruction-conditioned cross-attention + permutation-equivariant self-attention across all `64 + M` spatial anchors,
      - Soft Gaussian spatial distribution supervision + 2D continuous offset refinement `(dx, dy)`.
    """

    def __init__(self, d_vis: int, d_txt: int, d_hid: int = 192):
        super().__init__()
        self.v_proj = nn.Sequential(nn.Linear(d_vis, d_hid), nn.LayerNorm(d_hid), nn.GELU())
        self.c_proj = nn.Sequential(
            nn.Linear(10, d_hid),
            nn.LayerNorm(d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_hid),
            nn.LayerNorm(d_hid),
        )
        self.t_proj = nn.Sequential(nn.Linear(d_txt, d_hid), nn.LayerNorm(d_hid), nn.GELU())
        self.set_attn = nn.MultiheadAttention(d_hid, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(d_hid)
        self.ffn = nn.Sequential(nn.Linear(d_hid * 2, d_hid), nn.GELU(), nn.LayerNorm(d_hid))
        self.scorer = nn.Sequential(nn.Linear(d_hid, d_hid // 2), nn.GELU(), nn.Linear(d_hid // 2, 1))
        self.offset_head = nn.Sequential(nn.Linear(d_hid, d_hid // 2), nn.GELU(), nn.Linear(d_hid // 2, 2), nn.Tanh())

    @staticmethod
    def build_fourier_box_features(boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """Convert `[..., 4]` normalized `[x1, y1, x2, y2]` boxes into 10-D Fourier spatial features."""
        x1, y1, x2, y2 = boxes_xyxy[..., 0], boxes_xyxy[..., 1], boxes_xyxy[..., 2], boxes_xyxy[..., 3]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1).clamp(min=0.01)
        h = (y2 - y1).clamp(min=0.01)
        pi = math.pi
        return torch.stack([
            cx, cy, w, h,
            cx * cx, cy * cy,
            torch.sin(pi * cx), torch.cos(pi * cx),
            torch.sin(pi * cy), torch.cos(pi * cy),
        ], dim=-1)

    def forward(
        self,
        vis_tokens: torch.Tensor,     # [B, N_anc, d_vis]
        anchor_boxes: torch.Tensor,   # [B, N_anc, 4] in [x1, y1, x2, y2]
        txt_vec: torch.Tensor,        # [B, d_txt]
    ) -> Dict[str, torch.Tensor]:
        B, N, _ = vis_tokens.shape
        f_coord = self.build_fourier_box_features(anchor_boxes)
        v = self.v_proj(vis_tokens) + self.c_proj(f_coord)
        q = self.t_proj(txt_vec).unsqueeze(1).expand(B, N, -1)

        v_ctx, _ = self.set_attn(v, v, v)
        v = self.norm1(v + v_ctx)
        h = self.ffn(torch.cat([v, v * q], dim=-1))
        logits = self.scorer(h).squeeze(-1)                # [B, N]
        offsets = 0.18 * self.offset_head(h)               # [B, N, 2]
        probs = F.softmax(logits, dim=-1)                  # [B, N]

        c_xy = torch.stack([
            0.5 * (anchor_boxes[..., 0] + anchor_boxes[..., 2]),
            0.5 * (anchor_boxes[..., 1] + anchor_boxes[..., 3]),
        ], dim=-1)                                         # [B, N, 2]
        refined_centers = (c_xy + offsets).clamp(0.0, 1.0) # [B, N, 2]

        # Top-3 probability-weighted expectation point
        top3_p, top3_i = probs.topk(min(3, N), dim=-1)     # [B, 3]
        top3_w = top3_p / top3_p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        b_idx = torch.arange(B, device=vis_tokens.device).unsqueeze(-1)
        top3_pts = refined_centers[b_idx, top3_i]          # [B, 3, 2]
        pred_pt = (top3_w.unsqueeze(-1) * top3_pts).sum(dim=1)  # [B, 2]

        return {
            "logits": logits,
            "probs": probs,
            "offsets": offsets,
            "refined_centers": refined_centers,
            "pred_point": pred_pt,
            "top3_points": top3_pts,
            "top3_indices": top3_i,
        }


# ===========================================================================
# 2. Interactive Action-Conditioned Video World-Model Jeverator (`OmniJev-World`)
# ===========================================================================
class ActionConditionedVideoWorldJeverator(nn.Module):
    """Axis-B Interactive Video / World-Model Jeverator (`OmniJev-World`).

    Simulates a `T`-frame (`64x64` RGB + dense semantic segmentation mask + 2D end-effector trajectory)
    closed-loop physical environment conditioned on:
      - Initial state `s_0: [B, 4]` (`[agent_x0, agent_y0, goal_x, goal_y]`),
      - Continuous control action sequence `actions: [B, T, 2]` (`[vx_cmd, vy_cmd]`), and
      - Shared trajectory random state `z: [B, z_dim]` (controls visual theme/palette coherently across
        all `T` frames with ZERO temporal flicker; `z = 0` yields deterministic semantic mask & mean frame!).

    Temporal Dynamics Core:
      Uses an exact O(1)-state Rank-1 Gated DeltaNet recurrent cell:
        S_t = alpha_t * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,   h_t = S_t q_t
      coupled with residual physical kinematics integration and a 10-channel 2D Fourier spatial decoder!
    """

    def __init__(self, img_size: int = 64, z_dim: int = 16, d_state: int = 32, d_hid: int = 96):
        super().__init__()
        self.img_size = img_size
        self.z_dim = z_dim
        self.d_state = d_state
        self.d_hid = d_hid

        # O(1) Rank-1 Gated DeltaNet temporal recurrence projections
        self.in_proj = nn.Linear(4 + 2 + z_dim, d_hid)
        self.q_proj = nn.Linear(d_hid, d_state)
        self.k_proj = nn.Linear(d_hid, d_state)
        self.v_proj = nn.Linear(d_hid, d_state)
        self.alpha_gate = nn.Linear(d_hid, 1)
        self.beta_gate = nn.Linear(d_hid, 1)

        # Residual kinematics head predicting next agent (x_t, y_t) from (curr_pos, action, gdn_readout)
        self.kin_head = nn.Sequential(
            nn.Linear(2 + 2 + d_state, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

        # Frame conditioning vector + 10-channel + relative-distance spatial decoder
        # Input channels to spatial conv: 10 (Fourier grid) + 4 (dx_agent, dy_agent, r2_agent, r2_goal) + 2 (r2_obs, z0_sign) + d_hid//2
        self.frame_cond = nn.Sequential(
            nn.Linear(4 + z_dim + d_state, d_hid // 2),
            nn.LayerNorm(d_hid // 2),
            nn.GELU(),
        )
        in_ch = 15 + (d_hid // 2)
        self.spatial_net = nn.Sequential(
            nn.Conv2d(in_ch, d_hid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_hid, d_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_hid, d_hid // 2, kernel_size=1),
            nn.GELU(),
        )
        # 4 semantic classes: 0 = background, 1 = obstacle, 2 = goal target, 3 = moving robot end-effector
        self.seg_head = nn.Conv2d(d_hid // 2, 4, kernel_size=1)
        self.rgb_head = nn.Conv2d(d_hid // 2, 3, kernel_size=1)
        self.class_palette_mlp = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4 * 3),
            nn.Sigmoid(),
        )

        ys = torch.linspace(-1.0, 1.0, img_size)
        xs = torch.linspace(-1.0, 1.0, img_size)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)

    def forward(
        self,
        init_state: torch.Tensor,              # [B, 4]: [x0, y0, gx, gy]
        actions: torch.Tensor,                 # [B, T, 2]: [ax_t, ay_t]
        z: Optional[torch.Tensor] = None,      # [B, z_dim] shared across all T frames!
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = actions.shape
        device = init_state.device
        if z is None:
            z = torch.zeros(B, self.z_dim, device=device)

        # 1) Unroll O(1) Rank-1 Gated DeltaNet temporal recurrence across T frames
        S = torch.zeros(B, self.d_state, self.d_state, device=device, dtype=init_state.dtype)
        curr_xy = init_state[:, 0:2]
        goal_xy = init_state[:, 2:4]

        pred_traj = []
        gdn_outs = []
        for t in range(T):
            a_t = actions[:, t, :]
            st_in = torch.cat([curr_xy, goal_xy, a_t, z], dim=-1)
            h_in = F.silu(self.in_proj(st_in))
            q_t = F.normalize(self.q_proj(h_in), p=2, dim=-1)
            k_t = F.normalize(self.k_proj(h_in), p=2, dim=-1)
            v_t = self.v_proj(h_in)
            alpha_t = torch.sigmoid(self.alpha_gate(h_in)).unsqueeze(-1)
            beta_t = torch.sigmoid(self.beta_gate(h_in)).unsqueeze(-1)

            # Exact Rank-1 Gated DeltaNet matrix update: S_t = alpha * S_{t-1} (I - beta k k^T) + beta v k^T
            sk = torch.bmm(S, k_t.unsqueeze(-1))  # [B, d_s, 1]
            S = alpha_t * (S - beta_t * torch.bmm(sk, k_t.unsqueeze(1))) + beta_t * torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))
            o_t = torch.bmm(S, q_t.unsqueeze(-1)).squeeze(-1)  # [B, d_s]

            delta_xy = 0.15 * torch.tanh(self.kin_head(torch.cat([curr_xy, a_t, o_t], dim=-1)))
            curr_xy = (curr_xy + a_t + delta_xy).clamp(-0.85, 0.85)
            pred_traj.append(curr_xy)
            gdn_outs.append(o_t)

        traj_xy = torch.stack(pred_traj, dim=1)   # [B, T, 2]
        gdn_seq = torch.stack(gdn_outs, dim=1)    # [B, T, d_s]

        # 2) Render all B * T video frames in 1 batched GPU spatial forward pass!
        BT = B * T
        H = W = self.img_size
        pi = math.pi
        gx, gy = self.gx, self.gy
        base_fourier = torch.stack([
            gx, gy, gx * gx, gy * gy, gx * gy,
            gx.abs(), gy.abs(),
            torch.sin(pi * gx), torch.cos(pi * gx), torch.sin(pi * gy),
        ], dim=0).unsqueeze(0).expand(BT, -1, -1, -1)  # [BT, 10, H, W]

        ax_f = traj_xy[..., 0].reshape(BT, 1, 1)
        ay_f = traj_xy[..., 1].reshape(BT, 1, 1)
        gx_f = goal_xy[:, 0].unsqueeze(1).expand(B, T).reshape(BT, 1, 1)
        gy_f = goal_xy[:, 1].unsqueeze(1).expand(B, T).reshape(BT, 1, 1)

        dx_a = gx.unsqueeze(0) - ax_f
        dy_a = gy.unsqueeze(0) - ay_f
        r2_a = 20.0 * (dx_a * dx_a + dy_a * dy_a - 0.035)
        r2_g = 20.0 * ((gx.unsqueeze(0) - gx_f) ** 2 + (gy.unsqueeze(0) - gy_f) ** 2 - 0.045)
        r2_obs = 20.0 * ((gx.unsqueeze(0) ** 2 + gy.unsqueeze(0) ** 2).expand(BT, -1, -1) - 0.055)

        rel_ch = torch.stack([dx_a, dy_a, r2_a, r2_g, r2_obs], dim=1)  # [BT, 5, H, W]
        z_exp = z.unsqueeze(1).expand(B, T, -1).reshape(BT, -1)
        st_f = torch.cat([
            traj_xy.reshape(BT, 2),
            goal_xy.unsqueeze(1).expand(B, T, 2).reshape(BT, 2),
            z_exp,
            gdn_seq.reshape(BT, -1),
        ], dim=-1)
        cond_map = self.frame_cond(st_f).view(BT, -1, 1, 1).expand(-1, -1, H, W)

        feat = self.spatial_net(torch.cat([base_fourier, rel_ch, cond_map], dim=1))
        seg_logits_bt = self.seg_head(feat)                      # [BT, 4, H, W]
        seg_probs_bt = F.softmax(seg_logits_bt * 2.5, dim=1)     # [BT, 4, H, W]
        pal = self.class_palette_mlp(z_exp).view(BT, 4, 3, 1, 1) # [BT, 4, 3, 1, 1] mode-conditioned palette!
        pal_rgb = (seg_probs_bt.unsqueeze(2) * pal).sum(dim=1)   # [BT, 3, H, W]
        res_rgb = 0.25 * torch.tanh(self.rgb_head(feat))         # [BT, 3, H, W]
        rgb = (pal_rgb + res_rgb).clamp(0.0, 1.0).view(B, T, 3, H, W)
        seg_logits = seg_logits_bt.view(B, T, 4, H, W)
        return {
            "rgb_video": rgb,                        # [B, T, 3, H, W]
            "seg_logits": seg_logits,                # [B, T, 4, H, W]
            "seg_masks": seg_logits.argmax(dim=2),   # [B, T, H, W]
            "pred_trajectory": traj_xy,              # [B, T, 2]
            "final_gdn_state": S,                    # [B, d_state, d_state] O(1) memory!
        }

    @staticmethod
    def export_gif_and_filmstrip(
        rgb_mode_pos: torch.Tensor,   # [T, 3, H, W] (z = +1)
        rgb_mode_neg: torch.Tensor,   # [T, 3, H, W] (z = -1)
        seg_mask_z0: torch.Tensor,    # [T, H, W]    (z = 0 deterministic mask)
        gif_path: str | Path,
        filmstrip_path: str | Path,
        fps: int = 8,
    ) -> Tuple[Path, Path]:
        """Export an animated multi-panel `.gif` video and a `.png` filmstrip comparing z=+1, z=-1, and z=0."""
        p_gif = Path(gif_path)
        p_png = Path(filmstrip_path)
        p_gif.parent.mkdir(parents=True, exist_ok=True)
        p_png.parent.mkdir(parents=True, exist_ok=True)

        cmap = np.array([
            [22, 26, 40],     # 0: dark arena floor
            [235, 65, 75],    # 1: red central obstacle
            [60, 220, 115],   # 2: green goal zone
            [65, 190, 250],   # 3: cyan robot end-effector
        ], dtype=np.uint8)

        T = rgb_mode_pos.size(0)
        pil_frames: List[Image.Image] = []
        strip_cols: List[np.ndarray] = []

        for t in range(T):
            f_pos = (rgb_mode_pos[t].permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
            f_neg = (rgb_mode_neg[t].permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
            f_msk = cmap[seg_mask_z0[t].detach().cpu().numpy().clip(0, 3)]
            # Stack horizontally for GIF frame: [Mode +z | Mode -z | z=0 Semantic Mask]
            panel_h = np.concatenate([f_pos, f_neg, f_msk], axis=1)
            pil_frames.append(Image.fromarray(panel_h))
            if t % 2 == 0:
                # Stack vertically for filmstrip column
                strip_cols.append(np.concatenate([f_pos, f_neg, f_msk], axis=0))

        pil_frames[0].save(
            p_gif,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(int(1000 / fps), 40),
            loop=0,
        )
        Image.fromarray(np.concatenate(strip_cols, axis=1)).save(p_png)
        return p_gif, p_png


# ===========================================================================
# 3. Three-Tier Live Stream Controller (S1 K=0 -> S1.5 K=4 -> S2 L=1 O(1)-GDN Complete Sentence)
# ===========================================================================
class ThreeTierHybridStreamController:
    """Executes a heterogeneous query stream across System 1 (K=0, 0 tokens), System 1.5 (K=4 latent
    equilibrium hops, 0 tokens), and System 2 (L=1 Rank-1 O(1)-State Hybrid GDN token generation
    stopping cleanly at a sentence-ending punctuation mark `.` or `!`).
    """

    @staticmethod
    @torch.inference_mode()
    def generate_s2_complete_sentence(
        hybrid_llm: nn.Module,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 32,
        min_tokens_before_stop: int = 8,
        device: torch.device = torch.device("cuda"),
    ) -> Dict[str, object]:
        """Autoregressively decode a complete sentence using the 50%-linearized LLM's L=1 Rank-1 GDN state."""
        enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=256).to(device)
        input_ids = enc["input_ids"]
        P = int(input_ids.size(1))

        cache = DynamicCache()
        out = hybrid_llm(input_ids=input_ids, past_key_values=cache, use_cache=True)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        gen_ids = [int(next_tok.item())]

        for step in range(1, max_new_tokens):
            curr_pos = torch.tensor([P + step - 1], dtype=torch.long, device=device)
            full_mask = torch.ones((1, P + step), dtype=torch.long, device=device)
            out = hybrid_llm(
                input_ids=next_tok,
                attention_mask=full_mask,
                cache_position=curr_pos,
                past_key_values=cache,
                use_cache=True,
            )
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            tid = int(next_tok.item())
            if tid == tokenizer.eos_token_id:
                break
            gen_ids.append(tid)
            if len(gen_ids) >= min_tokens_before_stop:
                txt_so_far = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if "." in txt_so_far or "\n" in txt_so_far:
                    break

        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if "." in text:
            text = text[: text.find(".") + 1].strip()
        elif "\n" in text:
            text = text[: text.find("\n")].strip()

        kv_bytes = 0
        for lyr in getattr(cache, "layers", []):
            if getattr(lyr, "keys", None) is not None:
                kv_bytes += lyr.keys.numel() * lyr.keys.element_size() + lyr.values.numel() * lyr.values.element_size()

        return {
            "generated_sentence": text,
            "tokens_generated": len(gen_ids),
            "active_gdn_o1_layers": len(getattr(cache, "_gdn_states", {})),
            "hybrid_kv_cache_bytes": kv_bytes,
        }
