"""Universal Multimodal Input Bus for OmniJev.

Encodes heterogeneous domain inputs (Text/JSON/Code, Images/UI/Video, Audio,
Proprioception, 3D Points, and Graph/Table entities) into a unified latent token
stream plus spatial/structural side-channels (2D feature pyramid, proprio skip).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OmniJevConfig


MODALITY_IDS = {
    "text": 0,
    "image": 1,
    "audio": 2,
    "proprio": 3,
    "point3d": 4,
    "graph": 5,
    "action_prefix": 6,
}


@dataclass
class ModalityBatch:
    """Container for any combination of multimodal inputs in a single pass."""

    text_ids: Optional[torch.Tensor] = None          # [B, L_t] long
    images: Optional[torch.Tensor] = None            # [B, C, H, W] float
    audio_mels: Optional[torch.Tensor] = None        # [B, T_a, F_mel] float
    proprio: Optional[torch.Tensor] = None           # [B, D_prop] or [B, T_p, D_prop] float
    points3d: Optional[torch.Tensor] = None          # [B, N_pts, D_3d] float
    graph_nodes: Optional[torch.Tensor] = None       # [B, N_nodes, D_g] float
    action_prefix: Optional[torch.Tensor] = None     # [B, L_pre, D_act] float (for RTC)


class SpatialPatchEncoder(nn.Module):
    """ViT-style patch stem emitting compact patch tokens + CoordConv-enriched high-res 2D feature pyramid."""

    def __init__(self, in_channels: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        mid = max(d_model // 2, 32)
        # Fine stem (stride 4) keeps high spatial resolution for tiny UI buttons; coarse stem merges to patch_size
        s1 = min(4, max(patch_size // 2, 1))
        s2 = max(patch_size // s1, 1)
        self.fine_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, mid), mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, stride=max(s1 // 2, 1), padding=1),
            nn.GroupNorm(min(8, mid), mid),
            nn.GELU(),
        )
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(mid, d_model, kernel_size=s2, stride=s2),
            nn.GroupNorm(min(8, d_model), d_model),
            nn.GELU(),
        )
        # CoordConv fusion on high-res 2D feature map so every pixel knows BOTH appearance & metric (x, y)!
        self.fine_coord_fuse = nn.Sequential(
            nn.Conv2d(mid + 2, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, d_model), d_model),
            nn.GELU(),
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (patch_tokens [B, H_p*W_p, D], spatial_map [B, D, H_f, W_f])."""
        fine = self.fine_conv(images)       # [B, mid, H_f, W_f]
        coarse = self.coarse_conv(fine)     # [B, D, H_p, W_p]
        B, D, Hp, Wp = coarse.shape

        # 2D coordinates for coarse patch tokens
        ys = torch.linspace(0.0, 1.0, Hp, device=images.device, dtype=images.dtype)
        xs = torch.linspace(0.0, 1.0, Wp, device=images.device, dtype=images.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx, gy], dim=-1).reshape(1, Hp * Wp, 2)  # [1, L_v, 2]
        tokens = coarse.flatten(2).transpose(1, 2) + self.pos_mlp(coords)

        # CoordConv 2D metric grid concatenated directly onto high-res fine pyramid
        _, _, Hf, Wf = fine.shape
        yf = ((torch.arange(Hf, device=images.device, dtype=images.dtype) + 0.5) / Hf).view(1, 1, Hf, 1).expand(B, 1, Hf, Wf)
        xf = ((torch.arange(Wf, device=images.device, dtype=images.dtype) + 0.5) / Wf).view(1, 1, 1, Wf).expand(B, 1, Hf, Wf)
        fine_with_xy = torch.cat([fine, xf, yf], dim=1)  # [B, mid+2, H_f, W_f]

        coarse_up = F.interpolate(coarse, size=(Hf, Wf), mode="bilinear", align_corners=False)
        spatial_map = self.fine_coord_fuse(fine_with_xy) + coarse_up  # [B, D, H_f, W_f]
        return tokens, spatial_map


class UniversalMultimodalBus(nn.Module):
    """Projects all modalities into a unified token stream + side channels."""

    def __init__(self, cfg: OmniJevConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        self.modality_embed = nn.Embedding(len(MODALITY_IDS), D)
        self.text_embed = nn.Embedding(cfg.vocab_size, D)
        self.vision_encoder = SpatialPatchEncoder(cfg.image_channels, D, cfg.image_patch_size)
        self.audio_encoder = nn.Sequential(
            nn.Linear(cfg.audio_mels, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(cfg.proprio_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.point3d_encoder = nn.Sequential(
            nn.Linear(cfg.point3d_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(cfg.graph_feat_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.action_prefix_encoder = nn.Sequential(
            nn.Linear(cfg.vla_action_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.out_norm = nn.LayerNorm(D)

    def _tag(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        mid = torch.tensor(MODALITY_IDS[modality], device=x.device, dtype=torch.long)
        return x + self.modality_embed(mid).view(1, 1, -1)

    def forward(self, batch: ModalityBatch) -> Dict[str, Optional[torch.Tensor]]:
        """Encode present modalities and track exact slice boundaries for downstream heads."""
        chunks = []
        slices: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        spatial_map: Optional[torch.Tensor] = None
        proprio_raw: Optional[torch.Tensor] = None

        if batch.text_ids is not None:
            t_emb = self._tag(self.text_embed(batch.text_ids), "text")
            chunks.append(t_emb)
            slices["text"] = (cursor, cursor + t_emb.size(1))
            cursor += t_emb.size(1)

        if batch.images is not None:
            v_toks, spatial_map = self.vision_encoder(batch.images)
            v_toks = self._tag(v_toks, "image")
            chunks.append(v_toks)
            slices["image"] = (cursor, cursor + v_toks.size(1))
            cursor += v_toks.size(1)

        if batch.audio_mels is not None:
            a_toks = self._tag(self.audio_encoder(batch.audio_mels), "audio")
            chunks.append(a_toks)
            slices["audio"] = (cursor, cursor + a_toks.size(1))
            cursor += a_toks.size(1)

        if batch.proprio is not None:
            p_in = batch.proprio
            if p_in.dim() == 2:
                proprio_raw = p_in
                p_in = p_in.unsqueeze(1)
            else:
                proprio_raw = p_in[:, -1, :]
            p_toks = self._tag(self.proprio_encoder(p_in), "proprio")
            chunks.append(p_toks)
            slices["proprio"] = (cursor, cursor + p_toks.size(1))
            cursor += p_toks.size(1)

        if batch.points3d is not None:
            pt3 = self._tag(self.point3d_encoder(batch.points3d), "point3d")
            chunks.append(pt3)
            slices["point3d"] = (cursor, cursor + pt3.size(1))
            cursor += pt3.size(1)

        if batch.graph_nodes is not None:
            g_toks = self._tag(self.graph_encoder(batch.graph_nodes), "graph")
            chunks.append(g_toks)
            slices["graph"] = (cursor, cursor + g_toks.size(1))
            cursor += g_toks.size(1)

        if batch.action_prefix is not None:
            ap_toks = self._tag(self.action_prefix_encoder(batch.action_prefix), "action_prefix")
            chunks.append(ap_toks)
            slices["action_prefix"] = (cursor, cursor + ap_toks.size(1))
            cursor += ap_toks.size(1)

        if not chunks:
            raise ValueError("ModalityBatch must contain at least one non-None input modality.")

        tokens = self.out_norm(torch.cat(chunks, dim=1))
        return {
            "tokens": tokens,
            "spatial_map": spatial_map,
            "proprio_raw": proprio_raw,
            "slices": slices,
        }
