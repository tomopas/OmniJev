"""OmniJev v3 Pretrained Hugging Face Foundation Model Bridge.

Wraps ANY pretrained causal LLM (Qwen2.5, Qwen3, Gemma 4, Llama, etc.) into a live
OmniJev v3 System 1 / 1.5 Decision Engine in ONE prefill (zero autoregressive tokens!):
  - Exact O(1) permutation-invariant OrderInvariantChoiceHead + OOI Sink
  - Zero-hallucination BiaffineSpanHead extractive reader over real token sequences
  - Calibrated Noul & Ordinal Score heads
  - System 1.5 ScratchpadEquilibriumCore attached directly to pretrained hidden states
  - Online Adaptive Conformal Inference (ACI)
  - Two-timescale S2->S1 consolidation (Fast-Weight Delta Bank + KL-Anchored Micro-LoRA Sleep!)
  - Built-in apples-to-apples baselines (Autoregressive JSON generation & Vanilla Prompt-Logit AnyJev-0)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import (
    BiaffineSpanHead,
    NoulHead,
    OrderInvariantChoiceHead,
    OrdinalScoreHead,
    QueryCrossPooler,
)
from .v2_frontier import (
    FastWeightSkillConsolidator,
    OnlineAdaptiveConformal,
    ScratchpadEquilibriumCore,
)
from .v3_consolidation import MicroLoRASleepConsolidator


class PretrainedOmniJevBridge(nn.Module):
    """Drop-in OmniJev v3 System 1 / 1.5 adapter on top of a frozen/LoRA Hugging Face LLM."""

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer,
        d_jev: int = 256,
        n_heads: int = 4,
        max_classes: int = 16,
        freeze_backbone: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.d_llm = int(hf_model.config.hidden_size)
        self.d_jev = d_jev
        self.device_ref = device or next(hf_model.parameters()).device

        if freeze_backbone:
            for p in self.hf_model.parameters():
                p.requires_grad_(False)
            self.hf_model.eval()

        # Bottleneck projector from pretrained LLM hidden space -> FP32 calibrated OmniJev space
        self.state_proj = nn.Sequential(
            nn.Linear(self.d_llm, d_jev),
            nn.LayerNorm(d_jev),
            nn.GELU(),
            nn.Linear(d_jev, d_jev),
            nn.LayerNorm(d_jev),
        )
        self.opt_proj = nn.Sequential(
            nn.Linear(self.d_llm, d_jev),
            nn.LayerNorm(d_jev),
            nn.GELU(),
            nn.Linear(d_jev, d_jev),
            nn.LayerNorm(d_jev),
        )

        # System 1.5 Scratchpad Equilibrium Core
        self.scratchpad = ScratchpadEquilibriumCore(d_jev, n_heads=n_heads, d_ff=d_jev * 2, num_registers=4)
        self.pooler = QueryCrossPooler(d_jev, n_heads=n_heads)
        self.default_q = nn.Parameter(torch.randn(1, d_jev) * 0.02)

        # Calibrated Axis-A Primitives in FP32
        self.choice_head = OrderInvariantChoiceHead(d_jev, n_heads=n_heads, set_layers=2)
        self.span_head = BiaffineSpanHead(d_jev)
        self.noul_head = NoulHead(d_jev)
        self.score_head = OrdinalScoreHead(d_jev, num_levels=5)

        # Axis-D Live Calibration & Two-Timescale S2->S1 Consolidation
        # CRITICAL INSIGHT: Key OOD, Fast-Weights, and Micro-LoRA Sleep directly off the UNCOLLAPSED
        # pretrained LLM manifold (d_llm) so closed-world supervision never crushes unseen class subspaces!
        self.online_aci = OnlineAdaptiveConformal(alpha_target=0.10, gamma_aci=0.04)
        self.fast_weights = FastWeightSkillConsolidator(
            d_model=self.d_llm, max_classes=max_classes, familiarity_threshold=0.52, evidence_gain_multiplier=3.5
        )
        self.sleep_consolidator = MicroLoRASleepConsolidator(d_model=self.d_llm, max_classes=max_classes, lora_rank=16)

        # Keep adapter/head weights in FP32 for calibration preservation (Axis E rule!)
        for name, mod in self.named_children():
            if name != "hf_model":
                mod.to(device=self.device_ref, dtype=torch.float32)
        self.default_q.data = self.default_q.data.to(device=self.device_ref, dtype=torch.float32)

    @torch.no_grad()
    def _prefill_hf(self, texts: List[str], max_length: int = 160) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Single forward prefill of frozen HF LLM (zero decoding steps!)."""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device_ref)
        # Access base transformer directly (skips LM head projection for extra speed!)
        base = getattr(self.hf_model, "model", self.hf_model)
        out = base(**enc, use_cache=False)
        hidden = out.last_hidden_state.to(torch.float32)  # [B, L, d_llm]
        mask = enc["attention_mask"].to(torch.float32)    # [B, L]
        return hidden, mask, enc

    def encode_texts(self, texts: List[str], max_length: int = 160, deliberation_steps: int = 0) -> Dict[str, object]:
        raw_h, mask, enc = self._prefill_hf(texts, max_length=max_length)
        h = self.state_proj(raw_h)                        # [B, L, d_jev]
        # Masked mean pooling over real content tokens
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom

        if deliberation_steps > 0:
            eq = self.scratchpad(h, num_steps=deliberation_steps, init_seed=pooled)
            q_vec = eq["pooled"]
        else:
            q_vec = self.pooler(h, self.default_q.expand(h.size(0), -1) + pooled)

        return {
            "state_h": h,
            "pooled": pooled,
            "q_vec": q_vec,
            "attention_mask": mask,
            "input_ids": enc["input_ids"],
        }

    def encode_isolated_option_texts(self, option_texts: List[str], max_length: int = 48) -> torch.Tensor:
        """Encode each candidate option independently starting at RoPE position 0 -> exact O(1) order invariance!"""
        raw_h, mask, _ = self._prefill_hf(option_texts, max_length=max_length)
        h = self.opt_proj(raw_h)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h * mask.unsqueeze(-1)).sum(dim=1)        # [C, d_jev]

    def decide_choice(
        self,
        texts: List[str],
        option_texts: List[str],
        deliberation_steps: int = 0,
        include_ooi_sink: bool = True,
        cached_opt_embs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode_texts(texts, deliberation_steps=deliberation_steps)
        opt_embs = cached_opt_embs if cached_opt_embs is not None else self.encode_isolated_option_texts(option_texts)
        B = len(texts)
        opt_batch = opt_embs.unsqueeze(0).expand(B, -1, -1)
        out = self.choice_head(enc["state_h"], enc["q_vec"], opt_batch, include_ooi_sink=include_ooi_sink)
        out["q_vec"] = enc["q_vec"]
        out["pooled"] = enc["pooled"]
        return out

    def decide_span(self, contexts: List[str], questions: List[str], max_length: int = 192) -> Dict[str, object]:
        """Zero-hallucination extractive Biaffine Span over real English context + question."""
        ctx_enc = self.encode_texts(contexts, max_length=max_length)
        q_enc = self.encode_texts(questions, max_length=48)
        sp = self.span_head(ctx_enc["state_h"], q_enc["pooled"], max_span_len=24, top_k=1)

        extracted_texts = []
        input_ids = ctx_enc["input_ids"]
        for i in range(len(contexts)):
            s = int(sp["start_idx"][i].item())
            e = int(sp["end_idx"][i].item())
            toks = input_ids[i, s: e + 1]
            extracted_texts.append(self.tokenizer.decode(toks, skip_special_tokens=True).strip())
        sp["extracted_texts"] = extracted_texts
        return sp

    @torch.no_grad()
    def vanilla_prompt_choice_probs(self, text: str, option_texts: List[str]) -> torch.Tensor:
        """Baseline: Vanilla prompt-concatenated next-token logit readout on the SAME HF LLM (shows order bias!)."""
        letters = [chr(ord("A") + i) for i in range(len(option_texts))]
        opts_str = "\n".join(f"{l}) {o}" for l, o in zip(letters, option_texts))
        prompt = (
            f"Classify the following text into exactly one category.\n"
            f"Text: {text}\n"
            f"Options:\n{opts_str}\n"
            f"Answer with only the single letter ({', '.join(letters)}): "
        )
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device_ref)
        logits = self.hf_model(**enc, use_cache=False).logits[0, -1, :]  # [vocab]
        letter_ids = [self.tokenizer.encode(l, add_special_tokens=False)[-1] for l in letters]
        sel = logits[letter_ids].to(torch.float32)
        return F.softmax(sel, dim=-1)

    @torch.no_grad()
    def autoregressive_json_classify(self, text: str, option_texts: List[str], max_new_tokens: int = 32) -> str:
        """Baseline: Autoregressive JSON text generation on the SAME HF LLM (shows autoregressive latency tax!)."""
        prompt = (
            f'Classify the text into one of {option_texts}. '
            f'Reply ONLY as valid JSON: {{"category": "<label>", "confidence": 0.95}}\n'
            f'Text: {text}\nJSON:'
        )
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=224).to(self.device_ref)
        gen = self.hf_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(gen[0, enc["input_ids"].size(1):], skip_special_tokens=True)
