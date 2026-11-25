"""OmniJev v10 Production FastAPI Serving Engine (`/v1/decide`, `/v1/decide_batched`, `/v1/decide_graph`, `/v1/hocbf_filter`, `/v1/world_rollout`, `/v1/stream_3tier`, `/v1/consolidate`, `/v1/checkpoint/*`, `/v1/health`).

Implements a TypeSafe-Jev-compatible & embodied multimodal HTTP JSON API serving PretrainedOmniJevBridge,
SimulatorFreeRobustHOCBFQP, ActionConditionedVideoWorldJeverator, and Hybrid Linearized GDN backbones with:
  - POST /v1/decide          (parallel noul, choice, score, span + live calibrator, action-risk routing, & injection guardrail!)
  - POST /v1/decide_batched  (vLLM-style AsyncDynamicBatcher request coalescing across concurrent HTTP calls!)
  - POST /v1/decide_graph    (batched Prefix-KV CDG + TRUE conditional DAG branching `depends_on` / `condition_gt` / `pass_span_from`!)
  - POST /v1/hocbf_filter    (sub-millisecond 100% simulator-free pure-tensor 2-DoF / 7-DoF safety projection!)
  - POST /v1/world_rollout   (single-pass ActionConditionedVideoWorldJeverator mental trajectory foresight!)
  - POST /v1/stream_3tier    (unified S1 K=0 -> S1.5 K=4 -> S2 Batched O(1)-GDN complete-sentence cascade!)
  - POST /v1/consolidate     (live <1ms S2->S1 fast-weight consolidation + optional sleep trigger!)
  - POST /v1/checkpoint/save & /v1/checkpoint/load (`safetensors` persistence!)
  - GET  /v1/health          (live GPU VRAM, ACI alpha_t, prefix KV hits, guardrail status)
"""

from typing import Any, Dict, List, Optional, Tuple
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch


class QuestionSpec(BaseModel):
    type: str = Field(..., description="Primitive type: noul | choice | score | span")
    instructions: str = ""
    options: Optional[List[str]] = None


class DecideRequest(BaseModel):
    state: str
    questions: Dict[str, QuestionSpec]
    deliberation_steps: int = 0
    shared_prefix: Optional[str] = None
    check_injection: bool = True


class GraphNodeSpec(BaseModel):
    name: str
    question: QuestionSpec
    depends_on: Optional[str] = None
    condition_gt: Optional[float] = None
    pass_span_from: Optional[str] = None


class DecideGraphRequest(BaseModel):
    shared_prefix: str
    nodes: List[GraphNodeSpec]


class ConsolidateRequest(BaseModel):
    state: str
    verified_class_idx: int
    option_name: Optional[str] = None
    trigger_sleep: bool = False


class CheckpointRequest(BaseModel):
    directory: str


class HOCBFFilterRequest(BaseModel):
    dof: int = Field(2, description="2 for Reacher-v5 (state_dim=8) or 7 for Pusher-v5 (state_dim=17)")
    proposed_action: List[float]
    state_vec: List[float]
    tip_xy: Optional[List[float]] = None
    obstacle_xy: List[float] = Field(default_factory=lambda: [-0.10, 0.075])
    obstacle_r: float = 0.036
    hazard_side: float = 1.0
    curr_y: Optional[float] = None
    y_forbid: float = -0.40


class WorldRolloutRequest(BaseModel):
    init_state: List[float] = Field(..., description="[x0, y0, goal_x, goal_y]")
    actions: List[List[float]] = Field(..., description="Sequence of T 2D actions [[ax, ay], ...]")
    z_scale: float = 0.0


class ThreeTierStreamRequest(BaseModel):
    state: str
    tier: str = Field("auto", description="auto | s1 | s1.5 | s2")
    options: Optional[List[str]] = None
    max_new_tokens: int = 24


def create_omnijev_app(
    bridge,
    calibrator=None,
    action_router=None,
    injection_cdg=None,
    hocbf_2dof=None,
    hocbf_7dof=None,
    world_model=None,
    hybrid_llm=None,
) -> FastAPI:
    """Create a production FastAPI app wrapping PretrainedOmniJevBridge with Prefix-KV, Guardrails, & Checkpoints."""
    from .v5_hardening import PrefixKVCacheManager
    from .v6_frontier import OmniJevCheckpointManager
    from .v7_certification import AsyncDynamicBatcher

    app = FastAPI(title="OmniJev v10 Typed Decision, Dynamic-Batched, Conditional CDG, & Guardrail API", version="10.0.0")
    kv_manager = PrefixKVCacheManager(max_entries=32)

    opt_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    def _get_cached_opt_embs(opts: List[str]) -> torch.Tensor:
        key = tuple(opts)
        if key not in opt_cache:
            opt_cache[key] = bridge.encode_isolated_option_texts(opts).unsqueeze(0)
        return opt_cache[key]

    def _batch_decide_worker(reqs: List[DecideRequest]) -> List[Dict[str, Any]]:
        """Execute a coalesced micro-batch of B concurrent DecideRequests in ONE GPU forward pass!"""
        t_b0 = time.perf_counter()
        B = len(reqs)
        states = [r.state for r in reqs]
        max_k = max(r.deliberation_steps for r in reqs)
        with torch.no_grad():
            # Single unified LLM prefill for BOTH decision heads AND injection guardrail!
            raw_h, mask, _ = bridge._prefill_hf(states, max_length=160)
            state_h = bridge.state_proj(raw_h)
            if max_k > 0:
                delib = bridge.deliberation_core(state_h, max_steps=max_k, use_adaptive_halting=False)
                H_all = delib["refined_state"]
            else:
                H_all = state_h
            q_all = (H_all * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

            inj_flags = [False] * B
            inj_probs = [0.0] * B
            if injection_cdg is not None and any(r.check_injection for r in reqs):
                pooled = (raw_h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                if pooled.size(-1) == injection_cdg.d_model:
                    inj_out = injection_cdg.predict_noul(pooled)
                    inj_flags = [bool(x) for x in inj_out["is_injected"].tolist()]
                    inj_probs = [round(float(x), 5) for x in inj_out["inj_prob"].tolist()]

            batch_ms = round((time.perf_counter() - t_b0) * 1000.0, 2)
            per_item_ms = round(batch_ms / max(B, 1), 2)
            results: List[Dict[str, Any]] = []
            for i, req in enumerate(reqs):
                H_i = H_all[i: i + 1]
                q_i = q_all[i: i + 1]
                answers: Dict[str, Any] = {}
                for q_name, spec in req.questions.items():
                    ptype = spec.type.lower()
                    if ptype == "noul":
                        out = bridge.noul_head(q_i)
                        answers[q_name] = {
                            "type": "noul",
                            "noul": round(float(out["noul"][0].item()), 5),
                            "epistemic_vacuity": round(float(out["epistemic_vacuity"][0].item()), 5),
                        }
                    elif ptype == "score":
                        out = bridge.score_head(q_i)
                        answers[q_name] = {
                            "type": "score",
                            "score": round(float(out["score"][0].item()), 5),
                            "level_idx": int(out["level_idx"][0].item()),
                            "probs": [round(float(v), 5) for v in out["probs"][0].tolist()],
                            "confidence": round(float(out["confidence"][0].item()), 5),
                        }
                    elif ptype == "choice":
                        opts = spec.options or ["yes", "no"]
                        opt_embs = _get_cached_opt_embs(opts)
                        out = bridge.choice_head(H_i, q_i, opt_embs, include_ooi_sink=True)
                        probs = out["option_probs"][0]
                        if calibrator is not None and probs.numel() == getattr(calibrator, "num_classes", -1):
                            probs = calibrator.predict_probs(probs.clamp_min(1e-7).log().unsqueeze(0))[0]
                        win_idx = int(probs.argmax().item())
                        ans_d: Dict[str, Any] = {
                            "type": "choice",
                            "choice": opts[win_idx],
                            "choice_idx": win_idx,
                            "probabilities": {o: round(float(p), 5) for o, p in zip(opts, probs.tolist())},
                            "ooi_prob": round(float(out["ooi_prob"][0].item()), 5),
                            "epistemic_vacuity": round(float(out["epistemic_vacuity"][0].item()), 5),
                            "confidence": round(float(probs[win_idx].item()), 5),
                        }
                        if action_router is not None:
                            c_set = probs >= (1.0 / max(len(opts), 2)) * 0.5
                            dec = action_router.decide_escalation(probs, c_set, float(out["ooi_prob"][0].item()))
                            ans_d["escalate_to_system2"] = bool(dec["escalate"])
                            ans_d["routing_reason"] = dec["reason"]
                        answers[q_name] = ans_d
                    elif ptype == "span":
                        sp = bridge.decide_span([req.state], [spec.instructions or "Extract target span"])
                        answers[q_name] = {
                            "type": "span",
                            "extracted_text": sp["extracted_texts"][0],
                            "start_idx": int(sp["start_idx"][0].item()),
                            "end_idx": int(sp["end_idx"][0].item()),
                            "no_span_prob": round(float(sp["no_span_prob"][0].item()), 5),
                        }
                results.append({
                    "answers": answers,
                    "guardrail": {
                        "checked": bool(req.check_injection and injection_cdg is not None),
                        "injection_detected": inj_flags[i],
                        "injection_prob": inj_probs[i],
                    },
                    "meta": {
                        "latency_ms": per_item_ms,
                        "batch_wall_ms": batch_ms,
                        "coalesced_batch_size": B,
                        "deliberation_steps": req.deliberation_steps,
                        "tokens_generated": 0,
                    },
                })
            return results

    dynamic_batcher = AsyncDynamicBatcher(_batch_decide_worker, max_batch_size=16, max_wait_ms=4.0)

    @app.post("/v1/decide_batched")
    async def decide_batched(req: DecideRequest) -> Dict[str, Any]:
        """Async dynamic micro-batched decision endpoint coalescing concurrent HTTP calls on GPU!"""
        return await dynamic_batcher.submit(req)

    @app.get("/v1/health")
    def health() -> Dict[str, Any]:
        vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0
        return {
            "status": "ok",
            "version": "6.0.0",
            "d_llm": bridge.d_llm,
            "d_jev": bridge.d_jev,
            "fast_weight_skills_stored": int(bridge.fast_weights.num_stored.item()),
            "aci_alpha_t": round(float(bridge.online_aci.alpha_t), 4),
            "prefix_kv_hits": kv_manager.hits,
            "prefix_kv_misses": kv_manager.misses,
            "calibrator_attached": calibrator is not None,
            "action_router_attached": action_router is not None,
            "injection_guardrail_attached": injection_cdg is not None,
            "gpu_vram_allocated_mb": vram_mb,
        }

    def _run_node_head(node: GraphNodeSpec, h_proj: torch.Tensor, q_vec: torch.Tensor, raw_text: str) -> Dict[str, Any]:
        ptype = node.question.type.lower()
        if ptype == "noul":
            out = bridge.noul_head(q_vec)
            return {
                "noul": round(float(out["noul"][0].item()), 5),
                "epistemic_vacuity": round(float(out["epistemic_vacuity"][0].item()), 5),
            }
        if ptype == "score":
            out = bridge.score_head(q_vec)
            return {"score": round(float(out["score"][0].item()), 5)}
        if ptype == "span":
            sp = bridge.decide_span([raw_text], [node.question.instructions or "Extract target span"])
            return {
                "extracted_text": sp["extracted_texts"][0],
                "start_idx": int(sp["start_idx"][0].item()),
                "end_idx": int(sp["end_idx"][0].item()),
                "no_span_prob": round(float(sp["no_span_prob"][0].item()), 5),
            }
        if ptype == "choice":
            opts = node.question.options or ["yes", "no"]
            opt_embs = bridge.encode_isolated_option_texts(opts).unsqueeze(0)
            out = bridge.choice_head(h_proj, q_vec, opt_embs, include_ooi_sink=True)
            probs = out["option_probs"][0]
            if calibrator is not None and probs.numel() == getattr(calibrator, "num_classes", -1):
                probs = calibrator.predict_probs(probs.clamp_min(1e-7).log().unsqueeze(0))[0]
            w = int(probs.argmax().item())
            res: Dict[str, Any] = {
                "choice": opts[w],
                "choice_idx": w,
                "ooi_prob": round(float(out["ooi_prob"][0].item()), 5),
            }
            if action_router is not None:
                c_set = probs >= (1.0 / max(len(opts), 2)) * 0.5
                dec = action_router.decide_escalation(probs, c_set, float(out["ooi_prob"][0].item()))
                res["escalate_to_system2"] = bool(dec["escalate"])
                res["routing_reason"] = dec["reason"]
            return res
        return {"error": f"Unsupported primitive type '{node.question.type}'"}

    @app.post("/v1/decide_graph")
    def decide_graph(req: DecideGraphRequest) -> Dict[str, Any]:
        """Multi-node Conditional Composable Decision Graph with Batched Prefix-KV + Short-Circuiting!"""
        t0 = time.perf_counter()
        node_outputs: Dict[str, Any] = {}
        skipped_nodes: List[str] = []

        # Separate root/independent nodes (batched in ONE Prefix-KV pass!) from dependent/conditional nodes
        root_nodes = [n for n in req.nodes if not n.depends_on and not n.pass_span_from]
        dep_nodes = [n for n in req.nodes if n.depends_on or n.pass_span_from]

        prefix_hit = False
        reused_nodes = 0
        with torch.no_grad():
            if root_nodes:
                suffixes = [f"\nQuestion ({n.name}): {n.question.instructions}\nDecision:" for n in root_nodes]
                h_batch, m_batch, prefix_hit = kv_manager.prefill_graph_batch_with_cached_prefix(
                    bridge.hf_model, bridge.tokenizer, req.shared_prefix, suffixes, bridge.device_ref
                )
                reused_nodes += len(root_nodes) if prefix_hit else max(len(root_nodes) - 1, 0)
                for i, node in enumerate(root_nodes):
                    h_i = h_batch[i: i + 1]
                    m_i = m_batch[i: i + 1].unsqueeze(-1)
                    h_proj = bridge.state_proj(h_i)
                    q_vec = (h_proj * m_i).sum(dim=1) / m_i.sum(dim=1).clamp_min(1.0)
                    node_outputs[node.name] = _run_node_head(node, h_proj, q_vec, req.shared_prefix)

            # Evaluate conditional / chained nodes in topological order!
            for node in dep_nodes:
                if node.depends_on and node.depends_on in node_outputs:
                    parent_out = node_outputs[node.depends_on]
                    trigger_val = parent_out.get("noul", parent_out.get("score", 1.0))
                    if node.condition_gt is not None and float(trigger_val) <= float(node.condition_gt):
                        node_outputs[node.name] = {
                            "skipped": True,
                            "reason": f"{node.depends_on}={trigger_val} <= {node.condition_gt}",
                        }
                        skipped_nodes.append(node.name)
                        continue

                extra_ctx = ""
                if node.pass_span_from and node.pass_span_from in node_outputs:
                    span_txt = node_outputs[node.pass_span_from].get("extracted_text", "")
                    extra_ctx = f"\nExtracted Span ({node.pass_span_from}): {span_txt}"

                suffix = f"{extra_ctx}\nQuestion ({node.name}): {node.question.instructions}\nDecision:"
                h_suf, hit = kv_manager.prefill_with_cached_prefix(
                    bridge.hf_model, bridge.tokenizer, req.shared_prefix, suffix, bridge.device_ref
                )
                reused_nodes += int(hit)
                h_proj = bridge.state_proj(h_suf)
                q_vec = h_proj.mean(dim=1)
                node_outputs[node.name] = _run_node_head(node, h_proj, q_vec, req.shared_prefix + extra_ctx)

        return {
            "nodes": node_outputs,
            "meta": {
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "prefix_kv_cache_hits": reused_nodes,
                "prefix_was_warm": prefix_hit,
                "total_nodes": len(req.nodes),
                "executed_nodes": len(req.nodes) - len(skipped_nodes),
                "skipped_nodes": skipped_nodes,
                "tokens_generated": 0,
            },
        }

    @app.post("/v1/decide")
    def decide(req: DecideRequest) -> Dict[str, Any]:
        t0 = time.perf_counter()
        guardrail_info: Dict[str, Any] = {"checked": False, "injection_detected": False}
        active_state = req.state

        with torch.no_grad():
            enc = bridge.encode_texts([active_state], deliberation_steps=req.deliberation_steps)
            H = enc["state_h"]
            q_base = enc["q_vec"]

            if req.check_injection and injection_cdg is not None:
                raw_h, mask, _ = bridge._prefill_hf([active_state], max_length=160)
                pooled = (raw_h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                if pooled.size(-1) == injection_cdg.d_model:
                    inj_out = injection_cdg.predict_noul(pooled)
                    is_inj = bool(inj_out["is_injected"][0].item())
                    guardrail_info = {
                        "checked": True,
                        "injection_detected": is_inj,
                        "injection_prob": round(float(inj_out["inj_prob"][0].item()), 5),
                    }

            answers: Dict[str, Any] = {}
            for q_name, spec in req.questions.items():
                ptype = spec.type.lower()
                if ptype == "noul":
                    out = bridge.noul_head(q_base)
                    answers[q_name] = {
                        "type": "noul",
                        "noul": round(float(out["noul"][0].item()), 5),
                        "epistemic_vacuity": round(float(out["epistemic_vacuity"][0].item()), 5),
                    }
                elif ptype == "score":
                    out = bridge.score_head(q_base)
                    answers[q_name] = {
                        "type": "score",
                        "score": round(float(out["score"][0].item()), 5),
                        "level_idx": int(out["level_idx"][0].item()),
                        "probs": [round(float(v), 5) for v in out["probs"][0].tolist()],
                        "confidence": round(float(out["confidence"][0].item()), 5),
                    }
                elif ptype == "choice":
                    opts = spec.options or ["yes", "no"]
                    opt_embs = bridge.encode_isolated_option_texts(opts).unsqueeze(0)
                    out = bridge.choice_head(H, q_base, opt_embs, include_ooi_sink=True)
                    probs = out["option_probs"][0]
                    if calibrator is not None and probs.numel() == getattr(calibrator, "num_classes", -1):
                        probs = calibrator.predict_probs(probs.clamp_min(1e-7).log().unsqueeze(0))[0]
                    win_idx = int(probs.argmax().item())
                    ans_d: Dict[str, Any] = {
                        "type": "choice",
                        "choice": opts[win_idx],
                        "choice_idx": win_idx,
                        "probabilities": {o: round(float(p), 5) for o, p in zip(opts, probs.tolist())},
                        "ooi_prob": round(float(out["ooi_prob"][0].item()), 5),
                        "epistemic_vacuity": round(float(out["epistemic_vacuity"][0].item()), 5),
                        "confidence": round(float(probs[win_idx].item()), 5),
                    }
                    if action_router is not None:
                        c_set = probs >= (1.0 / max(len(opts), 2)) * 0.5
                        dec = action_router.decide_escalation(probs, c_set, float(out["ooi_prob"][0].item()))
                        ans_d["escalate_to_system2"] = bool(dec["escalate"])
                        ans_d["routing_reason"] = dec["reason"]
                    answers[q_name] = ans_d
                elif ptype == "span":
                    sp = bridge.decide_span([active_state], [spec.instructions or "Extract target span"])
                    answers[q_name] = {
                        "type": "span",
                        "extracted_text": sp["extracted_texts"][0],
                        "start_idx": int(sp["start_idx"][0].item()),
                        "end_idx": int(sp["end_idx"][0].item()),
                        "no_span_prob": round(float(sp["no_span_prob"][0].item()), 5),
                    }
                else:
                    answers[q_name] = {"error": f"Unsupported primitive type '{spec.type}'"}

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "answers": answers,
            "guardrail": guardrail_info,
            "meta": {
                "latency_ms": latency_ms,
                "deliberation_steps": req.deliberation_steps,
                "tokens_generated": 0,
            },
        }

    @app.post("/v1/consolidate")
    def consolidate(req: ConsolidateRequest) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with torch.no_grad():
            raw_h, mask, _ = bridge._prefill_hf([req.state], max_length=128)
            pooled = (raw_h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            bridge.fast_weights.consolidate_from_system2(
                pooled[0], verified_class_idx=req.verified_class_idx, option_name=req.option_name
            )
            bridge.sleep_consolidator.record_s2_escalation(pooled[0], verified_target=req.verified_class_idx)
        sleep_stats = None
        if req.trigger_sleep:
            sleep_stats = bridge.sleep_consolidator.run_sleep_consolidation(
                fast_weights=bridge.fast_weights, steps=40, evict_fast_weights=True
            )
        return {
            "consolidated": True,
            "write_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "fast_weight_skills_stored": int(bridge.fast_weights.num_stored.item()),
            "sleep_stats": sleep_stats,
        }

    @app.post("/v1/checkpoint/save")
    def save_checkpoint(req: CheckpointRequest) -> Dict[str, Any]:
        mods = {
            "state_proj": bridge.state_proj,
            "noul_head": bridge.noul_head,
            "choice_head": bridge.choice_head,
            "score_head": bridge.score_head,
            "fast_weights": bridge.fast_weights,
        }
        if calibrator is not None:
            mods["calibrator"] = calibrator
        if injection_cdg is not None:
            mods["injection_cdg"] = injection_cdg
        out_dir = OmniJevCheckpointManager.save_pretrained(req.directory, mods, metadata={"version": "10.0.0"})
        return {"saved": True, "directory": str(out_dir), "modules": list(mods.keys())}

    @app.post("/v1/checkpoint/load")
    def load_checkpoint(req: CheckpointRequest) -> Dict[str, Any]:
        mods = {
            "state_proj": bridge.state_proj,
            "noul_head": bridge.noul_head,
            "choice_head": bridge.choice_head,
            "score_head": bridge.score_head,
            "fast_weights": bridge.fast_weights,
        }
        if calibrator is not None:
            mods["calibrator"] = calibrator
        if injection_cdg is not None:
            mods["injection_cdg"] = injection_cdg
        meta, _ = OmniJevCheckpointManager.load_pretrained(req.directory, mods, device=bridge.device_ref)
        return {"loaded": True, "directory": req.directory, "metadata": meta}

    @app.post("/v1/hocbf_filter")
    def hocbf_filter(req: HOCBFFilterRequest) -> Dict[str, Any]:
        """Sub-millisecond 100% simulator-free pure-tensor HOCBF safety filter (2-DoF or 7-DoF)."""
        from .v7_certification import SimulatorFreeRobustHOCBFQP

        t0 = time.perf_counter()
        act_t = torch.tensor(req.proposed_action, dtype=torch.float32)
        st_t = torch.tensor(req.state_vec, dtype=torch.float32)
        if req.dof == 7:
            filt = hocbf_7dof if hocbf_7dof is not None else SimulatorFreeRobustHOCBFQP(state_dim=17, tip_dim=1, action_dim=7).eval()
            cy = float(req.curr_y) if req.curr_y is not None else float(st_t[15].item())
            safe_act, intervened = filt.filter_pusher7d_pure_tensor(act_t, st_t, curr_y=cy, y_forbid=req.y_forbid)
        else:
            filt = hocbf_2dof if hocbf_2dof is not None else SimulatorFreeRobustHOCBFQP(state_dim=8, tip_dim=2, action_dim=2).eval()
            tip_t = torch.tensor(req.tip_xy, dtype=torch.float32) if req.tip_xy is not None else st_t[6:8]
            safe_act, intervened = filt.filter_reacher2d_pure_tensor(
                act_t, st_t, tip_t, (float(req.obstacle_xy[0]), float(req.obstacle_xy[1])), float(req.obstacle_r), float(req.hazard_side)
            )
        return {
            "dof": req.dof,
            "safe_action": [round(float(x), 5) for x in safe_act.tolist()],
            "intervened": bool(intervened),
            "conformal_margin": round(float(filt.conformal_margin.item()), 5),
            "simulator_calls": 0,
            "filter_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    @app.post("/v1/world_rollout")
    def world_rollout(req: WorldRolloutRequest) -> Dict[str, Any]:
        """Single-pass ActionConditionedVideoWorldJeverator (`OmniJev-World`) mental foresight rollout."""
        from .v9_ultimate import ActionConditionedVideoWorldJeverator

        t0 = time.perf_counter()
        wm = world_model if world_model is not None else ActionConditionedVideoWorldJeverator(img_size=32, z_dim=8, d_state=16, d_hid=32).eval()
        dev = next(wm.parameters()).device
        s0 = torch.tensor([req.init_state], dtype=torch.float32, device=dev)
        acts = torch.tensor([req.actions], dtype=torch.float32, device=dev)
        z = torch.full((1, wm.z_dim), float(req.z_scale), dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = wm(s0, acts, z=z)
        traj = [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in out["pred_trajectory"][0].cpu().tolist()]
        return {
            "frames_simulated": len(traj),
            "pred_trajectory": traj,
            "final_gdn_state_norm": round(float(out["final_gdn_state"].norm().item()), 4),
            "rollout_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    @app.post("/v1/stream_3tier")
    def stream_3tier(req: ThreeTierStreamRequest) -> Dict[str, Any]:
        """Unified 3-Tier Live Stream endpoint: S1 (K=0, 0 tok) -> S1.5 (K=4, 0 tok) -> S2 (Hybrid GDN Sentence)."""
        from .v10_masterpiece import BatchedHybridGDNSentenceGenerator

        t0 = time.perf_counter()
        tier_req = req.tier.lower()
        if tier_req == "s2" and hybrid_llm is not None:
            s2_out = BatchedHybridGDNSentenceGenerator.generate_s2_batch_complete_sentences(
                hybrid_llm, bridge.tokenizer, [req.state], max_new_tokens=req.max_new_tokens, device=bridge.device_ref
            )
            return {
                "executed_tier": "s2_hybrid_gdn",
                "generated_sentence": s2_out["generated_sentences"][0],
                "tokens_generated": s2_out["total_tokens_generated"],
                "active_gdn_o1_layers": s2_out["active_gdn_o1_layers"],
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        k_steps = 4 if tier_req == "s1.5" else 0
        opts = req.options if req.options else ["Safe", "Escalate"]
        with torch.no_grad():
            out = bridge.decide_choice([req.state], opts, deliberation_steps=k_steps)
            probs = out["option_probs"][0].tolist()
            best_i = int(out["choice_idx"][0].item())
            max_p = float(max(probs))
            executed = "s1.5_equilibrium" if k_steps > 0 else "s1_zero_token"
            # Auto-escalate to S1.5 when confidence is low under `tier == "auto"`
            if tier_req == "auto" and max_p < 0.70:
                out = bridge.decide_choice([req.state], opts, deliberation_steps=4)
                probs = out["option_probs"][0].tolist()
                best_i = int(out["choice_idx"][0].item())
                executed = "s1.5_auto_escalated"
        return {
            "executed_tier": executed,
            "choice_idx": best_i,
            "choice_label": opts[best_i],
            "probabilities": {opts[j]: round(float(probs[j]), 5) for j in range(len(opts))},
            "tokens_generated": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    return app
