"""OmniJev CLI entry point (`omnijev info`, `omnijev test`, `omnijev serve`)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="omnijev", description="OmniJev v10 Unified System 1 / 1.5 Decision Engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Print OmniJev version, GPU status, and model tier table")
    sub.add_parser("test", help="Run the full OmniJev pytest verification suite")
    sub.add_parser("summary", help="Print verified benchmark logs, exported media artifacts, and package summary")

    p_srv = sub.add_parser("serve", help="Launch live FastAPI /v1/* multimodal decision & embodied server via uvicorn")
    p_srv.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace causal LM backbone ID")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8080)
    p_srv.add_argument("--nf4", action="store_true", help="Load backbone in real 4-bit NF4 bitsandbytes")

    args = parser.parse_args(argv)
    if args.command == "info":
        import torch
        from omnijev import MODEL_TIERS
        info = {
            "omnijev_version": "10.0.0",
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "model_tiers": list(MODEL_TIERS.keys()),
        }
        print(json.dumps(info, indent=2))
        return 0

    if args.command == "summary":
        import omnijev
        root = Path(__file__).resolve().parents[2]
        exp_dir = root / "experiments"
        out_dir = root / "outputs"
        logs = sorted(p.name for p in exp_dir.glob("*.json")) if exp_dir.exists() else []
        media = sorted(p.name for p in out_dir.iterdir() if p.is_file()) if out_dir.exists() else []
        v10_path = exp_dir / "v10_definitive_masterpiece_results.json"
        v10_metrics = json.loads(v10_path.read_text(encoding="utf-8")) if v10_path.exists() else {}
        summary = {
            "omnijev_version": "10.0.0",
            "exported_symbols_count": len(omnijev.__all__),
            "benchmark_logs_count": len(logs),
            "exported_media_artifacts": media,
            "v10_headline_metrics": {
                "blind_screenspot_stage2_pass1_pct": v10_metrics.get("exp14a_two_stage_blind_foveal_screenspot", {}).get("v10_stage2_native_foveal_pass1_hit_pct"),
                "blind_screenspot_stage2_top3_pct": v10_metrics.get("exp14a_two_stage_blind_foveal_screenspot", {}).get("v10_stage2_native_foveal_top3_hit_pct"),
                "simfree_hocbf_2dof_joint_safe_goal_pct": v10_metrics.get("exp14b_subms_tangential_simfree_hocbf", {}).get("joint_safe_and_goal_pct"),
                "simfree_hocbf_2dof_hz": v10_metrics.get("exp14b_subms_tangential_simfree_hocbf", {}).get("control_loop_hz"),
                "stream_3tier_speedup_x": v10_metrics.get("exp14c_batched_hybrid_gdn_100query_stream", {}).get("v10_end_to_end_stream_speedup_x"),
                "embodied_agent_safe_pct": v10_metrics.get("exp14d_unified_axes_a_to_e_embodied_agent", {}).get("physical_safe_episodes_pct"),
                "embodied_agent_hijack_asr_pct": v10_metrics.get("exp14d_unified_axes_a_to_e_embodied_agent", {}).get("prompt_injection_hijack_asr_pct"),
            },
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "test":
        root = Path(__file__).resolve().parents[2]
        return subprocess.call([sys.executable, "-m", "pytest", str(root / "tests" / "test_omnijev_suite.py"), "-v"])

    if args.command == "serve":
        import torch
        import uvicorn
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from omnijev import (
            ActionConditionedVideoWorldJeverator,
            PretrainedOmniJevBridge,
            SimulatorFreeRobustHOCBFQP,
            create_omnijev_app,
            load_mixed_precision_nf4_llm,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.nf4:
            hf, tok = load_mixed_precision_nf4_llm(args.model)
        else:
            tok = AutoTokenizer.from_pretrained(args.model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
        bridge = PretrainedOmniJevBridge(hf, tok, d_jev=256, max_classes=8, device=device)
        hocbf_2dof = SimulatorFreeRobustHOCBFQP(state_dim=8, tip_dim=2, action_dim=2).eval()
        hocbf_7dof = SimulatorFreeRobustHOCBFQP(state_dim=17, tip_dim=1, action_dim=7).eval()
        world_model = ActionConditionedVideoWorldJeverator(img_size=32, z_dim=8, d_state=16, d_hid=32).to(device).eval()
        app = create_omnijev_app(
            bridge,
            hocbf_2dof=hocbf_2dof,
            hocbf_7dof=hocbf_7dof,
            world_model=world_model,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
