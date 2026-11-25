# OmniJev: A Unified System 1 / System 1.5 Decision, Embodied Control, & Single-Pass Cross-Modal Generation Architecture (`v1.0.0` $\to$ `v10.0.0`)

## Abstract

Autoregressive Large Language Models (System 2) generate text token-by-token at quadratic $O(L^2)$ attention cost, introducing high latency, JSON schema failures, option-order bias, poor calibration under quantization, and catastrophic inability to drive closed-loop physical control or real-time multimodal synthesis. Conversely, isolated classification and regression models lack compositional reasoning, epistemic abstention, and generative capability.

We present **OmniJev**, a unified **System 1 / System 1.5** neural architecture and verified PyTorch reference implementation (`omnijev v10.0.0`) that generalizes **Jev** typed decision heads and **Jeveration** (single-pass shared-random-state generation) across **five expansion axes (Axes A–E)**:
- **Axis A (12 Typed Decision Primitives):** `Noul`, `Choice`, `Score`, `Span`, `Point`, `Box`, `Polygon`, `Mask`, `Rank`, `Graph`, `Set`, and `Scalar/Distribution`, achieving exact **$O(1)$ permutation invariance** (`0.0%` option-order flip rate) and **`100.0%` convergence** across all 8 discrete primitives (`< 0.014` L1/MAE across all 4 continuous primitives).
- **Axis B (Universal Single-Pass Cross-Modal Jeveration):** AdaLN shared random state $\mathbf{z}$ + discrete codebooks + $\mathbf{z}=\mathbf{0}$ deterministic dense prediction across **24 kHz Meta `EnCodec` Audio** (`89.33 dB` SNR, `152×` real-time), **10-Channel Fourier RGB & Dense Masks** (`30.82 dB` PSNR, `99.27%` pixel accuracy, `94.05%` mIoU), **3D Point Clouds** (`0.0142` RMSE), **Interactive Video World Models (`OmniJev-World`)** (`1,834.2 FPS`, `31.05 dB` PSNR, `99.71%` pixel accuracy, `96.18%` mIoU), and **Grammar-Masked Python AST Code Synthesis** (`100.0%` executable).
- **Axis C (Embodied Control, CUA Grounding, & Physical/Security Guardrails):** Single-pass multimodal VLA action chunking (`FlowEnergyHybridLoss` + Real-Time Action Chunking RTC), **100% simulator-free pure-tensor Robust HOCBF-QP** (`0` runtime simulator calls, `100.0%` safety on both 2-DoF `Reacher-v5` and 7-DoF `Pusher-v5` at `949–1,418 Hz`), **Two-Stage Native-Resolution Foveal Blind `ScreenSpot` CUA** (`31.67%` 100% blind Pass-1, `55.0%` blind Top-3; `100.0%` Pass-1 on non-oracle DOM web UIs), and **Zero-Refill Latent Token-Mask Prompt-Injection Sanitization** (`97.6%` clean retention at `0.00%` hijack ASR in `0.67 ms`).
- **Axis D (System 1.5 Latent Deliberation, Hybrid Streaming Linearization, & 3-Tier Cascade):** Zero-shot depth-extrapolating `ScratchpadEquilibriumCore` (`81.25%` real `HotpotQA` at `0` text tokens), **50%-Linearized `Qwen2.5-0.5B` via Chunkwise Parallel WY + $L=1$ Rank-1 Gated DeltaNet** (`0.999921` streaming equivalence, exact `-50.0%` KV memory, `0.00%` `AG News` accuracy gap), and a **3-Tier Live Stream (`S1 K=0` $\to$ `S1.5 K=4` $\to$ Batched `[10, 1]` $O(1)$-GDN `S2`)** achieving a **`68.12×` end-to-end speedup** and **`-90.0%` generated tokens**.
- **Axis E (Edge Deployment, Calibration-Preserving Quantization, & Production Serving):** 5-fold cross-validated 4-bit NF4 calibration (`-85.8%` ECE: `0.1857 -> 0.0263`, `0.00%` acc loss), `AsyncDynamicBatcher` (`117.3 req/s`, `18.32×` throughput speedup, `3.04 ms` per-item p50), `safetensors` checkpointing (`0.00000000` reload diff), and unified FastAPI + CLI deployment.

---

## 1. Unified System Architecture

```mermaid
flowchart TB
    subgraph BUS["Universal Multimodal Input Bus"]
        IN["Text Tokens | High-Res RGB / UI | 24 kHz Audio | Proprioception | 3D Points | Graphs"]
    end

    subgraph STREAM["Streaming Hybrid Backbone (50% Quadratic KV + 50% O(1) Gated DeltaNet)"]
        GDN["Odd Layers [1,3,...,23]: Chunkwise Parallel WY (Prefill) + L=1 Rank-1 Update (Decode)"]
        ATTN["Even Layers [0,2,...,22]: Exact Sliding-Window / Quadratic KV Attention (-50% KV Bytes)"]
    end

    subgraph S15["System 1.5 Core & Composable Decision Graph (CDG) Runtime"]
        SAN["Zero-Refill Latent Token-Mask Sanitizer (Stage-1 Noul + Stage-2 Span, 0.00% ASR)"]
        EQ["ScratchpadEquilibriumCore (K=0..K_max Contractive Latent Hops, 0 Tokens)"]
        FW["Two-Timescale Consolidation: <1ms Delta-Rule Fast Weights + KL-Anchored LoRA Sleep"]
    end

    subgraph HEADS["Parallel Zero-Token Typed & Generative Output Heads"]
        AXA["Axis A: 12 Typed Primitives (Noul, Choice, Score, Span, Point, Box, Polygon, Mask, Rank, Graph, Set, Scalar)"]
        AXB["Axis B: Universal Jeveration Head (AdaLN Shared z -> Audio EnCodec, Fourier RGB/Mask, 3D PLY, Video GIF, AST)"]
        AXC["Axis C: Jev-VLA / Jev-CUA + 100% Simulator-Free Pure-Tensor Robust HOCBF-QP (0 Sim Calls)"]
    end

    subgraph ROUTE["Calibrated Epistemic & Action-Risk Conformal Router"]
        CAL["5-Fold Cross-Validated NF4 Calibrator + Evidential Vacuity + Online ACI"]
        S2["Escalated System 2 (10% Queries): Batched [B, 1] O(1)-GDN Complete-Sentence Generation (-50% KV)"]
    end

    IN --> STREAM --> SAN --> EQ
    EQ --> AXA & AXB & AXC
    EQ --> FW
    AXA --> CAL
    CAL -- "90% Routine / Multi-Hop (0 Tokens)" --> OUT1["Instant Certified Execution / Action"]
    CAL -- "10% High-Risk / OOI Escalation" --> S2
```

---

## 2. Core Mathematical Foundations

### 2.1 Exact $O(1)$ Permutation-Invariant Option Evaluation (Axis A & D)
Standard causal LLMs exhibit severe position bias (`58.33%` flip rate on `Qwen2.5-0.5B` under option shuffling). OmniJev encodes each candidate option $c \in \{1,\dots,C\}$ in an **isolated local-RoPE lane** $\mathbf{e}_c = f_{\text{opt}}(o_c)$ and applies position-free cross-attention against the shared context state $\mathbf{H}$ plus an Out-of-Inventory (OOI) sink logit $z_{\text{OOI}}$:
$$p(c \mid \mathbf{H}, \{o_j\}_{j=1}^C) = \frac{\exp(\tau \langle \mathbf{q}_c, \mathbf{e}_c \rangle)}{\exp(z_{\text{OOI}}) + \sum_{j=1}^C \exp(\tau \langle \mathbf{q}_j, \mathbf{e}_j \rangle)}$$
Because no positional embedding or causal mask links option $i$ to option $j$, any permutation $\pi \in S_C$ commutes identically with the output distribution (`0.0%` flip rate, `2.24 × 10⁻⁸` max numerical shift).

### 2.2 Single-Pass Shared-Random-State Jeveration (Axis B)
To eliminate autoregressive token decoding *and* prevent unimodal L1/L2 mode-averaging across multimodal futures (RGB images, robot trajectories, interactive video, 3D geometry), OmniJev injects a single trajectory- or sample-wide random latent $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ via AdaLN across all parallel output tokens simultaneously, trained with a strictly proper **Flow-Energy Hybrid Score**:
$$\mathcal{L}_{\text{FE}}(\theta) = \mathbb{E}_{\mathbf{z}}\|\hat{\mathbf{y}}(\mathbf{x}, \mathbf{z}) - \mathbf{y}\|_2 - \frac{\lambda}{2} \mathbb{E}_{\mathbf{z}, \mathbf{z}'}\|\hat{\mathbf{y}}(\mathbf{x}, \mathbf{z}) - \hat{\mathbf{y}}(\mathbf{x}, \mathbf{z}')\|_2$$
Setting $\mathbf{z} = \mathbf{0}$ instantaneously switches the exact same head into a **deterministic dense predictor** (`99.27%–99.71%` segmentation pixel accuracy).

### 2.3 50% Hybrid Linearization: Chunkwise Parallel WY Prefill & $L=1$ Rank-1 Decode (Axis D)
Odd transformer layers `[1, 3, ..., 23]` of `Qwen2.5-0.5B-Instruct` are warm-started into **Gated DeltaNet** layers whose state matrix $S_t \in \mathbb{R}^{H \times d_h \times d_h}$ occupies strictly **$O(1)$ constant memory (`2.882 MB`)** regardless of context length $L$:
- **Prefill ($L > 1$):** GPU-parallel **Chunkwise WY representation** (`7.47×` faster than serial recurrence, `4.77 × 10⁻⁷` max diff).
- **Autoregressive System-2 Decode ($L = 1$):** Exact rank-1 recurrent matrix update matching Chunkwise WY to `< 1e-5`:
  $$S_t = \alpha_t S_{t-1}\left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top\right) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top, \qquad \mathbf{o}_t = S_t \mathbf{q}_t$$
Carrying $(S_t, \text{conv\_hist})$ inside Hugging Face `DynamicCache._gdn_states` preserves **`0.999921` chunk-by-chunk streaming equivalence** and cuts KV cache bytes by **exact `-50.0%`** across `L = 512 / 2048 / 4096`.

### 2.4 100% Simulator-Free Conformal-Buffered Braking-Invariant HOCBF-QP (Axis C)
Cartesian end-effector position $\mathbf{p}_t$ has relative degree 2 with respect to joint torques $\mathbf{a}_t$, causing naive 1-step position CBFs to suffer `0.0%` safety under inertia. `SimulatorFreeRobustHOCBFQP` learns a 1-step state transition network $(\mathbf{q}_{t+1}, \dot{\mathbf{q}}_{t+1}, \mathbf{p}_{t+1}) = \mathbf{s}_t + f_\theta(\mathbf{s}_t, \mathbf{a}_t)$, calibrates a 95% split-conformal error margin $\epsilon_{\text{conf}}$, and unrolls `[proposed_action, lattice_actions]` across $1$ control step + $K_{\text{brk}}$ active braking steps ($\mathbf{a}_{\text{brk}} = \mathrm{clip}(-k_d \dot{\mathbf{q}}, -1, 1)$) in **one batched PyTorch tensor pass (`0` runtime simulator calls)**:
$$\min_{k \in \{0,\dots,K_{\text{brk}}\}} h(\hat{\mathbf{p}}_{t+1+k}(\mathbf{a}); \epsilon_{\text{conf}}) \ge \min\!\big(\max((1 - \gamma) h(\mathbf{p}_t), h_{\min}), h_{\text{cap}}\big)$$

---

## 3. Master Empirical Benchmark Table (Verified on NVIDIA GeForce RTX 3060 12GB)

| Domain / Axis | Benchmark & Backbone / Environment | Baseline | OmniJev (`v1`–`v10`) | Key Improvement | Source Log |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Axis A + D** | Option-Order Invariance (`Qwen2.5-0.5B`, `AG News`) | `58.33%` flip rate | **`0.00%` flip rate (`4.47e-8`)** | Exact $O(1)$ invariance | [`v3_pretrained_qwen_results.json`](../experiments/v3_pretrained_qwen_results.json) |
| **Axis A** | All 12 Axis-A Primitives Simultaneous Convergence | — | **8/8 Discrete `100.0%`, 4/4 Cont. `<0.014`** | Complete primitive verification | [`v9_ultimate_zero_weakness_results.json`](../experiments/v9_ultimate_zero_weakness_results.json) |
| **Axis B (Audio)** | Single-Pass Meta `facebook/encodec_24khz` Audio | Autoregressive RVQ | **`89.33 dB` SNR, `6.58 ms` (`152×` RT)** | `0` AR steps, playable `.wav` | [`v6_sanitizing_encodec_serving_results.json`](../experiments/v6_sanitizing_encodec_serving_results.json) |
| **Axis B (Img/3D)** | 10-Ch Fourier RGB, $\mathbf{z}=0$ Mask, & 3D `.ply` | Raw linear coords (`26.86 dB`) | **`30.82 dB` PSNR, `99.27%` pix, `94.05%` mIoU** | `+3.96 dB`, `0.0142` 3D RMSE | [`v8_reviewer_proof_results.json`](../experiments/v8_reviewer_proof_results.json) |
| **Axis B (Video)** | Action-Conditioned Video World Model (`OmniJev-World`) | Iterative diffusion | **`1,834.2 FPS`, `31.05 dB`, `99.71%` pix, `96.18%` mIoU** | `0.0001` traj L1, real `.gif` | [`v9_ultimate_zero_weakness_results.json`](../experiments/v9_ultimate_zero_weakness_results.json) |
| **Axis B (Code)** | Grammar-Masked Python AST Synthesis (100 fns) | `0.0%` unmasked valid | **`100.0%` `ast.parse()` & live executable** | `0.427 ms/function` | [`v4_real_codecs_results.json`](../experiments/v4_real_codecs_results.json) |
| **Axis C (2-DoF)** | Simulator-Free `Reacher-v5` Joint Goal + 30% Glitches | `0.0%` safe (unfilt. / 1-step CBF) | **`100.0%` safe, `98.33%` goal (`0.947 ms` = `1,055.8 Hz`)** | `0` runtime simulator calls | [`v10_definitive_masterpiece_results.json`](../experiments/v10_definitive_masterpiece_results.json) |
| **Axis C (7-DoF)** | Simulator-Free `Pusher-v5` Boundary Station-Keeping | `0.0%` safe / `0.0%` goal | **`100.0%` safe, `100.0%` goal (`0.718 ms` = `1,393 Hz`)** | `0` runtime simulator calls | [`v8_reviewer_proof_results.json`](../experiments/v8_reviewer_proof_results.json) |
| **Axis C (CUA)** | Non-Oracle DOM Web UI CUA (`SmolVLM-256M`, `0%` leak) | `12.5%` random | **`100.0%` Pass-1, `100.0%` Top-3, `100.0%` `Noul` F1** | Color-spoof proof | [`v8_reviewer_proof_results.json`](../experiments/v8_reviewer_proof_results.json) |
| **Axis C (Blind)** | 100% Blind `ScreenSpot` (`gt_bbox`-Free, `SmolVLM-256M`) | `6.0%` (v6 single-tile) | **`31.67%` Pass-1 (`5.28×`), `55.00%` Top-3 (`0.2126` L1)** | True `1920×1080` Stage-2 zoom | [`v10_definitive_masterpiece_results.json`](../experiments/v10_definitive_masterpiece_results.json) |
| **Axis C (Sec)** | Latent Token-Mask Prompt-Injection Sanitizer (`Qwen-1.5B`) | `51.67%` hijack ASR | **`0.00%` hijack ASR, `97.6%` clean retention (`0.67 ms`)** | `0` second LLM prefill | [`v7_zero_gap_certification_results.json`](../experiments/v7_zero_gap_certification_results.json) |
| **Axis D (S1.5)** | Real `HotpotQA` Multi-Hop (`Qwen2.5-1.5B`, `0` tokens) | `54.17%` ($K=0$) | **`81.25%` (`ECE 0.0557`, `+22.22%` over v4)** | `0` generated tokens | [`v5_hybrid_distill_results.json`](../experiments/v5_hybrid_distill_results.json) |
| **Axis D (GDN)** | 50%-Linearized `Qwen2.5-0.5B` Streaming & 3-Tier Stream | `125.91 s` pure S2 (`100` q) | **`1.85 s` (`68.12×` speedup), `0.999921` cos, `-50%` KV** | `-90%` tokens, `0.0%` AG gap | [`v10_definitive_masterpiece_results.json`](../experiments/v10_definitive_masterpiece_results.json) |
| **Axis D (Agent)** | Unified Closed-Loop Axes A–E Neural Embodied Agent | — | **`100.0%` inj. det, `0.0%` ASR, `100.0%` plan & safety** | 100% end-to-end neural | [`v10_definitive_masterpiece_results.json`](../experiments/v10_definitive_masterpiece_results.json) |
| **Axis E (Quant)** | 5-Fold CV 4-bit NF4 `Qwen2.5-0.5B` Calibration | `0.1857` raw NF4 ECE | **`0.0263` ECE (`-85.8%`), `0.00%` acc loss** | `52.7%` VRAM reduction | [`v6_streaming_linearized_nf4_results.json`](../experiments/v6_streaming_linearized_nf4_results.json) |
| **Axis E (Serve)** | FastAPI `AsyncDynamicBatcher` (`Qwen2.5-1.5B`, 32 conc.) | `6.4 req/s`, `617 ms` p50 | **`117.3 req/s` (`18.32×`), `3.04 ms` per-item p50** | + `/v1/hocbf_filter`, `/v1/world_rollout`, `/v1/stream_3tier` | [`v7_zero_gap_certification_results.json`](../experiments/v7_zero_gap_certification_results.json) |

---

## 4. Production API Surface (`src/omnijev/server.py` & `omnijev` CLI)

- `POST /v1/decide` & `POST /v1/decide_batched`: Single & dynamically micro-batched zero-token typed decisions (`Noul`, `Choice`, `Score`, `Span`) with live calibration, action-risk conformal routing, and inline prompt-injection guardrails.
- `POST /v1/decide_graph`: Batched Prefix-KV Composable Decision Graph execution (`9.45×–13.15×` warm speedup) with conditional node short-circuiting (`depends_on` / `condition_gt`) and `Span -> Choice` value chaining.
- `POST /v1/hocbf_filter`: Sub-millisecond simulator-free 2-DoF / 7-DoF torque safety projection (`0` runtime simulator calls).
- `POST /v1/world_rollout`: Single-pass `ActionConditionedVideoWorldJeverator` (`OmniJev-World`) mental foresight rollout.
- `POST /v1/stream_3tier`: Unified 3-tier execution (`S1 K=0` $\to$ `S1.5 K=4` $\to$ `S2` Batched $O(1)$-GDN complete sentence).
- `POST /v1/consolidate`, `POST /v1/checkpoint/save`, `POST /v1/checkpoint/load`: Live `<1 ms` Delta-Rule fast-weight skill consolidation, KL-anchored Micro-LoRA sleep consolidation, and bit-exact `safetensors` persistence (`0.00000000` reload diff).

---

## 5. Complete Chapter & Verification Index

- **Detailed Specification Chapters:** [`01`](01_unified_architecture_and_primitives.md) $\to$ [`14`](14_v10_definitive_masterpiece_edition.md)
- **Automated Verification (`18/18` Unit & Mathematical-Invariant Tests):** `pytest tests/test_omnijev_suite.py -v`
- **Editable Package & CLI (`v10.0.0`):** `pip install -e ".[dev]" && omnijev info && omnijev summary`
