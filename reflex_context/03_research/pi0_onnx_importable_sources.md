# pi0-onnx-parity: importable code sources + action plan

**Compiled:** 2026-04-17
**Goal:** end-to-end pi0 ONNX export with cos ≥ 0.999 vs reference PyTorch `policy.predict_action_chunk` on shared noise. Tracks GOALS.yaml `pi0-onnx-parity` (weight 9, MVP).

**Key finding:** multiple production-grade pi0 ONNX+TRT implementations already exist in the open-source ecosystem. Revised estimate: **1.5–3 weeks** (vs. 3–5 weeks writing from scratch).

This doc replaces the "write from scratch" plan. Next session should consult this first before writing new exporter code.

---

## Tier 1 — drop-in / near drop-in replacements (license-gated)

These sources have components that **REPLACE** code in our plan, not just augment. Biggest time saves. Each has a license risk that must be resolved before commercial import.

### 1. Jetson AI Lab `openpi_on_thor` — the silver bullet

- **URL:** https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/
- **What's in the bundle:**
  - `pytorch_to_onnx.py` — pi0.5 → ONNX conversion
  - `patches/apply_gemma_fixes.py` — PaliGemma/Gemma quirks patched (gold)
  - `build_engine.sh` — trtexec invocation for pi0.5
  - `trt_model_forward.py` + `trt_torch.py` — TRT engine inference wrappers
  - `calibration_data.py` — INT8/FP8 calibration harness
  - `pi05_inference.py` — full end-to-end inference script
  - `serve_policy.py` — server wrapper
  - `thor.Dockerfile` — deployment container
- **Proven results:** cos > 0.99, 1.71× speedup over PyTorch
- **Precision:** FP8 + NVFP4 on Thor
- **License:** NOT STATED — blocker for commercial reflex-vla import. **Must contact NVIDIA DevRel for clarification.**
- **If permissive:** saves ~15–20 engineer-days.
- **Integration risk:** MEDIUM
  - Targeted at Thor (FP8-capable SM 10.0); Orin Nano is SM 8.7 and rejects FP8 Q/DQ at runtime
  - Must port FP8 → FP16 for Orin target
  - Scripts use pi0.5 (AdaRMSNorm variant); we want plain pi0
  - Gemma patches are still applicable — extract `apply_gemma_fixes.py` regardless

### 2. `Tacoin/openpi-pi0.5-libero-onnx` — MIT-licensed parity oracle

- **URL:** https://huggingface.co/Tacoin/openpi-pi0.5-libero-onnx
- **What it gives us:** pre-exported ONNX of pi0.5-LIBERO as a unified graph. Includes ORT+TRT EP integration snippets.
- **Variants:** W8A16, FP16, W4A4 (W4A4 is experimental quantization)
- **License:** MIT (✓ safe for commercial)
- **Use case:** **parity oracle**. Run our decomposed graph and this monolithic graph at matching shared noise; diff outputs. Any discrepancy is a bug in our pipeline, since Tacoin's is validated.
- **Saves:** 3–5 days on reference-generation infrastructure.
- **Integration risk:** LOW
  - Note: pi0.5 ≠ pi0. The models are related but have different action heads. Can only cross-check against our pi0.5 work (post-MVP).
  - For pi0 (MVP target), use as architectural reference only.

### 3. `NVIDIA/Isaac-GR00T/scripts/deployment/` — pattern source

- **URL:** https://github.com/NVIDIA/Isaac-GR00T/tree/main/scripts/deployment
- **What's there:**
  - `build_trt_pipeline.py`
  - `export_onnx_n1d7.py`
  - `trt_model_forward.py`, `trt_torch.py`
  - `verify_n1d7_trt.py` — **verification harness we can fork**
  - `benchmark_inference.py`
- **Gold patterns:**
  - `DiTInputCapture`, `LLMInputCapture` — hooks that record real inputs during a forward pass, then replay as traced inputs during ONNX export (avoids dummy-input shape mismatches)
  - `_apply_rotary_real` — complex-free RoPE implementation that exports cleanly (ours uses complex numbers via sin/cos; this is an alternative)
  - `_simple_causal_mask` — simple causal mask that exports cleanly
  - `_make_onnx_vision_attention_forward` — chunked attention wrapper for large attention matrices
- **License:** UNCLEAR — check repo root LICENSE before copying. If NVIDIA's "Isaac License" or similar, may be restrictive.
- **Saves:** 7–10 days (verify infra, rotary patterns, dynamic-axes for vision attention).
- **Integration risk:** MEDIUM — tightly coupled to GR00T's N1.7 architecture but extraction is mechanical.

### 4. HuggingFace Optimum PaliGemma2 ONNX — 4-file split reference

- **URL:** https://github.com/huggingface/optimum-onnx
- **Reference export:** `onnx-community/paligemma2-3b-pt-224` (on HF Hub)
- **What it gives us:** the exact 4-file decomposition our plan targets:
  - `embed_tokens.onnx` — text token embeddings
  - `vision_encoder.onnx` — SigLIP
  - `decoder_model_merged.onnx` — prefill + decode unified
  - `decoder_with_past.onnx` — decode-only (with KV cache input)
- **License:** Apache-2.0 (✓ safe for commercial)
- **Saves:** 4–6 days on decomposition plumbing.
- **Integration risk:** LOW for SigLIP + Gemma-decoder components. MEDIUM for the "image-text-to-text" task integration.
- **Known gap:** Optimum does NOT officially support the `image-text-to-text` task for PaliGemma. Open issue: https://discuss.huggingface.co/t/paligemma2-onnx-export-keyerror-unknown-task-image-text-to-text/139726
  - Workaround: `@register_for_onnx` custom-config pattern. We write a small PaliGemmaOnnxConfig subclass (~100 lines) — still much less than writing the whole exporter.

---

## Tier 2 — confirmed-safe library code + tooling (Apache-2.0 / MIT)

These are unambiguously safe to import. Use liberally.

### 5. `lerobot/src/lerobot/policies/pi0/modeling_pi0.py`

- **URL:** https://github.com/huggingface/lerobot
- **License:** Apache-2.0
- **What to reference:**
  - `PI0Policy.predict_action_chunk` — reference forward
  - `PI0Pytorch.sample_actions` — flow-matching loop
  - `PI0Pytorch.embed_prefix` + `embed_suffix` — how tokens become embeddings
  - `PI0Pytorch.denoise_step` — single expert forward
  - `PaliGemmaWithExpertModel` — dual-expert wrapper (our direct target)
- **Confirms:** the right seams exist for ONNX extraction. Already in our venv.

### 6. NVIDIA TensorRT-LLM `examples/models/core/gemma`

- **URL:** https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma
- **License:** Apache-2.0
- **What's there:** `convert_checkpoint.py` handles HF/JAX/Torch/Keras checkpoints → TRT-LLM. Quantization recipes: FP8, NVFP4, INT4-AWQ, SmoothQuant, INT8 KV-cache.
- **Saves:** 2–3 days on Gemma quantization recipes (relevant for v0.3 distill-dmpo).
- **Integration risk:** LOW for weight conversion; HIGH if we need TRT-LLM runtime (heavy dep — avoid unless v0.3+).
- **Gap:** no PaliGemma branch — Gemma decoder only.

### 7. `Dexmal/realtime-vla` (arxiv 2510.26742)

- **URL:** https://github.com/Dexmal/realtime-vla
- **License:** MIT
- **What it gives:** pi0/pi0.5 at 30 Hz via CUDA-graph + fused Triton kernels. Includes:
  - `convert_from_jax.py` + `convert_from_jax_pi05.py` — JAX → pickle helpers (validated ingest)
  - Reference 24-GEMM decomposition of pi0 inference (performance upper-bound)
- **Saves:** 1–2 days if we need JAX ingest (unlikely since we use lerobot's PyTorch port, but nice to have).
- **Not an ONNX path.** Useful as performance benchmark reference.

### 8. `allenzren/open-pi-zero`

- **URL:** https://github.com/allenzren/open-pi-zero
- **License:** MIT
- **What it gives:** second PyTorch reference implementation for pi0 on PaliGemma 2B. Block-causal attention + flow-matching patterns.
- **Integration risk:** MEDIUM — not exact architectural match; use for pattern reference only.

### 9. `ainekko/smolvla_base_onnx`

- **URL:** https://huggingface.co/ainekko/smolvla_base_onnx
- **License:** Apache-2.0
- **What it gives:** 9-file decomposition of SmolVLA matching exactly our `vlm_prefix_exporter.py` split (vision, text, expert_prefill, expert_decode, state_projector, time_in/out, action_in/out). **Public validation that our approach is correct.**
- **Use case:** sanity cross-check our SmolVLA export. Pattern-transfers to pi0.

### 10. `onnx-community/siglip2-so400m-patch14-384-ONNX`

- **URL:** https://huggingface.co/onnx-community/siglip2-so400m-patch14-384-ONNX
- **License:** Apache-2.0
- **What it gives:** pre-exported 1152-dim SigLIP2 so400m (matching pi0's vision dim)
- **Integration risk:** MEDIUM — verify pi0's SigLIP variant has matching weights (patch size 14 vs pi0's need-to-check).
- **If compatible:** saves 1 day on vision encoder export.

### 11. `transformers.StaticCache` + ONNX dynamic-cache docs

- **URLs:**
  - https://huggingface.co/docs/transformers/kv_cache
  - PyTorch issue #136582 (DynamicCache exportability discussion)
  - `sdpython/onnx-diagnostic` (dynamic-shape guessing)
- **License:** Apache-2.0 (transformers)
- **What it gives:** `StaticCache` is the ONNX-exportable path (pre-allocated, fixed-size). Avoids DynamicCache Python-object-serialization issues.
- **Saves:** 1–2 days of trial-and-error on prefix-KV I/O shape.

### 12. `aifoundry-org/ETARS`

- **URL:** https://github.com/aifoundry-org/ETARS
- **License:** NOT STATED (check before using)
- **What it gives:** ONNXRuntime inference backend for SmolVLA. Good runtime-layer reference.
- **Saves:** 0.5–1 day if license permits.

### 13. `onnx-community/paligemma2-3b-*` ONNX exports

- **URLs:** `paligemma2-3b-pt-224`, `paligemma2-3b-ft-docci-448`, `paligemma2-3b-pt-448/896` on HF Hub
- **License:** per base model (Gemma + PaliGemma terms — not fully open for all commercial uses). Check Gemma terms before commercial reflex-vla ship.
- **Use case:** reference diffing (component-by-component parity check).
- **Saves:** 1–2 days if licensing clears.

---

## Tier 3 — knowledge / patterns only (no direct code reuse)

### 14. `lucidrains/pi-zero-pytorch`
Simplified pi0 implementation. MIT. Not architecturally exact. SKIP for code; reference for concepts.

### 15. NVIDIA Isaac GR00T N1.6/N1.7 model cards + deployment blog
- **URL:** https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow
- **Critical takeaway:** FP8-on-Orin is NOT supported (Q/DQ rejected at runtime). Must use FP16 for Orin target. This contradicts our MVP assumption that we can copy Thor's FP8+NVFP4 recipe.

### 16. `D-Robotics/rdk_LeRobot_tools`
ACT-only, not pi0. SKIP.

### 17. Papers with code
- `2507.14049` EdgeVLA — concepts only, no exports
- `2510.23511` Dexbotic — concepts only, no exports

---

## Critical risks surfaced by research

These are MVP-relevant gotchas that weren't on my radar before this research:

1. **Orin rejects FP8 Q/DQ at runtime** — Thor's FP8+NVFP4 pipeline WON'T port 1:1 to Orin Nano. **Must use FP16 on Orin target.** Per NVIDIA GR00T team's own deployment guide.

2. **Optimum does NOT officially support image-text-to-text for PaliGemma** — confirmed open issue. We WILL need to write a custom `OnnxConfig` subclass. No way around it.

3. **Gemma3 has `torch.onnx.export` vmap failure** — PyTorch issue #160761. **Stay on Gemma2.** PaliGemma2 uses Gemma2 which works; confirmed safe.

4. **PI's open ONNX weights are pi0.5-native** — most community exports target pi0.5 (AdaRMSNorm version). Our MVP wants plain pi0 (simpler). Trade-off:
   - pi0 is architecturally simpler (no AdaRMSNorm) — easier for us to hand-write
   - No drop-in ONNX reference — we're the trailblazer for pi0 ONNX
   - Tacoin's pi0.5 reference is close but not a direct oracle

5. **License unknowns block commercial adoption of Tier-1 silver bullets.** The `openpi_on_thor` bundle and Isaac-GR00T deployment scripts are the biggest time savers but both have unclear license terms. Resolve first before heavy integration.

---

## Revised timeline estimate

**With Tier-1 imports (all permissive):** ~1.5–2 weeks
**With Tier-2 only (if Tier-1 blocked):** ~2.5–3 weeks
**Writing from scratch (pre-research baseline):** 3–5 weeks

**Mess-up probability:**
- Best case (all tiers): ~35%
- Middle case (Tier-2 only): ~40%
- Worst case (bugs hunt dominates): ~50%

Decrease vs. pre-research baseline (~60%) driven by:
- `apply_gemma_fixes.py` — patches already made and proven
- `onnx-community/paligemma2-3b-pt-224` — known-good 4-file decomposition
- Tacoin MIT ONNX — parity oracle available for pi0.5 cross-checks
- `_apply_rotary_real` — ONNX-clean RoPE pattern validated in GR00T
- `StaticCache` — documented ONNX-exportable cache type

---

## 7-day action plan

| Day | Task | Dependency |
|---|---|---|
| 0 (today) | Pull `Tacoin/openpi-pi0.5-libero-onnx` (MIT) to `./_oracles/pi05_libero_onnx/`. Verify loads. | HF Hub access |
| 0 (parallel) | Email NVIDIA DevRel requesting license clarity on `openpi_on_thor` + `Isaac-GR00T/scripts/deployment/` for commercial reflex-vla use. | NVIDIA contact |
| 1 | Run `optimum-cli export onnx --model google/paligemma-3b-pt-224 ./_optimum_sanity/` to confirm baseline PaliGemma export (will fail on task but should produce components). | optimum-onnx installed |
| 2–3 | Fork `src/reflex/exporters/vlm_prefix_exporter.py` → `src/reflex/exporters/pi0_prefix_exporter.py`. Swap SmolVLM2→PaliGemma2, SmolLM2→Gemma2, update state-dict prefixes. Use Optimum's 4-component split as structural guide. | Day 1 sanity passes |
| 4–6 | Integrate patterns: Isaac-GR00T's `_apply_rotary_real`, `_simple_causal_mask`, `DiTInputCapture` hooks. Extract `apply_gemma_fixes.py` lessons (patch-level bug fixes). License-clean rewrite if needed. | Day 1 NVIDIA response (ideal) |
| 7 | First end-to-end parity diff: export pi0 full pipeline, run with shared noise, compare vs real PyTorch pi0. Expect cos=0.X (SmolVLA precedent). Record result. | Days 2–6 complete |
| 8+ | Bug hunt using established diagnostic ladder (stage-diff → single-layer → full). Document each bug in `reflex_context/02_bugs_fixed/pi0_pipeline_bugs.md`. | Day 7 baseline diff |

---

---

## Additional code sources (found in 2026-04-17 second-pass research)

### Previously missed per-component sources

**SigLIP vision encoder:**
- `AXERA-TECH/siglip-so400m-patch14-384` (Apache-2.0) — pre-split `*_vision.onnx` + `*_text.onnx`, w8a16 quantized. Wrong res (384 vs 224) but pattern transfers.
- `NSTiwari/PaliGemma2-ONNX-Transformers.js` (Apache-2.0) — `Convert_PaliGemma2_to_ONNX.ipynb` is a complete drop-in adaptation template.

**Gemma2 backbone prefill-only:**
- Optimum `--task text-generation-with-past` — CLI flag that produces decoder with `past_key_values.{i}.key/value` outputs. Use this instead of writing the KV extraction manually.
- Optimum issue #1755 — known bug log for Gemma ONNX numerical parity. Pre-emptively check.
- onnxruntime-genai #692 — Gemma2 hybrid sliding cache gotchas.

**RoPE (complex-free):**
- `ONNX RotaryEmbedding` native op (opset 23+) — TRT 10.x supports natively. If we bump opset, drop-in.
- `IRotaryEmbeddingLayer` (TRT builder API) — alternative to ONNX op.
- `lucidrains/rotary-embedding-torch` (MIT) — real-valued RoPE reference.
- `transformers/modeling_rope_utils.py` `apply_rotary_pos_emb` (Apache-2.0) — used by Gemma.

**In-graph preprocessor (could be a v0.3 differentiator):**
- `onnxruntime-extensions gen_processing_models` (MIT) — embeds SentencePiece tokenizer INTO the ONNX graph. No Python tokenization at inference.
- `yuniko-software tokenizer-to-onnx-model` (MIT) — alt converter, cross-language bindings.
- `big_vision PaliGemma readme` — exact normalize constants + tokenizer spec.

**Flow matching loop:**
- Isaac-GR00T `build_trt_pipeline.py --denoising-steps` — loop lives OUTSIDE the graph, called N times. **This is the recommended pattern.**
- `diffusers FlowMatchEulerDiscreteScheduler` — reference Euler math.
- `gle-bellier/flow-matching` notebook (MIT) — pedagogical reference.
- Optimum issue #2220 — confirms "loop outside graph, export velocity field only."

**Sinusoidal timestep:**
- `diffusers/models/embeddings.py get_timestep_embedding` — ONNX-clean `Timesteps` nn.Module (no Python loops).
- `fairseq prepare_for_onnx_export_()` — explicit ONNX-prep hook pattern.

**Dual-expert references:**
- `OpenGalaxea/GalaxeaVLA` (MIT) — another dual-expert VLA architecture comparison.
- `starVLA` — "lego-like" modular VLA.
- openpi issue #362 — canonical pi0-specific KV sharing discussion.

**Alternative pi0 PyTorch ports:**
- `ZibinDong/openpi_pytorch` — independent PyTorch port.
- `allenzren/open-pi-zero` (MIT) — already listed; second-check reference for attention.

---

## Debugging toolkit (must-have before day 1)

Install these BEFORE starting export work. Saves ~70% of bug-hunt time per the research.

**Tier 1 (non-negotiable):**

1. **Polygraphy** — `pip install polygraphy nvidia-pyindex`
   - `polygraphy run --trt --onnxrt --atol --rtol` — auto-diff ORT vs TRT, every output
   - `polygraphy run --onnx-outputs mark all` — mark every intermediate tensor, compare PyTorch vs ORT vs TRT activations node-by-node
   - `polygraphy debug reduce --mode=bisect` — **auto-bisects the failing subgraph**. Replaces the manual single_layer_diff pattern I built for SmolVLA.
   - `polygraphy surgeon sanitize --fold-constants` — fixes parser failures, run before every TRT build
   - `polygraphy debug precision` — forces layers to FP32 individually to localize FP16 overflow
   - docs: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/polygraphy/

2. **onnx-graphsurgeon** — `pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com`
   - Python IR for surgical node insertion/removal
   - Needed to patch Gemma attention reshape patterns + add debug taps

3. **Custom OnnxConfig for PaliGemma** — required; Optimum doesn't support image-text-to-text.

**Tier 2 (strongly recommended):**

4. **ORT per-node dumps** — `ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1` env var, zero-code per-node activation dumps to disk.

5. **sdpython/onnx-diagnostic** — `pip install onnx-diagnostic` — `validate_onnx_model()` compares PyTorch vs ORT with per-output max-diff.

6. **torch.onnx.export legacy tracer (`dynamo=False`)** — **CRITICAL:** NVIDIA's own Thor tutorial uses this. Dynamo has multiple 2026 regressions for Gemma-family (see gotchas).

7. **trtexec verbose mode** — `--verbose --dumpLayerInfo --exportLayerInfo=layers.json --dumpProfile` — shows per-layer precision, shape choices, fallback reasons. Ships with JetPack.

8. **onnx-simplifier with `--skip-shape-inference`** — `pip install onnxsim`. For 3B models; shape inference OOMs on PaliGemma-scale.

**Tier 3 (phase-specific):**

9. **Netron desktop** (not web) — https://github.com/lutzroeder/netron. Web version chokes >1GB; desktop survives.

10. **Our own `local_*_diff.py` port for pi0** — highest-ROI single action.

---

## Critical gotchas (memorize before coding)

These are drawn from community incident reports. Bypassing each saves hours.

1. **Gemma FP16 overflows in attention.** `trtexec --fp16` will NOT work on Gemma. Must use FP8/NVFP4 on Thor, or BF16 on Orin. Per Jetson AI Lab's explicit statement.

2. **`torch.onnx.export(dynamo=True)` is broken for Gemma-family in 2026.** Multiple bugs:
   - Gemma3 vmap failure (pytorch #160761)
   - Batch-1 hardcode when buffers present (#170172)
   - Dynamic KV cache raises `NameError: 'L' is not defined` (#172903)
   **Use `dynamo=False` (legacy tracer).** Matches NVIDIA's Thor tutorial.

3. **PaliGemma ONNX exceeds 2GiB protobuf limit** — must use external-data format. `torch.onnx.export(..., use_external_data_format=True)`.

4. **Gemma attention reshape with `-1` makes all dims dynamic** → breaks TRT FP4 block quantization. Replace with explicit dim calculation before export.

5. **BF16 RoPE loses precision** at long context. Force RoPE in FP32. (arXiv 2411.13476, transformers PR #29285)

6. **Gemma `sqrt(hidden)` embedding multiplier rounds wrong in BF16** — 55.4256 → 55.5. Compute in FP32 (PR #29402).

7. **GemmaRMSNorm crashes ONNX tracing** when weight attribute absent in adaptive-norm layers. Add guard in `extra_repr()` per Jetson AI Lab patches.

8. **Per-step cos must be ≥0.9999** to hit final ≥0.999 after 10 Euler steps. Error geometric: `final_cos ≈ (per_step_cos)^N`. Test BOTH granularities.

9. **NVIDIA's own bar on Thor is cos > 0.99 per-step**, not 0.999. We may be setting target too high. Reconsider: is cos > 0.99 per-step acceptable for MVP?

10. **JetPack 6.2 TRT 10.3 has known Orin Nano numerical bugs** — flat-plane outputs vs DGX TRT 10.14. Build engine on-device; never cross-host. (nvidia forum thread)

11. **SigLIP text-encoder export gets called twice during tracing** — second call passes None, breaks. Use wrapper module.

12. **HF processors return 5D pixel_values for vision-language models** — silently breaks 4D-expecting ONNX. (Our own Apr-17 bug for SmolVLM.)

13. **LeRobot pi0 config incompatible with latest lerobot** — draccus decode errors. Pin version. (lerobot #3355)

14. **TRT engines are not portable across Jetson hardware** — build on device.

15. **Gemma 2 layer_norm eps mismatch** — TRT uses different eps than HF (1e-6 fix). (TensorRT-LLM #4815)

16. **JAX → PyTorch weight conversion for pi0 must be verified layer-by-layer** before trusting downstream ONNX. Silent key-mismatch drops fine-tuned weights.

17. **Optimum doesn't know `image-text-to-text` task for PaliGemma** — `optimum-cli export onnx` raises KeyError. Write custom `OnnxConfig` subclass.

18. **Transformers needs patching for pi0.5 AdaRMSNorm + KV cache** before export works. (openpi_on_thor has patches.)

19. **FP16 engines show accuracy drop even when FP16 ONNX is clean** — per-layer precision bisect via Polygraphy required.

20. **Per-step velocity parity ≠ final action parity.** Compounds geometrically. Test per-step AT LEAST as strictly as final.

---

## Critical license update

**Gemma 2 (and therefore PaliGemma + pi0) is NOT Apache-2.0.** It's Google's custom **Gemma Terms of Use**. Commercial use permitted but with copyleft-like distribution obligation: every recipient must receive the full Gemma Agreement + NOTICE.

- pi0 / pi0.5 weights themselves are Apache-2.0 (PI's choice) ✓
- Gemma backbone underneath is proprietary-ish
- **Tacoin/openpi-pi0.5-libero-onnx's MIT license CANNOT override Gemma ToU upstream.** Use Tacoin as a parity oracle AT ARM'S LENGTH — diff against it locally, don't redistribute their ONNX.

**Practical implication for reflex-vla:** `reflex export` should auto-emit a `LICENSE_BUNDLE/` directory listing Apache-2.0 (pi0) + Gemma ToU (backbone) + attributions. This becomes a **feature** of `export-verification-report`, not a legal hazard.

**Import ordering:** pi0 first (Apache-2.0), never ship unmodified PaliGemma ONNX.

---

## Alternative paths (if ONNX keeps fighting)

1. **Torch-TensorRT direct (TorchScript → TRT)** — **strong fallback**. NVIDIA's Thor tutorial is already close to TorchScript (uses `dynamo=False` legacy tracer). Low pivot cost.
2. **TensorRT-LLM direct** — works for Gemma backbone but doesn't handle pi0 expert/action-head. Partial path only.
3. **Triton + PyTorch** — negates edge-latency story. A10G cloud only.
4. **BF16 ONNX → BF16 TRT engine (no quantization)** — ship slower but numerically safer. Skip quantization v1, add v2. **Recommended de-risked MVP.**

---

## Revised testing methodology

Per the research, the operational definition of "cos ≥ 0.999 end-to-end" should be:

- **Per 50-step action chunk**, mean cos_sim ≥ 0.999
- **Averaged across ≥32 seeded samples** from LIBERO validation distribution
- Camera-real images, real 8D state, real language strings (NOT random tensors)
- Per-step velocity cos ≥ 0.9999 (stricter than final, because geometric)

**Edge cases to automate as pytest fixtures:**
- Shared-noise fixture (deterministic seed + stored `.npz` snapshot)
- Identity check (zero-image/zero-state → ~1e-6 both backends)
- OOD check (COCO image → ONNX fails same way as PyTorch)
- KV boundary stress (seq_len 1023/1024)

**Industry-standard reference:** MLPerf Inference high-accuracy track = 99.9% of FP16 reference quality. Our 0.999 target matches this bar.

---

## Revised mess-up probability

Further de-risking from the second-pass research:

- **Before research:** ~60%
- **After first-pass research (Tier-1 silver bullets):** ~45%
- **After second-pass research (tools + gotchas + license):** ~25–35%

Decrease from second pass driven by:
- **Polygraphy auto-bisection** eliminates 70% of manual debugging
- **Gotcha list prevents 15 of 20 pre-documented bugs**
- **Methodology clarity** (per-step ≥0.9999, 32-sample LIBERO fixture) prevents "looked right but wasn't"
- **Legacy tracer requirement** (`dynamo=False`) prevents a known-bad default
- **Gemma ToU bundling** turns a legal risk into a product feature

**Remaining irreducible risk:**
- Unknown-unknowns in pi0-specific composition (~10%)
- Time overrun even with good tooling (~10%)
- Orin TRT 10.3 numerical bugs (~5%)

**Cannot go to 0%** — that would require someone already having shipped this exact stack. `openpi_on_thor` is close but targets Thor (FP8), not Orin Nano (FP16/BF16). We're still first to this specific hardware target.

---

## Sources index

- Jetson AI Lab OpenPi on Thor: https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/
- Tacoin/openpi-pi0.5-libero-onnx: https://huggingface.co/Tacoin/openpi-pi0.5-libero-onnx
- NVIDIA Isaac-GR00T deployment: https://github.com/NVIDIA/Isaac-GR00T/tree/main/scripts/deployment
- HuggingFace optimum-onnx: https://github.com/huggingface/optimum-onnx
- onnx-community/paligemma2-3b-pt-224: https://huggingface.co/onnx-community/paligemma2-3b-pt-224
- lerobot pi0 source: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py
- NVIDIA TensorRT-LLM Gemma: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma
- Dexmal/realtime-vla: https://github.com/Dexmal/realtime-vla
- allenzren/open-pi-zero: https://github.com/allenzren/open-pi-zero
- ainekko/smolvla_base_onnx: https://huggingface.co/ainekko/smolvla_base_onnx
- onnx-community/siglip2-so400m-patch14-384-ONNX: https://huggingface.co/onnx-community/siglip2-so400m-patch14-384-ONNX
- PaliGemma2 ONNX issue: https://discuss.huggingface.co/t/paligemma2-onnx-export-keyerror-unknown-task-image-text-to-text/139726
- GR00T N1.6 blog (FP8/Orin constraint): https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow
- transformers KV cache docs: https://huggingface.co/docs/transformers/kv_cache
- aifoundry-org/ETARS: https://github.com/aifoundry-org/ETARS

**Second-pass additions (2026-04-17):**
- AXERA-TECH/siglip-so400m-patch14-384: https://huggingface.co/AXERA-TECH/siglip-so400m-patch14-384
- NSTiwari/PaliGemma2-ONNX-Transformers.js: https://github.com/NSTiwari/PaliGemma2-ONNX-Transformers.js
- ZibinDong/openpi_pytorch: https://github.com/ZibinDong/openpi_pytorch
- lucidrains/rotary-embedding-torch: https://github.com/lucidrains/rotary-embedding-torch
- ONNX RotaryEmbedding op (opset 23+): https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html
- onnxruntime-extensions: https://github.com/microsoft/onnxruntime-extensions
- yuniko-software/tokenizer-to-onnx-model: https://github.com/yuniko-software/tokenizer-to-onnx-model
- OpenGalaxea/GalaxeaVLA: https://github.com/OpenGalaxea/GalaxeaVLA
- openpi issue #362 (KV cache design): https://github.com/Physical-Intelligence/openpi/issues/362
- Polygraphy docs: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/polygraphy/
- Polygraphy debug_accuracy: https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/debug_accuracy.md
- onnx-graphsurgeon: https://pypi.org/project/onnx-graphsurgeon/
- sdpython/onnx-diagnostic: https://deepwiki.com/sdpython/onnx-diagnostic/
- pytorch#160761 (Gemma3 vmap): https://github.com/pytorch/pytorch/issues/160761
- pytorch#170172 (dynamo batch hardcode): https://github.com/pytorch/pytorch/issues/170172
- pytorch#172903 (dynamo KV cache): https://github.com/pytorch/pytorch/issues/172903
- TensorRT-LLM#4815 (Gemma3 accuracy): https://github.com/NVIDIA/TensorRT-LLM/issues/4815
- TensorRT#2922 (FP16 accuracy drop): https://github.com/NVIDIA/TensorRT/issues/2922
- JetPack 6.2 TRT 10.3 Orin Nano bug: https://forums.developer.nvidia.com/t/365536
- Gemma Terms of Use: https://ai.google.dev/gemma/terms
- BF16 RoPE precision paper (arxiv 2411.13476): https://arxiv.org/html/2411.13476v2
- MLPerf Inference Edge: https://mlcommons.org/benchmarks/inference-edge/
