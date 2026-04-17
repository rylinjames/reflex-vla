# Reflex VLA — Git History Knowledge Base

Date range: 2026-03-01 through 2026-04-16 (inclusive).
Commits processed: 113 commits on `main`/feature branches merged into `main`.
Author: RomirJ (single author). Heavy Claude Opus 4.6 / 4.7 pair-programming, as reflected by `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` on most substantive commits.

Usage: grouped by THEME (not chronologically). Each theme has a commit range, what was accomplished, and per-commit detail (SHA, date, file footprint, learning worth re-reading). Use Ctrl-F on a keyword (GR00T, AdaRMSNorm, adaptive, TRT, LIBERO, vlm_kv, etc) to find prior art.

---

## Theme: v0.1.0 baseline — first Reflex CLI skeleton (commit range: e8fe39f..e8fe39f, 1 commit)

### What was accomplished
Laid down the whole project scaffolding in a single large commit: CLI (export/validate/benchmark/targets), hardware profiles, safetensors + HF Hub checkpoint I/O, ONNX-friendly RMSNorm + RoPE decomposition, naive split exporter (vision/backbone/denoising), trtexec wrapper, flow-matching denoising inference loop, numerical diff validation, latency measurement. 20 tests passing.

### Key commits
- **e8fe39f 2026-04-13 19:59** Reflex v0.1.0 — VLA deployment CLI with export pipeline, op decomposition, validation
  - Files: 19 files, +1094 LOC. Core modules: `cli.py`, `config.py`, `checkpoint.py`, `decompose.py`, `exporters/onnx_export.py`, `exporters/trt_build.py`, `inference.py`, `validate.py`, `benchmark.py`.
  - Learning: Initial abstraction "split VLA into vision/backbone/denoising" — this was superseded within 24 hours once SmolVLA's real architecture was discovered. Keep `decompose.py`'s RMSNorm + RoPE decompositions — they survive every subsequent rewrite.

---

## Theme: SmolVLA expert + VLM backbone discovery (commit range: 1ed46ab..fb9a317, 4 commits, all Apr-14 night)

### What was accomplished
End-to-end SmolVLA exploration on Modal A100: loaded `lerobot/smolvla_base`, reconstructed architecture from raw state_dict, exported each component (suffix encoder, action projection, expert GQA layer, full 16-layer expert stack, VLM backbone). First time SmolVLA runs end-to-end outside of lerobot. Landing performance: 253.3ms / 3.9Hz on A100 PyTorch eager, 1.82GB peak.

### Key commits
- **1ed46ab 2026-04-14 00:48** Real SmolVLA export working: suffix encoder + action projection to ONNX validated on A100
  - Files: +`src/reflex/models/smolvla.py` + 3 Modal test scripts (+1018 LOC).
  - Numbers: suffix encoder (1.58M) max_diff 2.15e-06; action projection max_diff 1.07e-06; 5.07ms mean / 197Hz (no expert transformer).
  - Learning: SmolVLA expert_hidden=720, action_dim=32, 450M total params.
- **74d24c3 2026-04-14 01:35** Expert transformer ONNX export working: GQA layer with decomposed RMSNorm + RoPE validated on A100
  - Files: +`scripts/modal_expert_export.py` (+422 LOC).
  - Learning: Expert arch = 720 hidden, 15 Q heads, 5 KV heads (GQA 3:1), head_dim=64, intermediate=2048, 16 layers, 98.2M. **RMSNorm decomposed to elementwise + RoPE decomposed to precomputed cos/sin tables → cleanly exports at opset 19.** GQA done via KV-head expansion before attention matmul.
- **47f3d5d 2026-04-14 01:46** Full 16-layer SmolVLA expert stack: 10-step denoise + ONNX export validated on A100
  - Files: +`scripts/modal_full_pipeline.py` (+441 LOC).
  - Learning: 16 layers = 8 self-attn even + 8 cross-attn odd (indices [1,3,5,7,9,11,13,15]). 99.8M params. Full 10-step Euler: 202.1ms / 4.9Hz. ONNX 1.1MB, max_diff 4.77e-06.
- **da237f5 2026-04-14 01:55** VLM backbone explored: SmolVLM2 vision encoder + truncated decoder fully characterized
  - Files: +`scripts/modal_vlm_export.py` (+301 LOC).
  - Learning: SmolVLM2 full=507.5M, **truncated to 16 layers = 350.2M** (86.4M vision + 263.8M decoder). Vision encoder output [1, 576, 768] from 512x512 image with 16px patches. Decoder: hidden=960, 15 heads, 5 KV heads GQA, head_dim=64. VLM prefix: 48.3ms on A100. 2.01GB peak GPU. NOTE: later corrected in Apr-16 GQA spike to **32 layers, not 16** (see VLM real forward theme).
- **fb9a317 2026-04-14 02:06** End-to-end SmolVLA inference working: image in, 50 actions out, 253ms/3.9Hz on A100
  - Files: +`scripts/modal_e2e_pipeline.py` (+397 LOC).
  - Learning: **VLM hidden (960) projected to cross-attn KV dim (320) for expert layers**. Cross-attn on odd indices. Actions range [-1.204, 1.670]. 1.82GB peak → fits Orin Nano 8GB.

---

## Theme: SmolVLA exporter unified as `reflex export` (commit range: c1726e7..c1726e7, 1 commit)

### What was accomplished
Unified the scripting work into a single CLI command: `reflex export lerobot/smolvla_base --target desktop` produces `expert_stack.onnx` + `.data` sidecar + `reflex_config.json`. Validated on Modal A100 at 64.9s export time.

### Key commits
- **c1726e7 2026-04-14 02:18** CLI export pipeline complete: reflex export lerobot/smolvla_base works end-to-end
  - Files: +`src/reflex/exporters/smolvla_exporter.py` (full load→build→ONNX→validate→TRT pipeline, +339 LOC), +`scripts/modal_cli_export.py`, CLI rewired.
  - Learning: Output files: `expert_stack.onnx` (406.4MB) + `.data` sidecar + `reflex_config.json` (16 layers, cross-attn indices, dimensions). ONNX max_diff 3.81e-06. `reflex targets` lists 5 hardware targets.

---

## Theme: 7-wedge product assembly — serve/guard/turbo/split/adapt/check (commit range: c6618e5..89aaae4, 4 commits Apr-14 02:40-02:47)

### What was accomplished
Laid down the full 7-wedge product surface: serve (realtime_runtime), guard (safety_monitoring), turbo (action_head_optimization), split (cloud_edge_orchestration), adapt (cross_embodiment_transfer), check (training_fine_tuning). Tests: 67 passing.

### Key commits
- **c6618e5 2026-04-14 02:40** Add reflex serve — VLA inference server (Wedge 2: realtime_runtime)
  - Files: +`src/reflex/runtime/server.py` (ReflexServer class, +239 LOC), CLI `reflex serve`.
  - Learning: FastAPI POST /act (image base64 + instruction + state → actions), GET /health, GET /config. ONNX Runtime backend with CUDA provider. `[serve]` extra = fastapi + uvicorn + Pillow.
- **44d6a93 2026-04-14 02:42** Add reflex guard — safety constraints + EU AI Act logging (Wedge 3: safety_monitoring)
  - Files: +`src/reflex/safety/guard.py` (+248 LOC).
  - Learning: ActionGuard with clamp + reject modes. EU AI Act Article 12 logging: timestamp, input hash, raw/safe actions, violations, model version. Sub-millisecond check time. Limits from URDF (yourdfpy), JSON config, or defaults.
- **b9b89b5 2026-04-14 02:44** Add reflex turbo — adaptive denoising + benchmarking (Wedge 4: action_head_optimization)
  - Files: +`src/reflex/kernels/turbo.py` (+184 LOC).
  - Learning: TurboOptimizer has Fixed (10-step Euler baseline) + Adaptive (early-stop on velocity convergence) strategies. Per-step velocity norm tracking. Configurable min/max steps, warmup, convergence threshold.
- **89aaae4 2026-04-14 02:47** Complete all 7 wedges: split, adapt, check (Wedges 5-7)
  - Files: +`src/reflex/runtime/split.py` (SplitOrchestrator, +217 LOC), +`src/reflex/models/adapt.py` (EmbodimentAdapter, +212 LOC), +`src/reflex/validate_training.py` (5 pre-deploy checks, +169 LOC).
  - Learning: Split = cloud/edge routing with health checking, latency history, fallback (last_action/zero). Adapt = URDF → framework configs (LeRobot/OpenPI/GR00T) + action-space mapping (pad/truncate/project). Check = loadable/size/structure/dtype/nan_inf pre-deploy gate.

---

## Theme: Simulated robot test gate (commit range: 751f9d2..751f9d2, 1 commit)

### What was accomplished
End-to-end simulated robot test on A100: 10 episodes, 7/7 pass.

### Key commits
- **751f9d2 2026-04-14 03:04** Simulated robot test: 10 episodes, safety validation, adaptive comparison on A100
  - Files: +`scripts/modal_sim_test.py` (+380 LOC).
  - Numbers: 298.5ms avg / 3.4Hz, 0/300 safety violations, 947MB peak.
  - Learning: **"Adaptive denoising: needs threshold tuning for real tasks (without VLM conditioning, velocities don't converge naturally)"** — this foreshadows the Apr-14 Phase IV finding + the whole VLM-real-forward push.

---

## Theme: pi0 + pi0.5 + GR00T exporters (commit range: 45794b0..ff9fc3a, 9 commits Apr-14 10:04-11:08)

### What was accomplished
Added 3 more VLA families to the supported model list. Auto-dispatch in `reflex export` + per-model exporters + `reflex models` listing + E2E benchmark across all 4. GR00T gets TWO commits: expert-only (shape mismatch bug), then full-stack (action_encoder + action_decoder wrap, closes the denoise loop).

### Key commits
- **45794b0 2026-04-14 10:04** Add pi0 support + fix serve startup (full E2E passes)
  - Files: +`src/reflex/exporters/pi0_exporter.py` (reuses SmolVLA's ExpertGQALayer/ExpertStack, +278 LOC), +`scripts/modal_e2e_demo.py` + `scripts/modal_test_pi0.py`, checkpoint auto-dispatch, runtime/server.py fix.
  - Learning: pi0 arch = 18 layers, 16Q/2KV GQA, head_dim=128. Prefix: `paligemma_with_expert.gemma_expert.model.*`. pi0 full export max_diff 3.73e-08. **Bug fix: FastAPI HealthResponse BaseModel defined inside create_app() → Pydantic 2.13 can't resolve ForwardRef for locally-scoped classes. Moved to module scope + switched from deprecated @app.on_event("startup") to async lifespan context manager. Also: E2E scripts redirect server logs to file, NOT subprocess.PIPE — 64KB buffer deadlocks child process.**
- **062470f 2026-04-14 10:06** dry-run: flag unsupported pi0.5 + unknown model types
  - Files: `src/reflex/cli.py` (+7 lines).
  - Learning: Dry-run now explicitly warns pre-AdaRMSNorm.
- **c0a3a7b 2026-04-14 10:12** pi0.5 support: AdaRMSNorm expert stack + CLI auto-dispatch
  - Files: +`DecomposedAdaRMSNorm` in `decompose.py`, +`ExpertAdaRMSLayer` + `Pi05ExpertStack` in `pi0_exporter.py`, tests.
  - Learning: **pi0.5 = time-conditioned RMSNorm (AdaRMSNorm): time_emb → dense → chunk(3) → x*rsqrt(var+eps)*(1+scale)+shift.** 3-chunk (scale/shift/gate), distinct from GR00T's 2-chunk AdaLN. Time MLP runs at stack level (separate from action, UNLIKE pi0 where time is concatenated with action). Verified `lerobot/pi05_base` 3.62B, 18 layers, 426.9M expert (pi0 was 314.6M — AdaRMS `dense` layers add ~112M). ONNX max_diff 2.37e-06.
- **ca7eb21 2026-04-14 10:14** Add `reflex models` + update README with supported VLA table
  - Files: `src/reflex/cli.py` +31 lines.
- **e887598 2026-04-14 10:22** E2E benchmark all 3 models + GR00T stub
  - Files: +`scripts/modal_e2e_all_models.py` + gr00t_exporter stub (+330 LOC).
  - Numbers: smolvla 47.3s/246ms, pi0 96.8s/588ms, pi05 95.3s/704ms (A100).
  - Learning: Latency scales with expert param count. Stub documents GR00T arch (diffusers-style attn, GEGLU, AdaLN 2-chunk, alternating cross/self-attn with VLM-KV at 2048) for the next implementer.
- **68119b7 2026-04-14 10:37** GR00T N1.6 support: DiT expert with AdaLN + alternating cross/self-attn
  - Files: gr00t_exporter.py grows to full implementation (+537 LOC), +`scripts/modal_test_gr00t.py`.
  - Learning: **GR00T = 32-block DiT (not GQA, 32-head MHA head_dim=48, hidden=1536).** Diffusers attention API (to_q/to_k/to_v/to_out.0, bias=True). AdaLN 2-chunk (scale+shift). Alternating: cross-attn on EVEN blocks (KV from VLM at 2048-dim), self-attn on ODD blocks (1536). **Plain GELU-approx MLP — NOT GEGLU: ff.net.0.proj outputs ff_inner directly, NOT 2×.** Non-affine LayerNorms. Output via final AdaLN + proj_out_2 → 1024. BFloat16 storage (cast to fp32 for export). 3.29B params, 448 DiT keys. Expert stack 1091.7M, 32 blocks. Export 63.5s, max_diff 2.18e-05.
- **da537ad 2026-04-14 10:38** README: GR00T joins supported models table
- **c00ca82 2026-04-14 10:43** OpenVLA: postprocess helper instead of redundant exporter
  - Files: +`src/reflex/exporters/openvla_exporter.py` (raises NotImplementedError), +`src/reflex/postprocess/openvla.py` (decode_actions, +129 LOC), checkpoint auto-dispatch, `reflex models` shows yellow for OpenVLA.
  - Learning: **OpenVLA is NOT a flow-matching VLA — its action head is `argmax(lm_logits[:, -7:])` + 256-bin lookup on top of Llama-2 vocab. There is no dedicated action expert to reconstruct.** HF optimum-onnx already handles Llama-2-7B + DINOv2 + SigLIP + projector. Building a full Reflex exporter here would duplicate optimum-onnx for zero architectural insight. Ship the postprocess helper + clear NotImplementedError.
- **9087da3 2026-04-14 10:52** E2E benchmark across all 4 models + GR00T serve limitation docs
  - Numbers: smolvla 417ms/2.4Hz, pi0 968ms/1.0Hz, pi05 1036ms/1.0Hz, gr00t export-only (A100).
  - Learning: **GR00T serve fails in denoising loop because expert emits velocity in 1024-dim action-token space while input is 1536-dim action tokens — `noisy + velocity*dt` can't cross dimensions. The proper fix is a GR00T-aware serve path with the action_decoder (1024→native DoF).** ONNX export itself still passes validation.
- **ff9fc3a 2026-04-14 11:08** GR00T full-stack export: raw actions in, raw actions out
  - Files: gr00t_exporter.py grows another +270 LOC, +`scripts/modal_probe_gr00t.py` + `scripts/modal_test_gr00t_full.py`, CLI default changes.
  - Learning: **Fix: wrap DiT expert with GR00T's per-embodiment action_encoder (3 linears) + action_decoder (2 linears) pinned to embodiment_id=0 by default.** Input and output are now both `[b, chunk, raw_action_dim=128]` so the denoise loop works. **Encoder: action(128) → W1(1536) → silu → cat(h1, time_emb) → W2(1536) → silu → (h2 + h1) → W3(1536). Residual + gating pattern.** Decoder: velocity_tokens(1024) → L1(1024) → silu → L2(128). Weight shape is `[embodiment, in, out]` with leading dim of 32 — slice at embodiment_id and transpose for F.linear compat. Full stack 1091.7M + 10M buffers. Export 80.5s, max_diff 3.77e-06. Serve ready 8s. POST /act 2.35s (CPU provider).

---

## Theme: 7-wedge CLI wiring + shell verification (commit range: 4c447e8..88b1e78, 3 commits Apr-14 11:08-11:35)

### What was accomplished
Plumbed the 7 wedges into actual CLI commands (4 were Python modules with no entry point). End-to-end shell verification on Modal (5/5 pass).

### Key commits
- **4c447e8 2026-04-14 11:08** README: all 4 VLAs now full-parity in reflex serve
- **e177ecb 2026-04-14 11:32** cli: wire the remaining 4 wedges (turbo, split, adapt, check)
  - Files: `src/reflex/cli.py` +178 LOC.
  - Learning: `reflex turbo` bench action-head denoising strategies (fixed/adaptive/cuda_graph); `reflex split` configure cloud/edge orchestration; `reflex adapt` URDF → embodiment summary + framework config (lerobot/openpi/gr00t); `reflex check` 5 pre-deploy checks. **All thin CLI wrappers over existing library code — plumbing the 7-wedge story.** Total after this: 11 commands (7 wedges + validate + benchmark + targets + models).
- **88b1e78 2026-04-14 11:35** Verify all 7 wedges runnable at shell (Modal, 5/5 pass)
  - Files: +`scripts/modal_verify_cli.py` (+106 LOC).

---

## Theme: Apr-14 GPU benchmark post-mortem — ORT silently CPU-fallback (commit range: f2cd906..5b21296, 2 commits Apr-14 12:09-13:13)

### What was accomplished
The landmark post-mortem of the morning. A100 benchmark revealed torch.compile appearing to beat ONNX 6-14x per denoising step. Root cause: **`onnxruntime-gpu` was silently falling back to CPU on Modal due to CUDA 12 vs 13 library mismatch.** Every "GPU" benchmark published earlier was actually CPU execution on an A100 box. This produced two fixes: (1) capture the full 10-step denoise loop as a CUDA graph to eliminate per-step Python-CUDA launch overhead; (2) make `reflex serve` refuse to silently fall back to CPU.

### Key commits
- **f2cd906 2026-04-14 12:09** turbo: implement denoise_cuda_graph (full-loop capture) + benchmarks
  - Files: +`src/reflex/kernels/turbo.py` (+107 LOC), +`scripts/modal_bench_onnx_vs_torch.py` + `scripts/modal_bench_path_b.py` (+595 LOC).
  - Numbers reported in the (wrong) benchmark: smolvla 19ms eager / 2.86ms compile / 29.1ms ONNX-CPU-fallback; pi0 23.9 / 5.56 / 76.9; pi0.5 28.6 / 6.12 / 80.9; gr00t 25.3 / 13.1 / 198.9.
  - Learning: **TurboOptimizer.denoise_cuda_graph preallocates static input/output tensors, runs 3-stream-warmup passes required by CUDA graph capture, then captures the 10-step loop. replay_cuda_graph() exposes sub-millisecond replay. In-place ops (.copy_, .add_) throughout to keep caching allocator happy during capture.** Also pins torch 2.5.1 (CUDA 12.4) + onnxruntime-gpu 1.20.1 to fix ORT CUDA provider loading. **Bonus: all 4 VLAs fit on Orin Nano 8GB in FP16 with 2x overhead (gr00t 4.36 GB, tight but fits). $500 Orin Nano is the right starter SKU, not Thor ($1500) or Orin 64 ($2500).**
- **5b21296 2026-04-14 13:13** Phase I.1: reflex serve refuses to silently fall back to CPU
  - Files: `src/reflex/runtime/server.py` (+100/-14 LOC), `src/reflex/cli.py` (+62 LOC), +`scripts/modal_verify_strict_providers.py` (+254 LOC).
  - Learning: **New contract (codified in tests TestStrictProviderMode): Strict + CUDA requested + CPU only → RuntimeError. ReflexServer gets `providers` + `strict_providers=True` args. CLI gets `--providers "A,B,C"` and `--no-strict-providers` flags. Pre-flight: if --device cuda but ORT has no CUDAExecutionProvider, print multi-line hint + exit 1.** Install-hint distinguishes "you installed onnxruntime not onnxruntime-gpu" footgun from "CUDA 12 libs missing from path" footgun.

---

## Theme: Wedge composition through serve flags (commit range: 9df6daa..9df6daa, 1 commit)

### What was accomplished
The 7-wedge pitch is no longer "7 separate CLIs" — users can layer them through one `reflex serve` command.

### Key commits
- **9df6daa 2026-04-14 13:27** Phase I.2: compose wedges in `reflex serve` via flags
  - Files: `src/reflex/runtime/server.py` (+208 LOC), `src/reflex/cli.py` (+47 LOC), +`scripts/modal_verify_wedge_compose.py` (+237 LOC).
  - Learning: ReflexServer accepts safety_config (path), adaptive_steps (bool), cloud_fallback_url (str), deadline_ms (float). Load builds ActionGuard + SplitOrchestrator when respective flags set. **`_run_denoise()` honors adaptive_steps (early-stops when velocity norm delta < 0.01, after at least 2 steps).** `predict()` composes wedges in order: denoise(adaptive or fixed) → guard.check() → deadline check → return. **Telemetry surfaced in response dict: safety_violations, safety_detail, adaptive_enabled, deadline_exceeded, deadline_misses_total, split_enabled.** Deadline miss falls back to last known good action (or zeros on first call) + logs miss count. Banner at startup shows composed wedges.

---

## Theme: TRT FP16 benchmark — the right numbers (commit range: d201a03..fce8a6f, 4 commits Apr-14 14:27-15:16)

### What was accomplished
Phase II: rebuilt the cloud-GPU benchmark with the correct image (nvcr.io/nvidia/tensorrt:24.10-py3, CUDA 12 + cuDNN 9 bundled) and discovered TRT FP16 wins 2.6-3.3x over torch.compile. Retired the "edge-only moat" framing for "cloud AND edge, same TRT toolchain."

### Key commits
- **d201a03 2026-04-14 14:27** Phase II.1: TRT FP16 engine benchmark script on A10G
  - Files: +`scripts/modal_bench_trt_fp16.py` (+341 LOC).
- **28e8906 2026-04-14 14:43** TRT bench: drop explicit shape flags (ONNX has static shapes)
  - Learning: **trtexec rejects --minShapes/--optShapes/--maxShapes when input ONNX has fully static shapes ("Static model does not take explicit shapes...").** Our exporters bake static shapes, so omit those flags. **Bonus finding: with nvcr.io/nvidia/tensorrt base image, ORT-GPU FP32 actually beats torch.compile on 3 of 4 models.** The earlier "torch.compile crushes us 6-14x" finding was a CPU-fallback artifact.
- **78caac2 2026-04-14 14:57** TRT bench: use trtexec --loadEngine instead of Python bindings
  - Learning: Python TRT bindings require `/opt/tensorrt/python/python_setup.sh` that the container doesn't run by default. Use trtexec --loadEngine for inference timing instead.
- **fce8a6f 2026-04-14 15:16** README + bench: TRT FP16 wins cloud GPU 2.6-3.3x over torch.compile
  - Numbers (A10G, single denoising step ms): smolvla 3.06 compile / 3.26 ORT-GPU / 0.95 TRT-FP16 (3.2x); pi0 6.23 / 5.53 / 1.94 (3.2x); pi05 7.34 / 7.37 / 2.24 (3.3x); gr00t 14.61 / 14.45 / 5.59 (2.6x).
  - Per-chunk wall-clock: smolvla 9.5ms=105Hz; pi0 19.4ms=52Hz; pi05 22.4ms=45Hz; gr00t 55.9ms=18Hz. **All four meet or exceed the 20-30Hz target for real-time robot control.**
  - Learning: "Same engine that runs on A10G via trtexec runs on Jetson via Jetson TRT — no separate cloud and edge model variants."

---

## Theme: Phase III — continuous batching in serve (commit range: 899c02e..526dded, 4 commits Apr-14 15:21-16:15)

### What was accomplished
Added `--max-batch` + `--batch-timeout-ms` flags. FastAPI /act queues requests, async worker drains into ONE batched ONNX inference. Real-model test: 2.88x throughput at batch=16.

### Key commits
- **899c02e 2026-04-14 15:21** Phase III: continuous batching in `reflex serve`
  - Files: `src/reflex/runtime/server.py` (+220 LOC), `src/reflex/cli.py` (+17 LOC), +`scripts/modal_verify_batching.py` (+205 LOC).
  - Learning: **start_batch_worker / stop_batch_worker async, called from FastAPI lifespan handler. Worker is a single coroutine owning an asyncio.Queue dispatching batches. predict_async() new front-door for HTTP path — falls through to predict() when batching is off. _batch_worker_loop() drains: blocks for first request, drains up to max_batch items within batch_timeout_ms. _predict_batch_sync() runs ONE ONNX inference with batch dim = N, splits output back to N response dicts. Each result reports batch_size, request_index, amortized_latency_ms. Per-item guard wedge applies AFTER batched inference (each request clamped individually). Telemetry: batches_run_total + batched_requests_total.** Initial verify used fake Identity ONNX model (isolates queueing from real perf).
  - Future work notes: per-item conditioning (image, instruction, state) ignored in v0.1 batching, landing with VLM prefix. Adaptive denoising in batch mode needs per-item convergence — deferred. **TRT engines static-shape (batch=1) — real batching with TRT needs dynamic batch shape via trtexec --minShapes/--optShapes/--maxShapes (Phase III.2).**
- **5b58d35 2026-04-14 15:34** Phase III.2 + launch drafts: real-model batch bench + 3 post drafts
  - Files: +`scripts/modal_verify_batching_real.py` (+182 LOC).
  - Learning: Re-run with actual VLA (pi0) on A10G instead of Identity fake. Real model has ~50ms per chunk → batching amortizes. Also launch drafts: lerobot_3146_draft.md, show_hn_draft.md, reddit_robotics_draft.md, README.md sequencing. **All drafts UNPUBLISHED — user must explicitly say "post the LeRobot one."**
- **b4d3552 2026-04-14 15:50** real-batch test: 180s timeout + stdout capture on failure
- **492a351 2026-04-14 16:04** real-batch test: switch to nvcr.io/tensorrt image (cuDNN bundled) + README cuDNN note
- **48d76fe 2026-04-14 16:15** README: real-model batching results (2.88x throughput at batch=16)
- **526dded 2026-04-14 16:15** launch drafts: include batching throughput numbers

---

## Theme: Phase IV — adaptive denoising validation on real VLAs (commit range: 1c40f14..091074c, 2 commits Apr-14 16:33-16:37)

### What was accomplished
Validated the 0.01 velocity-norm-delta threshold (from synthetic toy model) on real VLAs. Finding: **only pi0 is a real win. smolvla never triggers, pi0.5 rarely triggers and degrades when it does, gr00t triggers too aggressively.** Server now warns when `--adaptive-steps` used with anything other than model_type="pi0".

### Key commits
- **1c40f14 2026-04-14 16:33** Phase IV: adaptive denoising validation script (real VLAs)
  - Files: +`scripts/modal_verify_adaptive_real.py` (+217 LOC).
- **091074c 2026-04-14 16:37** Phase IV: adaptive denoising — honest per-model verdict + warnings
  - Files: `src/reflex/runtime/server.py` +15 lines.
  - Numbers (A10G, 25 trials): smolvla 0/25 triggered, 0% savings. pi0 25/25, mean_step 4.2, 58.4% savings, action_diff 0.073 (small OK). pi05 3/25, 9.4 step, 5.6% savings, 0.762 diff (large). gr00t 25/25, 3.0 step, 70% savings, 0.674 diff (large — meaningful drift).
  - Learning: "pi0 is the only model where adaptive is a real win." Per-model threshold tuning deferred to v0.2. Docs say: **only quote the pi0 number, not a universal one.**

---

## Theme: 30-minute Getting Started guide (commit range: 467d2a1..467d2a1, 1 commit)

### What was accomplished
User-facing walkthrough complementing README: install (CPU + GPU paths), pick a model and export, serve + curl /act, safety limits, fleet mode, full production-server invocation, pre-flight checks, common workflows, troubleshooting.

### Key commits
- **467d2a1 2026-04-14 16:31** docs: getting_started.md — 30-min guide for first hour after install
  - Files: +`docs/getting_started.md`.
  - Learning: Troubleshooting covers cuDNN, trtexec missing, model_not_loaded race, **"random-looking actions because v0.1 has no VLM conditioning yet."** README points to guide just before 3-command quickstart.

---

## Theme: Pre-launch install-path verification (commit range: 0bca6f6..7727ad9, 3 commits Apr-14 16:39-16:50)

### What was accomplished
Fresh-box E2E test that `pip install reflex-vla[serve,gpu]` → `reflex export` → `reflex serve` works. Caught missing core deps (huggingface_hub, transformers, onnx, onnxscript) that "should have been in core" but lived in extras.

### Key commits
- **0bca6f6 2026-04-14 16:39** Pre-launch: verify README install path on fresh box (full E2E)
  - Files: +`scripts/modal_verify_install_path.py` (+186 LOC).
- **85ab1ea 2026-04-14 16:44** install verify: use stdlib urllib instead of httpx (correctly tests user-facing deps)
- **7727ad9 2026-04-14 16:50** Fix install path: huggingface_hub + transformers + onnx + onnxscript core deps
  - Learning: **Restructured optional extras: [serve] = fastapi+uvicorn+Pillow (HTTP only); [onnx] = onnxruntime (CPU); [gpu] = onnxruntime-gpu + cuDNN/cuBLAS (GPU). Pick ONE runtime.**

---

## Theme: Auto-TRT-FP16 in serve (commit range: 12e604f..60ecd39, 3 commits Apr-14 17:08-17:19)

### What was accomplished
Made TRT FP16 the default path in `reflex serve`: prefer TensorrtExecutionProvider when available + warmup at startup so first /act doesn't time out.

### Key commits
- **12e604f 2026-04-14 17:08** serve: prefer TensorrtExecutionProvider when available
  - Files: `src/reflex/runtime/server.py` (+46 LOC).
  - Learning: **Per-export-dir engine cache at `<export_dir>/.trt_cache`, fp16 default, 4GB max workspace.** inference_mode field in /act responses: onnx_trt_fp16 / onnx_gpu / onnx_cpu. strict-providers check extended to count BOTH CUDAExecutionProvider and TensorrtExecutionProvider as "GPU."
- **9a690ab 2026-04-14 17:14** serve: warm up at startup so first /act doesn't time out
  - Files: `src/reflex/runtime/server.py` (+21 LOC).
  - Learning: **First inference triggers TRT engine construction (30-90s for smolvla, longer for larger). Without warmup, first POST /act would time out. Fix: in FastAPI lifespan startup hook, run server.predict() once before yielding control. /health won't return model_loaded=true until warmup completes → health-check based readiness probes correctly wait.** Subsequent server starts hit `.trt_cache` and skip the build (~1-2s). Install verify /health poll bumped 90s→240s.
- **60ecd39 2026-04-14 17:19** README: latency=11.9ms via auto TRT FP16 (84 Hz on smolvla, A10G)
- **72d8658 2026-04-14 17:20** launch drafts: auto-TRT-FP16 + 11.9ms latency story

---

## Theme: `reflex bench` command (commit range: a190780..9e3dabb, 3 commits Apr-14 17:32-18:00)

### What was accomplished
Previously a stub. Now actually loads export, warms up, runs N denoise iterations, reports min/mean/p50/p95/p99 latency + Hz. Verify-all script pip-installs from git, exports all 4 VLA families, runs bench against each.

### Key commits
- **a190780 2026-04-14 17:32** Add `reflex bench` command + verify-all script
  - Files: `src/reflex/cli.py` (+82 LOC), +`scripts/modal_verify_bench_all.py` (+144 LOC).
  - Learning: Defaults 100 iter + 20 warmup, --device cuda. Surfaces `inference_mode` so users can verify TRT FP16 is active.
- **3071cad 2026-04-14 17:44** rename CLI command from `benchmark` to `bench`
- **9e3dabb 2026-04-14 18:00** Headline numbers from reflex bench all 4 VLAs (auto-TRT FP16 e2e)

---

## Theme: LICENSE + TRT × batching sharp edge (commit range: 7651c76..e76678c, 2 commits Apr-14 18:17-18:25)

### What was accomplished
Codified the known sharp edge: when ORT TensorRT EP encounters a batch shape it doesn't have an engine for, it rebuilds the engine on EACH CALL. With our static-shape ONNX (batch=1), first batched request = 34+ seconds, every subsequent = rebuild again. Fix: drop TRT EP when max_batch > 1.

### Key commits
- **7651c76 2026-04-14 18:17** LICENSE (Apache 2.0) + TRT × batching interaction test
  - Files: +`LICENSE`, +`scripts/modal_verify_trt_with_batch.py` (+191 LOC).
- **e76678c 2026-04-14 18:25** serve: skip TRT EP when --max-batch > 1 (avoids 34s rebuild penalty)
  - Files: `src/reflex/runtime/server.py` (+22 LOC).
  - Numbers: --max-batch 1 → 37.7 qps normal; --max-batch 4 → 0.2 qps / 34121 ms per call (rebuild every time); --max-batch 8 → 0.2 / 35129 ms.
  - Learning: Drop TRT EP; ORT falls through to CUDAExecutionProvider which handles dynamic batch shapes natively + gives 2.88x throughput at batch=16. **Proper long-term fix is dynamic batch shape export + TRT shape profiles (batch=1/4/8/16) — deferred to v0.2.**

---

## Theme: `reflex doctor` diagnostics (commit range: b0dff64..1c6ea79, 2 commits Apr-14 18:52-18:53)

### What was accomplished
Quick health-check command users run BEFORE opening a bug. Verifies Python, platform, torch+CUDA, ORT providers, trtexec on PATH, disk space, installed reflex version.

### Key commits
- **b0dff64 2026-04-14 18:52** Add `reflex doctor` — diagnoses install + GPU issues
  - Files: `src/reflex/cli.py` +169 LOC.
  - Learning: Surfaces the failure modes that bit us: "ORT installed but CUDA EP missing" → fix cuDNN/cublas setup (the silent CPU fallback from benchmarking); "trtexec not on PATH" → TRT engine build skipped; "huggingface_hub missing" → catches install-path bug before reflex export blows up. **rich.Table output with green ✓ / yellow ⚠.**
- **1c6ea79 2026-04-14 18:53** docs: point users to reflex doctor as first troubleshoot step

---

## Theme: `reflex validate` round-trip harness — 5-phase RPI epic (commit range: e1455f7..18f8038, 5 commits Apr-16 06:49-07:05)

### What was accomplished
Full round-trip numerical validation: scaffold → pytorch + onnx backends → CLI wiring + tests → vibe fixups → post-mortem. Epic result: 11/11 tests passing, 7/7 issues closed, 3 waves / 50 max, council verdict PASS.

### Key commits
- **e1455f7 2026-04-16 06:49** crank wave 1: scaffold ValidateRoundTrip + fixtures + CI template
  - Files: +`src/reflex/validate_roundtrip.py` (+243 LOC), +`src/reflex/fixtures/__init__.py` + `vla_fixtures.py` (+83 LOC), +`src/reflex/ci_template.py` (+149 LOC), + RPI artifacts.
- **ac81175 2026-04-16 06:53** crank wave 2: add pytorch + onnx backends (aligned flow-matching schedule)
  - Files: +`src/reflex/_pytorch_backend.py` (+315 LOC), +`src/reflex/_onnx_backend.py` (+300 LOC).
  - Learning: **Two parallel workers implementing forward passes for different runtimes MUST share the same integration scheme — otherwise parity tests silently fail.** Inline hand-verification caught Issue 4 using `t: 0→1` and Issue 5 using `t: 1→0` (opposite direction). Plan needed a 10-line appendix documenting the canonical scheme.
- **7a601a0 2026-04-16 07:00** crank wave 3: wire CLI validate handler + tests (11 passing)
  - Files: `src/reflex/cli.py` (+112 LOC), `src/reflex/validate_roundtrip.py` (+119/-59 LOC), +`tests/test_validate_roundtrip.py` (+320 LOC), README +24 LOC.
  - Learning: **test_seed_bridge_in_orchestrator asserts Python `is` identity on noise array across backends — catches regression on the most subtle part of the system.** `torch.manual_seed` ≠ numpy seeded. Pass the noise tensor once as an arg.
- **7ac265f 2026-04-16 07:04** vibe fixups: input validation, public exports, KeyboardInterrupt, version default, CHANGELOG
  - Files: `src/reflex/__init__.py` (+15 LOC), `_pytorch_backend.py` (+9/-2), `ci_template.py` (+8/-2), `cli.py` (+4), `validate_roundtrip.py` (+23/-1).
  - Learning: **`except Exception: pass` is a smell — replaced with `logger.warning`.** Input validation in orchestrator constructors: `num_test_cases < 1` would produce confusing "passed=False with 0 diff" — now raises clearly.
- **18f8038 2026-04-16 07:05** vibe report + phase-5 summary
- **316c1d4 2026-04-16 07:06** post-mortem: 8 next-work items harvested, 5-phase rpi complete
  - Files: +`.agents/council/2026-04-16-post-mortem-reflex-validate.md`, +`.agents/rpi/phase-6-summary-2026-04-16-reflex-validate.md`.
  - Harvested next-work items (all in the post-mortem file):
    1. Benchmark FP16 torch.compile vs TRT FP16 on A10G (high) — closes apples-to-oranges pitch attack surface.
    2. Real Jetson Orin Nano validation via reflex validate (high) — every claim is A10G-extrapolated.
    3. **VLM prefix encoder + KV-cache export (critical) — v0.1 serve + validate both use random-tensor VLM conditioning, outputs are action-shaped noise, not task-relevant actions. Prefix KV-cache per Dexmal realtime-vla (arXiv 2510.26742). Estimate 2 weeks.** ← the single biggest unfilled thing, drives the VLM waves next.
    4. Install radon + gocyclo in dev env (low).
    5. Reconcile SmolVLA fixture image size 512 vs spec 384 (low).
    6. Cache HuggingFace downloads in CI workflow (medium).
    7. TypedDict for ValidateRoundTrip.run() return shape (low).
    8. Reproducible dev env (.venv bootstrapping) (medium).
- **398e06d 2026-04-16 12:38** Merge branch 'rpi/reflex-validate-roundtrip-20260416'

---

## Theme: VLM prefix pipeline v1 — stub waves (commit range: f72b8b9..d641134, 3 commits Apr-16 11:17-11:25)

### What was accomplished
First crank on the critical "VLM prefix encoder + KV-cache" item from the post-mortem. Stub exporter + server wired + 9 tests — just the pipeline, still using dummy weights/tensors.

### Key commits
- **f72b8b9 2026-04-16 11:17** vlm wave 1: prefix exporter stub + expert accepts vlm_kv input
  - Files: +`src/reflex/exporters/vlm_prefix_exporter.py` (+268 LOC), `src/reflex/exporters/smolvla_exporter.py` (+/-29 LOC).
  - Learning: Expert accepts vlm_kv input.
- **8a3b52c 2026-04-16 11:20** vlm wave 2: server wired to run VLM prefix + pass vlm_kv to expert
  - Files: `src/reflex/runtime/server.py` (+161/-10 LOC).
- **d641134 2026-04-16 11:25** vlm wave 3: 9 tests for VLM prefix pipeline (all passing)

---

## Theme: VLM real forward pass — GQA spike + 4-file split (commit range: 4daf6ea..9fb6ddb, 7 commits Apr-16 11:45-12:35)

### What was accomplished
Re-plan: from 3-file split to 4-file split (vision_encoder + text_embedder + decoder_prefill + expert_stack). GQA+RoPE ONNX spike de-risked the decoder (non-issue). Then implemented all 4 components on top. Tests: 25 passing.

### Key commits
- **4daf6ea 2026-04-16 11:45** plan: real VLM forward pass (3-file split) + tighten GOALS.yaml check
- **3d90808 2026-04-16 11:57** plan: revised VLM real forward (4-file split, GQA spike first)
- **5259b77 2026-04-16 11:27** fix GOALS.yaml: use .venv/bin/python for import checks
- **6fedff3 2026-04-16 12:10** spike: GQA+RoPE ONNX export PASSES (max_diff 4e-05, opset 19)
  - Files: +`.agents/crank/spike-gqa-result.md` (+73 LOC).
  - Learning: **SmolLM2's GQA decoder (LlamaDecoderLayer) exports to ONNX cleanly first try — no patches, no custom ops, no workarounds. Approach: torch.onnx.export at opset 19, single decoder layer wrapped with RoPE computation included in wrapper. PyTorch 2.11 new exporter (torch.export.export strict=False) under the hood.** GQA confirmed: 15 Q heads, 5 KV heads, hidden=960, head_dim=64. **IMPORTANT: 32 layers, not 16 — full SmolVLM2-500M has 32 layers. "This may affect the VLM prefix split point calculation."** RoPE = standard HF LlamaRotaryEmbedding, lives on `model.text_model.rotary_emb`, computes (cos, sin) externally, attention layer receives position_embeddings as (cos, sin) tuple. max_diff PyTorch-vs-ORT 4.01e-05, mean 7.57e-07.
- **5869a3e 2026-04-16 12:22** vlm real wave 1: vision encoder export + text/state components + dim fix 512→960
  - Files: +`src/reflex/exporters/vlm_components.py` (+352 LOC), `src/reflex/exporters/vlm_prefix_exporter.py` big rewrite (+427/-171 LOC), `src/reflex/runtime/server.py` 1-line fix.
  - Learning: **VisionEncoderForONNX wraps SigLIP vision encoder + SmolVLM connector. Pre-computes position IDs to avoid the dynamic `index_put` / `bucketize` loop in SmolVLMVisionEmbeddings.forward() that produces ONNX nodes with int64/float type mismatches ORT cannot load.** Input pixel_values [B, 3, 512, 512]; output image_embeds [B, 64, 960]. `patch_onnx_type_mismatches` post-fixes any remaining type mismatches in the ONNX graph. **Numerical validation threshold ORT_MAX_DIFF_THRESHOLD = 5e-4 — SigLIP's 27 transformer layers accumulate fp32 rounding, so max_diff ~2-4e-4 is expected.** Constants: DEFAULT_VLM_MODEL_NAME="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", DEFAULT_IMAGE_SIZE=512 (SigLIP-SO400M native), DEFAULT_VLM_KV_DIM=960 (SmolLM2 hidden + connector output). **Dim fix: 512 → 960 (was incorrectly typed as 512 earlier).**
- **d5b6570 2026-04-16 12:27** vlm real wave 2: decoder_prefill export + 4-file orchestrator wired into server
  - Files: `vlm_prefix_exporter.py` (+212 LOC), +`src/reflex/runtime/vlm_orchestrator.py` (+389 LOC), `src/reflex/runtime/server.py` big refactor (+147/-106 LOC).
  - Learning: **VLMPrefixOrchestrator loads 3 ONNX sessions lazily (vision_encoder required, text_embedder optional, decoder_prefill may not exist yet). Caches tokenizer/processor. State encoder weights: inline linear, no ONNX needed for 32→960 (MAX_STATE_DIM=32, HIDDEN_SIZE=960). Pipeline: (1) image → image_embeds [1,64,960]; (2) instruction → text_embeds [1,T,960]; (3) state → state_embed [1,1,960]; (4) assemble prefix [1,64+T+1,960]; (5) decoder prefill (if available).** Fallback if no decoder: return assembled embeddings without decoder pass.
- **9fb6ddb 2026-04-16 12:35** vlm real wave 3: 25 tests passing (fixed imports + new pipeline tests)

---

## Theme: VLM orchestrator bug fixes + validate backend vlm_kv (commit range: 0838336..7ed41aa, 3 commits Apr-16 12:41-12:52)

### What was accomplished
Post-landing fixes to the VLM pipeline: ONNX input name mismatch, validate backends handling vlm_kv, tokenizer loads from base VLM id not policy checkpoint.

### Key commits
- **0838336 2026-04-16 12:41** fix: ONNX input name mismatch (prefix_embeds→inputs_embeds) + add close() + init _state_session
  - Files: `src/reflex/runtime/vlm_orchestrator.py` (+15 LOC).
- **7ed41aa 2026-04-16 12:52** fix: validate backends handle vlm_kv input (dim 320 from ONNX shape, not 960)
  - Files: `src/reflex/_onnx_backend.py` (+16 LOC), `src/reflex/_pytorch_backend.py` (+2/-1).
  - Learning: **Read the actual vlm_kv dim from the ONNX input shape (expert.vlm_kv_dim), NOT the VLM hidden_size — they differ (320 vs 960 for SmolVLA).** Expert cross-attn projects VLM hidden(960) to KV-dim(320). v0.2+ exports include `vlm_kv` as a named input; validation feeds zeros.

---

## Theme: GOALS.yaml gates — sim-smoke-test + text-embedder (commit range: b78ab48..ce384a0, 2 commits Apr-16 13:05-13:22)

### What was accomplished
Added two regression gates to GOALS.yaml — sim-smoke-test runs after every cycle (weight 10), text-embedder-onnx goal (replaces seeded-random fallback, weight 10).

### Key commits
- **b78ab48 2026-04-16 13:05** GOALS.yaml: add sim-smoke-test regression gate (weight 10, runs after every cycle)
- **ce384a0 2026-04-16 13:22** GOALS.yaml: add text-embedder-onnx goal (weight 10, replaces seeded-random fallback)

---

## Theme: Tokenizer loads from correct checkpoint + deterministic text fallback (commit range: f882dcb..f882dcb, 1 commit)

### What was accomplished
Bug: tokenizer was being loaded from the SmolVLA POLICY checkpoint (which doesn't have a tokenizer), not the base VLM.

### Key commits
- **f882dcb 2026-04-16 13:19** fix: deterministic text fallback (seed by token IDs) + tokenizer loads from vlm_model_id not policy checkpoint
  - Files: `src/reflex/exporters/vlm_prefix_exporter.py` (+1 LOC), `src/reflex/runtime/vlm_orchestrator.py` (+16/-10 LOC).
  - Learning: **`vlm_model_id` (the BASE VLM, e.g. "HuggingFaceTB/SmolVLM2-500M-Video-Instruct") used for tokenizer/processor, NOT `model_id` (which is the SmolVLA policy checkpoint and doesn't have a tokenizer).** Config now writes `vlm_model_id = checkpoint_path_or_id` at export. **When text_embedder ONNX missing, fallback now seeds by token IDs so same instruction always maps to same embedding (deterministic vs random).**

---

## Theme: Trajectory replay + sim smoke test scripts (commit range: 0516d81..af22008, 11 commits Apr-16 13:34-14:11)

### What was accomplished
Added trajectory replay (replay a real LeRobot dataset against the server) and sim smoke test (local synthetic test). Hours of "can't get the images out of LeRobot v2" debugging ended with deciding the native loader was the right answer, then pivoting entirely to a synthetic-image integration test.

### Key commits
- **0516d81 2026-04-16 13:34** add trajectory replay (Modal) + sim smoke test (local) scripts
  - Files: +`scripts/modal_trajectory_replay.py` (+403 LOC), +`scripts/sim_smoke_test.py` (+506 LOC).
- **3ca3070 2026-04-16 13:41** fix: trajectory replay adds state field + logs /act errors
- **9bcbf6e 2026-04-16 13:42** fix: trajectory replay handles torch.Tensor + dict[path] images + adds diagnostics
- **12bb4e8 2026-04-16 13:45** fix: trajectory replay logs all dataset columns + tries more image keys
- **0d18154 2026-04-16 13:46** fix: switch to lerobot/pusht dataset (has images, xarm_lift_medium has none)
- **484d68d 2026-04-16 13:53** fix: trajectory replay uses LeRobotDataset native loader (handles video-encoded images)
- **7cb1cd7 2026-04-16 14:00** fix: install lerobot from GitHub (PyPI package lacks lerobot.common)
- **a355ea5 2026-04-16 14:06** fix: use datasets non-streaming (full download decodes images properly)
- **af22008 2026-04-16 14:11** fix: replace dataset replay with synthetic-image server integration test (LeRobot v2 images need custom loader)
  - Files: `scripts/modal_trajectory_replay.py` big rewrite (+98/-175 LOC).
  - Learning: **LeRobot v2 image columns are video-encoded; need custom loader. Gave up on replay and pivoted to synthetic-image server integration test.**

---

## Theme: Real SmolLM2 embed_tokens → `text_embedder.onnx` (commit range: 36d8a40..36d8a40, 1 commit)

### What was accomplished
Exported SmolLM2's actual `embed_tokens` layer to ONNX, replacing the seeded-random fallback.

### Key commits
- **36d8a40 2026-04-16 14:20** add text_embedder.onnx export (real SmolLM2 embed_tokens, replaces seeded-random fallback)
  - Files: `src/reflex/exporters/vlm_prefix_exporter.py` (+118/-3 LOC).
  - Learning: After this, `reflex export lerobot/smolvla_base` produces 4 files: vision_encoder.onnx, text_embedder.onnx, decoder_prefill.onnx, expert_stack.onnx. **Exported AFTER vision encoder, BEFORE decoder_prefill. Config updated with `text_embedder_onnx = "text_embedder.onnx"` and `decoder_prefill_onnx = "decoder_prefill.onnx"`.**

---

## Theme: LIBERO-10 benchmark via vla-eval adapter — long debugging march (commit range: 2d60d6d..2c597b6, 18 commits Apr-16 14:50-16:05)

### What was accomplished
Hooked `vla-eval` (external LIBERO benchmark tool) into a Modal script so we can actually run the LIBERO-10 suite against our exported model. Had to fight through: LIBERO's interactive `input()` prompts, broken pip-install from git, missing stdin, robosuite version incompat, mirror hash mismatches, and shell-quoting hell.

### Key commits
- **2d60d6d 2026-04-16 14:50** add LIBERO-10 eval script (vla-eval adapter + Modal A10G)
  - Files: +`scripts/modal_libero10.py` (+338 LOC).
- **9cc3a14 2026-04-16 14:56** fix: use debian_slim base for LIBERO (nvidia/cuda had mirror hash mismatch)
  - Learning: nvidia/cuda image had mirror hash mismatch on some deps — switched to debian_slim as base for LIBERO.
- **736ec03 2026-04-16 15:02** fix: CPU provider + run_server API compat for vla-eval 0.1.0
- **bf0c9f5 2026-04-16 15:23** fix: correct vla-eval API (run_server takes class, write config YAML, --no-docker)
  - Files: `scripts/modal_libero10.py` big rewrite (+71/-55).
- **af6acba 2026-04-16 15:25** fix: accept **kwargs in model server __init__ (vla-eval auto-injects parent args)
- **766185f 2026-04-16 15:28** fix: add LIBERO + robosuite to Modal image (was missing at runtime)
- **c4c0ac2 2026-04-16 15:32** fix: diagnose LIBERO import failure + retry install at runtime
- **40f1933 2026-04-16 15:43** fix: git clone + pip install -e for LIBERO (pip install from git doesn't work)
- **a38cfa6 2026-04-16 15:44** fix: skip LIBERO import check during build (reads from stdin)
- **a189177 2026-04-16 15:48** fix: set LIBERO_DATA_DIR + LIBERO_BASE env vars (skips stdin prompt)
- **6cae528 2026-04-16 15:50** fix: patch LIBERO __init__.py to replace input() with env var (no stdin in containers)
  - Learning: LIBERO reads stdin during import for a "custom path wizard" — no stdin in containers.
- **8862afc 2026-04-16 15:51** fix: use sed to patch LIBERO input() (simpler, no quoting issues)
- **a9ef6c9 2026-04-16 15:53** fix: replace all 3 LIBERO input() calls with 'n' (decline custom path wizard)
- **6ef91a3 2026-04-16 15:54** fix: use python3 regex to patch LIBERO input() calls (avoids shell quoting)
- **dd03edb 2026-04-16 15:55** fix: separate patch_libero.py script (avoids shell quoting issues entirely)
- **91e3fd0 2026-04-16 15:56** fix: move patch_libero.py copy before LIBERO install step in image build
- **cb742db 2026-04-16 15:57** fix: aggressive regex patch for ALL input() patterns + nuke .pyc caches
- **2d29065 2026-04-16 15:58** debug: dump LIBERO __init__.py lines 60-80 before patch
- **b72d2ca 2026-04-16 16:00** fix: handle multi-line input() calls in LIBERO patch (split across 3 lines)
- **2c597b6 2026-04-16 16:05** fix: pin robosuite==1.4.1 for LIBERO compat (1.5+ changed module paths)
  - Learning: robosuite 1.5+ changed module paths — pin 1.4.1.

Pattern: **burst of 18 commits over ~75 minutes signaling a problem being chased.** The eventual full working path: separate `scripts/patch_libero.py`, git-clone LIBERO, pip-install-e it, set LIBERO_DATA_DIR+LIBERO_BASE env vars, patch all 3 input() calls with regex, nuke .pyc caches, pin robosuite==1.4.1, use debian_slim base.

---

## Theme: CLI unification — Apr-16 final session (commit range: fdd9bb3..ed8157c, 5 commits Apr-16 22:14-22:23)

### What was accomplished
Steps 1-5 of a planned CLI cleanup: unify `reflex export` (SmolVLA produces all 4 ONNX files in one command); deprecate turbo/split/adapt (hidden from --help, warn + forward); merge check into validate --quick (with --pre-export for raw checkpoints); add --benchmark flag to reflex bench (plugin framework for LIBERO/SimplerEnv/ManiSkill); scaffold reflex distill (DMPO + pi-Flow recipes, training loop in v0.2.1).

### Key commits
- **fdd9bb3 2026-04-16 22:14** step 1: unify reflex export — SmolVLA now produces all 4 ONNX files in one command
  - Files: `src/reflex/cli.py` (+37/-4 LOC).
  - Learning: **For SmolVLA: after expert export, also call `export_vlm_prefix` so `reflex serve` can run with real task-conditioned actions instead of noise. VLM weights come from BASE SmolVLM2-500M — fine-tuned SmolVLA VLM weight transfer tracked as v0.3 item.** Printed note to user: "VLM uses base SmolVLM2-500M weights. Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)."
- **a90a4ab 2026-04-16 22:16** step 2: deprecate turbo/split/adapt commands (hidden from --help, warn + forward)
  - Files: `src/reflex/cli.py` (+28/-126 LOC).
  - Learning: Net -98 LOC. These were experiment-level commands that made the CLI surface too wide for v0.2.
- **369f70c 2026-04-16 22:19** step 3: merge check into validate --quick (+ --pre-export for raw checkpoints)
  - Files: `src/reflex/cli.py` (+106/-16 LOC).
- **c768c54 2026-04-16 22:21** step 4: add --benchmark flag to reflex bench (plugin framework for LIBERO/SimplerEnv/ManiSkill)
  - Files: `src/reflex/cli.py` (+57/-4), +`src/reflex/eval/__init__.py` (+49 LOC), +`src/reflex/eval/libero.py` (+75 LOC), +`src/reflex/eval/maniskill.py` (+12 LOC), +`src/reflex/eval/simpler.py` (+12 LOC).
- **ed8157c 2026-04-16 22:23** step 5: scaffold reflex distill (DMPO + pi-Flow recipes, training loop in v0.2.1)
  - Files: `src/reflex/cli.py` (+71), +`src/reflex/distill/__init__.py` (+36), +`src/reflex/distill/dmpo.py` (+87), +`src/reflex/distill/pi_flow.py` (+33).
  - Learning: Training loop deferred to v0.2.1 — this is a scaffolding commit, the recipes are not operational yet.

---

## Cross-theme patterns worth knowing

### Bursts → signal a problem being chased
- Apr-14 10:04-11:08 (15 min burst): pi0 support + pi0.5 auto-dispatch + GR00T stub + OpenVLA helper. This was the "fill out the supported-VLA table" push.
- Apr-14 12:09 → 13:13 (~1 hour, 2 big commits): GPU post-mortem response — turbo cuda_graph + strict providers.
- Apr-14 15:34-16:15 (41 min, 4 commits): real-model batching lands.
- Apr-16 13:34-14:11 (37 min, 9 commits): trajectory replay image-format fighting.
- Apr-16 14:50-16:05 (75 min, 18 commits): LIBERO integration death march.

### Consistent reversion pattern
- Static-shape ONNX assumptions got patched multiple times (trtexec flags dropped, TRT EP skipped under batching). The TRT × shape interaction is the single most-revisited failure mode.
- VLM conditioning: v0.1 ran with random conditioning ("action-shaped noise"), then seeded-random fallback by token IDs (f882dcb), then real SmolLM2 embed_tokens ONNX (36d8a40), then unified into `reflex export` (fdd9bb3). Each step was the honest v1 choice at the time.
- Adaptive denoising: promised as a universal win, validated as only-pi0 (091074c), now gated behind a per-model warning. Threshold tuning deferred to v0.2.

### New files added (big ones)
- `src/reflex/runtime/vlm_orchestrator.py` (d5b6570) — 4-file orchestrator.
- `src/reflex/exporters/vlm_components.py` (5869a3e) — VisionEncoderForONNX, state encoder inline.
- `src/reflex/_pytorch_backend.py` + `_onnx_backend.py` (ac81175) — round-trip validation backends.
- `src/reflex/validate_roundtrip.py` (e1455f7) — the round-trip harness.
- `src/reflex/ci_template.py` (e1455f7) — GitHub Actions template.
- `src/reflex/fixtures/vla_fixtures.py` (e1455f7) — test fixtures.
- `src/reflex/eval/libero.py` + `maniskill.py` + `simpler.py` (c768c54) — benchmark adapter stubs.
- `src/reflex/distill/dmpo.py` + `pi_flow.py` (ed8157c) — distill recipe stubs.
- `scripts/patch_libero.py` (dd03edb) — LIBERO input() regex patcher.

### Files never deleted
- `src/reflex/exporters/onnx_export.py` (from e8fe39f original) — the "split into vision/backbone/denoising" initial exporter, superseded by per-family exporters but still present.
- `src/reflex/exporters/openvla_exporter.py` — stub that raises NotImplementedError, kept as documentation.

### Key architectural constants worth re-reading
- SmolVLA: expert_hidden=720, action_dim=32, 16 layers, 15Q/5KV GQA, head_dim=64, intermediate=2048. Cross-attn on odd indices. VLM hidden(960) → KV-dim(320) projection.
- SmolVLM2-500M (base VLM): **32 layers total** (not 16 as initially assumed), vision 86.4M + decoder 263.8M = 350.2M truncated / 507.5M full. Vision output [B, 576, 768] from 512×512 with 16px patches. Decoder hidden=960.
- pi0: 18 layers, 16Q/2KV GQA, head_dim=128, 314.6M expert. Prefix `paligemma_with_expert.gemma_expert.model.*`.
- pi0.5: 18 layers, AdaRMSNorm 3-chunk, 426.9M (AdaRMS `dense` layers add ~112M vs pi0).
- GR00T N1.6: 32-block DiT (no GQA, 32-head MHA head_dim=48, hidden=1536), AdaLN 2-chunk (scale+shift), alternating cross/self, 1091.7M, plain GELU-approx MLP (NOT GEGLU).
- OpenVLA: NOT flow-matching — argmax(lm_logits[:, -7:]) + 256-bin lookup on Llama-2 vocab.
