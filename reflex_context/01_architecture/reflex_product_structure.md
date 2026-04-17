# Reflex Product Structure — 7-Wedge Surface (Now 5 After Consolidation)

**Status as of 2026-04-16:** Reflex shipped v0.1 with seven user-facing CLI wedges plus internal helpers. The Apr-16 CLI-unification pass (commits `fdd9bb3`..`ed8157c`) consolidated the surface to **5 public wedges** plus `distill` (scaffolded) and `doctor`/`models`/`targets` utilities. Three wedges are deprecated (still present, hidden from `--help`, warn + forward). One planned wedge (`distill`) has CLI scaffolding and recipe stubs but no training loop until v0.2.1.

**Tagline (README):** "The deployment layer for VLAs — take a Vision-Language-Action model off the training cluster and onto a robot."

**Mission (GOALS.yaml):** "Deploy any VLA model to any edge hardware. One command."

---

## Current public surface (post `ed8157c`)

```
reflex export    # checkpoint → ONNX (+ TensorRT engine on device)
reflex serve     # FastAPI inference server with composable wedges
reflex guard     # URDF-derived safety limits + EU AI Act logging  (library-only after deprecation)
reflex bench     # latency benchmark + --benchmark LIBERO/SimplerEnv/ManiSkill
reflex validate  # ONNX/TRT-vs-PyTorch round-trip parity + --quick (merged check) + --pre-export
reflex distill   # [scaffolded] pi-Flow / DMPO recipes; training loop lands v0.2.1
reflex doctor    # diagnostics: Python, torch+CUDA, ORT providers, trtexec, disk, install
reflex models    # lists supported VLA families + status
reflex targets   # lists hardware profiles (orin-nano, orin-agx, thor, desktop)
```

Deprecated (warn + forward, hidden from `--help` per commit `a90a4ab`):
- `reflex turbo`  → use `serve --adaptive-steps`
- `reflex split`  → use `serve --cloud-fallback <url>`
- `reflex adapt`  → removed; no forwarding replacement (cross-embodiment retraining deferred)
- `reflex check`  → use `validate --quick` (and `--pre-export` for raw checkpoints)

Internal (not CLI-invokable per design decision at current_session line 8097: *"No user should ever call `export_vlm_prefix()` directly. It should be internal."*):
- `export_vlm_prefix()` — called by `reflex export` for SmolVLA automatically after unification commit `fdd9bb3`
- `reflex.runtime.adapters.vla_eval` — WebSocket adapter for AllenAI's vla-eval; used by `bench --benchmark libero_10`

---

## Wedge-by-wedge

### 1. `reflex export` — checkpoint → ONNX artifacts

**Purpose:** Take a VLA checkpoint (from HuggingFace Hub or a local path) and emit a self-describing deployment artifact: one or more ONNX files plus a `reflex_config.json` that records `model_type`, `action_dim`, `chunk_size`, VLM file list, and target hardware metadata.

**CLI command:**
```
reflex export <model_id_or_path> --target <profile> [--output <dir>] [--dry-run]
```

**Status:** Shipped in v0.1.0 (commit `c1726e7`, 2026-04-14). Unified for SmolVLA in `fdd9bb3` (2026-04-16) so `reflex export lerobot/smolvla_base` auto-produces all 4 ONNX files (vision_encoder, text_embedder, decoder_prefill, expert_stack) instead of requiring a separate `export_vlm_prefix` invocation.

**Supported model families (from `reflex models` + commit history):**
- SmolVLA (`1ed46ab`..`c1726e7`) — 450M total, 98M expert, 350M VLM
- pi0 (`45794b0`) — 314.6M expert, 18 layers, 16Q/2KV GQA, head_dim=128
- pi0.5 (`c0a3a7b`) — 426.9M expert with AdaRMSNorm 3-chunk time conditioning
- GR00T N1.6 (`68119b7`, `ff9fc3a`) — 1091.7M DiT expert, 32 blocks, 32-head MHA, AdaLN 2-chunk, alternating cross/self-attn
- OpenVLA (`c00ca82`) — postprocess helper only; `optimum-onnx` handles Llama-2-7B + DINOv2 + SigLIP. `postprocess/openvla.decode_actions` does argmax + 256-bin lookup on top-7 vocab positions.

**Key flags:**
- `--target {orin-nano|orin-agx|thor|desktop}` — selects fp16/fp8 + memory budget
- `--output <dir>` — artifact directory; contains `expert_stack.onnx` + `.data` sidecar + `reflex_config.json`
- `--dry-run` — probes checkpoint, prints detected structure, skips export

**Hardware profiles (commit `e8fe39f`):**
- `orin-nano` — Jetson Orin Nano, 8GB, fp16
- `orin-agx` — Jetson Orin AGX, 64GB, fp16/fp8
- `thor` — Jetson Thor, 128GB, fp8
- `desktop` — 4090/A100/H100, 24+GB, fp16

**Validation threshold (commit `c1726e7`):** Exporter validates PyTorch vs ONNX at `max_diff < 1e-5` per-layer for smolvla (reports 3.81e-06), 3.73e-08 for pi0, 2.37e-06 for pi0.5, 2.18e-05 for GR00T.

---

### 2. `reflex serve` — FastAPI inference server

**Purpose:** Long-lived HTTP server that loads an export directory and serves `POST /act` + `GET /health` + `GET /config`. Composable wedges are flags on this one command instead of separate CLIs (per commit `9df6daa`, 2026-04-14, "Phase I.2: compose wedges in reflex serve via flags").

**CLI command:**
```
reflex serve <export_dir> [--device cuda|cpu] [--port 8765] [--safety-config path.json]
             [--adaptive-steps] [--deadline-ms 50] [--cloud-fallback https://...]
             [--max-batch N --batch-timeout-ms 20] [--api-key <key>]
             [--no-strict-providers] [--providers "A,B,C"]
```

**Status:** Shipped in v0.1.0 (commit `c6618e5`, 2026-04-14).

**Key flags and behaviors:**
- `--device cuda` is default on GPU boxes; `reflex serve` **refuses to silently fall back to CPU** after Apr-14 post-mortem (commit `5b21296`, "Phase I.1"). If CUDA requested but ORT only reports `CPUExecutionProvider`, exits 1 with multi-line install hint.
- Auto-prefers `TensorrtExecutionProvider` when available (commit `12e604f`). Engine cache at `<export_dir>/.trt_cache`, fp16 default, 4GB max workspace. Response `inference_mode` field: `onnx_trt_fp16` / `onnx_gpu` / `onnx_cpu`.
- Warms up at startup (commit `9a690ab`): first inference triggers TRT engine build (30-90s for smolvla, longer for pi0/gr00t). `/health` returns `model_loaded: false` until warmup completes, so readiness probes correctly wait.
- `--max-batch > 1` drops TRT EP and falls through to CUDA EP (commit `e76678c`). TRT engines compile per input shape, so batched shape triggers rebuild per call (34s/call observed — 200× pessimization). Long-term fix is dynamic batch shape export + TRT shape profiles; deferred to v0.2.
- `--adaptive-steps` gates early termination by velocity-norm delta `< 0.01` after ≥2 steps. **Per commit `091074c` (Phase IV), only pi0 is a real win** (25/25 trigger, mean_step 4.2, 58.4% savings, action_diff 0.073). smolvla never triggers; pi0.5 rarely triggers and degrades when it does (0.762 diff); gr00t triggers too aggressively (0.674 diff). Server warns when enabled on non-pi0 models. GOALS.yaml `adaptive-denoise-fix` (weight 5): "works on pi0, gated behind `--experimental` for smolvla/pi0.5/gr00t (unsafe)".
- `--safety-config` loads `SafetyLimits` JSON; `ActionGuard` clamps per-joint position/velocity. EU AI Act Article 12 log records timestamp, input hash, raw/safe actions, violations, model version. Sub-millisecond check time.
- `--deadline-ms` returns last-known-good action (or zeros on first call) when inference misses the budget. Telemetry: `deadline_misses_total`.
- `--cloud-fallback` configures `SplitOrchestrator` with health checks + latency history; unused pending ≥2 design partners explicitly requesting cloud offload (ADR `2026-04-14-wrap-not-rebuild-vla-eval.md` records this policy).

**Telemetry surfaced in `/act` response:**
`latency_ms`, `hz`, `inference_mode`, `batch_size`, `request_index`, `amortized_latency_ms`, `safety_violations`, `safety_detail`, `adaptive_enabled`, `deadline_exceeded`, `deadline_misses_total`, `split_enabled`, `vlm_conditioning`.

---

### 3. `reflex guard` — safety + compliance (library-only after deprecation cleanup)

**Purpose:** Derive URDF-based joint limits, clamp actions at serve time, emit EU AI Act Article 12 audit log (timestamp, input hash, raw actions, safe actions, violations, model version).

**CLI command (historical, before wedge cleanup):**
```
reflex guard --urdf <path> --output safety.json
```

**Status:** Shipped in v0.1.0 (commit `44d6a93`). Per GOALS.yaml, `nan-guard-hardening` (weight 7) wants "Guard rejects NaN/Inf actions and halts after N consecutive clamps (staleness kill-switch)" — not yet implemented. `sqlite-audit-log` (weight 3) wants "reflex serve --audit-db appends every /act call to a SHA-256 hash-chain SQLite log" — not built.

**Key interface:** Consumed by `reflex serve --safety-config safety.json` (since Apr-14 wedge composition commit `9df6daa`). The standalone `reflex guard` CLI path still exists as a URDF→JSON converter.

---

### 4. `reflex bench` — latency + task-success

**Purpose:** Load an export, warm up, run N iterations of the flow-matching denoise, report `min/mean/p50/p95/p99` latency + Hz. Also wraps task-success benchmarks via `--benchmark {libero_10|simpler|maniskill}`.

**CLI command:**
```
reflex bench <export_dir> [--iterations 100] [--warmup 20] [--device cuda|cpu]
             [--benchmark libero_10|simpler|maniskill] [--episodes N]
```

**Status:** Shipped in v0.1.0 as `benchmark`, renamed to `bench` (commit `3071cad`). `--benchmark` flag + plugin framework for LIBERO/SimplerEnv/ManiSkill landed in `c768c54` (2026-04-16 step 4). `src/reflex/eval/libero.py`, `simpler.py`, `maniskill.py` are adapter stubs.

**Key numbers (from `9e3dabb` + current_session line 5522, A10G, auto-TRT-FP16):**
| Model   | mean_ms | p95_ms | mode           | export_s | Hz |
|---------|---------|--------|----------------|----------|------|
| SmolVLA | 11.67   | 11.85  | onnx_trt_fp16  | 72.8     | 86   |
| pi0     | 23.57   | 24.22  | onnx_trt_fp16  | 112.2    | 42   |
| pi0.5   | 27.07   | 27.76  | onnx_trt_fp16  | 151.9    | 37   |
| GR00T   | 56.55   | 57.25  | onnx_trt_fp16  | 181.9    | 18 (borderline; needs optimization) |

All four meet or exceed the 20-30 Hz target for real-time robot control (except GR00T at 18 Hz).

---

### 5. `reflex validate` — round-trip numerical parity

**Purpose:** Run a real PyTorch-vs-ONNX/TRT round-trip parity check with seeded fixtures. Replaces the v0.1 stub that accepted any path. Output formats: JSON (machine-readable), Rich table (human), `--init-ci` emits `.github/workflows/reflex-validate.yml`.

**CLI command:**
```
reflex validate <export_dir> [--threshold 1e-4] [--quick] [--pre-export] [--json] [--init-ci]
```

**Status:** Unreleased in CHANGELOG (as of 2026-04-16). 5-phase RPI epic commits `e1455f7`..`18f8038` (2026-04-16 06:49-07:05). 11/11 tests passing. Post-mortem at `.agents/council/2026-04-16-post-mortem-reflex-validate.md`.

**BREAKING change:** Default `--threshold` changed from `0.02` (v0.1 placeholder that never actually validated anything) to `1e-4`. Pass `--threshold 0.02` explicitly to match previous behavior. `reflex validate` now requires a valid `reflex_config.json` inside the export directory.

**Subcommand flags:**
- `--quick` — merged from deprecated `reflex check`. Runs 5 pre-deploy checks: loadable, size, structure, dtype, nan_inf (commit `369f70c`, step 3)
- `--pre-export` — targets raw checkpoint (not an export dir) for the same 5 checks

**Public exports on `reflex` module:** `ValidateRoundTrip`, `load_fixtures`, `SUPPORTED_MODEL_TYPES`.

**Key learning (commit `7a601a0`):** `test_seed_bridge_in_orchestrator` asserts Python `is` identity on the noise array across backends. `torch.manual_seed` ≠ numpy seeded — the noise tensor must be passed once as an arg. Without this, flow-matching initial conditions diverge and the cos_sim test becomes meaningless. See `modal_pytorch_vs_onnx.py` for the same lesson captured in code.

---

### 6. `reflex distill` — [scaffolded] flow-matching step distillation

**Purpose:** Paid-tier anchor (per ADR `2026-04-14-ship-distill-first.md`). Train a 2-step student from a 10-step teacher using pi-Flow velocity-field matching (arXiv 2510.14974). Target <5% LIBERO accuracy drop, ~1770 Hz student throughput (DMPO alternative from arXiv 2601.20701 achieves true one-step at 1770 Hz without a teacher — see prior_sessions.md line 6961).

**CLI command (scaffolded only):**
```
reflex distill --recipe {dmpo|pi_flow} <teacher_export_dir> --output <student_dir>
```

**Status:** Scaffolded in `ed8157c` (step 5 of CLI unification, 2026-04-16). `src/reflex/distill/__init__.py`, `dmpo.py`, `pi_flow.py` exist but are recipe stubs. **Training loop deferred to v0.2.1.** GOALS.yaml `distill-dmpo` (weight 9): "reflex distill implements DMPO one-step generation (no teacher) targeting 1000+ Hz on consumer GPUs".

**Compute estimate (current_session line 5756 correction):** ~$200-500 A10G-hours, not $10. Random noise alone produces nonsense students — requires real (image, state, language) triples from LeRobot/LIBERO/DROID.

---

### 7. `reflex doctor` — environment diagnostics

**Purpose:** Run BEFORE opening a bug. Verifies Python version, platform, torch+CUDA, ORT providers, `trtexec` on PATH, disk space, installed reflex version. Surfaces the footgun failure modes from the Apr-14 post-mortem.

**CLI command:**
```
reflex doctor
```

**Status:** Shipped in `b0dff64` (2026-04-14). Uses `rich.Table` with green ✓ / yellow ⚠ markers.

**What it catches:**
- "ORT installed but CUDA EP missing" → the silent CPU fallback footgun (Apr-14 Modal benchmarks were all CPU on A100 boxes before this fix)
- "trtexec not on PATH" → TRT engine build will be skipped
- "huggingface_hub missing" → `reflex export` will fail; caught before user hits it
- "CUDA 12 vs 13 mismatch" → `libcublasLt.so.12` not found; install hint specific to `nvcr.io/nvidia/tensorrt:24.10-py3`

---

## Deprecated wedges (still present, warn + forward)

### `reflex turbo` → `serve --adaptive-steps`
**Original purpose:** Benchmark action-head denoising strategies (fixed/adaptive/cuda_graph). `TurboOptimizer` had `denoise_cuda_graph` (full-loop capture with 3-stream warmup) that was shown to duplicate work torch.compile already does internally (current_session line 4504, "I get CUDA-graph savings but lose kernel fusion"). Adaptive heuristic (velocity-norm delta < 0.01) was validated only on a synthetic 16-hidden toy model, and fails on smolvla/pi0.5/gr00t on real VLAs (commit `091074c`).

**Deprecation:** Commit `a90a4ab` (step 2 of CLI unification, 2026-04-16). Net -98 LOC. Experiment-level commands that made the CLI surface too wide for v0.2.

### `reflex split` → `serve --cloud-fallback <url>`
**Original purpose:** Configure cloud/edge orchestration with health checking, latency history, and fallback modes (last_action/zero). Never gained users. Policy (ADR `2026-04-14-wrap-not-rebuild-vla-eval.md`): *no cloud-edge split until (a) ≥2 design partners explicitly request it, (b) reference VLA that doesn't fit edge hardware, (c) their need is concrete. Only the `SplitConfig` latency monitor ships now.*

**Deprecation:** Same commit `a90a4ab`.

### `reflex adapt` → [removed; no forwarding replacement]
**Original purpose:** URDF → framework configs (LeRobot/OpenPI/GR00T) + action-space mapping (pad/truncate/project). Cross-embodiment retraining is still DIY config surgery — OpenPI alone has 6+ issues with zero responses (issues 872, 740, 714, 580, 449, 591, per prior_sessions.md). Deferred wedge, post-LIBERO-success.

**Deprecation:** Same commit `a90a4ab`. Council reprioritization (current_session line 6497): "adapt has no users, delete from v0.2 scope".

### `reflex check` → `validate --quick`
**Original purpose:** 5 pre-deploy checks (loadable, size, structure, dtype, nan_inf) on an export directory. Users guessed wrong between this and `validate` (ONNX-vs-PyTorch parity) and `guard` (URDF safety). Council finding (current_session line 8117): "Three commands pretend to 'validate' something".

**Merge:** Commit `369f70c` (step 3, 2026-04-16). `validate --quick` for export dirs; `validate --pre-export` for raw checkpoints.

---

## Architectural constraints shared across wedges

From current_session line 8097 ("single black-box command" principle):
- **Auto-detect model type** (already works; `reflex.checkpoint.detect_model_type` dispatches to per-family exporter)
- **Run ALL required sub-exporters in one pass** (expert + VLM for SmolVLA; expert for pi0/pi0.5; full stack with action_encoder/decoder for GR00T)
- **`reflex_config.json` is authoritative** — records `model_id`, `vlm_model_id`, `action_dim`, `chunk_size`, `image_size`, VLM files list, which sub-graphs exist, benchmarks this export is valid for
- **Output dir is a contract** — `reflex serve <dir>`, `reflex validate <dir>`, `reflex bench <dir>` all work without the user knowing about the internal ONNX split
- **No user should ever call `export_vlm_prefix()` directly** — it's internal

From commit `f882dcb`: `vlm_model_id` (BASE VLM, e.g. `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) is used for tokenizer/processor, NOT `model_id` (which is the SmolVLA policy checkpoint and doesn't have a tokenizer). Config writes both at export time.

---

## Install / dependency gotchas shared across wedges

**Extras (pyproject.toml):**
- `[onnx]` — `onnxruntime>=1.17.0` (CPU only)
- `[gpu]` — `onnxruntime-gpu>=1.20,<1.24` + `nvidia-cudnn-cu12>=9.0,<10.0` + `nvidia-cublas-cu12>=12.0,<13.0`. **Critical: this pin is what closes the Apr-14 silent-CPU-fallback footgun.**
- `[serve]` — `fastapi>=0.100.0` + `uvicorn>=0.23.0` + `Pillow>=10.0.0`. Use `[serve,gpu]` together for production.
- `[safety]` — `yourdfpy`
- `[eval]` — `vla-eval` + `mujoco>=3.0` + `robosuite==1.4.1` (pinned; 1.5+ changed module paths) + `gymnasium` + `h5py`
- `[dev]` — pytest 8+, ruff 0.4+, mypy 1.10+, httpx 0.24+

**Core deps always pulled:** `huggingface_hub>=0.20.0`, `transformers>=4.40,<5.0`, `onnx>=1.15.0`, `onnxscript>=0.1.0`, plus `torch`, `safetensors`, `typer`, `rich`, `pydantic`, `numpy`, `pyyaml`. Caught by install-path test (`modal_verify_install_path.py`) — earlier omitting `huggingface_hub`/`transformers` from core broke `reflex export` even with `[serve,gpu]`.

**Five recurring gotchas:**
1. **cuDNN 9 system library** — GPU install needs the FULL cuDNN 9 system library (incl. `libcudnn_adv.so.9`), not just the pip wheel. Easiest: `nvcr.io/nvidia/tensorrt:24.10-py3` container. `reflex serve` errors loudly (no silent CPU fallback).
2. **`CUDAExecutionProvider not available`** — ORT 1.20+ needs CUDA 12.x + cuDNN 9.x. Pip `nvidia-cudnn-cu12` is missing `libcudnn_adv.so.9`.
3. **First `reflex serve` takes 30-90s** (TRT engine build); cached in `<export_dir>/.trt_cache`, restart is 1-2s.
4. **"Model not loaded" 500** — lifespan handler hasn't finished; wait for `/health` → `model_loaded: true`.
5. **"Action values look random / nonsensical"** — Expected in v0.1. The ONNX export covers the action-expert with random VLM conditioning. Real per-image conditioning lands when the VLM prefix pipeline is fully wired (Phase II.4 / v0.2).

---

## References

**ADRs:**
- `01_decisions/2026-04-14-ship-distill-first.md`
- `01_decisions/2026-04-14-deprioritize-adapt-and-split.md`
- `01_decisions/2026-04-14-wrap-not-rebuild-vla-eval.md`
- `01_decisions/2026-04-14-disable-trt-when-batch-gt-1.md`
- `01_decisions/2026-04-14-strict-provider-no-silent-cpu-fallback.md`
- `01_decisions/2026-04-16-council-reprioritization.md`

**Commits of record:**
- v0.1.0 baseline: `e8fe39f`
- 7-wedge assembly: `c6618e5`..`89aaae4`
- Apr-14 GPU post-mortem: `f2cd906`..`5b21296`
- Wedge composition through flags: `9df6daa`
- TRT FP16 breakthrough: `fce8a6f` (3.2× over torch.compile)
- reflex validate round-trip: `e1455f7`..`18f8038`
- VLM prefix pipeline v1 stub: `f72b8b9`..`d641134`
- VLM prefix 4-file split + GQA spike: `4daf6ea`..`9fb6ddb`
- CLI unification (deprecations + unify + distill scaffold): `fdd9bb3`..`ed8157c`
