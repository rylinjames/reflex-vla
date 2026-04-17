# 2026-04-13 — Phase III Continuous Batching + Phase IV Adaptive Denoising Validation

**Session theme:** The day the "it works on a toy" wedges got retested against real VLAs. Two promises from Apr-10 — continuous batching and adaptive denoising — had been validated on synthetic models (16-hidden toy, Identity ONNX). Today they got pointed at real SmolVLA, pi0, pi0.5, and GR00T, and one of them collapsed under the scrutiny.

---

## Goal

1. **Phase III — Continuous batching.** Make `reflex serve --max-batch N` actually amortize inference cost across concurrent requests. Ship a queue-only fake-Identity-ONNX verification (Phase III.1), then re-run against a real pi0 model on A10G (Phase III.2).
2. **Phase IV — Adaptive denoising validation.** The 0.01 velocity-norm-delta threshold in `src/reflex/kernels/turbo.py` was only validated against a synthetic 16-hidden toy model (from `modal_sim_test.py`). Confirm or deny it works on real SmolVLA / pi0 / pi0.5 / GR00T.

The sub-text: *"are our 'works on synthetic' benchmarks lies?"* The Apr-14 silent-CPU-fallback post-mortem (see `2026-04-14_benchmark_postmortem.md`) made everyone paranoid about benchmarks that happen to confirm the thesis. Apr-13 was a pre-emptive re-check of two features before they calcified in marketing.

---

## The real-model vs synthetic-toy distinction

The recurring pattern in the session: **a feature passes when tested against an Identity ONNX graph or a 16-hidden toy**, then fails or degrades on a real VLA because the real model's behaviour depends on parameter magnitudes the toy never captured.

Identity-ONNX tests measure only the plumbing — queueing, HTTP dispatch, memory layout. A real VLA has:
- Flow-matching velocity norms that decay in a model-specific pattern.
- Cross-attention against a real VLM prefix (not zeros).
- Actual numerical precision budgets for FP16 engines.
- Per-request image + instruction + state tokenization cost.

Every Apr-13 commit added a companion "real-model" script to whatever "synthetic" script shipped on Apr-14. Table:

| Feature | Synthetic test | Real-model test |
|---------|----------------|-----------------|
| Continuous batching | `scripts/modal_verify_batching.py` (Identity ONNX, `cpu=4`, dynamic batch axis) | `scripts/modal_verify_batching_real.py` (real pi0, A10G, 32 concurrent, batch=1/4/8/16) |
| Adaptive denoising | `scripts/modal_sim_test.py` (16-hidden toy, simulated cross_attn=zeros) | `scripts/modal_verify_adaptive_real.py` (real SmolVLA/pi0/pi0.5/GR00T, A10G, 25 trials) |

---

## Phase III — Continuous batching lands

### Phase III.1: synthetic (`899c02e`, Apr-14 15:21)

The initial commit added:
- `start_batch_worker` / `stop_batch_worker` async, called from FastAPI lifespan handler.
- Single worker coroutine owning an `asyncio.Queue` dispatching batches.
- `predict_async()` — new front-door for HTTP path; falls through to `predict()` when batching off.
- `_batch_worker_loop()` drains: blocks for first request, drains up to `max_batch` items within `batch_timeout_ms`.
- `_predict_batch_sync()` runs **one** ONNX inference with batch dim = N, splits output back to N response dicts.
- Per-item guard wedge applies AFTER batched inference (each request clamped individually).
- Telemetry: `batches_run_total`, `batched_requests_total`, `batch_size`, `request_index`, `amortized_latency_ms`.

The synthetic test `scripts/modal_verify_batching.py` used a **fake Identity ONNX with dynamic batch axis**: `TensorProto.FLOAT, ["batch", 50, action_dim]`. Three configs (batch=1, 4, 8), 16 concurrent POST /act requests per config, measured `throughput_qps = n_concurrent / elapsed`. Ran on `cpu=4` with `--no-strict-providers` (fake ONNX, CPU-only).

The inline comment in the script is honest: *"the fake-Identity-op test only measured queue overhead."* That's why `modal_verify_batching_real.py` exists.

### Phase III.2: real-model batching (`5b58d35` Apr-14 15:34, plus 4 follow-up fixups)

Switched to a real exported pi0 (~50 ms/chunk on GPU) so batching should give 2–3× throughput. The image post-mortem tells its own story:

Commits:
- **`5b58d35`** — Add `modal_verify_batching_real.py` (+182 LOC), launch drafts.
- **`b4d3552`** — 180 s timeout + stdout capture on failure.
- **`492a351`** — Switch to `nvcr.io/tensorrt:24.10-py3` image (cuDNN bundled). *"Use NVIDIA's TRT container so cuDNN 9 (including libcudnn_adv) is already on the system path. The pip-installed nvidia-cudnn-cu12 wheel is missing libcudnn_adv.so.9 which ORT 1.20+ requires."*
- **`48d76fe`** — README: real-model batching results (2.88× throughput at batch=16).
- **`526dded`** — Launch drafts include batching numbers.

Critical embedded comment in `scripts/modal_verify_batching_real.py`:
> *"Use a real file for stdout — subprocess.PIPE deadlocks if the process logs more than one OS pipe buffer (~64KB) before we read."*

This is the same pipe-buffer lesson from `modal_e2e_demo.py`, re-learned. Canonical pattern: `log_fh = open(path, "wb"); Popen(..., stdout=log_fh, stderr=subprocess.STDOUT)`.

### The 2.88× result

32 concurrent pi0 requests, A10G, per-request latency also improves (no more serial queueing):

| max-batch | qps | notes |
|-----------|-----|-------|
| 1 | 17.1 | baseline |
| 4 | ~30 | near-linear |
| 8 | ~40 | |
| 16 | **49.3** | **2.88× vs batch=1** |

Per-request amortized latency drops 60–80% at batch=16 vs serial execution. Headline: *"fleet batching makes a single GPU serve multiple robots without latency blowback."*

### Known sharp edges documented

v0.1 batching limitations captured in commit notes:

- **Per-item conditioning** (image, instruction, state) ignored in v0.1 batching — all 16 concurrent pi0 requests use the same dummy input. Real per-item conditioning lands with VLM prefix wiring (Phase II.4 / v0.2).
- **Adaptive denoising in batch mode** needs per-item convergence tracking — deferred.
- **TRT engines static-shape (batch=1)** — real batching with TRT needs dynamic batch shape via `trtexec --minShapes/--optShapes/--maxShapes` (Phase III.2+). See `modal_verify_trt_with_batch.py`.

### TRT × batching sharp edge resolved (`e76678c`, Apr-14 18:25)

The diagnostic `scripts/modal_verify_trt_with_batch.py` surfaced the actual interaction: **when ORT TensorRT EP encounters a batch shape it doesn't have an engine for, it rebuilds the engine on EACH CALL.** Our static-shape ONNX (batch=1) hits this path whenever `--max-batch > 1`.

Measured numbers:
- `--max-batch 1` → 37.7 qps normal.
- `--max-batch 4` → 0.2 qps / 34,121 ms per call (rebuild every time).
- `--max-batch 8` → 0.2 qps / 35,129 ms per call.

**~200× pessimization.** Fix: drop TRT EP when `max_batch > 1`; ORT falls through to `CUDAExecutionProvider` which handles dynamic batch shapes natively and gives the 2.88× at batch=16.

ADR: `2026-04-14-disable-trt-when-batch-gt-1.md`. The proper long-term fix (dynamic batch shape export + TRT shape profiles for batch=1/4/8/16) is deferred to v0.2.

---

## Phase IV — Adaptive denoising: only pi0 survives

The adaptive-stop heuristic in `src/reflex/kernels/turbo.py`:
- Track per-step velocity norm during 10-step Euler denoise.
- When `abs(norm[i] - norm[i-1]) < threshold=0.01` (after step ≥ 2), stop early.
- Promised: on pi0 synthetic, showed ~4x speedup on the 16-hidden toy.

### The real-model check (`1c40f14` + `091074c`, Apr-14 16:33–16:37)

`scripts/modal_verify_adaptive_real.py` — A10G, 25 trials per model, `NUM_STEPS=10`, `THRESHOLD=0.01`. Records: trigger rate, mean trigger step, % savings, action-diff at trigger-step vs full-10-step output.

Verdict table (A10G, 25 trials):

| Model | Triggered | Mean step | % savings | Action diff | Verdict |
|-------|-----------|-----------|-----------|-------------|---------|
| smolvla | 0/25 | — | 0% | — | **Never triggers.** Velocities never converge within 10 steps. |
| pi0 | 25/25 | 4.2 | **58.4%** | 0.073 (small, OK) | **Only real win.** |
| pi0.5 | 3/25 | 9.4 | 5.6% | 0.762 (**large**, degraded) | Rarely triggers; when it does, output is wrong. |
| gr00t | 25/25 | 3.0 | 70% | 0.674 (**large**, degraded) | Triggers too aggressively; output wrong. |

### Honest per-model verdict

Commit `091074c` added warnings to `src/reflex/runtime/server.py` so `--adaptive-steps` **warns** when used with anything other than `model_type == "pi0"`. Per-model threshold tuning deferred to v0.2 (goal `adaptive-denoise-fix`, weight 5, GOALS.yaml).

Docs rule set: *"only quote the pi0 number, not a universal one."*

The full `--adaptive-steps` flag remains functional, but the launch drafts only cite the pi0 result. The pre-Apr-13 marketing headline *"adaptive denoising halves inference latency"* was retired — it was measuring the synthetic toy's behaviour, not the real model.

---

## Post-mortem lessons (embedded in code comments)

Both Phase III and Phase IV validations embedded future-warning comments for the next engineer:

`scripts/modal_verify_adaptive_real.py` docstring:
> *"The heuristic was validated on a synthetic 16-hidden toy model (= `modal_sim_test.py`), and this script exists to test whether it survives on real VLAs. If velocities never converge OR action diff is large, adaptive needs to be removed or rewritten."*

`scripts/modal_verify_batching.py` inline comment:
> *"the fake-Identity-op test only measured queue overhead."*

The pattern: **every synthetic test that ships to verify a feature, gets a sibling real-model script that measures the feature's honest performance.** The scripts diverge in hardware (CPU vs A10G), image (debian_slim vs `nvcr.io/tensorrt:24.10-py3`), and concurrency (16 vs 32 concurrent), but run the same feature flag.

---

## Carry-over

Unfinished work from this session:

- **Adaptive denoising per-model threshold tuning** — deferred to v0.2. GOALS.yaml goal `adaptive-denoise-fix` (weight 5) codifies: *"Adaptive denoising works on pi0 (supported), is gated behind --experimental for smolvla/pi0.5/gr00t (unsafe)."* Task #14 on Apr-16 added the deprecation warning wrapper.
- **TRT × batching proper fix** — dynamic batch shape ONNX export + TRT shape profiles. Deferred to v0.2 (`export_v2.md`). Current v0.1 skips TRT EP when `max_batch > 1`.
- **Per-item conditioning in batch mode** — waits for VLM prefix encoder to land. See `2026-04-17_libero_correctness_hunt.md` for the VLM prefix saga.
- **The turbo wedge** was consolidated away on Apr-16 (step 2 of CLI unification, `a90a4ab`). Adaptive denoising survives as `reflex serve --adaptive-steps` with model-specific warnings.
- **Fleet-mode batching** (planned larger rollout) — dropped after the Apr-13 result was in; revisit once VLM prefix lands so per-item conditioning works.
- **Launch drafts batching numbers** — in drafts but drafts still unpublished.

The biggest downstream consequence of this session: the adaptive-denoise validation killed one of the three speed-story pillars (adaptive + TRT + batching). The remaining two (TRT FP16 + batching) carried the launch pitch into the Apr-14 benchmark post-mortem — where they *also* got rescued only after a CPU-fallback scare. See `2026-04-14_benchmark_postmortem.md`.
