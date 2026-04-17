# Hardware Targets — Jetson / Edge Matrix

Reflex's hardware profiles are registered at build time and exposed via `reflex targets`. The live list: `orin-nano`, `orin`, `orin-64`, `thor`, `desktop`.

Dev proxy: **Modal A10G** (Ampere SM_86) — the closest cloud GPU to Jetson Orin (Ampere Tegra SM_87). Same compute family, different memory. A10G ≈ $1.10/hr on Modal.

---

## The target matrix

| Profile | Hardware | Memory | SM arch | Precision | Status |
|---|---|---|---|---|---|
| `orin-nano` | Jetson Orin Nano Super Dev Kit | 8 GB LPDDR5 | SM 8.7 (Ampere Tegra) | FP16, no FP8 | Primary starter SKU ($249) |
| `orin` | Jetson Orin AGX | 32 GB | SM 8.7 | FP16 | Supported |
| `orin-64` | Jetson Orin AGX 64GB | 64 GB | SM 8.7 | FP16 | Supported |
| `thor` | Jetson Thor | 128 GB | SM 10.0 | FP16, FP8 | Supported, primary Blackwell target |
| `desktop` | 4090 / A100 / H100 | 24+ GB | A100 / A10G / Ampere / Blackwell | FP16/FP8 | Dev / cloud |

Modal dev proxy: `A10G` — Ampere, 24GB, SM 8.6. Close enough to Jetson Orin Nano (SM 8.7) that benchmarks there are within ~10% of Jetson numbers at the same precision.

---

## Per-target SKU pricing + availability

From `current_session.md` lines 5774, 5783:

- **Jetson Orin Nano Super (8GB):** $249 — NVIDIA dropped the Super 8GB price. Earlier Reflex docs said $499; that was pre-drop.
- **Jetson Orin NX (16GB):** the "safe" recommendation per sizing analysis below — all 4 VLAs fit comfortably with FP16 + 2× overhead.
- **Jetson Orin AGX (32/64GB):** ~$2000–$2500.
- **Jetson Thor (128GB):** ~$1500 starter.
- **CloudJetson.com:** "Rent NVIDIA Jetson Hardware by the Hour" — ~$5 for a 30-min Orin Nano session. Use this to validate reflex on real Jetson before asking users to buy one.
- **Used / refurb:** eBay active listings around $200–240 for Orin Nano Super 8GB.
- **NVIDIA Inception:** does **NOT** give free Jetson kits. It's a 15–30% discount + $25k AWS credits + DLI training + VC intros.

---

## Fit table — all 4 VLAs × 4 Jetson SKUs

From `modal_bench_onnx_vs_torch.py` / `modal_bench_path_b.py`. Estimates `fp16_gb × 2` (overhead) vs the 8/32/64/128 GB SKUs.

| Model | Params | FP16 weights | FP16 + 2× overhead | Orin Nano 8GB | Orin 32GB | Orin 64GB | Thor 128GB |
|---|---|---|---|---|---|---|---|
| SmolVLA | 99.8M | ~200MB | ~0.4GB | fits | fits | fits | fits |
| pi0 | 314.6M | ~630MB | ~1.3GB | fits | fits | fits | fits |
| pi0.5 | 426.9M | ~855MB | ~1.7GB | fits | fits | fits | fits |
| GR00T | 1091.7M | ~2.2GB | ~4.4GB | fits (tight) | fits | fits | fits |

**"All 4 current VLAs fit on $500 Orin Nano 8GB in FP16 with 2× overhead. GR00T at 4.4GB is tight but fits."** (commit `f2cd906 2026-04-14`, L4 of strategic lessons from Path B post-mortem.)

**L4 strategic takeaway:** Orin Nano 8GB is the right starter SKU, not Thor ($1500) or Orin 64 ($2500).

---

## Per-target Hz expectations (A10G proxy; Jetson numbers extrapolated)

From `modal_bench_trt_fp16.py` (commit `fce8a6f 2026-04-14`, A10G):

### Single denoise step (ms), TRT FP16

| Model | Params | `torch.compile` | ORT-GPU FP32 | **TRT FP16** | Speedup (TRT vs compile) |
|---|---|---|---|---|---|
| SmolVLA | 99.8M | 3.06 | 3.26 | **0.95** | **3.2×** |
| pi0 | 314.6M | 6.23 | 5.53 | **1.94** | **3.2×** |
| pi0.5 | 426.9M | 7.34 | 7.37 | **2.24** | **3.3×** |
| GR00T | 1091.7M | 14.61 | 14.45 | **5.59** | **2.6×** |

### Per-chunk (10-step denoise) wall-clock + effective Hz

From `reflex bench --all` (commit `9e3dabb 2026-04-14`, A10G auto-TRT FP16):

| Model | Mean ms | p95 ms | Effective Hz | Mode |
|---|---|---|---|---|
| SmolVLA | 11.67 | 11.85 | **86 Hz** | onnx_trt_fp16 |
| pi0 | 23.57 | 24.22 | **42 Hz** | onnx_trt_fp16 |
| pi0.5 | 27.07 | 27.76 | **37 Hz** | onnx_trt_fp16 |
| GR00T | 56.55 | 57.25 | **18 Hz** | onnx_trt_fp16 (borderline; needs optimization) |

**All four meet or exceed the 20–30 Hz target for real-time robot control** — except GR00T which sits at 18 Hz on A10G and likely stays there on Orin Nano (same compute family).

---

## The `reflex_bench` latency baseline table (canonical)

Published in `reflex_context/03_experiments/2026-04-14-trt-fp16-vs-torch-compile.md` and quoted across launch drafts:

```
Model       torch.compile   reflex TRT FP16    speedup
SmolVLA     25.8 ms         9.5 ms (per chunk) 2.6-3.3×
pi0         47.5 ms         19.4 ms            2.6-3.3×
pi0.5       52.9 ms         22.4 ms            2.6-3.3×
GR00T       113.2 ms        55.9 ms            2.6-3.3×
```

(The "25.8ms SmolVLA" torch.compile baseline is from per-step × 10 steps with CUDA-graph internal; the "9.5ms per chunk" TRT FP16 is the total 10-step denoise via `trtexec --loadEngine --useCudaGraph`.)

**Marketing headline (post review):** *"pi-Flow distillation from 10 to 2 steps, <5% accuracy drop on LIBERO (per arXiv 2510.14974)"* — replacing the prior "5× faster" claim flagged as weak. The TRT speedup story is also maintained but framed as "TRT FP16 beats `torch.compile` on cloud GPU by 2.6–3.3× *and* runs on Jetson."

---

## The FP16-vs-FP32 baseline honesty passage

From Show HN draft (`modal_apps_and_pm_docs.md`):

> "I went into this thinking the moat was edge-only because `torch.compile` was crushing my early benchmarks. Turned out my `onnxruntime-gpu` was silently falling back to CPU due to a CUDA 12-vs-13 library mismatch. Once that was fixed, TRT FP16 wins by 2.6-3.3× across the board."

Honest comparison is FP16-vs-FP16: `torch.compile(mode='reduce-overhead')` in FP32 vs `trtexec --fp16`. Per reviewer: FP16-vs-FP32 is a gotcha sophisticated buyers catch. The current numbers are FP16-vs-FP32; an honest FP16-vs-FP16 re-run is a post-mortem item (#3 of RPI post-mortem, current_session.md line 335).

---

## Jetson validation status

From `reflex_validate` post-mortem:

> **Every claim is A10G-extrapolated.** No run on actual Jetson hardware yet.

GOALS.yaml weight 9: **"Real Jetson Orin Nano validation via `reflex validate`"** — the second-highest priority behind VLM prefix encoder.

Path to validation:
1. Buy Jetson Orin Nano Super 8GB ($249) OR rent via CloudJetson.com (~$5 for 30 min).
2. Install JetPack 6.1 (torch 2.5, cuDNN 9).
3. Run `reflex export lerobot/smolvla_base --target orin-nano`.
4. Run `reflex bench` against the exported engine.
5. Compare to A10G numbers. If Jetson within 10–30% of A10G (memory bandwidth differs), the extrapolation holds.

---

## Hardware compatibility — SM version and FP8

- **SM 8.7 (Orin Nano / Orin AGX / Orin AGX 64GB):** Ampere Tegra. **No FP8 support** — only FP16 or FP32. Uses INT8 for fastest path in some cases.
- **SM 8.6 (A10G, A100-SXM4):** Ampere. No FP8. FP16 / TF32.
- **SM 9.0 (H100):** Hopper. **First FP8 support** (E4M3, E5M2).
- **SM 10.0 (Thor / Blackwell):** Grace Blackwell. **FP8 + NVFP4** — the microscaling format that makes Thor the target for ultra-low-power inference.

Reflex precision per target:
- `orin-nano` / `orin` / `orin-64`: FP16 default. INT8 via TRT calibration (reserved for Pro tier).
- `thor`: FP16 default. **FP8 path** — Pro tier feature, planned for v0.2+. NVFP4 is the "stretch goal" for Blackwell.
- `desktop`: FP16. Research mode allows FP32 for parity testing.

---

## TRT × hardware sharp edges

From sessions_md Theme "TRT × batching sharp edge":

- **Static-shape ONNX** forces TRT to rebuild engine when batch shape changes → **34s / call, 200× pessimization** (commit `e76678c 2026-04-14`).
- **Fix v1:** drop TRT EP when `--max-batch > 1`. CUDA EP handles dynamic batch shapes natively. CUDA EP at batch 16 is 2.88× throughput over batch 1 — good enough.
- **Proper long-term fix:** dynamic batch shape export + TRT shape profiles (batch=1/4/8/16). Deferred to v0.2 (export v2).

### Cold-start times (TRT engine build)

- **First `reflex serve`:** 30–90s (engine build). Subsequent starts hit `<export_dir>/.trt_cache` — 1–2s.
- **GR00T cold-start:** can exceed 90s on cold cache. Install verify `/health` poll bumped 90s → 240s (commit `9a690ab 2026-04-14`).
- **Warmup:** in FastAPI lifespan startup, `server.predict()` runs once before yielding. `/health` won't return `model_loaded=true` until warmup completes. Readiness probes wait correctly.

---

## Modal image basline for hardware benches

- **`nvcr.io/nvidia/tensorrt:24.10-py3`** — TRT 10.5, CUDA 12.6, cuDNN 9, trtexec included.
- **Torch pin:** `torch==2.5.1` (CUDA 12.4 — matches ORT 1.20's cuDNN 9 + CUDA 12.x requirements). Torch 2.11 installs cu13, which breaks onnxruntime-gpu 1.20+.
- **ORT pin:** `onnxruntime-gpu==1.20.1`.
- **cuDNN:** use NVIDIA container's bundled libs, NOT the pip wheel. `pip install nvidia-cudnn-cu12` is missing `libcudnn_adv.so.9` that ORT 1.20+ requires.

Key install gotcha (README `[gpu]` extra comment): *"Apr-14 post-mortem: omitting these was the cause of silent CPU fallback in v0.1 benchmarks."*

---

## Strict providers mode — no silent CPU fallback

Commit `5b21296 2026-04-14`: `reflex serve` refuses to silently fall back to CPU when `onnxruntime-gpu` isn't available.

**Contract (codified in `tests/test_server.py::TestStrictProviderMode`):**
- Strict + CUDA requested + CPU only → RuntimeError.
- `reflex serve` gets `--providers "A,B,C"` and `--no-strict-providers` flags.
- Pre-flight: if `--device cuda` but ORT has no CUDAExecutionProvider, print multi-line hint + exit 1.
- Install-hint distinguishes the "you installed onnxruntime not onnxruntime-gpu" footgun from the "CUDA 12 libs missing from path" footgun.

This guards users from the exact trap that produced the original CPU-fallback silent-bench scandal.

---

## Batching throughput table (pi0, A10G)

From `scripts/modal_verify_batching_real.py` / commit `48d76fe 2026-04-14`:

| `--max-batch` | Concurrent | QPS | Per-request latency | Speedup |
|---|---|---|---|---|
| 1 | 32 | 17.1 | ~2.0s (serial) | 1.0× |
| 4 | 32 | ~30 | ~1.1s | ~1.8× |
| 8 | 32 | ~38 | ~0.85s | ~2.2× |
| 16 | 32 | 49.3 | ~0.65s | **2.88×** |

At batch=16, per-request latency **drops 60–80%** — no more serial queueing.

**Caveat:** TRT EP skipped when `--max-batch > 1` (see sharp edge above). Serve falls back to CUDA EP.

---

## Reflex Compute Pack (Phase 3 hardware roadmap)

From `vla_to_hardware_roadmap/README.md`:

- Year 2+: "Reflex Compute Pack" — own-branded hardware.
- Co-design inference ASIC with Taiwan design houses (Phase 3).
- Phase 4: datacenter hardware (ODM AI server → silicon).
- Not scope for reflex-vla repo. Scoped to `axion_compute/vla_to_hardware_roadmap/`.

Reflex-VLA's hardware job: make the 7 profiles above work. Compute Pack is downstream.

---

## Adaptive-denoise per-model verdict (adaptive = `reflex turbo` path)

From `scripts/modal_verify_adaptive_real.py` / commit `091074c 2026-04-14`:

| Model | Triggers / 25 trials | Mean stop step | Savings | Action diff at trigger |
|---|---|---|---|---|
| SmolVLA | 0/25 | — | 0% | — |
| pi0 | 25/25 | 4.2 | **58.4%** | 0.073 (small, OK) |
| pi0.5 | 3/25 | 9.4 | 5.6% | 0.762 (large) |
| GR00T | 25/25 | 3.0 | 70% | 0.674 (large — meaningful drift) |

**Finding:** "pi0 is the only model where adaptive is a real win." Per-model threshold tuning deferred to v0.2. Server now warns when `--adaptive-steps` used with anything other than `model_type="pi0"`.

Implication for hardware Hz: on pi0, adaptive gets you 58% fewer denoise steps → roughly 58% latency win. So effective pi0 Hz with adaptive-TRT-FP16 could hit ~100 Hz on A10G. On SmolVLA / pi0.5 / GR00T, the non-adaptive numbers above are the real budgets.

---

## Deadline modes in serve

From commit `9df6daa 2026-04-14`:

`reflex serve --deadline-ms N` enables deadline checking. When a predict exceeds N ms, fall back to last known good action (or zeros on first call). Useful for real-time robot control where stale actions are worse than zero actions.

Planned for: hard-real-time scheduler (research flagship paper, `Action-Chunk Scheduling: A Serving Contract for Robot Foundation Models`, MLSys 2027 target).

---

## Related files

- `reflex_context/03_experiments/2026-04-14-trt-fp16-vs-torch-compile.md` — the experiment writeup.
- `scripts/modal_bench_trt_fp16.py` — the benchmark.
- `scripts/modal_bench_path_b.py` — the general GPU bench.
- `scripts/modal_verify_adaptive_real.py` — the adaptive per-model verdict.
- `scripts/modal_verify_batching_real.py` — the batching table.
- `src/reflex/config.py` — the hardware profile registry.
- `src/reflex/runtime/server.py` — TRT EP logic, strict providers, warmup, deadline handling.

---

## Gaps / unresolved on hardware

1. **No actual Jetson Orin Nano run.** All Jetson claims are A10G-extrapolated. Critical for credibility — Goals.yaml weight 9.
2. **No FP8 path.** Thor's FP8 is not yet wired; reserved for v0.2 pro tier.
3. **No INT8 TRT calibration.** Would compress GR00T from 4.4GB to ~2GB (comfortable on Orin Nano).
4. **No dynamic batch shape export.** Forces TRT EP drop at `--max-batch > 1`.
5. **No RTX 40/50 benchmarks.** A10G is Ampere; desktop tier should also cover Ada (4090) and Blackwell (5090) but no published numbers yet.
6. **GR00T at 18 Hz on A10G is borderline real-time** — needs either adaptive (which triggers too aggressively with 0.67 action diff) OR quantization OR a proper per-embodiment forward that avoids the 1024/1536 mismatch.
