# Latency Benchmarks

Per-model, per-execution-path latency numbers for the 4 supported VLAs (SmolVLA, pi0, pi0.5, GR00T N1.6). All numbers gathered on Modal A10G unless otherwise noted. See caveats at the bottom.

## Setup

- **Hardware proxy**: Modal A10G (Ampere SM_86). Jetson Orin Nano is Ampere Tegra SM_87 — same compute family, different memory budget. Used as Jetson proxy per `scripts/modal_bench_trt_fp16.py` docstring.
- **Modal image**: `nvcr.io/nvidia/tensorrt:24.10-py3` (TRT 10.5, CUDA 12.6, cuDNN 9, trtexec bundled). This image is the canonical one after the Apr-14 post-mortem — the pip `nvidia-cudnn-cu12` wheel is missing `libcudnn_adv.so.9` that ORT 1.20+ requires.
- **Dependency pins**: `torch==2.5.1` + `onnxruntime-gpu>=1.20,<1.24` + `transformers>=4.40,<5.0` + `numpy<2.0`. Any drift from this triple reintroduces the silent-CPU-fallback bug documented in `02_bugs_fixed/modal_gotchas.md`.
- **Precision**: FP16 for TensorRT engines; FP32 for torch.compile and ORT-GPU.
- **Denoise loop**: 10-step Euler integration. Chunk size = 50. Single-step latency × 10 ≈ per-chunk wall-clock (Python-driver overhead subsumed).
- **Bench methodology**: 10 warmup + 50 trials per `scripts/modal_bench_trt_fp16.py`; `trtexec --loadEngine` measures TRT via its own internal loop (200 warmup / 500 iterations / 100 avgRuns, `--useCudaGraph`). Python TRT bindings deliberately avoided — `nvcr.io` container doesn't pre-install them.

## Params

- `chunk_size=50` (flow-matching action chunk length).
- `num_steps=10` (Euler denoise steps).
- `dt=-1.0/10` (schedule direction t: 1→0; `_pytorch_backend` and `_onnx_backend` must share this or validation parity silently fails — learned the hard way in Wave 2 of the reflex-validate epic).
- Static-shape ONNX graphs (batch=1). Dynamic batch would require `trtexec --minShapes/--optShapes/--maxShapes`, which errors on static-shape inputs. Documented at `02_bugs_fixed/smolvla_pipeline_bugs.md`.

## Results — single denoising step (ms)

| Model   | Params    | torch eager | torch.compile (reduce-overhead) | ORT-GPU FP32 | TRT FP16 | Speedup (TRT vs compile) |
|---------|-----------|-------------|----------------------------------|--------------|----------|---------------------------|
| SmolVLA | 99.8M     | 19.1        | **3.06**                         | 3.26         | **0.95** | **3.2×**                  |
| pi0     | 314.6M    | 23.9        | **6.23**                         | 5.53         | **1.94** | **3.2×**                  |
| pi0.5   | 426.9M    | ~55.6*      | **7.34**                         | 7.37         | **2.24** | **3.3×**                  |
| GR00T   | 1091.7M   | —           | **14.61**                        | 14.45        | **5.59** | **2.6×**                  |

*eager-ms for pi0.5 extrapolated from per-step bench (torch.compile already uses CUDA graphs internally).

## Results — per-chunk wall-clock from `reflex bench`

| Model   | mean_ms | p95_ms  | mode           | export_s | Hz      |
|---------|---------|---------|----------------|----------|---------|
| SmolVLA | 11.67   | 11.85   | onnx_trt_fp16  | 72.8     | **86**  |
| pi0     | 23.57   | 24.22   | onnx_trt_fp16  | 112.2    | **42**  |
| pi0.5   | 27.07   | 27.76   | onnx_trt_fp16  | 151.9    | **37**  |
| GR00T   | 56.55   | 57.25   | onnx_trt_fp16  | 181.9    | **18**  |

Hz = `1000 / mean_ms`. All four meet or exceed the 20-30 Hz target for real-time robot control except GR00T on A10G (18 Hz is borderline; needs optimization for control rates above 20 Hz).

Source: `scripts/modal_verify_bench_all.py` output (git SHA `9e3dabb`, 2026-04-14 18:00).

## Memory (FP16 weights + peak forward, for Jetson fit)

From `scripts/modal_bench_onnx_vs_torch.py` "Jetson fit" table logic (`fp16_gb × 2` overhead):

| Model   | FP16 weights GB | Peak forward GB (2× overhead) | Orin Nano 8GB | Orin AGX 32GB | Thor 128GB |
|---------|------------------|-------------------------------|---------------|---------------|------------|
| SmolVLA | 0.2              | 0.4                           | fits          | fits          | fits       |
| pi0     | 0.6              | 1.3                           | fits          | fits          | fits       |
| pi0.5   | 0.9              | 1.7                           | fits          | fits          | fits       |
| GR00T   | 2.2              | 4.4                           | **tight**     | fits          | fits       |

All 4 VLAs fit on $249 Jetson Orin Nano Super 8GB in FP16 with 2× overhead. GR00T is the ceiling — if anything grows beyond GR00T N1.6, expect Orin NX 16GB ($500) to become the "safe" recommendation.

## Per-hardware target extrapolations

**Caveat up front: these are A10G benchmarks. Jetson-native numbers not yet captured** (see `missing benchmarks` section).

Rough scaling (Jetson cores run at lower clocks than A10G, memory bandwidth narrower). Order-of-magnitude estimate, not measured:

| Model   | A10G TRT FP16 (ms / Hz) | Orin Nano estimate (ms / Hz) | Orin AGX estimate (ms / Hz) | Thor estimate (ms / Hz) |
|---------|--------------------------|-------------------------------|------------------------------|--------------------------|
| SmolVLA | 11.7 / 86                | ~23 / ~43                     | ~14 / ~71                    | ~9 / ~111                |
| pi0     | 23.6 / 42                | ~47 / ~21                     | ~28 / ~36                    | ~18 / ~56                |
| pi0.5   | 27.1 / 37                | ~54 / ~19                     | ~33 / ~30                    | ~20 / ~50                |
| GR00T   | 56.6 / 18                | ~113 / ~9 (borderline)        | ~68 / ~15                    | ~42 / ~24                |

Scaling factor assumed: 2× vs A10G for Orin Nano, 1.2× for Orin AGX, 0.75× for Thor. These are rough and should be replaced with measured numbers once Jetson access is available (see `03_research/hardware_alternatives.md` for CloudJetson.com / used-refurb Jetson paths).

## Batching throughput (pi0 on A10G, from `scripts/modal_verify_batching_real.py`)

| max_batch | qps    | amortized_ms per request | mode      |
|-----------|--------|---------------------------|-----------|
| 1         | 17.1   | ~58                       | onnx_trt_fp16 |
| 4         | ~34    | ~30                       | onnx_cuda     |
| 8         | ~44    | ~23                       | onnx_cuda     |
| 16        | **49.3** | ~20                     | onnx_cuda     |

**2.88× throughput at batch=16.** Per-request latency drops because amortized kernel-launch and VLM-prefix costs shrink. TRT EP is auto-disabled when `max_batch > 1` (see `03_research/direct_torch_export_viability.md` — static-shape ONNX forces engine rebuild on every shape change = 34s/call). CUDA EP takes over.

## Adaptive denoising (per-model verdicts, from `scripts/modal_verify_adaptive_real.py`)

See `adaptive_denoising_validation.md` for full treatment. Headlines:

| Model   | Trigger rate | mean_step | savings | action_diff | Verdict |
|---------|--------------|-----------|---------|--------------|---------|
| SmolVLA | 0/25         | —         | 0%      | —            | Never triggers — adaptive never activates |
| pi0     | 25/25        | 4.2       | **58.4%** | 0.073 (small) | **Real win** — only model where adaptive is safe |
| pi0.5   | 3/25         | 9.4       | 5.6%    | 0.762 (large) | Rarely triggers and drifts when it does |
| GR00T   | 25/25        | 3.0       | 70%     | 0.674 (large) | Triggers too aggressively — meaningful drift |

**Only quote the pi0 58% latency-savings number externally.** Threshold `0.01` (velocity-norm delta) is a magic number tuned on the synthetic 16-hidden toy model in `scripts/modal_sim_test.py`; per-model thresholds deferred to v0.2.

## Caveats

1. **A10G ≠ Jetson.** These are Ampere SM_86 cloud GPU numbers standing in for Ampere Tegra SM_87 edge. Memory bandwidth, clocks, and SM count differ. Marketing copy says "A10G as Jetson proxy" — that framing survives only until Jetson-native validation lands.
2. **FP16 TRT vs FP32 torch.compile is the wrong comparison.** The marketing headline "2.6-3.3× faster than PyTorch" was flagged by reviewers as weak — FP16 vs FP32 is a known gotcha that sophisticated buyers spot. The honest comparison is TRT FP16 vs torch.compile FP16. Re-run pending (tracked as `03_research/direct_torch_export_viability.md` follow-up).
3. **SmolVLA Hz = 86** uses random VLM conditioning (v0.1). Real per-image-conditioned VLM prefix export works (vision_encoder.onnx + text_embedder.onnx + decoder_prefill.onnx all load), but action quality was 0% on LIBERO-10 as of 2026-04-17; see `02_bugs_fixed/smolvla_pipeline_bugs.md` and `05_sessions/2026-04-17_libero_correctness_hunt.md`.
4. **Static-shape ONNX** is the invariant TRT FP16 depends on for the single-shape engine build. Adding dynamic batch for true multi-batch TRT requires `--minShapes/--optShapes/--maxShapes` profiles at export — deferred to v0.2.
5. **torch.compile warm-up** takes 10-30s on first call; not amortized in "per-step" numbers. Fair bench budget assumed.
6. **GR00T 18 Hz on A10G** may fall under real-time threshold on Orin Nano. Embodiment 0 is pinned; other embodiments may differ.
7. **Modal preemption**: A100 spot instances can be reclaimed mid-bench. A10G on-demand is more stable. All cited numbers from on-demand runs.
8. **pi0.5 eager latency ~55.6ms is extrapolated**, not measured directly.

## Missing benchmarks

- **Jetson-native**: No Orin Nano / Orin AGX / Thor actual-hardware numbers yet. High-weight goal (`jetson-orin-nano-validation`, weight 9 in `GOALS.yaml`). CloudJetson.com and eBay refurb Orin Nano Super ($200-240) are the likely paths.
- **TRT FP16 vs torch.compile FP16** (apples-to-apples): current numbers are FP16 vs FP32. Published headlines need re-baseline. Tracked as `03_research/direct_torch_export_viability.md` follow-up.
- **INT4/FP8 quantization**: `reflex distill` + quantization wedge claim unsubstantiated until v0.2 lands.
- **FP8 KV throughput**: 0.97× (neutral) is reported once in the current-session notes; no dedicated bench script.
- **End-to-end with real VLM conditioning**: current Hz numbers use zero-tensor / random VLM prefix. Should be re-baselined once VLM prefix correctness is closed (pending `smolvla_pipeline_bugs.md`).
- **Multi-robot at real batch with dynamic-shape ONNX** — the export path doesn't exist yet.
- **GR00T with all embodiments**: only embodiment_id=0 is benched; per-embodiment variance unknown.

## Source scripts

- `scripts/modal_bench_trt_fp16.py` — single-step TRT FP16 vs compile vs ORT-GPU bench, 4 models.
- `scripts/modal_bench_path_b.py` — 4-path (eager / compile / ORT-GPU / Reflex CUDA graph) bench on A100.
- `scripts/modal_verify_bench_all.py` — `reflex bench` validator across all 4 VLAs (the source of the per-chunk Hz table).
- `scripts/modal_verify_batching_real.py` — pi0 batching throughput.
- `scripts/modal_verify_adaptive_real.py` — the adaptive-denoise validation (see `adaptive_denoising_validation.md`).
