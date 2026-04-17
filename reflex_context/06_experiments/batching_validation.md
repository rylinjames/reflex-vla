# Batching Validation (Phase III)

Phase III of the Apr-14 work: add continuous batching to `reflex serve` and validate on real VLAs. Outcome: **2.88× throughput at batch=16 on pi0**, with a TRT-EP × static-shape interaction that forced a fallback from TRT to CUDA EP when `max_batch > 1`.

## Setup

- **Hardware**: Modal A10G, image `nvcr.io/nvidia/tensorrt:24.10-py3`.
- **Scripts**:
  - `scripts/modal_verify_batching.py` — synthetic (fake Identity ONNX, dynamic batch axis). Queue-only measurement.
  - `scripts/modal_verify_batching_real.py` — real pi0 exported via `reflex export lerobot/pi0_base`. The source of the cited qps numbers.
- **Model**: pi0 (314.6M expert, ~50ms / chunk on GPU so batching amortizes meaningfully).
- **Concurrency**: 32 asyncio+httpx clients hammering POST `/act`.
- **Timeout**: 2400s (pi0 ONNX ~1.7GB + first ORT-GPU session creation are slow; 180s server boot timeout on top).

## Params

- `--max-batch {1, 4, 8, 16}` passed to `reflex serve`.
- `--batch-timeout-ms 20` (wait up to 20ms to fill the batch before dispatching).
- `chunk_size=50`, `num_steps=10` (flow-matching denoise).
- 5 sequential warmups before concurrent bench to keep the GPU hot.
- Batch-timeout behavior: first request blocks; subsequent requests aggregate into the same batch up to `max_batch` or `batch_timeout_ms`, whichever comes first.

## How it works (in one paragraph)

FastAPI `/act` calls `predict_async()` which enqueues the request on an `asyncio.Queue`. A single `_batch_worker_loop()` coroutine drains: it blocks on the first item, then drains up to `max_batch` items within `batch_timeout_ms`. `_predict_batch_sync()` runs ONE ONNX inference with batch dim = N and splits the output back into N response dicts. The guard wedge applies per-item AFTER batched inference (each request clamped individually). Telemetry exposes `batch_size`, `request_index`, `amortized_latency_ms`, plus `batches_run_total` + `batched_requests_total`. Started/stopped via FastAPI lifespan hooks.

## Results — pi0 on A10G, 32 concurrent requests

| max_batch | qps   | amortized_ms per request | inference mode |
|-----------|-------|---------------------------|-----------------|
| 1         | 17.1  | ~58                       | `onnx_trt_fp16` (TRT EP) |
| 4         | ~34   | ~30                       | `onnx_cuda` (CUDA EP)    |
| 8         | ~44   | ~23                       | `onnx_cuda` (CUDA EP)    |
| 16        | **49.3** | ~20                    | `onnx_cuda` (CUDA EP)    |

**2.88× throughput at batch=16 vs batch=1.** Per-request latency drops because kernel-launch and VLM-prefix costs amortize across the batch. Source: `git log` theme "Phase III — continuous batching", commits `899c02e..526dded` (2026-04-14 15:21-16:15).

## The TRT-EP × static-shape interaction (the sharp edge)

Our exporters bake static shapes (batch=1). TensorRT engines compile against a specific input shape. ORT's TRT EP, when it encounters a batch shape it doesn't have a cached engine for, rebuilds the engine on every call. For static-shape ONNX, every `max_batch > 1` request triggers a new engine build.

**Numbers (from `scripts/modal_verify_trt_with_batch.py`, commit `e76678c`):**

| max_batch | qps (with TRT EP auto-enabled) | per-call ms  | Behavior |
|-----------|---------------------------------|---------------|----------|
| 1         | 37.7                            | 27            | Normal TRT cache hit |
| 4         | **0.2**                         | **34121**     | **34s engine rebuild on every call** |
| 8         | **0.2**                         | **35129**     | Same, every call rebuilds |

A ~200× pessimization. The user would see 34-second `/act` responses and think the server was broken. Root cause: ONNX was exported with static shapes + the TRT EP's default behavior is to rebuild the engine when it sees an unrecognized input shape.

## Resolution (in production code as of commit `e76678c`)

`src/reflex/runtime/server.py` now drops TRT EP from the provider list when `max_batch > 1`:

```python
# gist
if self.max_batch > 1 and "TensorrtExecutionProvider" in providers:
    providers = [p for p in providers if p != "TensorrtExecutionProvider"]
    logger.warning(
        "Skipping TRT EP (max_batch > 1, static-shape ONNX would rebuild engine per call). "
        "Falling through to CUDA EP."
    )
```

ORT then falls through to `CUDAExecutionProvider`, which handles dynamic batch shapes natively and delivers the 2.88× speedup at batch=16. See ADR `01_decisions/2026-04-14-disable-trt-when-batch-gt-1.md` (in the raw session files; full ADR in `01_decisions/`).

## Proper long-term fix (deferred to v0.2)

Three documented options (from `scripts/modal_verify_trt_with_batch.py` docstring):
1. **Export with dynamic batch shape.** Add `dynamic_axes={"input": {0: "batch"}}` to `torch.onnx.export` and pass `--minShapes/--optShapes/--maxShapes` at TRT engine build. Lets TRT build one engine that handles batch=1..16.
2. **Pre-build engines for common batch sizes** (1, 4, 8, 16) and cache each under its own key. Trades disk + first-run time for hit latency.
3. **Current v0.1 fallback: drop TRT EP when `max_batch > 1`.** Ship's behavior as of today.

Option 1 is the correct long-term fix; deferred because static-shape is currently baked into every exporter and the entire test suite relies on it. Tracked as `export_v2` work in `04_product/prd/export_v2.md`.

## Caveats

1. **Synthetic (fake Identity ONNX) batching only measures queue/scheduling overhead**, not real model compute. `scripts/modal_verify_batching.py` gave ~8× throughput on batch=8 but that was all queue efficiency. `modal_verify_batching_real.py` with real pi0 is the honest number.
2. **Per-item conditioning is ignored in v0.1 batching.** Image, instruction, and state are currently shared across a batch. When VLM prefix is per-item (which it is for real serving), batching needs a batched VLM prefix orchestrator — not built.
3. **Adaptive denoising in batch mode needs per-item convergence detection.** Currently all items early-stop at the same step or none do. Deferred.
4. **Batch timeout 20ms is a magic number.** Too low = batches don't fill. Too high = first request waits too long.
5. **32 concurrent clients** on a single A10G saturates at batch=16. Real fleet workloads (100s of robots) need either larger batches or multi-GPU sharding. Not benchmarked.
6. **Guard wedge applies per-item post-batch** — correct semantically, but adds Python overhead proportional to batch size. Not yet vectorized.
7. **Memory pressure**: batch=16 on pi0 approaches VRAM limits on Orin Nano 8GB; not tested on Jetson hardware yet.
8. **qps numbers include HTTP overhead** — real robot fleets may use gRPC / shared memory, changing the headroom.

## Source commits

- `899c02e 2026-04-14 15:21` — Phase III: continuous batching in reflex serve
- `5b58d35 2026-04-14 15:34` — Phase III.2: real-model batch bench + 3 post drafts
- `b4d3552 2026-04-14 15:50` — 180s timeout + stdout capture on failure
- `492a351 2026-04-14 16:04` — switch to nvcr.io/tensorrt image (cuDNN bundled)
- `48d76fe 2026-04-14 16:15` — README: real-model batching results (2.88× throughput)
- `7651c76 2026-04-14 18:17` — LICENSE + TRT × batching sharp edge test
- `e76678c 2026-04-14 18:25` — **skip TRT EP when --max-batch > 1** (production fix)
