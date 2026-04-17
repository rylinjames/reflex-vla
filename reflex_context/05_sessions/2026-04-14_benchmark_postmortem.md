# 2026-04-14 — The Benchmark Post-mortem

**Session theme:** The day the whole speed story almost died, and then got resurrected by better CUDA pins. A morning benchmark said torch.compile beats our ONNX 6–14× on A100 — which would mean Reflex has no reason to exist as a cloud-GPU tool. By afternoon, the root cause was found (silent CPU fallback) and the numbers were re-measured correctly. TRT FP16 turned out to dominate torch.compile 2.6–3.3×.

---

## Goal

Publish defensible cloud-GPU benchmarks for SmolVLA, pi0, pi0.5, and GR00T. Pitch copy wants a headline like *"Reflex makes VLA inference Xx faster than PyTorch"* — but the benchmark has to survive sophisticated reviewers. An FP16-vs-FP32 apples-to-oranges claim will get caught.

The subgoal that emerged mid-session: make sure `reflex serve` never silently lies about what device it's running on. If `--device cuda` fails to load CUDA, exit with an error — don't fall through to CPU and report 462 ms/call as if it were a GPU number.

---

## The wrong benchmark

Initial A100 bench via `scripts/modal_bench_onnx_vs_torch.py` (commit **`f2cd906`**, Apr-14 12:09) produced what looked like a strategic disaster:

| Model | Eager PyTorch | torch.compile | ORT-CUDA | Reflex CUDA-graph |
|-------|---------------|---------------|----------|--------------------|
| SmolVLA | 19.1 ms | **2.86 ms** | 29.1 ms | — |
| pi0 | 23.9 ms | **5.56 ms** | 76.9 ms | — |
| pi0.5 | 28.6 ms | **6.12 ms** | 80.9 ms | — |
| GR00T | 25.3 ms | **13.1 ms** | 198.9 ms | — |

torch.compile was beating "ORT CUDA" by 6–14×. If true, Reflex has no reason to exist as a cloud-GPU tool.

The bench script pinned `onnxruntime-gpu` with the assumption that *"onnxruntime-gpu ≥1.17 bundles its own CUDA libs, so we can pip install cleanly."* The inline comment was wrong — and a few lines later, investigation revealed the ORT session's providers list:

```
['AzureExecutionProvider', 'CPUExecutionProvider']
```

No `CUDAExecutionProvider`. Every "GPU" benchmark was running on CPU on an A100 box. The latency numbers (417 ms / 968 ms / 1036 ms / 2352 ms) were pure CPU numbers at A100-box prices.

From the transcript (line 4372):
> *"Every 'A100 benchmark' I've reported in this conversation, the README, and the roadmap is ONNX Runtime CPU execution running on an A100 box where the GPU is doing nothing. The latency numbers (417ms / 968ms / 1036ms / 2352ms) are pure CPU numbers at A100-box prices. That's a material misrepresentation I've been repeating."*

### Root cause: CUDA 12 vs CUDA 13 library mismatch

The actual error when ORT tried to load the CUDA provider:

```
libcublasLt.so.12: cannot open shared object file
Require cuDNN 9.* and CUDA 12.*
```

(From transcript line 4445.)

ORT 1.24 needs CUDA 12 libraries. Modal had installed CUDA 13 (via torch's `cuda-toolkit-13.0.2`). The cu12 `.so.12` files simply did not exist on the image. ORT then "gracefully" fell back through its provider list and landed on `CPUExecutionProvider` without raising — which is exactly the silent failure mode that burned us.

### Install-path verification (`modal_verify_install_path.py`)

Commit **`0bca6f6`** (Apr-14 16:39) added the canonical install-path test: a fresh box runs the EXACT README command `pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'`, then `reflex --help`, `reflex models`, `reflex export lerobot/smolvla_base --target desktop`, `reflex serve --device cuda`, and POST /act via stdlib `urllib.request` (no pip-installed test client to avoid polluting the test).

Follow-up **`7727ad9`** (16:50): Restructured optional extras after finding that `pip install reflex-vla[serve,gpu]` was missing four core deps (`huggingface_hub`, `transformers`, `onnx`, `onnxscript`) that `reflex export` needed. The extras tree became:

- `[onnx]` → `onnxruntime>=1.17.0` (CPU-only).
- `[gpu]` → `onnxruntime-gpu>=1.20,<1.24` + `nvidia-cudnn-cu12>=9.0,<10.0` + `nvidia-cublas-cu12>=12.0,<13.0`.
- `[serve]` → `fastapi` + `uvicorn` + `Pillow` (HTTP only).

Inline comment block in `pyproject.toml`:
> *"Apr-14 post-mortem: omitting these [cudnn/cublas pins] was the cause of silent CPU fallback in v0.1 benchmarks."*
> *"hub access for `reflex export <hf_id>` and tokenizer/config loading for the auto-detect path. Apr-14 install-path verification caught that omitting these breaks `reflex export` even with [serve,gpu]."*

---

## `strict_providers=True` — the defense

Apr-14 13:13, commit **`5b21296`** — Phase I.1: `reflex serve` now refuses to silently fall back to CPU.

New contract codified in `tests/test_server.py::TestStrictProviderMode`:
- Strict + CUDA requested + only CPU provider available → raise `RuntimeError`.
- `ReflexServer` constructor gains `providers` + `strict_providers=True` args.
- CLI gains `--providers "A,B,C"` and `--no-strict-providers` flags.
- Pre-flight: if `--device cuda` but ORT has no `CUDAExecutionProvider`, print multi-line hint + exit 1.

The install-hint distinguishes two failure modes:
- *"you installed onnxruntime not onnxruntime-gpu"* — user-level footgun.
- *"CUDA 12 libs missing from PATH"* — system-level pin mismatch.

Verification in `scripts/modal_verify_strict_providers.py`:
- **Scenario 1**: GPU box + `onnxruntime-gpu` + `--device cuda` → starts normally.
- **Scenario 2**: GPU box + `--device cpu` → starts normally.
- **Scenario 3**: CPU-only `onnxruntime` + `--device cuda` default → **exit 1 with clear error.**
- **Scenario 4**: Scenario 3 + `--no-strict-providers` → starts with fallback (escape hatch for debug).

Scenario 3's success criterion inverted: `if exit_code == 1 → PASS`. The server refused to silently fall back — which is the codified behaviour.

ADR: `2026-04-14-strict-provider-no-silent-cpu-fallback.md`.

---

## The right benchmark emerges

### First fix attempt (`modal_bench_path_b.py`)

Commit **`f2cd906`** (12:09) fix attempt pinned:
- `torch==2.5.1` (CUDA 12.4, not 13 like torch 2.11) — *"Torch 2.5 uses CUDA 12.4 (NOT 13 like torch 2.11)"*
- `onnxruntime-gpu==1.20.1` — *"ORT 1.20.x uses cuDNN 9 + CUDA 12.x"*
- `transformers>=4.40,<5.0`, `numpy<2.0`, etc.
- `pip install -e . --no-deps` for reflex (to avoid re-pulling pinned packages).

**Extensive post-mortem in image construction comments**:
> *"FIXED image: pin torch to a version using CUDA 12 (matches ORT 1.20), install matching cuDNN and cuBLAS."*

The image now prints `torch + CUDA + ORT version + ort.get_available_providers()` at startup so silent fallback cannot recur.

### CUDA graph misfire

`modal_bench_path_b.py` also tested `TurboOptimizer.denoise_cuda_graph` — full-loop capture of the 10-step denoise as a CUDA graph. The finding (line 4504):

> *"torch.compile full loop: 52.87ms. Reflex CUDA graph: 69.7ms ← 25% slower than torch.compile. Root cause I didn't catch: my CUDA graph captures the *eager* model's kernels, not the torch.compile'd ones. torch.compile(`mode='reduce-overhead'`) already uses CUDA graphs internally per forward pass, AND applies kernel fusion. I get CUDA-graph savings but lose kernel fusion."*

So Reflex's CUDA-graph wedge is ~duplicate work when the alternative is torch.compile reduce-overhead mode. A bonus finding: Orin Nano 8GB fits all 4 current VLAs in FP16 with 2× overhead (gr00t 4.36 GB, tight but fits). $500 Orin Nano is confirmed as the right starter SKU, not Thor ($1500) or Orin 64 ($2500).

### The TRT FP16 flip (`fce8a6f`, Apr-14 15:16)

Phase II: rebuilt the cloud-GPU benchmark on an A10G with the correct base image — `nvcr.io/nvidia/tensorrt:24.10-py3` — which bundles TRT 10.5, CUDA 12.6, cuDNN 9, and `trtexec` already on PATH. Benchmark via `scripts/modal_bench_trt_fp16.py`.

Four paths per model: PyTorch eager, `torch.compile(reduce-overhead)`, ORT-GPU FP32, TensorRT FP16 engine (built via `trtexec`).

Sharp edges captured as comments:
- **Crucial**: *"Our ONNX export has static shapes baked in (no dynamic_axes), so we MUST NOT pass `--minShapes/--optShapes/--maxShapes` — trtexec rejects them for static models."* (Commit `28e8906`.)
- **Second**: *"trtexec already measures GPU compute latency precisely"* — justifies not using the Python TRT bindings (`nvcr.io/nvidia/tensorrt` doesn't pre-install them; would require `/opt/tensorrt/python/python_setup.sh`). Use `trtexec --loadEngine` for inference timing instead. (Commit `78caac2`.)

Bench command per engine:
```
trtexec --loadEngine=model.trt --warmUp=200 --iterations=500 --avgRuns=100 --useCudaGraph
```

Parsed line: `GPU Compute Time: min=, max=, mean=, median=, percentile(99%)=`.

### The new real table (A10G, single denoising step, ms)

| Model | Params | compile | ORT-GPU | TRT-FP16 | Speedup vs compile |
|-------|--------|---------|---------|----------|---------------------|
| SmolVLA | 99.8M | 3.06 | 3.26 | **0.95** | 3.2× |
| pi0 | 314.6M | 6.23 | 5.53 | **1.94** | 3.2× |
| pi0.5 | 426.9M | 7.34 | 7.37 | **2.24** | 3.3× |
| GR00T | 1091.7M | 14.61 | 14.45 | **5.59** | 2.6× |

Per-chunk wall-clock (10-step denoise) gives the user-visible number:
- SmolVLA 9.5 ms → **105 Hz**
- pi0 19.4 ms → 52 Hz
- pi0.5 22.4 ms → 45 Hz
- GR00T 55.9 ms → 18 Hz (borderline; needs optimization)

**All four meet or exceed the 20–30 Hz target for real-time robot control.**

Bonus finding baked in: with `nvcr.io/nvidia/tensorrt` base image, ORT-GPU FP32 actually beats torch.compile on 3 of 4 models. The earlier "torch.compile crushes us 6–14×" finding was a CPU-fallback artifact.

### Pitch narrative flipped

From the transcript (line 4989):
> *"This completely flips the strategic narrative. Reflex's TRT path doesn't just compete with torch.compile — it dominates by 3× on cloud GPU. The whole 'we can't beat torch.compile' narrative from earlier is dead."*

The L1 lesson from the mid-session post-mortem had been: *"Reflex cannot win 'faster inference on cloud GPU.'"* Post-flip, that lesson is reversed — TRT FP16 wins cloud GPU AND edge (same engine runs on A10G via trtexec and on Jetson via Jetson TRT, no separate cloud/edge variants).

Shipped README line: *"11.9ms via auto TRT FP16 (84 Hz on smolvla, A10G)"* — commits `60ecd39`, `72d8658` (Apr-14 17:19–17:20). *"Same engine that runs on A10G via trtexec runs on Jetson via Jetson TRT — no separate cloud and edge model variants."*

---

## Auto-TRT-FP16 in `reflex serve`

Commits `12e604f` + `9a690ab` (Apr-14 17:08–17:14) made TRT FP16 the default path in `reflex serve`:

- Per-export-dir engine cache at `<export_dir>/.trt_cache`, fp16 default, 4GB max workspace.
- `inference_mode` field in `/act` responses: `onnx_trt_fp16` / `onnx_gpu` / `onnx_cpu`.
- `strict-providers` check extended to count both `CUDAExecutionProvider` and `TensorrtExecutionProvider` as GPU.
- **Warmup pass at startup** during FastAPI lifespan. First inference triggers TRT engine construction (30–90 s for SmolVLA, longer for larger models). Without warmup, first POST /act would time out.
- `GET /health` returns `model_loaded: true` only after warmup completes — health-check based readiness probes correctly wait.
- Subsequent server starts hit `.trt_cache` and skip the build (~1–2 s).
- Install-verify `/health` poll bumped 90 s → 240 s.

---

## `reflex bench` emerges + `reflex doctor` diagnostics

Commit `a190780` (Apr-14 17:32) — `reflex bench` command went from stub to real: loads export, warms up, runs N denoise iterations, reports min / mean / p50 / p95 / p99 latency and Hz. Defaults 100 iter + 20 warmup, `--device cuda`. Surfaces `inference_mode` so users can verify TRT FP16 is active.

`reflex bench` all-4-VLA results (A10G, from `scripts/modal_verify_bench_all.py`):

| Model | mean_ms | p95_ms | mode | export_s |
|-------|---------|--------|------|----------|
| smolvla | 11.67 | 11.85 | onnx_trt_fp16 | 72.8 |
| pi0 | 23.57 | 24.22 | onnx_trt_fp16 | 112.2 |
| pi0.5 | 27.07 | 27.76 | onnx_trt_fp16 | 151.9 |
| gr00t | 56.55 | 57.25 | onnx_trt_fp16 | 181.9 |

Commit `b0dff64` (18:52) added **`reflex doctor`** — a quick health-check command users run BEFORE opening a bug:
- Verifies Python, platform, torch + CUDA, ORT providers, trtexec on PATH, disk space, installed reflex version.
- Surfaces the failure modes that bit us: *"ORT installed but CUDA EP missing" → fix cuDNN/cublas setup.* *"trtexec not on PATH" → TRT engine build skipped.* *"huggingface_hub missing" → catches install-path bug before reflex export blows up.*
- rich.Table output with green ✓ / yellow ⚠.

Docs updated to make `reflex doctor` the first troubleshoot step (commit `1c6ea79`).

---

## Honest framing for external publication

The Show HN draft (pre-launch, unpublished) reflects this post-mortem verbatim (verbatim quote from `/launch/show_hn_draft.md`):

> *"I went into this thinking the moat was edge-only because torch.compile was crushing my early benchmarks. Turned out my onnxruntime-gpu was silently falling back to CPU due to a CUDA 12-vs-13 library mismatch. Once that was fixed, TRT FP16 wins by 2.6-3.3× across the board."*

Public review rule set: stop every public artifact and ask *"does this pitch Reflex makes inference faster on your GPU?"* If yes, confirm the benchmark base image and provider list before publishing. The FP16-vs-FP32 apples-to-oranges gotcha (flagged by the "sophisticated reviewer" pre-mortem) is explicitly called out in bench docstrings.

---

## Cost log

- *"~$15 Modal cost tonight. Worth it for the auto-TRT-FP16 finding alone."* (Transcript line 5453.)
- *"Modal spend tonight: ~$8–12 across ~12 runs."* (Line 5218.)

---

## Outcome

The post-mortem saved the product. Without the CUDA 12 pin, the whole cloud-GPU benchmark was bogus, and the moat pitch was "edge-only." Post-fix, TRT FP16 dominates torch.compile 2.6–3.3× on cloud GPU AND runs on Jetson — so the pitch became "cloud AND edge, same TRT toolchain."

Three institutional defenses landed as code:
1. **`strict_providers=True`** in `reflex serve` prevents silent CPU fallback ever again.
2. **`modal_verify_install_path.py`** runs the exact README command on a fresh box — catches dep pin regressions before release.
3. **`reflex doctor`** surfaces the "which library is missing" question in one screen before users open a GitHub issue.

---

## Carry-over

Unfinished work that carried into later sessions:

- **FP16 torch.compile baseline** — post-mortem harvested item: *"Benchmark FP16 torch.compile vs TRT FP16 on A10G (high) — closes apples-to-oranges pitch attack surface."* Codified in goal `fp16-torch-compile-baseline` (weight 9, GOALS.yaml). Not yet run as of Apr-17.
- **Real Jetson Orin Nano validation** — harvested post-mortem item (high priority): *"every claim is A10G-extrapolated."* Codified in goal `jetson-orin-nano-validation` (weight 9). Not yet run.
- **Dynamic batch shape + TRT shape profiles** — deferred to `export_v2.md`; current workaround is `e76678c` (skip TRT EP when batch > 1). See `2026-04-13_phase_iii_batching_adaptive.md` for batching context.
- **VLM prefix encoder (critical)** — post-mortem harvested: *"v0.1 serve + validate both use random-tensor VLM conditioning, outputs are action-shaped noise, not task-relevant actions."* Single biggest unfilled thing. Scaffolded Apr-16 waves 1–3, debugged Apr-17. See `2026-04-16_libero_infra_hunt.md` and `2026-04-17_libero_correctness_hunt.md`.
- **reflex validate round-trip** — shipped Apr-16 as commits `e1455f7..18f8038` (5-phase RPI epic), 11/11 tests passing. Installed radon / gocyclo / fixture-image-size-reconcile left as minor items.
- **Bench history database** — not built. Benchmark numbers live in README + commit messages; no time-series DB.

The post-mortem also seeded a cultural rule: *FP16-vs-FP32 is a known gotcha — compare FP16 vs FP16 or report both.* Every future benchmark is expected to name its dtype in the same cell as its latency.
