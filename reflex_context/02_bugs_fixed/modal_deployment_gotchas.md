# Modal Deployment Gotchas

Modal-specific problems that bit us during Reflex VLA development. Modal is the deploy/test substrate for Reflex (A10G proxies for Jetson Orin Ampere SM_86; A100 for heavier workloads). These gotchas cost real hours of debugging when encountered mid-run.

Sources: `sessions_md.md`, `modal_scripts.md`, `git_history.md`, `current_session.md`, `modal_apps_and_pm_docs.md`.

---

## Gotcha 1: `subprocess.PIPE` deadlock — OS pipe buffer fills

**Symptom:** Child process (e.g. `reflex serve` launched as a subprocess, or `vla-eval` wrapped by the Modal script) hangs forever despite looking healthy. Python's `subprocess.run(capture_output=True)` or `Popen` with `stdout=subprocess.PIPE` never returns.

**Root cause:** OS pipe buffers are small — typically ~64KB on Linux. Once a subprocess fills the pipe buffer faster than the parent reads, the subprocess blocks on `write()`. If the parent is waiting for the child to exit before reading (e.g. `subprocess.run(capture_output=True)`), both processes deadlock.

**Fix:** Redirect child stdout/stderr to a FILE, not a pipe:
```python
log_fh = open(log_path, "wb")
process = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
```

Then tail the file from the parent for live visibility. Or use line-buffered `Popen` with explicit `bufsize=1` and an iterator over `process.stdout`.

**Scripts with post-mortem comments:**
- `scripts/modal_e2e_demo.py` — "server stdout is redirected to a FILE... to avoid pipe-buffer deadlock"
- `scripts/modal_verify_batching_real.py` — "Use a real file for stdout — subprocess.PIPE deadlocks if the process logs more than one OS pipe buffer (~64KB) before we read."
- `scripts/modal_libero10.py` — applies this pattern

**Sources:**
- `modal_scripts.md` cross-cutting pattern: "Pipe-buffer deadlock lesson: several scripts log 'Use a real file for stdout'"
- `git_history.md` commit `45794b0` ("E2E scripts redirect server logs to file, NOT subprocess.PIPE — 64KB buffer deadlocks child process")
- `sessions_md.md` line 22, 24

---

## Gotcha 2: `--detach` mode required for long runs; silent exit otherwise

**Symptom:** A long-running Modal function (e.g. a LIBERO-10 run, a 10-min benchmark) appears to exit cleanly from the local shell after a few minutes, but the actual Modal function either (a) never finished, or (b) ran for 30+ min and returned results Claude couldn't see.

**Root cause:** Without `--detach`, `modal run script.py` streams logs until the Modal function exits — and if the local SSH session drops, bash is interrupted, or the command times out locally, the tether breaks. Without detach, the local loss of the streaming connection may or may not terminate the remote function depending on Modal's version and mode.

**Fix:** Use `modal run --detach scripts/modal_libero10.py` for anything >10 min. The local command returns immediately; the Modal function runs to completion on its own. Check progress via:
```bash
modal app list
modal app logs <app_id>
```

Plan to check back by `app_id` rather than keeping a terminal open.

**Sources:**
- `sessions_md.md` line 98 ("`modal run --detach` returns immediately — the local bash task ends but the Modal function keeps running. Check via `modal app list` / `modal app logs <app_id>`.")
- `sessions_md.md` line 99 ("Modal A100 spot is preemptible; runs get new workers mid-execution.")

---

## Gotcha 3: Stdout buffering — `subprocess.run` vs `Popen` + line-buffered

**Symptom:** Modal function runs for 37 minutes with NO visible output. Then all output dumps at once when subprocess exits. Impossible to tell if the run is healthy or stuck during the 37 minutes.

**Root cause:** `subprocess.run(cmd, capture_output=True)` buffers stdout/stderr in a bytes buffer that's only surfaced after `.returncode` is set (process exit). During a long-running subprocess (LIBERO-10 eval ~40-60 min), this means zero telemetry. Compounded by: if stdout is pointed at a file handle, Python may still buffer writes until flush or close.

**Fix:** Use `Popen` with explicit `bufsize=1` (line-buffered) and an iterator:
```python
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
for line in process.stdout:
    print(line.rstrip(), flush=True)
```

Combined with Gotcha 1 — if the subprocess may produce more than 64KB of output total, write to a file and tail the file from the parent rather than reading Popen.stdout directly.

Add an **idle-timeout guard**: kill if no line arrives for N seconds. `scripts/modal_libero10.py` uses 600s for LIBERO (osmesa first-scene compile is slow) and comments on why.

**Sources:**
- `sessions_md.md` line 101 ("Line-by-line subprocess output: don't use capture_output=True with long-running subprocesses on Modal — buffering obscures progress. Stream.")
- `modal_scripts.md` `modal_libero10.py` ("subprocess.run(capture_output=True) ... buffers until the subprocess exits, which meant we couldn't tell if a run was hung vs mid-episode for 50+ minutes.")

---

## Gotcha 4: Image GC after ~3 days — `modal app logs --since` fails

**Symptom:** `modal app logs <app_id>` returns empty output even when the app appears in `modal app list`. Happens for older apps or after a Modal platform maintenance window.

**Root cause:** Modal garbage-collects images and log storage after a retention window. The `modal app list --json` returned 2026-04-05 app `ap-MrSsaMvCuiwlYLaTCs8gOb` but `--since 5d` returned no output — predates retention window.

**Fix:** Capture critical log output locally immediately after a run. For any run where the outcome is uncertain:
1. Use `--detach` for the run itself.
2. Poll `modal app logs <app_id>` and dump to a local file within the first hours.
3. Don't rely on Modal's log retention for long-term forensics.

For current debugging, if logs are GC'd, re-run the same Modal script (image is cached so build is fast) and capture fresh logs.

**Sources:**
- `modal_apps_and_pm_docs.md` ap-MrSsaMvCuiwlYLaTCs8gOb ("Unreadable — `modal app logs --since 5d` returned no output. Predates the 5-day window or logs GC'd.")

---

## Gotcha 5: `CUDAExecutionProvider` silent fallback when `onnxruntime-gpu` not installed

**Symptom:** Every "A100 benchmark" reported latency numbers of 462ms for SmolVLA, 999ms pi0, 1163ms pi0.5, 2352ms GR00T — i.e. CPU-class numbers running on A100 hardware. Compared against torch.compile numbers of 25-113ms on the same box. Made it appear torch.compile was 6-14× faster than Reflex on GPU.

**Root cause:** Modal's default base image pulls `onnxruntime` (CPU-only package), NOT `onnxruntime-gpu`. When we create an ORT session requesting `CUDAExecutionProvider`, ORT silently falls back to CPU without erroring if CUDA EP isn't loadable. The benchmark STATES CUDA but runs on CPU. The failure mode:
- `pip install onnxruntime` installs the CPU-only variant.
- OR `onnxruntime-gpu` installed but CUDA libs don't match (cf. Gotcha 6).
- `ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])` silently degrades.

**Fix:** 
1. Install `onnxruntime-gpu>=1.20,<1.24` explicitly (NOT `onnxruntime`).
2. Implement "strict provider mode" in `reflex serve`: if CUDA requested but CUDAExecutionProvider missing from `ort.get_available_providers()`, EXIT with error, do not silently fall back. Flag: `--strict-providers` default on, `--no-strict-providers` to opt out.
3. Pre-flight check: on server startup, verify `ort.get_available_providers()` includes `CUDAExecutionProvider` OR `TensorrtExecutionProvider` before calling `load_model()`.

Commit `5b21296` landed this: "Phase I.1: reflex serve refuses to silently fall back to CPU". Code in `src/reflex/runtime/server.py`, hint text in `src/reflex/cli.py`.

**Sources:**
- `sessions_md.md` line 14 ("ORT CUDA silent CPU fallback")
- `git_history.md` theme "Apr-14 GPU benchmark post-mortem — ORT silently CPU-fallback"
- `current_session.md` line 4372 ("Every 'A100 benchmark' I've reported in this conversation, the README, and the roadmap is ONNX Runtime CPU execution running on an A100 box where the GPU is doing nothing.")
- `modal_scripts.md` `modal_verify_strict_providers.py` section

---

## Gotcha 6: CUDA 12 vs 13 library mismatch (`libcublasLt.so.12: cannot open shared object file`)

**Symptom:** `onnxruntime-gpu` installed correctly, but CUDA EP loading fails with:
```
libcublasLt.so.12: cannot open shared object file
Require cuDNN 9.* and CUDA 12.*
```
Silent fallback to CPU (Gotcha 5) unless strict-providers active.

**Root cause:** Modal's default torch install pulls `torch` that may bundle CUDA 13 libs (via `cuda-toolkit-13.0.2`). `onnxruntime-gpu 1.24` wants CUDA 12. Mismatch: the CUDA 13 libs don't satisfy ORT's CUDA 12 symbol lookups. Also: pip's `nvidia-cudnn-cu12` wheel is missing `libcudnn_adv.so.9` that ORT 1.20+ requires.

**Fix:** Pin the stack explicitly in the Modal image:
```python
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1",        # Torch 2.5 uses CUDA 12.4 (NOT 13 like torch 2.11)
    "onnxruntime-gpu==1.20.1",  # ORT 1.20.x uses cuDNN 9 + CUDA 12.x
    "nvidia-cudnn-cu12>=9.0,<10.0",
    "nvidia-cublas-cu12>=12.0,<13.0",
    "numpy<2.0",
    "transformers>=4.40,<5.0",
)
```

Or use `nvcr.io/nvidia/tensorrt:24.10-py3` as the base image, which already has CUDA 12 + cuDNN 9 (including `libcudnn_adv.so.9`) on the system path. The pip-installed `nvidia-cudnn-cu12` wheel is insufficient — the TRT container is required for full cuDNN 9 coverage.

**Sources:**
- `current_session.md` line 4445 (`libcublasLt.so.12: cannot open shared object file... ORT 1.24 needs CUDA 12 libs, but Modal installed CUDA 13 (via torch's cuda-toolkit-13.0.2)`)
- `current_session.md` line 4504 ("missing libcudnn_adv.so.9. Need to explicitly install nvidia-cudnn-cu12==9.*")
- `modal_scripts.md` cross-cutting: "Use NVIDIA's TRT container so cuDNN 9 (including libcudnn_adv) is already on the system path. The pip-installed nvidia-cudnn-cu12 wheel is missing libcudnn_adv.so.9 which ORT 1.20+ requires."
- `modal_apps_and_pm_docs.md` "GPU — `onnxruntime-gpu>=1.20,<1.24` + `nvidia-cudnn-cu12>=9.0,<10.0` + `nvidia-cublas-cu12>=12.0,<13.0`. Comment says: 'Apr-14 post-mortem: omitting these was the cause of silent CPU fallback in v0.1 benchmarks.'"

---

## Gotcha 7: Pydantic `ForwardRef` inside `create_app()`

**Symptom:** `reflex serve` hangs at uvicorn startup, no error, no log. Server becomes unresponsive immediately.

**Root cause:** `HealthResponse(BaseModel)` was defined INSIDE the `create_app()` factory function. Pydantic 2.13 + FastAPI can't resolve `ForwardRef` for locally-scoped classes when FastAPI builds the `TypeAdapter`. The TypeAdapter construction raises, but uvicorn's startup wraps it silently and the server crashes without a useful message.

**Fix:** Move all `BaseModel` subclasses to module level. Also switch from deprecated `@app.on_event('startup')` to the async `lifespan` context manager (newer FastAPI pattern):
```python
# MODULE SCOPE
class HealthResponse(BaseModel):
    model_loaded: bool
    inference_mode: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown
```

Fixed in commit `45794b0` as part of the pi0 support work.

**Sources:**
- `current_session.md` line 3391, 3509 ("Pydantic 2.13 + FastAPI can't resolve `HealthResponse` defined inside `create_app()` — it's a locally-scoped class and the TypeAdapter ForwardRef fails. Moving the classes to module level.")
- `git_history.md` commit `45794b0` ("Bug fix: FastAPI HealthResponse BaseModel defined inside create_app() → Pydantic 2.13 can't resolve ForwardRef for locally-scoped classes. Moved to module scope + switched from deprecated @app.on_event('startup') to async lifespan context manager.")

---

## Gotcha 8: TRT engine rebuild on static-shape batches — 34s per call

**Symptom:** `reflex serve --max-batch 4` works — but every single `/act` request takes 34 seconds. Throughput collapses from 38 qps (batch=1) to 0.2 qps. 200× pessimization.

**Root cause:** Our ONNX exporters bake STATIC shapes (batch=1, chunk=50, action_dim=X). TRT engines are compiled against the static shape at engine-build time. When `--max-batch N > 1` fires a batched request, ORT's TensorrtExecutionProvider encounters an unknown shape and REBUILDS the engine from scratch for the new batch dimension. Each rebuild = 34s. Subsequent same-shape requests should hit the cache, but static-shape ONNX is a mismatch — TRT rebuilds anyway on each distinct shape.

**Fix:** Short-term (v0.1): drop TRT EP when `--max-batch > 1`. ORT falls through to `CUDAExecutionProvider`, which handles dynamic batch shapes natively. Gives 2.88× throughput at batch=16. Code in `src/reflex/runtime/server.py` (commit `e76678c`).

```python
# server.py pseudocode:
providers = []
if max_batch == 1 and "TensorrtExecutionProvider" in available:
    providers.append("TensorrtExecutionProvider")
providers.append("CUDAExecutionProvider")
```

Long-term (v0.2): dynamic batch shape export with `dynamic_axes={"input": {0: "batch"}}` + TRT shape profiles via `trtexec --minShapes/--optShapes/--maxShapes` for batch=1/4/8/16. Deferred (council reprioritization deleted it from v0.2 scope).

**Sources:**
- `sessions_md.md` line 13 ("TRT EP rebuilding engines per input shape — when reflex serve --max-batch > 1, TensorRT Execution Provider was rebuilding engines per call, producing 34s/call, a ~200× pessimization.")
- `git_history.md` commit `e76678c` ("serve: skip TRT EP when --max-batch > 1 (avoids 34s rebuild penalty)")
- `modal_scripts.md` `modal_verify_trt_with_batch.py` section (the diagnostic script)
- `current_session.md` line 5077, 6224

---

## Gotcha 9: `trtexec --minShapes/--optShapes/--maxShapes` rejected for static-shape ONNX

**Symptom:** `trtexec` build fails with error message like `Static model does not take explicit shapes...`.

**Root cause:** `trtexec` only accepts `--minShapes/--optShapes/--maxShapes` when the ONNX graph has DYNAMIC shapes (at least one input with a named dim). Our exporters bake static shapes. Passing these flags to a static-shape ONNX errors.

**Fix:** Strip the flags for static-shape ONNX. When exporting with `dynamic_axes={}` (static), `trtexec` uses the baked shape; no explicit-shape flags needed. Commit `28e8906` applies this.

Related: `scripts/modal_bench_trt_fp16.py` comments explicitly: "Our ONNX export has static shapes baked in (no dynamic_axes), so we MUST NOT pass --minShapes/--optShapes/--maxShapes — trtexec rejects them for static models. The engine is fixed at the export shape (batch=1, chunk=50, action_dim from model)."

**Sources:**
- `sessions_md.md` line 19 ("`trtexec --minShapes/--optShapes/--maxShapes` rejected — fails when the ONNX has static shapes.")
- `git_history.md` commit `28e8906` ("TRT bench: drop explicit shape flags (ONNX has static shapes)")
- `modal_scripts.md` `modal_bench_trt_fp16.py` crucial post-mortem comment

---

## Gotcha 10: `tensorrt.Logger` Python bindings missing in `nvcr.io` image

**Symptom:** `ImportError` or `AttributeError` when trying to use `tensorrt.Logger` or `tensorrt.Builder` in a Python script inside the `nvcr.io/nvidia/tensorrt:24.10-py3` container.

**Root cause:** `nvcr.io/nvidia/tensorrt:24.10-py3` ships with the TRT runtime + `trtexec` binary, but the Python bindings for TRT are NOT installed by default. They require running `/opt/tensorrt/python/python_setup.sh` which the container doesn't execute.

**Fix:** Don't use Python TRT bindings for benchmarking. Use `trtexec --loadEngine=... --warmUp=200 --iterations=500 --avgRuns=100 --useCudaGraph` instead. Parse the stdout for timing info:
```
GPU Compute Time: min=, max=, mean=, median=, percentile(99%)=
```

trtexec output is stable and parseable. Avoid the Python dance.

**Sources:**
- `sessions_md.md` line 20 ("`tensorrt.Logger` attribute missing — Python bindings not installed in `nvcr.io` container. Switched to `trtexec --loadEngine` for benchmarking (bypasses Python bindings).")
- `git_history.md` commit `78caac2` ("TRT bench: use trtexec --loadEngine instead of Python bindings")
- `modal_scripts.md` `modal_bench_trt_fp16.py` ("'trtexec already measures GPU compute latency precisely' — justifies not using the Python TRT bindings")

---

## Gotcha 11: Modal image build takes ~5-10 min for ML deps

**Symptom:** First Modal run for a new image configuration sits at "building image" for 5-10 minutes. Subsequent runs with cached image start in <30s.

**Root cause:** Heavy ML deps have long install times:
- torch: ~2 GB download + install
- transformers>=4.51: multiple C extensions compile
- onnx/onnxruntime-gpu: binary wheels but large
- lerobot from git: non-standard install structure (Gotcha 12)
- TensorRT container image: 10 GB pull alone

**Fix:** Accept the build time upfront; don't declare failure before 10 min. Modal caches the image afterward — repeated runs are fast. Batch image changes: change the `.pip_install(...)` list once, rebuild once.

For fast iteration, prefer local execution where possible (~100× cheaper than Modal, per `current_session.md` line 11574).

**Sources:**
- `sessions_md.md` line 97 ("Modal image builds are slow for ML-heavy deps: lerobot, torch, transformers, onnx, fastapi on Python 3.12 installs take ~5-10 min. Always allow build time before declaring failure.")
- `sessions_md.md` line 115 ("Modal's TRT container: 10GB pull. Extraction is slow (unpacking rootfs). Then deps install on top, then 4 checkpoints download, then TRT engines build. ~10 min full cycle.")

---

## Gotcha 12: `pip install git+lerobot` succeeds but `import lerobot` fails

**Symptom:** Image build completes. `pip show lerobot` returns a version. But `python -c "import lerobot"` fails with `ModuleNotFoundError`.

**Root cause:** `lerobot` has a non-standard install structure. The setup.py declares the package but the submodules aren't packaged correctly for pip's install-from-git. The install LOOKS successful but the package isn't usable.

**Fix:** 
1. `git clone https://github.com/huggingface/lerobot.git` to a known path.
2. `pip install -e /path/to/lerobot`.
3. Reference via known path if needed.

Alternatively, install `lerobot` from pypi (not git) — commit `7cb1cd7` shows this attempt but pypi version was missing `lerobot.common` submodule at the time.

**Sources:**
- `sessions_md.md` line 25 ("lerobot package install-structure quirk — `pip install git+lerobot` installs but `import lerobot` fails because the package has a non-standard install structure. Took 5 iterations of Modal debugging.")
- `git_history.md` commit `7cb1cd7` ("fix: install lerobot from GitHub (PyPI package lacks lerobot.common)")

---

## Gotcha 13: Modal A100 spot is preemptible mid-run

**Symptom:** A 40-min benchmark halfway through reports a new worker assignment in logs. Progress continues but cached state may be lost.

**Root cause:** A100 spot instances on Modal are preemptible. If a higher-priority user needs the GPU, Modal will migrate your function to a different worker. The migration is transparent to code but any in-memory state that wasn't yet persisted to disk/remote is lost.

**Fix:** Write intermediate results to disk / Modal volumes frequently. Don't rely on in-memory accumulation over hour+ runs. For critical runs, use dedicated (non-spot) A100 at higher cost.

**Sources:**
- `sessions_md.md` line 99 ("Modal A100 spot is preemptible; runs get new workers mid-execution.")
- `sessions_md.md` line 24

---

## Gotcha 14: `.pyc` cache persists across iterations

**Symptom:** "Old behavior" reappears after editing source code. Feels like the edit didn't take effect.

**Root cause:** Python caches bytecode as `.pyc` files in `__pycache__/` directories. When the interpreter can load a valid `.pyc` with a matching magic number and older mtime, it skips parsing the `.py`. In dev loops (especially Modal, where code is uploaded fresh each run), stale `.pyc` can persist if the pattern injects code into an installed package location (e.g. LIBERO patching).

**Fix:** Aggressively nuke `__pycache__/` directories after any code modification:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

Or set `PYTHONDONTWRITEBYTECODE=1` during dev to skip writing .pyc altogether.

**Sources:**
- `sessions_md.md` line 26 (".pyc cache / 4th input() call — old code was running despite edits because cached bytecode persisted. 'Must be a .pyc cache or a 4th call' — had to nuke caches + add catch-all.")
- `git_history.md` commit `cb742db` ("fix: aggressive regex patch for ALL input() patterns + nuke .pyc caches")

---

## Gotcha 15: Modal trial credits — ~$15 for a full night of iteration

**Symptom:** Burn through Modal credits faster than expected.

**Root cause:** Modal A100 spot ≈ $1.10/hr; A10G ≈ cheaper. A LIBERO run takes 60-120 min. A benchmark takes 10-30 min. Per-night iteration budget: 15-30 runs = ~$15-30 without aggressive caching.

**Fix:**
- Cache image: don't change pip_install deps for every iteration.
- Local-first workflow: ~100× cheaper than Modal. Export, validate, smoke-test locally before sending to Modal.
- Batch experiments: run 4 models in one Modal function (`modal_e2e_all_models.py` pattern) rather than 4 separate Modal runs.

**Sources:**
- `current_session.md` line 5453 ("~$15 Modal cost tonight. Worth it for the auto-TRT-FP16 finding alone.")
- `current_session.md` line 5218 ("Modal spend tonight: ~$8-12 across ~12 runs.")
- `current_session.md` line 11574 ("Local iteration is ~100× cheaper than Modal")

---

## Gotcha 16: `caffeinate -dimsu &` — prevent Mac sleep during long Modal streams

**Symptom:** Mac goes to sleep mid-benchmark. Modal stream disconnects.

**Root cause:** macOS defaults sleep after 15 min idle. A 40-min Modal streaming run where the user isn't typing will trigger sleep. Sleep disconnects the streaming session.

**Fix:** Before a long Modal run, run `caffeinate -dimsu &` to prevent sleep. Kill after via `killall caffeinate`.

Alternative: use `modal run --detach` (Gotcha 2) and poll logs by `app_id` — doesn't need a live streaming session.

**Sources:**
- `sessions_md.md` line 114 ("`caffeinate -dimsu &` to prevent Mac sleep during long-running Modal streams. Kill via `killall caffeinate`.")

---

## Gotcha 17: Background task output files in tmp dir — ~15 runs accumulated during LIBERO hunt

**Symptom:** `/private/tmp/claude-501/-Users-romirjain/.../tasks/*.output` fills with many output files.

**Root cause:** Each background task Claude spawns writes to a unique output file under the task temp directory. During the LIBERO hunt (18 iterations in 75 min), ~15 output files accumulated.

**Fix:** No user-facing action required. Claude's task output files are managed by the harness. If disk pressure becomes an issue, manually clean `/private/tmp/claude-501/.../tasks/` older than a few days.

**Sources:**
- `sessions_md.md` line 100 ("Background task output files land in /private/tmp/claude-501/-Users-romirjain/.../tasks/*.output. ~15 runs accumulated during the LIBERO hunt.")

---

## Cross-cutting: Image construction template

Canonical image template that avoids all the above gotchas, consolidated from `modal_bench_path_b.py`, `modal_verify_strict_providers.py`, `modal_libero10.py`:

```python
import modal

image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.10-py3", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2", "curl")
    .pip_install(
        "torch==2.5.1",            # CUDA 12.4
        "onnxruntime-gpu==1.20.1",  # cuDNN 9 + CUDA 12
        "onnx",
        "onnxscript",
        "transformers>=4.40,<5.0",
        "safetensors",
        "huggingface_hub",
        "numpy<2.0",
        "Pillow",
        "typer",
        "rich",
        "pydantic>=2.0",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "httpx",
    )
    .add_local_dir("src/reflex", remote_path="/root/src/reflex")
    .run_commands("cd /root && pip install -e . --no-deps")  # reuse pinned deps
)
```

Key points:
- TRT container base (cuDNN 9 on system path, including `libcudnn_adv.so.9`)
- Pin torch 2.5.1 (CUDA 12.4, NOT 13)
- Pin onnxruntime-gpu 1.20.1 (matches CUDA/cuDNN)
- `numpy<2.0` (transformers compat at the time)
- `--no-deps` for reflex install to avoid re-pulling pinned packages

## Files
- `scripts/modal_bench_path_b.py` — canonical fixed-image comment
- `scripts/modal_verify_strict_providers.py` — strict provider testing
- `scripts/modal_verify_batching_real.py` — real-model batching
- `scripts/modal_verify_trt_with_batch.py` — TRT × batch diagnostic
- `scripts/modal_libero10.py` — LIBERO image with apt-heavy deps
- `src/reflex/runtime/server.py` — strict providers + TRT batch bypass
- `src/reflex/cli.py` — `--strict-providers`, `--no-strict-providers`, `reflex doctor`

---

## Gotcha (2026-04-19): HF_TOKEN secret name ambiguity across scripts

**Symptom.** Ran `modal run scripts/modal_gr00t_monolithic_export.py` after four minutes of image build completed. Function start immediately errored: `Secret 'hf-token' not found in environment 'main'`. The pi0 and pi05 scripts use `modal.Secret.from_name("hf-token")` and work fine; GR00T, copied from the pi0 template, doesn't.

**Root cause.** My Modal account has a secret named `huggingface` (the default name when you create via the HF + Modal integration UI) — not `hf-token`. The pi0/pi05 scripts were written against a different account convention. `modal secret list` shows both names are legitimate patterns across Modal accounts; there's no canonical.

**Initial fallback attempt that didn't work.** Tried a try/except over `modal.Secret.from_name("hf-token")` falling through to `"huggingface"`. This fails because `from_name` is **lazy** — it returns a reference object without resolving, and the actual lookup happens when the Modal function starts. `from_name` never raises at call-time, so the try/except never trips.

**Fix.** Use the account-specific name directly. For this repo's GR00T script: hardcoded `"huggingface"`. For pi0/pi05 (pre-existing): kept `"hf-token"` since those ran fine for their original author. Longer-term: add `HF_TOKEN` to local env and let the `_hf_secret()` helper prefer `modal.Secret.from_dict({"HF_TOKEN": token})` — the local-env path is portable across accounts.

**Lesson.** When templating Modal scripts from another script on a different account, `from_name(...)` arguments are a silent trap. Either grep-replace the secret name, or prefer local-env fallbacks wired via `from_dict`.

---

## Gotcha (2026-04-19): Modal image build transient terminations

**Symptom.** `modal run scripts/modal_gr00t_monolithic_export.py` mid-way through pip install: `RemoteError('Image build for im-0vJfS37Mx4dtpQtNaRVco3 terminated due to external shut-down. Please try again.')`. Happens during torch/onnxruntime wheel downloads.

**Root cause.** Modal-side image build infrastructure is shared; occasional builder preemption drops in-flight builds. Not retryable automatically.

**Fix.** Re-run the command. Second attempt succeeded. No code change warranted given how rare this is (~1 in 10 first-builds of fresh images).

**Lesson.** Don't auto-retry via wrapping logic. The failure is transparent and a human re-run is cheaper than error-recovery complexity. Just note it in docs.

---

## Gotcha (2026-04-19): Docker ENTRYPOINT intercepting subprocess exec

**Symptom.** The CI docker-smoke test wanted to run `docker run reflex-vla python -c "import reflex"` to verify the package imports inside the published image. Instead of importing reflex, the container ran `reflex python -c "import reflex"` — i.e. treated `python` as an arg to `reflex` subcommand, then `reflex` errored "Unknown command: python".

**Root cause.** The `Dockerfile` has `ENTRYPOINT ["reflex"]` so that `docker run image <subcommand>` maps cleanly to `reflex <subcommand>`. This is the right ergonomics for end users but breaks subprocess-level smoke tests that need raw `python`.

**Fix options considered:**
1. `docker run --entrypoint python image -c "..."` — works but brittle; different test frameworks override differently
2. GitHub Actions workflow that invokes `docker build + docker run` natively in CI, bypassing the subprocess abstraction — **chosen**
3. Switch to `CMD` instead of `ENTRYPOINT` — worse ergonomics for end users who get the zero-install `docker run image serve ./export` story

**Implementation.** `.github/workflows/docker-smoke.yml` runs the build + a few `reflex <subcmd>` smoke tests in CI directly. The Modal-based `scripts/modal_docker_smoke.py` was dropped — could never escape the ENTRYPOINT intercept.

**Lesson.** If a Docker image has an ENTRYPOINT, don't try to test it via local subprocess exec; either `--entrypoint` override or use CI-native docker tooling.

---

## Gotcha (2026-04-19): Modal Volume sync lag on first parity after export

**Symptom.** After `modal run ...monolithic_export.py`, immediately running `modal run ...monolithic_export.py --parity` occasionally sees the ONNX external-data file truncated or missing. Second run of parity succeeds.

**Root cause.** `modal.Volume.commit()` completes and the Python function returns, but the filesystem-level sync to Modal's shared volume backend has a small lag (~10–30s). The next container that mounts the volume may see stale or partial state.

**Fix (pi0/pi05 scripts).** Sleep 60s before reading the freshly-exported ONNX in the parity function if it's invoked right after export. In practice we usually invoke parity as a separate `modal run`, which gives the volume time to sync naturally — only hit this when hotlooping export + parity in the same `local_entrypoint`.

**Lesson.** If you're testing exports by re-reading the volume right after a write, add a small sleep or check for file existence + correct size before running the parity logic. Modal Volumes are eventually-consistent, not strongly consistent.

---

## Gotcha (2026-04-19): ROS2 image construction — five attempts, one that works

**Context.** Launch gate #5 required a live `rclpy` test: real ROS2 humble + real rclpy (not mocked) verifying `reflex ros2-serve` can subscribe/publish on Modal.

**Attempts and failures:**

1. **`modal.Image.debian_slim().pip_install("rclpy")`** — `rclpy` is NOT on PyPI. Fails at pip install.
2. **`add_python="3.10"` on a ROS2 base image** — broke rclpy because Python path lookup uses setup.sh which goes missing when Modal injects its own Python.
3. **Without `add_python`, bare ROS2 base image** — pip 22 on older base image can't handle `--break-system-packages`, needed for installing reflex alongside ROS's system-packaged Python libs.
4. **Upgraded pip via pip install --upgrade pip** — distutils-installed ROS packages couldn't be replaced/upgraded; pip errored on dependency resolution.
5. **Added `--ignore-installed`** — numpy ABI mismatch (rclpy compiled against numpy 1.21 at ROS build time, but reflex-vla pulled numpy 2.4 via transformers).

**The working recipe (v5):** `ubuntu:22.04` + apt-install `ros-humble-ros-base` + `DEBIAN_FRONTEND=noninteractive` (for tzdata prompt) + pip constraint `numpy>=1.24,<2.0` + source `/opt/ros/humble/setup.bash` before invoking reflex.

```python
image = (
    modal.Image.from_registry("ubuntu:22.04")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install(
        "curl", "gnupg", "lsb-release",
        "ros-humble-ros-base",  # requires the ROS2 apt source to be added first
        "python3-pip", "git",
    )
    .pip_install("reflex-vla[serve,onnx]", "numpy>=1.24,<2.0")
)
```

**Lesson.** For anything that mixes apt-installed Python packages (like ROS) with pip packages, pin numpy to match the apt ABI. Don't try to upgrade apt-managed packages through pip.

**File.** `tests/test_ros2_bridge_live.py` — the landed test. Reads as "boots up ROS2 humble in a Modal ubuntu:22.04 container, publishes an Image to `/camera/image_raw`, expects action chunks on `/reflex/actions` within 10s."

---

## Gotcha (2026-04-19): Module stubbing for cross-model compat

**Symptom.** pi0/pi05 scripts import `lerobot.policies.pi0.modeling_pi0`; lerobot 0.5.1 also has `lerobot.policies.groot.groot_n1` and `lerobot.policies.groot.modeling_groot`. On Python 3.13 (used in some Modal images) the groot module has import-time compat breaks → ImportError → cascade failures before we ever touch GR00T code.

**Fix.** At the top of each pi0/pi05 Modal script:

```python
for _mod in ("lerobot.policies.groot.groot_n1",
             "lerobot.policies.groot.modeling_groot"):
    stub = types.ModuleType(_mod)
    stub.GrootPolicy = None
    stub.GR00TN15 = None
    sys.modules[_mod] = stub
```

This preemptively registers empty modules under those names so subsequent `from lerobot.policies.groot...` imports inside transitive dependencies succeed (returning None for the class, which they don't dereference during pi0 code paths).

**Lesson.** If you're tracing a complex package where one submodule has transitive import breaks you don't care about, stub it in `sys.modules` before importing the rest. Don't try to fix the broken submodule unless you actually need it.

---

## Gotcha (2026-04-19): Modal image cache serves stale code silently

**Symptom.** Customer dogfood v5 ran against a container that still showed v4's bug output — the tokenizer fallback warning was the old wording, the `/act` response still lacked the field I'd added. Four code fixes were pushed, the `pip install @ git+https://github.com/rylinjames/reflex-vla` pulls from `main`, yet the container behavior didn't change.

**Root cause.** Modal's `.run_commands(...)` caches the resulting image layer keyed by the **command string**, not by what the command actually fetches. A command like `pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'` is a static string — Modal sees it and reuses the cached layer from a prior run, even if `main` has advanced. The pip install inside the container never runs a second time. You're testing the version of your code that was HEAD when you FIRST built the image. Forever.

**The subtle part.** Build logs say nothing — you just see "✓ Created mount ..." and the Modal function boots. No "using cached image" notice. The stale test looks identical to a fresh test.

**Fix (the pattern for any Modal dogfood/verification script).** Inject the repo HEAD SHA into the command string so each commit produces a new cache key:

```python
import subprocess, os, modal

def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"  # fallback if not in a git repo

_HEAD = _repo_head_sha()

image = (
    modal.Image.from_registry(...)
    .run_commands(
        f"pip install 'reflex-vla[serve,gpu,monolithic] "
        f"@ git+https://github.com/rylinjames/reflex-vla@{_HEAD}'",
    )
)
```

The `@{_HEAD}` on the pip URL ALSO pins pip to install that exact commit (not latest `main`), so the test is deterministic against the code that triggered it.

**Safe-to-cache vs must-SHA-pin Modal scripts.** Many of our Modal scripts already work fine without this because their `run_commands` reference local code (`.add_local_dir("src/reflex", ...)`) — Modal hash-invalidates those when the local files change. The cache-key footgun only hits scripts that install via a remote git URL with a static ref (`main`, `master`, a branch name, or no ref at all). Audit test: any Modal script that installs from `git+https://...` without `@<sha>` is a candidate. The currently-known offenders are any dogfood-style script; the monolithic export scripts use `add_local_dir` and are safe.

**Lesson.** Modal's image cache is content-addressed by command STRING, not by what the command does. Remote git installs sidestep that assumption. Always SHA-pin for any verification script that's meant to test newly-committed code.

**Discovered:** customer dogfood v5 2026-04-19. Cost: one wasted ~12 min Modal run + a false-positive "fixes don't work" signal that almost triggered a second round of unnecessary debugging.

---

## Gotcha (2026-04-19): `pip install <pkg>` does not guarantee `import <pkg>` works

**Symptom.** LIBERO run 5: `pip list` shows `libero==0.1.0` installed, `/opt/LIBERO/libero/` source tree exists, BUT `import libero` fails with `ModuleNotFoundError: No module named 'libero'`. Reinstalling from source (`pip install /opt/LIBERO --no-deps`) succeeds cleanly (exit 0) — import STILL fails right after.

**Root cause.** LIBERO's `setup.py` is non-standard: `install_requires=[]` and an unusual package-discovery layout. pip writes `libero-0.1.0.dist-info` metadata and thinks the install succeeded, but the actual `libero/` package directory isn't placed on Python's module search path. Python can only import via the source tree at `/opt/LIBERO/libero/`.

**Fix.** Add the source root to `PYTHONPATH` at the image level:
```python
image = (...)
    .env({
        # libero installs to metadata only; source lives at /opt/LIBERO/libero/
        # — must be on PYTHONPATH for `from libero.libero import ...` to work.
        "PYTHONPATH": "/opt/LIBERO",
    })
```

PYTHONPATH applies to every subprocess (main function, vla-eval workers, smoke tests), so the import works everywhere.

**Lesson.** When a pip-installed package fails `import` despite "installation succeeded", check if the package uses a non-standard layout. A `pip show <pkg>` that lists a `Location` of `/usr/local/lib/python*/site-packages` but that directory has only `<pkg>-*.dist-info` and no `<pkg>/` subdirectory is the signature. Fix by adding the source root to PYTHONPATH.

---

## Gotcha (2026-04-19): `python` in subprocess != `sys.executable` in containers

**Symptom.** LIBERO runs 3–4: the main Modal function had `libero` importable (adapter server ran fine), but a subprocess spawned with `subprocess.run(["python", "-c", "import libero"])` raised `ModuleNotFoundError`. The `python` binary resolved to a DIFFERENT interpreter than the one running the main function.

**Root cause.** Modal containers often have multiple python interpreters on PATH (e.g. `/usr/bin/python` = system Python 3.x; `/usr/local/bin/python` = Modal-injected Python 3.11; virtualenv-local `python`). A bare `"python"` in `subprocess.run([...])` resolves via shell PATH, which may pick a different interpreter from the one your function is executing in. Packages installed in the "right" python's site-packages are invisible to the wrong python.

**Fix.** Always use `sys.executable` for subprocess python calls:
```python
import sys, subprocess
result = subprocess.run(
    [sys.executable, "-c", "import libero; print(libero.__file__)"],
    ...
)
```

This guarantees the subprocess uses the exact same interpreter as the caller.

**Also affects:** `pip` invocations from subprocess should be `[sys.executable, "-m", "pip", "install", ...]` not `["pip", "install", ...]` for the same reason.

**Lesson.** Never trust a bare `"python"` or `"pip"` in subprocess calls inside Modal containers. Use `sys.executable` + `-m <module>` form. Silent interpreter divergence is one of the most confusing classes of "I just installed it, why is it not importable" bug.

---

## Gotcha (2026-04-19): Python version vs package extras — lerobot 0.5.1 needs 3.12, LIBERO stack wants 3.11

**Symptom.** LIBERO image on `debian_slim(python_version="3.11")` — because LIBERO + robosuite 1.4.1 + MuJoCo stack is proven on 3.11. After flipping `reflex export` default to monolithic (2026-04-19), ran `pip install -e '.[monolithic]'` in the image. pip errored:
```
ERROR: Ignored the following versions that require a different python version:
  0.5.0 Requires-Python >=3.12;
  0.5.1 Requires-Python >=3.12
ERROR: Could not find a version that satisfies the requirement
  lerobot==0.5.1; extra == "monolithic"
```

**Root cause.** The `[monolithic]` extra pins `lerobot==0.5.1` exactly (required by `scripts/modal_*_monolithic_export.py` family). lerobot 0.5.x requires Python >=3.12. The LIBERO image is on 3.11. Can't satisfy both.

**Fix (for this case).** The LIBERO eval uses `REFLEX_NATIVE=1` which routes to `SmolVLANativeServer` — the PyTorch path, not the monolithic ONNX path. The export step still runs (to produce the config + normalizer safetensors that vla_eval reads), but it doesn't NEED the monolithic ONNX files. Pass `--decomposed` to `reflex export`:
```python
subprocess.run([
    "reflex", "export", "lerobot/smolvla_libero",
    "--target", "desktop", "--output", export_dir,
    "--decomposed",  # no [monolithic] extras needed, fine for REFLEX_NATIVE=1
], ...)
```

This sidesteps the `[monolithic]` install entirely.

**Larger lesson.** When Reflex's CLI defaults change (e.g. `--monolithic` becoming the default), Modal images that were stable suddenly break because their install command now pulls a newer-Python-requirement dep. Audit rule: if your image uses `python_version<current` and installs reflex-vla, either (a) match reflex-vla's supported Python range, or (b) pass the CLI flag that avoids the new-requiring extras.

**Cross-reference.** `scripts/modal_libero_monolithic.py` uses `--decomposed` + a comment explaining why.

