# local_vs_modal — the cost-per-iteration difference

## TL;DR

For correctness debugging on Reflex VLA exporters, **local iteration is ~100x cheaper and ~10-20x faster** than Modal. Use Modal only for benchmarks that require the fixed hardware profile (A10G for Jetson proxying, A100 for throughput claims).

## The numbers

### Modal iteration (current default)
- **Time per iteration**: 5-10 minutes end-to-end
  - Image build (cold cache): 3-5 min if deps change (lerobot, robosuite==1.4.1, bddl==1.0.1, osmesa, mujoco, etc.)
  - Image pull + warmup: 30-90s
  - Model download (SmolVLA base 907MB; pi0 3.5GB; GR00T 6.6GB): 10-60s
  - Actual exporter/diff run: 30s-3 min
  - Teardown + log drain: 30s
- **$ per iteration**: $0.30-0.60
  - A10G: ~$1.10/hr → $0.18-0.37 per 10-min run
  - A100-40GB: ~$3.00-4.00/hr → $0.30-0.60 per 10-min run
  - Each rebuild of the image when deps change adds ~$0.05-0.15 amortized
- **Feedback loop lag**: ~5-10 min wall-clock before you see stdout

### Local iteration (after one-time `pip install`)
- **Time per iteration**: 30s-2 min
  - No image build
  - Model already in HF cache after first run
  - Actual exporter/diff run: 30s-2 min (CPU is enough for single-layer diffs and stage diffs)
- **$ per iteration**: $0 (marginal)
- **Feedback loop lag**: <30s for diagnostic scripts

### Ratio
~**100x cheaper** ($0 vs $0.30-0.60), **~10-20x faster** wall-clock (60s vs 600s).

## The one-time setup

For local-first iteration, prime the environment once:

```bash
# Create a .venv for reflex-vla local work
cd ~/Desktop/building\ projects/reflex-vla
python3 -m venv .venv
source .venv/bin/activate

# Core deps
pip install -e .

# The three that unlock local correctness debugging
pip install lerobot num2words onnxruntime

# If you need the real VLM path (vision encoder local):
pip install 'transformers>=4.40,<5.0' safetensors pillow
```

Notes on those three deps:
- `lerobot` — gives you `SmolVLAPolicy.from_pretrained(...)` so the reference PyTorch forward is one import away. Install from PyPI (`pip install lerobot`); the `git+https://github.com/huggingface/lerobot.git` path has a non-standard install structure and `import lerobot` fails in about half the attempts (see Apr-14 session, `scripts/modal_libero10.py` history).
- `num2words` — required by the SmolVLM2 processor when tokenizing a task string. Without it, any call to `AutoProcessor(...)(text=[task], ...)` raises `ModuleNotFoundError`.
- `onnxruntime` (NOT `onnxruntime-gpu`) — CPU-only is fine for diagnostic scripts. You only need GPU for throughput benchmarks. Installing the CPU wheel avoids the CUDA 12 vs 13 library mismatch rabbit hole entirely.

Caveats:
- Local ORT runs CPU only — do NOT use local runs to benchmark latency.
- Mac Apple Silicon runs ORT CoreML EP which will give different numerics than x86 CPU ORT for FP16 ops; stay FP32 for local diff.
- LIBERO sim does not run on Mac (needs osmesa + mujoco GL + robosuite); LIBERO runs stay on Modal.

## The Apr-17 session that made this lesson concrete

The Apr-17 session (`ced2c4f1` transcript, approx lines 11524-11574) ran 6+ hours on Modal chasing a cos_sim=0.08 regression through 12 candidate bugs. Rough ledger:

- ~15 Modal runs across stage-diff, libero10, pytorch-vs-onnx, export
- Modal app list on 2026-04-17 shows 7 named runs:
  - `reflex-stage-diff` x 4 (6+6+4+2 = 18 min wall, ~$4-5)
  - `reflex-pytorch-vs-onnx` x 2 (3+4 = 7 min wall, ~$1.50-2)
  - `reflex-libero10` x 1 (12 min wall, user-killed, ~$2-3)
- Plus earlier iteration spend same day: ~$3-4
- **Total**: ~$5-10 Modal, 6h wall clock, still ended the session with cos_sim = -0.24 unresolved

Pivot that landed (transcript line 11524):
> "Pausing the kv_mask fix. We can, and we probably should have, copied lerobot code wholesale from the start. Three realistic paths, ordered by correctness-per-engineering-hour: (1) Wrap + export the REAL classes directly. (2) Copy lerobot's modeling files into `reflex/exporters/smolvla_native.py` verbatim. (3) Skip ONNX entirely, load `SmolVLAPolicy` and serve it via FastAPI in pure PyTorch."

The "~100x cheaper" note is this session's explicit takeaway (line 11574).

## When Modal IS the right tool

Do not read this as "Modal is a mistake." Modal is the right tool for:

- **Benchmarks** (`modal_bench_path_b.py`, `modal_bench_trt_fp16.py`, `modal_verify_bench_all.py`) — numbers must come from a fixed hardware profile (A10G = Ampere SM_86, Jetson Orin Nano = Ampere Tegra SM_87, same compute family) so results generalize. TRT FP16 engines cannot be built on Mac.
- **LIBERO** (`modal_libero10.py`) — robosuite + osmesa + LIBERO env needs a Linux box + decent GPU. Not Mac compatible.
- **Install-path gating** (`modal_verify_install_path.py`) — fresh-box E2E test that `pip install 'reflex-vla[serve,gpu]' @ git+...` actually works has to run on a box we don't control.
- **Scenarios that need CUDA EP / TensorRT EP** — provider-specific bugs (silent CPU fallback, cuDNN mismatches) surface only on CUDA 12 + cuDNN 9 Linux boxes. Mac has none of that pathology.

## The right rule

> **For correctness bugs: local first. For performance and integration: Modal.**

Practical decision tree:
1. Is the failure "numbers don't match reference"? → Local diagnostic script.
2. Is the failure "wrong hardware latency / throughput"? → Modal.
3. Is the failure "sim env crashes / rendering"? → Modal.
4. Is the failure "fresh install doesn't work"? → Modal (you want a clean box).
5. Is the failure "works on my mac but not on Modal"? → Modal to reproduce, then local to iterate the fix.

## File locations

- Local diagnostic scripts live at: `scripts/local_*.py`
  - `scripts/local_full_diff.py`
  - `scripts/local_stage_diff.py`
  - `scripts/local_single_layer_diff.py`
  - `scripts/local_expert_diff.py`
- Modal scripts live at: `scripts/modal_*.py` (32 files as of Apr-17)
- See `reflex_context/04_iteration_lessons/diagnostic_ladder.md` for the 4-level diagnostic progression these local scripts implement.

## Forgotten discipline flagged by this lesson

Pre-Apr-17 behavior was "every question runs on Modal." That's wrong. The lesson: **a local Python interpreter with `lerobot` + `onnxruntime` installed is enough to diff a PyTorch forward against an ONNX forward.** No cloud necessary. Every session thereafter should prefer local for correctness and restrict Modal to the four cases above.
