# Reflex

**The deployment layer for VLAs** — take a Vision-Language-Action model off the training cluster and onto a robot.

**Verified parity.** Reflex's native export path for SmolVLA matches the reference PyTorch policy to **cos = 1.0000** end-to-end (first-action, shared noise). Per-stage: vision encoder cos=1.0000, text embedder cos=1.0000, state projection cos=1.0000, single self-attn layer cos=1.0000 to 1e-5. Reproducer: `REFLEX_NATIVE=1 python scripts/local_full_diff.py` at commit `0616265`. The full claim ledger (verified / unverified / unmeasured) lives in [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

Cross-framework ONNX export, edge-first serving, composable runtime wedges (safety, adaptive denoising, cloud-edge split, pre-flight validation). One CLI, seven verbs.

**Want a deeper walkthrough?** See [docs/getting_started.md](docs/getting_started.md) for a 30-min guide covering safety configs, fleet-mode batching, deadline enforcement, and common troubleshooting.

**Something not working?** Run `reflex doctor` first — diagnoses install + GPU issues in one screen.

## Quickstart — 3 commands from zero to actions

```bash
# 1. Install (v0.1 — install from GitHub until we publish to PyPI)
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'
# Or CPU-only: pip install 'reflex-vla[serve,onnx] @ git+https://github.com/rylinjames/reflex-vla'
# Note: GPU install requires the FULL cuDNN 9 system library (incl. libcudnn_adv.so.9),
# not just the pip wheel. Easiest path is NVIDIA's container:
#   docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3
#   (then pip install reflex-vla[serve,gpu] inside)
# `reflex serve` errors loudly if cuDNN can't load — no silent CPU fallback.

# 2. Export any supported VLA to ONNX (auto-detects model type)
# SmolVLA fits on Orin Nano 8GB; pi0 (~12GB monolithic) needs Orin 16GB+ or desktop GPU.
reflex export lerobot/smolvla_base --target desktop --output ./smol

# 3. Serve it — POST /act to get 50-step action chunks
reflex serve ./smol --port 8000
```

In another terminal:

```bash
curl -X POST http://localhost:8000/act -H 'content-type: application/json' \
  -d '{"instruction":"pick up the red cup","state":[0.1,0.2,0.3,0.4,0.5,0.6]}'
```

```json
{
  "actions": [[...], [...], ...],   // 50 × action_dim chunk
  "num_actions": 50,
  "latency_ms": 11.9,                // smolvla on A10G, full 10-step denoise
  "denoising_steps": 10,
  "inference_mode": "onnx_trt_fp16"  // automatic — no engine flags needed
}
```

That's it. `reflex` auto-detects whether you gave it SmolVLA / pi0 / pi0.5 / GR00T and dispatches to the right exporter. No framework-specific flags.

### Docker — zero-install serve

```bash
# Pull the published image (x86_64 CUDA runtime)
docker pull ghcr.io/rylinjames/reflex-vla:latest

# Mount your exports dir at /exports, expose port 8000
docker run --gpus all \
  -v $(pwd)/p0:/exports \
  -p 8000:8000 \
  ghcr.io/rylinjames/reflex-vla:latest
```

The container's default command is `reflex serve /exports --host 0.0.0.0 --port 8000`. Override with any `reflex` subcommand: `docker run ... ghcr.io/rylinjames/reflex-vla:latest export <hf_id>` etc. Jetson arm64 images land in v0.3 (contact us if you need one sooner).

### ROS2 — `reflex ros2-serve`

Wraps the inference loop as a ROS2 node. Subscribes to `sensor_msgs/Image`, `sensor_msgs/JointState`, and `std_msgs/String`; publishes action chunks as `std_msgs/Float32MultiArray` at a configurable rate.

```bash
# rclpy is NOT pip-installable. Install ROS2 via apt or robostack first:
source /opt/ros/humble/setup.bash   # or iron / jazzy

reflex ros2-serve ./my_export \
  --image-topic /camera/image_raw \
  --state-topic /joint_states \
  --task-topic  /reflex/task \
  --action-topic /reflex/actions \
  --rate-hz 20
```

Inference respects `--safety-config` (same limits file as HTTP serve).

When `onnxruntime-gpu` ships with the TensorRT execution provider (it does in v1.20+), `reflex serve` uses TRT FP16 automatically and caches the engine in `<export_dir>/.trt_cache` so subsequent server starts skip the engine-build cost. The first `reflex serve` takes ~30-90s to warm up; restart is ~1-2s.

## Validation — round-trip ONNX vs PyTorch parity

After exporting, run `reflex validate` to confirm the ONNX graph matches the PyTorch reference within numerical tolerance:

```bash
# After exporting, validate that ONNX parity holds vs the reference:
reflex validate ./p0 --model lerobot/pi0_base --threshold 1e-4
```

Sample passing output (abbreviated):

```
Per-fixture results
fixture_idx  max_abs_diff  mean_abs_diff  passed
0            3.21e-06      8.40e-07       PASS
1            2.98e-06      7.92e-07       PASS
...
Summary
max_abs_diff_across_all  3.21e-06
passed                   PASS
```

Exit codes: `0` pass, `1` fail (any fixture above threshold), `2` error (missing ONNX, bad config). Pipe `--output-json` for CI consumption, or run `reflex validate --init-ci` to scaffold a GitHub Actions workflow at `.github/workflows/reflex-validate.yml`.

## Composable wedges

Every wedge is a flag on `reflex serve`:

```bash
reflex serve ./p0 \
  --safety-config ./robot_limits.json \   # joint-limit clamping + EU AI Act audit log
  --adaptive-steps \                       # stop denoise loop early when velocity converges
  --deadline-ms 33 \                       # return last-known-good action if over budget
  --cloud-fallback http://cloud:8000      # edge-first with cloud backup
```

The response JSON surfaces telemetry from each enabled wedge so you can see what's actually happening (`safety_violations`, `deadline_exceeded`, `adaptive_enabled`, etc.).

## Supported VLA models

| Model | HF ID | Params | Export status |
|---|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | 450M | ONNX + validated (max_diff=3.3e-06) |
| pi0 | `lerobot/pi0_base` | 3.5B | ONNX + validated (max_diff=6.0e-08) |
| pi0.5 | `lerobot/pi05_base` | 3.62B | ONNX + AdaRMSNorm (max_diff=2.5e-06) |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | 3.29B | ONNX + DiT/AdaLN (max_diff=3.8e-06) |
| OpenVLA | `openvla/openvla-7b` | 7.5B | `optimum-cli export onnx` + `reflex.postprocess.openvla.decode_actions` |

`reflex models` lists current support at any time. OpenVLA is a vanilla Llama-2-7B VLM — there's no custom action expert to reconstruct, so we defer to the standard HuggingFace export path and ship only the bin-to-continuous postprocess helper.

## Hardware targets

| Target | Hardware | Memory | Precision |
|---|---|---|---|
| `orin-nano` | Jetson Orin Nano | 8 GB | fp16 |
| `orin` | Jetson Orin | 32 GB | fp16 |
| `orin-64` | Jetson Orin 64 | 64 GB | fp16 |
| `thor` | Jetson Thor | 128 GB | fp8 |
| `desktop` | RTX / A100 | 40 GB | fp16 |

**Memory fit (monolithic ONNX on disk, FP32):** SmolVLA 1.6GB, pi0 12.5GB, pi0.5 ~14GB (v0.3), GR00T ~5GB (v0.3). SmolVLA fits comfortably on Orin Nano 8GB; **pi0 realistically needs Orin 16GB+ or a desktop NVIDIA GPU** — the 12.5GB monolithic ONNX cannot load on the 8GB Orin Nano even in FP16 (~6GB weights plus activations + OS). FP16 engine rebuild + Orin Nano fit work is tracked for v0.3.

`reflex targets` lists current profiles.

## The 7 wedges (+ 1 planned)

```
reflex export   # checkpoint → ONNX + TensorRT
reflex serve    # HTTP inference server, composable wedges
reflex guard    # URDF-derived safety limits + EU AI Act logging
reflex turbo    # adaptive denoising (stops early on convergence)
reflex split    # cloud-edge orchestration with fallback modes
reflex adapt    # cross-embodiment action-space mapping
reflex check    # 5 pre-deployment checks (loadable, size, structure, dtype, nan_inf)
reflex distill  # [planned] flow-matching step distillation (10 → 2)
```

Each wedge works standalone for scripting, and every wedge that belongs in the inference path is composable through `reflex serve` flags.

## What Reflex is and isn't

**Is:** the deployment layer between a trained VLA and a real robot. Cross-framework export (SmolVLA + pi0 verified at cos=+1.0000000, pi0.5/GR00T in v0.3), composable runtime (serve + safety + turbo + split), edge-first design targeting Jetson + desktop NVIDIA GPUs.

**Isn't:** a training framework (PyTorch/JAX own that) or a cloud inference provider (vLLM/Baseten own that). Reflex's moat is the deployment toolchain: cross-framework ONNX with verified numerical parity, composable safety wedges, ROS2 + Docker + HTTP serving, and a deterministic export receipt (`VERIFICATION.md`) your QA team can audit.

## Verified parity (the only load-bearing numbers)

Both SmolVLA and pi0 match the reference PyTorch policy to machine precision on shared seeded inputs:

| Model | Path | first-action cos | first-action max_abs | full-chunk cos | full-chunk max_abs |
|---|---|---|---|---|---|
| SmolVLA | PyTorch native (DecomposedRMSNorm swap) | 1.0000 | 0.000 | — | — |
| SmolVLA | Monolithic ONNX (num_steps=1) | **+1.0000000** | **1.55e-06** | **+1.0000000** | **3.34e-06** |
| pi0 | PyTorch native wrapper vs raw sample_actions | **1.0000000000** | **0.000e+00 (bit-exact)** | 1.0 | 0.0 |
| pi0 | Monolithic ONNX (num_steps=1) | **+1.0000000** | **1.43e-06** | **+1.0000000** | **2.98e-06** |

Full ledger of verified / unverified / unmeasured numbers: [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

**Latency numbers are intentionally not in the README yet** — earlier TRT FP16 tables were measured on a now-abandoned decomposed-ONNX path. Desktop GPU + Jetson latency re-measurement is tracked for v0.3. `reflex bench <export_dir>` reproduces on any hardware.

Reproduce on your own GPU with one command:

```bash
reflex bench ./pi0 --iterations 100
```

### Multi-robot batching (`reflex serve --max-batch N`)

Continuous batching on the HTTP layer: each `/act` request enters an asyncio queue; the server flushes the queue every `--batch-timeout-ms` (default 5ms) into one batched ONNX inference. Earlier measurements on the decomposed-ONNX path showed 2.3-2.9× throughput scaling at batch sizes 4-16; those numbers are being re-measured on the monolithic path for v0.3.

## License

Apache 2.0

## Status

v0.1 — active development. Install, kick the tires, open issues loudly. We're looking for the first 20 robotics teams actually deploying this; your feedback shapes v0.2.
