# Reflex

**The deployment layer for VLAs** — take a Vision-Language-Action model off the training cluster and onto a robot.

Cross-framework ONNX export, edge-first serving, composable runtime wedges (safety, adaptive denoising, cloud-edge split, pre-flight validation). One CLI, seven verbs.

## Quickstart — 3 commands from zero to actions

```bash
# 1. Install (v0.1 — install from GitHub until we publish to PyPI)
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'
# Or CPU-only: pip install 'reflex-vla[serve] @ git+https://github.com/rylinjames/reflex-vla'

# 2. Export any supported VLA to ONNX (auto-detects model type)
reflex export lerobot/pi0_base --target orin-nano --output ./p0

# 3. Serve it — POST /act to get 50-step action chunks
reflex serve ./p0 --port 8000
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
  "latency_ms": 47.3,
  "denoising_steps": 10,
  "inference_mode": "onnx_gpu"
}
```

That's it. `reflex` auto-detects whether you gave it SmolVLA / pi0 / pi0.5 / GR00T and dispatches to the right exporter. No framework-specific flags.

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

All 4 flow-matching VLAs fit on a `$500` Orin Nano 8GB in FP16 with 2× overhead (verified empirically) — no need to jump to Thor for most deployments.

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

**Is:** the deployment layer between a trained VLA and a real robot. Cross-framework export (4 VLA families covered), composable runtime (serve + safety + turbo + split), Jetson-first.

**Isn't:** a training framework (PyTorch/JAX own that), a cloud inference provider (vLLM/Baseten own that), or a "fastest raw forward pass on A100" tool (`torch.compile` dominates that and it's free). Reflex's moat is the edge: Jetson TRT engines, deterministic deploy graph, and the wedge composition for production robot deployments.

## Verification

End-to-end (`reflex export → reflex serve → POST /act`) tested on Modal A100 with all 4 flow-matching VLAs. Full benchmark table and honest cloud-GPU vs `torch.compile` comparison in [the roadmap repo](https://github.com/rylinjames/vla_to_hardware_roadmap/blob/main/phase_1_vla_software/deployment_export/build_candidates.md).

Jetson Orin Nano benchmarks are the headline numbers — landing in a follow-up once we've validated on real hardware.

## License

Apache 2.0

## Status

v0.1 — active development. Install, kick the tires, open issues loudly. We're looking for the first 20 robotics teams actually deploying this; your feedback shapes v0.2.
