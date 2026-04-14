# Reflex

Deploy any VLA (Vision-Language-Action) model to any edge hardware. One command.

```bash
pip install -e '.[dev,serve]'
reflex export lerobot/smolvla_base --target orin-nano
reflex serve ./reflex_export/ --port 8000
curl -X POST http://localhost:8000/act -d '{"instruction":"pick up the cup","state":[...]}'
```

## Supported VLA models

| Model | HF ID | Params | Status |
|---|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | 450M | ONNX + validated (max_diff=3.8e-06) |
| pi0 | `lerobot/pi0_base` | 3.5B | ONNX + validated (max_diff=3.7e-08) |
| pi0.5 | `lerobot/pi05_base` | 3.62B | ONNX with AdaRMSNorm (max_diff=2.4e-06) |
| GR00T | `nvidia/GR00T-N1.6` | 3.3B | planned |
| OpenVLA | `openvla/openvla-7b` | 7.5B | planned |

Run `reflex models` to see current support.

## Hardware targets

| Target | Name | Memory | FP8 | Precision |
|---|---|---|---|---|
| `orin-nano` | Jetson Orin Nano | 8 GB | no | fp16 |
| `orin` | Jetson Orin | 32 GB | no | fp16 |
| `orin-64` | Jetson Orin 64 | 64 GB | no | fp16 |
| `thor` | Jetson Thor | 128 GB | yes | fp8 |
| `desktop` | RTX/A100 | 40 GB | yes | fp16 |

Run `reflex targets` to see hardware options.

## 7-wedge product

```
reflex export   # checkpoint → ONNX + TensorRT
reflex serve    # HTTP inference server (POST /act)
reflex guard    # safety constraints (joint limits, EU AI Act logging)
reflex turbo    # denoising loop optimization (adaptive step count)
reflex split    # cloud-edge orchestration (fallback on latency spikes)
reflex adapt    # cross-embodiment action mapping
reflex check    # pre-deployment validation
```

## Verification

Full E2E (export → serve → POST /act) tested on Modal A100, ONNX Runtime CPU provider, 10 denoising steps:

| Model | Expert params | Export | ONNX max_diff | Mean latency | Hz |
|---|---|---|---|---|---|
| SmolVLA | 100M | 47s | 3.3e-06 | 246ms | 4.1 |
| pi0 | 314M | 97s | 6.0e-08 | 588ms | 1.7 |
| pi0.5 | 427M | 95s | 2.5e-06 | 704ms | 1.4 |

GPU provider and TensorRT builds (on Jetson) expected to push these 5-10x faster.
Validation diff is measured against the original PyTorch forward pass.
