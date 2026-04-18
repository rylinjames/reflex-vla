# Draft: r/robotics post

## Title

**Open-source tool to export & serve SmolVLA / pi0 on Jetson or desktop (cos=1.0 verified)**

## Body (markdown)

Hi r/robotics —

I built [Reflex](https://github.com/rylinjames/reflex-vla), an open-source CLI for taking a trained Vision-Language-Action model from a HuggingFace checkpoint to a working inference server you can hit from a robot.

**Verified today**: two of the most-used VLAs match the reference PyTorch policy to machine precision on shared seeded inputs:

| Model | Native parity | Monolithic ONNX parity |
|---|---|---|
| SmolVLA (`lerobot/smolvla_base`) | cos=1.0000 (DecomposedRMSNorm swap) | **cos=+1.0000000, max_abs=1.55e-06** |
| pi0 (`lerobot/pi0_base`) | cos=1.0 bit-exact (wrapper vs raw) | **cos=+1.0000000, max_abs=1.43e-06** |

Three commands from zero to serving:

```bash
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'
reflex export lerobot/pi0_base --output ./p0
reflex serve ./p0 --port 8000
```

Then `POST /act` returns flow-matching action chunks. Composable wedges let you build a real production pipeline without writing your own runtime:

- `--safety-config` — joint limits + NaN/Inf kill-switch + EU AI Act audit logging
- `--adaptive-steps` — early-stop denoising
- `--deadline-ms` — WCET fallback
- `--max-batch` — fleet serving with HTTP-layer continuous batching

**Plus:**
- **Docker image** at `ghcr.io/rylinjames/reflex-vla:latest` (x86 CUDA) — zero install
- **ROS2 bridge** (`reflex ros2-serve`) — subs `sensor_msgs/Image` + `sensor_msgs/JointState` + `std_msgs/String`, pubs action chunks as `Float32MultiArray` at configurable Hz
- **`VERIFICATION.md`** — every export directory gets an auto-generated manifest (sha256 of every file, ONNX opset, parity results after `reflex validate`) that your QA team can audit

**Honest disclaimers:**
- Alpha, single maintainer, Apache 2.0
- Jetson Orin Nano numbers not yet published — CloudJetson waitlisted, Orin Nano dev kit not on hand. Launch latency data is from Modal A10G; real Jetson numbers land when someone runs `reflex bench` on a dev kit (happy to credit + thank-you gift)
- pi0.5 (AdaRMSNorm) and GR00T (DiT + AdaLN) parity are v0.3 items — planned but not shipped
- Earlier TRT FP16 latency tables were on a decomposed-ONNX path that's now abandoned; latency re-measurement on the monolithic path is a v0.3 item

What I'm specifically asking for:

1. **Testers** — install, point at your robot, open issues. <24h response commitment.
2. **Jetson Orin Nano benchmark contributor** — 30 min on a dev kit + you get real edge numbers published with your credit.
3. **Wedge feedback** — does `--safety-config / --adaptive-steps / --deadline-ms / --max-batch` match how you actually want to deploy?

Repo: https://github.com/rylinjames/reflex-vla
Verified numbers ledger: [measured_numbers.md](https://github.com/rylinjames/reflex-vla/blob/main/reflex_context/measured_numbers.md)

Happy to answer questions in the thread.

---

## Posting notes

- r/robotics values implementation depth + honesty over hype
- Lead with verified cos numbers — that's the differentiator
- Mention Apache 2.0 early
- Be ready for "why not just use [X]" — answer with specific code links
- Respond to every comment within 24h post-launch
