# Draft: Show HN — Reflex (VLA deployment toolchain)

## Title (~80 chars)

**Show HN: Reflex — ONNX export + serve for Vision-Language-Action models (cos=1.0 verified)**

## Body (text post)

Hi HN —

I built Reflex because the path from "we have a trained Vision-Language-Action model" to "it runs on a real robot" is painful. Every VLA team writes their own export pipeline. Most break silently under FP16 / TRT / Jetson constraints.

**What's verified today:** I export three of the most-used open VLAs — SmolVLA (HuggingFace LeRobot), pi0, and pi0.5 (Physical Intelligence, via lerobot) — as monolithic ONNX covering the full 10-step flow-matching denoise. Measured on shared seeded inputs against PyTorch eager:

- **SmolVLA num_steps=10 ONNX**: cos = +1.0000000, max_abs = 5.96e-07 (first-action) vs `sample_actions(num_steps=10)`. **Machine precision.**
- **pi0 num_steps=10 ONNX**: cos = +1.0000000, max_abs = 2.09e-07 (first-action) vs `sample_actions(num_steps=10)`. **Machine precision.**
- **pi0.5 num_steps=10 ONNX**: cos = +1.0000000, max_abs = 2.38e-07 (first-action) vs `sample_actions(num_steps=10)`. **Machine precision.**

Getting pi0 / pi0.5 to cos=1.0 at num_steps=10 required three interacting patches (under `torch.export` + `transformers==5.3.0` + DynamicCache): (1) F.pad + logical AND for the block-causal mask instead of `torch.cat` (cat loses the suffix dim under FakeTensor); (2) freeze `DynamicLayer.update` during the unrolled Euler loop so the cache doesn't grow across iterations; (3) use `past_kv.get_seq_length()` instead of the pad-mask shape for mask assembly.

```bash
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'
reflex export lerobot/smolvla_base --output ./smol
reflex serve ./smol --port 8000
# POST /act returns 50-step action chunks
```

**Also ships:**
- Docker image published to GHCR (`ghcr.io/rylinjames/reflex-vla:latest`) — no CUDA driver wrangling
- ROS2 bridge (`reflex ros2-serve`) — subs image/state/task, pubs action chunks
- Safety guard with NaN/Inf rejection + consecutive-clamp kill-switch
- Auto-generated `VERIFICATION.md` per export directory — sha256 of every file, opset, and (after `reflex validate`) per-fixture cos/L2 numbers for audit

**The hard parts were non-obvious.** Three issues under transformers 5.x took most of the session to isolate:

1. `transformers 5.4+` has a `q_length` scalar regression in `masking_utils.sdpa_mask` that breaks onnx-diagnostic patches. Pinning `transformers==5.3.0` fixes it.
2. SmolVLM2's vision embedder does `torch.where(bool_mask, torch.full(fill_value=0), float_tensor)` where `fill_value=0` creates an int64 branch. `torch.export` traces this with mismatched dtypes and the resulting ONNX `Where` op is rejected by onnxruntime at load time. Fix: wrap `torch.where` to insert explicit `torch.promote_types`.
3. Even with a clean aten graph, `torch.onnx.export` sometimes lowers `index_put` to a `Where(bool, int64, float)` ONNX node. Fix: post-export pass that walks Where nodes and inserts Cast nodes targeting the declared output dtype.

**What's explicitly NOT done:**
- GR00T (DiT + AdaLN) ONNX parity — v0.3
- Jetson latency numbers — CloudJetson has only AGX Orin 64GB available; Orin Nano waitlisted. Launch numbers are from Modal A10G; real Jetson data comes when someone runs `reflex bench` on a dev kit
- **Orin Nano 8GB fit for pi0 / pi0.5.** The pi0 / pi0.5 monolithic ONNX is 12.5–13GB (FP32) and does not fit on Orin Nano 8GB in any precision once activations + OS are counted. SmolVLA (1.6GB) fits fine. pi-family models realistically need Orin 16GB+ or a desktop NVIDIA GPU. FP16 engine rebuild + Orin Nano fit is a v0.3 item
- Earlier TRT FP16 latency tables were on a now-abandoned decomposed-ONNX path; latency re-measurement on the monolithic path is in v0.3

Repo: https://github.com/rylinjames/reflex-vla
Verified numbers ledger: [reflex_context/measured_numbers.md](https://github.com/rylinjames/reflex-vla/blob/main/reflex_context/measured_numbers.md)

Apache 2.0, single maintainer. Looking for testers — especially anyone with a real robot or a Jetson Orin Nano dev kit. Open an issue, I respond fast.

---

## Tone notes

- Lead with verified cos numbers; avoid hype
- Disclose what's NOT done before commenters ask
- Don't oversell (no "revolutionary" / "game-changing")
- No "we" — single maintainer
