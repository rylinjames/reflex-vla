# Issue 4 result — PyTorch reference backend

## Summary

Created `src/reflex/_pytorch_backend.py` implementing the v1 PyTorch reference
surrogate for `reflex validate` round-trip parity. Reuses existing exporter
helpers — no copy-paste — so the surrogate matches the ONNX graph by
construction.

## What shipped

- New module `src/reflex/_pytorch_backend.py` (~290 LOC).
- Public API:
  - `load_pytorch_backend(export_dir, model_id, device="cpu") -> PyTorchBackend`
  - `class PyTorchBackend` with `forward(image, prompt, state, initial_noise)`
  - `UNSUPPORTED_MODEL_MESSAGE` constant (matches plan's verbatim string)
- Per-model dispatch:
  - `smolvla` -> `reflex.exporters.smolvla_exporter.build_expert_stack`
    (head_dim resolved via AutoConfig with the same fallback the exporter uses)
  - `pi0` -> `reflex.exporters.pi0_exporter.build_pi0_expert_stack` (head_dim=128)
  - `gr00t` -> `reflex.exporters.gr00t_exporter.build_gr00t_full_stack`
    pinned to `embodiment_id=0`. **Picked the full stack (not the bare DiT
    expert) so `forward()` accepts and returns raw-action shapes** — required
    for the ONNX denoise loop comparison to be apples-to-apples.
  - `pi05`, `openvla` -> `NotImplementedError(UNSUPPORTED_MODEL_MESSAGE)`
- `forward()` runs the same Euler flow-matching loop as
  `reflex.inference.flow_matching_denoise`: `actions += velocity * dt` over
  `num_denoising_steps` (default 10 smolvla/pi0, 4 gr00t — overridable via
  `reflex_config.json`).
- Wraps the loop in `torch.no_grad()`.
- Validates `initial_noise` shape against `[chunk_size, action_dim]` (accepts
  2D or 3D). Converts noise to torch on `self.device` exactly once.
- Logs via `logging.getLogger("reflex.validate.pytorch")`: which arch,
  param count (M), device.

## Design decisions / deviations

- **GR00T uses `build_gr00t_full_stack` not `build_gr00t_expert_stack`.** The
  bare DiT expert emits velocities in pre-decoder token space (1024-dim), not
  in raw-action space. Comparing that against an ONNX export that emits raw
  actions would be a shape mismatch. The full stack (encoder + DiT + decoder),
  pinned to `embodiment_id=0`, exactly matches what `export_gr00t_full`
  serializes.
- **`image`, `prompt`, `state` are accepted but unused** in v1. This is honest:
  the actual exporter graphs (smolvla / pi0 / gr00t) consume only
  `(noisy_actions, timestep, position_ids)` plus a zero-placeholder for VLM-KV
  on cross-attn blocks. Threading real image/prompt encoders into the surrogate
  would diverge from the ONNX graph and defeat the point of the comparison.
  Documented in the class docstring.
- **No new exporter refactor needed.** The three `build_*` helpers already
  exist as importable, factored functions — directly reused.
- A defensive `velocity.shape == actions.shape` check raises a clear error
  pointing at the GR00T full-stack-vs-expert distinction if a future tweak
  picks the wrong stack.

## Verification

```
$ cd "/Users/romirjain/Desktop/building projects/reflex-vla"
$ source .venv/bin/activate
$ PYTHONPATH=src python -c "..."
pytorch-backend-imports-ok

$ grep -c "class PyTorchBackend" src/reflex/_pytorch_backend.py
1
$ grep -c "def forward" src/reflex/_pytorch_backend.py
1
$ grep -c "initial_noise" src/reflex/_pytorch_backend.py
10
```

All assertions pass; module imports cleanly without loading a real checkpoint.

## Files touched

- `src/reflex/_pytorch_backend.py` (new, ~290 LOC)

## Out of scope (handed off)

- Wiring `_pytorch_backend` into `ValidateRoundTrip._load_pytorch` (Issue 6).
- Tests for the backend (Issue 7).
