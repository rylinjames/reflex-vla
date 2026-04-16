# Issue 5 — ONNX inference backend

**Status:** complete
**File:** `src/reflex/_onnx_backend.py` (~290 LOC)

## What landed

- `load_onnx_backend(export_dir, device="cpu") -> ONNXBackend`
  - Resolves ONNX path via `config.files.expert_onnx` then falls back to `expert_stack.onnx`.
  - Pins `providers=["CPUExecutionProvider"]` (v1 reproducibility, per plan).
  - `device != "cpu"` is logged at INFO and honored as CPU intentionally.
  - Reads `reflex_config.json` into `self.config`.
  - Logs active provider (`session.get_providers()[0]`) and ONNX opset (via `onnx.load`).
  - Detects model_type from config or filename heuristic (`smolvla` / `pi0` / `gr00t`).
  - Logger name: `reflex.validate.onnx`.
- `ONNXBackend.forward(image, prompt, state, initial_noise) -> np.ndarray`
  - Consumes the supplied `initial_noise` exactly (cast to float32, batch-dim added if needed). Never calls `np.random` / `torch.randn` internally — seed bridge preserved.
  - Builds inputs dict `{"noisy_actions", "timestep", "position_ids"}`.
  - Defensive `embodiment_id` handling: only fed when the loaded session actually exposes it as a graph input. Today's `gr00t_exporter.py` bakes `embodiment_id=0` into the graph, so the standard path will not feed it; logged at DEBUG. If a future exporter version surfaces it, we feed `np.array([0], dtype=np.int64)` automatically.
  - Asserts `out.shape == (chunk_size, action_dim)` before returning.

## Denoise schedule chosen

Used the canonical Euler scheme matching `runtime/server.py`:

```
dt = -1.0 / num_steps
for step in range(num_steps):
    t = 1.0 + step * dt
    velocity = session.run(...)
    current = current + velocity * dt
```

Step count from `config.num_denoising_steps` (defaults: 4 for gr00t, 10 elsewhere). Issue 4's PyTorch backend should mirror this exactly; Issue 6 can add a parity test that asserts both backends produce the same output given the same `initial_noise` and a no-op velocity model.

## Verification

```
$ PYTHONPATH=src python -c "from reflex._onnx_backend import load_onnx_backend, ONNXBackend; ..."
onnx-backend-imports-ok
$ grep -c "class ONNXBackend" src/reflex/_onnx_backend.py        # 1
$ grep -c "CPUExecutionProvider" src/reflex/_onnx_backend.py     # 7
$ grep -c "initial_noise" src/reflex/_onnx_backend.py            # 9
$ grep -c "embodiment_id" src/reflex/_onnx_backend.py            # 7
```

Smoke test with a fake session (`velocity=0` → output should equal `initial_noise`) passes for both smolvla (chunk=50, action=6) and gr00t (chunk=50, action=32) shapes.

## Notes for downstream issues

- Issue 6 (CLI): wire `_onnx_backend.load_onnx_backend(self.export_dir)` from `ValidateRoundTrip._load_onnx`. `load_onnx_backend` raises `FileNotFoundError` for missing config / ONNX — translate to exit code 2.
- Issue 6 should also assert PyTorch-vs-ONNX denoise schedules match. Both should use `dt = -1.0/N`, `t = 1.0 + step*dt`, integrating from t=1 toward t=0.
- v2: when full-stack ONNX export lands, `image` / `prompt` / `state` parameters become real graph inputs. The signature is already in place.
