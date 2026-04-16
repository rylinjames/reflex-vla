# VLM-2 Result: Update SmolVLA expert ONNX to accept vlm_kv input

**Status:** DONE
**File modified:** `src/reflex/exporters/smolvla_exporter.py`

## Changes

1. **ExpertStack.forward()** — Added optional `vlm_kv: torch.Tensor | None = None` parameter. When None, falls back to zeros (backward compat). When provided, uses the tensor as cross-attention KV input.

2. **ONNX export** — Added `vlm_kv` as 4th input with dynamic axes `{0: "batch", 1: "seq"}`. Dummy tensor `torch.zeros(1, 1, vlm_kv_dim)` used for export trace.

3. **ONNX validation** — Updated to pass `vlm_kv` to both ORT session and PyTorch forward call.

4. **Config metadata** — Added `"vlm_kv_input": True` and `"vlm_kv_dim": vlm_kv_dim` to export config.

## Verification

- `vlm-kv-param-ok` — ExpertStack.forward signature includes vlm_kv
- `grep -c vlm_kv` = 19 (>= 5 required)
- Backward compatible: vlm_kv defaults to None, which produces zeros identical to v0.1 behavior
