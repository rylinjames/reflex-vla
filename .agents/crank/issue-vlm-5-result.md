# VLM-5 Result: Tests for VLM prefix pipeline

**Status:** DONE
**File created:** `tests/test_vlm_prefix.py`

## Test results

```
9 passed, 8 warnings in 17.75s
```

## Tests written (9 total across 6 test classes)

1. **test_vlm_prefix_export_produces_valid_onnx** — Exports VLMPrefixEncoder to ONNX, loads in ORT, verifies input names (image, instruction_ids), output name (prefix_kv), output shape (1, 50, 512), and PyTorch-vs-ORT max_diff < 1e-4.

2. **test_expert_with_real_vlm_kv_differs_from_zeros** — Builds a tiny ExpertStack (2 layers, 1 cross-attn), runs forward with randn vlm_kv vs zeros, asserts L2 > 1e-3. Proves cross-attention actually uses the VLM input.

3. **test_expert_vlm_kv_none_fallback** — Verifies ExpertStack.forward() with vlm_kv=None runs without error and produces correct shape (backward compat).

4. **test_server_vlm_conditioning_real** — Wires ReflexServer with mock expert + VLM ORT sessions, calls predict() with image + instruction, asserts vlm_conditioning="real" and VLM session was called once.

5. **test_server_backward_compat_v01** — Wires server with v0.1 export dir (no vlm_prefix.onnx), calls predict(), asserts vlm_conditioning="dummy" and no errors.

6. **test_v01_with_image_still_dummy** — Even when image+instruction are provided, v0.1 server (no VLM session) stays dummy.

7. **test_config_v02_schema** — Creates v0.2 config JSON, loads it, verifies all fields present and correctly typed (vlm_prefix_onnx, vlm_image_size, vlm_kv_dim, export_version, vlm_prefix_seq_len).

8. **test_v01_config_missing_vlm_fields** — Verifies v0.1 config loads without VLM fields and .get() defaults work.

9. **test_vlm_prefix_exporter_updates_config** — Calls export_vlm_prefix with mocked checkpoint, verifies config file updated with vlm_image_size, vlm_kv_dim, vlm_prefix_onnx, export_version, and original fields preserved.

## Approach

- Used `tmp_path` fixtures for all file I/O
- Mocked `load_checkpoint` for export tests to avoid downloading real checkpoints
- Built a tiny ExpertStack fixture (64-dim hidden, 2 layers) for fast PyTorch tests
- Manually wired ReflexServer internals for server tests (avoids real ORT loading)
- All tests run in ~18s total (dominated by two ONNX export calls)
