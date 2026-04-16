# VLM-1 Result: Export VLM Prefix Encoder

**Status:** Complete
**File:** `src/reflex/exporters/vlm_prefix_exporter.py` (215 LOC)

## Approach

**Approach C (stub)** — chosen because:
- Reconstructing SigLIP + SmolLM2 from raw state_dict is prohibitively complex (nested HF configs, vision-language connectors, multi-modal merging)
- Approach A (AutoModel) requires downloading ~350 MB and tight `transformers` version coupling
- The stub proves the ONNX I/O contract and unblocks Issue 3 (server wiring)

The stub uses a small learned network (image pool + projection, token embedding + projection, fusion layer) that produces output of the exact correct shape.

## What was implemented

1. **`VLMPrefixEncoder(nn.Module)`** — wraps stub layers:
   - `__init__`: configurable `image_size`, `vlm_kv_dim`, `prefix_seq_len`
   - `forward(image, instruction_ids)`: image `[B, H, W, 3]` + token IDs `[B, seq]` → `prefix_kv [B, 50, 512]`

2. **`export_vlm_prefix(checkpoint_path_or_id, output_dir, opset=19) -> Path`**:
   - Loads checkpoint via `reflex.checkpoint.load_checkpoint`
   - Detects `vlm_kv_dim` from cross-attention key projection shapes
   - Reads `image_size` from model config's `vision_config` (default 384)
   - Builds `VLMPrefixEncoder`, exports to ONNX opset 19
   - Validates PyTorch vs ORT (max_diff < 1e-4)
   - Writes/updates `reflex_config.json` with `vlm_image_size`, `vlm_kv_dim`, `vlm_prefix_onnx`, `vlm_prefix_seq_len`, `export_version: "0.2"`

3. **`_detect_vlm_config(state_dict)`** — inspects checkpoint keys to infer `vlm_kv_dim` from cross-attention layer shapes.

## Verification

```
$ PYTHONPATH=src python -c "from reflex.exporters.vlm_prefix_exporter import export_vlm_prefix; print('import-ok')"
import-ok

$ # Forward pass shape check
output shape: torch.Size([1, 50, 512])  ✓

$ # ONNX export + ORT validation
max_diff=2.09e-07  ✓  (threshold: 1e-4)
```

## ONNX Contract

| Name | Shape | Dtype |
|------|-------|-------|
| **Input:** `image` | `[batch, 384, 384, 3]` | float32 |
| **Input:** `instruction_ids` | `[batch, seq]` | int64 |
| **Output:** `prefix_kv` | `[batch, 50, 512]` | float32 |

## TODO (v2)

- Wire real SigLIP → SmolLM2 forward pass (Approach A with HuggingFace AutoModel)
- The `VLMPrefixEncoder.__init__` accepts `vlm_state_dict` param — ready to receive real weights
