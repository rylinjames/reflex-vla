# VLM-Real Issue 2 Result: Export text embedder + state encoder + concatenation logic

**Status:** DONE
**File created:** `src/reflex/exporters/vlm_components.py`

## Components implemented

1. **`export_text_embedder(model, output_dir, opset=19)`** — Extracts `embed_tokens` from a SmolVLM2/SmolVLAPolicy model and exports it as `text_embedder.onnx`. Input: `input_ids [B, seq] int64` → Output: `text_embeds [B, seq, 960] float32`. Uses `do_constant_folding=False`. Searches multiple attribute paths to find the embedding table.

2. **`StateEncoder(state_proj_weight, state_proj_bias, max_state_dim=32, hidden_size=960)`** — `nn.Module` wrapping `nn.Linear(32, 960)`. Forward: `state [B, 32]` → `state_embed [B, 1, 960]`. v1 uses random init; real weights load in orchestrator (Issue 3/5). Optional `export_state_encoder()` function also provided.

3. **`assemble_prefix(image_embeds, text_embeds, state_embed)`** — NumPy concatenation of `[image_embeds, text_embeds, state_embed]` along sequence dim. Returns `(prefix_embeds, attention_mask)` where mask = 0 for bidirectional (image+text) and 1 for causal (state token).

4. **`pad_state(state, max_state_dim=32)`** — Zero-pads 1D or 2D robot state arrays to `max_state_dim`. Validates bounds.

## Verification

```
vlm-components-ok  — assemble_prefix produces (1, 75, 960) prefix + (1, 75) mask
StateEncoder ok    — (2, 1, 960) output from (2, 32) input
pad_state 1D ok    — (32,) from (3,)
pad_state 2D ok    — (4, 32) from (4, 7)
attention mask ok  — 74 bidirectional + 1 causal
all-checks-passed
```

## Constants

- `HIDDEN_SIZE = 960` (SmolVLM2-500M / SmolVLA)
- `MAX_STATE_DIM = 32` (zero-padded from actual DoF)
- `IMAGE_SEQ_LEN = 64` (post-SigLIP pooled tokens)
