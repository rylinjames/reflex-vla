# Spike: SmolLM2 GQA Decoder ONNX Export

**Result: PASS**

**Date:** 2026-04-15

## Summary

SmolLM2's GQA decoder layer (LlamaDecoderLayer) exports to ONNX cleanly on the first attempt. No fallbacks needed.

## Approach That Worked

- `torch.onnx.export` with **opset 19**, using `torch.export.export(..., strict=False)` under the hood (PyTorch 2.11 new exporter)
- Single decoder layer wrapped with RoPE computation included in the wrapper
- No patches, no custom ops, no workarounds required

## GQA Head Config (Confirmed)

| Parameter | Value |
|---|---|
| num_attention_heads | 15 |
| num_key_value_heads | 5 |
| hidden_size | 960 |
| head_dim | 64 |
| num_layers | 32 (not 16 — full SmolVLM2-500M has 32 layers) |
| GQA ratio | 3:1 (15 Q heads / 5 KV heads) |

## RoPE Details

- **Standard HF `LlamaRotaryEmbedding`** — not custom
- Lives on `model.text_model.rotary_emb`, computes (cos, sin) externally
- Attention layer receives position_embeddings as a (cos, sin) tuple
- Exports cleanly — no `apply_rope` issues

## Numerical Accuracy

| Metric | Value |
|---|---|
| max_diff (PyTorch vs ORT) | 4.01e-05 |
| mean_diff | 7.57e-07 |

Well within acceptable tolerance for fp32 inference.

## ONNX File

- Path: `/tmp/smollm2_single_layer.onnx`
- Size: 0.1 MB (single layer)
- Opset: 19
- ORT loads and runs without errors — no unsupported ops

## Environment

- PyTorch: 2.11.0
- ONNX Runtime: 1.24.4
- Model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Loaded via: `transformers.AutoModel` (lerobot not installed)
- Underlying arch: `LlamaDecoderLayer` / `LlamaAttention`

## Key Takeaway

The decoder is vanilla Llama architecture with standard HF components. ONNX export is a non-issue. The VLM prefix plan is **not blocked** by decoder export.

## Note on Layer Count

The model has **32 layers**, not 16 as initially assumed. This may affect the VLM prefix split point calculation. Verify whether the "16 layers" in the plan refers to a prefix subset or was based on stale info.

## Fallbacks (Not Needed)

None of the fallbacks were triggered:
- torch.jit.trace — not needed
- Replacing apply_rope — not needed (standard HF RoPE)
- Full model export — not needed
- Opset 17 — not needed (19 worked)
