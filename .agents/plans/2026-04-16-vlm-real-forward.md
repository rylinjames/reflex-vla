# Plan: Wire Real VLM Forward Pass into SmolVLA Export + Serve

**Date:** 2026-04-16
**Source:** reflex_context/02_research/2026-04-16-vlm-real-export.md
**Branch:** will be created by /evolve as evolve/vlm-prefix-encoder
**Approach:** Manual 3-file split (per SmolVLM-256M pattern), ETARS notebook as reference

## Overview

Replace the stub VLMPrefixEncoder (fake learned projection, wrong dim 512) with a real export of SmolVLA's VLM backbone: SigLIP vision encoder + pixel shuffle connector + SmolLM2 decoder (first 16 of 32 layers). Output: real prefix_kv of shape [batch, prefix_len, 960] that feeds the expert's cross-attention. Server runs the real VLM per /act request.

## Boundaries

**Always:**
- SmolVLA only (pi0/pi0.5 have no cross-attn; GR00T deferred)
- 3-file ONNX split: vision_encoder.onnx, embed_tokens.onnx, decoder.onnx
- vlm_kv_dim = 960 (SmolLM2 hidden_size), NOT 512
- SmolVLA uses first 16 of 32 SmolLM2 layers — truncate at export
- SigLIP input: 512×512, patch_size=16, produces 1024 patches × 768
- Pixel shuffle scale_factor=4: 1024→64 tokens, 768→12288 dims
- Connector: Linear(12288→960, no bias)
- Use `transformers.AutoModel` to load the VLM — don't reconstruct from raw state_dict
- ONNX opset 19
- CPU-only ORT sessions for VLM in v1
- Backward compat: old stub exports still work (server checks export_version)

**Never:**
- Do NOT export all 32 SmolLM2 layers — only the first 16 that SmolVLA actually uses
- Do NOT attempt KV-cache persistence across requests in this cycle
- Do NOT change the expert ONNX (already accepts vlm_kv from prior work)
- Do NOT touch pi0/pi0.5/gr00t exporters
- Do NOT use optimum-cli (confirmed unsupported for SmolVLM2/Idefics3)

## Baseline audit

| Metric | Value |
|--------|-------|
| Current stub VLMPrefixEncoder | 268 LOC, outputs [B, 50, 512] (WRONG) |
| Real VLM output shape | [B, prefix_len, 960] where prefix_len = 64 + T + 1 |
| SigLIP params | ~400M (SO400M variant) |
| SmolLM2 16-layer params | ~180M (half of 360M) |
| Total VLM download | ~350MB from HF |
| Expert cross-attn expects | dim 960 → k_proj/v_proj project to 720 |
| Reference implementations | ETARS notebook, poad42 RK3588 exporter, SmolVLM-256M HF ONNX |

## Issues

### Issue 1: Export SigLIP vision encoder as ONNX
**Dependencies:** None
**Files:** `src/reflex/exporters/vlm_prefix_exporter.py` (replace stub class)
**What:**
- Load SmolVLM2 via `AutoModel.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")`
- Extract `model.vision_model` (SigLIP ViT)
- Extract `model.connector` (pixel_shuffle + Linear(12288→960))
- Build a wrapper nn.Module: `VisionEncoder(vision_model, connector)`
  - forward(pixel_values [B, 3, 512, 512]) → image_tokens [B, 64, 960]
- Export to ONNX opset 19 as `vision_encoder.onnx`
- Validate: max_diff < 1e-4 vs PyTorch

**Acceptance:**
- `vision_encoder.onnx` exists in export dir
- Input: pixel_values [B, 3, 512, 512] float32
- Output: image_tokens [B, 64, 960] float32
- Numerical validation passes

### Issue 2: Export SmolLM2 decoder (16 layers) as ONNX
**Dependencies:** None
**Files:** `src/reflex/exporters/vlm_prefix_exporter.py` (add decoder export)
**What:**
- Extract `model.text_model` (SmolLM2 / LlamaForCausalLM)
- Truncate to first 16 layers (SmolVLA doesn't use layers 16-31)
- Build wrapper: `DecoderPrefix(text_model_truncated)`
  - forward(inputs_embeds [B, seq, 960], attention_mask [B, seq]) → last_hidden_state [B, seq, 960]
- Export to ONNX opset 19 as `decoder_prefix.onnx`
- Validate: max_diff < 1e-4 vs PyTorch
- Handle GQA attention (15 heads, 5 KV heads) — ONNX supports this natively

**Acceptance:**
- `decoder_prefix.onnx` exists
- Input: inputs_embeds [B, seq, 960], attention_mask [B, seq]
- Output: last_hidden_state [B, seq, 960]
- Only 16 layers (not 32)

### Issue 3: Export embed_tokens + build inference orchestrator
**Dependencies:** Issue 1, Issue 2
**Files:** `src/reflex/exporters/vlm_prefix_exporter.py` (replace export_vlm_prefix function)
**What:**
- Extract `model.text_model.embed_tokens` (Embedding layer)
- Export as `embed_tokens.onnx`: input_ids [B, seq] → embeddings [B, seq, 960]
- Build `VLMPrefixOrchestrator` class that:
  1. Loads all 3 ONNX sessions (vision_encoder, embed_tokens, decoder_prefix)
  2. forward(image, instruction):
     - Preprocess image → pixel_values [1, 3, 512, 512]
     - Tokenize instruction → input_ids [1, T]
     - Run vision_encoder → image_tokens [1, 64, 960]
     - Run embed_tokens → text_embeds [1, T, 960]
     - Concatenate: [image_tokens, text_embeds, state_embed] → [1, 64+T+1, 960]
     - Run decoder_prefix → last_hidden_state [1, prefix_len, 960]
     - Return prefix_kv = last_hidden_state
- Replaces the old stub export_vlm_prefix function
- Updates reflex_config.json with correct shapes + 3 ONNX file paths

**Acceptance:**
- `export_vlm_prefix()` produces 3 ONNX files + updated config
- `VLMPrefixOrchestrator.forward()` returns [B, prefix_len, 960]
- End-to-end: image + text → prefix_kv in correct shape for expert cross-attn

### Issue 4: Fix vlm_kv_dim 512→960 everywhere
**Dependencies:** None (can run parallel with Issues 1-2)
**Files:** Multiple: vlm_prefix_exporter.py constants, smolvla_exporter.py, server.py, tests, validate_roundtrip backends
**What:**
- Update DEFAULT_VLM_KV_DIM from 512 to 960
- Update ExpertStack's vlm_kv_dim detection logic (it should already detect 960 from checkpoint — verify)
- Update server's fallback vlm_kv_dim
- Update test fixtures + assertions that hardcode 512
- Update _pytorch_backend.py and _onnx_backend.py vlm_kv shapes (on the validate branch — may need merge)

**Acceptance:**
- `grep -r "512" src/reflex/exporters/vlm_prefix_exporter.py` returns 0 hits (for vlm_kv context)
- All tests pass with 960 dim
- Expert ONNX vlm_kv input shape is [B, seq, 960]

### Issue 5: Wire real VLM into server + update tokenizer
**Dependencies:** Issue 3, Issue 4
**Files:** `src/reflex/runtime/server.py`
**What:**
- Replace the single `_vlm_session` with `VLMPrefixOrchestrator` (loads 3 sessions)
- Use `AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")` for real tokenization (replace the ordinal char fallback)
- Use `AutoProcessor` for image preprocessing (SigLIP-specific resize + normalize)
- predict() now produces real prefix_kv [1, prefix_len, 960]
- /act response includes `vlm_conditioning: "real"` with actual VLM inference time

**Acceptance:**
- Server loads 3 VLM ONNX sessions at startup
- predict(image, instruction) produces task-conditioned actions
- Different instructions produce meaningfully different action trajectories
- /health reports vlm_loaded with model details

### Issue 6: Tests + validation
**Dependencies:** Issues 1-5
**Files:** `tests/test_vlm_prefix.py` (update existing), `tests/test_vlm_real.py` (new)
**What:**
- Update existing tests for 960 dim
- New test: real VLM export produces 3 valid ONNX files
- New test: VLMPrefixOrchestrator end-to-end (image + text → prefix_kv shape)
- New test: different instructions → different prefix_kv (semantic sensitivity)
- New test: server with real VLM produces actions that change when instruction changes
- Integration test (gated): full SmolVLA export + serve + /act with real image
- All tests use mocks except integration (which downloads 350MB from HF)

**Acceptance:**
- All existing tests still pass (backward compat)
- New tests verify real VLM behavior
- pytest -v shows all green

## Execution order

**Wave 1** (parallel): Issue 1, Issue 2, Issue 4
**Wave 2** (after W1): Issue 3, Issue 5
**Wave 3** (after W2): Issue 6

## Dependencies on external packages

- `transformers>=4.47.0` — already in pyproject.toml core deps
- `Pillow` — already in deps (image preprocessing)
- `AutoTokenizer` + `AutoProcessor` — from transformers, no new dep
- `AutoModel` — from transformers, no new dep
- HuggingFace Hub download: ~350MB first time, cached after

## Risk register

1. **SmolLM2 layer truncation:** Need to verify that simply slicing `model.text_model.layers[:16]` produces the same output as SmolVLA's internal 16-layer forward. If SmolVLA has custom layer indexing, this breaks.
2. **Pixel shuffle ONNX export:** reshape + permute pattern. Should export cleanly but verify shapes at every stage.
3. **GQA attention in ONNX:** SmolLM2 uses 15 heads / 5 KV heads (grouped query attention). ONNX opset 19 handles this natively, but verify no shape mismatches.
4. **HF download at export time:** First `reflex export` will download 350MB. Must be cached and not re-downloaded per call.
5. **Tokenizer max_length:** SmolVLA likely has a max instruction length. Need to discover and enforce.
6. **State token:** SmolVLA may encode the robot state as a special token in the prefix sequence. The stub skipped this. Research the SmolVLA policy's `embed_prefix()` to understand how state integrates.
