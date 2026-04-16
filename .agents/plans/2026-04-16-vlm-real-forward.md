# Plan: Wire Real VLM Forward Pass into SmolVLA Export + Serve (REVISED)

**Date:** 2026-04-16 (revised after council FAIL on v1 plan)
**Source:** reflex_context/02_research/2026-04-16-vlm-real-export.md + 2026-04-16-vlm-issue-research.md
**Approach:** 4-file ONNX split per ETARS pattern, using SmolVLAPolicy (not AutoModel)

## What changed from v1 plan (council corrections applied)

1. ~~3-file split~~ → **4-file split**: vision.onnx, text_embedder.onnx, expert_prefill.onnx, expert_decode.onnx
2. ~~AutoModel.from_pretrained~~ → **SmolVLAPolicy.from_pretrained** (gets 16 layers automatically)
3. ~~Custom VisionEncoder wrapper~~ → **`model.embed_image()` wrapper** (proven in ETARS)
4. ~~GQA "works natively"~~ → **2-hour spike first** (untested, highest risk)
5. Added: state encoder export, attention mask construction, patch_gather post-fix, tokenizer caching
6. Fixed: vlm_kv_dim 960 (not 512)

## Overview

Export SmolVLA's real VLM backbone as 4 ONNX files matching the ETARS pattern. Load via `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")` which gives the correct 16-layer VLM + action expert. Wire into server so /act with real image + instruction produces task-conditioned actions.

## Boundaries

**Always:**
- Load via `SmolVLAPolicy.from_pretrained` (NOT raw `AutoModel`)
- 4-file ONNX split: vision.onnx, text_embedder.onnx, expert_prefill.onnx, expert_decode.onnx
- `do_constant_folding=False` in ALL torch.onnx.export calls
- Run `patch_gather_indices_once()` on vision.onnx post-export (fixes float Gather indices)
- vlm_kv_dim = 960 everywhere
- State encoded via `nn.Linear(32, 960)` → 1 token
- Attention mask: bidirectional for image+text, causal for state token
- Cache AutoTokenizer + AutoProcessor at server startup (not per-request)
- SmolVLA only (no pi0/pi0.5/gr00t)
- CPU-only VLM inference in v1
- Backward compat with v0.1 exports (zeros fallback)

**Never:**
- Do NOT use `AutoModel.from_pretrained("SmolVLM2-500M")` — gives 32 layers
- Do NOT use `do_constant_folding=True` — breaks vision encoder export
- Do NOT skip the Gather index patch — produces float indices ORT can't run
- Do NOT assume GQA exports cleanly — spike first

## Issues

### Issue 0: SPIKE — GQA + manual RoPE ONNX export (DE-RISK)
**Dependencies:** None (runs first, gates everything)
**Effort:** 2 hours, time-boxed
**What:**
- Load SmolVLA via `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`
- Extract the 16-layer SmolLM2 decoder
- Attempt `torch.onnx.export` on a single decoder layer with GQA attention (15 heads, 5 KV heads)
- Check if SmolVLA's manual `apply_rope` traces cleanly
- Run through ORT, verify output matches PyTorch within 1e-4
**Gate:** If this fails, STOP and find a workaround (eager attention fallback, SDPA replacement, or custom op) before proceeding. If it passes, all downstream issues are unblocked.

### Issue 1: Export vision encoder via embed_image()
**Dependencies:** Issue 0 passes
**Files:** `src/reflex/exporters/vlm_prefix_exporter.py` (rewrite)
**What:**
- Load `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`
- Wrap `model.vlm_with_expert.embed_image()` in a thin nn.Module
- `torch.onnx.export` with input `pixel_values [1, 3, 512, 512]` → output `image_embeds [1, 64, 960]`
- `do_constant_folding=False`, opset 17
- Post-export: `patch_gather_indices_once("vision.onnx")`
- Validate max_diff < 1e-4

### Issue 2: Export text embedder + state encoder
**Dependencies:** Issue 0 passes
**Files:** `src/reflex/exporters/vlm_prefix_exporter.py` (add)
**What:**
- Extract `model.vlm_with_expert.text_model.embed_tokens` → export as `text_embedder.onnx`
  - Input: `input_ids [1, seq]` int64 → Output: `text_embeds [1, seq, 960]` float32
- Extract `model.state_proj` (nn.Linear(32, 960)) → export as part of the orchestrator (or inline)
  - Input: `state [1, 32]` float32 → Output: `state_embed [1, 1, 960]` float32
- Build the concatenation logic: `[image_special_tokens + image_embeds + text_embeds + state_embed]`
- Construct attention mask: bidirectional (0) for image+text, causal (1) for state

### Issue 3: Export expert_prefill (16-layer decoder producing prefix KV)
**Dependencies:** Issue 0 passes, Issue 1, Issue 2
**Files:** `src/reflex/exporters/vlm_prefix_exporter.py` (add)
**What:**
- Extract the 16-layer SmolLM2 decoder from SmolVLAPolicy
- Wrap as: `inputs_embeds [1, prefix_len, 960]` + `attention_mask [1, prefix_len]` → `last_hidden_state [1, prefix_len, 960]`
- Export as `expert_prefill.onnx`, opset 19 (needed for GQA support)
- `do_constant_folding=False`
- This is the compute-heavy part (~180M params, 16 transformer layers)

### Issue 4: Fix vlm_kv_dim 512→960
**Dependencies:** None (parallel with Issues 0-3)
**Files:** vlm_prefix_exporter.py, server.py, tests
**What:**
- Change DEFAULT_VLM_KV_DIM from 512 to 960
- Change server fallback from 512 to 960
- Update 5 test assertions from 512 to 960
- Verify expert's k_proj/v_proj handle 960→720 projection (should already work via checkpoint-inferred shapes)

### Issue 5: Wire 4-file orchestrator into server
**Dependencies:** Issues 1, 2, 3, 4
**Files:** `src/reflex/runtime/server.py`
**What:**
- Replace single `_vlm_session` with `VLMPrefixOrchestrator` that loads 4 ONNX sessions
- Cache `AutoTokenizer` + `AutoProcessor` at startup (not per-request)
- Remove char-ordinal tokenizer fallback
- `predict()` flow: preprocess image → vision.onnx → concat with text_embeds + state_embed → expert_prefill.onnx → prefix_kv → expert_decode denoise loop
- /act response: `vlm_conditioning: "real"`, VLM inference latency
- /health: `vlm_loaded: true` with file details
- Backward compat: v0.1 exports still fall back to zeros

### Issue 6: Tests
**Dependencies:** Issues 1-5
**Files:** `tests/test_vlm_prefix.py` (update), `tests/test_vlm_real.py` (new)
**What:**
- Update existing tests for 960 dim
- Test: GQA spike passes (single-layer export + ORT roundtrip)
- Test: vision.onnx produces [1, 64, 960] from image
- Test: 4-file orchestrator end-to-end produces [1, prefix_len, 960]
- Test: different instructions → different prefix_kv (semantic sensitivity)
- Test: state vector changes → different prefix_kv
- Test: backward compat with v0.1 exports
- Integration test (gated): full SmolVLA export + serve + /act

## Execution order

**Wave 0** (serial, gates everything): Issue 0 (GQA spike, 2h)
**Wave 1** (parallel, after spike passes): Issue 1, Issue 2, Issue 4
**Wave 2** (after W1): Issue 3, Issue 5
**Wave 3** (after W2): Issue 6

## Risk register

1. **GQA + manual RoPE ONNX export (CRITICAL)** — untested. SmolLM2's 15/5 head split + custom apply_rope may not trace. Spike de-risks this before any other work. Fallback: replace manual RoPE with HF's built-in RotaryEmbedding before export.
2. **Gather float indices in vision encoder** — known issue, mitigated by patch_gather_indices_once() post-export.
3. **Constant folding breaks export** — mitigated by do_constant_folding=False.
4. **HF download at export time** — 350MB first time. Cached after. Document in CLI output.
5. **State encoder shape** — max_state_dim=32, zero-padded. Must match SmolVLA's training convention exactly.
6. **Attention mask construction** — bidirectional vs causal split must match training. Wrong masking = silent quality degradation.
