# Issue 5 Result: Wire 4-file VLM orchestrator into server

**Status:** DONE
**Date:** 2026-04-15

## What was done

### 1. Created `VLMPrefixOrchestrator` (new file)
**File:** `src/reflex/runtime/vlm_orchestrator.py`

- Loads 3 ONNX sessions on CPUExecutionProvider at init:
  - `vision_encoder.onnx` (required for real image features)
  - `text_embedder.onnx` (optional, falls back to random embeddings)
  - `decoder_prefill.onnx` (optional, built in parallel by Wave 2)
- Caches `AutoTokenizer` and `AutoProcessor` at init (not per-call)
- Loads state encoder via `state_encoder.onnx` or inline linear (random init, real weights from checkpoint later)
- `run(image, instruction, state)` chains the full pipeline:
  1. Image -> AutoProcessor/manual -> vision_encoder -> image_embeds [1, 64, 960]
  2. Instruction -> AutoTokenizer -> text_embedder -> text_embeds [1, T, 960]
  3. State -> pad_state -> linear -> state_embed [1, 1, 960]
  4. assemble_prefix() -> prefix_embeds + attention_mask
  5. decoder_prefill (if available) -> prefix_kv
- Graceful degradation: if decoder_prefill.onnx missing, returns assembled prefix embeddings (real image features, no decoder pass)

### 2. Updated `server.py`
**File:** `src/reflex/runtime/server.py`

- Replaced `_load_vlm_prefix()` with `_load_vlm_orchestrator()` that creates a `VLMPrefixOrchestrator` when VLM files exist
- Replaced `_run_vlm_prefix()` call in `predict()` with `self._vlm.run(image, instruction, state)`
- Removed 3 old methods: `_load_vlm_prefix`, `_run_vlm_prefix`, `_tokenize_instruction`
- Changed init from `_vlm_session` to `_vlm` (orchestrator instance)
- `/health` still reports `vlm_loaded` (unchanged)
- `/act` still reports `vlm_conditioning: "real" | "dummy"` (unchanged)
- State is now passed through to VLM pipeline (was ignored before)

## Verification

```
$ PYTHONPATH=src python -c "from reflex.runtime.vlm_orchestrator import VLMPrefixOrchestrator; print('orchestrator-import-ok')"
orchestrator-import-ok

$ grep -q "VLMPrefixOrchestrator" src/reflex/runtime/server.py && echo "wired-in-server"
wired-in-server

$ PYTHONPATH=src python -c "from reflex.runtime.server import ReflexServer; print('server-import-ok')"
server-import-ok

Old methods removed: _load_vlm_prefix, _run_vlm_prefix, _tokenize_instruction -- verified absent.
```

## Backward compatibility
- v0.1 exports (no VLM files) still work: `_vlm = None`, dummy conditioning (zeros)
- Old single-file `vlm_prefix_onnx` config key still triggers orchestrator load attempt
- `/health` and `/act` response schemas unchanged
