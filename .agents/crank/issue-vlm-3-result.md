# VLM-3 Result: Wire VLM prefix into server predict()

**Status:** DONE
**File modified:** `src/reflex/runtime/server.py`

## Changes

### 1. Startup / __init__
- Added `_vlm_session`, `_vlm_loaded`, `_expert_input_names` instance vars
- `load()` now caches expert input names after loading ONNX and calls `_load_vlm_prefix()`

### 2. `_load_vlm_prefix()` (new method)
- Checks `reflex_config.json` for `vlm_prefix_onnx` field
- If present and file exists, loads as CPU-only `ort.InferenceSession`
- Logs VLM provider info (inputs/outputs) on success
- Falls back gracefully with warning if absent, missing, or fails to load

### 3. `_run_vlm_prefix()` (new method)
- Resizes image to `vlm_image_size` from config (default [512,512]) using PIL
- Normalizes to float32 [0,1]
- Calls `_tokenize_instruction()` for text encoding
- Runs VLM ONNX session, returns `prefix_kv` array

### 4. `_tokenize_instruction()` (new method)
- Tries `transformers.AutoTokenizer` first (uses `model_id` from config)
- Falls back to ordinal char encoding (each char -> ord % 50257, pad to 32)
- Returns int64 [1, max_seq] array

### 5. `predict()` changes
- Runs VLM prefix when `_vlm_session` is not None AND image AND instruction provided
- Passes `vlm_kv` to `_run_denoise()`
- Adds `"vlm_conditioning": "real" | "dummy"` to response

### 6. `_run_denoise()` changes
- Added `vlm_kv: np.ndarray | None = None` parameter
- Checks if expert ONNX has `vlm_kv` input (backward compat with v0.1)
- When expert expects `vlm_kv` but none provided, passes zeros of correct shape
- When expert does NOT have `vlm_kv` input (v0.1 exports), omits it entirely

### 7. `/health` endpoint
- `HealthResponse` model gains `vlm_loaded: bool = False`
- Response includes `vlm_loaded` from server state

## Backward compatibility
- v0.1 exports (no `vlm_prefix_onnx` in config): server logs "dummy conditioning" and works as before
- v0.1 expert ONNX (no `vlm_kv` input): `_run_denoise` detects via `_expert_input_names` and skips

## Cross-cutting constraints preserved
- Denoise loop scheme (dt, timesteps) unchanged
- CPU-only for VLM session
- No KV-cache persistence -- recomputes every call
- Existing batching logic untouched (vlm_kv is per-request path only)

## Verification
```
_run_denoise params: ['self', 'noisy_actions', 'position_ids', 'vlm_kv']
server-import-ok
grep count (vlm_kv|vlm_session|vlm_loaded|vlm_conditioning): 32
```
