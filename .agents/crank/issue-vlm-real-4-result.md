# Issue 4 Result: Fix vlm_kv_dim from 512 to 960

## Status: DONE

## Changes Made

### Source files
1. `src/reflex/exporters/vlm_prefix_exporter.py:37` — `DEFAULT_VLM_KV_DIM = 512` changed to `960`
2. `src/reflex/runtime/server.py:442` — `self.config.get("vlm_kv_dim", 512)` fallback changed to `960`

### Test files
3. `tests/test_vlm_prefix.py` — 5 instances updated:
   - Line 87: `"vlm_kv_dim": 512` -> `960` (v02_export_dir fixture)
   - Line 251: `(1, 50, 512)` -> `(1, 50, 960)` (mock VLM session output shape)
   - Line 342: `"vlm_kv_dim": 512` -> `960` (test_config_v02_schema config)
   - Line 360: `== 512` -> `== 960` (assertion on loaded vlm_kv_dim)
   - Line 378: `get("vlm_kv_dim", 512) == 512` -> `get("vlm_kv_dim", 960) == 960` (v0.1 fallback test)

## Verification

- `grep -n "vlm_kv_dim.*512\|DEFAULT_VLM_KV_DIM.*512"` across all three files returns no matches
- All 9 tests in `tests/test_vlm_prefix.py` pass (11.90s, warnings only)
