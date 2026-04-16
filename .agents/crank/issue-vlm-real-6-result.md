# Issue 6 Result: Update tests for real VLM pipeline

## Status: DONE

## What changed

### Fixed `tests/test_vlm_prefix.py`

**Broken imports removed:**
- `VLMPrefixEncoder` (no longer exists -- replaced by `VisionEncoderForONNX`)
- `DEFAULT_INSTRUCTION_SEQ_LEN` (no longer exists)

**New imports added:**
- `VisionEncoderForONNX`, `export_vlm_prefix` from `vlm_prefix_exporter`
- `assemble_prefix`, `StateEncoder`, `pad_state`, `HIDDEN_SIZE`, `MAX_STATE_DIM` from `vlm_components`
- `VLMPrefixOrchestrator` from `vlm_orchestrator`

**Existing tests updated:**
- `TestVLMPrefixExport` (Test 1) replaced: old test instantiated `VLMPrefixEncoder` which no longer exists. Replaced with `TestVLMArchitectureConstants` that validates `DEFAULT_VLM_KV_DIM=960`, `HIDDEN_SIZE=960`, `MAX_STATE_DIM=32`.
- `TestServerVLMConditioning` (Test 3) updated: now mocks `VLMPrefixOrchestrator` instead of a raw `_vlm_session`. Added `test_server_vlm_conditioning_still_works` to verify `/act` always includes `vlm_conditioning` field.
- `TestVLMPrefixExporterUpdatesConfig` (Test 6) rewritten: old test tried to mock `load_checkpoint` which no longer exists. New test validates the config schema update logic directly.
- `v02_export_dir` fixture updated: `export_version` changed from `"0.2"` to `"0.3"`, `vlm_prefix_onnx` changed from `"vlm_prefix.onnx"` to `"vision_encoder.onnx"`, added `text_embedder.onnx` and `decoder_prefill.onnx` files.

**Backward-compat tests preserved:**
- `TestServerBackwardCompat` (v0.1 config, no VLM files -> dummy conditioning) -- unchanged and passing.
- `TestConfigV02Schema` (v0.1 config missing VLM fields) -- unchanged and passing.

### New tests added

| Test | Class | What it verifies |
|------|-------|-----------------|
| `test_assemble_prefix_shapes` | `TestAssemblePrefix` | Concatenation produces `[B, N_img+T+1, hidden]` and mask `[B, N_img+T+1]` |
| `test_assemble_prefix_attention_mask` | `TestAssemblePrefix` | Mask is 0 for image+text (bidirectional), 1 for state (causal) |
| `test_assemble_prefix_content_order` | `TestAssemblePrefix` | Verifies [image, text, state] concatenation order |
| `test_state_encoder_shape` | `TestStateEncoder` | StateEncoder 32->960 produces `[B, 1, 960]` |
| `test_state_encoder_custom_dims` | `TestStateEncoder` | Works with non-default dimensions |
| `test_state_encoder_with_weights` | `TestStateEncoder` | Provided weights are correctly loaded |
| `test_pad_state_1d` | `TestPadState` | `pad_state([1,2,3], 32)` -> `[32]` with zero padding |
| `test_pad_state_2d` | `TestPadState` | Works with batched `[B, D]` input |
| `test_pad_state_already_full` | `TestPadState` | No-op when already at max_state_dim |
| `test_pad_state_too_large_raises` | `TestPadState` | ValueError when state exceeds max |
| `test_orchestrator_vision_only` | `TestOrchestratorGracefulDegradation` | Vision-only dir: `is_loaded=True`, `is_complete=False` |
| `test_orchestrator_no_files` | `TestOrchestratorGracefulDegradation` | No ONNX files: `is_loaded=False` |
| `test_orchestrator_full_pipeline` | `TestOrchestratorFullPipeline` | All 4 ONNX sessions mocked, end-to-end produces `[1, seq, 960]` |
| `test_server_vlm_conditioning_still_works` | `TestServerVLMConditioning` | `/act` response always includes `vlm_conditioning` field |
| `test_different_instructions_different_prefix` | `TestDifferentInstructions` | Integration test gated by `REFLEX_INTEGRATION=1` |

## Test results

```
25 passed, 1 skipped in 11.38s
```

No regressions in other test files (test_validate, test_server, test_config: 21 passed, 1 skipped).
