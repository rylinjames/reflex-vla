# Plan: VLM Prefix Encoder — SmolVLA Only

**Date:** 2026-04-16
**Source:** .agents/research/2026-04-16-vlm-prefix-encoder.md
**Branch:** evolve/vlm-prefix-encoder

## Overview

Replace dummy-zero VLM conditioning in SmolVLA's action expert with a real VLM prefix encoder. Export SigLIP + SmolLM2 backbone as a separate ONNX, update the expert ONNX to accept real vlm_kv input, wire the server to run VLM then expert per request. SmolVLA only in this cycle; GR00T in v2; pi0/pi0.5 don't need it (no cross-attention).

## Boundaries

**Always:**
- Backward compatible: if vlm_prefix.onnx is absent, server falls back to dummy zeros (v0.1 behavior)
- SmolVLA only — do NOT touch pi0/pi0.5/gr00t exporters in this cycle
- ONNX opset 19 for VLM prefix export
- CPU-only for v1 VLM inference in server (same as validate harness)
- No KV-cache persistence across requests in v1 — recompute VLM prefix per /act call
- Config versioning: add `"export_version": "0.2"` to distinguish new exports from v0.1

**Never:**
- Do NOT add KV-cache amortization (caching prefix across action chunks) in this cycle
- Do NOT change the denoise loop scheme (Euler, dt=-1/N, t=1→0)
- Do NOT break existing v0.1 exports — server must handle both old and new config formats
- Do NOT export pi0/pi0.5 VLM (they have no cross-attention)
- Do NOT attempt dynamic image resolution — fix to SmolVLA's SigLIP input size (384×384 if that's what it expects, or 512×512 — verify from checkpoint)

## Baseline audit

| Metric | Command | Result |
|--------|---------|--------|
| SmolVLA exporter LOC | `wc -l src/reflex/exporters/smolvla_exporter.py` | 352 |
| Server LOC | `wc -l src/reflex/runtime/server.py` | 782 |
| dummy_kv references | `grep -c dummy_kv src/reflex/exporters/smolvla_exporter.py` | 4 |
| vlm_kv_dim | `grep vlm_kv_dim src/reflex/exporters/smolvla_exporter.py` | 512 |
| cross_attn_layers | from exporter | [1, 5, 9, 13] (4 of 16) |
| predict() ignores image | `grep -n "image\|instruction" src/reflex/runtime/server.py` | params accepted but unused |

## Issues

### Issue 1: Export VLM prefix encoder (SmolVLA)
**Dependencies:** None
**Files:** `src/reflex/exporters/vlm_prefix_exporter.py` (new)
**Acceptance:**
- New module exports SigLIP + SmolLM2 backbone from SmolVLA checkpoint as `vlm_prefix.onnx`
- Inputs: `image [1, H, W, 3]` (float32), `instruction_ids [1, seq]` (int64)
- Outputs: `prefix_kv [1, seq_out, vlm_kv_dim]` (float32) — the projected VLM output that feeds cross-attention
- Uses `reflex.checkpoint.load_checkpoint` to load SmolVLA weights
- Extracts SigLIP vision encoder + SmolLM2 layers from state_dict
- Runs `torch.onnx.export` with opset 19
- Numerical validation: max_diff < 1e-4 between PyTorch and ONNX prefix outputs
- `export_vlm_prefix(checkpoint_path, output_dir, opset=19) -> Path`

**Description:** Extract the VLM backbone from SmolVLA's checkpoint. The forward pass is: image → SigLIP → patch embeddings → SmolLM2 transformer → final hidden states → project to vlm_kv_dim=512. The projection maps VLM hidden to the shape expected by the expert's cross-attention layers. Instruction tokenization uses the model's tokenizer (loaded from HF config). The ONNX output is a single `prefix_kv` tensor ready to feed into the expert's cross-attention.

### Issue 2: Update SmolVLA expert ONNX to accept vlm_kv input
**Dependencies:** None
**Files:** `src/reflex/exporters/smolvla_exporter.py` (modify)
**Acceptance:**
- `ExpertStack.forward()` accepts optional `vlm_kv` parameter instead of generating dummy zeros
- When `vlm_kv` is None, falls back to zeros (backward compat)
- ONNX export includes `vlm_kv` as a dynamic input with shape `[batch, seq, 512]`
- Old expert_stack.onnx (without vlm_kv input) still loadable by server (v0.1 compat)
- New expert_stack.onnx: inputs are `noisy_actions`, `timestep`, `position_ids`, `vlm_kv`

**Description:** Modify `ExpertStack.__init__` and `forward()` at line 98-141 to accept a `vlm_kv` tensor instead of allocating dummy zeros. Add `vlm_kv` to the ONNX export's `input_names` and `dynamic_axes`. The change is minimal — replace line 132's `dummy_kv = torch.zeros(...)` with `vlm_kv if vlm_kv is not None else torch.zeros(...)`.

### Issue 3: Wire VLM prefix into server predict()
**Dependencies:** Issue 1, Issue 2
**Files:** `src/reflex/runtime/server.py` (modify)
**Acceptance:**
- Server loads `vlm_prefix.onnx` as `_vlm_session` at startup (if present in export dir)
- `predict(image, instruction, state)` runs VLM session when image + instruction provided
- VLM output passed as `vlm_kv` input to expert session in `_run_denoise()`
- If `vlm_prefix.onnx` not found, logs warning and falls back to zeros (v0.1 behavior)
- `/health` endpoint reports `vlm_loaded: true/false`
- `/act` response includes `vlm_conditioning: "real" | "dummy"` field

**Description:** Add a second ONNX session for the VLM prefix. In predict(), preprocess the image (resize to SigLIP input size, normalize), tokenize the instruction, run VLM session, extract prefix_kv. Pass prefix_kv into the denoise loop alongside noisy_actions/timestep/position_ids. Update _run_denoise() to accept optional vlm_kv and feed it to the expert session. Backward compat: old exports without vlm_prefix.onnx still work with zeros.

### Issue 4: Update config schema + export orchestration
**Dependencies:** Issue 1, Issue 2
**Files:** `src/reflex/config.py` (modify), `src/reflex/cli.py` (modify export command output), `src/reflex/exporters/__init__.py` (if exists)
**Acceptance:**
- `reflex_config.json` gains: `"vlm_prefix_onnx": "vlm_prefix.onnx"`, `"export_version": "0.2"`, `"vlm_image_size": [H, W]`, `"vlm_kv_dim": 512`
- `reflex export lerobot/smolvla_base` now produces both `expert_stack.onnx` AND `vlm_prefix.onnx`
- CLI output shows both files + sizes after export
- Old configs (without vlm_prefix_onnx) still load — server detects v0.1 and skips VLM

**Description:** Extend the export orchestration for SmolVLA to call `export_vlm_prefix()` after `export_smolvla()`. Write updated config with new fields. CLI prints the new output file.

### Issue 5: Tests for VLM prefix pipeline
**Dependencies:** Issue 1, 2, 3
**Files:** `tests/test_vlm_prefix.py` (new)
**Acceptance:**
- Test: VLM prefix export produces valid ONNX (loadable by ORT)
- Test: Expert with real vlm_kv != expert with zeros (outputs differ)
- Test: Server predict() with image+instruction returns vlm_conditioning="real" in response
- Test: Backward compat — server with v0.1 export dir (no vlm_prefix.onnx) returns vlm_conditioning="dummy"
- Test: Config v0.2 schema loads with all new fields
- Uses mocks/stubs for heavy model loads (don't download 450M checkpoint in unit tests)
- Optional integration test gated by REFLEX_INTEGRATION=1

**Description:** Unit tests verifying the full pipeline from export through serve. Mock the PyTorch model with a small synthetic network that has the same interface (cross-attention layers, vlm_kv_dim) but tiny parameters.

## Execution order

**Wave 1** (parallel): Issue 1, Issue 2
**Wave 2** (parallel, after W1): Issue 3, Issue 4
**Wave 3** (after W2): Issue 5

Total: 5 issues, 3 waves.
