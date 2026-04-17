# VLM Prefix Pipeline — 4-File ONNX Export

**Purpose:** SmolVLA's VLM conditioning (image + task + state → per-layer K/V cache consumed by the expert) is split across four ONNX files so the serve path can orchestrate them as pipeline stages. This document describes each file's inputs, outputs, shapes, dtypes, and how `VLMPrefixOrchestrator` chains them.

**Status:** Shipped via commits `5869a3e` (wave 1: vision + text + state + dim fix), `d5b6570` (wave 2: decoder_prefill + orchestrator), `9fb6ddb` (wave 3: 25 tests passing), `36d8a40` (real `text_embedder.onnx`), `fdd9bb3` (unified into `reflex export`).

**Historical context:** The plan was originally **3 files** (vision + embed_tokens + decoder). Research on the ETARS repo's SmolVLA ONNX notebook (current_session line 7195, `aifoundry-org/ETARS::smolVLA_libero_export.ipynb`) revealed SmolVLA actually needs **4 files** (commit `3d90808`). The ETARS pattern uses `expert_prefill` + `expert_decode` splits; Reflex adapted this to `decoder_prefill` + `expert_stack` because we already had a standalone expert exporter.

---

## File 1 — `vision_encoder.onnx`

**Purpose:** SigLIP vision tower (SmolVLM2 base) + SmolVLM connector (pixel_shuffle + linear projection) → image embeddings at VLM hidden dim.

### Inputs

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `pixel_values` | `[B, 3, 512, 512]` | float32 | **MUST be in [-1, 1] range** (SigLIP normalization; bug #4 in current_session line 10835). Caller applies `img * 2.0 - 1.0` before feeding. |

### Outputs

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `image_embeds` | `[B, 64, 960]` | float32 | 64 image tokens after pixel_shuffle; 960 = VLM `hidden_size`. |

### Architecture details

From commit `5869a3e` (`vlm_components.py::VisionEncoderForONNX`):
- Wraps `vlm.model.vision_model` (SigLIP SO400M) + `vlm.model.connector` (pixel_shuffle + Linear)
- **Pre-computes position IDs as a persistent buffer** to avoid the dynamic `index_put` / `bucketize` loop in `SmolVLMVisionEmbeddings.forward()` — that loop produces ONNX nodes with int64/float type mismatches ORT can't load.
- Post-export, `patch_onnx_type_mismatches(onnx_path)` scans for any remaining float-indexed Gather ops and inserts Cast nodes.
- Export flags: `do_constant_folding=False` (per ETARS research — folding corrupts the graph), opset_version=19.

### Export size and cost

- Output ONNX: **394.1 MB** (from modal_apps_and_pm_docs.md `ap-uKaH8uEPuCeoKz0C6TCqbV` healthy run)
- Param count: **86.4M** vision + ~5M connector
- Numerical threshold: `ORT_MAX_DIFF_THRESHOLD = 5e-4`. SigLIP's 27 transformer layers accumulate fp32 rounding; max_diff ~2-4e-4 is expected and passing.

### Known numerics

From stage-diff `ap-YrnHF0WgFXQ2Y7HWlYHPaI` (healthy run after `AutoModelForImageTextToText` unwrap fix in commit `5869a3e`):
```
Vision stage: cos=+1.0000, L2=2.2022e-03, max_abs=1.0681e-04
```
From `ap-oXrqhfnQFJLuuY4A9GbPSv`: max_abs=8.7738e-05 (tighter).

**Pre-fix state:** `cos=0.6983, torch||=1644 vs onnx||=1104`. Root cause was loader wrapping — checkpoint has `model.connector.*` / `model.text_model.*` but our loader loaded as `connector.*` / `text_model.*` (extra `model.` prefix). Fix: use `AutoModelForImageTextToText` + unwrap inner `.model`.

---

## File 2 — `text_embedder.onnx`

**Purpose:** SmolLM2 `embed_tokens` lookup table. Token IDs → embeddings at VLM hidden dim. Replaces the seeded-random fallback used in v0.1.

**Landed:** Commit `36d8a40` (2026-04-16 14:20). Before this, the VLM orchestrator used a deterministic seeded fallback keyed on token IDs (commit `f882dcb`) which was deterministic but not numerically correct.

### Inputs

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `token_ids` | `[B, T]` | int64 | Tokenized task description. Must include trailing newline (bug #11 in 12-bugs table). |

### Outputs

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `text_embeds` | `[B, T, 960]` | float32 | Raw embeddings — caller applies `sqrt(hidden)` scale downstream. |

### Architecture details

- Isomorphic to `nn.Embedding(vocab_size, 960)` — just the `embed_tokens` layer extracted from SmolLM2.
- Vocab size is SmolVLM2-500M's text model vocab (~50k).
- Dynamic axis: sequence length `T`. All other axes static.

### Export size

- Output ONNX: **189.2 MB** (per stage-diff healthy run)

### Tokenizer source

Per commit `f882dcb`:
> *"`vlm_model_id` (the BASE VLM, e.g. "HuggingFaceTB/SmolVLM2-500M-Video-Instruct") used for tokenizer/processor, NOT `model_id` (which is the SmolVLA policy checkpoint and doesn't have a tokenizer). Config now writes `vlm_model_id = checkpoint_path_or_id` at export."*

This is critical: the SmolVLA policy checkpoint does NOT ship a tokenizer. Loading `tokenizer = AutoTokenizer.from_pretrained(policy_id)` fails silently. Load from `vlm_model_id` instead. `reflex_config.json` records both.

### Known numerics

From stage-diff `ap-YrnHF0WgFXQ2Y7HWlYHPaI`: `Text embed: cos=+1.0000` (perfect match).

**Determinism test (current_session line 7541):**
| Test | Before fix | After fix |
|---|---|---|
| Determinism (same input → same output) | FAIL (max_diff 34.7) | PASS (max_diff 0.0) |
| Semantic sensitivity (different instructions → different output) | PASS (0.15) | PASS (0.12-0.14) |

---

## File 3 — `decoder_prefill.onnx`

**Purpose:** The SmolLM2 text decoder's **first 16 layers**, running in prefill mode on the assembled prefix `[image_embeds; text_embeds; state_embed]`. Produces per-layer K/V tensors that the expert cross-attention layers consume.

**Landed:** Commit `d5b6570` (wave 2).

### Inputs

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `inputs_embeds` | `[B, prefix_len, 960]` | float32 | Concat of image (64) + text (T) + state (1). Already scaled by √hidden on image/text paths by caller. |
| `attention_mask` | `[B, prefix_len]` | int64 | 1 for valid tokens, 0 for padding. |
| `position_ids` | `[B, prefix_len]` | int64 | Canonical positions `0, 1, ..., prefix_len-1`. Caller computes via `mask.long().cumsum(-1) - 1` then clamps min=0 (per modal_stage_diff.py). |

### Outputs

For each of the 16 layers, two tensors:

| Name pattern | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `layer_k_{i}` for i in 0..15 | `[B, prefix_len, 320]` | float32 | Post-RoPE K for layer i. 320 = nkv × head_dim = 5 × 64. |
| `layer_v_{i}` for i in 0..15 | `[B, prefix_len, 320]` | float32 | V for layer i. **No RoPE applied to v** (per smolvla_forward_pass.md). |

Plus final residual stream (optional, not used by orchestrator):

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `last_hidden_state` | `[B, prefix_len, 960]` | float32 | Decoder output after all 16 layers — not consumed in current pipeline but exported for debug. |

### Architecture details

- Exports only the first 16 of 32 layers of SmolVLM2-500M's text model (as SmolVLA uses — the checkpoint ships pre-truncated; our loader respects that).
- Each layer runs full LlamaDecoderLayer: input_layernorm → self-attn → residual → post_attention_layernorm → SwiGLU MLP → residual.
- **GQA 15Q/5KV** (hidden_size=960, head_dim=64, num_kv_heads=5). Per commit `6fedff3`'s spike: "GQA confirmed: 15 Q heads, 5 KV heads, hidden=960, head_dim=64."
- RoPE applied per layer to Q and K (not V).
- `cos, sin` computed externally as a `(cos, sin)` tuple, passed via `position_embeddings` kwarg. Per commit `6fedff3`: "RoPE = standard HF LlamaRotaryEmbedding, lives on `model.text_model.rotary_emb`, computes (cos, sin) externally, attention layer receives position_embeddings as (cos, sin) tuple."

### Export size

- Output ONNX: **596.6 MB** (per stage-diff healthy run — largest of the 4 files)
- Param count: **263.8M** (decoder text-model half)

### Known numerics

From stage-diff `ap-YrnHF0WgFXQ2Y7HWlYHPaI` (per-layer post-unwrap):
```
layer_0_k  cos=+1.0000,   layer_0_v cos=+0.9117   ← outlier
layer_8_k  cos=+0.9997,   layer_8_v cos=+0.9967
layer_15_k cos=+0.9994,   layer_15_v cos=+0.9954
```

**The `layer_0_v` cos=0.9117 outlier is the reproducible structural discrepancy** currently blocking LIBERO task success. Consistent across runs `oXrqhfnQFJLuuY4A9GbPSv` and `2tNsuBRSnvuQ9kWPwm55Ob`. This is what task #25 "Per-layer vlm_kv ONNX export" is chasing.

### Open gap

Per commit `fdd9bb3`: "VLM weights come from BASE SmolVLM2-500M — fine-tuned SmolVLA VLM weight transfer tracked as v0.3 item." The 16 layers are currently loaded from the base VLM, not the policy checkpoint's fine-tuned VLM layers. Stage-diff logs confirm: *"Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)"*.

---

## File 4 — `expert_stack.onnx`

**Purpose:** The 16-layer action expert. Takes noisy actions + timestep + per-layer vlm_kv → velocity for flow-matching integration step.

**Landed:** Commit `47f3d5d` (full 16-layer stack), `c1726e7` (unified export), `7ed41aa` (vlm_k/vlm_v named inputs).

### Inputs

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `noisy_actions` | `[B, chunk=50, max_action_dim=32]` | float32 | Current flow-matching iterate |
| `timestep` | `[B]` or scalar | float32 | Current flow-matching time (1.0 → 0.0 over 10 steps) |
| `position_ids` | `[B, chunk=50]` | int64 | Position IDs for expert — renormalized to 0..49 for cross-attn, offset by prefix_len for self-attn |
| `vlm_k` | `[B, num_cross_attn_layers=8, prefix_len, 320]` | float32 | Stacked K from decoder_prefill, cross-attn layers only (odd indices) |
| `vlm_v` | `[B, num_cross_attn_layers=8, prefix_len, 320]` | float32 | Stacked V |

Note: some exports fold `timestep` into the `noisy_actions` preprocessing (pi0.5 uses time conditioning per-layer via AdaRMSNorm). SmolVLA concatenates time with actions in the suffix encoder before the expert sees them.

### Outputs

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `velocity` | `[B, chunk=50, max_action_dim=32]` | float32 | Flow-matching velocity; Euler integration: `x_new = x + velocity * dt` where `dt = -0.1` |

### Architecture details

- 16 expert layers alternating self-attn (even indices 0,2,4,...,14) and cross-attn (odd indices 1,3,5,...,15).
- Each layer = `ExpertGQALayer` (commit `74d24c3`).
- GQA: 15 Q heads, 5 KV heads, head_dim=64 — same as VLM.
- Expert hidden = 720 (not 960). Attention operates at VLM scale (960) via Q/K/V projections, then `o_proj: 960 → 720` back to expert hidden.
- 98.2M params; commit `47f3d5d`: "99.8M params. Full 10-step Euler: 202.1ms / 4.9Hz."
- Final `RMSNorm → action_out_proj (720 → max_action_dim=32)` outside the expert stack.

### Export size

- Output ONNX: **406.4 MB** + `.data` sidecar (commit `c1726e7`)

### Known numerics

- Full stack export max_diff: 3.81e-06 (commit `c1726e7`)
- Per-step velocity cos=0.977 (~2% error) on real LIBERO inputs (current_session line 11435)
- 10-step integration final cos=-0.24 (the remaining bug)

---

## `VLMPrefixOrchestrator` — how the 4 files chain together

Location: `src/reflex/runtime/vlm_orchestrator.py` (commit `d5b6570`, fixes `0838336` + `7ed41aa` + `f882dcb`, `36d8a40`).

### Initialization (lazy loading)

```python
class VLMPrefixOrchestrator:
    def __init__(self, export_dir, vlm_model_id, device="cuda"):
        self.export_dir = export_dir
        self.vlm_model_id = vlm_model_id  # e.g. "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

        # Loaded on first use:
        self._vision_session = None          # vision_encoder.onnx (required)
        self._text_session = None            # text_embedder.onnx (optional; falls back to seeded random by token IDs)
        self._decoder_session = None         # decoder_prefill.onnx (optional; returns assembled embeds if missing)
        self._state_session = None           # No ONNX — state_proj is inline numpy Linear

        # Cached:
        self._tokenizer = None               # AutoTokenizer.from_pretrained(vlm_model_id)
        self._processor = None               # AutoProcessor.from_pretrained(vlm_model_id)
        self._state_weight = None            # np.load("state_proj_weight.npy")
```

### Config values

```python
MAX_STATE_DIM = 32      # SmolVLA state_proj input dim (state is padded to this)
HIDDEN_SIZE = 960       # VLM hidden_size
```

### Forward pass (called from `reflex serve::predict`)

```python
def forward(self, image, instruction, state):
    # Stage 1a: Vision
    pixel_values = self._preprocess_image(image)        # [1, 3, 512, 512], scaled to [-1, 1]
    image_embeds = self._vision_session.run(
        ["image_embeds"],
        {"pixel_values": pixel_values}
    )[0]                                                  # [1, 64, 960]
    image_embeds = image_embeds * math.sqrt(HIDDEN_SIZE)  # √960 scale

    # Stage 1b: Text
    token_ids = self._tokenizer(instruction + "\n")      # trailing newline
    if self._text_session is not None:
        text_embeds = self._text_session.run(["text_embeds"], {"token_ids": token_ids})[0]
    else:
        text_embeds = self._seeded_text_fallback(token_ids)   # deterministic: seed by token IDs
    text_embeds = text_embeds * math.sqrt(HIDDEN_SIZE)   # √960 scale — same factor as image

    # Stage 1c: State
    state_padded = self._pad_state(state, MAX_STATE_DIM)
    state_embed = (state_padded @ self._state_weight.T).reshape(1, 1, HIDDEN_SIZE)  # [1, 1, 960]
    # NO √hidden scale on state path

    # Stage 2: Assemble
    prefix_embeds = np.concatenate([image_embeds, text_embeds, state_embed], axis=1)
    # Shape: [1, 64 + T + 1, 960]

    prefix_len = prefix_embeds.shape[1]
    attention_mask = np.ones((1, prefix_len), dtype=np.int64)
    position_ids = np.arange(prefix_len, dtype=np.int64).reshape(1, -1)

    # Stage 3: Decoder prefill (if available)
    if self._decoder_session is not None:
        decoder_outputs = self._decoder_session.run(
            None,
            {
                "inputs_embeds": prefix_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        # Outputs: layer_k_0, layer_v_0, layer_k_1, layer_v_1, ..., layer_k_15, layer_v_15, [last_hidden_state]
        vlm_k, vlm_v = self._extract_per_layer_kv(decoder_outputs)
        # Stack cross-attn layers only (odd indices 1, 3, ..., 15 → 8 layers)
        # Shape: [1, 8, prefix_len, 320]
    else:
        vlm_k, vlm_v = None, None   # expert falls back to zeros

    return {
        "prefix_embeds": prefix_embeds,
        "vlm_k": vlm_k,
        "vlm_v": vlm_v,
        "prefix_len": prefix_len,
    }
```

### State encoder — inline, no ONNX

Per commit `d5b6570`: *"State encoder weights: inline linear, no ONNX needed for 32→960 (MAX_STATE_DIM=32, HIDDEN_SIZE=960)."* The exporter saves `state_proj_weight.npy` (and bias if any) from the checkpoint; orchestrator does `state_padded @ weight.T` in numpy.

**Bug #1 (the smoking gun, current_session line 10679):** Our early exporter saved random weights instead of loading the real `state_proj` from the checkpoint. Fix: save actual `state_proj.weight` during export.

### Fallback chain

- **vision_encoder.onnx required** — server fails to start without it.
- **text_embedder.onnx optional** — if missing, fallback seeds `np.random.RandomState(hash(token_ids))` and draws deterministically. This was the v0.1 state; landing `text_embedder.onnx` in commit `36d8a40` closed the gap.
- **decoder_prefill.onnx optional** — if missing, orchestrator returns `prefix_embeds` + `vlm_k=None, vlm_v=None`. The expert stack consumes `vlm_k/vlm_v = zeros` which produces "action-shaped noise" (the README disclaimer).

### Session lifecycle

Per commit `0838336`:
- Each ONNX session opened lazily on first use.
- `close()` method iterates through sessions and closes them.
- `__del__` hook calls `close()` to prevent leaking sessions (fixed the "ONNX sessions never closed" bug from sessions_md.md line 28).

### ONNX input-name gotcha

Per commit `0838336`: The decoder_prefill session's first input is named `inputs_embeds` (ONNX-canonical), not `prefix_embeds` (our internal name). Fix: orchestrator maps internal name to ONNX-canonical name at session call time.

### Known gap — fine-tuned VLM weights

Per commit `fdd9bb3`: *"VLM uses base SmolVLM2-500M weights. Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)."* The decoder_prefill weights come from HuggingFace SmolVLM2 base, not the fine-tuned VLA policy checkpoint. Stage-diff logs confirm this is a known source of divergence.

---

## Where `reflex serve` uses the orchestrator

From commit `8a3b52c` (vlm wave 2):

```python
# Inside reflex serve /act handler:
orch = VLMPrefixOrchestrator(export_dir, vlm_model_id)   # Loaded once at startup

def predict(image, instruction, state, initial_noise=None):
    conditioning = orch.forward(image, instruction, state)

    vlm_k = conditioning["vlm_k"]    # [1, 8, prefix_len, 320] or None
    vlm_v = conditioning["vlm_v"]

    if vlm_k is None:
        # Fallback: expert runs with zero conditioning
        vlm_k = np.zeros((1, 8, 1, 320), dtype=np.float32)
        vlm_v = np.zeros_like(vlm_k)

    # Run expert flow-matching denoise loop
    actions = self._denoise(vlm_k, vlm_v, initial_noise)
    return actions
```

The `predict` path honors `--adaptive-steps` (early stop on velocity-norm convergence, pi0-only), `--deadline-ms` (fall back to last known good), `--safety-config` (ActionGuard clamp), and `--cloud-fallback` (SplitOrchestrator routing).

---

## Files on disk after `reflex export lerobot/smolvla_base`

```
<export_dir>/
├── vision_encoder.onnx                           # 394.1 MB
├── text_embedder.onnx                            # 189.2 MB
├── decoder_prefill.onnx                          # 596.6 MB
├── expert_stack.onnx                             # 406.4 MB
├── expert_stack.onnx.data                        # .data sidecar (external initializers)
├── state_proj_weight.npy                         # state encoder weights (32, 960)
├── policy_preprocessor_step_5_normalizer_processor.safetensors   # state normalizer (if LIBERO fine-tune)
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors  # action unnormalizer
├── reflex_config.json                            # model_type, action_dim, VLM file list, vlm_model_id, etc.
└── .trt_cache/                                   # created on first `reflex serve` (TRT engines)
```

Total ~1.6 GB per export. Plus the `.trt_cache` grows as the server compiles engines for the TRT EP.

---

## References

- Commits: `f72b8b9` (wave 1 stub), `8a3b52c` (wave 2 server wiring), `d641134` (wave 3 9 tests), `4daf6ea` (plan 3-file), `3d90808` (plan 4-file revision), `6fedff3` (GQA spike), `5869a3e` (real vision_components + dim fix 512→960), `d5b6570` (decoder_prefill + orchestrator), `9fb6ddb` (25 tests passing), `0838336` (ONNX input-name + close() + _state_session init), `7ed41aa` (validate backends vlm_kv dim 320), `f882dcb` (tokenizer from vlm_model_id), `36d8a40` (real text_embedder.onnx), `fdd9bb3` (unified export)
- Source files: `src/reflex/exporters/vlm_prefix_exporter.py`, `src/reflex/exporters/vlm_components.py`, `src/reflex/runtime/vlm_orchestrator.py`, `src/reflex/runtime/server.py`
- Diagnostic scripts: `scripts/modal_stage_diff.py` (per-stage cos_sim), `scripts/modal_pytorch_vs_onnx.py` (end-to-end diff)
- Research basis: ETARS repo `aifoundry-org/ETARS::smolVLA_libero_export.ipynb` (current_session line 7195)
- Reference paper: PaliGemma2 ONNX pattern — `onnx-community/paligemma2-3b-pt-224` has the 3-file split (vision_encoder + embed_tokens + decoder_model_merged)
