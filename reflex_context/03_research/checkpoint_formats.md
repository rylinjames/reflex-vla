# Checkpoint Formats — Per-Model state_dict Structure

Every exporter in `src/reflex/exporters/` detects its model type by inspecting the loaded `state_dict`'s key prefixes. This document catalogs the canonical per-model layout so new exporters can be built without trial-and-error.

Detection logic lives in `src/reflex/checkpoint.py::detect_model_type()`. The CLI's auto-dispatch in `src/reflex/cli.py` calls `detect_model_type(state_dict)` first and routes to the matching exporter.

---

## SmolVLA (`lerobot/smolvla_base`, `lerobot/smolvla_libero`)

### Size + geometry

- **Total:** ~450M params, 907MB single safetensors file.
- **VLM backbone:** 350.2M (SmolVLM2-500M-Video-Instruct truncated to 16 decoder layers). Vision 86.4M + decoder 263.8M.
- **Action expert:** 98.2M (16 layers × 6.27M).
- **Projections:** 1.6M (action_in_proj, action_out_proj, state_proj, action_time_mlp).

### Canonical state_dict keys (from sessions_md line 2513)

```
model.vlm_with_expert.vlm:        345 tensors, 350.2M params  (VLM backbone)
model.vlm_with_expert.lm_expert:  145 tensors,  98.2M params  (Action expert)
model.action_time_mlp_in:          1.0M params  (timestep embedding)
model.action_time_mlp_out:         0.5M params  (timestep projection)
model.action_in_proj:              projection into action space
model.action_out_proj:             projection out of action space
model.state_proj:                  robot state projection (32 → 960)
```

### Detection signal

- **Prefix:** `model.vlm_with_expert.vlm.*` and `model.vlm_with_expert.lm_expert.*` together uniquely identify SmolVLA.
- **Layer count:** 16 transformer layers (detected from `model.vlm_with_expert.lm_expert.layers.0` through `.layers.15`).

### Expert geometry (probed from state_dict)

- `expert_hidden = 720` (derived from `model.action_in_proj.weight` shape, which is `[720, action_dim]`).
- `action_dim = 32` (derived from the same shape).
- `num_layers = 16`.
- `num_q_heads = 15`, `num_kv_heads = 5` (GQA 3:1 — from `q_proj` vs `k_proj` shape diff: `q_proj=[960, 720]`, `k_proj=[320, 720]`; 960 = 15·64, 320 = 5·64).
- `head_dim = 64` (expert_hidden / num_q_heads = 720 / ... actually VLM hidden 960 / 15 heads).
- `intermediate = 2048` (expert MLP).
- **Cross-attn layers:** odd indices `[1, 3, 5, 7, 9, 11, 13, 15]`. Detected by comparing `q_shape[1]` (input dim) to `k_shape[1]` — cross-attn has input dim = expert_hidden (720) but k_shape[1] = VLM KV dim (320). Self-attn has both = 720.
- **VLM→KV projection:** `vlm_to_kv_proj: nn.Linear(960, 320, bias=False)` — projects VLM hidden to cross-attn KV dim.

### VLM sub-structure (when loaded via `AutoModelForImageTextToText`)

- `model.connector.*` — SigLIP→SmolLM2 pixel-shuffle connector.
- `model.text_model.*` — SmolLM2 text decoder.
- `model.vision_model.*` — SigLIP vision encoder.

**Critical bug (current_session.md line 11063):** if loaded with `AutoModel.from_pretrained(...)` instead of `AutoModelForImageTextToText`, the prefix structure becomes `connector.*`/`text_model.*`/`vision_model.*` (no `model.` leader), which **silently drops the vision/text weights at merge time** (488 missing, 345 unexpected). Every inference then runs with randomly-initialized vision. Fix: unwrap `ForConditionalGeneration` and use `AutoModelForImageTextToText`.

### Normalizer checkpoint files (for LIBERO fine-tune)

The `lerobot/smolvla_libero` repo also ships:
- `policy_preprocessor_step_5_normalizer_processor.safetensors` — state input normalizer.
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors` — action output unnormalizer.

Keys inside the normalizer safetensors: `action_mean`, `action_std`, `state_mean`, `state_std`.

### Exporter file

`src/reflex/exporters/smolvla_exporter.py` — and the 4-file VLM split exports `vision_encoder.onnx`, `text_embedder.onnx`, `decoder_prefill.onnx`, `expert_stack.onnx`, with 960-dim hidden (NOT 512 as the v1 stub used).

---

## pi0 (`lerobot/pi0_base`, Physical Intelligence)

### Size + geometry

- **Total:** 3.5GB.
- **Expert:** 314.6M, 18 layers.
- **GQA:** 16 Q / 2 KV heads, head_dim=128.
- **FF layer:** plain GELU (NOT GEGLU — see sessions_md line 3846: `ff.net.0.proj` outputs `ff_inner` directly, not `2*ff_inner`).

### Canonical state_dict prefix

```
paligemma_with_expert.gemma_expert.model.*
```

### Detection signal

- **Prefix:** `paligemma_with_expert.*` uniquely identifies pi0.
- **Expert layers:** `paligemma_with_expert.gemma_expert.model.layers.0..17`.

### Exporter file

`src/reflex/exporters/pi0_exporter.py` — reuses SmolVLA's `ExpertGQALayer` and `ExpertStack` patterns. Export max_diff: 3.73e-08.

---

## pi0.5 (`lerobot/pi05_base`, Physical Intelligence)

### Size + geometry

- **Total:** 3.62GB. Expert 426.9M, 18 layers.
- **AdaRMSNorm `dense` layers add ~112M** over pi0 (314.6M → 426.9M).
- **Norm type:** AdaRMSNorm — time-conditioned RMSNorm: `time_emb → dense → chunk(3) → x*rsqrt(var+eps)*(1+scale)+shift`.
- **3-chunk AdaRMSNorm** (scale / shift / gate), distinct from GR00T's 2-chunk AdaLN.

### Canonical state_dict markers (from `modal_test_pi05.py`)

Two key markers for AdaRMSNorm detection:
- `input_layernorm.dense*` — the scale/shift projection matrix.
- `time_mlp*` — time conditioning MLP (runs at stack level, separate from action — distinct from pi0 where time is concatenated with action).

### Detection signal

- Same `paligemma_with_expert.*` prefix as pi0 PLUS AdaRMSNorm markers listed above.
- `detect_model_type` sets `model_type="pi05"`, `uses_adarms=True`.

### Exporter file

`src/reflex/exporters/pi0_exporter.py` contains both `ExpertStack` (pi0 path) and `Pi05ExpertStack` with `ExpertAdaRMSLayer` (pi0.5 path). Export max_diff: 2.37e-06.

---

## GR00T N1.6 (`nvidia/GR00T-N1.6-3B`, NVIDIA)

### Size + geometry

- **Total:** 6.6GB, 2 safetensors shards.
- **Expert:** 1091.7M + 10M buffers, 32-block DiT (NOT GQA — 32-head MHA, head_dim=48, hidden=1536).
- **Norm:** AdaLN 2-chunk (scale + shift — distinct from pi0.5's 3-chunk).
- **Attention pattern:** alternating — cross-attn on EVEN blocks (KV from VLM at 2048-dim), self-attn on ODD blocks (1536).
- **FF:** plain GELU-approx MLP (NOT GEGLU) — `ff.net.0.proj` outputs `ff_inner` directly.
- **LayerNorms:** non-affine.
- **Storage dtype:** BFloat16 (must cast to fp32 for export).

### Canonical state_dict prefix

```
action_head.model.transformer_blocks.*
```

Plus:
- `action_encoder.*` — per-embodiment input encoder (128 → 1536).
- `action_decoder.*` — per-embodiment output decoder (1024 → 128).
- `state_encoder.*`, `position_embedding.*`, `vlln.*`, `timestep_encoder.*`, `proj_out.*`.

### Detection signal

- **Prefix:** `action_head.model.transformer_blocks.*` uniquely identifies GR00T.
- **DiT block count:** 32. 448 DiT-related keys total.
- **3.29B total params** including VLM.

### Per-embodiment weight shape quirk

Weight shape is `[embodiment, in, out]` with **leading dim of 32** (32 embodiments in the model). The exporter slices at `embodiment_id=0` by default and transposes for `F.linear` compat. Other embodiment_ids can be selected at export time.

### Action encoder / decoder architecture

**Encoder (128 → 1536):**
```
action(128) → W1(1536) → silu → cat(h1, time_emb) → W2(1536) → silu → (h2 + h1) → W3(1536)
```
Residual + gating pattern.

**Decoder (1024 → 128):**
```
velocity_tokens(1024) → L1(1024) → silu → L2(128)
```

### The GR00T serve fix (commit `ff9fc3a 2026-04-14`)

**Bug:** expert emits 1024-dim velocity tokens but input is 1536-dim action tokens; `noisy + velocity*dt` can't cross dimensions.

**Fix:** wrap DiT expert with `action_encoder` (3 linears) + `action_decoder` (2 linears) pinned to `embodiment_id=0` by default. Input and output are now both `[b, chunk, raw_action_dim=128]` so the denoise loop works.

### Exporter file

`src/reflex/exporters/gr00t_exporter.py`. Export max_diff: 2.18e-05 (expert-only) or 3.77e-06 (full-stack). Full stack = 1091.7M + 10M buffers.

---

## OpenVLA (`openvla/openvla-7b`)

### Size + geometry

- **Total:** 7.5B params — Llama-2-7B backbone.
- **Action head:** NOT flow-matching. **`argmax(lm_logits[:, -7:])` + 256-bin lookup** on top of Llama-2 vocab.

### Canonical state_dict prefixes

```
vision_backbone.featurizer.*       (DINOv2)
vision_backbone.fused_featurizer.* (SigLIP)
projector.fc1.*, projector.fc2.*   (projector into LM)
model.*                            (Llama-2-7B LM)
```

### Detection signal

- **Prefix:** `vision_backbone.featurizer.*` + `projector.fc1.*` is the unique combination.
- `detect_model_type` sets `model_type="openvla"`.

### Exporter file

`src/reflex/exporters/openvla_exporter.py` — raises `NotImplementedError` deliberately. OpenVLA uses HF optimum-onnx (which already handles Llama-2 + DINOv2 + SigLIP + projector). Reflex ships only `src/reflex/postprocess/openvla.py::decode_actions` — the 256-bin lookup that isn't covered by optimum-onnx.

**Rationale (commit `c00ca82 2026-04-14`):** *"Building a full Reflex exporter here would duplicate optimum-onnx for zero architectural insight. Ship the postprocess helper + clear NotImplementedError."*

`reflex models` shows OpenVLA in **yellow** (partial / postprocess-only) vs full-support models in green.

---

## How detection actually works — `src/reflex/checkpoint.py`

Ordering matters. The detector checks in this priority (so that SmolVLA doesn't get misclassified as a random HF checkpoint):

1. **SmolVLA first** — key `model.vlm_with_expert.vlm.*` exists.
2. **pi0.5 second** — `paligemma_with_expert.*` exists AND `input_layernorm.dense*` or `time_mlp*` markers present.
3. **pi0 third** — `paligemma_with_expert.*` exists but no AdaRMSNorm markers.
4. **GR00T fourth** — `action_head.model.transformer_blocks.*` exists.
5. **OpenVLA fifth** — `vision_backbone.featurizer.*` + `projector.fc1.*` exist.
6. **Unknown** — none match. CLI falls through with a helpful `reflex models` hint.

Detected type is recorded in the output `reflex_config.json` alongside geometry (num_layers, expert_hidden, action_dim, cross-attn indices) so `reflex serve`, `reflex validate`, `reflex bench` downstream all know which denoise loop to run.

---

## Per-model ONNX artifact set

`reflex export` produces per-model:

| Model | Files produced |
|---|---|
| SmolVLA | `vision_encoder.onnx`, `text_embedder.onnx`, `decoder_prefill.onnx`, `expert_stack.onnx`, `reflex_config.json`, normalizer .safetensors (if LIBERO fine-tune) |
| pi0 | `expert_stack.onnx`, `reflex_config.json` |
| pi0.5 | `expert_stack.onnx`, `reflex_config.json` |
| GR00T | `expert_stack.onnx` (wrapped with action_encoder/decoder for embodiment_id=0), `reflex_config.json` |
| OpenVLA | **NOT PRODUCED** — exporter raises `NotImplementedError` with a pointer to `optimum-cli export onnx` + `reflex.postprocess.openvla.decode_actions` |

---

## Known checkpoint-detection edge cases

1. **Different save formats across HF checkpoint versions** — some SmolVLA checkpoints have the `model.` prefix stripped (flat `vlm_with_expert.*`). Prefix auto-detection in `modal_expert_export.py` handles this by scanning for "layers.0" variants with and without `.model.` embedded.
2. **Per-layer prefix auto-detection** (from `modal_expert_export.py`): compare `q_shape[1]` (input dim) to `k_shape[1]` — self-attn → both equal expert_hidden; cross-attn → k_shape[1] smaller (VLM KV dim).
3. **`AutoModel` vs `AutoModelForImageTextToText` for SmolVLA VLM** — see SmolVLA section above. Load the `ForConditionalGeneration` wrapper and unwrap the `.model` attribute, so state_dict has `model.connector.*` / `model.text_model.*` / `model.vision_model.*`. Using the plain `AutoModel` shifts the prefix by one and produces `144 missing, 0 unexpected` vs a healthy run — immediately visible in Modal logs (modal_apps line 39).
4. **BFloat16 storage for GR00T** — reflex exporter casts to fp32 at export time.
5. **Multiple embodiments per GR00T** — `action_encoder` / `action_decoder` weights are `[32, in, out]`. CLI defaults `embodiment_id=0`; meta reports 32 embodiments available.

---

## Cross-cutting gotchas for all checkpoints

- **`huggingface_hub.snapshot_download()`** downloads checkpoint AND all `.safetensors` shards; takes 30–90s per model on Modal. Cache by container session.
- **Shared-noise test methodology** for any PyTorch-vs-ONNX comparison: seed a single `np.random.RandomState(seed).randn(1, chunk, max_action_dim)` and inject the same noise into both paths, else cos_sim is dominated by noise drift (see `scripts/modal_pytorch_vs_onnx.py`).
- **VLM fine-tune weights are NOT preserved** by the current exporter — base VLM weights are loaded from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` and the SmolVLA-specific VLM fine-tuning is discarded. This is the known **v0.3 item** ("Fine-tuned SmolVLA VLM layers not yet preserved").
