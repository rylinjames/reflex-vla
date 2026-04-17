# SmolVLA Forward Pass — Canonical Reference

**Purpose:** This document is the single source of truth for how a real SmolVLA computes actions. Every ONNX exporter, every validation script, every serve-path implementation in Reflex must conform to this reference. Numerical divergence from this pipeline on LIBERO-10 killed task success on 5+ runs (per sessions_md.md line 85); the bugs are always subtle differences from this canonical path.

**Source of truth:** `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy` + the Apr-16 stage-diff debug campaign (commits `5869a3e`, `d5b6570`, `7ed41aa`, `f882dcb`, `36d8a40`). The "12 pipeline bugs" in `02_bugs_fixed/smolvla_inference_bugs.md` are all concrete ways our early exporter diverged from this.

---

## Architecture at a glance

```
Image (512×512)  ─┐
Task string      ─┼──> VLM prefix (SmolVLM2-500M, 16 of 32 layers)  ──> vlm_kv cache
Robot state (8D) ─┘                                                     │
                                                                        │
Noise (50 × 32)  ──> Expert (16 layers, self/cross alt) <───────────────┘
                     └── 10-step flow-matching Euler integration ──> actions (50, 32)
```

**Total params:** 450M (350M VLM truncated to 16 layers + 98M expert + 1.6M projections).

**Disagreement in the sources:** Early scripts assumed SmolVLM2-500M has 16 decoder layers (commit `da237f5` note: "truncated to 16 layers = 350.2M"). The Apr-16 GQA spike (commit `6fedff3`) confirmed **full SmolVLM2 is 32 layers; SmolVLA uses only the first 16.** Resolution: `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")` ships with 16 layers already truncated; loading `AutoModel.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")` gives all 32 and requires explicit truncation. Use the policy loader.

---

## Numeric constants (cite everywhere)

| Constant | Value | Source |
|----------|-------|--------|
| `hidden` (VLM + prefix) | **960** | SmolLM2 text_model hidden_size |
| `expert_hidden` | **720** | `int(vlm_hidden * 0.75)` per commit `1ed46ab` |
| `nq` (expert Q heads) | **15** | `expert_hidden / head_dim = 720 / 64 * ... wait, actually 960/64=15` |
| `nkv` (expert KV heads) | **5** | GQA 3:1; `k_proj.shape = [320, 720]` |
| `head_dim` | **64** | SmolLM2 head_dim |
| `rope_theta` | **100000** | SmolLM2 config (NOT 10000 — bug #8 in current_session) |
| `max_action_dim` | **32** | Expert output dim; LIBERO truncates to 7 |
| `chunk_size` | **50** | Actions per forward call |
| `num_denoise_steps` | **10** | Flow matching Euler |
| `dt` | **-0.1** | `-1.0 / num_steps` |
| `state_dim` | **8** | LIBERO: eef_pos(3) + axis_angle(3) + gripper_qpos(2); SmolVLA's `state_proj` is `Linear(32, 960)` — pads state to 32 |
| `image_size` | **512** (SigLIP SO400M native) | `modal_stage_diff.py` |
| `vision_output_shape` | `[B, 64, 960]` | After SigLIP vision tower + SmolVLM connector (pixel_shuffle) |
| `vlm_kv_dim` (for expert cross-attn) | **320** | = nkv × head_dim = 5 × 64; projected from VLM 960 inside the decoder |
| `cross_attn_every_n_layers` | **2** | Alternating: indices [0,2,4,6,8,10,12,14] self; [1,3,5,7,9,11,13,15] cross |
| VLM layers used | **16** (of 32) | First 16 decoder layers only |

**Disagreements to track:**
- state_dim 6 vs 8: LIBERO uses 8D (`eef_pos(3) + axis_angle(3) + gripper_qpos(2)`). Earlier scripts used 6D (joint positions). Bug #5 in current_session line 10679 catalogs this. Resolution: **8D**. SmolVLA's `state_proj` is `Linear(32, 960)` so it pads whatever state dim the env sends to 32 before projecting.
- `vlm_kv_dim` 320 vs 960 vs 512: early stub used 512 (commit `4daf6ea`); one exporter path confused VLM hidden (960) with expert cross-attn input (320). Resolution (commit `7ed41aa`): **read actual vlm_kv dim from the ONNX input shape** (`expert.vlm_kv_dim = 320`), NOT VLM `hidden_size = 960`. The expert cross-attn projects VLM-hidden → 320 internally.

---

## Stage 1: `embed_prefix` — image + text + state → prefix embeddings

Code path: `SmolVLAPolicy.model.embed_prefix`.

### 1.a Image path
```
pixel_values: [B, N_images, 3, 512, 512]   # SmolVLM expects 5D with num_images dim
image_embeds = vlm.model.embed_image(pixel_values)  # SigLIP tower + pixel_shuffle connector
# Shape: [B, 64, 960]
image_embeds = image_embeds * sqrt(hidden)          # √960 scale
```

**Critical gotcha (current_session bug #3, line 9605):** The VLM processor returns 5D `[B, N, 3, H, W]`. Our early ONNX wrapper expected 4D and silently dropped to zeros. Fix: accept 5D and reshape, or pre-compute the flatten.

**Critical gotcha (current_session bug #4, line 10835):** SigLIP expects images in `[-1, 1]` range. LIBERO/standard preprocessing gives `[0, 1]`. Must do `img = img * 2.0 - 1.0` before SigLIP.

**Critical gotcha (stage-diff modal_apps):** Vision output norm in torch is 1644 vs ONNX 1104 when the `AutoModelForImageTextToText` wrapper isn't unwrapped. Fix (current_session bug #2, line 11063): checkpoint's VLM structure is `model.connector.*` / `model.text_model.*` — our loader must unwrap the outer `model.` prefix.

### 1.b Text path
```
text_ids = tokenizer(task_string + "\n")            # Must include trailing newline — bug #11
text_embeds = embed_tokens(text_ids)                # Real SmolLM2 embed_tokens (text_embedder.onnx)
# Shape: [B, T, 960]
text_embeds = text_embeds * sqrt(hidden)            # √960 scale — same factor as image
```

**Critical gotcha (current_session bug #10, line 7534):** `_encode_text()` in our early fallback produced non-deterministic embeddings (max_diff 0.135 between consecutive calls with same input). Root cause: fallback hit `np.random.randn()` when `text_embedder.onnx` was missing. Resolution in commit `f882dcb`: seed RNG by token IDs so same instruction → same embedding. Then commit `36d8a40` added the real `text_embedder.onnx` (SmolLM2 `embed_tokens`) and retired the fallback.

**Tokenizer source (commit `f882dcb`):** Load tokenizer from **base VLM id** (`vlm_model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`), NOT the policy checkpoint (which has no tokenizer). Config records both.

### 1.c State path
```
state_padded = F.pad(state, (0, 32 - state_dim))    # Pad state (8D) to 32 with zeros
state_embed = state_proj(state_padded)              # Linear(32, 960)
# Shape: [B, 1, 960]
# NOTE: NO sqrt(hidden) scale on state path — only image + text
```

**Critical gotcha (current_session bug #1, line 10679 — "the smoking gun"):** Our early exporter saved `state_proj` as random weights, never loading the real checkpoint weights. `vlm_orchestrator._load_state_encoder` fell through to `np.random.randn(...) * 0.02`. Fix: save `state_proj_weight.npy` during export, load during serve.

### 1.d Assembly
```
prefix_embeds = concat([image_embeds, text_embeds, state_embed], dim=1)
# Shape: [B, 64 + T + 1, 960]
# Example: image=64 tokens, task_len=5 tokens, state=1 token → prefix_len=70
```

### 1.e Attention mask for prefix

```
attention_mask = [1]*64 + [1]*T + [1]*1             # All valid (no padding)
# Bidirectional inside prefix; causal not required for prefix assembly
# In decoder_prefill forward below, this becomes a full 2D mask (no causal triangle inside prefix)
```

**Critical gotcha (12-bugs table, "KV mask for cross-attn"):** We don't mask padded prefix positions in the expert cross-attn. Real SmolVLA does. Per-layer kv match after fixes but end-to-end still diverged to cos=-0.27 — the remaining gap likely lives here and in softmax-fp32 upcast.

---

## Stage 2: `vlm_with_expert.forward` — VLM prefill + expert denoise share transformer stack

This is the canonical reference. Per commit `6fedff3` (GQA spike): the decoder layer is `LlamaDecoderLayer` from SmolLM2. The "expert" is a separate set of 16 `ExpertGQALayer` instances that alternate self-attention and cross-attention into the VLM's per-layer KV cache.

### Layer structure (16 layers total)

Cross-attention every `self_attn_every_n_layers = 2`:
- Even indices (0, 2, 4, 6, 8, 10, 12, 14) → **self-attention** (8 layers)
- Odd indices (1, 3, 5, 7, 9, 11, 13, 15) → **cross-attention** from expert Q into VLM K/V (8 layers)

Per commit `47f3d5d`: "16 layers = 8 self-attn even + 8 cross-attn odd (indices [1,3,5,7,9,11,13,15])."

### Per-layer computation

Each expert layer does (from `_pytorch_backend.py` + `modal_expert_export.py`):

```python
# Layer pre-norm
residual = hidden
hidden = expert_layer.input_layernorm(hidden)       # DecomposedRMSNorm(eps=1e-6)

# Attention split
if is_self_attn(layer_idx):
    q = hidden @ q_proj.T                           # [B, S, 960]  (note: expert_hidden=720 but Q projects to VLM scale)
    k = hidden @ k_proj.T                           # [B, S, 320]
    v = hidden @ v_proj.T                           # [B, S, 320]

    # RoPE on q,k only (NOT v) — per commit 6fedff3
    q, k = apply_rope(q, k, cos, sin)

    # GQA: expand k,v from 5 heads → 15 (repeat_interleave factor 3)
    k = k.repeat_interleave(nq // nkv, dim=head_dim_axis)
    v = v.repeat_interleave(nq // nkv, dim=head_dim_axis)

    attn = softmax(q @ k.T / sqrt(head_dim), fp32) @ v   # attention mask applies here
    hidden = attn @ o_proj.T                        # 960 → 720 projection
else:  # cross-attn
    q = hidden @ q_proj.T                           # [B, chunk, 960]
    # k, v come from vlm_kv cache at this layer index
    k = vlm_k[layer_idx]                            # [B, prefix_len, 320]  (pre-RoPE applied during prefill)
    v = vlm_v[layer_idx]                            # [B, prefix_len, 320]  (no RoPE on v)

    q = apply_rope(q, cos_expert, sin_expert)       # expert gets renormalized position IDs
    # No RoPE on k here — VLM k was already rotated during prefill

    # Same GQA expansion + softmax + o_proj
    k = k.repeat_interleave(3, dim=head_dim_axis)
    v = v.repeat_interleave(3, dim=head_dim_axis)
    attn = softmax(q @ k.T / sqrt(head_dim), fp32) @ v
    hidden = attn @ o_proj.T                        # 960 → 720

hidden = residual + hidden

# MLP
residual = hidden
hidden = expert_layer.post_attention_layernorm(hidden)
gate = F.silu(gate_proj(hidden))
up = up_proj(hidden)
hidden = down_proj(gate * up)                       # SwiGLU

hidden = residual + hidden
```

### Position IDs — self-attn vs cross-attn

**Critical gotcha (12-bugs table, "prefix_offset for self-attn"):**
- **Self-attn layers:** `position_ids = arange(prefix_len, prefix_len + chunk_size)` — offset by `prefix_offset = prefix_len` so the expert's self-attention positions come AFTER the prefix.
- **Cross-attn layers:** `position_ids = arange(0, chunk_size)` — RENORMALIZED to start at 0 because the expert Q is just the 50 action tokens, not a continuation of the prefix.

This asymmetry is what "expert gets renormalized position IDs" means above.

### RoPE cos/sin computation

From commit `6fedff3` (the GQA spike, which confirmed PyTorch 2.11 new exporter handles RoPE cleanly):

```python
# Pre-compute cos/sin caches as model buffers
freqs_base = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))   # rope_theta = 100000 (bug #8)
pos = arange(max_seq_len)
freqs = outer(pos, freqs_base)                     # [max_seq_len, head_dim // 2]
cos_cache = cat([freqs.cos(), freqs.cos()], dim=-1)  # duplicate for full head_dim (bug: half-dim concat)
sin_cache = cat([freqs.sin(), freqs.sin()], dim=-1)
```

**Critical gotcha (sessions_md.md):** Early RoPE exporter concatenated `freqs.cos()` at wrong dim (half instead of full), giving half-size cos/sin that silently broke attention. The fix is `cat([freqs.cos(), freqs.cos()], -1)` — duplicate, not half.

**Critical gotcha (commit `6fedff3`):** HF `LlamaRotaryEmbedding` lives at `model.text_model.rotary_emb` and is called once externally to produce `(cos, sin)` tuple; attention layer receives it via the `position_embeddings` kwarg. Don't try to re-compute inside the attention layer in ONNX — hoist it.

### Attention at VLM scale (the subtle architectural point)

Per `expert_stack_layer_structure.md`: expert attention happens at **VLM scale** (15 heads × 64 head_dim = **960**), then `o_proj: 960 → 720` projects back to expert hidden. The expert's "hidden" dimension (720) is the residual stream dim; attention itself operates at VLM dim.

This is why:
- `q_proj: 720 → 960` (expert hidden → VLM scale Q)
- `k_proj: 720 → 320` (expert hidden → VLM scale KV)
- `v_proj: 720 → 320`
- `o_proj: 960 → 720` (back to expert hidden for residual)

---

## Stage 3: `embed_suffix` — noisy_actions + timestep → expert-input embeddings

```python
# Sinusoidal timestep embedding
time_emb = create_sinusoidal_pos_embedding(
    time_val,
    dimension=expert_hidden,
    min_period=4e-3,
    max_period=4.0,
)
# Shape: [B, expert_hidden=720]

# Action projection
action_emb = action_in_proj(noisy_actions)          # Linear(max_action_dim=32 → 720)
# Shape: [B, chunk=50, 720]

# Combine: SmolVLA concatenates timestep WITH action along hidden dim, then MLP-projects back
# (from modal_real_export.py:encode_suffix reconstruction)
time_emb_expanded = time_emb.unsqueeze(1).expand(-1, chunk, -1)   # [B, 50, 720]
combined = cat([action_emb, time_emb_expanded], dim=-1)           # [B, 50, 1440]
hidden = action_time_mlp_in(combined)                             # Linear(1440, 720)
hidden = F.silu(hidden)
hidden = action_time_mlp_out(hidden)                              # Linear(720, 720)
# Shape: [B, 50, 720] — ready to enter expert stack
```

**Disagreement:** Early `modal_real_export.py` built this as `silu(cat([action, t_emb]))` with hidden*2 → hidden MLP. Final `_pytorch_backend.py` uses the `action_time_mlp_in` / `action_time_mlp_out` path. Same math, the canonical names are `action_time_mlp_in` (concat → hidden) and `action_time_mlp_out` (hidden → hidden, post-silu).

**pi0.5 deviation:** pi0.5 uses AdaRMSNorm (commit `c0a3a7b`): time is NOT concatenated with action; instead the expert's per-layer RMSNorm is time-conditioned via a `dense` projection of `time_emb` that produces 3 chunks (scale/shift/gate). "time_mlp runs at stack level (separate from action), UNLIKE pi0 where time is concatenated with action." Different time conditioning architecture; same denoising loop topology.

---

## Stage 4: `denoise_step` — flow-matching Euler integrator

SmolVLA uses rectified flow matching (from noise to action) with 10 Euler steps. Time goes 1 → 0 (noise at t=1, action at t=0).

```python
def denoise(initial_noise, vlm_k, vlm_v, state):
    x = initial_noise                               # [B, 50, 32], sampled from N(0, I)
    num_steps = 10
    dt = -1.0 / num_steps                           # -0.1

    for step in range(num_steps):
        t = 1.0 - step * (1.0 / num_steps)          # 1.0, 0.9, ..., 0.1
        velocity = expert_forward(x, t, vlm_k, vlm_v, state)     # [B, 50, 32]
        x = x + velocity * dt                       # Euler step: x_new = x_old - v * (1/N)

    return x    # [B, 50, max_action_dim=32], clamp or slice to env action dim downstream
```

**Critical gotcha (current_session line 11435):** Per-step cos_sim > 0.999 required to survive 10 integration steps. Current ONNX has ~2% per-step velocity error, which compounds catastrophically: per-step cos=0.977 → final cos=-0.24.

**Critical gotcha (modal_pytorch_vs_onnx.py):** Both paths MUST be seeded with SHARED noise for the cos_sim comparison to be meaningful. Without shared noise, cos_sim is dominated by random-noise drift, not export correctness.

**Per-model note:** pi0 uses the same Euler loop; pi0.5 uses AdaRMSNorm per-layer but same outer loop; GR00T uses a DiT expert with action_encoder / action_decoder wrappers (commit `ff9fc3a`) so `raw_action_dim=128` in/out.

---

## Stage 5: action post-processing

```python
# At this point x has shape [B, chunk=50, max_action_dim=32]
actions_raw = x[:, :, :env_action_dim]              # Slice to env dim (LIBERO=7, DROID=8)
# Normalizer un-normalize — must be applied for LIBERO/real robots
actions = unnormalizer(actions_raw)                 # mean, std from postprocessor safetensors
```

**Critical gotcha (current_session bug #9, line 9226):** SmolVLA LIBERO checkpoint ships with `policy_preprocessor_step_5_normalizer_processor.safetensors` (input normalizer — normalizes state) and `policy_postprocessor_step_0_unnormalizer_processor.safetensors` (output unnormalizer — un-normalizes actions). Our early pipeline applied neither. Model expects normalized state → returns normalized actions → LIBERO interprets raw joint values → 0% task success. The stats block shape is `(8,)` for state mean/std (bug #5 — state_dim 8 not 6).

---

## Integration test: the cos_sim ladder (from modal_stage_diff.py)

Canonical per-stage diagnostic (current_session lines 10969, 11089, 11132):
```
Stage 1a (Vision encoder):   cos = 1.0000  ✓ (after AutoModelForImageTextToText unwrap fix)
Stage 1b (Text embedder):    cos = 1.0000  ✓ (after real text_embedder.onnx export)
Stage 1c (State projection): cos = 1.0000  ✓ (after state_proj weight load fix)
Stage 2   (Per-layer vlm_kv): cos = 1.0000 on k, 0.91-1.00 on v  (layer_0_v cos=0.9117 outlier)
Stage 4   (Per-step velocity): cos = 0.977  per-step (~20% norm error)
Full pipeline (shared noise): cos = -0.24   final (composition bug somewhere, likely mask or softmax-fp32)
```

The per-step velocity match is close but the 10-step integration catastrophically amplifies the error. **Self-attention layer 0 matches to cos=1.0000 in isolation** (current_session line 11468); the composition across cross-attention layers is the remaining bug.

---

## References

- **Authoritative PyTorch code:** `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy` (the "copy lerobot code" strategy from current_session line 11574)
- **Our reconstructions (reference):** `src/reflex/_pytorch_backend.py`, `src/reflex/exporters/smolvla_exporter.py`
- **GQA spike report:** `.agents/crank/spike-gqa-result.md` (commit `6fedff3`)
- **Per-stage diff:** `scripts/modal_stage_diff.py`
- **Full vs pipeline diff:** `scripts/modal_pytorch_vs_onnx.py`
- **Key commits:** `1ed46ab` (suffix + action_proj), `74d24c3` (expert GQA layer), `47f3d5d` (16-layer stack), `da237f5` (VLM exploration), `fb9a317` (E2E pipeline), `6fedff3` (GQA/RoPE spike — 32 vs 16 layer discovery), `5869a3e` (vision_components), `d5b6570` (decoder_prefill + orchestrator), `7ed41aa` (vlm_kv dim=320 read from ONNX), `f882dcb` (tokenizer from vlm_model_id, deterministic text fallback), `36d8a40` (real text_embedder.onnx)
