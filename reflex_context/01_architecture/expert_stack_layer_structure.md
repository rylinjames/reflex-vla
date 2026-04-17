# Expert Stack — Per-Layer Structure

**Purpose:** Reference for the 16-layer action expert in SmolVLA. Covers the self-attention vs cross-attention alternation, how cross-attn layers consume `vlm_k`/`vlm_v` from the VLM prefix, the position-ID handling asymmetry between self- and cross-attn, and the subtle point that expert attention operates at **VLM scale** (15 heads × 64 head_dim = 960) while the residual stream is at **expert hidden** (720).

**Source of truth:** `src/reflex/exporters/smolvla_exporter.py::ExpertGQALayer` + `src/reflex/_pytorch_backend.py`. Discovery history in commits `74d24c3` (single layer), `47f3d5d` (16-layer stack), `fb9a317` (E2E with VLM), `7ed41aa` (vlm_kv dim 320 from ONNX shape).

---

## High-level structure

16 layers. `self_attn_every_n_layers = 2` (commit `47f3d5d`): alternating starting with self-attn at layer 0.

| Layer index | Type | Description |
|-------------|------|-------------|
| 0, 2, 4, 6, 8, 10, 12, 14 | **Self-attention** (8 layers) | Expert's chunk attends to itself + the prefix (via position offset) |
| 1, 3, 5, 7, 9, 11, 13, 15 | **Cross-attention** (8 layers) | Expert chunk queries attend to VLM-provided K/V at this layer index |

Each layer structure (pre-norm, like Llama):
```
residual_1 = hidden
hidden = input_layernorm(hidden)                        # DecomposedRMSNorm(eps=1e-6)
hidden = attention(hidden, ...)                         # self OR cross (see below)
hidden = residual_1 + hidden

residual_2 = hidden
hidden = post_attention_layernorm(hidden)               # DecomposedRMSNorm(eps=1e-6)
hidden = mlp(hidden)                                    # SwiGLU: silu(gate) * up → down
hidden = residual_2 + hidden
```

After layer 15:
```
hidden = final_norm(hidden)                             # DecomposedRMSNorm
velocity = action_out_proj(hidden)                      # Linear(720 → max_action_dim=32)
```

---

## Dimensions (the subtle architectural point)

The expert has **two coexisting dimensions**:

| Role | Dimension | Notes |
|------|-----------|-------|
| Residual stream / expert hidden | **720** | `expert_hidden = int(vlm_hidden * 0.75) = int(960 * 0.75) = 720` |
| Attention Q/K/V scale (VLM scale) | **960** | = nq × head_dim = 15 × 64 |
| KV heads scale | **320** | = nkv × head_dim = 5 × 64 (GQA 3:1) |
| head_dim | **64** | Same as SmolLM2 |
| Q heads (nq) | **15** | |
| KV heads (nkv) | **5** | GQA factor nq/nkv = 3 |

**The subtlety:** `q_proj` projects **720 → 960** (up-projects expert hidden to VLM scale for attention), and `o_proj` projects **960 → 720** (back down to expert hidden for residual add). This is how the expert plays in VLM's 960-dim attention space while carrying a smaller residual.

From current_session line 2619: *"k_proj shape mismatch — k_proj is `[320, 720]` not `[960, 720]`. That's because the expert uses GQA: fewer KV heads than Q heads. Q has 15 heads (960/64), K/V have 5 heads (320/64)."*

From current_session line 2660: *"Expert layer architecture: 720 hidden, 15 Q heads, 5 KV heads (GQA), 64 head_dim, 2048 intermediate, 6.27M params per layer. ONNX export: 0.1MB per layer, max_diff=5.36e-07."*

### Per-layer weight shapes

| Tensor | Shape |
|--------|-------|
| `q_proj.weight` | `[960, 720]` — 720 → 960 |
| `k_proj.weight` | `[320, 720]` — 720 → 320 (self-attn; cross-attn projects from VLM 960 instead) |
| `v_proj.weight` | `[320, 720]` — same |
| `o_proj.weight` | `[720, 960]` — 960 → 720 |
| `input_layernorm.weight` | `[720]` |
| `post_attention_layernorm.weight` | `[720]` |
| `gate_proj.weight` | `[intermediate=2048, 720]` |
| `up_proj.weight` | `[2048, 720]` |
| `down_proj.weight` | `[720, 2048]` |

Per-layer params: **6.27M** (commit `74d24c3` log). 16 layers → 99.8M expert total.

---

## Self-attention layer forward

```python
def self_attn_forward(self, hidden_expert, position_ids_expert):
    # hidden_expert: [B, chunk=50, expert_hidden=720]
    B, S, _ = hidden_expert.shape

    residual = hidden_expert
    hidden = self.input_layernorm(hidden_expert)

    # Project to attention scale
    q = hidden @ self.q_proj.weight.T                   # [B, S, 960]
    k = hidden @ self.k_proj.weight.T                   # [B, S, 320]
    v = hidden @ self.v_proj.weight.T                   # [B, S, 320]

    # Reshape for heads
    q = q.reshape(B, S, self.nq, self.head_dim).transpose(1, 2)   # [B, 15, S, 64]
    k = k.reshape(B, S, self.nkv, self.head_dim).transpose(1, 2)  # [B, 5, S, 64]
    v = v.reshape(B, S, self.nkv, self.head_dim).transpose(1, 2)  # [B, 5, S, 64]

    # RoPE on q, k only — NOT v
    q, k = self.rope(q, k, position_ids_expert)

    # GQA: expand k, v to nq heads by repeating each KV head 3 times
    k = k.repeat_interleave(self.nq // self.nkv, dim=1)           # [B, 15, S, 64]
    v = v.repeat_interleave(self.nq // self.nkv, dim=1)           # [B, 15, S, 64]

    # Attention
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, 15, S, S]
    # Softmax upcast to fp32 (HF convention)
    attn = torch.softmax(scores.to(torch.float32), dim=-1).to(hidden.dtype)
    attn_out = attn @ v                                            # [B, 15, S, 64]

    # Merge heads, project back to expert hidden
    attn_out = attn_out.transpose(1, 2).reshape(B, S, self.nq * self.head_dim)  # [B, S, 960]
    attn_out = attn_out @ self.o_proj.weight.T                    # [B, S, 720]

    hidden = residual + attn_out

    # MLP
    residual2 = hidden
    hidden = self.post_attention_layernorm(hidden)
    gate = F.silu(hidden @ self.gate_proj.weight.T)               # [B, S, 2048]
    up = hidden @ self.up_proj.weight.T                           # [B, S, 2048]
    hidden = (gate * up) @ self.down_proj.weight.T                # [B, S, 720]
    hidden = residual2 + hidden

    return hidden
```

### Position IDs — the prefix_offset subtlety

**Critical gotcha (12-bugs table, "prefix_offset for self-attn"):** self-attn layers must use `position_ids = arange(prefix_offset, prefix_offset + chunk_size)` where `prefix_offset = prefix_len`. This is because the expert's chunk is conceptually "continuing" the VLM prefix positionally.

Example: if prefix_len=70 (64 image + 5 text + 1 state):
- Self-attn positions for expert chunk = `[70, 71, 72, ..., 119]` (50 positions)
- These position IDs index into the RoPE cos/sin cache sized `[max_seq_len=512, head_dim]`

If you use `arange(0, 50)` instead, the expert attention sees its chunk as if it were at the start of the sequence — catastrophic for models that rely on absolute positioning.

---

## Cross-attention layer forward

```python
def cross_attn_forward(self, hidden_expert, vlm_k_at_layer, vlm_v_at_layer, position_ids_expert_renorm):
    # hidden_expert: [B, chunk=50, expert_hidden=720]
    # vlm_k_at_layer: [B, prefix_len, vlm_kv_dim=320]  (from decoder_prefill.onnx)
    # vlm_v_at_layer: [B, prefix_len, vlm_kv_dim=320]
    # position_ids_expert_renorm: [B, chunk=50] starting at 0  ← renormalized, NOT offset
    B, S, _ = hidden_expert.shape
    T_prefix = vlm_k_at_layer.shape[1]

    residual = hidden_expert
    hidden = self.input_layernorm(hidden_expert)

    # Expert Q projects from expert_hidden
    q = hidden @ self.q_proj.weight.T                   # [B, S, 960]
    q = q.reshape(B, S, self.nq, self.head_dim).transpose(1, 2)  # [B, 15, S, 64]

    # K, V come from VLM prefill — already at KV scale (320)
    # No k_proj / v_proj here! The VLM decoder already applied them during prefill.
    k = vlm_k_at_layer.reshape(B, T_prefix, self.nkv, self.head_dim).transpose(1, 2)  # [B, 5, T, 64]
    v = vlm_v_at_layer.reshape(B, T_prefix, self.nkv, self.head_dim).transpose(1, 2)  # [B, 5, T, 64]

    # RoPE on Q only (expert gets its own renormalized positions)
    # NO RoPE on K here — VLM K was already rotated during prefill
    q, _ = self.rope(q, q, position_ids_expert_renorm)   # reuse apply_rope helper; ignore second output

    # GQA expand
    k = k.repeat_interleave(self.nq // self.nkv, dim=1)
    v = v.repeat_interleave(self.nq // self.nkv, dim=1)

    # Attention (scores shape [B, 15, S_query=50, T_key=prefix_len])
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    attn = torch.softmax(scores.to(torch.float32), dim=-1).to(hidden.dtype)
    attn_out = attn @ v                                  # [B, 15, 50, 64]

    attn_out = attn_out.transpose(1, 2).reshape(B, S, self.nq * self.head_dim)  # [B, 50, 960]
    attn_out = attn_out @ self.o_proj.weight.T          # [B, 50, 720]

    hidden = residual + attn_out

    # MLP (same as self-attn)
    residual2 = hidden
    hidden = self.post_attention_layernorm(hidden)
    gate = F.silu(hidden @ self.gate_proj.weight.T)
    up = hidden @ self.up_proj.weight.T
    hidden = (gate * up) @ self.down_proj.weight.T
    hidden = residual2 + hidden

    return hidden
```

### Key cross-attn characteristics

1. **No `k_proj` / `v_proj`:** The cross-attn layer accepts K/V directly from the VLM's prefill. The VLM has already applied its own K/V projections at that layer index. The expert's cross-attn `k_proj` / `v_proj` weights **don't exist** for cross-attn layers — only `q_proj` and `o_proj`.
2. **`vlm_kv_dim = 320` not 960:** Per commit `7ed41aa`: *"Read the actual vlm_kv dim from the ONNX input shape (expert.vlm_kv_dim), NOT the VLM hidden_size — they differ (320 vs 960 for SmolVLA). Expert cross-attn projects VLM hidden(960) to KV-dim(320)."* This confused our v1 stub (current_session line 7195: *"Our v1 stub uses vlm_kv_dim=512. The real dimension is 960. SmolLM2 hidden_size is 960. The expert's cross-attention projects 960→720 internally."*). Final resolution: `vlm_kv_dim = 320`, not 512 and not 960. The projection lives in the VLM's decoder layer, consumed as K/V by the expert.
3. **RoPE on Q only, not K:** Cross-attn K was already rotated during VLM prefill at that layer's VLM-scale positions. The expert Q gets a fresh rotation at renormalized positions.
4. **Position IDs are renormalized:** expert Q uses positions `arange(0, 50)` (not offset by `prefix_len`). This matches the intuition that the expert's chunk is a "new" sequence attending TO the prefix, not continuing it.

**Mnemonic:** self-attn offsets positions; cross-attn renormalizes them.

---

## How `vlm_k` and `vlm_v` are shaped

The `decoder_prefill.onnx` emits 16 `layer_k_*` and 16 `layer_v_*` outputs (one per VLM layer). But the expert only has **8 cross-attn layers** (odd indices 1, 3, 5, 7, 9, 11, 13, 15). The orchestrator extracts just those 8 and stacks them.

`vlm_k` as passed to expert: `[B, num_cross_attn=8, prefix_len, 320]`
`vlm_v`: same shape.

Inside the expert, each cross-attn layer `i` indexes `vlm_k[:, cross_layer_index(i), :, :]` where `cross_layer_index` maps {1 → 0, 3 → 1, 5 → 2, ..., 15 → 7}.

**Why only odd layers:** Because the VLM has 16 layers total (SmolVLA uses the first 16 of SmolVLM2-500M's 32), and we alternate self/cross starting from layer 0. The VLM's K/V at odd layer indices (1, 3, ..., 15) is what the expert's 8 cross-attn layers consume.

**Open question:** Whether the pairing is actually layer-1 VLM → expert-layer-1, or some other mapping. Current code pairs them identically (expert cross-attn at index `i` consumes `vlm_k[:, cross_layer_index(i)]`). The stage-diff `ap-YrnHF0WgFXQ2Y7HWlYHPaI` per-layer k/v match was good (cos ≥ 0.91 on v, mostly 1.00) so the pairing likely is correct; the `layer_0_v cos=0.9117` outlier is a numerics question, not a pairing question.

---

## Full forward through 16 layers

```python
def expert_stack_forward(self, noisy_actions, timestep, vlm_k, vlm_v, state_embed, prefix_len):
    # noisy_actions: [B, 50, 32], timestep: [B]
    # vlm_k/vlm_v: [B, 8, prefix_len, 320]
    # state_embed: [B, 720] (or concatenated differently; see suffix encoder)

    # Suffix encoding: actions + time → expert-input
    hidden = self.embed_suffix(noisy_actions, timestep)  # [B, 50, 720]

    B, S, _ = hidden.shape

    # Precompute position IDs
    pos_ids_self = torch.arange(prefix_len, prefix_len + S).unsqueeze(0).expand(B, -1)
    pos_ids_cross = torch.arange(S).unsqueeze(0).expand(B, -1)

    cross_idx = 0
    for layer_idx in range(16):
        if layer_idx % 2 == 0:
            # Self-attention layer
            hidden = self.layers[layer_idx].self_attn_forward(hidden, pos_ids_self)
        else:
            # Cross-attention layer
            hidden = self.layers[layer_idx].cross_attn_forward(
                hidden,
                vlm_k[:, cross_idx],
                vlm_v[:, cross_idx],
                pos_ids_cross,
            )
            cross_idx += 1

    hidden = self.final_norm(hidden)
    velocity = hidden @ self.action_out_proj.weight.T    # [B, 50, 32]

    return velocity
```

---

## Cross-checks: per-layer numerics from modal_stage_diff.py

Healthy stage-diff run `ap-YrnHF0WgFXQ2Y7HWlYHPaI` (after `AutoModelForImageTextToText` unwrap fix):

```
layer_0_k  cos=+1.0000,   layer_0_v cos=+0.9117   ← the persistent outlier
layer_8_k  cos=+0.9997,   layer_8_v cos=+0.9967
layer_15_k cos=+0.9994,   layer_15_v cos=+0.9954
```

**Single-layer isolation test (current_session line 11468):**
> *"Single SELF-attn layer (layer 0) matches to 1e-5 precision, cos=1.0000. The bug is somewhere in COMPOSITION — probably cross-attention layers."*

Interpretation:
- Individual self-attn layers compute identically between PyTorch and ONNX.
- Per-layer VLM K/V match well (cos ≥ 0.91).
- Full pipeline final cos = -0.24.

Therefore the bug is either:
1. **Cross-attention composition** — interaction between cross-attn and the K/V it consumes from VLM.
2. **Attention mask handling** — we don't mask padded prefix positions in expert cross-attn; real SmolVLA does. This was flagged in `02_bugs_fixed/smolvla_inference_bugs.md`.
3. **Softmax fp32 upcast** — we upcast, but may be missing some subtlety around where the downcast happens.
4. **The 10-step integration** amplifies per-step ~2% velocity error into catastrophic final divergence.

Per current_session line 11435: *"The expert_stack ONNX has residual ~2% per-step error that COMPOUNDS catastrophically over 10 Euler steps."*

---

## Why the "attention at VLM scale, residual at expert hidden" design

This architectural quirk — expert_hidden=720 but Q/K/V/attention dims = 960/320 via `q_proj: 720→960` then `o_proj: 960→720` — is inherited from SmolLM2's design for the action expert. Hypothesis: the 720 residual saves MLP compute (intermediate 2048 vs 2560 if expert_hidden=960) while preserving attention expressiveness.

From commit `74d24c3`'s derivation:
```
expert_hidden = int(vlm_hidden * 0.75)           # 720
expert_intermediate = round_to_256(2/3 * 4 * expert_hidden)  # ~2048
```

The `0.75` factor and `2/3 * 4` formula are SmolLM2 / Llama conventions for sizing FFN relative to hidden dim.

---

## Suffix encoder — action + timestep → expert input

From `modal_real_export.py` (commit `1ed46ab`):

```python
def embed_suffix(noisy_actions, timestep):
    # noisy_actions: [B, 50, 32], timestep: scalar or [B]

    # Sinusoidal timestep embedding
    time_emb = create_sinusoidal_pos_embedding(
        timestep,
        dimension=expert_hidden,                # 720
        min_period=4e-3,
        max_period=4.0,
    )
    # [B, 720]

    # Action projection
    action_emb = action_in_proj(noisy_actions)  # Linear(32 → 720)
    # [B, 50, 720]

    # SmolVLA: concatenate along hidden dim, MLP back to 720
    time_emb_expanded = time_emb.unsqueeze(1).expand(-1, 50, -1)  # [B, 50, 720]
    combined = torch.cat([action_emb, time_emb_expanded], dim=-1)  # [B, 50, 1440]
    hidden = action_time_mlp_in(combined)       # Linear(1440 → 720)
    hidden = F.silu(hidden)
    hidden = action_time_mlp_out(hidden)        # Linear(720 → 720)
    # [B, 50, 720] — ready for expert stack
```

Parameter names in the checkpoint:
- `model.action_in_proj` (projection into action space)
- `model.action_time_mlp_in` (1.0M params)
- `model.action_time_mlp_out` (0.5M params)
- `model.action_out_proj` (projection out of action space, applied after final_norm)

---

## References

- `src/reflex/exporters/smolvla_exporter.py::ExpertGQALayer` — production per-layer implementation
- `src/reflex/_pytorch_backend.py` — PyTorch reference for validate round-trip
- `scripts/modal_expert_export.py` — original single-layer exploration (commit `74d24c3`)
- `scripts/modal_full_pipeline.py` — 16-layer stack + E2E (commit `47f3d5d`)
- `scripts/modal_stage_diff.py` — per-stage diagnostic including per-layer k/v
- **Commits of record:** `74d24c3` (single expert layer), `47f3d5d` (16-layer stack with alternating self/cross), `fb9a317` (E2E with VLM), `7ed41aa` (vlm_kv dim from ONNX shape, not config), `6fedff3` (GQA spike — 15Q/5KV/head_dim=64/rope_theta=100000)
- **Numerics:** per-layer max_diff 5.36e-07 (single layer), 4.77e-06 (16-layer stack), 3.81e-06 (full SmolVLA export)
- **Open question:** the `layer_0_v cos=0.9117` outlier in decoder_prefill + the cross-attention composition bug that produces final cos=-0.24 while per-step cos=0.977
