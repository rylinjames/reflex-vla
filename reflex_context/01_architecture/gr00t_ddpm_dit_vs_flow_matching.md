# GR00T N1.6 — DDPM DiT vs flow-matching decoders

**Why this doc exists.** GR00T is the fourth VLA we export, and unlike SmolVLA / pi0 / pi0.5 it needs **zero** torch.export patches. Plain `torch.onnx.export(opset=19)` traces the whole stack cleanly, first try. This doc explains why — because when the next diffusion-style VLA lands, knowing which category it falls into determines whether you're in for a 30-minute export or a 3-hour patch hunt.

---

## Two worlds

### World A: flow-matching decoders (pi0, pi0.5, SmolVLA)

Architecture:
- **Backbone:** PaliGemma (pi0, pi0.5) or SmolLM2 (SmolVLA) — a vanilla decoder-only LLM
- **Action head:** a second Gemma-like expert ("action expert") that shares attention with the backbone via a DynamicCache-backed block-causal mask
- **Integration:** flow matching — the model learns a velocity field; at inference, take N (=10 canonical) Euler steps from noise to action
- **Export shape:** the **full** `sample_actions(num_steps=N)` is wrapped and traced; the 10-step loop unrolls into the graph

What this means for tracing:
- DynamicCache lives in Python across iterations → sharing state across 10 unrolled attention forward passes
- Attention masks are re-assembled every step from (prefix_pad_masks, suffix_pad_masks) via `torch.cat` → cat loses dim under FakeTensor
- `use_cache=False` is erased under `torch.export` → `DynamicLayer.update` mutates the cache anyway → grows across iterations
- Three patches required to make it trace cleanly (see `02_bugs_fixed/pi0_num_steps_10_three_patch_stack.md`)

### World B: DDPM DiT (GR00T N1.6)

Architecture:
- **VLM conditioning:** Eagle-2-HG VLM (NVIDIA's) produces KV features — **separately** from the action head
- **Action head:** a 32-layer DiT (Diffusion Transformer) with AdaLN modulation
- **Integration:** DDPM-style discrete-timestep diffusion — the model learns a noise-prediction / velocity field at each timestep; at inference, 4 canonical DDIM steps
- **Export shape:** the **per-step velocity function** is wrapped and traced; the 4-step loop lives in `reflex serve` (Python host-loop)

What this means for tracing:
- No DynamicCache — VLM conditioning is cross-attn over *fixed* KV features, not growing past_kv
- No block-causal mask assembly inside the action head — cross-attn queries action tokens over a fixed-size KV projection
- No `torch.cat` across iterations to lose dims under FakeTensor
- AdaLN is just `(1 + scale) * LayerNorm(x) + shift` — trivially traceable
- Plain `torch.onnx.export(opset=19)` traces cleanly

---

## Why the DiT architecture is easier

GR00T's DiT has three properties that make export painless:

1. **No causal / block-causal masking.** DiT attention is bidirectional over a fixed-length action-token set. No mask = no cat-loses-dim problem.
2. **No growing KV cache.** Cross-attention reads from a precomputed VLM KV tensor that's fixed per-denoise-step invocation. The KV length is known at trace time; no DynamicCache.
3. **Alternating cross-attn / self-attn blocks, but both with static K length.** Even blocks do cross-attn over VLM features (length N_vlm). Odd blocks do self-attn over action tokens (length N_act = chunk_size = 50). Both lengths are static at trace time.

Result: the full stack `(action_encoder → 32 DiT blocks → action_decoder)` is a pure data-flow graph with no stateful mutation, no dynamic control flow, no cache side effects. `torch.export` and `torch.onnx.export` both handle it trivially.

---

## GR00T's AdaLN — 2-chunk, not 3-chunk

A subtle difference from pi0.5 (which also uses AdaLN-style modulation): GR00T's AdaLN is **2-chunk**:

```python
# From src/reflex/exporters/gr00t_exporter.py, GR00TDiTBlock
scale, shift = norm1_linear(time_emb).chunk(2, dim=-1)
y = (1 + scale) * layer_norm(x) + shift
```

pi0.5's AdaRMSNorm has **3-chunk** modulation (scale + shift + gate) applied inside the block with a residual gate. Neither matters for export — both chunk via `chunk(2, dim=-1)` or `chunk(3, dim=-1)` which traces as a static split. The only consequence is which weight tensor to load from the checkpoint (`norm1_linear`'s output is 2×hidden for GR00T vs 3×hidden for pi0.5's AdaRMSNorm).

---

## What's exported vs what's in the runtime

GR00T's exported ONNX is **per-step velocity**, not the full 4-step denoise loop. The decision:

| Choice | Pro | Con |
|---|---|---|
| Export per-step, loop in runtime | Simpler graph, smaller ONNX (4.4GB), runtime can change num_steps without re-export | One more round-trip through the runtime per denoise step |
| Export full 4-step loop unrolled | Single ONNX call per inference, fixed num_steps baked in | Larger ONNX (~17GB estimated), re-export to change num_steps, loses the pattern symmetry with pi0/pi0.5 which DO bake the loop |

We chose per-step because:
- GR00T's DDPM denoise loop is trivial (uniform t schedule, simple x_{t-1} = x_t + dt·velocity) — implementing in Python is 5 lines
- Per-step matches Isaac-GR00T's own reference implementation, so customers doing side-by-side comparisons see identical runtime behavior
- num_steps is a hyperparameter customers may want to sweep (2, 4, 6, 8) without re-exporting

**Verified parity at both levels:**
- Single-step ONNX vs PyTorch `GR00TFullStack.forward`: cos=+1.000000, max_abs=8.34e-07
- 4-step denoise loop (Python loop over ONNX) vs same loop (Python loop over PyTorch): cos=+1.000000, max_abs=4.77e-07

The per-step machine precision composes to end-of-chunk machine precision across the 4-step integration — empirically confirmed, not just theorized.

---

## The zero-VLM-KV placeholder

The current GR00T export wires a **zero tensor** as the VLM-KV input to the cross-attn blocks. This is fine for numerical parity testing (PyTorch and ONNX both see zeros, so the comparison is valid) but it means the exported ONNX is not yet a real multi-modal controller — without real VLM conditioning, actions are purely a function of the initial noise.

Same convention pi0 and SmolVLA use for their prefix=None exports. The real multi-modal story needs:

1. Export the Eagle-2-HG VLM backbone as a separate ONNX (vision + text → KV features)
2. Chain: VLM ONNX produces KV → feed into DiT ONNX per denoise step

**Tracked for v0.3.** The DiT + AdaLN stack is already machine precision — the expansion is purely "wire up the VLM input, verify parity holds with real conditioning."

---

## Patch-cost comparison

| Model | torch.export patches needed | Export time (Modal A100) | Monolithic ONNX size |
|---|---|---|---|
| SmolVLA | `torch.where` dtype fix + Cast post-pass | ~4 min | 1.6 GB |
| pi0 | 3-patch stack (F.pad mask + frozen DynamicLayer + past_kv.get_seq_length) | ~7 min | 12.5 GB |
| pi0.5 | 3-patch stack (same as pi0, 4-arg signature) | ~8 min | 13 GB |
| GR00T N1.6 | none | ~3 min | 4.4 GB |

GR00T's simplicity is largely an architectural gift — NVIDIA chose DiT over decoder-only precisely because DiT parallelizes better on their hardware. That architectural choice also happens to make ONNX export trivial.

---

## When the next diffusion VLA shows up

Heuristic questions to ask before attempting export:

1. **Is the action head decoder-only with block-causal attention?** If yes → expect the 3-patch stack.
2. **Does the action head share a DynamicCache with the backbone?** If yes → definitely the 3-patch stack.
3. **Is the action head a standalone transformer (DiT, encoder-only, etc.) that reads from a precomputed KV tensor?** If yes → expect clean export.
4. **Is inference flow matching (Euler loop over velocity field) or DDPM (loop over noise predictor)?** Flow matching typically unrolls the loop into the ONNX (harder); DDPM typically exports per-step + runtime loops it (easier).

If you can't answer these without reading the code: spend 30 min reading `sample_actions` / `get_action` first. Catching the category upfront saves hours.

---

## References

- `src/reflex/exporters/gr00t_exporter.py` — the actual exporter (757 lines, pre-session)
- `scripts/modal_gr00t_monolithic_export.py` — Modal export + parity + loop-parity
- `02_bugs_fixed/pi0_num_steps_10_three_patch_stack.md` — what the 3-patch stack does (and why GR00T doesn't need it)
- `measured_numbers.md` — Verified rows for GR00T single-step + 4-step loop
- `01_architecture/pi0_monolithic_wrap_pattern.md` — flow-matching wrap pattern
