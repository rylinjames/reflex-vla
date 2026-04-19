# pi0 monolithic ONNX wrap pattern (Path C)

**Discovered:** 2026-04-18 via deep research (agent synthesis of openpi_on_thor, Tacoin, Dexmal).

**Why this doc exists:** our first two attempts at pi0 ONNX export (custom ExpertStack + concat prefix_kv; Optimum with past_kv as tensor inputs) both hit fundamental blockers. Every production pi0/pi0.5 ONNX in the open-source world uses the same third pattern — monolithic wrap. Writing this down so next session doesn't re-discover.

## The pattern

Wrap pi0's `sample_actions` in a thin `nn.Module` whose `forward(images, img_masks, lang_tokens, lang_masks, state, noise)` calls the full policy forward end-to-end. `past_key_values` stays as a Python variable **inside** the traced graph — it never becomes an ONNX input. The 10-step Euler flow-matching loop is **unrolled** at trace time.

```python
class Pi0MonolithicWrapper(nn.Module):
    def __init__(self, pi0_model, num_steps=10):
        super().__init__()
        self.model = pi0_model
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        return self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=self.num_steps,
        )

torch.onnx.export(
    wrapper, dummy_inputs, onnx_path,
    dynamo=False,           # MANDATORY — dynamo broken for Gemma in 2026
    opset_version=19,       # NOT 23; TRT parser compat
    do_constant_folding=True,
    input_names=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
    output_names=["actions"],
)
```

## Why this works

1. **`dynamo=False` legacy tracer** traces Python-level control flow (loops, conditionals, cache ops) into tensor ops. DynamicCache.update → `torch.cat(past_k, new_k)` nodes baked into the graph.

2. **past_kv never crosses the ONNX boundary**, so we never trip the 3 unfixed PyTorch bugs:
   - pytorch#160761 — Gemma3 vmap in masking_utils
   - pytorch#170172 — dynamo batch=1 hardcode
   - pytorch#172903 — DynamicCache "NameError: 'L' not defined"

3. **Denoise loop unrolls** at `num_steps=10` at trace time — inspecting the graph shows 10× expert forward calls with growing attention matrices.

4. **Uses pi0's own tested forward code.** We inherit every correctness property (normalizer, attention masks, position IDs, state projection, block-causal pattern). No re-implementation bugs.

## Required Gemma source patches

Per openpi_on_thor's `apply_gemma_fixes.py`, two patches are needed (reimplement, don't copy — Thor bundle has no license):

### 1. `GemmaRMSNorm.extra_repr` hasattr guard

Issue: during trace, `extra_repr` may be called on a partially-initialized module → `AttributeError`. 

Fix: guard with `hasattr(self, 'weight')` before accessing.

### 2. `GemmaAttention.forward` explicit dim reshape

Issue: `hidden_shape = (*input_shape, -1, self.head_dim)` uses `-1` which makes that dim dynamic → breaks TRT FP4 block quantization.

Fix: replace `-1` with explicit `num_attention_heads` (for q_proj reshape) or `num_key_value_heads` (for k/v_proj reshape).

Both patches are monkey-patches; no source edits needed to transformers.

## Pros

- **Single ONNX file** — simpler deployment, fewer engine builds on Jetson
- **Proven working** — Tacoin, openpi_on_thor, Dexmal all produce pi0/pi0.5 ONNX this way
- **Inherits pi0's correctness** — no re-implementation bugs from Path A
- **No DynamicCache battles** — past_kv lives in Python vars, not as ONNX inputs
- **Low dev time** — ~50-100 lines of wrapper code + 2 patches

## Cons

- **~14GB monolithic ONNX** — single file with external data. Fine for production but debugging is harder (can't diff per-stage)
- **Fixed num_steps at export time** — changing denoise steps requires re-export. Acceptable for pi0 (10 steps canonical)
- **Larger TRT engine build time** — one big graph vs 5 smaller ones
- **License care** — Thor bundle has no license file. Pattern is public knowledge; implementation must be clean-room

## Trade-off vs our 5-file decomposition

Our existing decomposition (vision_encoder, multi_modal_projector, text_embedder, decoder_prefill, expert_stack) is still valuable:
- Per-stage parity diffing (we used this heavily today)
- Runtime flexibility (can replace individual stages without re-exporting)
- Smaller per-file sizes for debugging

Keep both. Use monolithic for **deployment** (production MVP), use 5-file for **debugging + research**.

## What the ONNX graph looks like (from openpi_on_thor inspection)

Single graph with ~10× unrolled denoise steps. Attention concat ops for past_kv extension baked in as `Concat` nodes. ~14GB total with external data (Gemma-2b backbone weights are the bulk).

## Ship status (2026-04-19 — UPDATE: all four VLAs verified)

**All four major open VLAs verified at cos=+1.000000 machine precision.** The 835→886 bug documented in the earlier version of this doc was resolved 2026-04-19 via the **three-patch stack** (see `02_bugs_fixed/pi0_num_steps_10_three_patch_stack.md`). Historical context below retained for posterity; current status reflects the April 19 fix.

| Model | Reproducer | cos | max_abs |
|---|---|---|---|
| SmolVLA num_steps=10 | `modal run scripts/modal_smolvla_monolithic_export.py --parity --num-steps 10` | +1.000000 | 5.96e-07 |
| pi0 num_steps=10 | `modal run scripts/modal_pi0_monolithic_export.py --parity --num-steps 10` | +1.000000 | 2.09e-07 |
| pi0.5 num_steps=10 | `modal run scripts/modal_pi05_monolithic_export.py --parity --num-steps 10` | +1.000000 | 2.38e-07 |
| GR00T N1.6 (per-step) | `modal run scripts/modal_gr00t_monolithic_export.py --parity` | +1.000000 | 8.34e-07 |
| GR00T N1.6 (4-step loop) | `modal run scripts/modal_gr00t_monolithic_export.py --loop` | +1.000000 | 4.77e-07 |

### Per-model deltas from this base pattern

**SmolVLA.** Same monolithic wrap pattern, `SmolVLAPolicy.model.sample_actions(num_steps=10)`. SmolLM2 backbone doesn't need the 3-patch stack; just the SmolVLM2 `torch.where` dtype fix + post-export Cast insertion. Simpler than pi0.

**pi0.** Full 3-patch stack (F.pad mask + frozen DynamicLayer.update + past_kv.get_seq_length). Wrapper signature: `(images, img_masks, lang_tokens, lang_masks, state, noise)` — 6 inputs.

**pi0.5.** Same 3-patch stack ported verbatim. **Wrapper signature drops the `state` arg** — pi0.5 tokenizes state into language tokens upstream, so `PI05Pytorch.denoise_step` is `(prefix_pad_masks, past_key_values, x_t, timestep)` — 4 args vs pi0's 5. Wrapper `forward(images, img_masks, lang_tokens, lang_masks, noise)` — 5 inputs.

**GR00T N1.6.** **Does not use this monolithic wrap pattern at all.** GR00T is DDPM DiT (not flow-matching decoder-only), no DynamicCache, no prefix-pad mask → plain `torch.onnx.export(opset=19)` on `GR00TFullStack` traces cleanly. See `01_architecture/gr00t_ddpm_dit_vs_flow_matching.md` for why.

### Three-patch stack (required for pi0 + pi0.5)

Summary — full detail in the dedicated bug report:

1. **F.pad instead of torch.cat** for block-causal mask assembly (cat loses suffix dim under FakeTensor)
2. **Freeze `DynamicLayer.update` during denoise phase** (prevents 784→835→886 cache growth across iterations)
3. **Use `past_kv.get_seq_length()`** not `prefix_pad_masks.shape[1]` (the two diverge under tracing)

The patches only work together. Removing any one drops cos back to 0.977 or breaks the export.

## Historical context (pre-2026-04-19) — how we got here

The original v0.2 pi0 num_steps=10 export used a `create_causal_mask → None` shim to dodge the 835→886 shape-expand bug. This let the export complete but silently skipped PaliGemma's prefix-pad masking, costing ~2% cos per step. Shipped as a v0.2 approximation with an honest disclaimer. SmolVLA was unaffected (SmolLM2's attention path doesn't use that mask for correctness).

The three alternative paths considered in the original version of this doc turned out to have a better fourth option: keep the monolithic wrap, but patch around the three interacting trace bugs. That's what `bac658a` landed.

## Follow-up goals (tracked)

- ~~**pi0-onnx-parity-multistep**~~ — ✅ SOLVED 2026-04-19 via three-patch stack
- **GR00T VLM conditioning (Eagle VLM backbone export)** — currently zero-stubbed, full multimodal control v0.3
- **Distribution** — publish ONNX artifacts to HF Hub once customer demand materializes

## Related

- `02_bugs_fixed/pi0_num_steps_10_three_patch_stack.md` — the patches, each explained with root cause and diagnostic
- `01_architecture/gr00t_ddpm_dit_vs_flow_matching.md` — why GR00T doesn't need this pattern at all
- `05_sessions/2026-04-19_all_four_vlas.md` — session narrative of landing all four VLAs
- `03_research/pi0_onnx_importable_sources.md` — Tier-1 source analysis (Thor, Tacoin, GR00T) — historical
- `scripts/modal_pi0_monolithic_export.py` — the shipped pi0 exporter with patches
- `scripts/modal_pi05_monolithic_export.py` — the shipped pi0.5 exporter (same stack, 4-arg signature)
- `scripts/modal_gr00t_monolithic_export.py` — the shipped GR00T exporter (plain torch.onnx.export)
