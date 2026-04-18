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

## Related

- `reflex_context/03_research/pi0_onnx_importable_sources.md` — Tier-1 source analysis (Thor, Tacoin, GR00T)
- `reflex_context/03_research/pi0_empirical_derisk_findings.md` — the Apr-17 component-level Optimum exports
- `src/reflex/exporters/pi0_prefix_exporter.py` — 5-file decomposition (still valuable for debug)
- `scripts/export_pi0_monolithic.py` — the Path C export script (this pattern)
