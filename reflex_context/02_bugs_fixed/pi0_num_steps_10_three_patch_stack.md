# pi0 num_steps=10 cos=0.977 → cos=1.000 — the three-patch stack

**Date fixed:** 2026-04-19. **Commit:** `bac658a`. **Reproducer:** `modal run scripts/modal_pi0_monolithic_export.py --parity --num-steps 10`.

The hardest bug squashed in v0.2. This doc explains why each of the three patches is load-bearing — removing any one drops cos back to 0.977 or breaks export entirely.

---

## Symptom

pi0 monolithic ONNX at num_steps=10 matched PyTorch `sample_actions(num_steps=10)` at cos=0.977, max_abs=1.31e-01 — an approximation, shipped as a known v0.2 limitation. The v0.2 workaround was a `create_causal_mask → None` shim: it let the export complete but silently skipped PaliGemma's prefix-pad masking. SmolVLA was unaffected (SmolLM2's attention path doesn't need that mask for correctness) but pi0's 4-camera + state + language prefix became partially masked, costing ~2% cos per step.

Goal: fix the shim without losing the export, reach cos=1.0 at machine precision.

## Why three patches, not one

Three different invariants break under `torch.export` FakeTensor tracing of pi0's 10-step unrolled Euler loop. Each one, fixed alone, leaves the others broken. They interact: fix (1) and (3) without (2) → cache grows 784→835→886 across iterations and mask can't be built. Fix (2) and (3) without (1) → the cat op still loses the suffix dim. Fix (1) and (2) without (3) → mask dim mismatches attention scores.

### Patch 1 — F.pad instead of torch.cat for the block-causal mask

**File:** `scripts/modal_pi0_monolithic_export.py`, patched `denoise_step`.

**Before:**
```python
full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
```

**After:**
```python
prefix_allowed = F.pad(prefix_pad_2d_masks, (0, suffix_len), value=True)
suffix_allowed = F.pad(suffix_att_2d_masks, (prefix_len, 0), value=True)
full_att_2d_masks = prefix_allowed & suffix_allowed
```

**Why.** `torch.cat` along dim=2 under FakeTensor tracing loses the suffix dim. The mask arrives at attention as `[1,1,51,835]` (prefix only) instead of `[1,1,51,886]` (prefix + 51 suffix). `F.pad` has a concrete output size at every call — pads with True (allowed), then ANDs. Both operations preserve shape under tracing.

**Diagnostic that revealed it:** instrumented `DebugAttentionMaskCapture` hook that printed the mask shape at the attention entry point. Observed `[1,1,51,835]` where `[1,1,51,886]` was expected. Tracing the graph showed the cat's output shape was treated as `[1,1,51,835]` (prefix dim only) by the ONNX exporter.

### Patch 2 — Freeze DynamicLayer.update during denoise phase

**File:** `scripts/modal_pi0_monolithic_export.py`.

**Install:**
```python
from transformers.cache_utils import DynamicLayer as _DL
_orig_layer_update = _DL.update
_denoise_phase = [False]

def _frozen_layer_update(self, key_states, value_states, cache_kwargs=None):
    if _denoise_phase[0] and getattr(self, "is_initialized", False):
        past_k = self.keys
        past_v = self.values
        if past_k is not None and past_v is not None:
            new_k = torch.cat([past_k, key_states], dim=-2)
            new_v = torch.cat([past_v, value_states], dim=-2)
            return new_k, new_v
    return _orig_layer_update(self, key_states, value_states, cache_kwargs)

_DL.update = _frozen_layer_update
```

**Wrap** `PI0Pytorch.denoise_step` to flip the flag to `True` during its execution, `False` elsewhere.

**Why.** Transformers' attention layer unconditionally calls `past_kv.update(k, v, layer_idx)` to get the concatenation of past + current, and `DynamicCache.update()` APPENDS to `self.keys`/`self.values` regardless of `use_cache=False`. Under `torch.export`, the cache is a traced Python object shared across the 10 unrolled iterations. Iteration 1 appends suffix K/V (784 → 835). Iteration 2 appends again (835 → 886). Iteration N → attention sees past of size `(orig + N · suffix_len)`. Eager PyTorch doesn't have this problem because each call creates a fresh cache or `use_cache=False` is honored; tracing flattens the whole loop into one graph with shared state.

The patched `update` returns `cat(past, current)` **without appending**. Prefix forward runs with the flag off (normal cache build). Denoise loop runs with the flag on (cache frozen at prefix size). Semantic equivalence to eager PyTorch at num_steps=10.

**Diagnostic that revealed it:** a `print(past_key_values.get_seq_length())` at the top of each patched denoise_step. Observed 784 → 835 → 886 across three iterations. Should have been 784 throughout (prefix is fixed).

### Patch 3 — Use past_kv.get_seq_length() for mask K dim, not prefix_pad_masks.shape[1]

**File:** `scripts/modal_pi0_monolithic_export.py`, patched `denoise_step`.

```python
cached_prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
prefix_len = max(cached_prefix_len, prefix_pad_masks.shape[1])
```

**Why.** Under tracing, `prefix_pad_masks.shape[1]` is the *static* shape of the pad-mask tensor passed in (784) — but the actual K dimension at attention is whatever `past_kv` has (784 or 835 or 886 pre-patch-2). Using the pad-mask shape for mask assembly → mask's K is smaller than attention scores' K → broadcast failure `expand: 835 -> 886`.

Even with patch 2 in place (cache frozen at 784), this patch is still needed because the original code reads `prefix_pad_masks.shape[1]` as the K length; the safer invariant is "use the source of truth for K, which is the cache itself." Defense against any future refactor that changes where K is allocated.

---

## Why patch 2 is the hardest

Patches 1 and 3 are local rewrites of one `denoise_step` function. Patch 2 is a global monkey-patch of `transformers.cache_utils.DynamicLayer.update` — a class method that ships inside the transformers library, called from deep inside attention layers we don't own. It has to:

- Know when we're in denoise phase vs prefix phase (a flag)
- Preserve the exact return signature (`(keys, values)` tuple)
- Not break transformers 5.3's internal state machine (hence the `is_initialized` check)
- Not leak across tests (flag is flipped via try/finally in the wrapped denoise_step)

The reason it's necessary (not a hack): canonical eager PyTorch at `use_cache=False` **does not mutate the cache**. Under `torch.export` tracing, `use_cache=False` is erased and the attention layer always calls `update`. Patch 2 restores the eager semantics under tracing.

## What wasn't the fix

For the record — things that looked promising and turned out to be red herrings:

1. **Swapping the ONNX opset.** Tried 17, 19, 23. Opset doesn't change trace-level semantics; this is a Python-level mutation bug, not an op translation bug.
2. **`torch.export(strict=True)`.** Got strict mode working at one point; same cos=0.977. The mutation happens before `torch.export` sees the graph.
3. **Dynamo-based `torch.onnx.export(dynamo=True)`.** Fails on Gemma with a different bug; not a viable path until PyTorch fixes the 3 upstream blockers (160761, 170172, 172903).
4. **Re-exporting at different dummy input shapes.** Thought maybe the issue was shape-specialization. Same 0.977 at every variant.
5. **Forcing `_attn_implementation = "eager"`** on the Gemma expert. Already set correctly pre-session; irrelevant to the bug.

## Numbers before / after

| Measurement | Before (v0.2 `ccm_shim → None`) | After (3-patch stack) |
|---|---|---|
| first-action cos | 0.977 | +1.000000 |
| first-action max_abs | 1.31e-01 | 2.09e-07 |
| full-chunk cos | 0.977 | +1.000000 |
| full-chunk max_abs | 1.4e-01 (approx) | 4.17e-07 |

Reduction in max_abs: ~6 orders of magnitude.

## Generalization — does pi0.5 need this too

Yes. pi0.5 has the same PaliGemma backbone, same DynamicCache, same `denoise_step` structure. The pi0.5 port (commit `c604962`) applies the 3-patch stack verbatim with two signature changes (no `state` arg in `denoise_step`; no `state` tensor in wrapper). First-try PASS at cos=+1.000000, max_abs=2.38e-07.

GR00T does NOT need this. It's DDPM (not flow matching), DiT (not decoder-only Gemma), no DynamicCache, no prefix-pad mask. Plain `torch.onnx.export(opset=19)` traces cleanly. See `01_architecture/gr00t_ddpm_dit_vs_flow_matching.md`.

SmolVLA does NOT need the full stack — its SmolLM2 backbone's attention is structured differently enough that the `create_causal_mask → None` shim has no semantic impact. SmolVLA stayed at cos=1.0 through v0.2 with just the shim.

## If you remove a patch

- **Remove patch 1 only:** cos drops back to 0.977 (same as v0.2). Export succeeds.
- **Remove patch 2 only:** cos drops to ~0.85 or fails during parity with shape mismatch. Export often fails at iteration 3+ with "expand 835 → 886".
- **Remove patch 3 only:** with patches 1 + 2 in place, often still works at cos=1.0 because cache is frozen at 784 and `prefix_pad_masks.shape[1]` equals 784 in happy path. Remove and hope nothing else changes K sizing → future brittleness.

## References

- Commit `bac658a` — the fix
- Commit `0c09f80` — prior investigation (concluded the fix was v0.3)
- Commit `21043a6` — partial-progress revert
- `scripts/modal_pi0_monolithic_export.py` lines 200–310 — the patches as shipped
- `scripts/modal_pi05_monolithic_export.py` — same stack, pi0.5 signature
- `01_architecture/pi0_monolithic_wrap_pattern.md` — wrapper-level overview
- `01_architecture/gr00t_ddpm_dit_vs_flow_matching.md` — why GR00T doesn't need this
