# Pi0 does NOT need DecomposedRMSNorm swap

**Discovered:** 2026-04-17 (this session, building multi-model-native-parity)

## Claim

Unlike SmolVLA (which uses `nn.RMSNorm` / `LlamaRMSNorm` and needs `DecomposedRMSNorm` swap for TRT opset 23 compat), **pi0's `PiGemmaRMSNorm` is already written in elementwise form** and does not need decomposition.

Swapping it would be wasted work.

## Evidence

Source: `.venv/lib/python3.13/site-packages/lerobot/policies/pi_gemma.py:85-128`

```python
class PiGemmaRMSNorm(nn.Module):
    """Adaptive RMSNorm for PI Gemma (AdaRMS).
    When cond_dim is set, uses cond to modulate scale/shift/gate;
    otherwise behaves like standard GemmaRMSNorm.
    forward(x, cond=None) returns (output, gate) for use with _gated_residual.
    """

    def __init__(self, dim, eps=1e-6, cond_dim=None):
        ...
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype
        normed = self._norm(x)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None
        # AdaRMS (time-conditioned) path:
        modulation = self.dense(cond)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)
```

All operations (`torch.mean`, `torch.square`, `torch.rsqrt`, `torch.add`, `torch.mul`, `nn.Linear`, `chunk`) are ONNX-native and TensorRT-compatible. There is no `torch.nn.functional.rms_norm` or opset-23 `RMSNormalization` op in the graph.

## Why this differs from SmolVLA

SmolVLA uses `torch.nn.RMSNorm` / `transformers.LlamaRMSNorm` / `SmolLM2RMSNorm`. These implementations call optimized CUDA kernels that — when traced to ONNX — emit the opset-23 `RMSNormalization` op. TensorRT's ONNX parser doesn't support it yet (NVIDIA TRT issue #4639). So we swap them with `DecomposedRMSNorm` before export.

Pi0's authors (Physical Intelligence) **hand-wrote the elementwise form from the start** (see the docstring reference to openpi's `gemma.py`). This choice was likely made so openpi could JIT-compile to JAX/TPU without relying on framework-specific fused kernels. The TRT-compat benefit is free.

## What pi0 export DOES still need

The RMSNorm situation is clean. The rest of the pi0 export pipeline has its own challenges:

1. **RoPE** — `GemmaRotaryEmbedding` still emits ops that need decomposition for TRT. The pattern matches what we already do for SmolVLA (see `decompose.py` `_DecomposedRoPE`).
2. **GQA attention** — 16 Q heads, 2 KV heads, head_dim=128. Exporting cleanly requires flattening the grouped-query repeat_kv pattern.
3. **Expert block** — 18 layers, hidden=1024. Separate from the backbone language model.
4. **Flow matching loop** — Same 10-step Euler integration as SmolVLA. The denoise step must be exportable as a single callable.
5. **PaliGemma vision encoder** — SigLIP with 1152-dim hidden, 27 layers. Similar to SmolVLA's SigLIP but different dims.
6. **`_gated_residual`** — custom function called by `_PiGemmaDecoderLayerBase` that applies the `gate` from the tuple return. Must be preserved or inlined in the export graph.

## Second gotcha: tuple return

`PiGemmaRMSNorm.forward` returns `(output, gate)` — a tuple — even in the non-adaptive path (where `gate = None`). This means:

- You CANNOT naively swap it with a class that returns only a tensor.
- Any wrapper or replacement module must preserve the tuple-return shape.
- The caller `_PiGemmaDecoderLayerBase` unpacks `(hidden_states, gate)` and passes `gate` to `_gated_residual`.

If we ever DO decide to swap `PiGemmaRMSNorm` (e.g., for performance optimization), the replacement must return `(tensor, None | tensor)` and implement both the non-adaptive and adaptive forward paths.

## Third gotcha: AdaRMS mode is different from pi0.5's AdaRMSNorm

- **pi0's `PiGemmaRMSNorm(cond_dim=...)`** — adaptive mode optional, controlled at construction. Uses `dense.weight` shape `[3 * dim, cond_dim]`.
- **pi0.5's AdaRMSNorm** — a distinct module type, ALWAYS adaptive. Decomposed into our existing `DecomposedAdaRMSNorm` (`decompose.py:32`).

Do NOT assume `DecomposedAdaRMSNorm` can substitute for the adaptive path of `PiGemmaRMSNorm`. Verify shapes and the gate-return semantics first.

## Practical implications for GOALS.yaml

- `multi-model-native-parity` (weight 9) description must NOT say "same DecomposedRMSNorm-swap pattern" — it doesn't apply to pi0.
- `pi0-onnx-parity` (weight 9, added 2026-04-17) tracks end-to-end ONNX parity verification; RMSNorm is already exportable so focus shifts to attention, RoPE, expert, flow matching loop, and PaliGemma vision encoder.
- `swap_rmsnorm_variants` helper in `src/reflex/decompose.py` will find 0 layers matching on pi0 — **this is correct and expected**, not a bug.

## Related artifacts

- `src/reflex/exporters/pi0_exporter.py` — 584-line pi0 ONNX exporter, pre-dates this finding.
- `src/reflex/decompose.py:85` — `swap_rmsnorm_variants` helper; pi0 returns count=0 by design.
- `scripts/local_pi0_rmsnorm_swap_diff.py` — the test that surfaced this (trivially passed with swap count=0).
- `scripts/local_pi0_inspect_norms.py` — the introspection script that found `PiGemmaRMSNorm` is the actual class name (74 instances, vs `GemmaRMSNorm` which is 0 instances).
- `reflex_context/measured_numbers.md` — pi0 ONNX parity still in Unmeasured as of 2026-04-17.
