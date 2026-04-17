# Direct `torch.onnx.export(dynamo=True)` Viability

Can we replace our hand-decomposed exporter (RMSNorm + RoPE + ExpertGQALayer + ExpertStack written from scratch in `src/reflex/decompose.py` and `src/reflex/exporters/*`) with a direct call to `torch.onnx.export(..., dynamo=True)` on the real lerobot classes?

**Summary recommendation:** **No — not wholesale. But yes for most of the graph.**

The correct posture is a **hybrid**: copy lerobot's `SmolVLAPolicy.sample_actions` + `embed_prefix` + `embed_suffix` + `forward_cross_attn_layer` into `src/reflex/runtime/smolvla_native.py`, swap only `RMSNorm → DecomposedRMSNorm`, and let `torch.onnx.export(dynamo=True)` handle attention, RoPE, GQA, MLP, and everything else.

---

## The TensorRT opset 23 RMSNorm gap (NVIDIA TRT issue #4639)

Source: `current_session.md` line 11574.

> "TRT's ONNX parser doesn't support opset 23 `RMSNormalization` op yet (issue #4639). Modern torch export emits this op when it sees `nn.RMSNorm`. So if we use `torch.onnx.export` with default settings, RMSNorm won't compile on Jetson."

- **NVIDIA TensorRT GitHub issue #4639** tracks this gap. As of the mining window (through 2026-04-16), no fix ships.
- PyTorch's dynamo exporter emits opset 23 `RMSNormalization` for any `nn.RMSNorm`. Any VLA using RMSNorm (SmolVLA, pi0, pi0.5, GR00T AdaRMSNorm) breaks at TRT compile time on opset 23.
- **Implication:** TRT opset-23 RMSNorm support is the single hardest dependency in the exporter. Until NVIDIA closes #4639, we must continue emitting decomposed RMSNorm.
- Our `DecomposedRMSNorm` in `src/reflex/decompose.py` implements `variance = x.to(float32).pow(2).mean(-1, keepdim=True); x * rsqrt(variance + eps) * weight` — the upcast-to-fp32 is load-bearing for numerical parity (sessions_md line 20).

---

## TensorRT SigLIP fp16 accuracy issues (NVIDIA TRT issues #3908 and #4373)

Source: `current_session.md` line 11574.

> "TRT has known fp16 accuracy issues on SigLIP (issues #3908, #4373) — independent of how you export. This is an NVIDIA bug that bites regardless of our decomposition."

- **#3908** and **#4373** are both active NVIDIA TensorRT GitHub issues covering SigLIP FP16 accuracy loss.
- Not fixable in Reflex: the issue is in TRT's handling of SigLIP's LayerNorm / GeLU / attention under FP16 precision.
- **Implication:** reflex exports SigLIP with caveats. Launch drafts currently live with the 2–4e-4 `ORT_MAX_DIFF_THRESHOLD` (`vlm_components.py`: `ORT_MAX_DIFF_THRESHOLD = 5e-4`; rationale: SigLIP's 27 transformer layers accumulate fp32 rounding, max_diff 2-4e-4 is expected).
- The question of whether dynamo-export would improve these numbers is **moot** — the bug lives downstream in TRT's FP16 path.

---

## Shape inference mismatches (NVIDIA TRT issue #4203)

Source: aggregated — issue is tracked as a known class of problem when the ONNX file declares static shapes but TRT attempts shape-profile handling on top of them.

- **#4203** covers shape-inference mismatches on `trtexec` when ONNX has no explicit dynamic axes.
- This is the root cause behind our "drop `--minShapes/--optShapes/--maxShapes`" fix in `scripts/modal_bench_trt_fp16.py`:
  > "Our ONNX export has static shapes baked in (no dynamic_axes), so we MUST NOT pass `--minShapes/--optShapes/--maxShapes` — trtexec rejects them for static models. The engine is fixed at the export shape (batch=1, chunk=50, action_dim from model)."
- **Implication:** if we want TRT to handle multiple batch shapes (the `--max-batch > 1` path), we need **dynamic-axis export** + shape-profile building, which we don't have yet. Hence `src/reflex/runtime/server.py` drops TRT EP when `--max-batch > 1` (commit `e76678c 2026-04-14`).

---

## What `torch.onnx.export(dynamo=True)` would do well

Based on the GQA+RoPE spike (commit `6fedff3 2026-04-16`, result in `.agents/crank/spike-gqa-result.md`):

> "SmolLM2's GQA decoder (LlamaDecoderLayer) exports to ONNX cleanly first try — no patches, no custom ops, no workarounds. Approach: `torch.onnx.export` at opset 19, single decoder layer wrapped with RoPE computation included in wrapper. PyTorch 2.11 new exporter (`torch.export.export` strict=False) under the hood."
>
> "max_diff PyTorch-vs-ORT 4.01e-05, mean 7.57e-07."

**Direct dynamo export handles:**
- Attention (MHA and GQA).
- RoPE when implemented as `LlamaRotaryEmbedding` — standard HF pattern.
- GeLU (elementwise Mul+Tanh decomp by the exporter, no work on our end).
- Linear + Softmax + reshape.
- MLP blocks (SiLU + Linear stacks).

**Direct dynamo export does NOT handle:**
- `nn.RMSNorm` — emits opset-23 `RMSNormalization` TRT can't parse. **Must be overridden with `DecomposedRMSNorm`.**
- Dynamic SigLIP vision embedding position IDs — `SmolVLMVisionEmbeddings.forward()` has a dynamic `index_put` / `bucketize` loop that emits int64/float mismatches ORT cannot load. **Must be pre-computed and wrapped** (our `VisionEncoderForONNX` already does this).
- Constant folding on vision_encoder — **must use `do_constant_folding=False`** (folding corrupts the graph, per ETARS notebook).
- Float-valued Gather indices from vision — **must run `patch_gather_indices_once()` post-export** (per ETARS).

---

## Practical recommendation: hybrid approach

Three paths ranked by correctness-per-engineering-hour (from `current_session.md` line 11524):

### Path 1 (guaranteed correct) — wrap and export real classes

- Take `policy.model.vlm_with_expert.lm_expert` as-is, wrap in a thin `nn.Module`, call `torch.onnx.export` with `dynamo=True`.
- Pro: **correctness by construction** — all 4 "our reimplementation bugs" (sinusoidal timestep formula, RoPE base 10000 vs 100000, prefix_offset for self-attn, KV mask for cross-attn) disappear.
- Con: output ONNX may include RMSNorm opset-23 / other unsupported ops. RMSNorm at least must be swapped.

### Path 2 (recommended) — copy lerobot modeling files + override RMSNorm

- Copy lerobot's `SmolVLAPolicy.sample_actions` + `embed_prefix` + `embed_suffix` + `forward_cross_attn_layer` into `src/reflex/runtime/smolvla_native.py` verbatim.
- **Swap only `nn.RMSNorm → DecomposedRMSNorm`** for TRT compat.
- Let `torch.onnx.export(dynamo=True)` handle everything else (attention, RoPE, GQA, MLP).
- Pro: Hours of work, correct by construction, Jetson compatible.
- Pro: Much shorter bug list — the 4 "reimplementation bugs" still disappear.
- Con: must track upstream lerobot changes. **This is the path the repo is converging toward.**

### Path 3 (shortcut) — skip ONNX entirely for first launch

- Load the real `SmolVLAPolicy`, serve it via `reflex.runtime.ReflexServer` in pure PyTorch.
- Ship LIBERO success on day one.
- Pro: unblocks the launch narrative immediately.
- Con: no TRT path, loses the 2.6–3.3× speedup story on cloud GPU, loses Jetson FP16 deployment.
- Decision: **not the path** — Reflex's explicit positioning is "the deployment toolchain with TRT," not "a PyTorch server wrapper."

---

## 12-bug breakdown: which go away with direct export (from current_session.md line 11574)

From the table in Theme "The 12 pipeline bugs":

| Bug | ONNX-related? | Goes away with direct lerobot-code export? |
|---|---|---|
| State proj random weights | No — CLI plumbing bug | **Yes** (lerobot loads real weights) |
| Base vs fine-tuned VLM | No — CLI plumbing bug | **Yes** (AutoModelForImageTextToText handled upstream) |
| 5D pixel_values | No — wrong preprocessor | **Yes** (lerobot uses its own preprocessor) |
| Missing √hidden scaling | No — we skipped step | **Yes** (lerobot does it) |
| SigLIP [-1,1] range | No | **Yes** (lerobot does it) |
| Missing newline on task | No | **Yes** (lerobot does it) |
| 8D vs 6D state | No | **Yes** (lerobot does it) |
| Sinusoidal timestep formula | Our reimpl bug | **Yes** (not reimplemented) |
| RoPE base 10000 vs 100000 | Our reimpl bug | **Yes** (not reimplemented) |
| prefix_offset for self-attn | Our reimpl bug | **Yes** (not reimplemented) |
| KV mask for cross-attn | Our reimpl bug | **Yes** (not reimplemented) |
| `obs.get(k) or obs.get(other)` on np | Adapter bug | No — outside the export pipeline |

**"8 of 12 bugs disappear if we use lerobot's actual code."**

---

## "Local iteration is ~100× cheaper than Modal" (current_session.md line 11574)

Not specific to dynamo export but relevant: the decomposition-vs-direct-export decision has been **iterated on Modal**, where each run costs Modal-GPU-hour and takes 5–30 min. Key workflow lesson:

> "Rather than fighting the PyTorch API (which is a rabbit hole), let me make a cheaper test... Local iteration is ~100× cheaper than Modal."

Before the next exporter refactor, prototype locally against a small SmolVLA slice. Only then commit to Modal.

---

## Single-layer cliffhanger (current_session.md line 11468)

The final unresolved frontier at the end of the current session:

> "Single SELF-attn layer (layer 0) matches to **1e-5 precision, cos=1.0000**. The bug is somewhere in COMPOSITION — probably cross-attention layers."

- Our decomposed SELF-attn is numerically perfect per-layer.
- End-to-end cos drops to -0.27 / -0.24 after 10 Euler denoise steps.
- The ~2% per-step error must live in:
  1. `DecomposedRMSNorm` vs real `nn.RMSNorm` numerics (most likely culprit; direct export would fix).
  2. `F.silu(gate_proj(x)) * up_proj(x)` ordering vs real.
  3. Attention mask handling (we don't mask padded prefix positions; real does).
  4. Attention softmax upcast to fp32.

**Switching to hybrid Path 2** would address #1 (no more DecomposedRMSNorm variance) and #2 (no more our reimpl of the MLP block). Worth **committing to the hybrid path before chasing cross-attn numerically.**

---

## Decision record

- **Status:** the decomposed exporter ships today (v0.1). It is numerically correct per-layer for self-attn to 1e-5.
- **Gap:** end-to-end parity fails at 10-step integration (cos=-0.24).
- **Next action:** implement hybrid Path 2 in `src/reflex/runtime/smolvla_native.py`. Override only RMSNorm. Keep `DecomposedRoPE` if it's faster — but verify direct `apply_rope` works at opset 19 first.
- **Blocker:** none external. This is pure engineering time.
- **Kill-criteria for current decomposed path:** if hybrid Path 2 hits LIBERO cos≥0.99 end-to-end, deprecate the hand-decomposed `ExpertGQALayer` in favor of wrapped lerobot code.

---

## Related files

- `src/reflex/decompose.py` — `DecomposedRMSNorm`, `DecomposedRoPE`, `DecomposedAdaRMSNorm` patterns.
- `src/reflex/exporters/smolvla_exporter.py` — current hand-built exporter.
- `src/reflex/exporters/vlm_prefix_exporter.py` — 4-file VLM split.
- `src/reflex/exporters/vlm_components.py` — `VisionEncoderForONNX` + pre-computed position IDs + `patch_onnx_type_mismatches`.
- `.agents/crank/spike-gqa-result.md` — the GQA+RoPE dynamo-export spike result (PASS, max_diff 4e-5).
- `scripts/modal_pytorch_vs_onnx.py` — the diagnostic comparing real PyTorch vs exported ONNX on identical noise.
