# PyTorch-vs-ONNX Cosine Similarity Timeline

Chronology of how the PyTorch-reference-vs-ONNX-export action agreement (measured as `cos_sim` on the first action of a 50-step chunk) evolved as fixes landed. Each data point corresponds to a specific fix to the SmolVLA ONNX pipeline.

**Why this document exists:** the progression is the most compact view of why the LIBERO-10 task-success number is still 0% despite 12+ fixes. Per-step numerics are good; 10-step Euler integration compounds per-step error to a catastrophic final drift.

## Setup

- **Hardware**: Modal A10G, image `nvcr.io/nvidia/tensorrt:24.10-py3`.
- **Scripts**:
  - `scripts/modal_pytorch_vs_onnx.py` — end-to-end action diff (full 10-step denoise).
  - `scripts/modal_stage_diff.py` — per-stage diff (vision → text → state → per-layer kv → expert velocity).
- **Model**: `lerobot/smolvla_libero` (fine-tuned SmolVLA).
- **Task**: `"put the red bowl on the plate"`.
- **Input**: synthetic 256×256 image (3 copies for 3 cameras), 8-dim state (`eef_pos(3) + axis_angle(3) + gripper_qpos(2)`), shared flow-matching initial noise via `RandomState(99)`.
- **Metric**: `cos_sim` on the first predicted action (PyTorch policy's `predict_action_chunk()` output vs Reflex ONNX pipeline `ReflexServer.predict()` output), with **shared noise** injected into both so we aren't comparing noise-driven randomness.

## Params

- `chunk_size=50`, `num_steps=10`, `dt=-1/10` (t: 1→0).
- Normalizer + unnormalizer loaded from the `lerobot/smolvla_libero` checkpoint (safetensors) — per Bug #9 fix.
- PyTorch policy forced to `float32` + `cpu` before comparison (ONNX is fp32 on CPU; matching dtype avoids `Input FloatTensor vs weight BFloat16` errors).
- Vision normalized to `[-1, 1]` for SigLIP (Bug #4 fix).

## Timeline

| # | cos_sim | Fix that landed just before this measurement | Source |
|---|---------|----------------------------------------------|--------|
| 1 | **0.28**   | Baseline diff harness working. No component-level fixes yet. State proj still using random weights. VLM conditioning was random. | current_session line ~10668 |
| 2 | **0.498**  | `state_proj` now loads fine-tuned weights from checkpoint; manual preprocessing (SigLIP `[-1, 1]`, newline on task, 8D state) applied. Still MAJOR divergence but moving. | current_session line 10809 |
| 3 | **0.305**  | Shared flow-matching noise injected into both pipelines. Was non-deterministic before — "both paths got different noise, so we were comparing noisy vs noisy." This reveals that the earlier 0.28 / 0.498 had noise contamination. | current_session line 10912 |
| 4 | **0.08**   | Unwrapped `AutoModelForImageTextToText` → inner `AutoModel` correctly. Vision, text, and state-proj now cos=1.0000 each. Per-layer kv match to cos≥0.91. But end-to-end action cos dropped from 0.305 to 0.08 — composition is wrong somewhere, not component-level. | current_session line 11089, 11109 |
| 5 | **-0.27**  | Added √hidden scaling, RoPE base 10k→100k, prefix_offset for self-attn. Per-layer kv good. End-to-end went NEGATIVE — meaningful sign reversal on the velocity field. | current_session line 11325 |
| 6 | **0.977** per-step / **-0.24** final | Expert velocity measured per step: cos=0.977 per step, ~20% norm error. Over 10 Euler integration steps the error compounds to cos=-0.24 on final action. | current_session line 11435 |

Source: keyword grep across session `ced2c4f1-a341-45bf-ae1b-ba9f6ab0931c.jsonl` for `cos=`, `cos_sim`, `MAJOR DIVERGENCE`, `HEADLINE`.

## Per-stage snapshot after all fixes

See `stage_diff_snapshot.md` for the full per-stage breakdown. Headline:

| Stage | cos_sim |
|------|---------|
| vision_embeds   | 1.0000 |
| text_embeds     | 1.0000 |
| state_proj      | 1.0000 |
| layer_0_k       | 1.0000 |
| layer_0_v       | 0.9117 |
| layer_8         | 0.999  |
| layer_15        | 0.999  |
| expert velocity (per step) | 0.977 |
| **final action (10-step compounded)** | **-0.24** |

## Why cos_sim keeps "moving sideways" across fixes

Each fix was real — it fixed a genuine bug. But task success is an integration test. ONE subtly wrong operation out of ~500 in the pipeline pins the final cos to near-zero or negative. The sequence:

- **0.28 → 0.498**: state_proj random-weight fix swung most of the component-level noise out. +0.22.
- **0.498 → 0.305**: shared-noise methodology fix revealed that previous numbers were partly noise-inflated. Real number was lower. -0.19.
- **0.305 → 0.08**: vision AutoModel unwrap fixed per-stage (vision cos 0.70 → 1.0) but end-to-end went DOWN. This is the diagnostic that composition is broken, not components. -0.23.
- **0.08 → -0.27**: more plausible fixes (RoPE base, √hidden, prefix_offset). Per-step velocity got closer to correct but the error sign flipped, integrating over 10 steps away from ground truth. -0.35.
- **-0.27 → -0.24**: per-step velocity cos=0.977 is encouragingly close but still 20% norm error and sign flips on dims 2 (axis) and 6 (gripper). Compounds catastrophically over 10 Euler steps.

## The numerical budget observation

**Per-step cos > 0.999 is required to survive 10 Euler integration steps** (verbalized but not formally captured in tests — see `04_iteration_lessons/diagnostic_ladder.md`). At 0.977 per-step, the drift compounds roughly as `(1 - 0.977)^10 → 0.79` error accumulation, explaining the cos=-0.24 final.

Therefore: any remaining fix that moves per-step from 0.977 → 0.999 would close the LIBERO gap in a single pass. Candidates (from current_session line 11435):
1. `DecomposedRMSNorm` vs real `RMSNorm` numerics (the decomposed op upcasts to fp32; real one may not).
2. `F.silu(gate_proj(x)) * up_proj(x)` ordering vs real model order.
3. Attention mask (we don't mask padded prefix positions; real model does).
4. Attention softmax upcast to fp32 (real models frequently upcast).
5. Cross-attention composition (self-attn cos=1.0000 at layer 0, but cross-attn not individually tested — likely suspect).

## Caveats

1. **Shared noise is essential.** Without it, both paths get different random initial conditions and cos_sim is dominated by noise drift, not export correctness. Methodology fix at step 3 above (commit where `RandomState(99)` was pinned into both paths).
2. **Cos_sim on action 0 only.** A chunk is 50 actions; only first is compared. If the first action is fine but actions 1-49 drift, this metric misses it. Tradeoff for fast iteration.
3. **LIBERO task-success = 0% across all fixes.** Task success is downstream of the 10-step integration; cos_sim negative predicts task failure cleanly but doesn't say whether fixing per-step → 0.999 would cross 50% task success.
4. **"20% norm error"** in per-step velocity: the direction is approximately right (cos=0.977) but the magnitude is off. Euler integration amplifies both.
5. **Sign flips on dims 2 and 6 (gripper)**: noted in the session but not yet formally root-caused. Could be a single wrong sign in the expert's output projection.
6. **Single-layer self-attn matches to 1e-5 precision (cos=1.0000)**. That means the bug is in COMPOSITION — probably cross-attention layers — not individual self-attn math. Last unresolved frontier as of session end.
7. **Fine-tuned SmolVLA VLM layers not preserved** is a known-gap (`v0.3 item`) — the current export uses BASE SmolVLM2-500M VLM weights, not the LIBERO-fine-tuned ones. Any residual drift traceable to VLM prefix KV-cache composition may be this.
8. **vla-eval sends 1 camera (agentview) while model trained on 3 (camera1/2/3)** — camera mismatch is a remaining structural divergence candidate not isolated by shared-noise cos_sim.
9. **states vs controller_states** — LIBERO obs has both. Model may have been trained on `controller_states` (controller output) not `states` (raw env). Candidate for residual drift.

## Methodology discovery highlights

- **Shared noise is THE methodology fix** for flow-matching diff tests. Non-obvious — requires understanding that flow-matching initializes from random noise at t=1.
- **Stage diff reveals composition bugs that end-to-end diff hides.** Vision cos=1.0 but end-to-end cos=0.08 is a loud signal that components are fine but composition is broken.
- **Per-layer kv matching to ≥0.99 does NOT imply end-to-end correctness** — subtle attention-mask / padding / softmax differences can produce per-layer-OK-but-composition-wrong patterns.
- **Single-layer forward test** (run one expert layer in PyTorch with known inputs, then in our `ExpertGQALayer` with same weights+inputs, compare) is the fastest bisection tool. cos=1.0000 on single self-attn layer pinpoints cross-attn as the remaining suspect.

## Source scripts

- `scripts/modal_pytorch_vs_onnx.py` — the canonical diff harness (current).
- `scripts/modal_stage_diff.py` — per-stage diff; companion.
- `scripts/local_full_diff.py`, `scripts/local_stage_diff.py`, `scripts/local_expert_diff.py`, `scripts/local_single_layer_diff.py` — local variants (per `04_iteration_lessons/local_vs_modal.md`, local iteration is ~100× cheaper; prefer local for these diffs).

## Related

- `02_bugs_fixed/smolvla_pipeline_bugs.md` — the 12 bugs found during this hunt.
- `04_iteration_lessons/diagnostic_ladder.md` — the stage-diff → single-layer → composition strategy.
- `05_sessions/2026-04-17_libero_correctness_hunt.md` — session log.
- `stage_diff_snapshot.md` — the per-stage cos_sim table referenced throughout.
