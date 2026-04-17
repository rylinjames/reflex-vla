# Stage Diff Snapshot

Per-stage PyTorch-vs-ONNX cosine similarity after all 12 pipeline bugs were fixed. This is the canonical snapshot showing WHY LIBERO task success is still 0% — every component is nearly perfect, but per-step expert velocity error compounds over 10 Euler integration steps to a catastrophic final-action drift.

**The punchline: components are fine, composition compounds.**

## Setup

- **Hardware**: Modal A10G, image `nvcr.io/nvidia/tensorrt:24.10-py3`.
- **Script**: `scripts/modal_stage_diff.py`.
- **Model**: `lerobot/smolvla_libero` (fine-tuned SmolVLA for LIBERO).
- **Fixes applied** (all 12 bugs, see `02_bugs_fixed/smolvla_pipeline_bugs.md`):
  - state_proj loads real checkpoint weights (not random)
  - `AutoModelForImageTextToText` unwrapped to inner model (144 missing, 0 unexpected = healthy)
  - 5D→4D pixel_values handled
  - SigLIP `[-1, 1]` range normalization
  - 8D state (eef_pos(3) + axis_angle(3) + gripper_qpos(2))
  - newline appended to task instruction
  - RoPE base 100000 (not 10000)
  - normalizer + unnormalizer loaded from checkpoint
  - shared flow-matching noise via `RandomState(7)` injected into both paths
  - tokenizer loaded from BASE SmolVLM2 (`vlm_model_id`), not the policy checkpoint
  - `text_embedder.onnx` exports real `embed_tokens` (not seeded-random fallback)
  - policy forced to fp32+cpu before comparison

## Params

- Input image: 512×512 synthetic (matches vision ONNX export size).
- 3 cameras = same image replicated (LIBERO sends `agentview` only in practice; this is the structural-diff setup).
- State: 8-dim `eef_pos(3) + axis_angle(3) + gripper_qpos(2)`, normalized.
- Task: deterministic string.
- `num_keep = min(16, len(tm.layers))` — SmolVLA truncates VLM text decoder to 16 layers.
- Noise: `RandomState(7).randn(1, chunk, max_action_dim)` shared between paths.

## Snapshot

| Stage                                 | cos_sim    | Notes |
|----------------------------------------|------------|-------|
| `vision_embeds`                        | **1.0000** | SigLIP vision tower output. max_abs ≈ 8.7e-05. Perfect match after the `AutoModelForImageTextToText` unwrap fix |
| `text_embeds`                          | **1.0000** | `text_embedder.onnx` exports real SmolLM2 `embed_tokens`. Perfect match after deterministic RNG + real-ONNX fix |
| `state_proj`                           | **1.0000** | `nn.Linear(32, 960)` loaded from checkpoint (Bug #1: was using random weights). Perfect match after `state_proj_weight.npy` loads correctly |
| `layer_0_k` (decoder VLM KV, post-RoPE) | **1.0000** | First VLM decoder layer K matches |
| `layer_0_v` (decoder VLM KV)           | 0.9117     | **Outlier.** First layer V has a ~9% drop. Reproducible across multiple runs (ap-YrnH, ap-oXrq, ap-2tNs all show the same number). Structural divergence candidate |
| `layer_8_k / layer_8_v` (middle VLM)    | 0.999      | Middle layers match to 3 nines |
| `layer_15_k / layer_15_v` (final VLM)   | 0.999      | Final layer matches too |
| **Full expert velocity (per step)**     | **0.977**  | Expert produces velocity with 20% magnitude error + sign inversions on dims 2 (axis) and 6 (gripper) |
| **Final action (10-step Euler)**        | **-0.24**  | Per-step 0.977 error compounds through 10 integration steps to catastrophic final-action drift |

Source: `modal app logs` on `ap-YrnHF0WgFXQ2Y7HWlYHPaI` (Apr-17 10:57), `ap-oXrqhfnQFJLuuY4A9GbPSv` (Apr-17 11:08), `ap-2tNsuBRSnvuQ9kWPwm55Ob` (Apr-17 11:34). Expert velocity norm first 7 dims: `[3.636, -0.145, 0.934, 0.380, -1.624, -0.960, 0.177]`, norm = 90.3 — a useful reproducibility fingerprint.

## Explanation

**Per-step small error compounds over 10 Euler integration steps to catastrophic final-action drift.**

Flow-matching inference integrates a velocity field from `t=1` to `t=0` in 10 Euler steps. Each step applies:
```
action_{t+1} = action_t + velocity(action_t, t, vlm_kv) * dt    (dt = -1/10)
```

If the velocity field has per-step `cos=0.977` error (≈2.3% angular deviation), and 20% magnitude error, Euler integration amplifies both:
- Angular drift compounds roughly geometrically: 10 steps of 2.3% drift → ~25% drift.
- Magnitude error stretches each step → each integrated action overshoots/undershoots.
- Sign flips on dims 2 and 6 (gripper open/close) mean the gripper action is literally inverted.

The result: cos_sim on final action = -0.24. The action vector points **away** from ground truth, so the robot performs the opposite intent. LIBERO task success = 0%.

## Why layer_0_v is 0.9117 (the outlier)

Layer 0's V tensor drops from cos=1.0 down to 0.91 while K matches perfectly. Candidates:
- **K and V use different projection norms** — the K projection may be followed by RMSNorm differently than V in the reference model. Decomposed RMSNorm in our exporter may be slightly off on V path specifically.
- **V padding mask** — if the reference model masks padded-prefix V differently than K (some architectures do this asymmetrically), our ONNX would skip that mask.
- **Attention mask is applied to V through scores × V** — not directly to V — so the 0.91 on V output suggests the V *values* themselves are drifting, not just their weighting. That points at RMSNorm or projection numerics.

This is the specific residual under active investigation (task #25 "Per-layer vlm_kv ONNX export").

## Why per-step velocity is 0.977 (not higher)

The expert is a 16-layer GQA transformer with RMSNorm, RoPE, and cross-attention to VLM KV. Components:
- Self-attention layers (layer 0 tested in isolation): cos=1.0000, max_abs=1e-5. **These compose perfectly.**
- Cross-attention layers: not individually tested. **Primary suspect.**
- AdaRMSNorm / RMSNorm decomposition: per-layer cos=0.999+ up to layer 15, so almost certainly not the culprit.
- Attention mask on padded prefix positions: real model masks these; our ONNX may not. Untested.
- Attention softmax upcast to fp32: real models upcast; decomposed path may stay in BF16/FP16. Untested.

The 0.977 per-step velocity cos represents the *accumulated* per-layer drift. Half-percent-per-layer errors add up over 16 layers to ~2-3% per step.

## Per-step cos_sim budget

**Per-step cos > 0.999 is required to survive 10 Euler integration steps** with final cos > 0.9. This is a verbalized-but-not-tested rule-of-thumb from current_session line 11435:

```
(1 - per_step_cos) * N_steps ≈ 1 - final_cos    (for small deviations)

Per-step 0.999 × 10 → final ≈ 0.99
Per-step 0.977 × 10 → final ≈ 0.77 (model predicts 0.977^10 ≈ 0.79 agreement)
Per-step 0.93  × 10 → final drops to ~0.5
```

Our measured (per-step 0.977, final -0.24) doesn't fit this cleanly — suggests the 20% magnitude error + dim-2/6 sign flips amplify drift beyond what angle-only analysis predicts. Euler integration is sensitive to direction *and* magnitude, and magnitude errors accumulate linearly (10× for 10 steps) not geometrically.

## Caveats

1. **`layer_0_v cos=0.9117` is the single reproducible structural discrepancy.** It's the next thing to root-cause. Fixing it may cascade into lower expert velocity error.
2. **Single-layer self-attn isolation test (cos=1.0000)** only tested layer 0. Layers 1, 3, 5, 7, 9, 11, 13, 15 (the cross-attn layers on odd indices) not yet individually tested. They are the remaining primary suspect.
3. **Expert velocity cos=0.977 per step** is measured on a single chunk (50 actions). Per-action cos within the chunk not broken out — front of chunk may be worse than tail, or vice versa.
4. **"Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)"** — current export uses BASE SmolVLM2-500M VLM weights for the prefill path, not the LIBERO-fine-tuned versions. Any residual drift from VLM prefix composition may trace to this. Tracked.
5. **Shared noise** is only injected into flow-matching initial condition, not into attention dropout / layer-norm stochastic paths (these are deterministic in eval mode anyway, but worth noting).
6. **Stage 5 expert velocity** is measured with identical `vlm_k, vlm_v` fed into both paths — i.e., we give the ONNX expert the CORRECT PyTorch-derived vlm_kv, not our ONNX-derived (imperfect) kv. Even with perfect kv input, the expert produces 0.977 cos. So the residual lives in the expert stack itself, not in VLM prefix composition.
7. **cos_sim on first action only** — a full chunk diff across all 50 actions might surface stronger patterns.
8. **Numbers are reproducible** (3 runs on 2026-04-17 show identical output to the fourth decimal) — so this is not numerical jitter, it's a structural residual.

## Verdict

**Infrastructure win is real and worth shipping:**
- Unified CLI (`reflex export` auto-produces 4 ONNX files for SmolVLA)
- vla-eval adapter (`ReflexVlaEvalAdapter`)
- Normalizer pipeline (state + action)
- Per-layer vlm_kv ONNX export (with 0.9117 V residual as open gap)
- 5D→4D pixel_values fix
- LIBERO sim on Modal (bddl, gym, osmesa, robosuite pin) fully packaged

**LIBERO task-success: 0% unresolved.** Primary remaining candidate: cross-attention composition inside the expert stack. Secondary candidates: attention padding mask, softmax fp32 upcast, decomposed RMSNorm V-path numerics.

**The path forward** (per `04_iteration_lessons/diagnostic_ladder.md` + current_session pivot at line 11524): copy `lerobot`'s `SmolVLAPolicy.sample_actions` + `embed_prefix` + `forward_cross_attn_layer` verbatim into `reflex/runtime/smolvla_native.py`. Swap only `RMSNorm → DecomposedRMSNorm` for TRT compatibility. Let `torch.onnx.export` handle the rest. Correct by construction; 8 of 12 bugs disappear.

## Related

- `02_bugs_fixed/smolvla_pipeline_bugs.md` — the 12 bugs listed in this document's setup.
- `pytorch_vs_onnx_cos_sim_timeline.md` — chronology of cos_sim measurements as fixes landed.
- `04_iteration_lessons/diagnostic_ladder.md` — stage-diff → single-layer → composition methodology.
- `05_sessions/2026-04-17_libero_correctness_hunt.md` — session log of the stage-diff + pytorch-vs-onnx Modal runs.
