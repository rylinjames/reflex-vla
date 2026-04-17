# Adaptive Denoising Validation (Phase IV)

Phase IV of the Apr-14 push: validate the early-stop heuristic in `src/reflex/kernels/turbo.py` and `src/reflex/runtime/server.py::_run_denoise` against real VLAs. The heuristic was originally validated on a synthetic 16-hidden toy model in `scripts/modal_sim_test.py`. Phase IV asks: does it survive on real policies?

**Bottom line: only pi0 validates. Quote only the pi0 number externally.**

## Setup

- **Hardware**: Modal A10G, image `nvcr.io/nvidia/tensorrt:24.10-py3`.
- **Script**: `scripts/modal_verify_adaptive_real.py`.
- **Models**: SmolVLA, pi0, pi0.5, GR00T — all 4 supported VLAs, exported via `reflex export`.
- **Timeout**: 2400s.
- **Execution**: ORT-GPU (CUDAExecutionProvider), FP32. TRT FP16 disabled for adaptive (the CUDA-graph capture replays all 10 steps; early-stop would invalidate the graph).

## Params

- `NUM_STEPS = 10` (full Euler schedule).
- `NUM_TRIALS = 25` (per model).
- `THRESHOLD = 0.01` — the magic number for velocity-norm delta convergence. Inherited from the synthetic toy in `modal_sim_test.py`. Early-stop fires when `abs(v_norm[t] - v_norm[t-1]) < 0.01` after step ≥ 2.
- Records per-step velocity norms, trigger step, and max-abs action diff at trigger-step vs full-10-step output.

## Results

| Model   | Triggered (of 25) | mean_step on trigger | Latency savings | Action diff at trigger vs full-10 | Verdict |
|---------|--------------------|----------------------|-----------------|-----------------------------------|---------|
| SmolVLA | **0/25**           | —                    | **0%**          | — (never activates)               | Adaptive never fires — velocities don't converge under this threshold for this model |
| pi0     | **25/25**          | **4.2**              | **58.4%**       | **0.073** (small)                 | **Real win.** Latency savings are meaningful and the action diff is within tolerance |
| pi0.5   | 3/25               | 9.4                  | 5.6%            | 0.762 (large)                     | Rarely triggers; when it does, it triggers late and drifts meaningfully |
| GR00T   | 25/25              | 3.0                  | 70%             | **0.674** (large)                 | Triggers too aggressively — meaningful drift in final action |

Source: `git log` theme "Phase IV — adaptive denoising validation on real VLAs", commits `1c40f14..091074c` (2026-04-14 16:33-16:37).

## Interpretation

**pi0 is the only model where adaptive denoising is a real win.** At step 4.2 (of 10), the velocity field has converged enough that stopping early produces an action within `action_diff=0.073` of the full 10-step result — typically below the per-joint action granularity of the robot. 58% latency savings on pi0 is worth shipping.

**SmolVLA velocities never converge under the 0.01 threshold.** The 99.8M expert has a richer velocity field that keeps evolving to step 10. Early-stop never fires, so the wedge is a no-op. Not harmful, but not useful either.

**pi0.5 triggers only 3/25 trials (12%) and drifts when it does.** The AdaRMSNorm architecture (3-chunk time-conditioned RMSNorm) makes velocity convergence behave differently from pi0. The 0.762 action diff is large — half the unit range on a normalized action. Would degrade task success.

**GR00T triggers very aggressively (step 3.0 mean) with 0.674 action diff — a meaningful drift.** The DiT expert's velocity collapses fast under the 0.01 threshold but that collapse doesn't mean the final action is correct. Using adaptive on GR00T degrades trajectory quality.

## Gate / production behavior

Since commit `091074c` (2026-04-14 16:37), `reflex serve --adaptive-steps` now warns when `model_type != "pi0"`:

```python
# src/reflex/runtime/server.py (gist)
if self.adaptive_steps and self.model_type != "pi0":
    logger.warning(
        f"--adaptive-steps was validated only on pi0. "
        f"model_type={self.model_type} may produce drifted actions. "
        f"Pass --no-adaptive-steps for safe serving."
    )
```

Same gate is intended for `src/reflex/kernels/turbo.py` behind an `--experimental` flag (see `GOALS.yaml::adaptive-denoise-fix`, weight 5). Not yet fully enforced.

## Caveats

1. **Threshold 0.01 is a magic number** with no theoretical backing. It was chosen on a synthetic 16-hidden toy transformer where velocities happen to converge under 0.01 for almost all inputs. Real VLA velocity fields have different magnitudes and decay rates; per-model calibration was deferred to v0.2.
2. **Action diff is measured against the 10-step full result, not against ground truth.** If the 10-step result is itself drifted from the true action, "diff=0.073" is measuring agreement with a possibly-wrong baseline. LIBERO task-success is the real gate; adaptive-on-pi0 LIBERO number not yet captured.
3. **Adaptive disables TRT FP16's CUDA-graph replay** — early-stop invalidates a captured graph. If TRT FP16 is the deployment path, `--adaptive-steps` auto-forces CUDA EP. Documented in `latency_benchmarks.md` § batching throughput.
4. **Threshold = 0.01 was measured with `float32` velocity norms**. Moving to FP16 for edge deployment may shift the effective threshold. Not re-validated for FP16 serving.
5. **The 25-trial sample** is thin for statistical conclusions on pi0.5 (3 positives). If thresholds get tuned per model, re-run with ≥100 trials.
6. **"Action diff" is max-abs over all dims**. Some drift could be concentrated in gripper channel (dim 6 on LIBERO); position dims might be fine. Not currently decomposed.

## What to cite externally

**Only the pi0 58% latency savings.** Do not quote a universal "adaptive denoising" speed-up — it's per-model and only pi0 currently works.

For pitches:
> "On pi0 (Physical Intelligence's 314M action-head VLA), enabling Reflex's adaptive denoising (`reflex serve --adaptive-steps`) triggers in 100% of chunks at step 4.2 on average, saving 58% of inference latency while keeping action diff within 0.07 of the full 10-step schedule."

Do NOT say:
- "Adaptive denoising saves 58% on all VLAs" (SmolVLA: 0%)
- "Adaptive denoising on GR00T is a 70% win" (action_diff is 0.67, a meaningful drift)

## Next steps (deferred to v0.2)

- Per-model threshold tuning. Sweep thresholds [0.001, 0.005, 0.01, 0.02, 0.05] × all 4 models × ≥100 trials.
- LIBERO task-success gate on pi0 with adaptive ON vs OFF.
- Threshold as a function of training schedule: flow-matching teacher / distilled student may need different thresholds.
- Make `--adaptive-steps` auto-pick threshold from `reflex_config.json` rather than hardcoded.
