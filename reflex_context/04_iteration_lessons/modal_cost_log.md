# modal_cost_log — rough $ per experiment class

## Per-run rough costs (A10G or A100, 2026-04 rates)

Modal A10G billed at ~$1.10/hr. Modal A100-40GB billed at ~$3.00-4.00/hr. Times below include image pull, deps install, checkpoint download, execution, and teardown.

| Experiment class | Hardware | Wall time | Cost/run | Notes |
|---|---|---|---|---|
| Export-only verification (1 model) | A100 | ~3 min | **$0.05-0.15** | `modal_cli_export.py`; ONNX export + validate only |
| Single benchmark run (1 model, 1 path) | A10G | ~5-10 min | **$0.10-0.20** | `modal_bench_onnx_vs_torch.py`; 100 trials |
| 4-model benchmark table | A10G | ~30-40 min | **$0.60-0.80** | `modal_e2e_all_models.py`, `modal_bench_path_b.py` |
| TRT engine build per model | A10G | ~15 min | **$0.25-0.35** | `modal_bench_trt_fp16.py`; workspace 4GB, trtexec --fp16 |
| Multi-model TRT FP16 bench | A10G | ~60 min | **$1.00-1.20** | all 4 models; TRT engine build + bench |
| Install-path verification | A10G | ~15-20 min | **$0.30-0.40** | `modal_verify_install_path.py`; pip install from git |
| Verify-bench-all (4 models) | A10G | ~60 min | **$1.00-1.20** | `modal_verify_bench_all.py` |
| Stage diff (diagnostic) | A10G | ~4-6 min | **$0.10-0.15** | `modal_stage_diff.py`; 3 stages, per-layer k/v |
| Pytorch-vs-onnx action diff | A10G | ~3-5 min | **$0.10-0.15** | `modal_pytorch_vs_onnx.py` |
| Real-model batching (pi0) | A10G | ~40 min | **$0.70-0.90** | `modal_verify_batching_real.py`; 4 scenarios x 32 concurrent |
| Adaptive-denoise verify (4 models) | A10G | ~40 min | **$0.70-0.90** | `modal_verify_adaptive_real.py` |
| LIBERO-10 full run (10 tasks x 10 eps) | A10G | ~2 hours | **$2.20-2.50** | `modal_libero10.py`; full eval |
| LIBERO-10 quick (1 task, 1 ep) | A10G | ~5-10 min | **$0.10-0.20** | smoke test path |
| LIBERO-10 partial (aborted at task 2/10) | A10G | ~12 min | **$0.22** | ap-QAG1Pk9w3DkuZnVs9VC8Ke, Apr-17 11:14 |
| Wedge composition verify (fake ONNX) | cpu=2 | ~10 min | **$0.03-0.05** | `modal_verify_wedge_compose.py`; no GPU |
| Strict providers verify (4 scenarios) | A10G x4 | ~20 min | **$0.35-0.45** | `modal_verify_strict_providers.py` |

Rough rule of thumb:
- **Export-only / diagnostic**: ~$0.10
- **Benchmark (1 model)**: ~$0.20
- **Benchmark (4 models)**: ~$0.80-1.20
- **TRT engine build**: ~$0.30
- **Install or bench verify**: ~$0.30-1.20
- **LIBERO full**: ~$2.50; partial ~$0.20-0.40

---

## Session-level aggregate spend (actual ledger)

Reconstructed from assistant messages in `ced2c4f1` transcript + Modal app list.

### Apr-13 — scaffolding (single commit, no Modal)
- $0.

### Apr-14 — v0.1.0 build + wedge assembly + benchmarks + post-mortem
Reported in-line across session: "~$15 Modal cost tonight" (line 5453), "~$8-12 across ~12 runs" (line 5218).

Itemized from commit history:
- Initial smolvla export + expert + full pipeline (scripts `modal_test_export`, `modal_real_export`, `modal_expert_export`, `modal_full_pipeline`, `modal_vlm_export`, `modal_e2e_pipeline`, `modal_cli_export`): ~8 runs, ~$2-3 total
- pi0 + pi0.5 + GR00T exporter E2E (`modal_test_pi0`, `modal_test_pi05`, `modal_test_gr00t`, `modal_test_gr00t_full`, `modal_probe_gr00t`): ~5 runs at larger checkpoints, ~$2-3
- 4-model benchmark (`modal_e2e_all_models`, `modal_bench_onnx_vs_torch`, `modal_bench_path_b`): ~3 runs, ~$2-4
- TRT FP16 bench (`modal_bench_trt_fp16`): 1 run, ~$1
- Strict provider + wedge compose + batching verify (`modal_verify_strict_providers`, `modal_verify_wedge_compose`, `modal_verify_batching`, `modal_verify_batching_real`): ~5 runs, ~$2-3
- Adaptive denoise validation (`modal_verify_adaptive_real`): 1 run, ~$0.80
- Install path + TRT+batch verify (`modal_verify_install_path`, `modal_verify_trt_with_batch`, `modal_verify_bench_all`): ~3 runs, ~$2-3
- `reflex doctor` + launch-drafts updates: ~$0

**Apr-14 total: ~$12-18 across ~25 runs**

### Apr-16 — validate roundtrip + VLM waves + LIBERO death march
Session reports ~$5-8 for the LIBERO runs + trajectory-replay image-format fighting.

- 5-phase `reflex validate` RPI (`test_validate_roundtrip.py` local; no Modal)
- VLM prefix stubs and waves (d641134, 4daf6ea, etc.) — mostly local test edits
- Trajectory-replay iteration: 9 commits, 37 min, ~2-3 Modal runs (image format fighting), ~$0.30-0.80
- LIBERO integration death march: 18 commits, 75 min. Each run `modal_libero10.py` is 5-15 min. Say 8-12 actual runs, ~$3-5
- Text embedder export (36d8a40): 1 run, ~$0.20

**Apr-16 total: ~$5-8**

### Apr-17 — stage-diff hunt + libero continuation
Apr-17 Modal app list has 7 named runs:

| App | Wall | ~$ |
|---|---|---|
| ap-uKaH8uEPuCeoKz0C6TCqbV (reflex-stage-diff) | 6 min | $0.12 |
| ap-YrnHF0WgFXQ2Y7HWlYHPaI (reflex-stage-diff) | 6 min | $0.12 |
| ap-oXrqhfnQFJLuuY4A9GbPSv (reflex-stage-diff) | 4 min | $0.08 |
| ap-2tNsuBRSnvuQ9kWPwm55Ob (reflex-stage-diff) | 2 min | $0.04 |
| ap-v6gmsosx9ayiGJxoWUfs6o (reflex-pytorch-vs-onnx) | 3 min | $0.06 |
| ap-oBhVQcQnjsd4uMK6lSy98D (reflex-pytorch-vs-onnx) | 4 min | $0.08 |
| ap-QAG1Pk9w3DkuZnVs9VC8Ke (reflex-libero10) | 12 min | $0.22 |

**Named Modal: ~$0.72** — plus iteration on earlier failed runs + image rebuilds when deps changed: likely another $2-4 in unnamed or retried runs. **Apr-17 session total: ~$5-10 inclusive of the ~6h debugging.**

### Running total across Apr 13-17
- **~$30-50** total Modal spend on Reflex VLA across ~5 active days.

---

## What drove cost

Three main drivers, in order:

1. **Image build time when deps change.** Every new pip dep (`lerobot`, `num2words`, `bddl`, `robosuite`, `osmesa`, `mujoco`) triggers a 2-5 min image build that burns on the clock. LIBERO integration's 18-commit death march cost because each commit was a new image. Mitigation: use `detach` mode to avoid keeping the local shell hot, pin the image aggressively, bundle deps in one `.pip_install()` call.

2. **Checkpoint download** (SmolVLA 907MB; pi0 3.5GB; pi0.5 3.62GB; GR00T 6.6GB). Every cold run re-pulls. Mitigation: Modal volumes for HF cache; `huggingface_hub.snapshot_download(...)` once, then reuse. Not consistently done in our scripts.

3. **Actual GPU time.** Usually 30s-3min per experiment. Dominated by TRT engine build (30-90s for smolvla, 180-300s for gr00t).

### Specific waste in the Apr-17 session
- 6h wall clock to confirm cos_sim -0.24
- ~15 Modal runs across stage-diff, pytorch-vs-onnx, libero10
- **Pivot that saved the session**: move to local diagnostic scripts (`local_stage_diff.py`, `local_single_layer_diff.py`) which run in ~30s at $0. Had those existed from the start, Rung 4 (single-layer copy diff) would have been reachable in 20 min of dev time instead of burning ~$5 Modal spend on the same answer.

---

## Optimizations that worked

### `modal run --detach`
Returns immediately; the Modal function keeps running. Check logs via `modal app list` / `modal app logs <app_id>`. Stops you from wasting local shell time waiting for a 12-min LIBERO run.

### Line-by-line stdout streaming
Default `subprocess.run(capture_output=True)` inside a Modal function buffers all stdout until the subprocess completes, making a 40-min LIBERO run look hung. Fix in `modal_libero10.py`: stream stdout line-by-line via `select`. Also: redirect server stdout to a file, not a pipe (pipe buffers ~64KB and deadlocks). Multiple scripts learned this the hard way.

### Local diagnostic scripts (the key optimization)
- `scripts/local_stage_diff.py` — $0, 60s instead of $0.12, 6 min.
- `scripts/local_single_layer_diff.py` — $0, 30s.
- `scripts/local_full_diff.py` — $0, 2 min.

For ~15 iterations of the diagnostic ladder, local saves **$2-5 and 60-90 min of wall time** per session.

### `scaledown_window=60s`
Default is longer; pay for longer idle. Specify `scaledown_window=60s` on `@app.function` so containers shut down fast when done.

### Cache HF checkpoints in a Modal Volume
Not consistently done. Recommended pattern:
```python
hf_cache_vol = modal.Volume.from_name("reflex-hf-cache", create_if_missing=True)
@app.function(volumes={"/root/.cache/huggingface": hf_cache_vol})
def run():
    ...
```
Saves 10-60s per run on checkpoint downloads. Over 30 runs, that's 5-30 min of billable time saved.

### Single image for related scripts
Instead of one image per script (each causes a rebuild), share an image definition. Most of the `modal_bench_*` scripts share the `nvcr.io/nvidia/tensorrt:24.10-py3` base; centralize that.

---

## What would have saved the most money in retrospect

**Written as advice to future-sessions:**

1. **Build `scripts/local_stage_diff.py` before the first Modal run.** At Apr-17 rates, every "I'll just run it on Modal quickly" costs $0.10-0.25. Ten of them cost more than a day of local setup.

2. **Never iterate sim environments on Modal.** Changing `robosuite==1.4.1` vs `1.5` or trying a different `osmesa` version shouldn't cost $0.15 per attempt. Pin once, commit the image fingerprint, move on.

3. **Batch related experiments into one Modal call.** Instead of 4 separate apps for `reflex-stage-diff`, one app with 4 parameterized invocations. Saves image pull + deps install per extra run.

4. **Use `--detach` and check logs later.** Kills the habit of staring at terminal output for 12 min.

5. **Don't re-run a benchmark at a new date unless something changed.** A benchmark table from 2 weeks ago is still the benchmark table today unless hardware or code moved.

---

## Budget envelope going forward

Based on current burn rate, a productive weekly budget for Reflex VLA is:

- **Normal week:** $10-20 Modal, 20-40 experiments, 70% diagnostic (should be local) + 30% benchmark/integration (Modal-only).
- **Benchmark publication week:** +$30 one-off for big TRT bench + real-robot validation.
- **Emergency debug week (like Apr-17):** budget up to $30 with explicit kill-switch at $30 to force a pivot (in Apr-17 it was "copy lerobot code wholesale").

Target: **under $40/week** Modal + $0/week local iteration, with local covering 80% of diagnostic churn.
