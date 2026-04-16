# Trajectory Replay Smoke Test -- Implementation Result

**Status:** Implemented  
**Date:** 2026-04-15

## Files Created

1. **`scripts/modal_trajectory_replay.py`** -- Modal A10G trajectory replay
2. **`scripts/sim_smoke_test.py`** -- Local smoke test with `--quick` flag

## modal_trajectory_replay.py

Runs on Modal with A10G GPU. Pipeline:
1. Installs reflex-vla + all dependencies via `pip install -e .`
2. Exports SmolVLA via `reflex export lerobot/smolvla_base`
3. Starts `reflex serve` in background on port 8321 (CPU mode for reliability)
4. Downloads 5 episodes from `lerobot/xarm_lift_medium` (fallback: `lerobot/pusht`) via HuggingFace `datasets` streaming
5. For each episode: extracts image observations + expert actions, sends frames to `POST /act`, computes L2 error per step
6. Reports: mean L2, max L2, per-episode table
7. Pass/fail: mean L2 < 2.0

Run: `modal run scripts/modal_trajectory_replay.py`

## sim_smoke_test.py

Two modes:
- **`--quick`**: Generates 3 synthetic frames, runs 10-step denoising through `expert_stack.onnx` directly (no server), verifies output shapes (chunk_size x action_dim), bounded actions (no NaN/Inf, |action| < 50), and smoothness (consecutive frame L2 diff < 5.0). Targets <30s runtime. This is the GOALS.yaml `sim-smoke-test` gate.
- **Full mode** (no flag): Same as Modal script but runs locally -- export, serve, download data, replay, L2 comparison.

Run: `python scripts/sim_smoke_test.py --quick` or `python scripts/sim_smoke_test.py`

## GOALS.yaml Integration

The `sim-smoke-test` goal (weight 10) checks:
```
test -f scripts/sim_smoke_test.py && .venv/bin/python scripts/sim_smoke_test.py --quick
```

Quick mode auto-exports if no export directory is found, then validates ONNX inference directly.

## Design Decisions

- Uses `lerobot/xarm_lift_medium` as primary dataset (small, has image obs + 4-dim actions). Falls back to `lerobot/pusht` if unavailable.
- Streams dataset to avoid downloading full dataset to disk.
- Caps at 500 frames / 10 frames per episode to keep runtime under 15 minutes on Modal.
- L2 threshold of 2.0 is generous because text embeddings are still dummy/zero-vector -- real VLM prefix conditioning will tighten this.
- Server runs on CPU in both scripts to avoid GPU provider issues on diverse hardware.
