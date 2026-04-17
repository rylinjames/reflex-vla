# VLA-Eval Integration

How Reflex integrates with AllenAI's `vla-eval` harness (arXiv 2603.13966) to drive LIBERO / SimplerEnv / ManiSkill evaluations. Lives at `src/reflex/runtime/adapters/vla_eval.py`.

**Design posture (ADR `2026-04-14-wrap-not-rebuild-vla-eval.md`):** wrap, don't rebuild. Expose Reflex as a `PredictModelServer` subclass so `vla-eval` runs it as if it were any other VLA backend.

---

## Wire format: WebSocket + msgpack, NOT HTTP

Key surprise surfaced during integration (sessions_md line 32):

> "vla-eval uses WebSocket+msgpack, not HTTP — had to build a thin adapter (`reflex.runtime.adapters.vla_eval`)."

- `vla-eval` connects to the model server over WebSocket, serializes observations as msgpack frames, expects action chunks back in the same format.
- Reflex's normal FastAPI `/act` endpoint (HTTP + JSON) does **not** talk to vla-eval directly.
- Adapter bridges the two: it owns a `ReflexServer` instance (same one that powers `reflex serve`) and exposes it via `PredictModelServer.on_message()`.

---

## `PredictModelServer` class lineage

From `scripts/modal_libero10.py` Step 2 and the adapter design:

- `vla-eval` provides a base class `vla_eval.PredictModelServer`.
- Reflex subclasses as `ReflexVlaEvalAdapter(PredictModelServer)`.
- The subclass's `__init__(self, export_dir, device, out_dim, camera, vlm, norm, **kwargs)` accepts all reflex-specific config.
- **`**kwargs` is load-bearing** — vla-eval's `run_server()` auto-injects additional kwargs from the parent `PredictModelServer.__init__` signature (commit `af6acba 2026-04-14`: "accept **kwargs in model server __init__ (vla-eval auto-injects parent args)").
- The adapter then initializes a `ReflexServer(export_dir, device=device, strict_providers=...)` internally and routes each observation through `ReflexServer.predict()`.

Concrete adapter startup log (from Modal app `ap-QAG1Pk9w3DkuZnVs9VC8Ke`):
```
Expert ONNX inputs: ['noisy_actions', 'timestep', 'position_ids', 'vlm_k', 'vlm_v']
VLM orchestrator loaded complete=True (vision_encoder.onnx + text_embedder.onnx + decoder_prefill.onnx — all CPU)
Loaded normalizer stats from 2 file(s): ['action_mean', 'action_std', 'state_mean', 'state_std']
Adapter config: export=/tmp/reflex_libero_export device=cuda out_dim=7 camera=<first> vlm=on norm=on
```

---

## Observation schema per benchmark (LIBERO)

vla-eval sends an `obs` dict per step. The critical discovery: **`send_state=True` must be set explicitly in the run config** or the adapter only sees `images` and `task_description`, never the robot state.

### LIBERO obs schema (when configured correctly)

```python
obs = {
    "images": {
        "agentview": np.ndarray[256, 256, 3] uint8,   # main camera
        "wrist":     np.ndarray[256, 256, 3] uint8,   # wrist camera (if send_wrist_image=True)
    },
    "states":             np.ndarray[8] float64,        # env state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)
    "controller_states":  np.ndarray[?] float64,        # controller output — different semantic
    "task_description":   str,                         # "put the red bowl on the plate"
}
```

### Critical observations from debugging (current_session.md)

- **Without `send_state=True`** (Bug #6, line 10117): obs schema is `{'images', 'task_description'}`. Model gets `state=None` → adapter passes `zeros(6)` → garbage actions regardless of VLM.
- **Without `send_wrist_image=True`:** LIBERO sends only the agentview camera. SmolVLA-LIBERO was trained on **3 cameras** (`camera1/2/3`), so single-camera input is a distribution mismatch but must work regardless.
- **`states` vs `controller_states`** (Bug #7, line 10152): model may have been trained on `controller_states` (controller output) not `states` (raw env). Unresolved which one SmolVLA-LIBERO prefers.
- **`obs.get(k) or obs.get(other)` fails on numpy arrays** (Bug #12, line 10221): `ValueError: truth value is ambiguous`. Adapter must use `x if x is not None else y` instead.

### LIBERO action space

- 7-dim: 6 joints + gripper.
- `REFLEX_ACTION_DIM_OUT=7` env var forces reflex to slice its 32-dim action output down to the first 7 dims.
- Action-normalizer has 7-dim stats.

---

## `run_server` kwargs auto-injection from `__init__` signature

The adapter's `__init__` signature IS its CLI:

```python
class ReflexVlaEvalAdapter(PredictModelServer):
    def __init__(
        self,
        export_dir: str,
        device: str = "cuda",
        out_dim: int = 7,
        camera: str = "<first>",
        vlm: bool = True,
        norm: bool = True,
        **kwargs,    # mandatory — vla-eval injects parent args here
    ):
        ...
```

When vla-eval's `run_server(ReflexVlaEvalAdapter, config_yaml)` runs:

1. It reads the YAML config.
2. Introspects `ReflexVlaEvalAdapter.__init__` via `inspect.signature`.
3. Passes matching config keys as kwargs, reserves unknown ones for `**kwargs`.
4. Starts the WebSocket server.

This means **new adapter params = add them to `__init__`, document them in config**. No separate CLI boilerplate.

---

## Key benchmark knobs (LIBERO config YAML)

The YAML written to `/tmp/libero_10_config.yaml` in `scripts/modal_libero10.py`:

```yaml
model_server:
  class: reflex.runtime.adapters.vla_eval.ReflexVlaEvalAdapter
  kwargs:
    export_dir: /tmp/reflex_libero_export
    device: cuda
    out_dim: 7
    camera: <first>
    vlm: on
    norm: on

benchmark:
  suite: libero_10
  episodes_per_task: 1          # or 10 for the full LIBERO-10 matrix
  max_steps: 150                # SmolVLA paper uses 600 — we use 150 for speed
  seed: 7
  num_steps_wait: 10

observation:
  send_state: true              # CRITICAL — without this, state is None
  send_wrist_image: true
  image_size: 256
```

### Knob-by-knob consequences

- **`send_state: true`** — must be set; otherwise every episode predicts from zeros. (Bug #6.)
- **`send_wrist_image: true`** — adds wrist camera obs; SmolVLA-LIBERO uses multi-camera at train time.
- **`episodes_per_task`** — 1 for a smoke run (~3 min/episode × 10 tasks = 30 min); 10 for a real matrix (~5 hours on A10G).
- **`max_steps`** — 150 is the fast default. SmolVLA paper uses 600 — if task fails at 150, the budget may be the issue (rabbit-hole warning line 9682 of current_session.md).
- **`seed`** — 7 is the canonical reflex seed.
- **`num_steps_wait`** — 10 steps of warmup before starting the evaluation.
- **`image_size: 256`** — LIBERO renders at 256×256; VLM reflex vision encoder expects 512×512. Adapter handles upscaling.

---

## The Modal-wrapped runner: `scripts/modal_libero10.py`

Fundamental flow:

1. **Image build** (~10 min first time): `debian_slim` + heavy sim deps (torch, ONNX, vla-eval, mujoco, robosuite==1.4.1, bddl==1.0.1, LIBERO via git+clone, osmesa). See `current_session.md` Theme "LIBERO-10 simulation blockers" for the install-minefield ladder.
2. **Export**: `reflex export lerobot/smolvla_libero --target desktop --output /tmp/reflex_libero_export` (~173s).
3. **Adapter launch**: `python -m reflex.runtime.adapters.vla_eval` on port 8000. Waits for `/health` `model_loaded: true`.
4. **Env smoke**: `env.reset()` for LIBERO-10 task 0. Must complete in <10s (osmesa can hang silently on EGL→osmesa boundary).
5. **vla-eval run**: `vla-eval run --config /tmp/libero_10_config.yaml --no-docker`.

### Critical infrastructure lessons (all from debugging commit `2d60d6d..2c597b6 18 commits Apr-16`)

- **LIBERO's `setup.py` has `install_requires=[]`** — none of its requirements.txt deps get installed. Install narrowly (bddl==1.0.1, robosuite==1.4.1, gym [not gymnasium], hydra-core, easydict, einops) else full requirements.txt downgrades transformers / numpy and nukes the ONNX stack.
- **`robosuite 1.5+ changed module paths`** — pin `robosuite==1.4.1` which LIBERO expects.
- **LIBERO imports `bddl` at reset() time, not at package import.** bddl==1.0.1 is on pypi, just isn't in the setup.py install_requires.
- **LIBERO `__init__.py` reads stdin during import** for a "custom path wizard." No stdin in Modal containers. Fix: `scripts/patch_libero.py` replaces all 3 input() calls with a hard-coded 'n' via Python regex. Shell patching was abandoned — quoting issues caused 6 failed commits.
- **LIBERO's `.pyc` cache persists after source patching** — must nuke caches before first import. See commit `cb742db 2026-04-16`.
- **`MUJOCO_GL=osmesa` over `egl`** — EGL hangs silently on some debian_slim+NVIDIA combos; osmesa is slower but reliable.
- **`LIBERO_DATA_DIR` + `LIBERO_BASE` env vars** skip the "where is your data" prompt.
- **`num2words`** is a transitive dependency of `SmolVLMProcessor` — easy to miss; throws `ModuleNotFoundError` only at first processor call.
- **`subprocess.run(capture_output=True)` buffers until exit** — LIBERO runs for 50+ min; looks hung. Fix: stream line-by-line via `select()` on the subprocess pipes.

---

## Idle-timeout + streaming diagnostics

`scripts/modal_libero10.py` surfaces:
- **Idle-timeout guard** (600s / 10 min) — kills vla-eval if no stdout for 10 minutes. Avoids a 40-min hang on osmesa first-scene compilation.
- **Overall timeout** 3600s (1 hour) for the full 10-task × 1-episode matrix.
- **Log tail dump** for key markers at completion: "First predict", "First predict actions", "VLM orchestrator failed", "dummy conditioning", "ERROR", "Traceback".
- **Normalizer file presence check** — asserts both normalizer safetensors are present in the export dir before launching vla-eval.
- **Env smoke subprocess** — `env.reset()` in a separate process before running the full eval: "If this hangs, vla-eval would hang too — fail fast here with a clear message."

---

## What vla-eval returns

For each task × episode, vla-eval emits a status line:
```
Task 1: put both the alphabet soup and the tomato sauce in the basket
  ep0 → FAIL (steps=150, max)
```

Per-episode record: `{task, ep, status, steps, success_flag}`. Wall-clock ~3 min per episode at 150 steps.

Final JSON (what `reflex bench --benchmark libero_10` eventually consumes): `{benchmark, model, episodes, success_rate, per_task, latency_p50, latency_p99}`.

---

## The 0% LIBERO task-success state-of-play (as of session end)

From `current_session.md` line 10268 — all 12 bugs fixed, infrastructure all green, but LIBERO-10 still 0%:

| Component | Status |
|---|---|
| Unified CLI (`reflex export` auto-produces 4 ONNX files for SmolVLA) | PASS |
| vla-eval adapter (`ReflexVlaEvalAdapter`) | PASS |
| Normalizer pipeline (state + action) | PASS |
| Per-layer `vlm_kv` ONNX export | PASS |
| 5D→4D pixel_values fix | PASS |
| LIBERO sim on Modal (bddl, gym, osmesa, robosuite pin) | PASS |
| **LIBERO task success** | **0% unresolved** |

Why 0%? The remaining unknown is a systemic ~2%-per-step error in the expert denoise that compounds to cos=-0.24 after 10 Euler steps. See `direct_torch_export_viability.md` for the path forward.

The adapter and vla-eval wiring is **not the problem.** The export pipeline's numerical correctness is.

---

## First-class `reflex eval` (aspirational)

From `current_session.md` line 8097 — the long-term vision:

> "Don't hardcode a LIBERO-specific script. The long-term fix is a first-class `reflex eval <export_dir> <benchmark>` CLI command that:
> - Reads action space, image size, and model capability from config.
> - Maps model output dims to benchmark action space automatically (SmolVLA 32-dim → LIBERO 7-dim, pi0 32-dim → DROID 8-dim, etc.).
> - Supports multiple benchmarks via a plugin pattern: LIBERO, SimplerEnv, ManiSkill. Not one script per benchmark.
> - Wraps vla-eval as one backend but can fall back to direct sim runners.
> - Publishes standardized JSON: `{benchmark, model, episodes, success_rate, per_task, latency_p50, latency_p99}`."

Today this is spelled `reflex bench --benchmark libero_10` (commit `c768c54 2026-04-16`); the plugin scaffolds in `src/reflex/eval/{libero,simpler,maniskill}.py` are the foundations.

---

## Related files

- `src/reflex/runtime/adapters/vla_eval.py` — the adapter (the thin wrapper).
- `src/reflex/runtime/server.py` — `ReflexServer` (the real inference engine the adapter delegates to).
- `src/reflex/runtime/vlm_orchestrator.py` — VLM 4-ONNX coordinator.
- `src/reflex/eval/libero.py`, `src/reflex/eval/simpler.py`, `src/reflex/eval/maniskill.py` — benchmark plugin stubs.
- `scripts/modal_libero10.py` — Modal runner; canonical LIBERO integration sample.
- `scripts/patch_libero.py` — the regex patcher for LIBERO's stdin-input calls.
- `tests/test_vla_eval_adapter.py` — adapter unit tests.
