# 2026-04-16 — The LIBERO Infrastructure Death March

**Session theme:** The day the team tried to get LIBERO-10 running on Modal so there would be an actual task-success number to cite alongside the latency benchmarks. Latency said 86 Hz on SmolVLA. Task-success said … nothing, because the sim wouldn't boot. 18 commits in 75 minutes fighting a cascade of install quirks — bddl, gym (not gymnasium), robosuite, osmesa, MuJoCo, LIBERO's interactive `input()` calls in `__init__.py`. The sim eventually ran end-to-end. The task-success was 0%.

---

## Goal

Ship LIBERO-10 on Modal. The exporter had been validated end-to-end with max_diff < 1e-5 across every SmolVLA component. The CLI had been consolidated (`reflex export` now auto-produces the 4-file VLM split for SmolVLA; `check` merged into `validate --quick`; `turbo/split/adapt` deprecated). `reflex bench` produced the A10G numbers. The one missing thing: **an actual task-success number from a real sim-benchmark run against the exported model.**

The north-star framing was stated on Apr-10 and repeated through Apr-16: *"LIBERO number = the north star benchmark. Without it, marketing architecture; with it, marketing results."* Task #23 in the working memory: *"Ship LIBERO-10 Modal run and capture task-success number."*

---

## The plan

Apr-16 morning started with three preceding landings:

1. **VLM prefix pipeline v1 — stub waves** (commits `f72b8b9..d641134`, 11:17–11:25). Prefix exporter stub, server wired to run VLM prefix + pass `vlm_kv` to expert, 9 tests passing.

2. **VLM real forward pass — 4-file split** (commits `4daf6ea..9fb6ddb`, 11:45–12:35). The re-plan from 3-file to 4-file split (vision_encoder + text_embedder + decoder_prefill + expert_stack). GQA+RoPE ONNX spike passed (commit `6fedff3`, `max_diff=4e-05`, opset 19) — confirming SmolLM2's GQA decoder exports cleanly first try, no patches, no custom ops. 25 tests passing.

3. **`text_embedder.onnx` real export** (commit `36d8a40`, 14:20). Replaced the seeded-random fallback with real SmolLM2 `embed_tokens` exported to ONNX. After this, `reflex export lerobot/smolvla_base` produces 4 files: `vision_encoder.onnx`, `text_embedder.onnx`, `decoder_prefill.onnx`, `expert_stack.onnx`.

With the VLM plumbing wired, all that's left is to point the exporter at `lerobot/smolvla_libero` (the LIBERO fine-tune) and run it against a real sim benchmark. That script is `scripts/modal_libero10.py` — a thin wrapper that exports, launches the `reflex.runtime.adapters.vla_eval` adapter as a background server, and runs `vla-eval run` on LIBERO-10.

Two hours later, the sim still wouldn't boot.

---

## Run ladder

The session settled into a pattern of Modal runs numbered by what broke each time. Transcript line 8776 captured the early ladder:

- **Run #1** — 0% because VLM files weren't in export (fixed by CLI Step 1 / commit `fdd9bb3`).
- **Run #2** — killed (wrong `requirements.txt` nuked the ONNX stack).
- **Run #3** — missing cmake for egl_probe.
- **Run #4** — bddl / future / gym cascade.
- **Run #5** — proper readiness polling, up to 5-min warmup budget.

Each run burned Modal time (~$5–8 per attempted episode). But the real cost was the serial discovery of a 9-element install-minefield chain.

---

## 18 commits in 75 minutes

The commit burst between Apr-16 14:50 and 16:05 contained the actual fight. This is the per-commit chronology (theme: LIBERO integration death march):

### The cascade

**14:50 `2d60d6d`** — Add `scripts/modal_libero10.py` (+338 LOC). First attempt. Ran, failed.

**14:56 `9cc3a14`** — Switch base image from `nvidia/cuda` to `debian_slim`. The `nvidia/cuda` image had a mirror hash mismatch on some deps. Symptom: `pip install` failed with hash verification error.

**15:02 `736ec03`** — CPU provider + `run_server` API compat for `vla-eval 0.1.0`.

**15:23 `bf0c9f5`** — Correct `vla-eval` API — `run_server` takes a class, write config YAML, pass `--no-docker`. Big rewrite (+71/-55 LOC).

**15:25 `af6acba`** — Accept `**kwargs` in model server `__init__`. `vla-eval` auto-injects parent args the adapter didn't know about.

**15:28 `766185f`** — Add LIBERO + robosuite to Modal image (was missing at runtime). Bddl error: *"every LIBERO episode fails with `ModuleNotFoundError: No module named 'bddl'`."* LIBERO's `setup.py` has `install_requires=[]` — the deps in its `requirements.txt` never land automatically.

**15:32 `c4c0ac2`** — Diagnose LIBERO import failure + retry install at runtime.

**15:43 `40f1933`** — git clone + `pip install -e` for LIBERO. `pip install from git` doesn't work because LIBERO's package has a non-standard install structure.

**15:44 `a38cfa6`** — Skip LIBERO import check during build (LIBERO's `__init__.py` reads from stdin). In Modal container: no stdin.

**15:48 `a189177`** — Set `LIBERO_DATA_DIR` + `LIBERO_BASE` env vars. These skip the stdin prompt in `__init__.py`. Theoretically. In practice, the prompt happens anyway because of older `.pyc` caches.

**15:50 `6cae528`** — Patch LIBERO `__init__.py` to replace `input()` with env var. LIBERO reads stdin during import for a "custom path wizard." No stdin in containers. Fix it at install time.

**15:51 `8862afc`** — Use `sed` to patch LIBERO `input()`. Simpler, but shell-quoting hell.

**15:53 `a9ef6c9`** — Replace all 3 LIBERO `input()` calls with `'n'` (decline custom path wizard). The 3rd call is the sneaky one — it's inside a nested conditional.

**15:54 `6ef91a3`** — Use python3 regex to patch LIBERO `input()` calls. Avoids shell quoting.

**15:55 `dd03edb`** — Separate `patch_libero.py` script. Avoids shell quoting issues entirely. Cleaner to maintain.

**15:56 `91e3fd0`** — Move `patch_libero.py` copy BEFORE LIBERO install step in image build. Ordering issue.

**15:57 `cb742db`** — Aggressive regex patch for ALL `input()` patterns + nuke .pyc caches. The `.pyc` cache pattern that hit Taptic is the same pattern hitting LIBERO here — old bytecode persists despite edits to source. Nuke and restart.

**15:58 `2d29065`** — Debug: dump LIBERO `__init__.py` lines 60–80 before patch. Verify the patch actually lands.

**16:00 `b72d2ca`** — Handle multi-line `input()` calls in LIBERO patch. Some prompts span three lines; the regex missed them.

**16:05 `2c597b6`** — Pin `robosuite==1.4.1` for LIBERO compat. robosuite 1.5+ moved module paths; LIBERO expects 1.4.1.

**Total commits: 18. Total time: 75 minutes. Total sim success: 0%.**

---

## Install minefield captured

The eventual working Modal image stanza lives in `scripts/modal_libero10.py`. Multi-paragraph post-mortem comment documents the full minefield:

```
# LIBERO's setup.py is install_requires=[]; its envs import bddl,
# robomimic, hydra-core at reset() time. Installing the full requirements.txt
# would downgrade transformers/numpy/etc. and nuke the ONNX export stack.
# Install only the runtime-required deps with flexible versions.
```

Pinned dependencies:
- `bddl==1.0.1` — LIBERO's BDDL (Boolean Domain Definition Language) task parser. Had to be pinned because the LIBERO version expects a specific API.
- `robosuite==1.4.1` — comment: *"robosuite 1.5+ moved module paths — pin 1.4.1 which LIBERO expects."*
- `gym` (not `gymnasium`) — *"LIBERO's venv.py uses gym not gymnasium."*
- `future` (py2/3 compat shim LIBERO still depends on).
- `robomimic`, `hydra-core>=1.1`, `easydict`, `einops`, `opencv-python-headless`.
- `h5py` — dataset I/O.

apt packages:
- `libosmesa6 libosmesa6-dev` — software MuJoCo rendering.
- `libegl1 libglvnd0` — for EGL alternative (kept even though unused, in case someone tries to flip back).
- `libgl1 libglib2 ffmpeg cmake build-essential` — standard cascade.

### osmesa > egl

From the script comment:
> *"Uses both osmesa and egl for MuJoCo rendering: osmesa for MuJoCo software rendering (EGL hangs silently on some debian_slim+NVIDIA combos with LIBERO; osmesa is reliable but slow)."*

Env var: `MUJOCO_GL=osmesa`.

This was itself a ~90-minute detour within the 75-minute commit burst. Transcript line 9100: *"Vla-eval got connected, LIBERO env started, 8× dataset warnings, then hung silently for 180s — killed by our idle-timeout guard. Classic MuJoCo EGL rendering hang. Trying MUJOCO_GL=osmesa instead."* Line 9147: *"Same silent hang with osmesa — so rendering backend isn't the cause. Something deeper in LIBERO env."* Line 9453: *"Sim is mid-first-episode (osmesa is slow). Last log at 21:18 UTC, ep0 should finish around 21:21 UTC."*

The pattern: **osmesa is slow but reliable; EGL is fast but hangs.** Keep osmesa.

### Trojan dependency: `num2words`

Transcript line 9574: *"Missing `num2words` dep for SmolVLM processor. Trivial fix."* Added to the LIBERO eval image. Not a LIBERO bug — a SmolVLM processor dep that only surfaces when the processor is exercised in the vla-eval request path.

### The `env.reset()` smoke test

Transcript line 9173: *"BREAKTHROUGH! 🎉 Smoke test: `env.reset()` works in 6.4s (not hung — buffering was the issue all along). First episode actually ran 150 steps: `[1/20]...ep0: FAIL (steps=150)` — model ran inference 150 times, just didn't complete task in the step budget."*

The breakthrough insight: **the hang was `subprocess.run(capture_output=True)` buffering, not the sim.** All `vla-eval` stdout/stderr was buffered inside the Modal function until the subprocess completed. Looked like container was hung. Fix: stream stdout line-by-line rather than capturing.

Same pipe-buffer lesson as `modal_e2e_demo.py`. Re-learned. **Third time.**

Step 3b smoke test in `modal_libero10.py`:
> *"If this hangs, vla-eval would hang too — fail fast here with a clear message."*

Baked-in post-mortem.

---

## Adapter wiring: `reflex.runtime.adapters.vla_eval`

Promoting the Modal script's inline inference into a proper library adapter was the parallel Apr-16 task (#18 complete). Previously, `scripts/modal_libero10.py` reimplemented the inference pipeline inline (denoising loop, VLM handling, action truncation). Fixing a bug meant editing a Modal script instead of the library.

From the transcript (line 8373): *"The real problem: scripts/modal_libero10.py reimplements the inference pipeline inline... The best long-term fix: promote the adapter into the library."*

Architecture:
1. `reflex.runtime.ReflexServer` — one class that owns vision encoder + text embedder + decoder prefill + expert denoising + action post-processing.
2. `reflex.runtime.adapters.VlaEvalAdapter` — thin wrapper (~20 LOC) that makes `ReflexServer` implement `vla-eval`'s `PredictModelServer` interface.
3. The Modal script becomes a 30-line runner: `reflex export → python -m reflex.runtime.adapters.vla_eval → vla-eval run`.

Modal log confirmed adapter came up in 12 s on port 8000:
```
Expert ONNX inputs: ['noisy_actions', 'timestep', 'position_ids', 'vlm_k', 'vlm_v']
VLM orchestrator loaded complete=True (vision_encoder.onnx + text_embedder.onnx + decoder_prefill.onnx — all CPU)
Loaded normalizer stats from 2 file(s): ['action_mean', 'action_std', 'state_mean', 'state_std']
Adapter config: export=/tmp/reflex_libero_export device=cuda out_dim=7 camera=<first> vlm=on norm=on
```

### Normalizer pipeline

Transcript line 9226: *"Found probable root cause: SmolVLA LIBERO checkpoint ships with `policy_preprocessor` (input normalizer) and `policy_postprocessor` (output unnormalizer) — we're not loading/applying either. Model expects normalized state, returns normalized actions. We feed raw state → get normalized actions → LIBERO interprets them as real joint values → failure."*

Two normalizer files in the HF repo:
- `policy_preprocessor_step_5_normalizer_processor.safetensors` (normalizes state)
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors` (un-normalizes actions)

Added normalizer support to the adapter — GOALS.yaml task #24. Modal log showed `norm=on` and 4 stats loaded (`action_mean`, `action_std`, `state_mean`, `state_std`).

Expected: apply preprocessor before inference, apply postprocessor after. Implementation landed.

---

## The flagship LIBERO-10 run (ap-QAG1Pk9w3DkuZnVs9VC8Ke)

Modal app `ap-QAG1Pk9w3DkuZnVs9VC8Ke` — created Apr-17 11:14 IST, stopped 11:26 IST (~12 min). This is the flagship LIBERO-10 eval run referenced by task #23.

### What worked

- **Step 1 Export** (`lerobot/smolvla_libero`): PASS in 173 s.
  - 3/3 VLM files.
  - 2/2 normalizer files.
- **Step 2 Adapter** (`reflex.runtime.adapters.vla_eval`): PASS in 12 s on port 8000.
  - Expert ONNX inputs: `['noisy_actions', 'timestep', 'position_ids', 'vlm_k', 'vlm_v']`.
  - VLM orchestrator loaded complete=True (vision + text_embedder + decoder_prefill — all CPU).
  - Normalizer stats loaded: 2 files, 4 keys.
- **Step 3 Config** written to `/tmp/libero_10_config.yaml`.
- **Step 3b env.reset() smoke**: PASS in 5.8 s, obs keys present.

### What failed

- **Step 4 vla-eval run**:
  - Task 1 (`put both the alphabet soup and the tomato sauce in the basket`) ep0 → **FAIL** (steps=150, max).
  - Task 2 (`put both the cream cheese box and the butter in the basket`) ep0 → **FAIL** (steps=150, max).
  - Tasks 3–10 never ran — user stopped from CLI before ep 3 started.
  - Extrapolating: if pattern held, task success would have trended near **0%**.

### Non-fatal warnings

- `datasets path /opt/LIBERO/libero/libero/../datasets does not exist!` (twice per ep) — LIBERO can init tasks without dataset dir; warning only.
- robosuite macros warnings.
- `gym` deprecation warnings.

---

## Result: infrastructure proven, task success 0%

Transcript summary (line 10268):

> *"Final data point: `0% with controller_states + per-layer vlm_kv + normalizer + all plumbing correct`. Actions are firmly in-distribution (within 2σ of training mean/std across all 7 dims). State flows. VLM pipeline complete. Everything measurable looks correct."*

The **infrastructure win is real**:
- Unified CLI (`reflex export` auto-produces 4 ONNX files for SmolVLA) ✅
- `reflex.runtime.adapters.vla_eval` adapter ✅
- Normalizer pipeline (state + action) ✅
- Per-layer `vlm_kv` ONNX export ✅
- 5D→4D pixel_values fix ✅
- LIBERO sim on Modal (bddl, gym, osmesa, robosuite pin) ✅

**LIBERO task success: 0% unresolved.**

The closing confession from the transcript (line 10630):
> *"Honest answer: I'm guessing at which subtle thing is wrong. Each fix I've made (5D pixel_values, normalizer, per-layer vlm_kv, layernorm on k, RoPE on keys, split k/v, newline on task, multi-camera, controller_states) was a real bug that would have made the model fail. Fixing them all and still getting 0% means there's ONE more thing — but I have 30+ candidates and no way to rank them without direct comparison."*
> *"The fundamental problem: task-success is an integration test. ONE subtly wrong operation out of ~500 in the pipeline = 0%. Without a side-by-side diff against the real PyTorch model, I'm iterating blind."*

The conclusion framed the Apr-17 session's methodology pivot: **stop guessing, start stage-diffing.**

---

## Lessons captured in the session

1. **pip-install-from-git doesn't work for all packages.** LIBERO needs git clone + `pip install -e .`. Same issue hit `lerobot` earlier.
2. **LIBERO has 3 `input()` calls that hang in containers.** Patch them with `patch_libero.py` regex.
3. **Nuke `.pyc` caches when patching Python source.** Old bytecode persists.
4. **osmesa > EGL for MuJoCo in containers** — EGL hangs silently on debian_slim + NVIDIA combos.
5. **`subprocess.run(capture_output=True)` buffers all output until subprocess completion.** Use streaming (`Popen` + line-by-line read) for long-running sims. Third re-learn.
6. **LIBERO's `setup.py` has empty `install_requires=[]`.** Install its deps explicitly — the requirements.txt would downgrade transformers / numpy / etc. and nuke the ONNX stack.
7. **Pin `robosuite==1.4.1`**; 1.5+ moved module paths.
8. **`bddl==1.0.1`**; LIBERO imports `bddl.parsing` at reset time.
9. **Env.reset smoke before vla-eval.** Fail fast before sitting on a 40-min sim run.

All captured in `scripts/modal_libero10.py` comment block and `scripts/patch_libero.py` regex patcher.

---

## Carry-over

Unfinished work from this session that lives in `2026-04-17_libero_correctness_hunt.md`:

- **LIBERO-10 task success > 0%** — the actual number. Apr-16 established infrastructure, Apr-17 hunted bugs in the exporter correctness.
- **PyTorch-vs-ONNX stage diff** — the "decisive test" script (`scripts/modal_pytorch_vs_onnx.py`) designed but not fully run until Apr-17.
- **Per-layer `vlm_kv` ONNX export** (task #25) — the stage-diff runs on Apr-17 drive this. Layer 0 `v` projection was the outlier (cos=0.9117).
- **VLM weight loading bug** — *"Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)"* — flagged as known gap. Turned out to be the headline bug on Apr-17 (base SmolVLM2 loaded instead of fine-tuned checkpoint weights). Fixed via `AutoModelForImageTextToText` unwrap.

The Apr-16 session delivered infrastructure. Apr-17 tried to deliver correctness — it got closer (single-layer self-attn matches to 1e-5) but not over the line. See `2026-04-17_libero_correctness_hunt.md`.
