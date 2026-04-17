# LIBERO / vla-eval Integration Bugs

Bugs specific to integrating the Reflex runtime with LIBERO (Stanford's Boolean Domain Definition Language benchmark) via the AllenAI `vla-eval` harness. Each bug dragged the LIBERO-10 Modal run through a death march of ~18 commits in 75 minutes (Apr-16 14:50–16:05) plus an earlier 9-commit burst on trajectory replay. The final working path is now codified in `scripts/modal_libero10.py` and `scripts/patch_libero.py`.

Sources referenced: `sessions_md.md`, `modal_scripts.md`, `git_history.md`, `current_session.md`, `prior_sessions.md`, `modal_apps_and_pm_docs.md`.

---

## Bug 1: `bddl` missing — LIBERO `setup.py` has empty `install_requires=[]`

**Symptom:** Every LIBERO episode fails at `env.reset()` with `ModuleNotFoundError: No module named 'bddl'`. All inference/export/server plumbing works — sim side missing a dependency.

**Root cause:** LIBERO uses Stanford's BDDL (Boolean Domain Definition Language) for task specifications. `bddl` is on PyPI. But LIBERO's `setup.py` has `install_requires=[]` (empty list!) — so `pip install -e .` on LIBERO does NOT pull any of its dependencies. The stated dependencies live in `requirements.txt` which nothing automatically installs. Running the full `requirements.txt` would downgrade transformers / numpy / torch etc. and nuke the ONNX export stack.

**Fix:** Install ONLY the runtime-required deps explicitly in the Modal image — pin `bddl==1.0.1`, install narrowly: `bddl==1.0.1, robosuite==1.4.1, future, robomimic, hydra-core>=1.1, easydict, einops, opencv-python-headless, gym, h5py`. Flexible versions where possible; hard-pins only for compatibility-broken deps.

```python
# scripts/modal_libero10.py image construction comment:
# "LIBERO's setup.py is install_requires=[]; its envs import bddl, robomimic, hydra-core
#  at reset() time. Installing the full requirements.txt would downgrade
#  transformers/numpy/etc. and nuke the ONNX export stack. Install only the
#  runtime-required deps with flexible versions."
```

**Sources:**
- `modal_scripts.md` `modal_libero10.py` section (multi-paragraph post-mortem in image comment)
- `current_session.md` lines 8596-8776 ("bddl==1.0.1 is on pypi. LIBERO's setup.py has empty install_requires=[]")
- `sessions_md.md` line 21 ("Run #4 bddl/future/gym cascade")
- `git_history.md` commit `766185f` ("fix: add LIBERO + robosuite to Modal image (was missing at runtime)")

---

## Bug 2: LIBERO `input()` prompts hang on import

**Symptom:** Modal container hangs silently during image build. No stdin attached — `input()` blocks forever. Build times out after 1800s with no useful error.

**Root cause:** LIBERO's `__init__.py` reads from stdin during import for a "custom path wizard" that asks the user whether they want to configure custom data directories. There are **3 separate `input()` calls**, some spanning **multiple lines** (the prompt, then the reading). None of these work in a containerized Modal function.

**Fix:** Use a separate `scripts/patch_libero.py` script with aggressive regex patching to replace ALL `input(...)` calls with the string `'n'` (declining the wizard). Copy this patch before installing LIBERO in the image build. Key details:
- Set `LIBERO_DATA_DIR` and `LIBERO_BASE` env vars so the wizard doesn't try to prompt.
- Use a Python regex (not shell sed — too much quoting hell) to handle multi-line patterns.
- Nuke `.pyc` caches after patching so the old bytecode doesn't run.
- Place patch_libero.py BEFORE the LIBERO install step in the Modal image.

The commits to get this right span 11 steps: `6cae528`, `8862afc`, `a9ef6c9`, `6ef91a3`, `dd03edb`, `91e3fd0`, `cb742db`, `2d29065`, `b72d2ca`, `a38cfa6`, `a189177`.

**Sources:**
- `git_history.md` "LIBERO-10 benchmark via vla-eval adapter — long debugging march" theme
- `sessions_md.md` line 26 (".pyc cache / 4th input() call")
- `current_session.md` related to LIBERO init hang
- `modal_scripts.md` `modal_libero10.py` ("patch_libero.py to fix interactive input prompts that hang on import")

---

## Bug 3: `gym` vs `gymnasium` — LIBERO uses old `gym`

**Symptom:** `AttributeError` or `ImportError` in LIBERO env init.

**Root cause:** LIBERO's `venv.py` imports `gym` (the OLD pre-split package), not `gymnasium`. Modern Python ML stacks have moved to `gymnasium`. Installing only `gymnasium` breaks LIBERO. Installing only `gym` breaks anything else that imports `gymnasium`.

**Fix:** Install BOTH packages in the Modal image. `gym` is needed by LIBERO's `venv.py`; `gymnasium` stays for anything else that imports it. `scripts/modal_libero10.py` explicitly comments "gym (old gym — LIBERO's venv.py uses gym not gymnasium)".

**Sources:**
- `modal_scripts.md` `modal_libero10.py` pip stack ("gym (old gym — LIBERO's venv.py uses gym not gymnasium)")
- `sessions_md.md` line 21 ("Run #4 bddl/future/gym cascade")

---

## Bug 4: `robosuite==1.4.1` hard version pin

**Symptom:** LIBERO env init fails with `ImportError` or `AttributeError` from `robosuite` — some module paths changed.

**Root cause:** `robosuite 1.5+` moved module paths (e.g. controller registration, env wrappers). LIBERO was built against `robosuite 1.4.1` and hardcodes the old paths. Installing a newer robosuite (even via pypi latest) silently breaks LIBERO at env import.

**Fix:** Pin `robosuite==1.4.1` in the Modal image. See commit `2c597b6` ("fix: pin robosuite==1.4.1 for LIBERO compat (1.5+ changed module paths)").

**Sources:**
- `git_history.md` commit `2c597b6`
- `modal_scripts.md` `modal_libero10.py` ("robosuite 1.5+ moved module paths — pin 1.4.1 which LIBERO expects.")

---

## Bug 5: `osmesa` vs EGL — rendering hang silently

**Symptom:** vla-eval connected, LIBERO env started, 8× dataset warnings, then hung silently for 180s. Killed by idle-timeout guard. No error message — process just stops producing output.

**Root cause:** Classic MuJoCo EGL rendering hang on `debian_slim + NVIDIA` combos with LIBERO. EGL driver is not reliably available in the Modal environment; MuJoCo's EGL backend blocks instead of erroring. `osmesa` (software renderer) is reliable but slow.

**Fix:** Set `MUJOCO_GL=osmesa` environment variable + install `libosmesa6` and `libosmesa6-dev` via apt. Install both osmesa AND egl deps (libegl1, libglvnd0) so MuJoCo can fall back if osmesa has its own issue.

```python
# scripts/modal_libero10.py apt deps:
# git, libgl1, libglib2, libegl1, libglvnd0, ffmpeg, cmake,
# build-essential, libosmesa6, libosmesa6-dev
```

"osmesa first-scene compilation can be slow" — bump idle-timeout to 600s.

**Sources:**
- `current_session.md` line 9100 ("Classic MuJoCo EGL rendering hang. Trying MUJOCO_GL=osmesa (software renderer) instead.")
- `current_session.md` line 9173 ("BREAKTHROUGH! env.reset() works in 6.4s (not hung — buffering was the issue all along)")
- `modal_scripts.md` `modal_libero10.py` ("Uses both osmesa and egl for MuJoCo rendering: 'osmesa for MuJoCo software rendering (EGL hangs silently on some debian_slim+NVIDIA combos with LIBERO; osmesa is reliable but slow).'")

---

## Bug 6: `send_state=True` / `send_wrist_image=True` missing from vla-eval config

**Symptom:** 100% of episodes fail at step 150 — max steps — with no progress toward task. Model outputs garbage even after all pipeline bugs fixed.

**Root cause:** Diagnostic dump revealed:
```
obs schema: top_keys=['images', 'task_description']
images=agentview:(256, 256, 3)/uint8
state=none ⚠️
```

vla-eval was NOT sending robot state at all, because our LIBERO config didn't set `send_state=True`. The model got `state=None` → server passed `zeros(6)` → the model had no idea where the robot was → garbage actions. This was EVEN WITH all pipeline bugs fixed. Also `send_wrist_image=True` was missing, meaning the wrist camera was never sent. The model was trained on 3 cameras (agentview + wrist + something else) but was getting 1 camera + nothing.

**Fix:** Set both in the LIBERO config YAML:
```yaml
send_state: true
send_wrist_image: true
```

`scripts/modal_libero10.py` comments: "CRITICAL: without these, vla-eval sends only images + task_description. Our first-predict dump showed state=none, which means the model was predicting actions from zero state vectors — garbage trajectories no matter what the VLM pipeline looked like."

**Sources:**
- `current_session.md` line 10117 ("FOUND THE BUG! The diagnostic dump reveals... state=none ⚠️")
- `modal_scripts.md` `modal_libero10.py` section ("send_state=True, send_wrist_image=True has a pointed comment...")

---

## Bug 7: `states` vs `controller_states` — which state is the model trained on?

**Symptom:** State flows now (8D float64), both cameras arriving (`agentview` + `wrist`). Actions in-distribution. Still 0%.

**Root cause:** LIBERO's obs contains BOTH `states` AND `controller_states` keys. The model may have been trained on `controller_states` (the robot controller's output — what the controller commanded) NOT `states` (the raw env observation — what actually happened). These differ when the robot is in contact with objects or at joint limits. Picking the wrong one silently corrupts the state distribution.

**Fix:** Try `controller_states` first if present; fall back to `states`. Still open candidate — neither choice has been proven to give non-zero task success yet. See Bug 13 in `smolvla_pipeline_bugs.md` for the related numpy-array truth-value issue when using `obs.get() or obs.get()`.

**Sources:**
- `current_session.md` line 10152 ("obs has BOTH states AND controller_states. Model may have been trained on controller_states (robot controller's output) not states (raw env)")
- `sessions_md.md` line 110 ("states vs controller_states in LIBERO obs — model may have been trained on controller_states...")

---

## Bug 8: `multi-line input()` regex in `patch_libero.py`

**Symptom:** Even after patching `input()` calls with a simple regex, the container STILL hung on import. The 4th `input()` call was being missed.

**Root cause:** LIBERO has at least one `input()` call that spans MULTIPLE LINES — the prompt string is on line N, the `input(...)` invocation is on lines N+1 through N+2 with continuation. A single-line regex can't catch this. ALSO: `.pyc` caches from previous imports kept the old code running even after the .py was patched.

**Fix:** `scripts/patch_libero.py` uses a multi-line-aware Python regex that matches `input(...)` spanning up to 3 lines. Commit `b72d2ca` explicitly fixes "handle multi-line input() calls in LIBERO patch (split across 3 lines)". Additionally `cb742db` nukes `.pyc` caches aggressively after patching. Commit `2d29065` dumps LIBERO's `__init__.py` lines 60-80 before patch for debugging.

Workflow:
1. Copy `patch_libero.py` into image BEFORE LIBERO install.
2. Clone LIBERO → `pip install -e .`.
3. Run `patch_libero.py` against the installed LIBERO path.
4. Nuke all `__pycache__/` directories under LIBERO.
5. Import should succeed with `LIBERO_DATA_DIR + LIBERO_BASE` env vars set.

**Sources:**
- `git_history.md` commits `6cae528`, `8862afc`, `a9ef6c9`, `6ef91a3`, `dd03edb`, `91e3fd0`, `cb742db`, `2d29065`, `b72d2ca`, `a38cfa6`, `a189177`
- `sessions_md.md` line 26 (".pyc cache / 4th `input()` call — old code was running despite edits because cached bytecode persisted. 'Must be a `.pyc` cache or a 4th call' — had to nuke caches + add catch-all.")
- `modal_scripts.md` `modal_libero10.py` (patch_libero.py workflow)

---

## Bug 9: `subprocess.run(capture_output=True)` buffering on long-running vla-eval

**Symptom:** Modal function appears hung for 37 min with no stdout / stderr output. Actually the vla-eval subprocess IS running — the capture buffer just waits until the subprocess exits before flushing.

**Root cause:** `subprocess.run(capture_output=True)` buffers all of the subprocess's stdout/stderr until the subprocess completes. Inside a Modal function that wraps `vla-eval run`, which can take 40-60 minutes for a full LIBERO-10 suite, this means ZERO telemetry for the entire duration. You cannot tell if the run is healthy vs stuck.

**Fix:** Stream stdout line-by-line instead:
```python
# Use Popen + read() in a select loop:
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
for line in process.stdout:
    print(line.rstrip())
```

Also add an **idle-timeout guard**: kill vla-eval if no stdout appears for 600s (10 min). This catches cases where osmesa first-scene compilation truly does take many minutes vs real hangs.

**Sources:**
- `sessions_md.md` line 22 ("`subprocess.run(capture_output=True)` buffering — inside the Modal function this buffered all vla-eval stdout/stderr until the subprocess completed. Looked like the container was hung (37 min with no output). Fix: stream stdout line-by-line rather than capturing.")
- `sessions_md.md` line 24 ("`capture_output` output only lands on completion — same buffering class as above; means you can't tell if a 40-min LIBERO run is healthy.")
- `modal_scripts.md` `modal_libero10.py` ("Streams vla-eval stdout line-by-line via select: 'subprocess.run(capture_output=True) ... buffers until the subprocess exits, which meant we couldn't tell if a run was hung vs mid-episode for 50+ minutes.'")

---

## Bug 10: vla-eval API version compat — `run_server` signature changed

**Symptom:** vla-eval Python API errors: `TypeError: run_server() takes N positional arguments but M were given`. Different vla-eval versions (0.1.0 vs 0.1.1+) changed the signature.

**Root cause:** Early in LIBERO integration, we called vla-eval's `run_server` as if passing a model INSTANCE. In `vla-eval 0.1.0+` it takes a CLASS and instantiates internally with parent args auto-injected as kwargs. The model class must accept `**kwargs` in `__init__` to absorb those parent args.

**Fix:**
1. Pass a class, not an instance: `run_server(ReflexVlaEvalAdapter, config)`.
2. Accept `**kwargs` in the adapter's `__init__`.
3. Write a proper `config.yaml` and pass `--no-docker` to skip Docker-based deployment.

Commits `736ec03`, `bf0c9f5`, `af6acba` fix these in sequence.

**Sources:**
- `git_history.md` commits `736ec03`, `bf0c9f5`, `af6acba`
- `modal_apps_and_pm_docs.md` ap-QAG1Pk9w... (adapter PASS in 12s with correct API)

---

## Bug 11: `cmake` missing for `egl_probe` build

**Symptom:** LIBERO install fails during a transitive dep install (`egl_probe` requires CMake).

**Root cause:** `egl_probe` is a C++ package for detecting EGL availability. It builds from source via CMake. The debian_slim base image doesn't have `cmake` or `build-essential` by default.

**Fix:** Install `cmake`, `build-essential`, `libegl1`, `libglvnd0` as apt deps in the Modal image BEFORE the LIBERO pip install.

**Sources:**
- `sessions_md.md` line 21 ("Run #3 missing cmake for egl_probe")
- `modal_scripts.md` `modal_libero10.py` apt deps

---

## Bug 12: Wrong base image — `nvidia/cuda` mirror hash mismatch

**Symptom:** Modal image build fails at apt update with hash mismatch errors on some debian mirror entries.

**Root cause:** The `nvidia/cuda:*` base images had intermittent mirror hash mismatches during the Apr-16 window. Debian security mirrors went out of sync with the image's apt snapshot.

**Fix:** Use `debian_slim` as the base for LIBERO instead of `nvidia/cuda`. For GPU acceleration elsewhere, we use `nvcr.io/nvidia/tensorrt:24.10-py3` (which has its own working apt sources). Commit `9cc3a14` applies this.

**Sources:**
- `git_history.md` commit `9cc3a14` ("fix: use debian_slim base for LIBERO (nvidia/cuda had mirror hash mismatch)")

---

## Bug 13: `pip install git+...` for LIBERO doesn't work

**Symptom:** `pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git` succeeds, but `import libero` at runtime fails with `ModuleNotFoundError`.

**Root cause:** LIBERO's `setup.py` has an unusual layout — it declares the package but doesn't package the submodules correctly for pip's install-from-git. The install LOOKS successful but the package isn't actually on the PYTHONPATH in a usable way.

**Fix:** `git clone` LIBERO to a known path + `pip install -e .` from that path. Keep the clone around so debugging can inspect file locations. Commit `40f1933` applies this.

**Sources:**
- `git_history.md` commit `40f1933` ("fix: git clone + pip install -e for LIBERO (pip install from git doesn't work)")
- `sessions_md.md` line 25 ("lerobot package install-structure quirk" — SAME pattern for lerobot)

---

## Bug 14: Skipping LIBERO import check at image build time

**Symptom:** Image build fails because LIBERO's import wants stdin (even with patch).

**Root cause:** During image build, Modal runs `RUN python -c "import libero"` as a sanity check. Even after patching `input()` calls, the import still wants to read from a (non-existent) stdin briefly to confirm the wizard decline.

**Fix:** Skip the LIBERO import check during image build (`a38cfa6`). Let the import happen at runtime when env vars are set and patch has been applied. Defer validation to the `env.reset()` smoke test at runtime.

**Sources:**
- `git_history.md` commit `a38cfa6` ("fix: skip LIBERO import check during build (reads from stdin)")
- `git_history.md` commit `a189177` ("fix: set LIBERO_DATA_DIR + LIBERO_BASE env vars (skips stdin prompt)")

---

## Bug 15: `env.reset()` smoke test must run before `vla-eval`

**Symptom:** vla-eval run STARTS successfully but hangs on first `env.reset()` with no clear error, burning 60+ minutes of Modal compute.

**Root cause:** LIBERO's `env.reset()` does a lot of file loading and scene compilation. If any of those steps fails silently (missing dataset files, wrong osmesa config), vla-eval won't surface the error — it'll just appear hung. By the time you notice, you've burned lots of compute.

**Fix:** Before running vla-eval, run a standalone subprocess smoke test: `env.reset()` alone, with a 60s timeout. If it hangs, fail fast with a clear message. `scripts/modal_libero10.py` comments: "If this hangs, vla-eval would hang too — fail fast here with a clear message."

```python
# Step 3b in modal_libero10.py:
smoke_test_result = subprocess.run(
    [sys.executable, "-c", "from libero.libero import benchmark, env; ..."],
    timeout=60
)
if smoke_test_result.returncode != 0:
    raise RuntimeError("env.reset() smoke failed — vla-eval would hang")
```

**Sources:**
- `modal_scripts.md` `modal_libero10.py` ("Step 3b: LIBERO `env.reset()` **smoke test** in a subprocess before running vla-eval")
- `current_session.md` line 9173 ("env.reset() works in 6.4s (not hung — buffering was the issue all along)")

---

## Bug 16: vla-eval uses WebSocket+msgpack, not HTTP

**Symptom:** Initial attempt to point vla-eval at Reflex's FastAPI `/act` endpoint failed — vla-eval didn't understand the HTTP JSON responses.

**Root cause:** vla-eval (AllenAI's VLA benchmark harness) speaks WebSocket + msgpack as its model-server protocol, NOT HTTP + JSON. Reflex's `reflex serve` is HTTP-first. We had to build a thin adapter that translates.

**Fix:** Build `src/reflex/runtime/adapters/vla_eval.py` — `ReflexVlaEvalAdapter` class that implements vla-eval's `PredictModelServer` interface on top of `ReflexServer`. ~20 LOC thin wrapper. vla-eval launches this adapter as a subprocess on a known port; Modal script starts the adapter, then runs vla-eval CLI pointed at it.

**Sources:**
- `sessions_md.md` line 32 ("vla-eval uses WebSocket+msgpack, not HTTP — had to build a thin adapter (reflex.runtime.adapters.vla_eval).")
- `sessions_md.md` decision "Wrap, don't rebuild vla-eval"
- `git_history.md` task #18 ("Build reflex.runtime.adapters.vla_eval")
- `modal_apps_and_pm_docs.md` ap-QAG1Pk9w ("Adapter (reflex.runtime.adapters.vla_eval): PASS in 12s on port 8000")

---

## Bug 17: LIBERO sends 1 image, SmolVLA trained on 3

**Symptom:** Model acts "reasonably" but can't solve tasks. Agent chooses wrong objects, misses approach angles.

**Root cause:** vla-eval/LIBERO default sends only 1 image (agentview camera, the first in its list), but SmolVLA-LIBERO was trained on 3 cameras keyed `camera1/2/3`. Our VLM pipeline runs with 1-camera input — very different distribution from training. Even though `send_wrist_image=True` (Bug 6 fix) got 2 cameras flowing, the 3rd camera is missing.

**Fix:** Open. Workarounds:
- Duplicate the agentview image across all 3 camera slots (not ideal — loses spatial info).
- Check if a 3rd camera is available in LIBERO's obs (`agentview`, `wrist`, `eye_in_hand`?) and route accordingly.

**Sources:**
- `sessions_md.md` line 108 ("LIBERO sends 1 image (first camera) while SmolVLA was trained on 3 cameras (camera1/2/3). Camera mismatch is candidate for remaining 0%.")
- `current_session.md` line 10679 ("vla-eval sends ONE camera (agentview) but SmolVLA-LIBERO was trained on THREE cameras (camera1/2/3)")

---

## Bug 18: Normalizer files — `policy_preprocessor` + `policy_postprocessor`

**Symptom:** Model state-flows correctly, actions in-distribution numerically, but LIBERO environment rejects them. Robot doesn't move to the right pose.

**Root cause:** The SmolVLA LIBERO checkpoint ships with TWO normalizer files in its HF repo:
- `policy_preprocessor_step_5_normalizer_processor.safetensors` — normalizes state input (mean/std)
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors` — un-normalizes action output

The pipeline wasn't loading/applying EITHER. Model expected normalized state, returned normalized actions. We fed raw state → got normalized actions → LIBERO interpreted them as real joint values → failure.

**Fix:** Task #24 "Add normalizer support to adapter" — partially shipped. `reflex export` now exports these files alongside. Adapter loads them at startup and applies:
1. State: `(raw_state - state_mean) / state_std` before ONNX call.
2. Actions: `(onnx_action * action_std) + action_mean` after ONNX.

Modal log confirms `norm=on` and `Loaded normalizer stats from 2 file(s): ['action_mean', 'action_std', 'state_mean', 'state_std']`.

**Sources:**
- `current_session.md` line 9226 ("Found probable root cause: SmolVLA LIBERO checkpoint ships with `policy_preprocessor` (input normalizer) and `policy_postprocessor` (output unnormalizer) — we're not loading/applying either.")
- `modal_apps_and_pm_docs.md` ap-QAG1Pk9w ("Loaded normalizer stats from 2 file(s)")
- TaskList #24 "Add normalizer support to adapter" (in_progress)

---

## Bug 19: `datasets path /opt/LIBERO/libero/libero/../datasets does not exist!`

**Symptom:** LIBERO prints warning twice per episode: `datasets path /opt/LIBERO/libero/libero/../datasets does not exist!`.

**Root cause:** LIBERO's task initialization looks for a datasets directory for some tasks, but LIBERO-10 doesn't actually need it. The warning is non-fatal — the tasks init correctly without the dataset dir.

**Fix:** Ignore the warning. It does not affect task-success. Known-benign noise in logs.

**Sources:**
- `modal_apps_and_pm_docs.md` ap-QAG1Pk9w ("Warnings: `datasets path /opt/LIBERO/libero/libero/../datasets does not exist!` (twice per ep; non-fatal since LIBERO can init tasks without the dataset dir)")

---

## The LIBERO Run Ladder (chronological)

Captured in `current_session.md` line 8776 — the sequence of failed runs before a working one:

- **Run #1** — 0% because VLM files weren't in export (fixed by CLI Step 1 unifying `reflex export`)
- **Run #2** — killed (requirements.txt nuked the stack)
- **Run #3** — missing cmake for egl_probe
- **Run #4** — bddl/future/gym cascade
- **Run #5** (working) — proper readiness polling, up to 5min warmup budget, all deps pinned narrowly

Each run represented a ~$1-3 Modal spend, plus 20-60 minutes of wall-clock time.

## Final status at session close

Per `current_session.md` line 10268 and `modal_apps_and_pm_docs.md` ap-QAG1Pk9w:

**Infrastructure PASS:**
- Unified CLI (reflex export auto-produces 4 ONNX files for SmolVLA) ✅
- vla-eval adapter (ReflexVlaEvalAdapter) ✅
- Normalizer pipeline (state + action) ✅
- Per-layer vlm_kv ONNX export (partial) ⚠
- 5D→4D pixel_values fix ✅
- LIBERO sim on Modal (bddl, gym, osmesa, robosuite pin, patch_libero) ✅

**LIBERO task success: 0%** unresolved. Task 1 ep0 FAIL (steps=150 max), Task 2 ep0 FAIL (steps=150 max), user stopped run before Task 3.

## Files
- `scripts/modal_libero10.py` — the working Modal script with all image fixes
- `scripts/patch_libero.py` — the regex input()-patcher
- `src/reflex/runtime/adapters/vla_eval.py` — the WebSocket adapter
- `src/reflex/eval/libero.py` — the benchmark adapter stub (for `reflex bench --benchmark libero_10`)

