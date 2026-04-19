# Launch verification gates — the 9 tests that guard v0.2 launch

**Purpose.** Nine regression gates must pass for v0.2 to launch. Each gate catches a specific class of ship-breaker that slipped past unit tests in earlier development. This doc explains what each gate guards against and why it was added — so a future session that sees one fail knows whether it's load-bearing or skippable.

**Current state (2026-04-19).** All 9 green.

**Philosophy.** The gates are intentionally *end-to-end*, not unit tests. They exercise the full chain from a fresh install through to HTTP `/act` returning a 200 with a well-shaped action payload. Unit tests live in the same `tests/` directory but aren't in this list.

---

## Gate 1 — `fresh-install`

**Test file:** `tests/test_fresh_install.py` • **Last run:** `reflex_context/fresh_install_last_run.json`

**What it guards.** `pip install 'reflex-vla[serve,gpu] @ git+...'` into a clean Python 3.12 venv succeeds, then `reflex --help` lists every subcommand. Catches dependency resolution bugs, missing `__init__.py` entries, broken optional extras.

**Why it exists.** During the 2026-04-10 → 2026-04-17 arc the repo accumulated enough pyproject.toml optional-deps churn that a fresh install silently dropped one wedge on some Python minor versions. The fix was to explicitly enumerate the `[serve]`, `[onnx]`, `[gpu]`, `[ros2]` extras and pin their minimums. This gate catches regressions in that pinning.

**Reproducer:** `modal run scripts/modal_fresh_install.py` (runs pip install in a clean Modal container + invokes `reflex --help`). Pass = exit 0 + every expected subcommand in stdout.

---

## Gate 2 — `cuda-runtime` (CUDAExecutionProvider parity vs CPU)

**Test file:** `tests/test_cuda_runtime_parity.py` • **Last run:** `reflex_context/cuda_runtime_last_run.json`

**What it guards.** `reflex serve --device cuda` actually uses CUDAExecutionProvider and not a silent CPU fallback. Compares ONNX output on CUDA vs CPU with identical seeded input; expects cos ≥ 0.9999 (float32 kernel-level non-determinism is tolerated; full-path dropout is not).

**Why it exists.** onnxruntime-gpu's EP fallback is quiet — if cuDNN can't load, CUDAExecutionProvider just de-registers and CPUExecutionProvider runs instead, with no error. We had one incident where a Modal image was missing `libcudnn_adv.so.9`; `reflex serve` claimed `inference_mode=onnx_trt_fp16` but was actually running on CPU. This gate explicitly verifies the provider used is CUDA.

**Latest numbers (2026-04-18):**
- pi0: cos=0.99999994, max_abs=8.74e-04, used_provider=CUDAExecutionProvider
- SmolVLA: cos=0.99999961, max_abs=1.19e-03, used_provider=CUDAExecutionProvider

**Reproducer:** `modal run scripts/modal_pi0_monolithic_export.py --cuda` (and the SmolVLA variant).

---

## Gate 3 — `num-steps-quality`

**Test file:** `tests/test_num_steps_quality.py` • **Last run:** `reflex_context/num_steps_quality_last_run.json`

**What it guards.** The num_steps=10 monolithic ONNX matches PyTorch `sample_actions(num_steps=10)` at machine precision. After the 2026-04-19 three-patch fix, this gate is the core correctness claim for v0.2.

**Why it exists.** Pre-fix, num_steps=10 ran at cos=0.977 (pi0) — a 2% drift that would silently break customers. This gate ensures any regression in the 3-patch stack (F.pad mask, frozen DynamicLayer.update, past_kv.get_seq_length) is caught before launch. It also covers num_steps=1 as a lower-quality fallback signal.

**Latest (2026-04-19):**
- pi0 num_steps=10: first-action cos=+1.000000, max_abs=2.09e-07 ✅
- pi0.5 num_steps=10: first-action cos=+1.000000, max_abs=2.38e-07 ✅
- SmolVLA num_steps=10: first-action cos=+1.000000, max_abs=5.96e-07 ✅
- GR00T N1.6 4-step loop: first-action cos=+1.000000, max_abs=4.77e-07 ✅

**Reproducer:** `modal run scripts/modal_pi0_monolithic_export.py --parity --num-steps 10` (and variants for each model).

---

## Gate 4 — `docker-smoke`

**Test file:** `tests/test_docker_image_smoke.py` + `.github/workflows/docker-smoke.yml` • **Last run:** `reflex_context/docker_smoke_last_run.json`

**What it guards.** `docker build` on the published Dockerfile succeeds. `docker run ghcr.io/rylinjames/reflex-vla:latest <subcmd>` executes each documented CLI subcommand (help, models, targets, doctor) and returns exit 0.

**Why it exists.** The Dockerfile uses `ENTRYPOINT ["reflex"]` so `docker run image serve ./export` works cleanly. But that entrypoint intercepts `docker run image python ...` which is what a naive "does it import?" smoke test would do. Gate 4 was re-architected via GitHub Actions workflow (not Modal subprocess) to bypass the intercept — see `02_bugs_fixed/modal_deployment_gotchas.md` "Docker ENTRYPOINT intercepting subprocess exec."

**Reproducer:** push a commit; GH Actions runs the workflow. Local repro: `docker build -t test-reflex . && docker run test-reflex --help`.

---

## Gate 5 — `ros2-bridge-live`

**Test file:** `tests/test_ros2_bridge_live.py` • **Last run:** `reflex_context/ros2_live_last_run.json`

**What it guards.** `reflex ros2-serve` subscribes to `sensor_msgs/Image` + `sensor_msgs/JointState` + `std_msgs/String`, inferences through the ONNX, and publishes `std_msgs/Float32MultiArray` action chunks at the configured Hz. Real `rclpy`, real ros2 humble container — not mocked.

**Why it exists.** There's a mocked `tests/test_ros2_bridge.py` that verifies the ROS2 bridge class logic (8 tests, all green, no rclpy). But mocked tests can't catch rclpy-version-skew or ROS2-environment-init issues that only surface with real rclpy. This gate was the hardest-to-construct of the nine — five image-build iterations before the v5 recipe (ubuntu:22.04 + apt-install ros-humble-ros-base + numpy<2.0 constraint) worked. Details in `02_bugs_fixed/modal_deployment_gotchas.md` "ROS2 image construction — five attempts."

**Reproducer:** `modal run scripts/modal_ros2_live_test.py`. Pass = action chunks on `/reflex/actions` within 10s of publishing to `/camera/image_raw`.

---

## Gate 6 — `cli-export-end-to-end`

**Test file:** `tests/test_cli_export_end_to_end.py`

**What it guards.** `reflex export --monolithic <hf_id> --output ./out` (with model auto-detection from the HF ID) routes to the correct exporter, writes `model.onnx` + `reflex_config.json` + `VERIFICATION.md`, and sets `export_kind=monolithic` in the config. Exercises the dispatcher in `src/reflex/exporters/monolithic.py` added over commits `40ce929`, `c604962`, `4090e76`.

**Why it exists.** Prior CLI used per-model flags (`reflex export-pi0`, `reflex export-smolvla`). The monolithic dispatcher was added so `reflex export --monolithic <hf_id>` auto-detects the model type (substring matching on the HF ID) and calls the right exporter. This gate ensures the dispatcher stays correct across model_type additions — when we add xVLA or OpenVLA to `[smolvla, pi0, pi05, gr00t]`, this gate catches routing errors.

**Reproducer:** `pytest tests/test_cli_export_end_to_end.py -v`.

---

## Gate 7 — `serve-act-roundtrip`

**Test files:** `tests/test_serve_act_roundtrip.py` + `tests/test_serve_http_roundtrip.py`

**What it guards.** Boot `create_app()` with a monolithic export config, POST `/act` with a base64-encoded image + instruction + state, verify:
1. HTTP 200 returned
2. Response body has `actions` (50 × action_dim), `num_actions`, `latency_ms`, `denoising_steps`, `inference_mode`
3. Actions pass the Guard (no NaN/Inf)
4. Response time is under 30s even with stub ORT (wiring, not latency)

**Why it exists.** Earlier evidence of `/act` working was via `Pi0OnnxServer.predict()` direct invocation — that bypasses the FastAPI layer (request parsing, base64 decode, async wrapping, JSON response). A customer hitting our HTTP endpoint uses the full chain. This gate covers the wiring from HTTP request byte stream → ORT run → JSONResponse byte stream. ORT sessions are stubbed (we don't need a real 13 GB ONNX for wiring validation); numerical correctness is covered by gate 3.

**Reproducer:** `pytest tests/test_serve_http_roundtrip.py -v` (3 tests, all green).

---

## Gate 8 — `runtime-num-steps-10`

**Test file:** `tests/test_runtime_num_steps_10.py`

**What it guards.** `Pi0OnnxServer` and `SmolVLAOnnxServer` both read `num_denoising_steps` from `reflex_config.json` at startup (not hardcoded). `create_app` dispatches to the correct runtime server class based on `export_kind: monolithic` + `model_type` in the config.

**Why it exists.** The runtime server classes (`Pi0OnnxServer`, `SmolVLAOnnxServer`) predated the num_steps=10 unroll — their `predict()` methods assumed num_steps=1 and hardcoded it. When num_steps=10 exports started shipping, the server silently ignored the `num_denoising_steps` field and kept reporting `denoising_steps: 1` in `/act` responses. This gate ensures the server surfaces the correct value and that future runtime refactors don't regress.

**Reproducer:** `pytest tests/test_runtime_num_steps_10.py -v`.

---

## Gate 9 — `guard-trip-integration`

**Test file:** `tests/test_guard_trip_integration.py`

**What it guards.** When `ActionGuard.tripped == True`:
- `POST /act` returns a `guard_tripped` error dict (HTTP 503), NOT actions
- `GET /guard/status` reports `tripped: true` and the trip reason
- `POST /guard/reset` clears state; subsequent `/act` succeeds

**Why it exists.** The unit tests in `tests/test_guard.py` verify `ActionGuard.check()` in isolation. But the HTTP-level behavior (does the server actually check the guard before inference? does reset clear state visible to subsequent requests?) is a separate wiring question. This gate covers the HTTP ↔ guard handshake — specifically that a robot that hits a joint-limit wall 10 times in a row won't keep receiving action chunks after the 11th `/act` call.

**Reproducer:** `pytest tests/test_guard_trip_integration.py -v`.

---

## Gate lifecycle

Each gate has three possible states:

1. **Green** (current) — last-run JSON dump exists in `reflex_context/<gate>_last_run.json` with a PASS verdict + date.
2. **Red** — a recent commit broke it. Block launch; investigate.
3. **Stale** — last run was >2 weeks ago. Re-run before launch to confirm still green.

Stale detection: every gate's last_run JSON has a timestamp. A pre-launch checker should flag any > 2 weeks old.

---

## How the 9 gates came to be

- **Gates 1–5** were identified during launch prep (commit `11b7c5d`, 2026-04-18) as the minimum bar for "claims work on a clean box" + "claims work in a ROS2 environment" + "claims work with TRT/CUDA." Five gates, five different failure modes previously observed in customer-style setups.
- **Gates 6–9** were added later (commit `b14eaf7`) when a post-analysis found four end-to-end chains that had unit-test coverage but no integration-test coverage. Each one was a real failure mode we'd seen at least once: dispatcher routing errors (6), HTTP wiring (7), num_steps propagation (8), and guard/HTTP handshake (9).

The 9 gates are NOT a theoretical correctness framework — each one was born from a real incident or a real gap that would have bitten a customer.

---

## References

- `tests/test_*.py` — each gate's implementation
- `reflex_context/*_last_run.json` — last-run verdict + timestamp per gate
- `.github/workflows/docker-smoke.yml` — CI version of gate 4
- `GOALS.yaml` — original goal IDs (`fresh-install`, `cuda-runtime`, etc.)
- `05_sessions/2026-04-19_all_four_vlas.md` — session that added gates 6–9 + verified all 9
- `02_bugs_fixed/modal_deployment_gotchas.md` — the Modal-side pain behind gates 4 and 5
