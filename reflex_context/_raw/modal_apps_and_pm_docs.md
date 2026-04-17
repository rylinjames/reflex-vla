# Modal apps + PM/documentation surface — raw notes

Generated 2026-04-17 against `modal app list --json` + repo docs.

Scope filter: Modal apps created 2026-04-05+ whose description starts with `reflex-` or `ap-`.
Non-reflex apps (`hikaflow-*` from 2026-02) are excluded. `easyinference-demo` is kept as a boundary case — from 2026-04-05 — though it is not a reflex app.

---

# Part A: Modal apps

All reflex runs sit on 2026-04-17 and are now `stopped`. Descriptions group them into three campaigns:

- `reflex-stage-diff` (x4) — iteration of a stage-by-stage vision/text/decoder/expert parity probe
- `reflex-pytorch-vs-onnx` (x2) — full PyTorch vs our ONNX pipeline action-diff probe
- `reflex-libero10` (x1) — the real LIBERO-10 Modal run (adapter + vla_eval)

## Modal app: ap-MrSsaMvCuiwlYLaTCs8gOb
**Description:** easyinference-demo
**Created:** 2026-04-05 10:30 IST
**State:** deployed
**Tasks:** 0
**Purpose (from logs):** Unreadable — `modal app logs --since 5d` returned no output. Predates the 5-day window or logs GC'd. Not a reflex app based on name.
**Outcome:** unreadable
**Key numbers:** n/a
**Errors / notable events:** none surfaced

## Modal app: ap-uKaH8uEPuCeoKz0C6TCqbV
**Description:** reflex-stage-diff
**Created:** 2026-04-17 10:39 IST
**Stopped:** 2026-04-17 10:46 IST (~6 min run)
**State:** stopped
**Tasks:** 0
**Purpose (from logs):** First stage-diff attempt. Export `lerobot/smolvla_libero`, then run per-stage torch-vs-ONNX parity across vision encoder → text embedder → state projection → decoder prefill → expert velocity.
**Outcome:** FAIL (weight-loading regression on vision_model)
**Key numbers:**
- Expert export: 66.7s, 16 layers 99.8M params, max_diff=2.86e-06 PASS
- VLM export: 103.3s total files ~1.18GB (394.1MB vision / 189.2MB text / 596.6MB decoder)
- **`[vlm-weights] load: 488 missing, 345 unexpected`** (compare to healthy runs with `144 missing, 0 unexpected`) — rebasing stopped at wrong prefix, vision weights never loaded.
- Vision stage 1: cos=+0.6983, torch||=1644 vs onnx||=1104 — wrong magnitude. All downstream stages meaningless.
**Errors / notable events:**
- Weight-prefix rebase bug — exporter did not unwrap `ForConditionalGeneration`, so 488 vision/text keys were missing and 345 SmolVLA-prefixed keys were unexpected.
- Fixed in later runs (starting ap-YrnH...) where log shows `[vlm-weights] unwrapped ForConditionalGeneration -> inner model` and `144 missing, 0 unexpected`.

## Modal app: ap-YrnHF0WgFXQ2Y7HWlYHPaI
**Description:** reflex-stage-diff
**Created:** 2026-04-17 10:57 IST
**Stopped:** 2026-04-17 11:03 IST (~6 min)
**State:** stopped
**Tasks:** 0
**Purpose (from logs):** Re-run of stage-diff with the `ForConditionalGeneration` unwrap fix.
**Outcome:** partial (vision now OK, decoder layer_0_v cos=0.9117 remains a concern)
**Key numbers:**
- Expert export: 45.1s, max_diff=3.34e-06 PASS
- VLM export: 97.5s, 142.8s total
- `[vlm-weights] load: 144 missing, 0 unexpected` (healthy)
- Vision stage: cos=+1.0000, L2=2.2022e-03, max_abs=1.0681e-04 — matches torch
- Text embed: cos=+1.0000
- State proj: cos=+1.0000
- Per-layer decoder prefill k/v (post-unwrap):
  - layer_0_k cos=+1.0000, layer_0_v cos=+0.9117 (outlier)
  - layer_8_k cos=+0.9997, layer_8_v cos=+0.9967
  - layer_15_k cos=+0.9994, layer_15_v cos=+0.9954
- Run stopped before Stage 5 expert velocity.
**Errors / notable events:**
- Note from log: `"Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)"` — known gap.
- `Stopping app - user stopped from CLI` — user killed before stage 5.

## Modal app: ap-oXrqhfnQFJLuuY4A9GbPSv
**Description:** reflex-stage-diff
**Created:** 2026-04-17 11:08 IST
**Stopped:** 2026-04-17 11:12 IST (~4 min)
**State:** stopped
**Tasks:** 0
**Purpose (from logs):** Stage-diff rerun with presumably same weight fix; slightly different vision numerics (max_abs=8.7738e-05 vs 1.0681e-04). Ran through full stage 5 this time.
**Outcome:** PASS on stages 1-3, partial on stage 4 (same layer_0_v=0.9117 outlier), stage 5 produced an expert velocity chunk.
**Key numbers:**
- Expert export: 28.7s, max_diff=3.34e-06 PASS (same 99.8M params / 16 layers)
- VLM export: 60.8s, 90.0s total
- Vision cos=+1.0000 (max_abs=8.7738e-05, tighter than 10:57 run)
- Stage 5 expert velocity: shape=(1,50,32), first 7 dims ≈ [3.636, -0.145, 0.934, 0.380, -1.624, -0.960, 0.177], norm=9.029e+01
**Errors / notable events:**
- Same "v0.3 item" note about fine-tuned VLM layers not preserved.
- Same layer_0_v cos=+0.9117 outlier persists → *this is the reproducible structural discrepancy*.

## Modal app: ap-2tNsuBRSnvuQ9kWPwm55Ob
**Description:** reflex-stage-diff
**Created:** 2026-04-17 11:34 IST
**Stopped:** 2026-04-17 11:36 IST (~2 min — shortest run)
**State:** stopped
**Tasks:** 0
**Purpose (from logs):** Final stage-diff iteration — same pipeline, same fixture. Log appears identical to oXrq run (same 8.7738e-05 vision max_abs, same 3.636 first velocity dim), suggesting a clean re-run against cached artifacts (28.7s expert, 60.8s VLM, 90.0s total).
**Outcome:** partial — same result as oXrq (stage 4 layer_0_v cos=0.9117 outlier, stage 5 completes).
**Key numbers:** identical to ap-oXrq above (cos=+1.0 vision/text/state, layer_0_v cos=+0.9117, expert velocity norm=9.029e+01).
**Errors / notable events:** none new.

## Modal app: ap-v6gmsosx9ayiGJxoWUfs6o
**Description:** reflex-pytorch-vs-onnx
**Created:** 2026-04-17 11:03 IST
**Stopped:** 2026-04-17 11:06 IST (~3 min)
**State:** stopped
**Tasks:** 0
**Purpose (from logs):** End-to-end action-diff — run full PyTorch policy (lerobot/smolvla_libero, with preprocessor/postprocessor) side-by-side with our ONNX pipeline on the same preprocessed batch. 7 steps: export → load policy → raw batch → preprocess → torch predict → onnx pipeline → action diff.
**Outcome:** **FAIL** — "MAJOR DIVERGENCE — a structural bug remains in the export"
**Key numbers:**
- Export: 129s
- Torch first action (post-processor): `[-0.112, 0.113, -0.173, 0.020, 0.023, 0.009, 1.025]`
- Torch first action (pre-postproc): `[-0.520, 0.070, -0.187, 0.488, 0.278, 0.183, 1.076]`
- ONNX first action (normalized, 32-dim output but compared on first 7): `[-0.314, -0.474, 0.200, 0.120, 0.373, 0.171, -0.188]`
- **abs diff: `[0.205, 0.545, 0.387, 0.368, 0.094, 0.011, 1.264]`**
- **L2 = 1.494, cos_sim = +0.082**
- Note: CUDAExecutionProvider not available → ONNX ran on CPU (warning: "Available providers: 'AzureExecutionProvider, CPUExecutionProvider'").
**Errors / notable events:**
- Batch preprocessing confirmed 3 cameras 256×256, state shape (1,8), tokens (1,48).
- Torch output shape (1, 50, 7) vs ONNX (50, 32) — dim mismatch is suspicious; ONNX appears to output full 32-dim action without slicing to 7 (the LIBERO action dim).

## Modal app: ap-oBhVQcQnjsd4uMK6lSy98D
**Description:** reflex-pytorch-vs-onnx
**Created:** 2026-04-17 11:28 IST
**Stopped:** 2026-04-17 11:32 IST (~4 min)
**State:** stopped
**Tasks:** 0
**Purpose (from logs):** Second iteration of the same torch-vs-ONNX action diff.
**Outcome:** **FAIL** — still MAJOR DIVERGENCE, but different ONNX numbers than the 11:03 run (non-deterministic somewhere).
**Key numbers:**
- Export: 151s
- Torch first action (post-processor): `[-0.112, 0.113, -0.173, 0.020, 0.023, 0.009, 1.025]` — same as v6gm (deterministic)
- Torch first action (pre-postproc): same `[-0.520, 0.070, -0.187, 0.488, 0.278, 0.183, 1.076]`
- ONNX first action (normalized): `[-0.541, -0.207, 0.066, 0.093, 0.441, 0.294, -0.723]` — different from v6gm!
- **abs diff: `[0.021, 0.277, 0.253, 0.395, 0.162, 0.112, 1.799]`**
- **L2 = 1.890, cos_sim = -0.209** (worse than v6gm's +0.082 — *sign flipped*)
**Errors / notable events:**
- Torch side deterministic; ONNX side varied → randomness lives in the ONNX path (likely the random VLM conditioning fallback when the VLM orchestrator isn't wired, per README disclaimer "Action values look random / nonsensical … Expected in v0.1").
- Same CUDAExecutionProvider not available warning.

## Modal app: ap-QAG1Pk9w3DkuZnVs9VC8Ke
**Description:** reflex-libero10
**Created:** 2026-04-17 11:14 IST
**Stopped:** 2026-04-17 11:26 IST (~12 min — longest run)
**State:** stopped
**Tasks:** 0
**Purpose (from logs):** The flagship LIBERO-10 eval — run adapter + vla_eval against 100 episodes (10 tasks × 10 episodes) of LIBERO-10 through the new reflex.runtime.adapters.vla_eval adapter on Modal.
**Outcome:** partial — adapter came up PASS, env smoke PASS, but the eval failed the first two tasks then was killed by user from CLI before completing.
**Key numbers:**
- Step 1 Export (`lerobot/smolvla_libero`): PASS in 173s; 3/3 VLM files; 2/2 normalizer files (`policy_preprocessor_step_5_normalizer_processor.safetensors`, `policy_postprocessor_step_0_unnormalizer_processor.safetensors`)
- Step 2 Adapter (`reflex.runtime.adapters.vla_eval`): PASS in 12s on port 8000
  - `Expert ONNX inputs: ['noisy_actions', 'timestep', 'position_ids', 'vlm_k', 'vlm_v']`
  - VLM orchestrator loaded complete=True (vision_encoder.onnx + text_embedder.onnx + decoder_prefill.onnx — all CPU)
  - `Loaded normalizer stats from 2 file(s): ['action_mean', 'action_std', 'state_mean', 'state_std']`
  - Adapter config: `export=/tmp/reflex_libero_export device=cuda out_dim=7 camera=<first> vlm=on norm=on`
- Step 3 config: LIBERO-10 config written to `/tmp/libero_10_config.yaml`
- Step 3b env smoke: PASS (env.reset() 5.8s, obs keys present)
- Step 4 vla-eval run:
  - Task 1: `put both the alphabet soup and the tomato sauce in the basket` ep0 → **FAIL** (steps=150, max)
  - Task 2: `put both the cream cheese box and the butter in the basket` ep0 → **FAIL** (steps=150, max)
  - Tasks 3-10 never ran — `Stopping app - user stopped from CLI` before ep 3 started.
- Extrapolating: if pattern held, task success would have trended near 0% (consistent with the "random VLM conditioning → meaningless actions" caveat documented in README getting_started).
**Errors / notable events:**
- Warnings: `datasets path /opt/LIBERO/libero/libero/../datasets does not exist!` (twice per ep; non-fatal since LIBERO can init tasks without the dataset dir).
- robosuite macros warnings, gym deprecation warning.
- **This is the run referenced by the in-progress task #23 "Ship LIBERO-10 Modal run and capture task-success number"** — the number is still TBD because the run was aborted. The failure of ep0 on both first two tasks is consistent with the known-issue block in README (action values look random in v0.1 until VLM prefix conditioning is fully wired).

---

# Part B: Reflex product surface (from PM docs)

## Wedges and their status

From `README.md` section "The 7 wedges (+ 1 planned)":

```
reflex export   # checkpoint → ONNX + TensorRT
reflex serve    # HTTP inference server, composable wedges
reflex guard    # URDF-derived safety limits + EU AI Act logging
reflex turbo    # adaptive denoising (stops early on convergence)
reflex split    # cloud-edge orchestration with fallback modes
reflex adapt    # cross-embodiment action-space mapping
reflex check    # 5 pre-deployment checks (loadable, size, structure, dtype, nan_inf)
reflex distill  # [planned] flow-matching step distillation (10 → 2)
```

Per-wedge status (merged from README, CHANGELOG, task list, and Modal runs):

- **export** — shipped (v0.1). Covers 4 flow-matching VLAs (SmolVLA, pi0, pi0.5, GR00T N1.6) + OpenVLA (falls back to `optimum-cli export onnx` + `reflex.postprocess.openvla.decode_actions`). Task #13 marked complete: "Step 1: Unify reflex export (call export_vlm_prefix automatically for SmolVLA)". Modal logs confirm it now auto-exports VLM prefix (vision_encoder.onnx / text_embedder.onnx / decoder_prefill.onnx) + normalizer stats for SmolVLA LIBERO ckpt.
- **serve** — shipped. FastAPI. Auto-prefers TRT FP16 when ORT has TRT EP. `POST /act`, `GET /health`, `GET /config`. Composable wedges as flags.
- **guard** — shipped. URDF → SafetyLimits JSON, clamps at request time, EU AI Act Art. 12 audit log.
- **turbo** — deprecation pending. Task #14 marked complete: "Step 2: Remove split, adapt, turbo commands with deprecation warnings". GOALS.yaml `adaptive-denoise-fix` (weight 5) says "Adaptive denoising works on pi0 (supported), is gated behind --experimental for smolvla/pi0.5/gr00t (unsafe)" — wedge exists but for safety reasons it now prints an experimental warning off-pi0.
- **split** — deprecation pending (same task #14 bundle). Was cloud-edge orchestration.
- **adapt** — deprecation pending (same task #14 bundle). Was cross-embodiment action-space mapping.
- **check** — merged into `validate --quick`. Task #15 complete: "Step 3: Merge check into validate --quick".
- **distill** — *scaffolded, DMPO recipe*. Task #17 complete: "Step 5: Scaffold reflex distill command (DMPO recipe)". GOALS.yaml `distill-dmpo` (weight 9) wants `src/reflex/distill/dmpo.py` importable as `DMPOTrainer` targeting 1000+ Hz.
- **validate** — new shipped item (Unreleased in CHANGELOG). Runs "real ONNX/TRT-vs-PyTorch round-trip parity check" with seeded fixtures for SmolVLA/pi0/GR00T (pi0.5/OpenVLA deferred to v2). JSON + Rich table + `--init-ci` workflow scaffolder. **BREAKING**: `--threshold` default changed from 0.02 (v0.1 placeholder) to 1e-4.
- **bench** — shipped w/ new `--benchmark` flag. Task #16 complete: "Step 4: Add --benchmark flag to reflex bench (wraps LIBERO/SimplerEnv)".

New-ish runtime surface (not in the 7-wedge list but present):
- **`reflex.runtime.adapters.vla_eval`** — built (task #18 complete). Adapts reflex → vla-eval via WebSocket/HTTP; Modal run confirms adapter ready in 12s, loads expert + VLM orchestrator + normalizer stats.
- **`reflex doctor`** — shipped. README: "`reflex doctor` first — diagnoses install + GPU issues in one screen."
- **`reflex models`** — shipped, lists current model support.
- **`reflex targets`** — shipped, lists hardware profiles.

In-progress (tasks still open):
- #23: Ship LIBERO-10 Modal run and capture task-success number — run attempted (ap-QAG1), aborted at task 2/10.
- #24: Add normalizer support to adapter (partly shipped per Modal log: "norm=on" + "Loaded normalizer stats from 2 file(s)").
- #25: Per-layer vlm_kv ONNX export (stage-diff runs are the experiment driving this — layer_0_v cos=0.9117 outlier is what's being chased).
- #26: Extract learnings from all sources into reflex_context/ — *this task*.

## GOALS.yaml live fitness

Mission: **"Deploy any VLA model to any edge hardware. One command."** (version 1)

Total goals: **15** (1 critical@10, 2 critical@10, 3 high@8-9, 3 med-high@7, 3 med@5-6, 3 low@3-4, 1 regression gate@10).

Weight distribution:
- Weight 10 (critical / regression gate): 3 goals (`vlm-prefix-encoder`, `text-embedder-onnx`, `sim-smoke-test`)
- Weight 9: 1 (`distill-dmpo`)
- Weight 8: 2 (`stripe-license-gating`, `ros2-bridge`)
- Weight 7: 3 (`nan-guard-hardening`, `xvla-exporter`, `api-key-auth`)
- Weight 6: 1 (`latency-histograms`)
- Weight 5: 2 (`adaptive-denoise-fix`, `determinism-version-hash`)
- Weight 4: 2 (`inference-test-coverage`, `openvla-exporter`)
- Weight 3: 1 (`sqlite-audit-log`)

Per goal (id — description → check command):

- **vlm-prefix-encoder** (10): "VLM backbone exports as real SigLIP+SmolLM2 ONNX (not stub) with vlm_kv_dim=960 so /act returns task-conditioned actions"
  - check: `test -f src/reflex/exporters/vlm_prefix_exporter.py && grep -q 'AutoModel\|from_pretrained' src/reflex/exporters/vlm_prefix_exporter.py 2>/dev/null && ! grep -q 'AdaptiveAvgPool2d' src/reflex/exporters/vlm_prefix_exporter.py 2>/dev/null`
  - Status (from Modal logs): partially achieved — vision_encoder.onnx (394MB), text_embedder.onnx (189MB), decoder_prefill.onnx (596MB) all exporting with `[vlm-weights] load: 144 missing, 0 unexpected` in healthy runs. But README still warns "Action values look random / nonsensical … Expected in v0.1."
- **text-embedder-onnx** (10): "reflex export produces text_embedder.onnx (real SmolLM2 embed_tokens) so text encoding uses real embeddings not seeded-random fallback"
  - check: `grep -q 'text_embedder' src/reflex/exporters/vlm_prefix_exporter.py 2>/dev/null && grep -q 'embed_tokens' src/reflex/exporters/vlm_prefix_exporter.py 2>/dev/null`
  - Status: shipped (Modal confirms text_embedder.onnx loaded on every run, text stage cos=+1.0000).
- **distill-dmpo** (9): "reflex distill implements DMPO one-step generation (no teacher) targeting 1000+ Hz on consumer GPUs"
  - check: `test -f src/reflex/distill/dmpo.py && .venv/bin/python -c 'from reflex.distill.dmpo import DMPOTrainer' 2>/dev/null`
  - Status: scaffolded (task #17 complete).
- **stripe-license-gating** (8): "Stripe subscription verification gates Pro-tier features (distill, fleet batching) behind a valid license key"
  - check: `test -f src/reflex/licensing.py && .venv/bin/python -c 'from reflex.licensing import verify_license' 2>/dev/null`
  - Status: not yet built.
- **ros2-bridge** (8): "reflex serve --ros2 wraps the HTTP endpoint with a thin rclpy action server for native ROS2 integration"
  - check: `grep -q 'ros2' src/reflex/runtime/server.py 2>/dev/null || test -f src/reflex/runtime/ros2_bridge.py`
  - Status: not yet built.
- **nan-guard-hardening** (7): "Guard rejects NaN/Inf actions and halts after N consecutive clamps (staleness kill-switch)"
  - check: `grep -q 'nan' src/reflex/safety/guard.py 2>/dev/null && grep -q 'staleness\|stale' src/reflex/safety/guard.py 2>/dev/null`
- **xvla-exporter** (7): "reflex export auto-detects and exports xVLA (880M, tokenized action head) to ONNX"
  - check: `test -f src/reflex/exporters/xvla_exporter.py && .venv/bin/python -c 'from reflex.exporters.xvla_exporter import export_xvla' 2>/dev/null`
- **api-key-auth** (7): "reflex serve --api-key enables X-Reflex-Key header auth that rejects unauthenticated requests with 401"
  - check: `grep -q 'api.key\|api_key\|X-Reflex-Key' src/reflex/runtime/server.py 2>/dev/null`
- **latency-histograms** (6): "/act response includes latency_p50, latency_p95, latency_p99, jitter_ms fields"
  - check: `grep -q 'latency_p95\|p95' src/reflex/runtime/server.py 2>/dev/null`
- **adaptive-denoise-fix** (5): "Adaptive denoising works on pi0 (supported), is gated behind --experimental for smolvla/pi0.5/gr00t (unsafe)"
  - check: `grep -q 'experimental\|EXPERIMENTAL' src/reflex/kernels/turbo.py 2>/dev/null`
- **determinism-version-hash** (5): "Every /act response includes model_hash, config_hash, reflex_version for reproducible debugging"
  - check: `grep -q 'model_hash' src/reflex/runtime/server.py 2>/dev/null`
- **inference-test-coverage** (4): "Unit tests for inference.py and individual exporter modules (currently zero direct coverage)"
  - check: `test -f tests/test_inference.py && .venv/bin/python -m pytest tests/test_inference.py -q --tb=no 2>/dev/null`
- **openvla-exporter** (4): "reflex export auto-detects and exports OpenVLA (7.5B, tokenized head) beyond the current stub"
  - check: `grep -c 'def export_openvla' src/reflex/exporters/openvla_exporter.py 2>/dev/null | grep -q '[1-9]'`
- **sqlite-audit-log** (3): "reflex serve --audit-db appends every /act call to a SHA-256 hash-chain SQLite log"
  - check: `test -f src/reflex/safety/audit_log.py && .venv/bin/python -c 'from reflex.safety.audit_log import AuditLog' 2>/dev/null`
- **sim-smoke-test** (10, regression gate): "Trajectory replay against Open X-Embodiment data — predicted actions within L2 threshold of expert"
  - check: `test -f scripts/sim_smoke_test.py && .venv/bin/python scripts/sim_smoke_test.py --quick 2>/dev/null`

*Currently passing count:* Not resolved without actually running the checks — each is a shell command. But from the Modal logs + task list we know at least `text-embedder-onnx`, `distill-dmpo` scaffold, and `vlm-prefix-encoder` (partial) are further along than goal definition; the pass/fail tally requires running `.venv/bin/python` against these checks.

## Launch + positioning

**Three drafts live in `/launch/` awaiting approval** (per `launch/README.md`: "Nothing here is published yet — all need user approval before going live.").

Sequencing: LeRobot issue #3146 first → 48-72h later, Show HN → same/next day, r/robotics. Rationale: "reduces signal in each, and means you can't respond to comments in any of them."

Pre-launch checklist includes Jetson benchmark, <24h GitHub Issues response, Discord/Slack link, fresh-box install test.

**Tagline (README line 3):**
> **The deployment layer for VLAs** — take a Vision-Language-Action model off the training cluster and onto a robot.

**Elevator (README line 5):**
> Cross-framework ONNX export, edge-first serving, composable runtime wedges (safety, adaptive denoising, cloud-edge split, pre-flight validation). One CLI, seven verbs.

**Show HN title (draft):**
> **Show HN: Reflex – ONNX/TensorRT export for VLA models, runs on Jetson**

**Show HN hook (verbatim):**
> "I built Reflex because the path from 'we have a trained Vision-Language-Action model' to 'it runs on a real robot' is brutal. Every VLA team writes their own export pipeline and most of them break."

**Show HN honesty passage (verbatim):**
> "I went into this thinking the moat was edge-only because torch.compile was crushing my early benchmarks. Turned out my onnxruntime-gpu was silently falling back to CPU due to a CUDA 12-vs-13 library mismatch. Once that was fixed, TRT FP16 wins by 2.6-3.3× across the board."

**r/robotics title:**
> **Open-source tool to deploy SmolVLA / pi0 / pi0.5 / GR00T to Jetson via ONNX + TensorRT**

**r/robotics three asks (verbatim):**
> 1. **Testers** — install it, point it at your robot, tell me what breaks
> 2. **Jetson benchmark contributor** — 30 min of your time on an Orin Nano dev kit would let me publish real edge numbers
> 3. **Critical feedback** on the wedge composition (`--safety-config`, `--adaptive-steps`, etc.) — does it match how you actually want to deploy?

**LeRobot #3146 close (verbatim):**
> "Honest disclaimer: this is alpha, single maintainer, no funding. If it works for you, great; if it doesn't, please tell me how it broke so I can fix it."

**Target user (across all three drafts):** "VLA teams deploying to edge hardware (Jetson Orin / Orin Nano / Thor)" + "fleet operators serving N robots through one GPU." The explicit ask in Show HN is "testers, especially anyone with a Jetson Orin or a real robot to point this at."

**Differentiation claims (consolidated):**
- Cross-framework ONNX + TRT — "4 VLA families covered" (SmolVLA, pi0, pi0.5, GR00T N1.6) via one CLI.
- Same pipeline cloud & edge: "The same ONNX → TRT pipeline is what runs on Jetson — there is no 'cloud version' vs 'edge version' of the model."
- TRT FP16 beats torch.compile 2.6-3.3× on cloud GPU (Modal A10G):
  - SmolVLA 99.8M: 0.95ms vs torch.compile 3.06ms (3.2×)
  - pi0 314.6M: 1.94ms vs 6.23ms (3.2×)
  - pi0.5 426.9M: 2.24ms vs 7.34ms (3.3×)
  - GR00T 1091.7M: 5.59ms vs 14.61ms (2.6×)
- End-to-end per-chunk (10-step denoise) — 86 Hz SmolVLA / 42 Hz pi0 / 37 Hz pi0.5 / 18 Hz GR00T on A10G.
- Fleet batching on pi0: 17.1 qps @ batch 1 → 49.3 qps @ batch 16 (2.88× throughput, per-request latency *drops*).
- All 4 VLAs fit on "$500 Orin Nano 8GB in FP16 with 2× overhead (verified empirically)" — biggest is GR00T at 4.4 GB.

**Not-a:** (README section "What Reflex is and isn't")
> **Is:** the deployment layer between a trained VLA and a real robot. Cross-framework export (4 VLA families covered), composable runtime (serve + safety + turbo + split), Jetson-first.
> **Isn't:** a training framework (PyTorch/JAX own that) or a cloud inference provider (vLLM/Baseten own that). Reflex's moat is the deployment toolchain: cross-framework ONNX, TensorRT FP16 engines that beat `torch.compile` on cloud GPU by 2.6-3.3× *and* run on Jetson, deterministic deploy graph, and the wedge composition for production robot deployments.

## CHANGELOG highlights per version

From `CHANGELOG.md` (as of 2026-04-16):

- **Unreleased** — "real" reflex validate ships:
  - New: `reflex validate` runs real ONNX/TRT-vs-PyTorch round-trip parity check.
  - New: Seeded fixtures for SmolVLA / pi0 / GR00T (pi0.5, OpenVLA defer to v2).
  - New: JSON + Rich-table output; `--init-ci` emits `.github/workflows/reflex-validate.yml`.
  - New: Exit codes 0 pass / 1 fail / 2 error.
  - New public exports on `reflex`: `ValidateRoundTrip`, `load_fixtures`, `SUPPORTED_MODEL_TYPES`.
  - **BREAKING (from stub):** `reflex validate` default `--threshold` changed from `0.02` (the v0.1 placeholder) to `1e-4`. "The stub never performed real validation so no existing deployments depended on the old default. Pass `--threshold 0.02` explicitly to match the previous behavior."
  - `reflex validate` now requires a valid `reflex_config.json` inside the export directory — the stub accepted any path.
  - Fix: `_pytorch_backend` SmolVLA path no longer swallows `AutoConfig` fetch errors silently (now logs warning + continues with fallback head_dim).
  - Fix: CLI handler catches `KeyboardInterrupt` explicitly (exits 130) instead of a raw traceback.
- **v0.1.0** — initial release, seven-wedge scope (see README). No detailed entries beyond "see README for the seven-wedge scope at that time."

No v0.2 or v0.3 milestones ship'd yet — CHANGELOG goes straight from Unreleased to v0.1.0. README status footer: "v0.1 — active development. Install, kick the tires, open issues loudly. We're looking for the first 20 robotics teams actually deploying this; your feedback shapes v0.2."

## Install / dependency gotchas

From `pyproject.toml` `[project.optional-dependencies]` and README quickstart:

**Extras:**
- `onnx` — `onnxruntime>=1.17.0` (CPU-only).
- `gpu` — `onnxruntime-gpu>=1.20,<1.24` + `nvidia-cudnn-cu12>=9.0,<10.0` + `nvidia-cublas-cu12>=12.0,<13.0`. Comment says (verbatim): *"Apr-14 post-mortem: omitting these was the cause of silent CPU fallback in v0.1 benchmarks."* (This is the "CUDA 12-vs-13 library mismatch" referenced in the Show HN draft.)
- `serve` — `fastapi>=0.100.0`, `uvicorn>=0.23.0`, `Pillow>=10.0.0`. Note: `[serve]` alone still installs `onnxruntime` (CPU) because `onnx` is in core deps; for GPU use `[serve,gpu]`.
- `safety` — `yourdfpy`.
- `eval` — `vla-eval` + `mujoco>=3.0` + `robosuite==1.4.1` + `gymnasium` + `h5py`. (heavy sim deps; gated behind extra so the base install stays light; use with `reflex bench --benchmark libero_10`.)
- `dev` — pytest 8+, ruff 0.4+, mypy 1.10+, httpx 0.24+.
- Commented out: `tensorrt = ["tensorrt>=10.0"]` — "install on Linux/Jetson only: pip install tensorrt".

**Core deps** always pulled (with inline comment block): "hub access for `reflex export <hf_id>` and tokenizer/config loading for the auto-detect path. Apr-14 install-path verification caught that omitting these breaks `reflex export` even with [serve,gpu]." → `huggingface_hub>=0.20.0`, `transformers>=4.40,<5.0`, `onnx>=1.15.0`, `onnxscript>=0.1.0`, plus `torch`, `safetensors`, `typer`, `rich`, `pydantic`, `numpy`, `pyyaml`.

**Install gotchas from README + getting_started.md:**
1. **cuDNN 9 system library** — "GPU install requires the FULL cuDNN 9 system library (incl. libcudnn_adv.so.9), not just the pip wheel. Easiest path is NVIDIA's container: `docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3`". `reflex serve` errors loudly if cuDNN can't load — **no silent CPU fallback** (deliberate, from the Apr-14 post-mortem).
2. **`CUDAExecutionProvider not available`** — ORT 1.20+ needs CUDA 12.x + cuDNN 9.x. Pip-installed `nvidia-cudnn-cu12` is missing `libcudnn_adv.so.9`. Fix: use nvcr.io container, or pass `--device cpu`, or `--no-strict-providers` (not recommended).
3. **`trtexec not found` warning** — normal on dev box without TRT; ONNX still exports; TRT engine built later on target with TRT installed (Jetson Jetpack or x86 `nvidia-tensorrt`).
4. **"Model not loaded" 500** — lifespan handler hasn't finished; big models (pi0, gr00t) take 30-60s for ORT-GPU session creation. Wait for `GET /health` `model_loaded: true`.
5. **"Action values look random / nonsensical"** — "Expected in v0.1. The current ONNX export covers the action-expert denoising loop with random VLM conditioning. Real per-image conditioning lands when the VLM prefix encoder is wired (Phase II.4 / v0.2)."
6. **First `reflex serve` takes 30-90s** (TRT engine build); restart is 1-2s because cached in `<export_dir>/.trt_cache`.
7. **Modal install path vs release:** pyproject declares `torch>=2.1.0` but Modal's base image ships `torch==2.10.0` (v6gm run) or `torch==2.11.0` (oBhV run) with different CUDA deps — the latter pulls `cuda-toolkit==13.0.2` + `nvidia-cudnn-cu13==9.19`, mismatching the cu12 pins in `[gpu]`. Dev on Modal + GPU serve in prod use different CUDA major versions.

## Cross-document status tell

The clearest single signal of project state is from README line 5 (positioning) + "What Reflex is and isn't" + the three launch drafts all sharing the same numbers: Reflex has converged on a well-articulated "deployment-layer-for-VLAs" story backed by 2.6-3.3× TRT-over-torch.compile benchmarks — but the **VLM prefix encoder is the one honest gap** holding up v0.2 launch, and it's exactly what the Modal stage-diff and pytorch-vs-onnx runs on 2026-04-17 are probing (and where they're still failing — L2=1.494-1.890, cos_sim=+0.082 to -0.209). The LIBERO-10 run (ap-QAG1) was the first attempt to get a task-success number through this same incomplete pipeline and it confirmed the disclaimer: first two tasks failed at step 150.
