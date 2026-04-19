# MVP Queue

Operational plan for shipping Reflex VLA v0.2 — the first sellable version. Derived from GOALS.yaml; stays in sync with it by referencing goal IDs, not duplicating descriptions.

**MVP thesis (EXPANDED 2026-04-19):** "Every major open VLA verified at cos=+1.000000 machine precision — SmolVLA, pi0, pi0.5, GR00T N1.6 — with ROS2 + Docker + safety + one-command install." The cross-framework moat claim is now load-bearing across the full open-VLA landscape (not just SmolVLA + pi0).

**Status: v0.2 ship-ready. All 9 launch-verification gates green. Awaiting launch trigger (user decision).**

---

## Current focus

**Up next (pick one):**
- **Tag + launch v0.2.1 / v0.3.0** — all the technical work is done. Launch sequencing: LeRobot #3146 comment → Show HN (48–72h later) → r/robotics.
- **GR00T VLM conditioning (Eagle backbone export)** — DiT + AdaLN stack is machine precision, but VLM conditioning is zero-stubbed. Full multi-modal control is v0.3.
- **Jetson latency** — blocks on hardware access (CloudJetson Orin Nano waitlisted). Community-bounty-friendly.

**Last completed:** `gr00t-onnx-parity` (2026-04-19, commit `4090e76`) — single-step cos=+1.000000 max_abs=8.34e-07; 4-step loop cos=+1.000000 max_abs=4.77e-07. All four major open VLAs now at machine precision.

---

## MVP ship list (12 goals)

Priority order reflects dependency + moat impact, not raw weight.

| # | goal_id (GOALS.yaml) | weight | status | ETA | notes |
|---|---|---|---|---|---|
| 1 | `native-path-parity` | 10 | ✅ done | 0 | commit 4c0f817 — SmolVLA PyTorch path cos=1.0000 |
| 2 | `vlm-prefix-encoder` | 10 | ✅ done | 0 | check passes (real SmolVLM2 + SmolLM2 ONNX) |
| 3 | `text-embedder-onnx` | 10 | ✅ done | 0 | check passes |
| 4 | `quickstart-docs` | 6 | ✅ done | 0 | check passes; may still need polish pre-launch |
| 5 | `smolvla-onnx-parity` | 9 | ✅ done | 0 | 2026-04-18: cos=+1.0000000, max_abs=1.55e-06 (first) / 3.34e-06 (full). `scripts/modal_smolvla_monolithic_export.py` via Modal A10G. Required transformers==5.3.0 pin + torch.where monkey-patch + post-export Where dtype Cast fix. |
| 6 | `pi0-onnx-parity` | 9 | ✅ done | 0 | 2026-04-18: single-step ONNX end-to-end cos=+1.0000000. Host-loop in Pi0OnnxServer for num_steps=10 (Path C decision — keeps adaptive denoising + safety + distill wedges viable). See [03_research/pi0_onnx_importable_sources.md](03_research/pi0_onnx_importable_sources.md), [01_architecture/pi0_monolithic_wrap_pattern.md](01_architecture/pi0_monolithic_wrap_pattern.md). |
| 7 | `jetson-benchmark-ci` | 9 | ⏭ deferred to v0.3 | 0 | 2026-04-18: explicitly deferred. CloudJetson only has AGX Orin 64GB available (3-day min, $122); Orin Nano waitlisted. Launch pitch reframes around verified A10G parity + Docker/ROS2 deployment. Real Orin Nano numbers land once a customer / bounty supplies hardware. |
| 8 | `multi-model-native-parity` | 9 | ✅ done | 0 | 2026-04-18: pi0 verified at cos=1.0000000000, max_abs=0.000e+00 (bit-exact) via `parity_native_pi0` function in `scripts/modal_pi0_monolithic_export.py`. pi0.5 + GR00T (AdaRMSNorm / AdaLN) deferred to v0.3. For MVP, pi0 + SmolVLA native parity is enough to claim the cross-framework moat. |
| 9 | `ros2-bridge` | 8 | ✅ done | 0 | 2026-04-18: `src/reflex/runtime/ros2_bridge.py` + `reflex ros2-serve` CLI subcommand. Subs `sensor_msgs/Image` + `sensor_msgs/JointState` + `std_msgs/String`; pubs `std_msgs/Float32MultiArray` at configurable Hz. rclpy import gated behind helpful error. 8 tests with mocked rclpy (`tests/test_ros2_bridge.py`, all green). |
| 10 | `nan-guard-hardening` | 7 | ✅ done | 0 | 2026-04-18: `ActionGuard.check()` now zeros the entire chunk on any NaN/Inf + consecutive-clamp kill-switch (default 10). Server checks `guard.tripped` before `/act` and refuses with a `guard_tripped` error. New endpoints: `GET /guard/status`, `POST /guard/reset`. 9 new tests in `tests/test_guard.py` (19 total, all green). |
| 11 | `docker-image-distribution` | 7 | ✅ done | 0 | 2026-04-18: `Dockerfile` + `.dockerignore` + `.github/workflows/docker-publish.yml` landed. Auto-publishes `ghcr.io/rylinjames/reflex-vla:<version>` + `:latest` on any `v*` tag push. Image: python:3.12-slim + reflex-vla[serve,gpu], ENTRYPOINT=`reflex`, CMD=`serve /exports --host 0.0.0.0 --port 8000`. x86_64 only; Jetson arm64 deferred to v0.3. README quickstart section added. |
| 12 | `export-verification-report` | 6 | ✅ done | 0 | 2026-04-18: `src/reflex/verification_report.py` writes `VERIFICATION.md` with sha256 manifest + opset + model metadata. Skeleton fires from `reflex export`; full parity rows written by `reflex validate`. Regression test at `tests/test_verification_report.py` (6 tests). |

**MVP SHIP READY.** All 11 software items + pi0 num_steps=10 fix + pi0.5 + GR00T landed by 2026-04-19. Nine launch-verification gates all green. Last commit: `c46386a` (knowledge capture). Awaiting user "go" on tag + launch.

### 2026-04-19 session additions (beyond the original 12 MVP goals)

| goal_id | status | commit | verified number |
|---|---|---|---|
| `pi0-num-steps-10-fix` (the 3-patch stack) | ✅ done | `bac658a` | cos=+1.000000, max_abs=2.09e-07 |
| `pi05-onnx-parity` (promoted from v0.3) | ✅ done | `c604962` | cos=+1.000000, max_abs=2.38e-07 |
| `gr00t-onnx-parity` (promoted from v0.3) | ✅ done | `4090e76` | cos=+1.000000, max_abs=8.34e-07 (single-step); 4.77e-07 (4-step loop) |
| 9 launch-verification gates | ✅ all green | various | see `06_experiments/launch_verification_gates.md` |

The v0.3 "deferred" list below has shrunk — pi0.5 and GR00T parity moved up to MVP. Remaining v0.3 items are genuine follow-ups, not reprioritizations.

---

## Deferred to v0.3

Everything below is tracked in GOALS.yaml but explicitly NOT in v0.2 ship. Revisit after launch + first paying customer signal.

| goal_id | weight | why deferred |
|---|---|---|
| `distill-dmpo` | 9 | Wow-factor (1000+ Hz one-step) but not a deal-blocker. Ship after launch, reposition around it for a v0.3 release. |
| `jetson-benchmark-ci` | 9 | CloudJetson Orin Nano waitlisted; no hardware access. Community bounty or customer-supplied hardware will unblock. |
| `gr00t-vlm-conditioning` | 8 | NEW 2026-04-19: GR00T's DiT stack is cos=1.0 machine precision, but VLM conditioning is zero-stubbed. Full multi-modal control needs Eagle-2-HG VLM backbone ONNX export + chained KV input to the DiT. |
| `calibration-metrics` | 8 | Deepest Leg C differentiator but zero paying customers asking. Build after first customer acquires the MVP and asks. |
| `prefix-kv-cache-reuse` | 8 | Dexmal pattern, 5–10× throughput. Hold for v0.3; v0.2 sells on baseline latency. |
| `xvla-exporter` | 7 | xVLA support expands model zoo. Now that pi0 + pi0.5 + GR00T are all solid, this is next in line. |
| `api-key-auth` | 7 | Tablestakes prod, but first customers won't need it (single-tenant deployments). v0.3. |
| `action-chunk-buffering` | 7 | Physical Intelligence pattern. Ship when first customer hits the 20Hz/100Hz timing problem. |
| `orin-nano-fp16-fit` | 7 | NEW: pi0/pi0.5 monolithic ONNX is 12.5–13GB FP32; doesn't fit on Orin Nano 8GB. FP16 engine rebuild + validation is v0.3. SmolVLA (1.6GB) and GR00T (4.4GB) already fit. |
| `latency-histograms` | 6 | Observability polish. v0.3. |
| `stripe-license-gating` | 6 | Invoice the first 5 customers manually. Automate in v0.3. |
| `adaptive-denoise-fix` | 5 | pi0-only feature, edge case. v0.3. |
| `determinism-version-hash` | 5 | Reproducibility polish. v0.3. |
| `inference-test-coverage` | 4 | Quality improvement. v0.3+. |
| `openvla-exporter` | 4 | Different (tokenized) path. v0.3. |
| `openvla-onnx-parity` | 4 | Depends on openvla-exporter. v0.3. |
| `sqlite-audit-log` | 3 | EU AI Act compliance for EU customers. Ship when first EU customer appears. |
| `docker-arm64-jetson-image` | 3 | x86_64 image shipped in v0.2; arm64 Jetson image needs native build. v0.3. |

---

## Pricing + customer outreach plan

**Pricing (MVP launch):**
- **Founding-customer offer:** $500/mo for first 3 months, then $2,000/mo
- **Invoice billing** (Stripe automation deferred to v0.3)
- **Target: 3 founding customers within 2 weeks of launch**

**Ideal first-customer persona:**
- 2–10 person robotics team
- Fine-tuned SmolVLA or pi0 on their own dataset
- Needs Jetson Orin Nano / Orin deployment at ≥20 Hz
- Uses ROS2 (industry standard)
- Currently hand-rolling their own export/serve pipeline
- NOT Physical Intelligence (they have their own stack)
- NOT HuggingFace internal (they use lerobot directly)

**Outreach channels (post-launch, sequenced):**
1. **LeRobot PR #3146 comment** — most strategic; active VLA users read it
2. **48–72h later: Show HN** — broader tech audience, signal amplification
3. **Same day or next day: r/robotics** — third audience
4. Direct outreach to 3 named companies (identify during weeks 5–6)

**Launch prerequisites** (from `launch/README.md` + 2026-04-19 updates):
- [x] All MVP goals landed (2026-04-18, +3 promoted ones on 2026-04-19)
- [ ] Jetson benchmark number — **explicitly deferred to v0.3** (disclosed in every draft)
- [x] SmolVLA + pi0 + pi0.5 + GR00T all at verified cos=+1.000000 (exceeded bar)
- [x] Docker image published to GHCR (`ghcr.io/rylinjames/reflex-vla:0.2.0`)
- [ ] 5-minute quickstart tested on fresh Mac + fresh Linux box (optional polish — CI + Modal fresh-install gate already cover this)
- [ ] GitHub Issues open + <24h response commitment set in profile

**Signal to invest in v0.3:**
- 2 of 3 founding-customer outreach says yes → PMF, build v0.3
- 0 of 3 says yes → pivot positioning, not features

---

## Update protocol

This doc tracks three categories of change:
1. **Status flip** — when a goal completes, move from `❌ blocked` to `✅ done`, add commit SHA, bump `Last updated`.
2. **Queue reorder** — when a dependency reveals itself (e.g., smolvla-onnx-parity gap), re-order and note why.
3. **Defer / promote** — when a goal moves between MVP and v0.3 deferred list, note the rationale.

**Any change here should also update:**
- `GOALS.yaml` if weight or description changes
- `reflex_context/measured_numbers.md` if verification status changes
- Commit message referencing this file

---

**Last updated:** 2026-04-19 (pi0.5 + GR00T + 9 launch gates landed; v0.2 ship-ready; awaiting tag + launch trigger)
