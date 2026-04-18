# MVP Queue

Operational plan for shipping Reflex VLA v0.2 — the first sellable version. Derived from GOALS.yaml; stays in sync with it by referencing goal IDs, not duplicating descriptions.

**MVP thesis:** "SmolVLA + pi0 on your Jetson, verified cos≥0.999 parity, with ROS2 + safety + one-command install." The cross-framework moat claim (both SmolVLA *and* pi0 verified) is load-bearing — without it, reflex is a one-model tool.

**Target: 4–7 weeks of solo dev to sellable v0.2 MVP.**

---

## Current focus

**Up next (pick one):**
- `jetson-benchmark-ci` (weight 9) — get the second real number (Jetson ms/step) via Amazon-return Orin Nano + community bounty
- `pi0-onnx-parity` (weight 9) — unlock the cross-framework claim (1–2 weeks of verification + bug hunting on existing 584-line pi0_exporter.py)

**Last completed:** `native-path-parity` (2026-04-17, commit `4c0f817`) — cos=+1.000, L2=0.000 end-to-end on SmolVLA.

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
| 7 | `jetson-benchmark-ci` | 9 | ❌ blocked | 2–3d | Amazon-return Orin Nano $249 + community bounty ($20–50) in parallel |
| 8 | `multi-model-native-parity` | 9 | ✅ done | 0 | 2026-04-18: pi0 verified at cos=1.0000000000, max_abs=0.000e+00 (bit-exact) via `parity_native_pi0` function in `scripts/modal_pi0_monolithic_export.py`. pi0.5 + GR00T (AdaRMSNorm / AdaLN) deferred to v0.3. For MVP, pi0 + SmolVLA native parity is enough to claim the cross-framework moat. |
| 9 | `ros2-bridge` | 8 | ❌ blocked | ~1w | `reflex serve --ros2` wrapping /act with rclpy action server |
| 10 | `nan-guard-hardening` | 7 | ✅ done | 0 | 2026-04-18: `ActionGuard.check()` now zeros the entire chunk on any NaN/Inf + consecutive-clamp kill-switch (default 10). Server checks `guard.tripped` before `/act` and refuses with a `guard_tripped` error. New endpoints: `GET /guard/status`, `POST /guard/reset`. 9 new tests in `tests/test_guard.py` (19 total, all green). |
| 11 | `docker-image-distribution` | 7 | ✅ done | 0 | 2026-04-18: `Dockerfile` + `.dockerignore` + `.github/workflows/docker-publish.yml` landed. Auto-publishes `ghcr.io/rylinjames/reflex-vla:<version>` + `:latest` on any `v*` tag push. Image: python:3.12-slim + reflex-vla[serve,gpu], ENTRYPOINT=`reflex`, CMD=`serve /exports --host 0.0.0.0 --port 8000`. x86_64 only; Jetson arm64 deferred to v0.3. README quickstart section added. |
| 12 | `export-verification-report` | 6 | ✅ done | 0 | 2026-04-18: `src/reflex/verification_report.py` writes `VERIFICATION.md` with sha256 manifest + opset + model metadata. Skeleton fires from `reflex export`; full parity rows written by `reflex validate`. Regression test at `tests/test_verification_report.py` (6 tests). |

**Total remaining: 3 goals, ~1–1.5 weeks solo** (smolvla-onnx-parity + multi-model-native-parity + docker-image-distribution + export-verification-report + nan-guard-hardening all landed 2026-04-18).

---

## Deferred to v0.3

Everything below is tracked in GOALS.yaml but explicitly NOT in the MVP ship. Revisit after MVP launch + first paying customer signal.

| goal_id | weight | why deferred |
|---|---|---|
| `distill-dmpo` | 9 | Wow-factor (1000+ Hz one-step) but not a deal-blocker. Ship after MVP, reposition around it for a v0.3 launch. |
| `calibration-metrics` | 8 | Deepest Leg C differentiator but zero paying customers asking. Build after first customer acquires the MVP and asks. |
| `prefix-kv-cache-reuse` | 8 | Dexmal pattern, 5–10× throughput. Hold for v0.3; MVP sells on baseline latency. |
| `pi05-onnx-parity` | 7 | AdaRMSNorm handling. 1–2 weeks after pi0 lands. Ships in v0.3. |
| `gr00t-onnx-parity` | 7 | DiT + AdaLN handling. Hardest of the four. v0.3. |
| `xvla-exporter` | 7 | xVLA support expands model zoo. Ship after pi0/pi0.5/GR00T are solid. |
| `api-key-auth` | 7 | Tablestakes prod, but first customers won't need it (single-tenant deployments). v0.3. |
| `action-chunk-buffering` | 7 | Physical Intelligence pattern. Ship when first customer hits the 20Hz/100Hz timing problem. |
| `latency-histograms` | 6 | Observability polish. v0.3. |
| `stripe-license-gating` | 6 | Invoice the first 5 customers manually. Automate in v0.3. |
| `adaptive-denoise-fix` | 5 | pi0-only feature, edge case. v0.3. |
| `determinism-version-hash` | 5 | Reproducibility polish. v0.3. |
| `inference-test-coverage` | 4 | Quality improvement. v0.3+. |
| `openvla-exporter` | 4 | Different (tokenized) path. v0.3. |
| `openvla-onnx-parity` | 4 | Depends on openvla-exporter. v0.3. |
| `sqlite-audit-log` | 3 | EU AI Act compliance for EU customers. Ship when first EU customer appears. |

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

**Launch prerequisites** (from `launch/README.md`):
- [ ] All MVP goals landed
- [ ] Jetson benchmark number measured and published
- [ ] pi0 + SmolVLA both at verified cos≥0.999
- [ ] Docker images published to GHCR
- [ ] 5-minute quickstart tested on fresh Mac + fresh Linux box
- [ ] GitHub Issues open + <24h response commitment

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

**Last updated:** 2026-04-17 (MVP ship list expanded 11 → 12 with smolvla-onnx-parity)
