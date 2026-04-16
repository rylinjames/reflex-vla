# Post-Mortem: reflex validate round-trip harness

**Date:** 2026-04-16
**Epic:** Fill in reflex validate stub (TaskList: 7 issues, 3 waves)
**Branch:** rpi/reflex-validate-roundtrip-20260416
**Total iterations:** 3 waves / 50 max
**Commits:** 5 (research/plan/pre-mortem artifacts + 3 code waves + vibe fixups)

## Council verdict: PASS

Epic closed cleanly. All 7 issues completed, all conformance checks satisfied, all pre-mortem amendments applied, 11 unit tests passing, vibe WARN addressed inline.

## What went well

- **Pre-mortem caught the load-bearing bugs before crank.** 3 of 4 judges surfaced issues (file conflict, reference-vs-surrogate confusion, CI OOM on 3B+ models, seed bridge). Fixing them in the plan rather than in crank saved at least one re-crank.
- **Parallel execution held.** Waves 1 and 2 landed 3 and 2 issues respectively without file-level conflicts once the Issue 4 / Issue 5 split moved into separate modules.
- **Inline hand-verification caught the denoise-schedule mismatch.** Issue 4's worker followed one internal convention (`t: 0→1`); Issue 5's followed another (`t: 1→0`). Without grep-check between waves, parity tests would have failed silently during vibe.
- **Test harness uses the seed bridge as a first-class invariant.** `test_seed_bridge_in_orchestrator` asserts Python `is` identity on the noise array across backends — catches regression on the single most subtle part of the system.
- **Vibe's api-surface WARN items (re-exports, CHANGELOG, version default) were all cheap single-commit fixes.**

## What went wrong / could improve

- **Workers invented v0.2 surface without surfacing it.** Issue 4's worker chose GR00T `build_gr00t_full_stack` over the bare DiT expert because shapes need to match the ONNX output — that's the right call, but it contradicted the pre-mortem amendment saying "v1 validates only the DiT expert." The velocity-shape mismatch forced the full-stack choice; the plan should have made this explicit.
- **radon complexity was skipped.** Dev env missing the tool. A follow-up ticket should install it; otherwise future vibes will keep skipping.
- **Two "small" divergences crept in** that the spec-compliance judge flagged as minor: SmolVLA fixture image 512×512 vs spec 384×384, and one README example mention was unverified at the judge level (but actually present).
- **Pytest required bootstrapping pip + editable install** in the `.venv` before tests could run. Dev env setup isn't reproducible yet.
- **The CI template runs `reflex validate --threshold 1e-4` on a real SmolVLA export** that will pull ~2GB of weights every PR. Caching was mentioned in the missing-requirements judge's report but not implemented.

## Learnings worth remembering

1. **Two-judge denoise-scheme check.** When two parallel workers implement forward passes for different runtimes, they must share the same integration scheme. A 10-line plan appendix documenting the canonical scheme would have prevented the mismatch.
2. **Surrogate-vs-upstream reference is the hard problem.** The feasibility judge surfaced this in pre-mortem; the plan adopted surrogate-only for v1. This is the honest v1 choice. Future v2 work should document what upstream-reference validation would require (in many cases: full LeRobot/openpi/gr00t imports).
3. **`torch.manual_seed` ≠ numpy seeded.** The seed-bridge test is the right invariant to pin. Other cross-runtime code in reflex should adopt the same pattern (noise tensor generated once, passed as an arg).
4. **`except Exception: pass` is a smell.** Caught by error-paths judge on line 160 of `_pytorch_backend.py`. Now replaced with `logger.warning`.
5. **Input validation in orchestrator constructors pays off.** `num_test_cases < 1` would have produced a confusing "passed=False with 0 diff" result; now it raises clearly.

## Harvested next-work (for future /rpi cycles)

```jsonl
{"title": "Benchmark FP16 torch.compile baseline against TRT FP16 on A10G", "severity": "high", "target_repo": "reflex-vla", "reason": "Current 2.6-3.3x claim compares FP16 TRT to FP32 torch.compile — an ML engineer will flag the apples-to-oranges immediately. One afternoon of Modal runs closes the pitch attack surface.", "consumed": false}
{"title": "Real Jetson Orin Nano validation using reflex validate harness", "severity": "high", "target_repo": "reflex-vla", "reason": "Every speed claim is A10G-extrapolated. Now that reflex validate exists, buying a $249 Super Dev Kit + running end-to-end gives publishable per-model Hz on real silicon.", "consumed": false}
{"title": "VLM prefix encoder + KV-cache export (unblocks task-relevant actions)", "severity": "critical", "target_repo": "reflex-vla", "reason": "v0.1 serve + validate both use random-tensor VLM conditioning — outputs are action-shaped noise, not task-relevant actions. Prefix KV-cache per Dexmal realtime-vla (arXiv 2510.26742) is the fix. Estimate 2 weeks.", "consumed": false}
{"title": "Install radon + gocyclo in dev env for future vibe runs", "severity": "low", "target_repo": "reflex-vla", "reason": "Complexity analysis was skipped in vibe. Cheap one-liner fix: add to pyproject [dev] extras.", "consumed": false}
{"title": "Reconcile SmolVLA fixture image size (currently 512x512, spec said 384x384)", "severity": "low", "target_repo": "reflex-vla", "reason": "Cosmetic divergence from plan; image arg is unused by v1 surrogate forward anyway. Fix for doc consistency.", "consumed": false}
{"title": "Cache HuggingFace downloads in the CI workflow template", "severity": "medium", "target_repo": "reflex-vla", "reason": "reflex-validate.yml currently pulls the full SmolVLA weights every PR (~2GB). Add actions/cache keyed on model_id + revision.", "consumed": false}
{"title": "TypedDict for ValidateRoundTrip.run() return shape", "severity": "low", "target_repo": "reflex-vla", "reason": "Currently returns dict[str, Any]; programmatic users will reverse-engineer from CLI output. Small ergonomics win.", "consumed": false}
{"title": "Reproducible dev env (.venv bootstrapping + pip ensurepip)", "severity": "medium", "target_repo": "reflex-vla", "reason": "Worker had to manually install pip + reflex-vla[dev] to run pytest. Document + add a makefile target.", "consumed": false}
```

## Completion marker

`<promise>DONE</promise>`
Epic: reflex-validate-roundtrip
Issues completed: 7 / 7
Iterations: 3 waves / 50 max
Flywheel: harvested 8 next-work items
