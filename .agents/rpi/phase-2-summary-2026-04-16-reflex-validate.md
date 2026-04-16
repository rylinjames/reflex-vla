# RPI Phase 2 Summary — plan for reflex validate round-trip harness

**Date:** 2026-04-16
**Plan doc:** `.agents/plans/2026-04-16-reflex-validate-roundtrip.md`
**Epic:** Fill in reflex validate stub (TaskCreate — bd not available)

## Decomposition

7 issues across 3 waves.

**Wave 1 (parallel, 3 issues, no dependencies):**
- Task #1: Scaffold ValidateRoundTrip class module
- Task #2: Per-model seeded fixture generators
- Task #3: GitHub Actions workflow template + --init-ci emission

**Wave 2 (parallel, 2 issues, depends on W1):**
- Task #4: PyTorch reference loading + forward pass
- Task #5: ONNX inference path

**Wave 3 (parallel, 2 issues, depends on W1 + W2):**
- Task #6: Wire CLI validate handler + output + exit codes
- Task #7: Tests for validate round-trip

## Fast-path check

ISSUE_COUNT = 7, BLOCKED_COUNT = 4 (tasks 4, 5, 6, 7 all have blockedBy). Not a micro-epic. **fast_path = false** — full council at every gate.

## Cross-cutting constraints

- Reuse `reflex.validate` utilities (don't replace).
- CPU-only PyTorch ref load by default.
- Seed-pin every run, log in output.
- Exit codes 0 pass / 1 fail / 2 error.
- Skip pi0.5 + OpenVLA in v1.

## Next

Phase 3 (pre-mortem) against this plan.
