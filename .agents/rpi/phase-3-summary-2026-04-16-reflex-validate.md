# RPI Phase 3 Summary — pre-mortem on reflex validate plan

**Date:** 2026-04-16
**Plan:** `.agents/plans/2026-04-16-reflex-validate-roundtrip.md`
**Pre-mortem report:** `.agents/council/2026-04-16-pre-mortem-reflex-validate.md`

## Council verdict: WARN

4 judges (missing-requirements, feasibility, scope, spec-completeness). 3 WARN + 1 PASS → overall WARN. /rpi gate logic: auto-proceed. Inline amendments applied to plan before crank.

## Amendments applied (pre-crank)

1. Issues 4 + 5 split into separate files (`_pytorch_backend.py`, `_onnx_backend.py`) — removes Wave 2 merge conflict
2. Reference semantics locked: v1 compares ONNX to exporter's decomposed PyTorch surrogate (proves ONNX export correctness); upstream-vs-exporter parity = v2
3. GR00T scope cut: v1 validates DiT expert only (embodiment_id=0), not full stack
4. Issue 3 CI template: SmolVLA on github-hosted only (7GB RAM limit); pi0/GR00T commented with `# self-hosted runner 16GB+ RAM`
5. Seed bridge: initial noise tensor generated ONCE in torch, same numpy array passed to both backends
6. Issue 6 CLI: README happy-path example added to acceptance
7. Issue 7 tests: added seed-bridge equivalence test as a critical case

## Tasks updated

- Task #4: split to `_pytorch_backend.py` + reference semantics note
- Task #5: split to `_onnx_backend.py` + seed bridge note + CPU-only provider
- Task #6: README happy-path added to acceptance
- Task #7: seed-bridge equivalence test added

## Ready for crank

All blocking concerns resolved. WARN → auto-proceed. Moving to Phase 4.
