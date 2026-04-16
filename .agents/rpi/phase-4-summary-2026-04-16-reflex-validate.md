# RPI Phase 4 Summary — crank complete

**Date:** 2026-04-16
**Epic:** Fill in reflex validate stub

## Execution

3 waves, 7 issues, 0 retries, 0 BLOCKED.

**Wave 1 (parallel, 3 issues):**
- Issue 1: `src/reflex/validate_roundtrip.py` (243 LOC) — `ValidateRoundTrip` scaffold
- Issue 2: `src/reflex/fixtures/` (80 LOC) — per-model seeded fixtures
- Issue 3: `src/reflex/ci_template.py` (149 LOC) — GitHub Actions template emitter
- Commit: `e1455f7`

**Wave 2 (parallel, 2 issues):**
- Issue 4: `src/reflex/_pytorch_backend.py` (~290 LOC) — exporter-surrogate PyTorch ref
- Issue 5: `src/reflex/_onnx_backend.py` (~290 LOC) — ORT CPU-only backend
- Post-wave fix: aligned PyTorch forward's denoise schedule to canonical `dt=-1/N, t=1+step*dt` scheme (matches `inference.py` + `runtime/server.py`)
- Commit: `ac81175`

**Wave 3 (parallel, 2 issues):**
- Issue 6: `src/reflex/validate_roundtrip.py` (wired up), `src/reflex/cli.py` (full handler), `README.md` (Validation section)
- Issue 7: `tests/test_validate_roundtrip.py` (~270 LOC, 11 passing + 1 integration skipped)
- Commit: `<HEAD>`

## Verification

- `pytest tests/test_validate_roundtrip.py -v` → 11 passed, 1 skipped (integration)
- `reflex validate --help` → shows all flags including `--init-ci` with threshold default `0.0001`
- `grep ValidateRoundTrip src/reflex/cli.py` → wired
- Import smoke across all new modules clean

## Completion marker

`<promise>DONE</promise>`

## Notes

- One post-wave-2 fix: PyTorch backend used wrong denoise direction (0→1 vs 1→0). Caught by hand-verification before Wave 3 spawned. Fixed inline.
- GR00T uses `build_gr00t_full_stack` not bare DiT expert — worker made right call to ensure velocity shape matches ONNX output shape.
- One test spec deviation: `test_missing_onnx_error` also stubs `load_pytorch_backend` because the `run()` loads PyTorch before ONNX. Documented.
- `.venv` required bootstrapping pip and installing `reflex-vla[dev]` editable during testing.
