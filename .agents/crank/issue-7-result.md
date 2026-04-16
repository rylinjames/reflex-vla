# Issue 7 — Tests for validate round-trip — RESULT

## Files created
- `tests/test_validate_roundtrip.py` (~270 LOC)

## pytest output
```
tests/test_validate_roundtrip.py::test_fixture_determinism PASSED
tests/test_validate_roundtrip.py::test_seed_bridge_equivalence PASSED
tests/test_validate_roundtrip.py::test_threshold_pass PASSED
tests/test_validate_roundtrip.py::test_threshold_fail PASSED
tests/test_validate_roundtrip.py::test_threshold_fail_then_pass_with_loose_threshold PASSED
tests/test_validate_roundtrip.py::test_missing_onnx_error PASSED
tests/test_validate_roundtrip.py::test_unsupported_model_raises_pi05 PASSED
tests/test_validate_roundtrip.py::test_unsupported_model_raises_openvla PASSED
tests/test_validate_roundtrip.py::test_unsupported_model_raises_unknown PASSED
tests/test_validate_roundtrip.py::test_ci_template_emission PASSED
tests/test_validate_roundtrip.py::test_seed_bridge_in_orchestrator PASSED
tests/test_validate_roundtrip.py::test_integration_smolvla_export SKIPPED

11 passed, 1 skipped in 2.31s
```

## Skipped tests
- `test_integration_smolvla_export` — gated on `REFLEX_INTEGRATION=1` env var (requires HF token + ~30 min real export).

## xfail / upstream-bug skips
None.

## Deviations from spec
- `test_missing_onnx_error` also monkeypatches `load_pytorch_backend` to a stub. Reason: `ValidateRoundTrip.run()` calls `_load_pytorch()` before `_load_onnx()`. With only `reflex_config.json` and a fake `model_id` ("stub/none"), the unmocked PyTorch loader hits HuggingFace and raises `RepositoryNotFoundError` before the missing-ONNX path is exercised. Stubbing PyTorch lets the test isolate the FileNotFoundError raised by `load_onnx_backend` for the missing `expert_stack.onnx`, which is the invariant the test is meant to lock in.
- Added a bonus test `test_threshold_fail_then_pass_with_loose_threshold` to sanity-check that the threshold parameter actually drives the pass/fail decision (cheap addition, no spec conflict).
- Added a bonus test `test_seed_bridge_in_orchestrator` that asserts (a) within one run, both backends receive the same numpy buffer object (`is`-identical, no copy), and (b) reseeding produces byte-identical noise across runs. This is a stronger end-to-end version of the seed-bridge invariant.
- Bonus `test_ci_template_emission` is included (the spec marked it optional).

## Environment notes
- `.venv` did not initially have `pip` bootstrapped; ran `python -m ensurepip --upgrade` then `python -m pip install -e .` plus `pytest numpy torch onnxruntime`. The verification command in the issue (`source .venv/bin/activate && pytest ...`) works after that bootstrap.

## Coverage map (spec → test)
| Spec test | Implemented as |
| --- | --- |
| 1. fixture_determinism | `test_fixture_determinism` |
| 2. seed_bridge_equivalence | `test_seed_bridge_equivalence` (+ `test_seed_bridge_in_orchestrator`) |
| 3. threshold_pass | `test_threshold_pass` |
| 4. threshold_fail | `test_threshold_fail` |
| 5. missing_onnx_error | `test_missing_onnx_error` |
| 6. unsupported_model_raises | `test_unsupported_model_raises_pi05` / `_openvla` / `_unknown` |
| 7. ci_template_emission (bonus) | `test_ci_template_emission` |
| Integration (opt-in) | `test_integration_smolvla_export` (skip-unless `REFLEX_INTEGRATION=1`) |
