# Issue 1 — Scaffold ValidateRoundTrip class module

## Files created
- `/Users/romirjain/Desktop/building projects/reflex-vla/src/reflex/validate_roundtrip.py`

## Verification

Commands run from repo root inside `.venv` with `PYTHONPATH=src` (the package
is not installed editable in this venv; using `PYTHONPATH=src` is equivalent
for import smoke-testing).

```
$ python -c "from reflex.validate_roundtrip import ValidateRoundTrip; print('import-ok')"
import-ok

$ grep -c "class ValidateRoundTrip" src/reflex/validate_roundtrip.py
1                              # >= 1, OK

$ grep -c "NotImplementedError" src/reflex/validate_roundtrip.py
6                              # >= 5, OK (5 stub methods + 1 docstring mention)
```

## Decisions / deviations

- **Backend imports** are wrapped in a `try/except ImportError` block per the
  cross-cutting constraint. Both `_pytorch_backend` and `_onnx_backend` are
  imported on the same line so a single failure leaves both as `None`. Logs a
  debug line referencing Issue 4/5.
- **`model_type` detection** in `__init__` first reads `reflex_config.json`'s
  `model_type` field; only if that's missing does it fall back to loading the
  checkpoint and calling `detect_model_type`. This avoids a multi-GB checkpoint
  load on the happy path. If neither `model_type` nor `model_id` is available,
  `__init__` raises `ValueError` with a clear message.
- **`UNSUPPORTED_MODEL_MESSAGE`** is exposed as a module-level constant (also
  re-exported in `__all__`) so tests in Issue 7 can assert against it without
  duplicating the literal.
- **`SUPPORTED_MODEL_TYPES`** is a module-level tuple `("smolvla", "pi0", "gr00t")`.
- All five required private stub methods raise `NotImplementedError("filled in
  by Issue N")`. `run()` also raises `NotImplementedError("filled in by Issue
  6")` since it requires the backends + comparison wiring.
- Type hints on every method signature; `from __future__ import annotations`
  at top so `str | None` works on Python 3.10+ at runtime via PEP 604 strings.
- Class docstring documents reference semantics, seed bridge rationale, and
  the 0/1/2 exit code convention as required.
- File is ~225 lines, well under the 400-line budget.

## Notes for downstream issues
- Issue 4 will populate `_load_pytorch` and `_generate_initial_noise`.
- Issue 5 will populate `_load_onnx`.
- Issue 6 will populate `run`, `_compare`, and `_aggregate`, plus replace the
  guarded backend import block once `_pytorch_backend.py` and
  `_onnx_backend.py` exist (the TODO comment there points at Issues 4/5).
