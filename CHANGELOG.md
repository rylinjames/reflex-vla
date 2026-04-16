# Changelog

## Unreleased

### Added
- `reflex validate` command now runs a real ONNX/TRT-vs-PyTorch round-trip parity check.
  - Seeded fixtures for SmolVLA, pi0, GR00T (pi0.5 and OpenVLA defer to v2).
  - Per-fixture max/mean L2 abs-diff + summary, JSON and Rich table output.
  - `--init-ci` emits a GitHub Actions workflow template at `.github/workflows/reflex-validate.yml`.
  - Exit codes: 0 pass, 1 fail, 2 error.
- Public exports added to `reflex`: `ValidateRoundTrip`, `load_fixtures`, `SUPPORTED_MODEL_TYPES`.

### Changed
- **BREAKING (from stub):** `reflex validate` default `--threshold` changed from `0.02` (the v0.1 placeholder) to `1e-4`. The stub never performed real validation so no existing deployments depended on the old default. Pass `--threshold 0.02` explicitly to match the previous behavior.
- `reflex validate` now requires a valid `reflex_config.json` inside the export directory — the stub accepted any path.

### Fixed
- `_pytorch_backend` SmolVLA path no longer swallows `AutoConfig` fetch errors silently — now logs a warning with the exception and continues with the fallback head_dim.
- CLI handler now catches `KeyboardInterrupt` explicitly (exits 130) instead of emitting a raw traceback.

## v0.1.0 (previous)

Initial release — see README for the seven-wedge scope at that time.
