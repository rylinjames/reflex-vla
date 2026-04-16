# Issue 6 — Wire CLI handler + output + exit codes + README example

Status: COMPLETE

## Part A — Filled in `ValidateRoundTrip` private methods

File: `src/reflex/validate_roundtrip.py`

- Replaced guarded `_pytorch_backend` / `_onnx_backend` import with hard imports
  of `load_pytorch_backend` and `load_onnx_backend` (Issues 4/5 are landed).
- Added `_ACTION_DIM_FALLBACK` map (smolvla=6, pi0=14, gr00t=64).
- `_load_pytorch()` -> `load_pytorch_backend(self.export_dir, self.model_id, self.device)`
- `_load_onnx()` -> `load_onnx_backend(self.export_dir, self.device)`
- `_generate_initial_noise(rng)` reads `action_chunk_size`/`chunk_size` (fallback 50)
  and `action_dim` (fallback per model_type), produces
  `torch.randn((chunk_size, action_dim), generator=rng).numpy().astype(np.float32)`.
- `_compare(pt, onnx)` calls `validate_outputs(..., threshold=self.threshold, name="roundtrip")`
  and returns `.to_dict()`.
- `_aggregate(per_fixture)` returns `{model_type, threshold, num_test_cases, seed,
  results, summary: {max_abs_diff_across_all, passed}}`.
- `run()` loads both backends once, loads fixtures, builds a single seeded CPU
  `torch.Generator`, loops fixtures generating fresh noise per fixture from the
  shared generator, calls both backend `.forward(image, prompt, state, noise)`,
  appends per-fixture compare result with `fixture_idx`, and returns aggregate.

## Part B — CLI handler (`src/reflex/cli.py`)

Replaced the 12-line stub with the full typer handler. Highlights:

- Positional `export_dir` (now optional so `--init-ci` works without one).
- Options: `--model`, `--threshold` (default `1e-4`, help text mentions the
  v0.1 -> 1e-4 change), `--num-cases` (5), `--seed` (0), `--device` (cpu),
  `--output-json` flag, `--init-ci` flag, `--verbose`.
- `--init-ci` short-circuits to `emit_ci_template(Path(".github/workflows/reflex-validate.yml"),
  reflex_version=__version__)` and exits 0 (or 2 on FileExists / other error).
- Validates `--device` is cpu or cuda (else exit 2).
- Catches `FileNotFoundError` (missing export dir or ONNX) -> exit 2.
- Catches `ValueError` from `ValidateRoundTrip.__init__` (unsupported model,
  missing config) -> exit 2.
- Catches generic `Exception` -> traceback in verbose mode, exit 2.
- Output:
  - `--output-json` -> `print(json.dumps(result, indent=2, default=str))`
  - Otherwise two Rich tables: per-fixture (fixture_idx, max_abs_diff,
    mean_abs_diff, passed) + summary (max_abs_diff_across_all, passed,
    num_cases, seed, threshold).
- Exits 0 if `summary.passed`, else 1.

## Part C — README example

File: `README.md`. Inserted a "Validation - round-trip ONNX vs PyTorch parity"
section after the Quickstart, before "Composable wedges". Includes the exact
suggested command (`reflex validate ./p0 --model lerobot/pi0_base --threshold 1e-4`),
a sample abbreviated passing output, and a note about exit codes,
`--output-json`, and `--init-ci`.

## Verification

All checks from the issue brief pass:

```
reflex validate --help | grep -qi init-ci          -> OK
reflex validate --help | grep -qi "1e-4|0.0001"    -> OK (default: 0.0001 visible)
python -c "from reflex.validate_roundtrip import ValidateRoundTrip" -> import-ok
grep -c "ValidateRoundTrip(" src/reflex/cli.py     -> 1
grep -c "init_ci|init-ci" src/reflex/cli.py        -> 3
grep -c "reflex validate" README.md                -> 3
```

`reflex validate --help` renders cleanly with all flags including the threshold
default (`0.0001`) and `--init-ci`.

## Notes / out of scope

- Tests for the round-trip are Issue 7's responsibility (still in_progress).
- Did not run the full pipeline end-to-end against a real export (no model
  downloaded in this sandbox); behavior is verified via the import + help
  surface plus the static checks above.
- The previous guarded `try: import _pytorch_backend / _onnx_backend` shim
  was removed since both modules are now landed; this is a deliberate
  hard-fail import to avoid silent NotImplementedError surfaces.
