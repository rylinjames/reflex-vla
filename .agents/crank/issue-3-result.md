# Issue 3 — Result

**Status:** completed
**Files created:** `src/reflex/ci_template.py` (~140 LOC)

## Summary
Implemented `emit_ci_template(output_path, reflex_version="0.1.0", *, overwrite=False)` and `validate_emitted_yaml(path)` in `src/reflex/ci_template.py`. The module-level `TEMPLATE` string contains a single `{reflex_version}` placeholder rendered via `str.format` (no jinja2). All literal `${{ ... }}` GitHub expression syntax is escaped as `${{{{ ... }}}}` in the template so it survives `.format()` cleanly.

## Acceptance criteria coverage
- `name: Reflex Validate`
- `on:` with `pull_request` and `push: branches: [main]`
- Single active job `validate-smolvla` on `ubuntu-latest`
- Python 3.11 setup (`actions/setup-python@v5`)
- Install step uses `pip install 'reflex-vla[serve,onnx,dev] @ git+https://github.com/rylinjames/reflex-vla@v{reflex_version}'`
- Export step: `reflex export lerobot/smolvla_base --target desktop --output ./sv_export`
- Validate step: `reflex validate ./sv_export --threshold 1e-4 --num-cases 3 --output-json > validate_result.json`
- Upload artifact step using `actions/upload-artifact@v4` for `validate_result.json`
- Commented-out `validate-pi0` and `validate-gr00t` blocks, each prefixed with `# Requires self-hosted runner with 16GB+ RAM — uncomment and update runs-on:`
- Job-level `permissions: contents: read`
- Top-level `concurrency:` block with `${{ github.workflow }}-${{ github.ref }}` and `cancel-in-progress: true`
- `FileExistsError` raised with exact message `f"{output_path} exists — pass overwrite=True to replace"` when target exists and `overwrite=False`
- Parent directories created via `mkdir(parents=True, exist_ok=True)`
- `validate_emitted_yaml` imports yaml locally and returns False on ImportError or parse failure

## Verification run
Ran the exact verification script provided in the assignment. Output:
```
yaml parse OK
ci-template-ok
```
PyYAML is installed in the env, so the yaml-parse branch executed and returned True.

## Notes for downstream issues
- Issue 6 should import `from reflex.ci_template import emit_ci_template` and call it with `Path(".github/workflows/reflex-validate.yml")`. The function already does the parent-dir creation and file-exists guard, so the CLI just needs to surface the `FileExistsError` to the user (auto mode: clear error message + exit non-zero).
- `reflex_version` defaults to `"0.1.0"`; CLI can plumb through `reflex.__version__` once that's settled.
