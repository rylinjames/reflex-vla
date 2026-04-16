# Plan: Fill in `reflex validate` — round-trip parity harness

**Date:** 2026-04-16
**Source:** `.agents/research/2026-04-16-reflex-validate-stub.md`

## Overview
Replace the stubbed `reflex validate` command with a real round-trip harness that loads an export directory, runs the same (image, prompt, state) fixtures through both the exported ONNX/TRT path and the upstream PyTorch checkpoint, computes per-action-dim L2 max_diff, and emits JSON + Rich output with a configurable pass/fail threshold (default 1e-4). Ship fixtures for SmolVLA / pi0 / GR00T; skip pi0.5 and OpenVLA in v1. Add `--init-ci` flag that writes a GitHub Actions workflow template.

## Boundaries

**Always:**
- Preserve existing `src/reflex/validate.py` utilities (`ValidationResult`, `validate_outputs`, `validate_decomposition`) — expand, don't replace.
- Reuse `reflex.checkpoint.load_checkpoint` and `detect_model_type` for PyTorch reference.
- Pin `torch.manual_seed(seed)` before every validation run; log `seed` in output.
- **Noise bridge:** generate initial noise ONCE in torch via a seeded generator, convert to numpy, pass the *identical array* to both PyTorch and ONNX forward paths. `torch.manual_seed()` does not seed numpy.
- **Reference semantics (v1):** "PyTorch reference" = the exporter's decomposed PyTorch model (same architecture used to produce the ONNX). This proves ONNX export correctness. Upstream-vs-exporter parity is a separate v2 concern.
- **GR00T scope (v1):** validate only the DiT expert with `embodiment_id=0` pinned, matching what `gr00t_exporter.py` actually emits. Full-stack validation (SigLIP2 + Qwen3 + action_encoder/decoder) is v2.
- Exit codes: 0 pass, 1 fail, 2 error (missing ONNX, invalid export dir).
- CPU-only PyTorch reference load (memory discipline; pi0 / GR00T are 3B+).
- Default threshold 1e-4; overridable via `--threshold`.
- All new modules under `src/reflex/` with matching tests under `tests/`.
- **Cross-runtime backends in separate files:** `_pytorch_backend.py` and `_onnx_backend.py` — avoids merge conflicts when the two paths are implemented in parallel.

**Ask First:** (auto mode — logged only, not blocking)
- Whether known-good images should be baked into the repo (v1 uses seeded synthetic tensors).
- Whether `reflex validate` should expose `--init-ci --provider gitlab|circleci` for multi-CI support (v1 ships GitHub only).

**Never:**
- Do NOT add pi0.5 support in v1 — `pi0_exporter.py:83` explicitly rejects it; wait until pi0.5 exporter path lands.
- Do NOT add OpenVLA validation in v1 — it uses a different (non-flow-matching) action head; handle in v2 as separate subcommand.
- Do NOT attempt to validate multiple export dirs in one invocation (single-export-at-a-time memory scope).
- Do NOT load PyTorch ref on CUDA by default (CPU only unless `--device cuda` explicitly passed).
- Do NOT change existing exporter validation thresholds (0.01) — only the new `validate` command uses 1e-4.

## Baseline audit

| Metric | Command | Result |
|--------|---------|--------|
| `reflex validate` stub LOC | `wc -l src/reflex/cli.py` (stub at 165-176) | 12 lines stubbed |
| Existing validate module LOC | `wc -l src/reflex/validate.py` | 106 lines (keep + extend) |
| Exporters with parity pattern | `ls src/reflex/exporters/*_exporter.py` | 4 (smolvla, pi0, gr00t, openvla — note openvla delegates) |
| Fixtures directories | `ls src/reflex/fixtures/ tests/fixtures/` | Neither exists — both to be created |
| CI workflows | `ls .github/workflows/` | None (to be created by --init-ci) |
| `bd` CLI availability | `which bd` | Not available — using TaskCreate for tracking |

## Conformance checks

| Issue | Check Type | Check |
|-------|-----------|-------|
| Issue 1 | files_exist | `["src/reflex/validate_roundtrip.py"]` |
| Issue 1 | content_check | `{file: "src/reflex/validate_roundtrip.py", pattern: "class ValidateRoundTrip"}` |
| Issue 2 | files_exist | `["src/reflex/fixtures/__init__.py", "src/reflex/fixtures/vla_fixtures.py"]` |
| Issue 2 | content_check | `{file: "src/reflex/fixtures/vla_fixtures.py", pattern: "def load_fixtures"}` |
| Issue 3 | files_exist | `["src/reflex/ci_template.py"]` |
| Issue 3 | content_check | `{file: "src/reflex/ci_template.py", pattern: "reflex validate"}` |
| Issue 4 | content_check | `{file: "src/reflex/validate_roundtrip.py", pattern: "_pytorch_forward"}` |
| Issue 5 | content_check | `{file: "src/reflex/validate_roundtrip.py", pattern: "_onnx_forward"}` |
| Issue 6 | content_check | `{file: "src/reflex/cli.py", pattern: "ValidateRoundTrip("}` |
| Issue 6 | content_check | `{file: "src/reflex/cli.py", pattern: "init_ci"}` |
| Issue 6 | command | `reflex validate --help | grep -q 'init-ci'` |
| Issue 7 | files_exist | `["tests/test_validate_roundtrip.py"]` |
| Issue 7 | tests | `pytest tests/test_validate_roundtrip.py -v` |

## Issues

### Issue 1: Scaffold `ValidateRoundTrip` class module
**Dependencies:** None
**Files:** `src/reflex/validate_roundtrip.py` (new)
**Acceptance:**
- File exists with `ValidateRoundTrip` class
- `__init__(export_dir, model_id=None, threshold=1e-4, num_test_cases=5, seed=0, device="cpu")` signature
- `run() -> dict` method stub
- Imports from `reflex.validate`, `reflex.checkpoint`, `onnxruntime`, `torch`, `numpy`
- Docstrings on class + public methods

**Description:** Create the orchestrator class that glues together fixture loading, PyTorch ref forward, ONNX forward, and comparison. Include type stubs for `_load_pytorch`, `_load_onnx`, `_pytorch_forward`, `_onnx_forward`, `_compare`, `_aggregate`. Raise `NotImplementedError` in method bodies for implementation in later issues. Import `ValidationResult` and `validate_outputs` from the existing `src/reflex/validate.py`.

### Issue 2: Per-model seeded fixture generators
**Dependencies:** None
**Files:** `src/reflex/fixtures/__init__.py`, `src/reflex/fixtures/vla_fixtures.py` (both new)
**Acceptance:**
- `load_fixtures(model_type: str, num: int, seed: int) -> list[tuple]` returns N `(image, prompt, state)` tuples
- Deterministic per (model_type, num, seed) input
- Per-model correct shapes: SmolVLA image 384x384, state 6-dim; pi0 image 224x224, state 14-dim (or per config); GR00T image 224x224, state per embodiment_0 config
- Pin `torch.manual_seed(seed)`; use `torch.randn` for images and states; use hardcoded prompt list per model type
- Raise `ValueError` for unsupported model types (pi05, openvla)

**Description:** Synthetic seeded fixtures keep v1 lightweight (no bundled images) and fully reproducible across environments. Images produced as 0-1 float tensors via `torch.randn` clamped to `[0, 1]`. Prompts are a small curated list (e.g., `["pick up the red cup", "move the block to the left", "place the object on the shelf"]`). States per-model shape derived from `ExportConfig` metadata or hardcoded per research findings.

### Issue 3: `--init-ci` GitHub Actions template
**Dependencies:** None
**Files:** `src/reflex/ci_template.py` (new)
**Acceptance:**
- `emit_ci_template(output_path: Path, reflex_version: str) -> None` writes a valid YAML
- Template includes `on: [pull_request, push]`, a `validate` job, pytest + `reflex validate` steps
- Passing `--init-ci` to the CLI calls this and writes to `.github/workflows/reflex-validate.yml`
- Creates `.github/workflows/` directory if missing
- If file already exists, errors with clear message (we're in --auto mode)
- **CI matrix constraint:** active job runs SmolVLA only on `ubuntu-latest` (fits in 7GB runner memory). pi0 and GR00T steps are included but commented with a `# Requires self-hosted runner with 16GB+ RAM` header.

**Description:** Emit a working CI workflow template that pins the current reflex version, runs on pushes + PRs, installs the package, and runs `reflex validate ./export_dir` for SmolVLA on GitHub-hosted runners. pi0 and GR00T blocks are present as commented templates with self-hosted runner labels. Workflow uses `ubuntu-latest` with Python 3.11 and installs `reflex-vla[serve,onnx,dev]` for CPU-only CI validation. Template is a Python string literal in `ci_template.py`; no jinja2 dependency needed.

### Issue 4: PyTorch reference loading + forward pass
**Dependencies:** Issue 1, Issue 2
**Files:** `src/reflex/_pytorch_backend.py` (new — separate from `validate_roundtrip.py` to avoid file conflict with Issue 5)
**Acceptance:**
- Module exports `load_pytorch_backend(export_dir, model_id, device) -> PyTorchBackend`
- `PyTorchBackend.forward(image, prompt, state, initial_noise) -> np.ndarray` returns action chunk (shape `[chunk_size, action_dim]`) — `initial_noise` is provided externally so ONNX path uses the identical tensor
- Per-model dispatch: reuses `build_expert_stack` from SmolVLA / pi0 / `build_gr00t_expert_stack` from GR00T exporter modules (import directly — they are already factored)
- Loads checkpoint via `reflex.checkpoint.load_checkpoint` onto `self.device` (default CPU)
- Logs which decomposed architecture was reconstructed
- Raises `NotImplementedError` for pi0.5 / OpenVLA with clear message

**Description:** Bridge between loaded safetensors and callable forward pass. v1 uses the exporter's decomposed PyTorch surrogate as the "reference" — this proves ONNX export correctness (the primary goal) and is the only honest comparison when upstream repos aren't importable. Load state_dict into the decomposed stack, run the denoising loop (10 steps for SmolVLA/pi0, 4 for GR00T) from the externally-provided `initial_noise`, return the final action chunk as numpy. GR00T forward uses `embodiment_id=0`.

### Issue 5: ONNX inference path
**Dependencies:** Issue 1, Issue 2
**Files:** `src/reflex/_onnx_backend.py` (new — separate from `validate_roundtrip.py` to avoid file conflict with Issue 4)
**Acceptance:**
- Module exports `load_onnx_backend(export_dir, device) -> ONNXBackend`
- `ONNXBackend.forward(image, prompt, state, initial_noise) -> np.ndarray` runs the same denoising loop via `onnxruntime.InferenceSession` — receives the *same* `initial_noise` numpy array the PyTorch path used
- Provider priority: `CPUExecutionProvider` for v1 (deterministic, CI-portable). `CUDAExecutionProvider` / `TensorrtExecutionProvider` deferred until the CUDA CI story lands.
- Records which provider loaded; logs ONNX opset version found in the graph
- Reads `reflex_config.json` for input shapes, step count
- Asserts output shape matches `[chunk_size, action_dim]` before returning

**Description:** Mirror of the PyTorch forward pass, executed through the exported ONNX graph. Uses the exact numpy array the PyTorch path received as `initial_noise` (critical for seed bridging — torch.manual_seed does not seed numpy). v1 runs CPU-only for reproducibility and CI portability; GPU providers deferred.

### Issue 6: Wire CLI handler + output + exit codes
**Dependencies:** Issue 1, Issue 3, Issue 4, Issue 5
**Files:** `src/reflex/cli.py` (modify stub at lines 165-176); `README.md` (happy-path example)
**Acceptance:**
- `reflex validate <export_dir>` runs `ValidateRoundTrip` end-to-end
- `--model <hf_id>` option for ref checkpoint override (auto-detected from `reflex_config.json` if omitted)
- `--threshold <float>` (default 1e-4) — help text notes this is a change from the v0.1 stub default of 0.02
- `--num-cases <int>` (default 5)
- `--seed <int>` (default 0)
- `--device {cpu,cuda}` (default cpu)
- `--output-json` flag switches output from Rich table to pure JSON
- `--init-ci` short-circuits validation and emits `.github/workflows/reflex-validate.yml`
- `reflex validate --help` shows all flags and the example
- Exit code: 0 pass, 1 fail (at least one fixture over threshold), 2 error (missing ONNX, bad config, etc.)
- **Happy-path example in README:** `reflex validate ./pi0 --model lerobot/pi0_base` with expected output sample

**Description:** Replace the stub with full handler. Load config, detect model_type, build `ValidateRoundTrip` (which internally wires `_pytorch_backend` + `_onnx_backend` + fixtures), call `.run()`, format output, exit with correct code. JSON output schema matches research spec. Rich table uses `rich.table.Table` with per-fixture rows + summary footer. Add a 5-line "Validation" section to README under Quickstart.

### Issue 7: Tests for validate round-trip
**Dependencies:** Issue 1, Issue 2, Issue 4, Issue 5
**Files:** `tests/test_validate_roundtrip.py` (new)
**Acceptance:**
- At least 6 test cases covering:
  - Fixture determinism (same seed → identical (image, prompt, state) tuples)
  - **Seed-bridge equivalence** — the `initial_noise` tensor generated once equals the numpy array consumed by both `_pytorch_backend` and `_onnx_backend`
  - Threshold pass
  - Threshold fail
  - Missing ONNX error (exit 2)
  - Unsupported model type (pi0.5 / OpenVLA raises ValueError with expected message)
- `pytest tests/test_validate_roundtrip.py -v` passes cleanly
- Tests use monkeypatching and a small mock export dir to avoid loading real 3B-param checkpoints
- Optional: one integration test gated by `REFLEX_INTEGRATION=1` env var that exercises a real SmolVLA export

**Description:** Unit-test the orchestrator with mocked backends so CI runs fast. The seed-bridge test is critical — verifies that PyTorch's noise and ONNX's input are byte-identical arrays. Integration test for a real SmolVLA export is opt-in because the model download + export is slow. Cover deterministic seed path, threshold decision logic, CI template emission, and error cases.

## Execution order

**Wave 1** (parallel, 3 issues): Issue 1, Issue 2, Issue 3
**Wave 2** (after Wave 1, parallel, 2 issues): Issue 4, Issue 5
**Wave 3** (after Wave 2, parallel, 2 issues): Issue 6, Issue 7

Total: 7 issues, 3 waves.

## Next Steps
- Phase 3: `/pre-mortem` against this plan
- Phase 4: `/crank` for autonomous execution
- Phase 5: `/vibe` on recent changes
- Phase 6: `/post-mortem` harvests learnings + next-work
