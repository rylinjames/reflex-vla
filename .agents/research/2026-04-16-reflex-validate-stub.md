# Research: reflex validate stub — round-trip parity harness

**Date:** 2026-04-16
**Backend:** claude-native-teams (Explore agent)
**Scope:** fill in `reflex validate` to round-trip (image, prompt, state) tuples through ONNX/TRT vs upstream PyTorch checkpoint; per-action-dim L2 max_diff; pass/fail on configurable threshold; known-good fixtures for SmolVLA / pi0 / pi0.5 / GR00T; emit GitHub Action template via `--init-ci`.

## Summary

`reflex validate` is stubbed at `src/reflex/cli.py:165` (argparse + placeholder print only). A validation module already exists at `src/reflex/validate.py` with `ValidationResult`, `validate_outputs()`, and `validate_decomposition()` utilities. Each exporter runs single-step ONNX parity checks at export time (threshold 0.01) but no post-export round-trip harness exists. The implementation must integrate checkpoint loading, ONNX session management, per-model preprocessing, fixture orchestration, JSON/Rich output, and CI template generation.

## Key Files

| File | Purpose |
|------|---------|
| `src/reflex/cli.py:165-176` | Stubbed `validate` command — needs full implementation |
| `src/reflex/validate.py` | Core utilities: `ValidationResult`, `validate_outputs()`, `validate_decomposition()` |
| `src/reflex/checkpoint.py:93-150` | `detect_model_type()`, `load_checkpoint()` |
| `src/reflex/exporters/smolvla_exporter.py:299-316` | Export-time ONNX validation pattern |
| `src/reflex/exporters/pi0_exporter.py:246-263` | pi0 validation (mirrors SmolVLA) |
| `src/reflex/exporters/pi0_exporter.py:12,83-84` | pi0 rejects pi0.5 checkpoints explicitly |
| `src/reflex/exporters/gr00t_exporter.py:705-720,735-736` | GR00T validation with `embodiment_id=0` pin |
| `src/reflex/exporters/openvla_exporter.py:1-66` | Delegates to optimum-cli; 30-line post-process helper |
| `src/reflex/decompose.py:15-85` | `DecomposedRMSNorm`, `DecomposedAdaRMSNorm`, `DecomposedRotaryEmbedding` |
| `src/reflex/runtime/server.py:189-261` | ONNX InferenceSession + provider selection |
| `src/reflex/config.py:96-108` | `ExportConfig` dataclass, `validate: bool = True` |
| `tests/test_validate.py` | Existing unit tests for `validate_outputs()` |

## Current state of `reflex validate`

```python
# src/reflex/cli.py:165-176
def validate(
    export_dir: str = typer.Argument(...),
    model: str = typer.Option("", ...),
    threshold: float = typer.Option(0.02, ...),
    verbose: bool = typer.Option(False, ...),
):
    """Validate exported model quality against PyTorch reference."""
    _setup_logging(verbose)
    console.print(f"\n[bold]Reflex Validate[/bold]")
    console.print(f"  Export: {export_dir}")
    console.print(f"  Threshold: {threshold}")
    # STUB: no actual validation logic
```

**What must be built:**
1. Load `reflex_config.json` from export_dir; infer model_type + architecture
2. Load PyTorch reference (from `--model` or inferred from config)
3. Generate or load per-model (image, prompt, state) fixtures
4. Run PyTorch forward + ONNX forward; compare per-action-dim L2 max_diff
5. Pass/fail on `--threshold` (default 1e-4)
6. Emit JSON + Rich table; exit code 0=pass, 1=fail, 2=error
7. `--init-ci` emits `.github/workflows/reflex-validate.yml`

## Existing validation patterns to reuse

**Export-time pattern** (`smolvla_exporter.py:299-316`):
```python
sess = ort.InferenceSession(str(onnx_path))
onnx_out = sess.run(None, input_dict)[0]
max_diff = np.abs(ort_out - torch_out).max()
```

**Reusable functions:**
- `reflex.validate.validate_outputs(reference, candidate, threshold, name)` → `ValidationResult`
- `reflex.checkpoint.load_checkpoint(path_or_id, device)` → `(state_dict, config)`
- `reflex.checkpoint.detect_model_type(state_dict)` → `"smolvla" | "pi0" | "pi05" | "gr00t" | "openvla" | None`

## Per-model specifics

### SmolVLA (450M, action_dim=6, 10 steps, chunk=50)
- Expert stack inputs: `noisy_actions [1,50,6]`, `timestep [1]`, `position_ids [1,50]` → `velocity [1,50,6]`
- Export-time max_diff: < 1e-5
- ONNX components: vision_encoder, backbone, expert_stack

### pi0 (3.5B, action_dim=32)
- PaliGemma-2B backbone + action expert
- **No AdaRMSNorm** (pi0.5 only). pi0 exporter rejects pi0.5 at `pi0_exporter.py:83`.

### pi0.5 (3.62B, action_dim=32)
- `DecomposedAdaRMSNorm` with (scale, shift, gate) triplets
- Detection marker: `input_layernorm.dense.weight`
- **Status: separate exporter path not yet in main codebase** — validate may need to skip pi0.5 or wait

### GR00T N1.6 (3.29B, action_dim=128, 4 steps)
- `DecomposedAdaLN` (2-chunk: scale+shift, NOT 3-chunk like pi0.5)
- 32-embodiment design but **`embodiment_id=0` pinned during export** (`gr00t_exporter.py:736`)
- Full-stack export includes `action_encoder` + DiT + `action_decoder`
- `full_stack: true` flag in config

### OpenVLA (7.5B)
- Not a custom exporter — delegates to `optimum-cli export onnx`
- Action head: `argmax(lm_logits[:, -7:])` + bin lookup (not flow matching)
- Reflex role: 30-line `decode_actions()` helper in `postprocess.openvla`
- Validate should check bin decoding separately or skip OpenVLA in v1

## Recommended architecture

### File layout
```
src/reflex/
├── validate.py                      # EXPAND: core utilities (keep existing)
├── validate_roundtrip.py            # NEW: orchestrator + ValidateRoundTrip class
├── fixtures/
│   ├── __init__.py
│   └── vla_fixtures.py              # NEW: per-model (image, prompt, state) generators
├── cli.py                           # MODIFY: fill in validate() + --init-ci flag
└── exporters/
    └── _validation_common.py        # NEW: extracted shared ONNX parity check

tests/
├── test_validate_roundtrip.py       # NEW: unit tests
└── fixtures/images/                 # NEW: optional known-good test images

.github/workflows/
└── reflex-validate.yml.template     # NEW: emitted by --init-ci
```

### Core signatures
```python
class ValidateRoundTrip:
    def __init__(self, export_dir: Path, model_id: str | None, threshold: float = 1e-4, num_test_cases: int = 5, seed: int = 0):
        ...
    def run(self) -> dict: ...

def load_fixtures(model_type: str, num: int, seed: int) -> list[tuple[np.ndarray, str, np.ndarray]]:
    ...

def emit_ci_template(output_path: Path) -> None:
    """Write .github/workflows/reflex-validate.yml with pinned reflex version."""
```

### Output JSON shape
```json
{
  "model_type": "smolvla",
  "status": "pass",
  "threshold": 0.0001,
  "num_test_cases": 5,
  "action_dim": 6,
  "chunk_size": 50,
  "seed": 0,
  "results": [
    {"fixture_idx": 0, "max_abs_diff": 2.3e-05, "mean_abs_diff": 8.1e-06, "passed": true},
    ...
  ],
  "summary": {"max_abs_diff_across_all": 4.7e-05, "passed": true, "latency_pytorch_ms": 145.2, "latency_onnx_ms": 23.4}
}
```

## Open questions

1. **Fixture portability** — bake images into repo (heavier) vs. seeded `torch.randn`? **Recommendation:** seeded synthetic tensors for v1; optional on-disk known-good images for v2.
2. **pi0.5 support** — rejected by pi0 exporter; scope out of v1? **Recommendation:** skip pi0.5 in v1 with clear error message.
3. **GR00T embodiment generalization** — test only embodiment_id=0 or all 32? **Recommendation:** only embodiment_id=0 in v1; document limitation.
4. **Stochastic output** — flow matching uses `torch.randn` init. **Recommendation:** pin `torch.manual_seed(seed)` before each validation; log seed in output.
5. **Threshold default** — export-time uses 0.01; goal specifies 1e-4. **Recommendation:** 1e-4 default with `--threshold` override.
6. **Memory footprint** — loading pi0/GR00T PyTorch ref (~7GB+) on small machines. **Recommendation:** single-export-at-a-time; `device="cpu"` for validation; FP16 option.
7. **OpenVLA in v1** — delegated export, bin-decoder post-process. **Recommendation:** skip OpenVLA in v1; add dedicated path in v2.
8. **Exit codes** — 0/1 for pass/fail; what about "not a valid export dir"? **Recommendation:** exit 2 for error/missing files.

## Risks

| Risk | Mitigation |
|------|-----------|
| Stochastic seed causes non-deterministic parity | Pin seed, log in output, document |
| PyTorch ref memory bloat (3B+ models) | CPU-only load; FP16 option; single-model scope |
| ONNX opset version drift | Pin `onnxruntime` in `[dev]` deps; validate early in CI |
| Fixture drift across PIL/torch versions | Fixed seed + deterministic transforms; embed preproc in fixture generator |
| Missing ONNX file in export_dir | Check at startup; exit 2 with clear error |
| GR00T embodiment_id mismatch in production | Document pinning; warn in output if config inconsistent |
| Threshold 1e-4 too strict for quantized exports | Support per-layer override; document |

## Implementation sequencing (feeds planning)

1. **Core class + CLI handler** — fill `validate()`, build `ValidateRoundTrip`
2. **Fixture generators** — seeded synthetic per model
3. **ONNX path** — session init, input prep, output decode
4. **PyTorch path** — load ref, run forward, extract matching output
5. **Comparison + aggregation** — reuse `validate_outputs()`, aggregate per-fixture
6. **Output formatting** — JSON + Rich table
7. **`--init-ci` flag** — emit workflow template
8. **Tests** — unit tests for fixture loading, comparison, thresholds
9. **Docs** — README + help strings

Each tier is a commit.
