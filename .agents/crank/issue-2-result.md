# Issue 2 — Per-model seeded fixture generators (DONE)

## Files created
- `src/reflex/fixtures/__init__.py` — re-exports `load_fixtures`.
- `src/reflex/fixtures/vla_fixtures.py` — `load_fixtures(model_type, num, seed=0)` implementation.

Total LOC: ~80 across both files (well under 150 cap).

## Behavior summary
- Uses `torch.Generator(device="cpu").manual_seed(seed)` (avoids mutating global RNG).
- Images: `torch.rand((H, W, 3), generator=g).numpy().astype(np.float32)` — float32 in `[0, 1)`, HWC.
- States: `torch.randn((state_dim,), generator=g).numpy().astype(np.float32)`.
- Prompts: cycled (with repetition for `num > 5`) from the curated 5-prompt list specified in the assignment.
- Per-model shapes: smolvla 512x512 / state 6, pi0 224x224 / state 14, gr00t 224x224 / state 64.
- Unsupported `pi05` / `openvla`: `ValueError("reflex validate v1 supports smolvla, pi0, gr00t. For pi0.5 / openvla see roadmap.")`.
- Unknown model_type: `ValueError(f"unknown model_type: {model_type}")`.
- Deterministic per `(model_type, num, seed)`: confirmed by `np.allclose` re-run check.

## Verification output
Ran the assignment's verification snippet plus extra coverage (pi0/gr00t shapes, prompt cycling for num=7, dtype/range, both unsupported model errors, unknown-type error):

```
fixtures-ok
```

Invocation: `PYTHONPATH=src .venv/bin/python -c "..."` from repo root.

## Deviations / notes
- The repo's editable `.pth` file (`_reflex_vla.pth`) points at `/Users/romirjain/Desktop/building projects/reflex-vla/src`. The path contains spaces, and CPython's site.py appears to skip it on this machine (the venv's `python` does not pick up `reflex` without `PYTHONPATH=src`). This is a pre-existing environment quirk unrelated to Issue 2 — verification was therefore run with explicit `PYTHONPATH=src`. Worth flagging for a follow-up env-fix issue (re-install with `uv pip install -e .` or quote the path), but not in scope here.
- Used `torch.Generator` rather than the cross-cutting "`torch.manual_seed(seed)`" wording so fixture generation does not leak global RNG state into the rest of the validate pipeline. The seeding intent (deterministic per seed input, CPU-only) is preserved exactly.
