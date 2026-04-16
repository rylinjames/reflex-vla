# RPI Phase 1 Summary — reflex validate stub research

**Date:** 2026-04-16
**Full research:** `.agents/research/2026-04-16-reflex-validate-stub.md`

## Takeaways (500-token condensation)

`reflex validate` is a stub in `src/reflex/cli.py:165-176` — only parses args and prints a header. A parallel module `src/reflex/validate.py` already ships real utilities: `ValidationResult`, `validate_outputs()`, `validate_decomposition()`. Each exporter runs single-step ONNX parity at export time with threshold 0.01; none run a post-export round-trip on real (image, prompt, state) tuples.

**Key reusable primitives already in repo:**
- `reflex.validate.validate_outputs()` — handles torch↔numpy, computes max_abs / mean_abs / max_rel, returns structured result
- `reflex.checkpoint.load_checkpoint()` — handles local safetensors, directories, HF Hub IDs
- `reflex.checkpoint.detect_model_type()` — returns model family from state_dict
- Per-exporter ONNX validation patterns in `smolvla_exporter.py:299-316`, `pi0_exporter.py:246-263`, `gr00t_exporter.py:705-720`

**Per-model status:**
- SmolVLA (6 dims, 10 steps, chunk 50) — clean path, max_diff <1e-5
- pi0 (32 dims) — clean path; exporter rejects pi0.5 explicitly
- pi0.5 (32 dims, AdaRMSNorm) — separate exporter path not yet merged; recommend skip in v1
- GR00T N1.6 (128 dims, 4 steps, AdaLN 2-chunk) — pinned `embodiment_id=0` in export; validate same
- OpenVLA — delegated export via optimum-cli; recommend skip in v1

**Recommended architecture:**
- New `src/reflex/validate_roundtrip.py` with `ValidateRoundTrip` class
- New `src/reflex/fixtures/vla_fixtures.py` for seeded synthetic (image, prompt, state) tuples
- Expand `cli.py` validate() handler + add `--init-ci` flag
- New `tests/test_validate_roundtrip.py`
- `.github/workflows/reflex-validate.yml.template` emitted by `--init-ci`

**Threshold:** default 1e-4 (goal spec) with `--threshold` override; export-time uses 0.01.

**Exit codes:** 0 pass, 1 fail, 2 error (missing ONNX, invalid export dir).

**Key open items for planning:**
- v1 scope: drop pi0.5 and OpenVLA, focus on SmolVLA / pi0 / GR00T
- Stochastic seed: pin `torch.manual_seed(seed)`, log in output
- Memory: CPU-only ref loading; single-export scope
- Fixtures: synthetic seeded tensors in v1; optional known-good images later

## Ready for /plan

All architecture decisions deferrable to plan phase. Research provides file paths, function signatures, and risk register. Proceed.
