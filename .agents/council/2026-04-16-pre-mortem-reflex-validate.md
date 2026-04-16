# Pre-Mortem: reflex validate round-trip harness

**Date:** 2026-04-16
**Plan:** `.agents/plans/2026-04-16-reflex-validate-roundtrip.md`

## Council Verdict: WARN

Proceeding to crank with inline amendments (applied below).

| Judge | Verdict | Key finding |
|-------|---------|-------------|
| Missing-Requirements | WARN | Silent threshold default change (0.02 → 1e-4) breaks scripted users; risk register items orphaned from acceptance criteria |
| Feasibility | WARN | Exporter helpers yield ONNX-export *surrogates* not upstream PyTorch ref — "round-trip" risks comparing exporter to itself. GR00T full-stack needs SigLIP2+Qwen3+action_encoder/decoder outside exporter. pi0/GR00T PyTorch FP32 OOMs on 7GB GitHub runners. |
| Scope | WARN | 7-issue plan over-fragmented for a stub; per-model fixture preprocessing smuggles in complexity; --init-ci near-zero recurring use |
| Spec-Completeness | PASS | Boundaries + conformance checks solid; but Issues 4 & 5 both edit `validate_roundtrip.py` — Wave 2 file conflict, not parallelism |

## Shared findings

- **Reference semantics** — the plan doesn't commit to whether "PyTorch reference" means the exporter's decomposed surrogate (what's available now) or true upstream model (much more work). Must pick.
- **CI reality** — GitHub-hosted runners won't load 3B+ FP32 models. SmolVLA only in CI; pi0/GR00T on self-hosted or local.
- **Seed bridging** — `torch.manual_seed()` does not seed numpy. Cross-runtime reproducibility needs a single noise tensor generated once and passed to both paths.
- **File conflict** — Wave 2 Issues 4 and 5 both extend the same file; need to split into separate modules or serialize.

## Applied amendments (before crank)

1. **Split Issue 4 / Issue 5 into separate modules:** `src/reflex/_pytorch_backend.py` and `src/reflex/_onnx_backend.py`. Removes Wave 2 file conflict.
2. **Reference-vs-surrogate decision (Boundary: Always):** v1 validates parity between the exporter's decomposed PyTorch model (surrogate) and its ONNX export. This proves export correctness — the primary goal of `validate`. Upstream-vs-exporter parity is a separate v2 problem.
3. **GR00T scope cut:** v1 validates only the DiT expert (pinned `embodiment_id=0`), mirroring what `gr00t_exporter.py` already emits. Full-stack (SigLIP2 + Qwen3 + action_encoder/decoder) is v2.
4. **CI matrix constraint (Issue 3):** default template runs SmolVLA only on `ubuntu-latest`. pi0 and GR00T steps are commented with a `# Requires self-hosted runner with 16GB+ RAM` header.
5. **Seed bridge (Issues 4 + 5 acceptance):** generate noise ONCE in torch, convert to numpy, pass identical array to both forward paths. Issue 7 tests this equivalence.
6. **Backward-compat deprecation (Issue 6):** the v0.1 stub shipped `--threshold 0.02` default. New default is `1e-4`. CLI help string documents the change; CHANGELOG entry added. No `--strict` flag needed — the stub never actually validated anything, so no real user scripts broke.
7. **Happy-path example (Issue 6 acceptance):** README gains a 5-line `reflex validate ./pi0 --model lerobot/pi0_base` example. Helpmsg includes it too.
8. **Error message spec (Issue 2):** pi0.5/OpenVLA `ValueError` message is `"reflex validate v1 supports smolvla, pi0, gr00t. For pi0.5 / openvla see roadmap."`

## Not applied (accepted risk or too conservative)

- **Cut Issue 3 (--init-ci)** — keeping. User explicitly requested; implementation is <50 LOC.
- **Cut per-model fixture preprocessing** — keeping. Model-agnostic fixtures would hide per-model shape bugs (the exact class of silent bugs the tool is supposed to catch).
- **Collapse to 4 issues** — keeping 7-issue structure. Fragmentation cost is acceptable vs clarity for the cranker.

## Recommendation

PROCEED with amendments applied. Feasibility WARN on GR00T full-stack is the load-bearing concern; amendment 3 neutralizes it by scoping to DiT-expert-only for v1.
