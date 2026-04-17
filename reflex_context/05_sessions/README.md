# Reflex VLA — Session Chronology

This folder is the chronological narrative of the Reflex VLA project's work sessions. Each file tells the story of one day: what the goal was, what got discovered, which dead-ends ate time, what pivoted, and what got carried forward.

## How to read this folder

1. **Start here (the README).** It indexes every session file.
2. **Pick the date relevant to your current debugging.** Each file opens with a `Goal` and closes with a `Carry-over` section that points to later sessions where unfinished work continued.
3. **Follow the carry-over chain.** The Apr-10 mega session set the direction; every subsequent day iterated on a specific unfilled gap.
4. **If you're looking for code-level findings, cross-reference:**
   - Bug catalogs → `02_bugs_fixed/` (if present).
   - Per-model architecture notes → `01_architecture/` (if present).
   - Benchmark tables and strategy decisions → `01_decisions/` ADRs.

## Session files

| File | Synopsis |
|------|----------|
| [2026-04-10_mega_session.md](2026-04-10_mega_session.md) | Naming (Reflex), 7-path strategy, no-VLA-exporter gap discovery, initial SmolVLA bring-up, TRT FP16 preview, 7-wedge CLI definition, pricing ladder, OSS/GTM motion. The session that defined everything. |
| [2026-04-13_phase_iii_batching_adaptive.md](2026-04-13_phase_iii_batching_adaptive.md) | Phase III continuous batching lands (2.88× at batch=16 on real pi0). Phase IV adaptive denoising validated on real VLAs — only pi0 is a real win; smolvla/pi0.5/gr00t degrade. Real-model vs synthetic-toy distinction formalized. |
| [2026-04-14_benchmark_postmortem.md](2026-04-14_benchmark_postmortem.md) | Post-mortem: "GPU" benchmarks were silently CPU. Root cause: CUDA 12 vs 13 library mismatch. `strict_providers=True` introduced. Install-path verification. TRT FP16 flip — dominates torch.compile 2.6–3.3× once pins are correct. |
| [2026-04-16_libero_infra_hunt.md](2026-04-16_libero_infra_hunt.md) | LIBERO-10 infrastructure death march. 18 commits in 75 min fighting bddl, gym, robosuite, osmesa, MuJoCo, lerobot install quirks. Result: LIBERO runs end-to-end but 0% task success. |
| [2026-04-17_libero_correctness_hunt.md](2026-04-17_libero_correctness_hunt.md) | Today. Per-bug chronology: 5D pixels → VLM dummy → state_proj random → AutoModel vs AutoModelForImageTextToText → √hidden → rope_theta → prefix_offset → multi-camera → state dim → newline → sinusoidal → obs.get. Stage-diff methodology debut. Local-iteration pivot. ~12 bugs fixed. Single-layer self-attn matches to 1e-5. LIBERO task success still 0%. |

## Conventions

- **Dates** are in ISO format and reflect the day work happened, not the day files were written.
- **Commit SHAs** cited are full 7-char prefixes from the git history (e.g., `ced2c4f1`, `47f3d5d`).
- **Transcript line numbers** reference the raw session JSONL files (see `reflex_context/_raw/`).
- **Numeric results** (cos_sim, max_diff, ms/chunk, Hz) are quoted verbatim from the session they were measured in — not aggregated or re-interpreted.
- **Carry-over** sections explicitly name the later session where unfinished work continued. Follow the trail.

## The arc in one sentence

*Apr-10 picked the product (Reflex: VLA deployment CLI); Apr-13 validated the speed wedges on real models; Apr-14 rescued the speed narrative from a silent CPU-fallback that would have killed the pitch; Apr-16 proved the LIBERO infrastructure could boot; Apr-17 chased 12 correctness bugs through the ONNX pipeline without yet cracking the final 2% per-step velocity error that keeps task-success at 0%.*

---

*Last updated: 2026-04-17.*
