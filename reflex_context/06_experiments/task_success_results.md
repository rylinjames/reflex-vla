# LIBERO-10 task-success on the Reflex pipeline (2026-04-19)

**Headline.** First measurable end-to-end LIBERO-10 run through Reflex's pipeline: **0/10 = 0.0% success rate** on the native PyTorch path. No crashes, no errors — all 10 tasks ran cleanly to their 300-step timeout. Same result as the April 2026 decomposed-path attempt; different root cause.

**Reproducer.** `modal run scripts/modal_libero_monolithic.py`. Takes ~25–30 min on Modal A10G after the image is cached (first build adds ~15 min for LIBERO/MuJoCo/robosuite install).

---

## What's a win, what isn't

**Phase 1 — infrastructure wins (first time ever):**
- The LIBERO-10 harness runs end-to-end through vla-eval against the reflex adapter. Prior attempts (April 2026) never got past smoke tests.
- All 10 tasks execute cleanly at 300 steps each (no crashes, no hangs, no error episodes).
- The vla-eval adapter boots with the reflex export directory, loads the normalizer, accepts WebSocket traffic.
- `SmolVLANativeServer` runs inference at a consistent rate (~2.2–2.5 min per 300-step episode).
- Every harness/plumbing/Modal gotcha encountered was captured as a reproducer-class fix:
  - Python 3.11 vs lerobot 0.5.1 incompatibility → use `--decomposed` flag (REFLEX_NATIVE=1 bypasses ONNX)
  - `libero` pip-installed but not importable → `PYTHONPATH=/opt/LIBERO` in image env
  - `python` vs `sys.executable` in smoke-test subprocess → use `sys.executable`

**Phase 1 — the honest task-success number:**
- **0.0% (0/10).** No task completed within 300 simulator steps.
- This is the FIRST honest, reproducible task-success number for Reflex on any benchmark.

---

## Per-task results

Every row: 1 episode, 300 max steps, FAIL (timeout), no error.

| # | Task | Result |
|---|---|---|
| 1 | put both the alphabet soup and the tomato sauce in the basket | FAIL (300 steps) |
| 2 | put both the cream cheese box and the butter in the basket | FAIL (300 steps) |
| 3 | turn on the stove and put the moka pot on it | FAIL (300 steps) |
| 4 | put the black bowl in the bottom drawer of the cabinet and close it | FAIL (300 steps) |
| 5 | put the white mug on the left plate and put the yellow and white mug on the right plate | FAIL (300 steps) |
| 6 | pick up the book and place it in the back compartment of the caddy | FAIL (300 steps) |
| 7 | put the white mug on the plate and put the chocolate pudding to the right of the plate | FAIL (300 steps) |
| 8 | put both the alphabet soup and the cream cheese box in the basket | FAIL (300 steps) |
| 9 | put both moka pots on the stove | FAIL (300 steps) |
| 10 | put the yellow and white mug in the microwave and close it | FAIL (300 steps) |

---

## Root-cause hypothesis — the adapter is running with `vlm=off`

Adapter startup log (captured):
```
ReflexVlaEvalAdapter ready: export=/tmp/reflex_libero_export device=cuda
                            out_dim=7 camera=<first> vlm=off norm=on
```

**`vlm=off` is the smoking gun.** Every LIBERO-10 task requires *language* conditioning to pick the right object — the model has to know "alphabet soup" vs "butter" vs "book" from the instruction text. The `lerobot/smolvla_libero` checkpoint was trained with full VLM language conditioning. The adapter, running in its current mode, feeds the model image + state + dummy language, so the model is effectively operating blind on the task intent.

This is consistent with: all tasks timing out cleanly (model IS generating actions, just not task-relevant ones). It's also consistent with the April 2026 decomposed-path result, which ran in the same no-VLM mode.

**This is NOT a model problem** — it's an adapter plumbing problem. The SmolVLA fine-tune presumably solves LIBERO-10 well above 0% when used with its full VLM pipeline.

---

## What this means for paid-pricing

**The cos=+1.000000 verified parity claim still holds** — ONNX matches PyTorch at machine precision for `sample_actions(num_steps=10)`. That's a correctness property of the export, not a promise about VLM wiring.

**But the "ships a working deployment" claim needs the asterisk.** Today, a customer following the README gets:
- An ONNX that matches PyTorch mathematically ✅
- A server that responds to POST /act with shape-valid actions ✅
- **NOT** an end-to-end pipeline that completes a LIBERO task ❌

**Honest buyer statement:** "Reflex today ships the correctness-of-export half of the deployment stack. The customer's fine-tune + VLM-conditioning path is their responsibility to wire — we provide the primitives but haven't yet shipped an end-to-end LIBERO-10-beating wrapper. Demo and pilot customers only; general availability waits on VLM-conditioning being baked into the adapter."

This is the paid-pricing unlock discussion we've been implicitly avoiding. Now we have a real number to anchor it.

---

## Next steps (Phase 2 work)

Three candidate fixes, roughly in order of investment:

1. **Turn `vlm=on` in the vla-eval adapter.** Probably a config-level thing in `src/reflex/runtime/adapters/vla_eval.py` — route images + language through the full VLM pipe before calling the expert. If this works, re-run and expect a non-zero number. **Rough effort: 0.5–1 day.**
2. **Extend adapter to route through monolithic ONNX (`SmolVLAOnnxServer`) with VLM conditioning wired.** This is the real "test the cos=1.0 path's task success" measurement. Depends on #1 working first. **Rough effort: 1–2 days.**
3. **Investigate preprocessing / camera-keying differences between SmolVLA LIBERO fine-tune's training and our adapter's runtime.** Dig into policy_preprocessor safetensors, per-camera resizing, state normalization. **Rough effort: 1–3 days, possibly more depending on what turns up.**

My pick: **#1 first.** Fastest signal. If VLM conditioning produces >0%, we have validated the hypothesis + a path to commercial numbers. If VLM conditioning still yields 0%, the adapter has a deeper bug and we investigate #3 before trying #2.

---

## Meta-finding: re-ran LIBERO for the first time

**It's been April since anyone at reflex-vla ran LIBERO-10 to completion.** The Apr-17 session captured a hunt for correctness bugs on the decomposed path but never actually completed LIBERO-10 (per `measured_numbers.md` Unverified section). Phase 1 shipped the **working reproducer** — that alone is a durable artifact. Any future "LIBERO-10 on reflex" measurement starts from `modal_libero_monolithic.py` + the PYTHONPATH + `sys.executable` + `--decomposed` fixes documented above.

---

## Compared to April's 0% (decomposed path)

The April run (see `05_sessions/2026-04-17_libero_correctness_hunt.md`) also got 0%, but the root cause was different:

| Dimension | April 2026 (decomposed path) | 2026-04-19 (this run, native path) |
|---|---|---|
| Model stack | Decomposed 5-file ONNX with 12 reimplementation bugs | Native PyTorch via `SmolVLANativeServer` — no reimpl |
| cos_sim vs reference | -0.24 per-step (catastrophic) | Not measured; inherits PyTorch correctness via native path |
| VLM conditioning in adapter | off | off |
| LIBERO-10 success | 0% | 0% |
| Root cause | Per-step velocity field corruption (reimpl bugs) | VLM adapter off — model can't resolve task description |
| Fix path | Rip the decomposed path (done; abandoned) | Turn vlm=on in adapter (Phase 2) |

The root causes are different — April's bugs are architecturally gone. The remaining gap is adapter-level, not model-level.

---

## Artifacts

- `scripts/modal_libero_monolithic.py` — the working reproducer
- `scripts/patch_libero.py` — LIBERO `input()` prompt patcher
- `/tmp/libero_run6.log` (local) — full run transcript
- Modal run URL: `https://modal.com/apps/hikaflow/main/ap-9Z7ekJa6gEGy7g4ZDKlLes` (run 4) + `https://modal.com/apps/hikaflow/main/ap-[bl33v3e01]` (run 6 final)

---

## Related

- `measured_numbers.md` — Verified section will now get a "LIBERO-10 success: 0/10 (adapter vlm=off)" row
- `05_sessions/2026-04-17_libero_correctness_hunt.md` — the original LIBERO hunt; context for the 0% decomposed result
- `06_experiments/customer_first_run_transcript.md` — the separate customer-dogfood exercise (also 2026-04-19)
- `02_bugs_fixed/modal_deployment_gotchas.md` — will capture the `python` vs `sys.executable`, PYTHONPATH, and Python 3.11 × lerobot 0.5.1 gotchas
- `GOALS.yaml` — `task-success-benchmark` goal; this file is its `check` artifact
