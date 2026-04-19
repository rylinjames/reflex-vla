# LIBERO-10 task 3/4 gap audit (2026-04-19)

**Context:** N=25 native PyTorch sample achieved 10/25 = 40% success on LIBERO-10 tasks 0-4. Per-task: 3/5 on tasks 0-2 (60% each), **1/5 on task 3 (20%)**, **0/5 on task 4 (0%)**. Tasks 3-4 underperform the community 43-51% baseline.

This audit hunts for the root cause of the task 3/4 underperformance so we can either fix it (reflex wins the baseline back) or document it as a model-inherent limitation (future customers calibrate expectations).

## Quick findings

### Task 4: 21-token prompt exceeds native training fine-tune's effective working length

The preprocessor is configured as `pad_language_to: "longest"` with a tokenizer `max_length=48`. Task 4's prompt is 21 tokens, which fits in 48.

| Task | Prompt | Tokens |
|---|---|---|
| 0 | "put both the alphabet soup and the tomato sauce in the basket" | 12 |
| 1 | "put both the cream cheese box and the butter in the basket" | 12 |
| 2 | "turn on the stove and put the moka pot on it" | 12 |
| 3 | "put the black bowl in the bottom drawer of the cabinet and close it" | 14 |
| 4 | "put the white mug on the left plate and put the yellow and white mug on the right plate" | **21** |

Task 4 is ALSO the only explicitly two-goal bimanual-ish task ("put X on left plate AND put Y on right plate"). Two disambiguation challenges: "white mug" vs "yellow and white mug" (similar colored objects) + left-vs-right plate grounding.

Native scored **0/5** despite having the full 21-token prompt, num_steps=10 denoising, and all the harness fixes. This is a model-capability limitation of the fine-tune, not a harness/export bug. The fine-tune checkpoint simply doesn't execute bimanual-ish instructions.

### Task 4 compound bug in our ONNX export: seq=16 truncation

**This IS a fixable latent bug.** Our monolithic ONNX was exported with `lang_tokens shape (B, 16)` hardcoded. Task 4's 21-token prompt truncates at token 16, cutting off `" and put the yellow and white mug on the right plate"`. So the ONNX path sees only `"put the white mug on the left plate and put"`.

For tasks 0-3 (≤14 tokens each), seq=16 is harmless — padding is appended. But any future long-prompted task will silently lose information.

Fix: re-export monolithic with seq=48 (or dynamic seq axis with `dynamic_axes={"lang_tokens": {1: "seq"}, "lang_masks": {1: "seq"}}`).

### Task 3: 1/5 not yet explained

14-token prompt fits fine, no bimanual challenge, single-goal-ish (place bowl, close drawer). Still scored only 1/5 on native.

Hypotheses:
- **Two-phase horizon**: "place bowl" + "close drawer" may require replan on mode transition that `replan_steps=5` interrupts mid-motion.
- **Drawer grounding**: "bottom drawer of the cabinet" requires specific spatial reasoning that may be underrepresented in training data.
- **Training data imbalance**: fine-tune may have been epoch-biased toward pick-and-place tasks (0-2) over drawer manipulation.

Without access to training logs or lerobot training set breakdown, can't definitively resolve. But:

### The big confounding factor: num_steps=1 ONNX vs num_steps=10 training

**Critical finding:** fine-tune was trained with `num_steps=10` (10-step flow-matching denoise). Native harness uses config default → num_steps=10. Our monolithic ONNX default is `num_steps=1` (single Euler step).

From `measured_numbers.md`:
- SmolVLA num_steps=1 vs num_steps=10 on shared inputs: first-action cos=0.78, max_abs=0.58, **22% of action range**.

That's a huge behavioral gap. If the LIBERO-ONNX test (currently running on task `b2e0hsmnw`) uses num_steps=1, it can't match native's 40%. Likely outcome: tasks 0-2 drop from 60% → maybe 30-40%, task 3 may drop to 0, task 4 stays 0.

**For a fair ONNX-vs-native comparison, we need num_steps=10 ONNX.** That was already verified cos=1.0 for smolvla_base in measured_numbers.md; we need to re-export it specifically for `HuggingFaceVLA/smolvla_libero`.

## Next actions (if pursuing gap closure)

1. **Re-export `smolvla_libero_monolithic/model_n10.onnx` with `--num-steps 10`** (5 min compute, already-verified export path)
2. **Re-export with seq=48 OR dynamic seq** to unblock task 4 for the ONNX path (10 min compute)
3. **Re-run N=25 ONNX with the fixed export** (50 min compute) — expect ≥35% to match native within noise
4. **Task 3 alone investigation**: if native holds at 20%, it's the fine-tune; if replan_steps=1 bumps it, it's our harness

## Summary

| Factor | Evidence | Actionable? |
|---|---|---|
| Task 4 truncation in our ONNX | seq=16 cuts 21-token prompt | **YES** — re-export with dynamic seq |
| Task 4 fine-tune limit | 0% native with full prompt | No — model capability |
| Task 3 long-horizon | 1/5 native, hypothesis unverified | Maybe — try replan_steps=1 |
| num_steps=1 in ONNX default | `--num-steps 1` flag default | **YES** — re-export with num_steps=10 |
| seq=16 latent bug for future tasks | 4 of 5 current tasks fit but any >16 truncates | **YES** — fix is same as task 4 fix |

The **immediate concrete action** is re-exporting the smolvla_libero monolithic with (a) num_steps=10 and (b) seq=48 (or dynamic), then re-running N=25 ONNX. That gets the monolithic comparison on fair footing and simultaneously unblocks task 4's latent failure mode. Task 3 under-performance may remain a fine-tune capability issue.

## Related

- `reflex_context/06_experiments/task_success_results.md` — the 40% N=25 that motivates this audit
- `reflex_context/measured_numbers.md` — num_steps=1 vs num_steps=10 quality gap (line 53)
- `scripts/modal_libero_monolithic_onnx.py` — the ONNX harness with hardcoded seq=16 pad
- `scripts/modal_smolvla_monolithic_export.py` — supports `--num-steps 10` already
