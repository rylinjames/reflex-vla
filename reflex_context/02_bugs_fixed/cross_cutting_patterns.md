# Cross-Cutting Patterns — What the Bugs Have in Common

Patterns that emerge when you stack all the bugs in the other files side-by-side. These are the "this is HOW bugs happen in ML infrastructure" observations — anti-patterns, debug methodologies, and cost-reality checks worth internalizing before the next deep debugging march.

Sources: all six raw files, with emphasis on the "verbalized-only" and meta-analysis threads in `current_session.md`.

---

## Pattern 1: "Silent fallback" as the dominant anti-pattern

**Pattern:** Layers of infrastructure silently degrade when their prerequisites aren't met. No error, no warning — just a fallback that looks similar but produces materially different results.

**Examples from the bug catalog:**

1. **CUDA → CPU silent fallback.** `onnxruntime-gpu` falls back to CPU when CUDA EP fails to load (`libcublasLt.so.12: cannot open shared object file`). Every "A100 benchmark" before the post-mortem was actually CPU-on-A100-hardware. Led to a real misunderstanding of competitive positioning — thought torch.compile crushed us 6-14×, actually TRT FP16 wins 2.6-3.3× when CUDA EP loads. `sessions_md.md` line 14, `current_session.md` line 4372.

2. **5D pixel_values → dummy conditioning.** `vision_encoder.onnx` expected 4D `pixel_values`, got 5D from SmolVLM's AutoProcessor, shape check failed, orchestrator caught the exception and returned DUMMY ZEROS. Every LIBERO episode for months ran without VLM conditioning. Only visible via careful stage-diff inspection. `current_session.md` line 9605.

3. **Text embedder → seeded random.** When `text_embedder.onnx` didn't exist, fallback produced `np.random.randn()` — fresh random numbers per call. Non-deterministic. Only caught when determinism test showed max_diff=34.7 on identical inputs. `current_session.md` line 7534, 7541.

4. **State proj → random weights.** `state_proj_weight.npy` missing? Fallback: random N(0, 0.02). State info → meaningless garbage. The smoking gun. `current_session.md` line 10679.

5. **VLM weights → partially loaded.** `AutoModel` vs `AutoModelForImageTextToText` prefix mismatch left 488 missing / 345 unexpected keys. Vision loaded half-random weights; pipeline kept running. Only caught via `[vlm-weights] load:` log line tracking. `current_session.md` line 11063.

**Remediation: `strict_providers=True` as defense pattern.**

Commit `5b21296` codifies the counter-design: if the requested execution path isn't available, EXIT with clear error; don't silently degrade.

```python
# src/reflex/runtime/server.py after strict-providers fix:
if strict_providers and device == "cuda":
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "CUDA requested but CUDAExecutionProvider not available. "
            "Possible fixes:\n"
            "  - pip install onnxruntime-gpu (not onnxruntime)\n"
            "  - docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3\n"
            "  - Pass --no-strict-providers to silently fall back"
        )
```

**Similar pattern elsewhere:**
- `reflex serve` refuses to load if ONNX files are missing (not silently ignore).
- `reflex doctor` diagnoses install issues ahead of `reflex serve` blowing up.
- VLM orchestrator should HARD-FAIL when state_proj_weight.npy missing, not fall back to random.

**Learning:** Wherever a fallback exists, ask: "Will the user notice the degraded behavior?" If the answer is "not until task success hits 0%," the fallback is a bug. Prefer loud errors with installation hints over silent degradation.

**Sources:**
- `sessions_md.md` lines 14, 45 (strict provider decision)
- `git_history.md` theme "Apr-14 GPU benchmark post-mortem"
- `current_session.md` multiple examples above

---

## Pattern 2: Copy-paste-test (vs decompose-from-scratch)

**Pattern:** 8 of 12 SmolVLA pipeline bugs disappear if you use lerobot's actual code instead of reimplementing the operations from scratch. The bugs aren't in the underlying algorithms — they're in subtle reimplementation details.

**Cases:**
- Sinusoidal timestep: missing 2π, wrong [cos, sin] order.
- RoPE base 10000 vs 100000 (10× error).
- prefix_offset for self-attention position_ids.
- KV mask for cross-attention.
- Missing √hidden scaling.
- 5D pixel_values handling.
- Missing newline on task string.
- State dim 8 vs 6.

Each is a faithful port error; lerobot's code does each one correctly.

**Remediation strategy (from `current_session.md` line 11524 hybrid option 2):**

> "Copy lerobot's `SmolVLAPolicy.sample_actions` + `embed_prefix` + `embed_suffix` + `forward_cross_attn_layer` into `reflex/runtime/smolvla_native.py`. Swap only `RMSNorm → DecomposedRMSNorm` for TRT compat. Let torch.onnx.export handle the rest. Hours of work, correct by construction, Jetson compatible."

**Key insight:** The value of "decomposition" is targeted at TRT / ONNX op coverage — specifically, ops that don't export cleanly (RMSNorm in opset 23, RoPE in torch.onnx pre-dynamo). For EVERY OTHER operation — attention, GQA, RoPE, time embedding — modern `torch.onnx.export(dynamo=True)` handles them correctly. Reimplementing them from scratch just invites faithful-port errors.

**Prior evidence:**
- Spike commit `6fedff3`: "SmolLM2's GQA decoder (LlamaDecoderLayer) exports to ONNX cleanly first try — no patches, no custom ops, no workarounds."
- Vision: `VisionEncoderForONNX` wraps the real SigLIP + connector, just pre-computes position IDs to avoid dynamic index_put.

**Learning:** For research-calibre models, the reference implementation is usually correct. Your job is NOT to re-derive the math from paper — it's to make the reference implementation ONNX-compatible by swapping only the ops that don't export. Use the reference implementation everywhere else.

**Sources:**
- `current_session.md` line 11524, 11574 (the hybrid proposal)
- `git_history.md` commit `6fedff3` (spike showing GQA+RoPE export cleanly)
- `modal_scripts.md` `modal_stage_diff.py` (reimplements per-layer k/v in PyTorch to compare — precisely because ONNX export diverges from that reference)

---

## Pattern 3: Modal cold-image build cost vs local iteration — ~100× factor

**Pattern:** Modal iteration (cloud) is 100× more expensive than local iteration in both time and dollar terms. Every Modal run: ~5-10 min image build (first time), ~$0.50-3 of compute, tethered to an A10G/A100 resource.

**Numerical reality:**
- Modal A100 spot: ~$1.10/hr.
- A10G: ~$0.40/hr cheaper proxy for Ampere.
- LIBERO-10 run: 60-120 min → $1-3 per iteration.
- Benchmark: 10-30 min → $0.20-1.
- Image build: 5-10 min for ML deps (lerobot, torch, transformers, onnx, fastapi).
- TRT container base: 10GB pull (slow).

During the LIBERO hunt, 18 commits in 75 min triggered ~15 Modal runs. ~$15-30 Modal cost per night of iteration. This is fine for validation but BRUTAL for debugging — if you're chasing a bug that requires per-layer diagnostic changes, you're paying cloud for something you could do locally.

**When to use Modal:**
- ✓ Validate ONNX export numerically (~$0.50 per model).
- ✓ Benchmark latency (requires real GPU).
- ✓ Integration tests (reflex serve + curl /act).
- ✓ LIBERO-10 (requires Modal; sim needs a real env setup).

**When to use local:**
- ✓ Per-stage numerical diffs (can use small inputs, CPU).
- ✓ Weight-load diagnostics.
- ✓ Reading/debugging code.
- ✓ Iterating on shape math / reimplementation ports.

**Learning (from `current_session.md` line 11574):** "Local iteration is ~100× cheaper than Modal." Run `scripts/local_stage_diff.py` / `scripts/local_single_layer_diff.py` before reaching for `scripts/modal_stage_diff.py`. Only jump to Modal after you've ruled out the cheap bugs locally.

**Sources:**
- `current_session.md` line 11574 ("Local iteration is ~100× cheaper than Modal")
- `current_session.md` line 5453, 5218 (specific spend numbers)
- `sessions_md.md` line 97 (image build timing)
- Scripts list in `sessions_md.md` section: `scripts/local_stage_diff.py`, `local_full_diff.py`, `local_expert_diff.py`, `local_single_layer_diff.py` — the local-first tools

---

## Pattern 4: The "diagnostic ladder" — stage diff → single layer → composition

**Pattern:** When end-to-end task success is 0%, you cannot debug by staring at final actions. You need a hierarchical bisection strategy: diff at a series of granularity levels, isolating where divergence first appears.

**The ladder (from `current_session.md` lines 10799, 11458):**

**Rung 1 — Stage diff (`scripts/modal_stage_diff.py`):**
- Diff vision encoder output (after SigLIP + connector).
- Diff text embedder output (after SmolLM2 embed_tokens).
- Diff state projection output (after state_proj linear).
- Diff per-layer decoder prefill k/v (after each transformer layer, RoPE applied).
- Diff final action chunk.

Result: vision `cos=1.0000`, text `cos=1.0000`, state `cos=1.0000`, per-layer k/v `cos=0.91-1.00`, FINAL `cos=0.08`. Divergence is in the expert stack composition, NOT in any isolated component.

**Rung 2 — Single layer diff (`scripts/local_single_layer_diff.py`):**
- Take ONE layer of the real expert (e.g. `policy.model.vlm_with_expert.lm_expert.layers[0]`).
- Run forward with known inputs (noise + k/v + position_ids).
- Run OUR `ExpertGQALayer` with the SAME weights and inputs.
- Compare outputs.

Result: single SELF-attn layer (layer 0) matches to `1e-5 precision, cos=1.0000`. The bug is somewhere in COMPOSITION — probably cross-attention layers or between-layer masking.

**Rung 3 — Composition tests:**
- Test single cross-attn layer (open at session close).
- Test full 2-layer composition (self-attn → cross-attn).
- Progressively stack layers until divergence appears.

**Key principle (current_session.md line 11458):** "The first stage where L2 diverges is the bug — we fix that specific thing."

**Key methodology refinement (current_session.md line 10846):** "Flow-matching uses fresh random noise each call — BOTH paths get different noise, so we're comparing noisy vs noisy. Need to inject the SAME noise into both paths." Without shared noise, cos_sim is meaningless. `scripts/modal_pytorch_vs_onnx.py` explicitly seeds with `np.random.RandomState(99)` for this reason.

**Learning:** Don't guess. Diff. Each attempted fix without bisection is a flyby shot — maybe it helps, maybe doesn't, you can't tell because the metric doesn't move smoothly. A proper diagnostic ladder makes the bug observable at its actual layer, not hidden behind 500 downstream operations.

**The cos_sim progression as empirical validation:**

| Rung | Method | Signal quality |
|---|---|---|
| End-to-end action diff | Single cos_sim number | Low — know something's wrong, don't know where |
| Stage diff (vision/text/state/layer KV/final) | 5 cos_sim numbers | Medium — localize to one stage |
| Single-layer diff (layer 0 self-attn) | 1 per-layer cos_sim | High — localize to one class of operation |
| Single-op diff (RMSNorm, softmax, etc.) | 1 per-op cos_sim | Highest — pinpoint the exact operation |

Descend only as far as you need to. But descend aggressively when end-to-end diff gives you 0.08 or 0.3.

**Sources:**
- `current_session.md` lines 10799, 11458, 11468 (ladder methodology)
- `scripts/modal_stage_diff.py`, `scripts/local_single_layer_diff.py`
- `current_session.md` line 10846 (shared-noise methodology)
- `modal_scripts.md` `modal_stage_diff.py` + `modal_pytorch_vs_onnx.py` sections

---

## Pattern 5: "Task-success is an integration test" — the 500-op pipeline problem

**Pattern:** When a pipeline has ~500 operations and you need per-step cos > 0.999 to survive 10 integration steps without compounding error, ONE subtly wrong operation = 0% task success.

**Numerical budget (current_session.md line 11435):**
> "Given cos=-0.24 final, task success will still be 0%. We need per-step cos >0.999 to survive 10 integration steps."

**Concrete math:** If per-step cos = 0.977 (20% error in velocity), then after 10 Euler integration steps, compounded drift reverses sign (cos ≈ -0.24). If per-step cos = 0.999, after 10 steps residual cos ≈ 0.99. The budget is tight.

**Implication for debugging:** You cannot tolerate "mostly right." Every single operation has to be numerically perfect (< 1e-5 max_diff). There is no "we're 90% done, ship it" phase — the 10% wrong is the difference between 0% success and 80% success.

**From the 12 pipeline bugs, the failure compounding pattern:**
- Bug 1 (state_proj random) alone = 0% (garbage state).
- Bug 4 (5D pixel_values) alone = 0% (no VLM conditioning).
- Bug 11 (position_ids offset) alone = per-layer 0.99 → compounded catastrophic.
- Bug 20 (composition mystery) alone = cos 0.977 per step → compounded cos=-0.24 final.

Each bug is catastrophic in isolation. Fixing 11 of 12 still gives you 0%.

**"Each fix was a real bug" confession (current_session.md line 10630):**
> "Honest answer: I'm guessing at which subtle thing is wrong. Each fix I've made (5D pixel_values, normalizer, per-layer vlm_kv, layernorm on k, RoPE on keys, split k/v, newline on task, multi-camera, controller_states) was a real bug that would have made the model fail. Fixing them all and still getting 0% means there's ONE more thing — but I have 30+ candidates and no way to rank them without direct comparison. The fundamental problem: task-success is an integration test. ONE subtly wrong operation out of ~500 in the pipeline = 0%. Without a side-by-side diff against the real PyTorch model, I'm iterating blind."

**Learning:** For ML integration, the "you have to fix it all" budget is real. Don't stop at "90% of bugs fixed." Task-success of 0% doesn't imply "most bugs remain" — it may imply "one last bug remains." The diagnostic ladder (Pattern 4) is the only way to know which.

**Sources:**
- `current_session.md` line 10630, 11435, 11574

---

## Pattern 6: Burst-of-commits = signal of a problem being chased

**Pattern:** Looking at `git log`, bursts of many small commits in a short window signal a debugging march — the team is iterating fast on a specific problem.

**Bursts observed in reflex-vla (from `git_history.md` cross-theme section):**
- **Apr-14 10:04-11:08 (15 min burst): 9 commits** — pi0 support + pi0.5 auto-dispatch + GR00T stub + OpenVLA helper. "Fill out the supported-VLA table" push.
- **Apr-14 12:09 → 13:13 (~1 hour, 2 big commits)** — GPU post-mortem response: turbo cuda_graph + strict providers. Bug discovery → fix.
- **Apr-14 15:34-16:15 (41 min, 4 commits)** — real-model batching lands. Progressive fix.
- **Apr-16 13:34-14:11 (37 min, 9 commits)** — trajectory replay image-format fighting. LeRobot v2 dataset format wrangling.
- **Apr-16 14:50-16:05 (75 min, 18 commits)** — LIBERO integration death march. 18 commits fixing bddl, gym, osmesa, input() patching, robosuite pin, pip install quirks.

**Reading the burst pattern:** Any commit log block with >5 commits in <1 hour signals a battle. For future archaeology:
- Block between SHAs is usually a discrete bug fight.
- Start reading from the FINAL commit — it's likely the working fix.
- Earlier commits in the block are intermediate failure modes (useful context).
- Sometimes the real fix is in a commit message; the actual code pattern is already in the block.

**Learning for process:** If you find yourself in a burst (3+ rapid commits on the same problem), STOP and do a step-back root cause analysis. You're iterating blind. The 18-commit LIBERO burst would have been shorter if a single `patch_libero.py` script had been designed upfront instead of evolved through 11 iterations of `sed` → `python regex` → separate file.

**Sources:**
- `git_history.md` "Cross-theme patterns worth knowing" section

---

## Pattern 7: Infrastructure-vs-task-success tension — the "ship and keep debugging" dilemma

**Pattern:** Infrastructure components (ONNX export, CLI, serve, validate) all pass their own tests. Task-success still 0%. Every visible metric looks correct but the downstream metric (LIBERO success) is rock bottom.

**Status at session close (current_session.md line 10268):**

| Component | Status |
|---|---|
| Unified CLI (reflex export auto-produces 4 ONNX files for SmolVLA) | ✅ |
| vla-eval adapter (ReflexVlaEvalAdapter) | ✅ |
| Normalizer pipeline (state + action) | ✅ |
| Per-layer vlm_kv ONNX export | ✅ (partial) |
| 5D→4D pixel_values fix | ✅ |
| LIBERO sim on Modal (bddl, gym, osmesa, robosuite pin) | ✅ |
| **LIBERO task success: 0%** | ⚠ unresolved |

**The dilemma:** You've built a real piece of infrastructure (4-file ONNX export, vla-eval adapter, Modal image). You've fixed 12 pipeline bugs. But the number that matters (LIBERO task-success) is still 0%. Do you:
- (a) Ship the infrastructure with "v0.1 uses random VLM conditioning" disclaimer and keep debugging?
- (b) Hold the release until task success > 0%?

The project chose (a): the infrastructure is real and valuable even without task success. Launch drafts (`launch/lerobot_3146_draft.md`, `show_hn_draft.md`, `reddit_robotics_draft.md`) all carry the caveat: "Action values look random / nonsensical ... Expected in v0.1. The current ONNX export covers the action-expert denoising loop with random VLM conditioning. Real per-image conditioning lands when the VLM prefix encoder is wired (Phase II.4 / v0.2)."

**Learning:** It is possible — and sometimes correct — to ship infrastructure before the integration test passes. The infrastructure is what composes into correctness once the final bug is found. But be explicit in the README about what doesn't work. Don't let marketing copy obscure the gap.

**Sources:**
- `current_session.md` line 10268 (status table)
- `modal_apps_and_pm_docs.md` launch drafts section
- README snippet: "Action values look random / nonsensical … Expected in v0.1."
- `current_session.md` line 6497 (council reprioritization: "VLM prefix encoder is #1, not #2... v0.1 benchmarks a noise generator. Every speed number... is theater until real VLM conditioning works.")

---

## Pattern 8: The "wrap, don't rebuild" principle

**Pattern:** When an external tool already solves a problem, wrapping it saves weeks vs. rebuilding. Demonstrated repeatedly:

**Cases:**
- **vla-eval**: AllenAI's LIBERO benchmark harness uses WebSocket+msgpack. We built a thin `VlaEvalAdapter` (20 LOC) that bridges to `ReflexServer` rather than reimplementing the LIBERO sim. ADR: `2026-04-14-wrap-not-rebuild-vla-eval.md`.
- **PaliGemma2 ONNX**: `onnx-community/paligemma2-3b-pt-224` already has the 3-file split (vision_encoder + embed_tokens + decoder_model_merged). Use as template for pi0's VLM export. `current_session.md` line 5756.
- **ETARS SmolVLA ONNX notebook**: `aifoundry-org/ETARS` had `smolVLA_libero_export.ipynb` — "Someone already solved this exact problem. Study their code, adapt to our exporter, validate. Fastest path." `current_session.md` line 7195.
- **OpenVLA via optimum-onnx**: instead of building a custom exporter, route to HF's `optimum-onnx` and ship only the bin-decode postprocess helper. `current_session.md` line 4009.
- **lerobot's modeling files (the hybrid pivot)**: instead of reimplementing SmolVLA's expert + preprocessing + cross-attn from scratch, copy lerobot's code into `reflex/runtime/smolvla_native.py` and swap only RMSNorm for TRT compat. `current_session.md` line 11574.

**Learning:** Before writing 500+ LOC of custom reimplementation, spend 30 min googling for an existing solution. "ETARS has this notebook" saved weeks. "Wrap vla-eval" saved rebuilding LIBERO's sim harness.

**Sources:**
- `sessions_md.md` line 42 (vla-eval wrap decision ADR)
- `current_session.md` lines 5756, 7195, 11524, 11574
- `git_history.md` commit `c00ca82` (OpenVLA → optimum-onnx + helper)

---

## Pattern 9: Open questions that remain at session end

**Pattern:** Some bugs never get fully resolved in a single session. Track them as "open questions" not "done."

**Open at session close (from `current_session.md` line 770, 10268):**

1. **Why does cos_sim final = -0.24 when all per-stage cos values are high?** Composition is the remaining bug; cross-attention layers are the suspect.
2. **Is cross-attention composition the bug, or is it attention mask / softmax-fp32?** Open.
3. **Would copying lerobot code wholesale (hybrid option 2 from line 11574) close the gap?** Untested, likely YES.
4. **Will the ~2% per-step expert velocity error ever be fixable in ONNX?** Open; depends on whether the reimplementation can get RMSNorm / softmax / F.silu-gate-up ordering perfect.
5. **Does multi-camera (all 3 cameras instead of 1) fix LIBERO, or is it architectural drift?** Depends on whether SmolVLA-LIBERO actually trained on 3 cameras or somehow tolerates 1.
6. **8 additional next-work items from `validate` epic post-mortem** (git_history.md commit `316c1d4`): FP16 torch.compile vs TRT FP16 bench, Jetson Orin Nano validation, VLM prefix KV-cache (partially done), radon/gocyclo, SmolVLA fixture image size, HF caches, TypedDict, reproducible dev env.
7. **GR00T embodiment multi-mode**: current code pins `embodiment_id=0`. Multi-embodiment selection via flag open.
8. **xVLA support**: 880M tokenized head, new model family. GOALS.yaml weight 7.

**Learning:** Track open questions explicitly. A debug session can produce "infrastructure complete + 12 bugs fixed + task success still 0%" — that's NOT a failure, but it's NOT a complete success either. Be honest about what's open. Use the post-mortem to harvest next-work items (as in the Apr-16 validate epic did).

**Sources:**
- `current_session.md` session summary (bottom section)
- `git_history.md` commit `316c1d4` (post-mortem harvest)
- GOALS.yaml open goals list

---

## Meta-pattern: "When in doubt, log more"

Every major bug in this catalog was caught because someone added a diagnostic log line. Examples:

- `[vlm-weights] load: 488 missing, 345 unexpected` caught Bug 2 (AutoModel wrapper).
- `obs schema: top_keys=['images', 'task_description']` caught Bug 6 (send_state missing).
- `state=none ⚠️` same caught.
- `Inference mode: onnx_trt_fp16` caught the TRT EP rebuild issue when you see `onnx_cpu` unexpectedly.
- `Expert ONNX inputs: ['noisy_actions', 'timestep', 'position_ids', 'vlm_k', 'vlm_v']` confirmed the VLM KV wiring.

**Learning:** When a pipeline has ~500 operations, debug prints ARE the diagnostic tool. Log:
- Input shapes at every boundary.
- Weight-load statistics (`X missing, Y unexpected`).
- Dispatched code path (`inference_mode=onnx_trt_fp16`).
- Feature flags in effect (`vlm=on, norm=on, adaptive=off`).

Make these surface via `GET /health` and `/config` endpoints so you can verify from a `curl` call without reading logs.

**Sources:**
- Scattered across all files; see `modal_apps_and_pm_docs.md` Modal app log excerpts

---

## Files
- All bug files in this directory reference these patterns
- `src/reflex/runtime/server.py` — strict providers, /health, /config
- `src/reflex/cli.py` — `reflex doctor` for diagnostics
- `scripts/local_*_diff.py`, `scripts/modal_stage_diff.py` — diagnostic ladder tools
- `scripts/modal_pytorch_vs_onnx.py` — the decisive cos_sim test
- `.agents/council/2026-04-16-post-mortem-reflex-validate.md` — 8-item harvest exemplar

