# knowledge_capture_pattern — why `reflex_context/` exists

## The hypothesis

> If the next session starts with full knowledge of prior bugs, decisions, and benchmarks, it won't rediscover them. The `reflex_context/` folder makes that transfer cheap, searchable, and durable across Claude sessions.

## The problem this solves

The Apr-10 marquee session (`ced2c4f1`, ~91KB of session text, ~1.12M tokens, 1661 lines) contained:
- 20+ discovered bugs
- 15 architectural decisions
- 4 competitive landscape probes
- ~90 distinct insights across 21 themes
- Every benchmark number published to the README

Without a knowledge-capture step, that session's knowledge lives in one of three places, each of which decays:

1. **Session transcript** — searchable but only by the person who knows the session ID. No Claude session afterward can find it by default.
2. **Code comments** — some insights land in commits (e.g., Apr-14 GPU post-mortem comments in `modal_bench_path_b.py`), but not all. Code only captures implementation facts, not negative results ("we tried X, it didn't work").
3. **Auto-memory** (`/Users/romirjain/.claude/projects/-Users-romirjain/memory/project_reflex_vla.md`) — persists across sessions but is capped, summarized, and easy to overwrite.

Without intentional capture, repeated findings get re-discovered, wasting entire Modal runs. The Apr-17 session had to rediscover:

- "onnxruntime-gpu silently falls back to CPU on CUDA 12 vs 13 mismatch" — found Apr-14, partially re-hit Apr-17.
- "LIBERO's `setup.py` has empty `install_requires`" — found Apr-16, had to be re-applied.
- "The expert's vlm_kv_dim is 320 not 960" — found Apr-16, re-asserted Apr-17.
- "shared noise required" — articulated only mid-Apr-17 after burning ~$3 of Modal on noise-dominated diffs.

## The taxonomy

`reflex_context/` layout (per session line 11597):

```
reflex_context/
├── README.md                     — index
├── 00_vision/                    — north star, positioning, moat (strategic)
├── 01_decisions/                 — ADRs (Architectural Decision Records)
│   ├── 2026-04-14-ship-distill-first.md
│   ├── 2026-04-14-deprioritize-adapt-and-split.md
│   ├── 2026-04-14-wrap-not-rebuild-vla-eval.md
│   ├── 2026-04-14-disable-trt-when-batch-gt-1.md
│   ├── 2026-04-14-strict-provider-no-silent-cpu-fallback.md
│   └── 2026-04-16-council-reprioritization.md
├── 02_research/
│   ├── papers/                   — arXiv summaries (pi-Flow, RTC, VLA-Eval, StarVLA, ...)
│   ├── competitors/              — PI, NVIDIA GR00T, LeRobot, vlagents, AllenAI vla-eval
│   ├── hardware_partners/
│   ├── customers/
│   └── market/
├── 03_experiments/               — per-experiment result writeups
│   ├── 2026-04-14-trt-fp16-vs-torch-compile.md
│   ├── 2026-04-14-batching-scale.md
│   └── 2026-04-14-adaptive-denoising.md
├── 04_iteration_lessons/         — THIS folder (meta-lessons about how to work)
│   ├── local_vs_modal.md
│   ├── diagnostic_ladder.md
│   ├── modal_cost_log.md
│   ├── shared_noise_discipline.md
│   └── knowledge_capture_pattern.md  (this file)
├── 05_inbox/                     — raw session outputs awaiting triage
└── 06_archive/                   — old/superseded material kept for history
```

Lower-numbered dirs = stabler, more-referenced. Higher-numbered dirs = flowing / transient.

## What to capture — the checklist

At the end of each major session:

### Decisions
- [ ] Every **architectural decision** (file `01_decisions/YYYY-MM-DD-<slug>.md`). One ADR per decision. Include: context, options considered, rejected alternatives, decision, consequences. Follow the standard ADR template.
- [ ] Every **deprioritization** — what was dropped and why (so a future session doesn't revive it cargo-culted).
- [ ] Every **pivot** — the old plan, the new plan, the trigger.

### Bugs
- [ ] Every bug found and fixed: one line per bug in `02_bugs_fixed/<topic>.md`. Include: symptom (exact error message), root cause, fix, evidence (commit SHA or file:line).
- [ ] Every bug found and NOT fixed: one line per bug in `05_inbox/open_bugs.md` with a best-guess next step.
- [ ] **Numeric thresholds** that came up (cos_sim > 0.999 per step; max_diff < 1e-5 per layer; L2 < 0.5 for end-to-end). These are budgets.

### Numbers
- [ ] Every **benchmark table** published to the README or launch draft: preserve in `03_experiments/` with (a) hardware profile, (b) inputs, (c) raw numbers, (d) caveats.
- [ ] Every **$ spend** on Modal for the session. Update `04_iteration_lessons/modal_cost_log.md`.
- [ ] **Deprecations** — "we used to claim X, now we claim Y, because ...".

### Research
- [ ] Every paper referenced: `02_research/papers/<arxiv-id>-<slug>.md`. Include claim, relevance to us, whether we adopted.
- [ ] Every competitor mentioned: `02_research/competitors/<name>.md`. Include what they do, pricing, threat level, relation to our wedge.

### Meta-lessons
- [ ] Any "forgotten discipline" surfaced. Add or update `04_iteration_lessons/<topic>.md`.
- [ ] Any tool or workflow that saved measurable time. Update `04_iteration_lessons/local_vs_modal.md` or similar.

### Session log (minimal)
- [ ] `05_inbox/YYYY-MM-DD_<slug>.md` — chronological 50-200 line log, just enough that a future grep hits the right session. Link from here into the per-topic files above.

## The trade-off

Capture-discipline cost: **~1-2h at session end**. Sometimes feels like theater.

Retrieval benefit per future session: **saves 2-8h** of rediscovery per session, depending on topic density. Breaks even after 1 session; compounds thereafter.

Apr-17 session spent ~6h of Modal debugging. If the Apr-14 post-mortem had been captured as `01_decisions/2026-04-14-silent-cpu-fallback.md` with the exact error string ("libcublasLt.so.12: cannot open shared object file"), that's ~30 min of rediscovery the Apr-17 session would not have paid.

## Anti-patterns

- **Writing notes inside a long README**. Scales poorly; no one greps the README for "cos_sim progression table" at midnight.
- **Only committing decisions as code comments**. Only implementation facts land. Negative results (why we chose X over Y) don't.
- **Putting everything in auto-memory**. Gets summarized away.
- **Recapitulating session text verbatim**. The point is *distillation*. A 40-line file in `02_bugs_fixed/` beats a 4000-line transcript.

## Recommended cadence

- **Mid-session:** when a decision lands, write the ADR **before** moving to the next task. Ideal: ADR within 10 min of decision. At end of week, ADRs compound.
- **End of session:** 10-min wrap — walk through this checklist, append to relevant files.
- **Major release / end of epic:** 1-2h — reorganize `05_inbox/` into permanent homes.
- **Every new session:** first action — skim `reflex_context/` index. Skill `/vault-research` (configured in `settings.json`) does this automatically.

## The specific "forgotten disciplines" this exercise surfaced

Articulating them here so they don't go missing in the text of the four other files in this folder:

1. **Write diagnostic scripts upfront, not reactively.** The Rung 2 / 3 / 4 stage-diff scripts should have been built in Apr-14 alongside the first exporter. Instead they were written mid-Apr-17 as an emergency measure. (`diagnostic_ladder.md`)

2. **Seed the flow-matching noise from day one.** Every cos_sim before the noise fix was noise-dominated. (`shared_noise_discipline.md`)

3. **Local iteration is the default, Modal is the exception.** Running a diff script on Modal is 100x more expensive than running it locally. (`local_vs_modal.md`)

4. **Budget Modal explicitly.** $40/week cap with a kill-switch at $30 on any single debug session. Without a budget, the "$0.15 per run" rounds to zero and you end up at $15+/session. (`modal_cost_log.md`)

5. **Capture decisions as they happen, not at end of quarter.** Apr-17 had to reconstruct Apr-14 decisions from git history and session text. (`knowledge_capture_pattern.md`, this file)

6. **Prefer wrapping upstream code over decomposing it.** The 12 SmolVLA pipeline bugs, of which 4 were in our reimplementation (RoPE base 10000 vs 100000, sinusoidal time, prefix_offset, kv_mask), would have been zero if we had imported `lerobot.policies.smolvla.modeling_smolvla` directly instead of rebuilding from state_dict keys. The decomposed approach makes sense for TRT compat (only RMSNorm actually needs decomposition; torch.onnx.export handles RoPE, GQA, attention cleanly). See session line 11574.

7. **Infrastructure is insufficient — task success is the only metric.** The Apr-17 infrastructure-vs-task-success table (line 10268) lists 6 shipped components, all green — and task success = 0%. Infrastructure wins without a benchmark number are vanity.

## Where to put this file

This file IS the knowledge-capture pattern. If the next session starts by reading `reflex_context/04_iteration_lessons/knowledge_capture_pattern.md`, it has the meta-rules. If any session after this deletes this file, the discipline slips.

## Writing template for future `04_iteration_lessons/` entries

```markdown
# <topic> — <one-sentence rule>

## The rule
<one paragraph)

## Why it matters
<numbers, concrete example, what fails without it>

## The canonical pattern
<code snippet or file reference>

## What breaks without it
<specific failure mode from a real session>

## Checklist
- [ ] ...

## Forgotten discipline
<what I should have done from the start>
```

Every file in this folder follows this shape. Keep them terse and code-heavy; avoid philosophy.
