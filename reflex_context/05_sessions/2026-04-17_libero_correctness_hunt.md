# 2026-04-17 — The LIBERO Correctness Hunt

**Session theme:** Today's session. Yesterday established that the LIBERO infrastructure works. Today asks the harder question — *why is task-success 0% when every component looks numerically correct?* The per-bug chronology below walks through ~12 distinct real bugs caught, fixed, and verified via a stage-diff methodology that became the session's signature diagnostic tool. The final result: infrastructure proven, single-layer self-attn matches to 1e-5 precision, full-pipeline cos_sim still negative, task success still 0%.

---

## Goal

Make LIBERO-10 task-success > 0%. The Apr-16 infrastructure hunt proved the sim runs, the adapter works, the normalizer pipeline is in, and actions are in-distribution — yet every episode fails at 150 steps. The honest confession at the end of Apr-16 (*"I'm guessing at which subtle thing is wrong. I have 30+ candidates and no way to rank them"*) set today's methodology: **stop guessing, start stage-diffing**.

The working hypothesis entering the day: the exported ONNX has a structural bug somewhere in the pipeline that makes it semantically different from the PyTorch reference model. A stage-by-stage diff between the PyTorch policy and our ONNX pipeline — on identical preprocessed inputs with shared flow-matching noise — should localize the divergence to a specific stage, which can then be fixed.

---

## The stage-diff methodology debut

Transcript line 10799 articulated the core insight:

> *"The current diff test isn't the fastest path. It only tells us if the FINAL action matches — cos_sim=0.28 means something is wrong but not WHERE. The truly fast path is:*
> *1. Instrument the PyTorch model with forward hooks to capture intermediate tensors during `predict_action_chunk`: vision-encoder output, text embeds, per-layer VLM k/v (post-RoPE), final action.*
> *2. Run our ONNX pipeline with identical inputs, capturing matching tensors.*
> *3. Compare at every stage in one Modal run.*
> *4. The first stage where L2 diverges is the bug — we fix that specific thing."*

Two scripts implement this:
- **`scripts/modal_pytorch_vs_onnx.py`** — the decisive end-to-end test. cos_sim classification: `>0.95 → export correct; 0.5–0.95 → minor drift; ≤0.5 → structural bug`.
- **`scripts/modal_stage_diff.py`** — per-stage L2 localization: (1) vision per camera, (2) text embeds, (3) state projection, (4) per-layer VLM k (post-RoPE) and v from `decoder_prefill.onnx`, (5) expert one-step velocity with identical `vlm_k`/`vlm_v`.

Both use **shared noise** — `RandomState(7).randn(1, chunk, max_action_dim)` injected into both PyTorch and ONNX paths — because without shared initial conditions, flow-matching's cos_sim is dominated by noise drift, not export correctness. Transcript line 10846:
> *"The cos_sim numbers vary because flow-matching uses fresh random noise each call — BOTH paths get different noise, so we're comparing noisy vs noisy. Need to inject the SAME noise into both paths."*

---

## Per-bug chronology

The day's work is a string of bug discoveries each found via stage-diff, fixed in the exporter, and re-verified. The 12-bug catalog from transcript line 11574 organizes them into two categories: **pipeline orchestration bugs** (our CLI's fault) and **reimplementation bugs** (would not happen if we used lerobot code verbatim).

### Bug 1: 5D `pixel_values` silently failing VLM

Transcript lines 9605, 9615:
> *"The bug is confirmed: 5D pixel_values dropped to ONNX which wanted 4D → every VLM call silently failed → zero VLM conditioning for all episodes."*
> *"Root cause: SmolVLM's `AutoProcessor` returns `[B, N, 3, H, W]` (5D with num_images dim); our ONNX expects 4D. Every prior LIBERO run silently fell back to dummy zeros. Every task-success data point so far was without real VLM conditioning — actions were purely a function of noise + random init state."*

Fixed in `src/reflex/runtime/vlm_orchestrator.py` by squeezing the num_images dim before passing to ONNX.

### Bug 2: VLM dummy conditioning (the silent fallback)

Corollary of bug 1: when VLM failed, the orchestrator fell back to zero tensors for `vlm_k` / `vlm_v`. No error, no warning. Every episode was a "random VLM conditioning" run.

Fixed by surfacing a warning when conditioning falls back and — once bug 1 was fixed — the fallback path only fires on genuine VLM errors.

### Bug 3: `state_proj` random weights (the "smoking gun")

Transcript line 10679:
> *"FOUND THE SMOKING GUN BUG. Our state encoder uses RANDOM weights, never loading the real `state_proj` from the checkpoint.*
> *```python*
> *# vlm_orchestrator.py _load_state_encoder:*
> *if state_weight_path.exists():*
> *    self._state_weight = np.load(str(state_weight_path))*
> *else:*
> *    self._state_weight = np.random.randn(...) * 0.02*
> *```*
> *And we never save `state_proj_weight.npy` during export. So every inference has the robot state projected through a random matrix into the VLM prefix. That means: State info → meaningless garbage, VLM sees random noise where the robot pose should be, Cross-attn keys never encode robot state, Model cannot solve tasks."*

Fixed by saving `state_proj_weight.npy` at export time and loading it in the orchestrator instead of initializing random weights.

### Bug 4: `AutoModel` vs `AutoModelForImageTextToText`

This was the biggest single-fix of the day. The first stage-diff run (Modal app `ap-uKaH8uEPuCeoKz0C6TCqbV` at 10:39 IST) produced:

```
[vlm-weights] load: 488 missing, 345 unexpected
Vision stage 1: cos=+0.6983, torch||=1644 vs onnx||=1104
```

488 missing weights and 345 unexpected — the weight-prefix rebase was stopping at the wrong level. All vision weights never actually loaded; the VLM ran with base SmolVLM2 weights instead of the fine-tuned SmolVLA weights from the checkpoint.

Transcript line 11063:
> *"Found it exactly. Checkpoint's VLM was `AutoModelForImageTextToText` (structure: `model.connector.*`, `model.text_model.*`), but we load `AutoModel` (structure: `connector.*`, `text_model.*`). Keys don't match because of the extra `model.` prefix."*

Fixed: switched to `AutoModelForImageTextToText.from_pretrained(...).model` (unwrap the `ForConditionalGeneration` wrapper), restoring the `model.` prefix to match checkpoint keys.

Second run (Modal app `ap-YrnHF0WgFXQ2Y7HWlYHPaI` at 10:57 IST):
```
[vlm-weights] unwrapped ForConditionalGeneration -> inner model
[vlm-weights] load: 144 missing, 0 unexpected  ← healthy
Vision cos=+1.0000, L2=2.2022e-03, max_abs=1.0681e-04 ← matches torch
```

Transcript line 11089:
> *"Vision `cos=0.70 → 1.0000`, `max_abs=1e-4` — perfect match! Text and state also perfect. The fine-tuned VLM weight loading was the real bug."*

Also noted: *"Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)"* — meaning the SmolVLA policy checkpoint technically stores fine-tuned VLM weights that we don't yet extract. The `AutoModelForImageTextToText` fix gets us to the base SmolVLM2 weights correctly loaded; the fine-tuned deltas remain a v0.3 item.

### Bug 5: √hidden scaling

Transcript line 11248:
> *"cos went NEGATIVE (-0.21). My √hidden scaling may have made things worse, not better."*
>
> Line 11310:
> *"Decoder output is HIGHLY sensitive to input scaling (cos=0.51 between scaled vs unscaled). The scaling matters."*
>
> Line 11325:
> *"Still `cos=-0.27`. Scaling helped per-layer kv match but end-to-end is still wrong."*

Real SmolLM2 applies √hidden scaling to embeds before the first layer. Our ONNX didn't. Added the multiplication. Per-layer kv match improved (some layers went 0.99x → 0.99y with y > x), but end-to-end action cos stayed negative. The scaling fix was necessary but not sufficient.

### Bug 6: `rope_theta` — 10000 vs 100000

Transcript line 11424:
> *"Another bug! Real SmolLM2 uses `rope_theta=100000`, but our `_DecomposedRoPE` defaults to `base=10000.0`. 10× different RoPE base frequencies."*

A classic reimplementation-from-defaults bug. Our `_DecomposedRoPE(base=10000.0)` matched the standard LlamaRotaryEmbedding default, but SmolLM2 uses 100000. Fixed by reading `config.rope_theta` explicitly.

Effect on cos: per-layer kv improved further. End-to-end still negative.

### Bug 7: `prefix_offset` for self-attention

The VLM prefix has position IDs 0..N-1 for N prefix tokens. The expert's self-attention layer must use position IDs starting at N (not 0) so RoPE rotates correctly for the suffix. Our decomposed RoPE was applying position 0..M-1 for M suffix tokens, effectively resetting the rotation frequency at the prefix/suffix boundary.

Fixed by adding a `prefix_offset` parameter that shifts position IDs in self-attention layers.

### Bug 8: Multi-camera input

Transcript line 10679:
> *"vla-eval sends ONE camera (`agentview`) but SmolVLA-LIBERO was trained on THREE cameras (`camera1/2/3`). Our VLM pipeline runs with 1-camera input — very different distribution from training."*

LIBERO fine-tune was trained on 3 camera views; the adapter was only passing the first. Fixed by pulling all three camera streams from the LIBERO obs and passing them to the VLM.

Note: LIBERO obs contains `agentview` (front), `robot0_eye_in_hand` (wrist) — the naming doesn't match training's `camera1/2/3`. Best-effort mapping. Possible residual distribution shift.

### Bug 9: State dim — 8 vs 6

Transcript line 10152:
> *"State IS flowing now (8D float64), both cameras arriving (`agentview` + `wrist`). Actions still in-distribution. But 0%. Noticed one more thing: obs has BOTH `states` AND `controller_states`. Model may have been trained on `controller_states` (robot controller's output) not `states` (raw env)."*

Two observations:
- LeRobot's LIBERO dataset uses 8D state: `eef_pos(3) + axis_angle(3) + gripper_qpos(2)`. The preprocessor's normalizer stats are shape (8,).
- vla-eval sends either `states` (8D, raw env) or `controller_states` (8D, post-controller) depending on config.

Modal script config change: `send_state=True, send_wrist_image=True` (transcript line 10117):
> *"FOUND THE BUG! The diagnostic dump reveals: obs schema: top_keys=['images', 'task_description'] images=agentview:(256, 256, 3)/uint8 state=none ⚠️. vla-eval isn't sending us the robot state at all — because our LIBERO config didn't set `send_state=True`."*

After fix: state flows (8D float64). Still 0% task success.

### Bug 10: Newline on task string

Pipeline bug: lerobot's tokenizer appends `\n` to the task description before tokenization. Our CLI did not. The tokens differ by one `<newline>` token at position -1. Cross-attention K/V for that token is wrong.

Fixed by appending `\n` to the instruction string before tokenization.

### Bug 11: Sinusoidal timestep

Transcript line 11574 (the 12-bug table):
> *"Sinusoidal timestep | Our reimplementation bug | lerobot's is correct"*

Our reimplementation of `create_sinusoidal_pos_embedding(time_val, dimension, min_period=4e-3, max_period=4.0)` had a minor numerical drift vs the lerobot reference. The formula is specific — it uses a log-linear frequency spacing, not the standard 10000^(2i/d) from Transformers. Fixed to match lerobot's formulation.

### Bug 12: `obs.get(k) or obs.get(other)` on numpy arrays

Transcript line 10221:
> *"Found the bug: `obs.get(k) or obs.get(other)` on numpy arrays raises `ValueError: truth value is ambiguous`."*

Not a correctness bug in the model — an adapter bug that raised in the Python code path. Fixed with explicit `None` checks: `v = obs.get(k); v = obs.get(other) if v is None else v`.

---

## The cos_sim numerical ladder (chronological)

Transcript line-by-line progression as each bug was fixed:

| Step | cos_sim | Action |
|------|---------|--------|
| Line 10668 baseline | first data | Test-specific |
| Line 10799 | 0.28 | "Something is wrong but not WHERE." |
| Line 10809 | 0.277 → 0.498 | After state_proj + manual preprocessing fixes. |
| Line 10846 | 0.305 | With shared noise. "Our ONNX is significantly misaligned from PyTorch." |
| Line 10969 | Vision 0.6983 | **Vision encoder broken** (norms differ torch=1644 vs onnx=1104). |
| Line 11089 | Vision 1.0000 (max_abs=1e-4) | **AutoModelForImageTextToText fix — vision perfect!** |
| Line 11109 | 0.082 | "Wait — cos_sim DROPPED despite vision/text/state being perfect. Something in decoder or expert broke." |
| Line 11132 | per-layer kv cos ≥ 0.91 | "All per-layer k/v match to cos≥0.91 (layer 0 v is 0.91, others 0.99+). Yet final action cos=0.08." |
| Line 11175 | 0.08 end-to-end | "That final number doesn't match the components. Either (1) expert layer mapping is off, (2) attention mask inside expert differs, (3) timestep embedding formula, (4) flow-matching dt direction or sign." |
| Line 11248 | **-0.21** | √hidden scaling made things WORSE. |
| Line 11325 | **-0.27** | After RoPE rescue. Scaling helped per-layer kv; end-to-end still negative. |
| Line 11435 | cos_sim=0.977 per-step, -0.24 final | "Expert step velocity has ~20% norm error; 10-step flow accumulation drops cos to -0.24." |
| Line 11468 | **Single self-attn layer cos=1.0000, 1e-5 precision** | "The bug is somewhere in COMPOSITION — probably cross-attention layers." |

### The final cos_sim table (line 11435)

| Fix | Effect on cos_sim |
|-----|-------------------|
| State_proj fine-tune weights | made things coherent (was garbage) |
| AutoModelForImageTextToText + unwrap | VLM cos 0.70 → 1.00 ✓ |
| √hidden scaling | minor |
| RoPE base 10k→100k | negligible |
| prefix_offset | negligible |
| Expert step velocity | cos=0.977 per step (~20% norm error) |
| 10-step flow accumulation | cos drops to -0.24 |

Analysis: *"The expert_stack ONNX has residual ~2% per-step error that COMPOUNDS catastrophically over 10 Euler steps. Remaining candidates for that 2%:*
1. DecomposedRMSNorm vs real RMSNorm numerics
2. `F.silu(gate_proj(x)) * up_proj(x)` ordering vs real
3. Attention mask (we don't mask padded prefix positions; real does)
4. Attention softmax upcast to fp32

*Given cos=-0.24 final, task success will still be 0%. We need per-step cos >0.999 to survive 10 integration steps."*

**Budget statement:** per-step cos must exceed 0.999 to survive 10 Euler integration steps without catastrophic drift. We're at 0.977 per step. That gap is what remains.

---

## Local-iteration pivot

Mid-session realization: Modal iteration for stage-diff is absurdly expensive. Transcript line 11574:
> *"Also the 'local iteration is ~100× cheaper than Modal' lesson."*
> *"Pausing the kv_mask fix. We can, and we probably should have [copied lerobot code] from the start."*

The pivot: **run stage-diff locally.** A new family of scripts in `/scripts/` prefixed `local_*_diff.py`:
- `scripts/local_stage_diff.py` — per-stage diff on CPU.
- `scripts/local_full_diff.py` — end-to-end diff on CPU.
- `scripts/local_expert_diff.py` — expert-only diff.
- `scripts/local_single_layer_diff.py` — single-layer isolation test.

The single-layer test (line 11468) is the most important finding:
> *"Single SELF-attn layer (layer 0) matches to 1e-5 precision, cos=1.0000. The bug is somewhere in COMPOSITION — probably cross-attention layers."*

Individual self-attention components work. The layer-to-layer composition (or cross-attention specifically) is where the ~2% per-step error lives. Candidates:
- Cross-attention layer with mismatched KV dimensions.
- Attention mask (padding positions).
- Softmax fp32 upcast missing.
- `F.silu(gate) * up` ordering reversed somewhere.

### The "copy lerobot code" option

Transcript line 11524:
> *"Three realistic paths, ordered by correctness-per-engineering-hour:*
> *1. Wrap + export the REAL classes directly. `policy.model.vlm_with_expert.lm_expert` as-is, wrap in a thin nn.Module, call `torch.onnx.export` with `dynamo=True`. Output ONNX may be larger / include unsupported ops for Jetson, but CORRECTNESS is guaranteed.*
> *2. Copy lerobot's modeling files into `reflex/exporters/smolvla_native.py` verbatim. Only override ops that don't export. Much shorter bug list.*
> *3. Skip ONNX entirely for first launch. Load the real `SmolVLAPolicy`, serve it via reflex's FastAPI in pure PyTorch. Ship LIBERO success on day one."*

Line 11574:
> *"Hybrid: Copy lerobot's `SmolVLAPolicy.sample_actions` + `embed_prefix` + `embed_suffix` + `forward_cross_attn_layer` into `reflex/runtime/smolvla_native.py`. Swap only `RMSNorm → DecomposedRMSNorm` for TRT compat. Let torch.onnx.export handle the rest. Hours of work, correct by construction, Jetson compatible."*

Not yet executed. This is the leading candidate for tomorrow's session — deliberately moving from "decompose everything from scratch" to "copy lerobot's proven path, override only the Jetson-incompatible ops."

---

## Bug-classification catalog (line 11574)

The bug taxonomy the author drew from the 12-bug experience:

**Pipeline orchestration bugs (our CLI's fault, not lerobot's):**
- State_proj random weights (never saved at export)
- Base vs fine-tuned VLM loading (wrong AutoModel class)
- 5D pixel_values (wrong preprocessor)
- Missing √hidden scaling (skipped step)
- SigLIP [-1,1] range
- Missing newline on task
- 8D vs 6D state

**Our reimplementation bugs (would not happen with lerobot code):**
- Sinusoidal timestep formula
- RoPE base 10000 vs 100000 (10× error)
- `prefix_offset` for self-attn
- KV mask for cross-attn

The author's meta-conclusion: **8 of 12 bugs disappear if we use lerobot's actual code**. Reimplementation from scratch is the cost driver.

---

## Modal apps run today

Snapshot of Modal apps created 2026-04-17 (all reflex-vla):

- **`ap-MrSsaMvCuiwlYLaTCs8gOb`** (`easyinference-demo`, 2026-04-05) — not reflex.
- **`ap-uKaH8uEPuCeoKz0C6TCqbV`** (reflex-stage-diff, 10:39, ~6min) — FAIL, 488 missing / 345 unexpected weights.
- **`ap-YrnHF0WgFXQ2Y7HWlYHPaI`** (reflex-stage-diff, 10:57, ~6min) — partial PASS, vision cos=1.0000, layer_0_v cos=0.9117 outlier.
- **`ap-oXrqhfnQFJLuuY4A9GbPSv`** (reflex-stage-diff, 11:08, ~4min) — identical layer_0_v 0.9117 (reproducible).
- **`ap-2tNsuBRSnvuQ9kWPwm55Ob`** (reflex-stage-diff, 11:34, ~2min) — same, clean re-run.
- **`ap-v6gmsosx9ayiGJxoWUfs6o`** (reflex-pytorch-vs-onnx, 11:03, ~3min) — FAIL, L2=1.494, cos_sim=+0.082.
- **`ap-oBhVQcQnjsd4uMK6lSy98D`** (reflex-pytorch-vs-onnx, 11:28, ~4min) — FAIL, L2=1.890, cos_sim=**-0.209** (sign-flipped — randomness in ONNX path).
- **`ap-QAG1Pk9w3DkuZnVs9VC8Ke`** (reflex-libero10, 11:14, ~12min) — partial: task 1&2 ep0 FAIL steps=150, aborted before full 10-task run.

The `layer_0_v cos=0.9117` outlier is the reproducible structural discrepancy — seen in every stage-diff run post-AutoModel fix. Layer 0 V projection diverges while K doesn't, suggesting the divergence is in one of two layer-0-specific computations (either `v_proj` application or its downstream use).

---

## Final status (end of session)

| Component | Status |
|-----------|--------|
| Unified CLI (`reflex export` auto-produces 4 ONNX files for SmolVLA) | ✅ |
| vla-eval adapter (`ReflexVlaEvalAdapter`) | ✅ |
| Normalizer pipeline (state + action) | ✅ |
| Per-layer `vlm_kv` ONNX export | ✅ (partial — layer 0 v outlier) |
| 5D→4D pixel_values fix | ✅ |
| LIBERO sim on Modal (bddl, gym, osmesa, robosuite pin) | ✅ |
| Vision encoder stage-diff (cos=1.0000, max_abs=1e-4) | ✅ |
| Text embedder stage-diff (cos=1.0000) | ✅ |
| State projection stage-diff (cos=1.0000) | ✅ |
| Per-layer VLM k/v stage-diff | ✅ 15/16 layers (layer_0_v cos=0.9117) |
| Single-layer self-attn diff | ✅ cos=1.0000, 1e-5 precision |
| **LIBERO task success** | **❌ 0% unresolved** |

**~12 distinct real bugs fixed.** Every one would have prevented task success on its own. Infrastructure proven. Single-layer diff matches to 1e-5. Final cos_sim still negative.

---

## The confession

Transcript line 10630:
> *"Honest answer: I'm guessing at which subtle thing is wrong. Each fix I've made (5D pixel_values, normalizer, per-layer vlm_kv, layernorm on k, RoPE on keys, split k/v, newline on task, multi-camera, controller_states) was a real bug that would have made the model fail. Fixing them all and still getting 0% means there's ONE more thing — but I have 30+ candidates and no way to rank them without direct comparison.*
> *The fundamental problem: task-success is an integration test. ONE subtly wrong operation out of ~500 in the pipeline = 0%. Without a side-by-side diff against the real PyTorch model, I'm iterating blind."*

---

## Unresolved questions at session end

- Why does cos_sim final = -0.24 when all per-stage cos values are high?
- Is cross-attention composition the bug, or is it attention mask / softmax-fp32?
- Would copying lerobot code wholesale (hybrid option 2 from line 11574) close the gap?
- Will the ~2% per-step expert velocity error ever be fixable in ONNX?
- Does multi-camera (using all 3 cameras instead of 1) fix LIBERO, or is it architectural drift?
- Layer 0 V projection outlier (cos=0.9117 vs 0.99+ everywhere else) — what's unique to layer 0?

---

## Carry-over (future sessions)

The next session should open with one of three choices:

1. **Copy lerobot path.** `reflex/runtime/smolvla_native.py` — copy `SmolVLAPolicy.sample_actions`, `embed_prefix`, `embed_suffix`, `forward_cross_attn_layer` verbatim; swap only `RMSNorm → DecomposedRMSNorm`. Let `torch.onnx.export` handle the rest. Correct by construction, Jetson-compatible. Estimated: hours.
2. **Wrap-and-export real classes.** `policy.model.vlm_with_expert.lm_expert` as-is, wrap in a thin `nn.Module`, call `torch.onnx.export` with `dynamo=True`. ONNX may be larger or include unsupported ops for Jetson, but correctness is guaranteed.
3. **Skip ONNX for launch.** Load real `SmolVLAPolicy`, serve via reflex's FastAPI in pure PyTorch. Ship LIBERO success day one. Jetson export deferred to v0.3.

Unfinished work that carries over:

- **Layer 0 V projection outlier** — the reproducible cos=0.9117 data point. Suspected to live in cross-attention composition or layer-0-specific weight handling.
- **Per-step 2% expert velocity error** — compounds to -0.24 over 10 Euler steps. Candidates: RMSNorm numerics, MLP ordering, attention mask on padded prefix, softmax fp32 upcast.
- **30+ remaining bug candidates** — no triaged list. The local-iteration pivot is meant to make that triage cheap.
- **Fine-tuned SmolVLA VLM layer extraction (v0.3)** — base SmolVLM2 now loads correctly; fine-tuned deltas still pending.
- **PyTorch-only fallback serve** — not built. Would let us ship LIBERO task-success > 0% while debugging ONNX correctness asynchronously.
- **`reflex.runtime.smolvla_native` hybrid** — proposed, not executed.
- **LIBERO-10 run with > 0% task success** — the north-star number. Still TBD.

The arc of today: **we stopped guessing, started stage-diffing, caught every bug the diff caught, and now need a different methodology because the remaining bug is composition-level, not per-stage.** Tomorrow's work lives at the boundary between "copy lerobot code" and "hybrid swap-only-what-Jetson-breaks."
