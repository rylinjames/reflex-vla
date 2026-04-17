# Current Session Mining — Reflex VLA Transcript Insights

**Source transcript:** `/Users/romirjain/.claude/projects/-Users-romirjain/ced2c4f1-a341-45bf-ae1b-ba9f6ab0931c.jsonl` (53MB, 11,664 lines, 4,530 assistant messages)

**Mining method:** Keyword grep across assistant messages for: root cause, smoking gun, THE BUG, FOUND IT, BREAKTHROUGH, post-mortem, HEADLINE, MAJOR DIVERGENCE, verdict, cos_sim, cos=, tradeoff, blocker, rabbit hole, vlm_k, vlm_v, RoPE, state_proj, SigLIP, embed_prefix, prefix_offset, kv_mask, controller_states, send_state, num2words, bddl, osmesa, egl, rope_theta, Modal. 548 assistant messages hit keywords; this document captures the substantive ones.

---

## Theme: Model export architecture decisions (SmolVLA/pi0/pi0.5/GR00T)

### Why decompose RMSNorm/RoPE for ONNX
"Exact blockers are known. RMSNorm and RoPE need manual decomposition (opset 19-20). Flow matching loop exported as single step + CUDA Graph replay. Vision encoder (SigLIP) exports cleanly." (line 2037/2051)

"We've been writing our own `ExpertGQALayer`, `DecomposedRMSNorm`, `_DecomposedRoPE`, decomposed attention etc. from scratch — optimized for ONNX/TensorRT op coverage and edge hardware. Every tiny mismatch (missing `2π` in sinusoidal, wrong `rope_theta`, missing pad mask) then requires us to discover it and fix it." (line 11524)

### Direct torch.onnx.export isn't a silver bullet
"TRT has known fp16 accuracy issues on SigLIP (issues #3908, #4373) — independent of how you export. This is an NVIDIA bug that bites regardless of our decomposition. TRT's ONNX parser doesn't support opset 23 `RMSNormalization` op yet (issue #4639). Modern torch export emits this op when it sees nn.RMSNorm. So if we use torch.onnx.export with default settings, RMSNorm won't compile on Jetson." (line 11574)

"For Jetson TRT, we STILL need decomposed RMSNorm. But only RMSNorm — torch.onnx.export handles attention, RoPE, GQA, etc. cleanly." (line 11574)

### SmolVLA structure — canonical reference (line 2513)
```
model.vlm_with_expert.vlm:        345 tensors, 350.2M params  (VLM backbone — SmolVLM2)
model.vlm_with_expert.lm_expert:  145 tensors,  98.2M params  (Action expert — flow matching head)
model.action_time_mlp_in:          1.0M params  (timestep embedding)
model.action_time_mlp_out:         0.5M params  (timestep projection)
model.action_in_proj:              projection into action space
model.action_out_proj:             projection out of action space
model.state_proj:                  robot state projection
```
"350M VLM backbone + 98M action expert + 1.6M projections = 450M total."

### GQA dimensions — line 2619
"k_proj shape mismatch — k_proj is `[320, 720]` not `[960, 720]`. That's because the expert uses GQA: fewer KV heads than Q heads. Q has 15 heads (960/64), K/V have 5 heads (320/64)."

### Expert layer shape reference — line 2660
"Expert layer architecture: 720 hidden, 15 Q heads, 5 KV heads (GQA), 64 head_dim, 2048 intermediate, 6.27M params per layer. ONNX export: 0.1MB per layer, max_diff=5.36e-07."

### vlm_kv dimension mismatch (line 2841 + 7195)
"The VLM hidden states are 960-dim but the cross-attn k/v projections expect 320-dim input. The VLM output is full hidden size (960) but in the real SmolVLA, cross-attention uses the VLM's KV states (which are already projected down to KV head dimension), not the full hidden states. vlm_kv_dim=320 = 5 KV heads x 64 head_dim."

"Our v1 stub uses vlm_kv_dim=512. The real dimension is 960. SmolLM2 hidden_size is 960. The expert's cross-attention projects 960→720 internally. The entire I/O contract in the stub is wrong-shaped and needs to be corrected regardless of which approach we pick." (line 7195)

Later (line 7419): "The expert ONNX expects `vlm_kv` dim 320 (cross-attention size), not 960 (VLM hidden)." — resolved by reading actual ONNX input shape instead of config.

### pi0 FF layer is plain GELU, not GEGLU — line 3846
"Found the bug. The FF layer is NOT GEGLU — it's a plain GELU-activated MLP. `ff.net.0.proj` outputs `ff_inner` directly (not `2*ff_inner`)."

### OpenVLA export insight — line 4009
"Researched + concluded: OpenVLA's 'action head' is just `argmax(lm_logits[:, -7:]) + bin-lookup` on Llama vocab. No custom expert to reconstruct; full model is standard Llama-2-7B + DINOv2 + SigLIP that optimum-onnx already handles. Shipped `src/reflex/postprocess/openvla.py` — the *only* OpenVLA-specific piece worth owning."

### GR00T serve limitation — line 4009
"Ran all-models Modal benchmark with GR00T added. Export passes (max_diff=3.5e-05), serve fails — discovered that GR00T's expert emits 1024-dim velocity but input is 1536-dim action tokens. Generic denoising loop can't `noisy + velocity * dt` across dims."

---

## Theme: The 12 pipeline bugs (SmolVLA→ONNX→LIBERO)

A catalog of bugs discovered during LIBERO-10 debugging. The assistant verbalized these as a table in line 11574:

| Bug | ONNX-related? | Applies to PyTorch serving? |
|---|---|---|
| State proj random weights | No — our CLI just didn't save them | Doesn't happen in lerobot |
| Base vs fine-tuned VLM | No — our CLI didn't reload | Doesn't happen in lerobot |
| 5D pixel_values | No — wrong preprocessor | lerobot uses its own preprocessor correctly |
| Missing √hidden scaling | No — we skipped the step | lerobot does it |
| SigLIP [-1,1] range | No | lerobot does it |
| Missing newline on task | No | lerobot does it |
| 8D vs 6D state | No | lerobot does it |
| Sinusoidal timestep | Our reimplementation bug | lerobot's is correct |
| RoPE base 10000 vs 100000 | Our reimplementation bug | lerobot's is correct |
| prefix_offset for self-attn | Our reimplementation bug | lerobot's is correct |
| KV mask for cross-attn | Our reimplementation bug | lerobot's is correct |

**"8 of 12 bugs disappear if we use lerobot's actual code."** (line 11574)

### Bug #1: state_proj was using random weights (the "smoking gun") — line 10679
"FOUND THE SMOKING GUN BUG. Our state encoder uses RANDOM weights, never loading the real `state_proj` from the checkpoint.
```python
# vlm_orchestrator.py _load_state_encoder:
if state_weight_path.exists():
    self._state_weight = np.load(str(state_weight_path))
else:
    self._state_weight = np.random.randn(...) * 0.02
```
And we never save `state_proj_weight.npy` during export. So every inference has the robot state projected through a random matrix into the VLM prefix. That means: State info → meaningless garbage, VLM sees random noise where the robot pose should be, Cross-attn keys never encode robot state, Model cannot solve tasks."

### Bug #2: Loading base SmolVLM2 instead of fine-tuned weights — line 11063
"Found it exactly. Checkpoint's VLM was `AutoModelForImageTextToText` (structure: `model.connector.*`, `model.text_model.*`), but we load `AutoModel` (structure: `connector.*`, `text_model.*`). Keys don't match because of the extra `model.` prefix."

"**Vision `cos=0.70 → 1.0000`, `max_abs=1e-4` — perfect match! Text and state also perfect. The fine-tuned VLM weight loading was the real bug.**" (line 11089)

### Bug #3: 5D pixel_values silently failing VLM — line 9605/9615
"The bug is confirmed: 5D pixel_values dropped to ONNX which wanted 4D → every VLM call silently failed → zero VLM conditioning for all episodes. Fix applied."

"Root cause: SmolVLM's `AutoProcessor` returns `[B, N, 3, H, W]` (5D with num_images dim); our ONNX expects 4D. Every prior LIBERO run silently fell back to dummy zeros. **Every task-success data point so far was without real VLM conditioning** — actions were purely a function of noise + random init state."

### Bug #4: SigLIP image range [-1,1] vs [0,1] — line 10835
"Task `b981ighwg`. Image now in `[-1, 1]` range for SigLIP (was `[0, 1]`). 7 min for next cos_sim data point."

### Bug #5: state dim 8 vs 6 truncation — line 10679 / 9486
"vla-eval sends ONE camera (`agentview`) but SmolVLA-LIBERO was trained on THREE cameras (`camera1/2/3`). Our VLM pipeline runs with 1-camera input — very different distribution from training. Also confirmed state dim (8D → 6D truncation) is correct."

### Bug #6: `send_state=True` missing from config — line 10117
"**FOUND THE BUG!** The diagnostic dump reveals:
```
obs schema: top_keys=['images', 'task_description']
images=agentview:(256, 256, 3)/uint8
state=none ⚠️
```
vla-eval isn't sending us the robot state at all — because our LIBERO config didn't set `send_state=True`. Result: model gets `state=None` → server passes `zeros(6)` → model has no idea where the robot is → garbage actions."

### Bug #7: states vs controller_states — line 10152
"State IS flowing now (8D float64), both cameras arriving (`agentview` + `wrist`). Actions still in-distribution. But 0%. Noticed one more thing: obs has BOTH `states` AND `controller_states`. Model may have been trained on `controller_states` (robot controller's output) not `states` (raw env)."

### Bug #8: RoPE base was 10000 not 100000 — line 11424
"**Another bug!** Real SmolLM2 uses `rope_theta=100000`, but our `_DecomposedRoPE` defaults to `base=10000.0`. **10× different RoPE base frequencies.**"

### Bug #9: Missing normalizer pipeline — line 9226
"Found probable root cause: SmolVLA LIBERO checkpoint ships with `policy_preprocessor` (input normalizer) and `policy_postprocessor` (output unnormalizer) — we're not loading/applying either. Model expects normalized state, returns normalized actions. We feed raw state → get normalized actions → LIBERO interprets them as real joint values → failure."

"The LIBERO-tuned SmolVLA checkpoint ships with two normalizer files in its HF repo that the pipeline wasn't using: `policy_preprocessor_*.safetensors` (normalizes state) and `policy_postprocessor_*.safetensors` (un-normalizes actions)." (line 9302)

### Bug #10: Text embedder non-determinism fallback — line 7534/7528
"Found it. The text encoding is non-deterministic — `_encode_text()` produces different embeddings on consecutive calls with the same input (max_diff 0.135). The decoder then amplifies that into 33.8 across 16 layers."

"Root cause found. Line 371-373: when there's a tokenizer but no `text_embedder.onnx`, it falls back to `np.random.randn()` — fresh random numbers every call. We have the tokenizer (GPT2TokenizerFast) but no `text_embedder.onnx` in the export dir, so it hits this path every time."

Determinism resolved: "Test suite: 36 passed → 36 passed. Determinism max_diff 34.7 → 0.0. Fixed by seeding the RNG with the token IDs." (line 7541)

### Bug #11: Missing newline on task
Referenced in the "12 bugs" table (line 11574). Task input needed a trailing newline that lerobot includes automatically.

### Bug #12: `obs.get(k) or obs.get(other)` raises on numpy arrays — line 10221
"Found the bug: `obs.get(k) or obs.get(other)` on numpy arrays raises `ValueError: truth value is ambiguous`."

### Related code
- `src/reflex/exporters/vlm_prefix_exporter.py`
- `src/reflex/runtime/vlm_orchestrator.py`
- `src/reflex/exporters/smolvla_exporter.py`
- `src/reflex/exporters/pi0_exporter.py`
- `scripts/modal_libero10.py`

---

## Theme: Cosine-similarity diagnostic ladder

### The diagnostic approach — lines 10799, 11458
"The current diff test isn't the fastest path. It only tells us if the FINAL action matches — cos_sim=0.28 means something is wrong but not WHERE. The truly fast path is:
1. Instrument the PyTorch model with forward hooks to capture intermediate tensors during `predict_action_chunk`: vision-encoder output, text embeds, per-layer VLM k/v (post-RoPE), final action.
2. Run our ONNX pipeline with identical inputs, capturing matching tensors.
3. Compare at every stage in one Modal run.
4. The first stage where L2 diverges is the bug — we fix that specific thing."

Later (line 11458), the bisection ladder got more formal:
"Take ONE layer of the real expert (e.g. `policy.model.vlm_with_expert.lm_expert.layers[0]`). Run it forward with known inputs (noise + k/v + position_ids). Run OUR `ExpertGQALayer` with the SAME weights and inputs. Compare outputs. Diff should be 0 or near-zero. If single-layer diff is tiny → the bug is in how layers compose (attention mask, between-layer norm). If single-layer diff is big → the bug is in one of: RMSNorm formula, MLP ordering, GQA reshape, RoPE, attention scale."

### The cos_sim numerical progression (chronological)
- **Line 10668:** First baseline cos_sim test. "Cos similarity tells us: >0.95 → export is correct; <0.5 → structural bug remains."
- **Line 10799:** "cos_sim=0.28 means something is wrong but not WHERE."
- **Line 10809:** "cos_sim 0.277 → 0.498 after state_proj + manual preprocessing fixes. Still MAJOR divergence but moving in the right direction."
- **Line 10846:** "The cos_sim numbers vary because flow-matching uses fresh random noise each call — BOTH paths get different noise, so we're comparing noisy vs noisy. Need to inject the SAME noise into both paths."
- **Line 10912:** "With shared noise, cos_sim=**0.305**. Our ONNX is significantly misaligned from PyTorch."
- **Line 10969:** "Stage diff shows: Text embedder: `cos=1.0000` ✓. State projection: `cos=1.0000` ✓. **Vision encoder: `cos=0.6983`** ❌ (norms differ: torch=1644, onnx=1104). The vision encoder is broken."
- **Line 11089:** "Vision `cos=1.0000`, `max_abs=1e-4` — perfect match! Text and state also perfect. The fine-tuned VLM weight loading was the real bug."
- **Line 11109:** "Wait — cos_sim dropped from 0.305 to 0.082 even though vision/text/state are perfect. Something in decoder_prefill or expert broke."
- **Line 11132:** "All per-layer k/v match to cos≥0.91 (layer 0 v is 0.91, others 0.99+). Yet final action cos=0.08. That means the expert_stack.onnx is diverging or there's something interleaved wrong."
- **Line 11175:** "Full action (shared noise): cos=0.08. That final number doesn't match the components. Either: (1) Expert layer mapping is off, (2) Attention mask inside expert differs, (3) Timestep embedding formula, (4) Flow-matching dt direction or sign. Any of these can make per-step error tiny but cumulative error catastrophic over 10 denoising steps."
- **Line 11248:** "cos went NEGATIVE (-0.21). My √hidden scaling may have made things worse, not better."
- **Line 11310:** "Decoder output is HIGHLY sensitive to input scaling (cos=0.51 between scaled vs unscaled). The scaling matters."
- **Line 11325:** "Still `cos=-0.27`. Scaling helped per-layer kv match but end-to-end is still wrong."
- **Line 11435:** Comprehensive status table shown below.
- **Line 11468:** "Single SELF-attn layer (layer 0) matches to **1e-5 precision, cos=1.0000**. The bug is somewhere in COMPOSITION — probably cross-attention layers."

### Final verdict on where cos_sim stalled (line 11435)
Fix / effect on cos_sim table:
| Fix | Effect on cos_sim |
|---|---|
| State_proj fine-tune weights | made things coherent (was garbage) |
| AutoModelForImageTextToText + unwrap | VLM cos 0.70 → 1.00 ✓ |
| √hidden scaling | minor |
| RoPE base 10k→100k | negligible |
| prefix_offset | negligible |
| Expert step velocity | cos=0.977 per step (~20% norm error) |
| 10-step flow accumulation | cos drops to -0.24 |

"The expert_stack ONNX has residual ~2% per-step error that COMPOUNDS catastrophically over 10 Euler steps. Remaining candidates for that 2%:
1. DecomposedRMSNorm vs real RMSNorm numerics
2. `F.silu(gate_proj(x)) * up_proj(x)` ordering vs real
3. Attention mask (we don't mask padded prefix positions; real does)
4. Attention softmax upcast to fp32

Given cos=-0.24 final, task success will still be 0%. We need per-step cos >0.999 to survive 10 integration steps."

---

## Theme: Strategic pivots

### The TRT FP16 breakthrough flip — line 4973, 4989
"TRT FP16 is 3.2× faster than torch.compile!"
```
Model      Params    compile   ORT-GPU   TRT-FP16   Speedup
smolvla    99.8M     3.06      3.26      0.95       3.2×
pi0        314.6M    6.23      5.53      1.94       3.2×
pi05       426.9M    7.34      7.37      2.24       3.3×
gr00t      1091.7M   14.61     14.45     5.59       2.6×
```
"This **completely flips the strategic narrative**. Reflex's TRT path doesn't just compete with torch.compile — it dominates by 3× on cloud GPU. The whole 'we can't beat torch.compile' narrative from earlier is dead."

### The earlier "silent CPU fallback" scare — line 4372
"Every 'A100 benchmark' I've reported in this conversation, the README, and the roadmap is ONNX Runtime CPU execution running on an A100 box where the GPU is doing nothing. The latency numbers (417ms / 968ms / 1036ms / 2352ms) are pure CPU numbers at A100-box prices. That's a material misrepresentation I've been repeating."

### Root cause of the silent CPU fallback — line 4445
"`libcublasLt.so.12: cannot open shared object file / Require cuDNN 9.* and CUDA 12.*`. That's the exact root cause — ORT 1.24 needs CUDA 12 libs, but Modal installed CUDA 13 (via torch's `cuda-toolkit-13.0.2`)."

### CUDA Graph over torch.compile — strategic loss — line 4504
"torch.compile full loop: 52.87ms. Reflex CUDA graph: 69.7ms ← 25% slower than torch.compile. Root cause I didn't catch: my CUDA graph captures the *eager* model's kernels, not the torch.compile'd ones. torch.compile(`mode='reduce-overhead'`) already uses CUDA graphs internally per forward pass, AND applies kernel fusion. I get CUDA-graph savings but lose kernel fusion."

### TRT × batching sharp edge — line 5077, 6224
"When `--max-batch > 1`, fall back from TRT EP to CUDA EP. TRT EP was rebuilding engines per input shape → 34s/call, 200x pessimization. Root cause of the 200x pessimization: static-shape ONNX exports force TRT to rebuild engine when batch shape changes. Fix targeted for weeks 3-4 of sprint (export v2)." (line 6224)

"Current v0.1: batch=1 → TRT EP (38.3 qps); batch 4-16 → CUDA EP (37-49 qps)" (line 6224)

### Strategic lessons after Path B post-mortem — line 4538 (L1-L5)
- **L1:** "Reflex cannot win 'faster inference on cloud GPU.' torch.compile beats our ONNX export 6-14x on A100. Conclusion: don't pitch cloud latency. The moat is cross-framework consistency + edge deployment (Jetson TRT) + deterministic graph, not datacenter throughput." (preceded the TRT FP16 discovery — see line 4973 for flip)
- **L2:** "Phase 1 is distribution, not revenue. Pure software tooling for robotics has near-zero revenue precedent. ROS, MoveIt, Isaac Sim, LeRobot all free. Paying customers in robotics buy: (a) hardware, (b) consulting hours, (c) compliance certs. Reflex-as-SaaS ceiling is realistically $0-2M ARR ever."
- **L3:** "Revenue lives in Phase 2-4, not Phase 1." (Hardware bundle → compliance SaaS → custom silicon)
- **L4:** "Orin Nano 8GB is the right starter SKU. Apr 14 benchmark showed all 4 current VLAs fit on $500 Orin Nano with FP16 + 2x overhead."
- **L5:** "Narrative discipline required. Stop every public artifact and ask: 'does this pitch Reflex makes inference faster on your GPU'? If yes, rewrite."

### Council reprioritization after brainstorm — line 6497
"**1. VLM prefix encoder is #1, not #2.** Robotics engineer, solo-founder critic, pre-mortem all say the same thing: v0.1 benchmarks a noise generator. Every speed number, every Jetson validation, every pitch line is theater until real VLM conditioning works. Pre-mortem: 'Fix the brain before timing it or compressing it.'

**2. External user is week 1, not week 7.** Three critics agree.

**3. Distill is seductive but wrong right now.** Robotics engineer calls it researcher catnip. Solo-founder critic calls it the biggest time-waster. Resolution: distill is the right *eventual* paid wedge, wrong *now* because you have no users to sell to.

**4. API-key auth, dynamic-batch ONNX, and audit-log/guard-v2 should be deleted from the v0.2 scope.**

**5. Silicon/datacenter narrative hurts you at pre-seed. VC says hide it; pattern-matches to Rivos cosplay.**

**6. $99/mo anchors you cheap. Restructure: Pro $299/mo, Team $1500/mo, Enterprise $30-80k/yr.**"

### CLI redesign — 13→9 commands — line 8117
"Three pain points:
1. Three commands pretend to 'validate' something: `check` (static pre-flight), `validate` (ONNX-vs-PyTorch parity), `guard` (URDF safety limits). Users will guess wrong.
2. Two commands pretend to 'benchmark': `bench` (latency) and `eval` (task success).
3. Three commands are dead weight: `split` (<10% use), `adapt` (no users), `turbo` (broken on 3/4 models)."

Proposed redesign: MERGE check → validate --quick. MERGE eval → bench --benchmark. DELETE split (replace with `--cloud-fallback` flag). DELETE adapt. DELETE turbo (move to `serve --adaptive-steps`).

### Pivot: Copy lerobot code instead of decomposing — line 11524
"Three realistic paths, ordered by correctness-per-engineering-hour:
1. Wrap + export the REAL classes directly. `policy.model.vlm_with_expert.lm_expert` as-is, wrap in a thin nn.Module, call `torch.onnx.export` with `dynamo=True`. Output ONNX may be larger / include unsupported ops for Jetson, but CORRECTNESS is guaranteed.
2. Copy lerobot's modeling files into `reflex/exporters/smolvla_native.py` verbatim. Only override ops that don't export. Much shorter bug list.
3. Skip ONNX entirely for first launch. Load the real `SmolVLAPolicy`, serve it via reflex's FastAPI in pure PyTorch. Ship LIBERO success on day one."

"**Hybrid:** Copy lerobot's `SmolVLAPolicy.sample_actions` + `embed_prefix` + `embed_suffix` + `forward_cross_attn_layer` into `reflex/runtime/smolvla_native.py`. Swap only `RMSNorm → DecomposedRMSNorm` for TRT compat. Let torch.onnx.export handle the rest. Hours of work, correct by construction, Jetson compatible." (line 11574)

---

## Theme: LIBERO-10 simulation blockers

### The cascade of dependencies — lines 8596-8776
- **bddl missing:** "every LIBERO episode fails with `ModuleNotFoundError: No module named 'bddl'`. All inference/export/server plumbing works — it's the sim side missing a dependency. LIBERO uses Stanford's BDDL (Boolean Domain Definition Language) for task specs."
- **LIBERO's setup.py has empty install_requires:** "`bddl==1.0.1` is on pypi. LIBERO's `setup.py` has empty `install_requires=[]`, so none of its requirements.txt deps get installed. Fix: install requirements.txt explicitly."
- **Run ladder (line 8776):**
  - Run #1 — 0% because VLM files weren't in export (fixed by CLI Step 1)
  - Run #2 — killed (requirements.txt nuked stack)
  - Run #3 — missing cmake for egl_probe
  - Run #4 — bddl/future/gym cascade
  - Run #5 (now) — proper readiness polling, up to 5min warmup budget

### EGL → osmesa rendering hang — lines 9100, 9147
"Vla-eval got connected, LIBERO env started, 8× dataset warnings, then hung silently for 180s — killed by our idle-timeout guard. Classic MuJoCo EGL rendering hang. Trying `MUJOCO_GL=osmesa` (software renderer) instead."

"Same silent hang with osmesa — so rendering backend isn't the cause. Something deeper in LIBERO env.reset."

"Sim is mid-first-episode (osmesa is slow). Last log at 21:18 UTC, ep0 should finish around 21:21 UTC." (line 9453)

### num2words dep for SmolVLM processor — line 9574
"Missing `num2words` dep for SmolVLM processor. Trivial fix."

### `env.reset()` actually works — line 9173
"BREAKTHROUGH! 🎉 Smoke test: `env.reset()` works in 6.4s (not hung — buffering was the issue all along). First episode actually ran 150 steps: `[1/20]...ep0: FAIL (steps=150)` — model ran inference 150 times, just didn't complete task in the step budget."

### Related code
- `scripts/modal_libero10.py`
- `src/reflex/runtime/adapters/vla_eval.py`
- `GOALS.yaml`

---

## Theme: Infrastructure vs task success tension (the infrastructure-is-proven dilemma)

### Final status of LIBERO-10 after 12 bugs fixed — line 10268
"Final data point: `0% with controller_states + per-layer vlm_kv + normalizer + all plumbing correct`. Actions are firmly in-distribution (within 2σ of training mean/std across all 7 dims). State flows. VLM pipeline complete. Everything measurable looks correct.

**The infrastructure win is real and worth shipping:**
| Component | Status |
|---|---|
| Unified CLI (`reflex export` auto-produces 4 ONNX files for SmolVLA) | ✅ |
| vla-eval adapter (`ReflexVlaEvalAdapter`) | ✅ |
| Normalizer pipeline (state + action) | ✅ |
| Per-layer `vlm_kv` ONNX export | ✅ |
| 5D→4D pixel_values fix | ✅ |
| LIBERO sim on Modal (bddl, gym, osmesa, robosuite pin) | ✅ |

**LIBERO task success: 0% unresolved.** Remaining unknowns (can only be isolated by a proper PyTorch-vs-ONNX side-by-side diff):
- Subtle numerical drift in per-layer kv computation (position embeddings, attention masks)
- Per-layer mapping (we apply k_proj to `hidden_states[i]` = input of layer i; real model may apply post-some-operation)
- Single camera used despite model trained on 3 (wrist IS in obs; we pick first=agentview only)"

### The "guessing at subtle things" confession — line 10630
"Honest answer: **I'm guessing at which subtle thing is wrong.** Each fix I've made (5D pixel_values, normalizer, per-layer vlm_kv, layernorm on k, RoPE on keys, split k/v, newline on task, multi-camera, controller_states) was a real bug that would have made the model fail. Fixing them all and still getting 0% means there's ONE more thing — but I have 30+ candidates and no way to rank them without direct comparison.

**The fundamental problem**: task-success is an integration test. ONE subtly wrong operation out of ~500 in the pipeline = 0%. Without a side-by-side diff against the real PyTorch model, I'm iterating blind."

---

## Theme: Modal iteration cost & local-first workflow

### The $/iteration cost observation
- "~$15 Modal cost tonight. Worth it for the auto-TRT-FP16 finding alone." (line 5453)
- "Modal spend tonight: ~$8-12 across ~12 runs." (line 5218)
- Item 1 (distill): "**~$200-500 Modal, needs data pipeline**, ~1 week of focused work." (line 5756) — correction from earlier "$10 Modal" estimate
- "Local iteration is ~100× cheaper than Modal" is called out as a lesson in line 11574.

### Instructions for future work — line 11574
"Also the 'local iteration is ~100× cheaper than Modal' lesson."

"Pausing the kv_mask fix. We can, and we probably should have [copied lerobot code] from the start." (line 11524)

### Rabbit hole warning — line 9682
"Rather than fighting the PyTorch API (which is a rabbit hole), let me make a cheaper test: run ONE LIBERO episode with `max_steps=600` instead of 150. That's the SmolVLA paper's eval setting. If a task succeeds at step ~400, step budget was the culprit."

---

## Theme: Benchmark results — headline numbers

### TRT FP16 wins over torch.compile (line 4989) — already cited above but canonical
```
Model      Params    compile   ORT-GPU   TRT-FP16   Speedup
smolvla    99.8M     3.06ms    3.26ms    0.95ms     3.2×
pi0        314.6M    6.23ms    5.53ms    1.94ms     3.2×
pi05       426.9M    7.34ms    7.37ms    2.24ms     3.3×
gr00t      1091.7M   14.61ms   14.45ms   5.59ms     2.6×
```

### `reflex bench` all-4-VLA results — line 5522
```
Model       mean_ms   p95_ms        mode           export_s
smolvla     11.67     11.85    onnx_trt_fp16        72.8s
pi0         23.57     24.22    onnx_trt_fp16       112.2s
pi05        27.07     27.76    onnx_trt_fp16       151.9s
gr00t       56.55     57.25    onnx_trt_fp16       181.9s
```
Per-chunk (10 denoise steps) effective Hz:
| Model | Per-chunk | Effective Hz |
|---|---|---|
| SmolVLA | 11.7ms | 86 Hz |
| pi0 | 23.6ms | 42 Hz |
| pi0.5 | 27.1ms | 37 Hz |
| GR00T | 56.6ms | 18 Hz (borderline; needs optimization) |

### Multi-robot batching scaling — line 5218
"32 concurrent pi0 requests: 17 → 49 qps (2.88×). Per-request latency drops 60-80% (no more serial queueing)."

### veRL log-prob divergence benchmark (earlier path / infrascope project) — line 1057
"Results from BF16 (training) vs FP16 (serving) on Qwen2.5-7B on A100:
- 10/10 requests divergent
- 5 critical, 5 drift
- 419/1209 tokens (35%) above 0.01 threshold
- Max delta: 2.05 (token '**' at position 126 — training says -0.05, serving says -2.11)
- P95 delta: 1.17"

---

## Theme: Research-derived architectural insights

### pi-Flow over "function matching on random noise" — line 5756
"**Winner: pi-Flow (arXiv 2510.14974, ICLR 2026).** Target 4 denoise steps (not 2 or 3). Student trained via imitation flow-matching loss on teacher's own trajectories.

**Uncomfortable correction to what I told you earlier:**
- ❌ My claim: 'no data needed, function-match on random noise'
- ✅ Reality: action head is conditioned on VLM features, so we need real (image, state, language) triples from LeRobot/LIBERO/DROID. Random noise alone gives nonsense students.
- Compute: ~200-500 A10G-hours (not ~$10 — more like $200-500)"

### DMPO replaces pi-Flow for one-step generation — line 6961
"DMPO (arXiv 2601.20701) achieves true one-step generation at 1770 Hz without a teacher — pi-Flow is no longer state-of-the-art for distillation."

### Cosmos-Reason, not Qwen3, for GR00T — line 5756
"GR00T N1.6 uses Cosmos-Reason-2B, NOT Qwen3. I had this wrong in the earlier pi0_exporter research. Need to revisit."

### PaliGemma2 ONNX prior art exists — line 5756
"PaliGemma2 ONNX is already a solved recipe — `onnx-community/paligemma2-3b-pt-224` has the 3-file split (vision_encoder + embed_tokens + decoder_model_merged). Use as template."

### Jetson sizing — line 5756
"**Jetson Nano fit problem:**
- PaliGemma FP16: 5.6 GB (fits)
- Plus SigLIP + expert + runtime overhead: tight on Orin Nano 8GB
- INT4 quantization brings PaliGemma to ~2GB → comfortable fit
- Orin NX 16GB is the 'safe' recommendation, not Orin Nano"

### Jetson Orin Nano Super is $249, not $499 — line 5774
"Correction 1: Jetson Orin Nano Super is $249, not $499. NVIDIA dropped the Super 8GB price.
Correction 2: CloudJetson.com exists — 'Rent NVIDIA Jetson Hardware by the Hour'. Estimated ~$5 for a 30-min Orin Nano session.
Correction 3: Used/refurb is viable. eBay active listings around $200-240 for Orin Nano Super 8GB."

### NVIDIA Inception correction — line 5783
"NVIDIA Inception does NOT give free Jetson kits. It's a 15-30% discount + $25k AWS credits + DLI training + VC intros."

### LeRobot pain ranking — line 5783
"Top 10 engaged issues are dominated by training reliability and install hell. Deployment is #3.

BUT — ONNX/TRT is a completely unclaimed slice:
- ONNX issues: 4 open / 0 closed
- TensorRT: 3 open / 0 closed
- Issue #3146 is OPEN, 5 reactions, 0 comments, untouched since March

The 5 pains that ACTUALLY dominate 2026 LeRobot discussion:
1. 'Trained but robot does wrong thing on real hardware' (real/sim gap)
2. Install hell on Jetson/Thor/aarch64
3. Version drift (transformers/torch/JAX upgrades break weight keys)
4. Async-inference flakiness + slowness
5. Dataset format churn (v2.1 → v3.0 → Lance)

Reflex hits #2, #3, #4 — **reframe positioning: 'Your Jetson deploy is broken — here's the fix.'**"

### ETARS repo has a SmolVLA ONNX notebook — line 7195
"`aifoundry-org/ETARS` already has a SmolVLA ONNX export notebook (`smolVLA_libero_export.ipynb`). Someone already solved this exact problem. Study their code, adapt to our exporter, validate. Fastest path."

---

## Theme: ETARS-inspired 4-file split correction — line 7252

"**What the original plan got wrong:**

1. **Wrong number of ONNX files.** Plan said 3 files (vision + embed_tokens + decoder). Research found SmolVLA needs **4 files** matching the ETARS pattern: vision.onnx, text_embedder.onnx, expert_prefill.onnx, expert_decode.onnx. The split is different because SmolVLA has a special prefill step where the decoder processes image+text+state to produce KV cache, then the expert consumes that cache during denoising.

2. **Wrong model loading.** Plan said use `AutoModel.from_pretrained('SmolVLM2-500M')` which loads all 32 SmolLM2 layers. SmolVLA only uses 16. The fix: load via `SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')` which gives you 16 layers automatically because the checkpoint already has them truncated.

3. **Vision export was overengineered.** Plan said build a custom `VisionEncoder(vision_model, connector)` wrapper. Research found SmolVLA already has `model.embed_image()` that does vision+pixel_shuffle+connector in one call. Just wrap that.

4. **Missed the highest risk.** Plan assumed 'ONNX handles GQA natively.' Research found nobody has actually verified SmolLM2's specific GQA config (15 heads, 5 KV heads) + custom `apply_rope` in ONNX.

5. **Missed critical export gotchas.** Two things that break the export silently:
   - Must use `do_constant_folding=False` (folding corrupts the graph)
   - Must run `patch_gather_indices_once()` post-export (vision encoder produces float Gather indices that ORT rejects)

6. **State token was underspecified.** SmolVLA encodes robot state via `nn.Linear(32, 960)` → exactly 1 token appended to the prefix.

7. **Wrong vlm_kv_dim.** Plan inherited 512 from the stub. Real dim is 960."

---

## Theme: Bug-classification catalog (pipeline vs decomposition)

Reference line 11574 — author's meta-analysis of where bugs came from:

**Pipeline orchestration bugs (our CLI's fault, not lerobot's):**
- State proj random weights
- Base vs fine-tuned VLM loading
- 5D pixel_values — wrong preprocessor
- Missing √hidden scaling
- SigLIP [-1,1] range
- Missing newline on task
- 8D vs 6D state

**Our reimplementation bugs (would not happen with lerobot code):**
- Sinusoidal timestep formula
- RoPE base 10000 vs 100000 (10× error)
- prefix_offset for self-attn
- KV mask for cross-attn

---

## Theme: Fastapi / Pydantic / Modal infrastructure bugs

### Pydantic 2.13 + FastAPI ForwardRef — line 3391, 3509
"Found the bug. Pydantic 2.13 + FastAPI can't resolve `HealthResponse` defined inside `create_app()` — it's a locally-scoped class and the TypeAdapter ForwardRef fails. Moving the classes to module level.

Root cause of earlier hang: `HealthResponse(BaseModel)` was defined *inside* `create_app()`. Pydantic 2.13 can't resolve ForwardRef for locally-scoped classes when FastAPI builds the TypeAdapter — server crashed silently at uvicorn startup. Fix: moved BaseModel subclasses to module level + switched from deprecated `@app.on_event('startup')` to async `lifespan` context manager."

### Modal CUDA version mismatch — line 4445
"`libcublasLt.so.12: cannot open shared object file`. That's the exact root cause — ORT 1.24 needs CUDA 12 libs, but Modal installed CUDA 13 (via torch's `cuda-toolkit-13.0.2`)."

### Install path — missing pkg — line 5329
"Found it — `pip install reflex-vla[serve,gpu]` doesn't include `huggingface_hub`, `transformers`, or `onnxscript` which `reflex export` needs. The README's quickstart command is incomplete."

### cuDNN gap — line 4504
"The ORT fallback fix only got us halfway: fixed `libcublasLt.so.12` (CUDA 12 via torch 2.5.1) but now missing `libcudnn_adv.so.9`. Need to explicitly install `nvidia-cudnn-cu12==9.*` in the Modal image."

---

## Theme: Design principles surfaced / verbalized

### The "single black-box command" principle for export — line 8097
"The long-term fix isn't 'add a flag' — it's making `reflex export <model>` a single black-box command that produces a complete, self-describing deployment artifact. Specifically:
- Auto-detect model type (already works), then run ALL required sub-exporters in one pass (expert + VLM for SmolVLA; just expert for pi0; full stack for GR00T).
- `reflex_config.json` becomes authoritative: records model_id, action_dim, image_size, VLM files, which sub-graphs exist, and which benchmarks this export is valid for.
- Output dir becomes a contract: `reflex serve <dir>`, `reflex validate <dir>`, `reflex eval <dir>` all work without the user ever knowing about the internal ONNX split.
- **No user should ever call `export_vlm_prefix()` directly. It should be internal.**"

### Promote adapter into library — line 8373
"The real problem: `scripts/modal_libero10.py` reimplements the inference pipeline inline (denoising loop, VLM handling, action truncation). That's why fixing bugs requires editing a Modal script instead of editing the library.

**Best long-term fix — promote the adapter into the library:**
1. Build `reflex.runtime.ReflexServer` — one class that owns vision encoder + text embedder + decoder prefill + expert denoising + action post-processing.
2. Add `reflex.runtime.adapters.VlaEvalAdapter` — thin wrapper (~20 LOC) that makes `ReflexServer` implement vla-eval's `PredictModelServer` interface.
3. Make the Modal script a 30-line runner that just: `reflex export → python -m reflex.runtime.adapters.vla_eval → vla-eval run`."

### First-class `reflex eval` — line 8097
"Don't hardcode a LIBERO-specific script. The long-term fix is a first-class `reflex eval <export_dir> <benchmark>` CLI command that:
- Reads action space, image size, and model capability from config.
- Maps model output dims to benchmark action space automatically (SmolVLA 32-dim → LIBERO 7-dim, pi0 32-dim → DROID 8-dim, etc.).
- Supports multiple benchmarks via a plugin pattern: LIBERO, SimplerEnv, ManiSkill. Not one script per benchmark.
- Wraps vla-eval as one backend but can fall back to direct sim runners.
- Publishes standardized JSON: `{benchmark, model, episodes, success_rate, per_task, latency_p50, latency_p99}`."

### Pre-mortem de-risking trick — line 6524
"Single de-risking trick from pre-mortem: **day 1 of VLM prefix**, run cosine-similarity of (PyTorch prefix KV) vs (ONNX prefix KV) on SmolVLA before touching the action head. If that fails, you know before burning a week."

---

## Theme: Semantic sensitivity + determinism tests

### Semantic sensitivity PASS — line 7489
"Semantic sensitivity: PASS. The VLM is working — three different instructions produce meaningfully different prefix_kv tensors:
```
L1 diff (pick cup vs push block):   0.155
L1 diff (pick cup vs open drawer):  0.140
L1 diff (push block vs open drawer): 0.139
```
These are real differences, not noise. The VLM is conditioning the expert based on the instruction."

### Determinism test — line 7541
"| Test | Before fix | After fix |
|---|---|---|
| Determinism (same input → same output) | FAIL (max_diff 34.7) | PASS (max_diff 0.0) |
| Semantic sensitivity (different instructions → different output) | PASS (0.15) | PASS (0.12-0.14) |
| Test suite | 36 passed | 36 passed |"

---

## Theme: The knowledge-capture plan itself

### The extraction plan — line 11618, 11635
"**Sources I can mine:**
1. `.agents/ao/sessions/*.md` (13 files) — pre-summarized narratives
2. `scripts/modal_*.py` docstrings (32 files)
3. `git log --stat src/ scripts/` + key `git log -p`
4. Current session transcript (`ced2c4f1-...jsonl`)
5. Prior transcripts (`799fe46d`, `c7962eb2`, `aca7560b`, `35017f59`) grep for keywords
6. `modal app logs` on currently-reachable apps

**Total: ~9 hours of pure synthesis. No Modal spend, no code changes.**

Grep keywords I'll use on transcripts: `bug`, `fix`, `found`, `root cause`, `post-mortem`, `does not`, `wrong`, `broken`, `HEADLINE`, `DECISION`, `LEARNING`, `verdict`, `blocked on`, `actually`, `turns out`.

**Biggest risk:** redundancy — same insight surfaces in session X, commit Y, and `.md` Z. Dedup as I write rather than during extraction (extract exhaustively first).

**Biggest win:** once `reflex_context/` exists, future sessions START with full knowledge instead of re-discovering."

### Taxonomy for reflex_context/ — line 11597
```
reflex-vla/
├── reflex_context/
│   ├── README.md                          — index
│   ├── 01_architecture/
│   │   ├── smolvla_forward_pass.md       — canonical ref: how real SmolVLA computes actions
│   │   ├── onnx_export_decisions.md      — why we decompose, what's tradable
│   │   └── reflex_server_stack.md        — pipeline diagram
│   ├── 02_bugs_fixed/
│   │   ├── smolvla_inference_bugs.md     — the 12 bugs
│   │   ├── libero_integration.md         — bddl/gym/osmesa/mujoco deps
│   │   └── modal_gotchas.md              — image builds, stdout buffering, detach mode
│   ├── 03_research/
│   │   ├── direct_torch_export.md        — TRT compat findings, opset 23 RMSNorm gap
│   │   ├── checkpoint_formats.md         — smolvla_base vs smolvla_libero state_dict layout
│   │   └── vla_eval_schema.md            — obs schema, camera naming, states vs controller_states
│   ├── 04_iteration_lessons/
│   │   ├── local_vs_modal.md             — ~100× cheaper, do this first next time
│   │   ├── diagnostic_ladder.md          — stage diff → single layer → composition
│   │   └── cost_log.md                   — rough Modal $ per attempt type
│   └── 05_sessions/
│       └── 2026-04-17_libero_hunt.md     — chronological log
```

### The GOALS.yaml snapshot (the weighted-goals approach) — line 6909
13 weighted goals including:
- vlm-prefix-encoder (weight 10) — "exports as separate ONNX with KV-cache; /act returns task-relevant actions, not noise"
- fp16-torch-compile-baseline (weight 9)
- jetson-orin-nano-validation (weight 9)
- reflex-verify-parity (weight 8)
- latency-histograms (weight 7)
- action-safety-clamps (weight 7)
- ros2-bridge (weight 6)
- telemetry-ping (weight 5)
- stripe-license-gating (weight 5)
- reflex-distill (weight 5)

---

## Theme: Flow matching noise accounting

### Shared noise discovery — line 10846
"The cos_sim numbers vary because flow-matching uses fresh random noise each call — BOTH paths get different noise, so we're comparing noisy vs noisy. Need to inject the SAME noise into both paths."

Without shared noise, the cos_sim test gives you essentially meaningless comparisons; this was a critical methodology fix for the PyTorch-vs-ONNX diff harness.

---

## Theme: Research batch methodology

### 7-parallel-agent research pattern — line 5747
| # | Topic | Tools |
|---|---|---|
| 1 | Distillation SOTA for flow-matching VLAs | paper-search, hf-mcp, github |
| 2 | VLM ONNX decomposition (SigLIP+PaliGemma/Qwen3) | github, hf-mcp, paper-search |
| 3 | Jetson access without $499 | WebFetch, crunchbase, github |
| 4 | LeRobot/OpenPI/GR00T issue engagement | github (issues) |
| 5 | Competitive landscape since March | github, crunchbase, arxiv, hf-mcp |
| 6 | Conference / YouTube deployment pain | youtube-transcript, WebSearch |
| 7 | Compute grants active in 2026 | WebFetch, WebSearch, crunchbase |

"Running in parallel — wall-clock ~45-60 min for slowest to finish."

### Council composition — line 6497
- Robotics engineer critic
- Solo-founder critic
- VC critic
- Revenue architect critic
- Pre-mortem critic

Three critics consensus = "settled." Disagreements flagged as "sharpest tensions for you to resolve."

### Batch 1 research findings aggregation — line 6961
"**5 agents returned rich findings. Key headlines:**
- DMPO (arXiv 2601.20701) achieves true one-step generation at 1770 Hz without a teacher
- xVLA (880M, tokenized head) is a new model family Reflex doesn't support
- Jetson deployment is the #1 pain across all repos (3 separate openpi issues, 15+ comments each)
- LiteVLA-Edge already ships GGUF quantization on Jetson at 6.6 Hz in a ROS2 pipeline — direct competitor
- No serious deployment-tool competitor yet (all under 5 stars), but the window is closing"

---

## Theme: Launch narrative / external-facing artifacts

### "95% → 60%" and other phrases to steal — line 5791
> - "The robot is idle while the VLA thinks" (Remi Cadene / HF blog on async inference)
> - "95% in the lab, 60% in the warehouse" (a16z Oliver Hsu)
> - "Memory-bound on Thor" (VLA-Perf paper, arxiv 2602.18397)
> - "3-5 FPS against a need for 20-30 Hz" (arxiv 2510.26742)
> - "The missing infrastructure layer for real-world robotics" (Foxglove's Banisadr at Actuate 2025)
> - "Integration tax"
> - "Not reactive" / "Open-loop while actions available, idle while waiting for next chunk"

### "OpenPI doesn't ship a serve layer" — line 5774
"Use in LeRobot post" — identified as the tagline.

### People to @-mention in LeRobot post — line 5783
- `@jashshah999` — author of #3146 itself, 19 PRs to LeRobot. First ally.
- `@imstevenpmwork` — most-active LeRobot staff (95 PRs)
- `@fracapuano` — commentator-in-chief, engages on deployment threads
- `@not-heavychevy` — torch.compile champion

### Issues to cross-link — line 5783
- `lerobot#1899` (smolvla ONNX) — companion
- `lerobot#2061` (torch.compile for policies)
- `lerobot#2356` (AsyncInference only runs one chunk)
- `openpi#386` (Deploying Pi0 on Jetson Orin errors)
- `openpi#826` (deployment pain)
- `gr00t#517` (GR00T TRT export, NVIDIA's in-progress work)

---

## Theme: Ship decisions (what NOT to build)

### Deliberately NOT doing (line 5453)
- Post anything publicly (user's call)
- Add Phase V (WCET stress test) or Phase VI (full split orchestrator) — feature creep
- Train a distilled VLA — `reflex distill` is v0.2 work
- Bench on T4/L4 — A10G is sufficient as Jetson proxy

### Delete from v0.2 scope (line 6497 council)
- API-key auth (no users to auth)
- Dynamic-batch ONNX + TRT shape profiles (today static-batch + CUDA EP fallback is good enough)
- Audit-log / guard-v2 (2027 revenue, not 2026)

### Deferred (line 6524)
- Hot-reload (blue/green swap) — "Build only after first enterprise explicitly asks"
- Signed audit-log bundle export — same

---

## Theme: What's not in code — verbalized-only insights

These are "I verbalized it but didn't commit it" items:

1. **Per-step cos >0.999 required to survive 10 integration steps** (line 11435) — this is a numerical budget, not captured in any test.
2. **"Task-success is an integration test. ONE subtly wrong operation out of ~500 in the pipeline = 0%"** (line 10630) — a calibration for why this problem is hard.
3. **30+ candidate subtle bugs remain after fixing 12** (line 10630) — effectively a list of suspects that was never written down.
4. **"Local iteration is ~100× cheaper than Modal"** (line 11574) — workflow guidance.
5. **"Reflex cannot win faster inference on cloud GPU"** was stated as L1 in Apr-14 post-mortem (line 4538), then FLIPPED by TRT FP16 discovery (line 4973). Both sides of the pivot should be in the record.
6. **"$99/mo anchors you cheap. Real comp set is Roboflow ($249-999), W&B enterprise ($30-80k)"** (line 6497) — pricing calibration.
7. **"When Physical Intelligence ships pi-1.5 with a first-party Jetson runtime (they will; deployment friction kills their adoption), Reflex's reason to exist evaporates"** (line 6497) — existential risk.
8. **"No user should ever call `export_vlm_prefix()` directly. It should be internal."** (line 8097) — API design constraint.
9. **"Every pitch line that says 'targets Jetson Orin Nano' becomes 'shown to hit X Hz on Jetson Orin Nano'"** (line 8076) — marketing preconditions.
10. **"Actions are firmly in-distribution (within 2σ of training mean/std across all 7 dims)"** (line 10268) — an unanswered question about why the model still gets 0%.

---

## Theme: Vision encoder norm check finding — line 10969

"Stage diff shows:
- Text embedder: `cos=1.0000` ✓
- State projection: `cos=1.0000` ✓
- **Vision encoder: `cos=0.6983`** ❌ (norms differ: torch=1644, onnx=1104)

The vision encoder is broken."

Followed by the root cause (line 11063): wrong AutoModel class. After fix (line 11089): "Vision `cos=1.0000`, `max_abs=1e-4` — perfect match!"

---

## Theme: The eleventh-hour composition-layer finding

### Self-attention layers match 1e-5 — line 11468
"Single SELF-attn layer (layer 0) matches to **1e-5 precision, cos=1.0000**. The bug is somewhere in COMPOSITION — probably cross-attention layers. Let me test a cross layer."

This is the final unresolved frontier at the time the session ended: self-attention numerics are perfect, but some cross-attention or composition operation is still wrong.

---

## Summary — total distinct insights captured in this document

Total distinct insights captured: **~90 substantive findings** organized into 21 themes:

1. Model export architecture decisions (RMSNorm/RoPE decomp rationale, SmolVLA structure, GQA dims, vlm_kv mismatch, pi0 FF, OpenVLA, GR00T)
2. The 12 pipeline bugs (state_proj, base-vs-finetuned VLM, 5D pixels, SigLIP range, state dim, send_state, controller_states, RoPE base, normalizer, text non-det, newline, `or` on np array)
3. Cosine-similarity diagnostic ladder (numbers 0.28 → 0.498 → 0.305 → 0.08 → 0.977 per-step / -0.24 final)
4. LIBERO-10 simulation blockers (bddl, num2words, EGL→osmesa, env.reset hang)
5. Infrastructure-vs-task-success tension (line 10268 state-of-play table)
6. Modal iteration cost & local-first workflow (100× factor)
7. Benchmark results (TRT FP16 2.6-3.3× over compile; reflex bench 86/42/37/18 Hz)
8. Research-derived architectural insights (pi-Flow → DMPO, Cosmos-Reason not Qwen3, PaliGemma2 prior art, Orin NX 16GB not Nano, Jetson pricing)
9. ETARS-inspired 4-file split correction (7 things plan got wrong)
10. Bug-classification catalog (pipeline orchestration vs reimplementation)
11. FastAPI/Pydantic/Modal infra bugs (ForwardRef, CUDA 12 vs 13, cuDNN gap, install path extras)
12. Design principles (single black-box export, library-adapter split, first-class reflex eval)
13. Semantic sensitivity + determinism tests (max_diff 0.0 after seeded-RNG fix)
14. Knowledge-capture methodology (reflex_context taxonomy, extraction plan)
15. Flow matching noise accounting (shared-noise methodology fix)
16. Research batch methodology (7-parallel-agents, council composition)
17. Launch narrative / phrases-to-steal / @-mentions / issues-to-cross-link
18. Ship decisions / what NOT to build
19. "Verbalized-only" insights list (10 items not in code)
20. Vision encoder root cause (AutoModelForImageTextToText)
21. Composition-layer cliffhanger (self-attn 1e-5 but full pipeline 0.08)

**Source line references are embedded throughout.** Each insight has a specific line number in `/Users/romirjain/.claude/projects/-Users-romirjain/ced2c4f1-a341-45bf-ae1b-ba9f6ab0931c.jsonl` that can be consulted for full context.

**Unresolved questions at session end:**
- Why does cos_sim final = -0.24 when all per-stage cos values are high?
- Is cross-attention composition the bug, or is it attention mask / softmax-fp32?
- Would copying lerobot code wholesale (hybrid option 2 from line 11574) close the gap?
- Will the ~2% per-step expert velocity error ever be fixable in ONNX?
- Does multi-camera (using all 3 cameras instead of 1) fix LIBERO, or is it architectural drift?
