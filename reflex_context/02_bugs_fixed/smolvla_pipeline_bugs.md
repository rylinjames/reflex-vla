# SmolVLA Pipeline Bugs — the LIBERO Task Success Hunt

A catalog of the bugs discovered and fixed while chasing LIBERO-10 task-success through the SmolVLA ONNX pipeline (vision_encoder + text_embedder + decoder_prefill + expert_stack). Each bug listed once; sources list all raw files where the bug appears. Symptoms, root cause, and fix location are captured for each.

Most of these bugs share a common pattern: they are "silent correctness" bugs — code runs, produces numerically valid tensors, and infrastructure looks green, but the action the model outputs is garbage because one tensor along the path has been subtly wrong all along. Task success is an integration test: ONE subtly wrong operation out of ~500 in the pipeline = 0%.

---

## Bug 1: `state_proj` uses random weights (THE SMOKING GUN)

**Symptom:** Every LIBERO episode failed at step 150 with in-distribution actions that still refuse to progress the task. All infrastructure green. Stage-diff showed `cos=1.0` for state projection — but the values being projected were garbage. Action trajectories looked sensible in aggregate but never actually tracked the state.

**Root cause:** In `src/reflex/runtime/vlm_orchestrator.py::_load_state_encoder`, when `state_proj_weight.npy` didn't exist, the fallback initialized `np.random.randn(...) * 0.02`. The export pipeline never wrote `state_proj_weight.npy`, so EVERY inference ran a random 8→960 projection of the robot state into the VLM prefix. State information became meaningless noise, cross-attention keys never encoded where the robot actually was, and the model could not solve tasks.

```python
# vlm_orchestrator.py _load_state_encoder (before fix):
if state_weight_path.exists():
    self._state_weight = np.load(str(state_weight_path))
else:
    self._state_weight = np.random.randn(...) * 0.02  # <-- smoking gun
```

**Fix:** Save the real `state_proj` weight during export (in `src/reflex/exporters/vlm_prefix_exporter.py`) and load it at runtime with NO fallback to random weights. Hard-fail if weight file missing.

**Sources:**
- `current_session.md` lines 78-86 ("FOUND THE SMOKING GUN BUG")
- `current_session.md` line 11435 "Fix / effect on cos_sim table" row `State_proj fine-tune weights → made things coherent (was garbage)`
- `sessions_md.md` line 27 ("Text-embedding non-determinism" pattern)
- `modal_apps_and_pm_docs.md` — stage-diff runs show state proj cos=1.0 only AFTER the fix

---

## Bug 2: `AutoModel` vs `AutoModelForImageTextToText` wrapper mismatch

**Symptom:** Vision encoder cos_sim suddenly 0.6983 against PyTorch reference. Norms: torch=1644 vs onnx=1104 — completely different magnitudes. State-diff log: `[vlm-weights] load: 488 missing, 345 unexpected` versus healthy `144 missing, 0 unexpected`.

**Root cause:** The checkpoint's VLM was stored under `AutoModelForImageTextToText` wrapper (keys start with `model.connector.*`, `model.text_model.*`), but our exporter was calling `AutoModel.from_pretrained(...)` which returns the inner model (keys are `connector.*`, `text_model.*`). The `model.` prefix mismatch meant 488 vision/text keys went missing and 345 SmolVLA-prefixed keys were unexpected. Vision loaded random-initialized weights for half its layers.

**Fix:** Load `AutoModelForImageTextToText.from_pretrained(...)` and unwrap the `.model` attribute before using it. Log `[vlm-weights] unwrapped ForConditionalGeneration -> inner model` to confirm. After fix: Vision cos=1.0000, max_abs=1e-4.

**Sources:**
- `current_session.md` line 11063 ("Found it exactly. Checkpoint's VLM was `AutoModelForImageTextToText`...")
- `current_session.md` line 11089 (post-fix verification)
- `modal_apps_and_pm_docs.md` ap-uKaH8uEPu... (488/345 regression), ap-YrnHF0Wg... (144/0 fix)

---

## Bug 3: Fine-tuned VLM weights never overlaid

**Symptom:** Stage-diff runs showing `"Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)"` in logs. Vision/text/state cos=1.0 but cross-attention drift.

**Root cause:** The VLM prefix exporter uses the BASE SmolVLM2-500M weights by default. Even after loading the SmolVLA policy checkpoint, the fine-tuned VLM layers (which differ from the base) are never overlaid on top of the base VLM during export. `reflex export` for SmolVLA instructs the user: "VLM uses base SmolVLM2-500M weights. Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)."

**Fix:** Known gap; partially addressed. Weight overlay deferred to v0.3. Workaround is to load the VLM weights from the policy checkpoint's safetensors and overwrite the base VLM's state dict before export.

**Sources:**
- `git_history.md` theme "CLI unification" commit `fdd9bb3` ("VLM uses base SmolVLM2-500M weights. Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item)")
- `modal_apps_and_pm_docs.md` stage-diff runs ap-YrnHF0Wg, ap-oXrqhfnQ, ap-2tNsuBRS all carry "v0.3 item" note

---

## Bug 4: 5D `pixel_values` from SmolVLM `AutoProcessor`

**Symptom:** Every VLM call silently failed → zero VLM conditioning → every previous LIBERO task-success data point was without real VLM conditioning. Actions were purely a function of noise + random init state.

**Root cause:** SmolVLM's `AutoProcessor` returns `pixel_values` with shape `[B, N, 3, H, W]` (5D, with a `num_images` dim). Our exported `vision_encoder.onnx` expected 4D `[B, 3, H, W]`. The orchestrator caught the shape exception and fell back to dummy zeros WITHOUT crashing loudly. Every inference path ran with zero VLM conditioning.

**Fix:** Squeeze the `num_images` dim before passing to the ONNX session. Code path in `src/reflex/runtime/vlm_orchestrator.py`. ALSO: the exporter should hard-fail (not silently succeed) when shape is wrong.

**Sources:**
- `current_session.md` lines 94-96 ("The bug is confirmed: 5D pixel_values dropped to ONNX which wanted 4D")
- `current_session.md` line 9605/9615 (root cause)
- `sessions_md.md` line 88 ("5D fix" referenced in list of LIBERO-0%-after-fixes)

---

## Bug 5: Missing √hidden scaling on image + text embeds

**Symptom:** Per-layer VLM kv cos still not 1.0 after other fixes; decoder output highly sensitive to input scaling (cos=0.51 between scaled vs unscaled).

**Root cause:** SmolVLM2's decoder expects the vision and text embeddings to be pre-scaled by `√hidden_size` (standard transformer embedding scale). Our pipeline was feeding unscaled embeddings to the decoder. The real model does this scaling inside `embed_prefix`.

**Fix:** Multiply image_embeds and text_embeds by `sqrt(hidden_size)` before assembling the prefix and feeding decoder_prefill. Discovered only when per-stage cos_sim converged on everything but the decoder pass-through.

**Sources:**
- `current_session.md` line 11248 ("cos went NEGATIVE (-0.21). My √hidden scaling may have made things worse, not better.")
- `current_session.md` line 11310 ("Decoder output is HIGHLY sensitive to input scaling")
- `current_session.md` line 11574 table "Missing √hidden scaling — No — we skipped the step — lerobot does it"

---

## Bug 6: SigLIP expects `[-1, 1]` pixel range

**Symptom:** Vision encoder cos_sim stuck even after unwrapping `ForConditionalGeneration` and loading real weights.

**Root cause:** SigLIP vision towers expect pixel values normalized to `[-1, 1]`, not `[0, 1]`. Our preprocessing divided by 255 and fed the resulting `[0, 1]` tensor directly. Vision features were systematically biased.

**Fix:** After dividing by 255, apply `img_f = img_f * 2.0 - 1.0` to map into `[-1, 1]`. Done in the stage-diff script (`scripts/modal_stage_diff.py`) before passing to `vision_encoder.onnx`.

**Sources:**
- `modal_scripts.md` `modal_stage_diff.py` section ("Vision normalization: img_f = img_f * 2.0 - 1.0 — SigLIP expects [-1, 1]")
- `current_session.md` line 10835 ("Image now in [-1, 1] range for SigLIP (was [0, 1])")
- `current_session.md` line 11574 table ("SigLIP [-1,1] range — No — lerobot does it")

---

## Bug 7: Missing newline on task string

**Symptom:** Text embedder cos=1.0 but downstream tokens subtly different; per-layer kv cos drifts.

**Root cause:** lerobot's tokenizer automatically appends a trailing newline to the task instruction string. Our pipeline fed the raw string without the newline. Different token IDs = different embeddings = different decoder KV.

**Fix:** Append `"\n"` to the task string before tokenization in the orchestrator. This matches lerobot's `SmolVLAPolicy.sample_actions` path.

**Sources:**
- `current_session.md` line 11574 table ("Missing newline on task — No — lerobot does it")
- `current_session.md` bug catalog line 131 ("Missing newline on task. Referenced in the '12 bugs' table. Task input needed a trailing newline that lerobot includes automatically.")

---

## Bug 8: State is 8D, not 6D

**Symptom:** Action trajectories looked reasonable in aggregate but LIBERO episodes failed at step 150.

**Root cause:** LIBERO's state vector for SmolVLA-LIBERO is 8-dimensional: `eef_pos(3) + axis_angle(3) + gripper_qpos(2)`. The preprocessor's normalizer stats are shape `(8,)`. Our pipeline was truncating or hardcoding 6D state. Model was fed zeros for the gripper state → could never decide when to grip.

**Fix:** Feed full 8D state into the pipeline. `scripts/modal_pytorch_vs_onnx.py` explicitly comments: "LeRobot's LIBERO dataset uses 8D state: eef_pos(3) + axis_angle(3) + gripper_qpos(2). The preprocessor's normalizer stats are shape (8,) so we need 8D here." State-dim truncation to 6 is correct when TARGET embodiment is 6D, but input state stays 8D.

**Sources:**
- `modal_scripts.md` `modal_pytorch_vs_onnx.py` section
- `current_session.md` line 10679, 9486 ("state dim 8 vs 6 truncation")
- `current_session.md` line 11574 table ("8D vs 6D state — No — lerobot does it")

---

## Bug 9: Sinusoidal timestep — missing 2π factor, wrong `[cos, sin]` order

**Symptom:** Per-layer kv matches, per-step expert velocity cos=0.977 (20% norm error), but 10-step flow accumulation catastrophically drifts to cos=-0.24.

**Root cause:** Two bugs in our `create_sinusoidal_pos_embedding(time_val, dimension, min_period=4e-3, max_period=4.0)` implementation:
1. Missing a factor of `2π` inside the sin/cos argument.
2. The `[cos, sin]` concatenation order was reversed from lerobot's canonical `[sin, cos]` (or vice versa depending on which side of the fix).

Either error alone produces ~2% per-step velocity error. Compounding over 10 Euler steps destroys trajectory integration.

**Fix:** Copy lerobot's timestep encoder verbatim (from `lerobot.common.policies.smolvla.modeling_smolvla`). Add the `2π` factor and match the `[sin, cos]` concat order.

**Sources:**
- `current_session.md` line 11524 ("missing `2π` in sinusoidal")
- `current_session.md` line 11435 table ("Expert step velocity — cos=0.977 per step (~20% norm error)")
- `current_session.md` line 11574 table ("Sinusoidal timestep — Our reimplementation bug — lerobot's is correct")

---

## Bug 10: RoPE base `10000` vs `100000`

**Symptom:** Per-layer vlm_k cos consistently off by 0.1-0.2 across layers. Everything else matches.

**Root cause:** Real SmolLM2 uses `rope_theta=100000`, but our `_DecomposedRoPE` defaulted to `base=10000.0`. **10× different RoPE base frequencies.** The position-encoded query and key vectors rotate at completely different rates. Cross-layer error compounds.

**Fix:** Set `base=100000` in the RoPE constructor. Commit: RoPE base 10k→100k. Note: cos_sim effect was "negligible" at final end-to-end level because other bugs dominated, but it was a real bug.

**Sources:**
- `current_session.md` line 11424 ("Another bug! Real SmolLM2 uses `rope_theta=100000`, but our `_DecomposedRoPE` defaults to `base=10000.0`. **10× different RoPE base frequencies.**")
- `sessions_md.md` line 88 ("RoPE fix" referenced as one of many applied before 0%)
- `current_session.md` line 11435 table ("RoPE base 10k→100k — negligible")
- `current_session.md` line 11574 table ("RoPE base 10000 vs 100000 — Our reimplementation bug — lerobot's is correct")

---

## Bug 11: Self-attention `position_ids` offset by `prefix_len`

**Symptom:** Self-attention layers (even indices) cos=1.0 when tested in isolation at layer 0, but in-pipeline drift accumulates.

**Root cause:** The expert's self-attention runs over action tokens that are APPENDED to the VLM prefix in the decoder's position space. Our code fed `position_ids=[0..49]` for the chunk of 50 action tokens. The real model feeds `position_ids=[prefix_len..prefix_len+49]` — offset by the prefix length (vision + text + state tokens = ~100). The RoPE rotation was applied at the wrong position indices.

**Fix:** Compute `prefix_len` from the assembled prefix and pass `position_ids=torch.arange(chunk_size) + prefix_len` to the self-attention RoPE. Fixed in `src/reflex/exporters/smolvla_exporter.py`.

**Sources:**
- `current_session.md` line 11574 table ("prefix_offset for self-attn — Our reimplementation bug — lerobot's is correct")
- `current_session.md` line 11435 table ("prefix_offset — negligible" after individual fix)

---

## Bug 12: KV mask for cross-attention on padded prefix

**Symptom:** Cross-attention layers (odd indices) produce non-zero attention weights on padded positions of the VLM prefix. Action prediction mildly wrong.

**Root cause:** The assembled prefix is padded to a max length for batched decoder_prefill. Cross-attention must MASK those padding positions so attention weights don't leak into the padding. Our code passed an all-ones KV mask; real model computes an attention mask based on `inputs_embeds` validity.

**Fix:** Compute the KV mask from the VLM input attention_mask and propagate it through the cross-attention path. Added to cross-attention layer in `smolvla_exporter.py::ExpertGQALayer`.

**Sources:**
- `current_session.md` line 11524 ("missing pad mask")
- `current_session.md` line 11574 table ("KV mask for cross-attn — Our reimplementation bug — lerobot's is correct")

---

## Bug 13: `obs.get(a) or obs.get(b)` on numpy arrays → `ValueError: ambiguous truth`

**Symptom:** Adapter crashed after 3 `env.step()` calls with `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`.

**Root cause:** In the vla-eval adapter (`src/reflex/runtime/adapters/vla_eval.py`), the observation parsing did `state = obs.get("states") or obs.get("controller_states")` to get state under either key. When `obs["states"]` is a numpy array (even a valid one), Python's `or` evaluates it as bool which raises on multi-element arrays.

**Fix:** Explicit existence check: `state = obs["states"] if "states" in obs else obs.get("controller_states")`. Or: `state = obs.get("states"); if state is None: state = obs.get("controller_states")`.

**Sources:**
- `current_session.md` line 10221 ("Found the bug: `obs.get(k) or obs.get(other)` on numpy arrays raises ValueError: truth value is ambiguous.")

---

## Bug 14: `num2words` missing

**Symptom:** SmolVLM processor import fails with `ModuleNotFoundError: No module named 'num2words'`. VLM pipeline hard-crashes at adapter startup.

**Root cause:** SmolVLM's tokenizer/processor uses the `num2words` library internally (for converting numbers in text to words). The Modal image didn't include it. Missing from any transitive dep.

**Fix:** Add `num2words` to the Modal image `pip_install(...)` list in `scripts/modal_libero10.py` and in the `[eval]` extra of `pyproject.toml`.

**Sources:**
- `current_session.md` line 9574 ("Missing `num2words` dep for SmolVLM processor. Trivial fix.")

---

## Bug 15: `bddl` package version mismatch

**Symptom:** Every LIBERO episode failed at import with `ModuleNotFoundError: No module named 'bddl'`. All inference/export/server plumbing works — sim side missing a dependency.

**Root cause:** LIBERO uses Stanford's BDDL (Boolean Domain Definition Language) for task specifications. `bddl` is on PyPI, but LIBERO's `setup.py` has empty `install_requires=[]`, so none of its `requirements.txt` deps get installed by default. Even when installed, wrong `bddl` versions silently fail at parse time.

**Fix:** Pin `bddl==1.0.1` explicitly in the Modal image. Do NOT install LIBERO's full `requirements.txt` (would downgrade transformers/numpy/etc. and nuke the ONNX export stack). Install only the runtime-required deps with flexible versions.

**Sources:**
- `modal_scripts.md` `modal_libero10.py` section ("bddl==1.0.1 (pinned — LIBERO imports bddl.parsing)")
- `current_session.md` lines 8596-8776 ("bddl missing... LIBERO's setup.py has empty install_requires=[]")
- `sessions_md.md` line 21 ("Run #4 `bddl`/`future`/`gym` cascade")

---

## Bug 16: Expert head_dim derivation (ended up 64, not 48)

**Symptom:** Expert layer export succeeded numerically but attention dimensions looked "off" during manual inspection.

**Root cause:** Initial expert reconstruction tried `head_dim = vlm_hidden // num_attention_heads`, which from the base SmolVLM2 config gave 48 (960/20). Turns out SmolVLA uses DIFFERENT num_heads for the expert (15 Q heads on 720 hidden), giving `head_dim = 720/15 = 48`. But the actual expert has `head_dim=64` with 15 Q heads, `q_proj` shape [960, 720]. The derivation from config was wrong; the real answer came from probing `q_proj.weight.shape`.

**Fix:** Detect head counts AND head_dim from `q_proj` / `k_proj` shapes directly. `expert_hidden = q_shape[1]`; `head_dim = q_shape[0] / num_heads_detected_elsewhere`. Explicit probe, not derivation from base VLM config.

**Sources:**
- `modal_scripts.md` `modal_expert_export.py` section ("Derives from HuggingFaceTB/SmolVLM2-500M-Video-Instruct config: `head_dim = vlm_hidden // num_attention_heads`, ... Detects actual head counts from `q_proj` / `k_proj` shapes.")
- `git_history.md` commit `74d24c3` ("Expert arch = 720 hidden, 15 Q heads, 5 KV heads (GQA 3:1), head_dim=64, intermediate=2048, 16 layers")
- `current_session.md` line 2660 ("Expert layer architecture: 720 hidden, 15 Q heads, 5 KV heads (GQA), 64 head_dim")

---

## Bug 17: Expert velocity sign flips on dims 2 and 6 (gripper)

**Symptom:** LIBERO task-success 0% even after all other fixes. Action trajectories in-distribution, but gripper fails: never grips at the right moment. Per-step cos ≈ 0.98, but 20% magnitude error + sign inversions on dims 2 and 6.

**Root cause:** Partially tracked. Expert emits velocity with sign flipped for dims 2 (vertical axis-angle component) and 6 (gripper qpos). Likely tied to mismatched action normalization or the shared-noise schedule direction. Trajectory integration destroyed task-success because the gripper opens when it should close.

**Fix:** Open. Candidates include:
- Expert layer mapping is off (we apply k_proj to `hidden_states[i]` = input of layer i; real model may apply post-some-operation)
- DecomposedRMSNorm vs real RMSNorm numerics
- `F.silu(gate_proj(x)) * up_proj(x)` ordering vs real
- Attention softmax upcast to fp32
- Per-embodiment encoder/decoder routing

**Sources:**
- `sessions_md.md` line 29 ("Expert's velocity sign flips on dims 2 and 6 (gripper). 20% magnitude error plus sign inversions → trajectory integration destroyed task-success.")
- `sessions_md.md` line 85 ("Velocity has 20% magnitude error + sign flips on dims 2 and 6")
- `current_session.md` line 11175 (4 hypothesis list)
- `current_session.md` line 11435 table (expert velocity ~2% per-step)

---

## Bug 18: `vlm_kv` dim 320 vs 960 mismatch

**Symptom:** Validate backend (`_onnx_backend.py`) fed zeros of shape `[batch, prefix_len, 960]` to the expert ONNX's `vlm_kv` input — got shape mismatch error at `Run()`.

**Root cause:** The VLM hidden states are 960-dim, but the expert's cross-attention KV projections expect 320-dim input (= 5 KV heads × 64 head_dim). The expert ONNX has `vlm_kv` input with shape `[batch, prefix_len, 320]`. Our validate stub inherited 512 and then 960 from the VLM hidden_size. Real dim: 320.

**Fix:** Read the actual `vlm_kv` dim from the ONNX input shape (`expert.vlm_kv_dim`), NOT the VLM hidden_size — they differ (320 vs 960 for SmolVLA). Fixed in `src/reflex/_onnx_backend.py` and `_pytorch_backend.py`. See commit `7ed41aa`.

**Sources:**
- `git_history.md` commit `7ed41aa` ("fix: validate backends handle vlm_kv input (dim 320 from ONNX shape, not 960)")
- `current_session.md` line 7195 ("Our v1 stub uses vlm_kv_dim=512. The real dimension is 960. SmolLM2 hidden_size is 960. The expert's cross-attention projects 960→720 internally. The entire I/O contract in the stub is wrong-shaped...")
- `current_session.md` line 7419 ("The expert ONNX expects `vlm_kv` dim 320 (cross-attention size), not 960 (VLM hidden)")
- `current_session.md` line 2841 ("vlm_kv_dim=320 = 5 KV heads x 64 head_dim")
- `sessions_md.md` line 17 ("Cross-attention K/V dim mismatch (320 vs 720)")

---

## Bug 19: Per-layer `vlm_k` / `vlm_v` vs single tensor

**Symptom:** Per-layer kv cos ≥ 0.91 (layer 0 v is 0.91, others 0.99+), yet final action cos=0.08. Expert stack ONNX diverging from PyTorch even though inputs match.

**Root cause:** The original expert ONNX export took a SINGLE `vlm_kv` tensor `[batch, prefix_len, vlm_kv_dim]` and broadcast it across all cross-attention layers. The real model has PER-LAYER `vlm_k` and `vlm_v` tensors that differ across layers because they are the OUTPUT of the decoder's hidden states at each layer (after RoPE, after layer_norm). Single tensor = layer-0 KV reused for all 16 layers.

**Fix:** Export `decoder_prefill.onnx` producing per-layer `vlm_k_0..vlm_k_15` and `vlm_v_0..vlm_v_15` outputs. Expert ONNX takes `vlm_k` and `vlm_v` as `[num_layers, batch, prefix_len, kv_dim]` tensors. Orchestrator wires per-layer. Task #25 "Per-layer vlm_kv ONNX export" still in progress; stage-diff runs (`scripts/modal_stage_diff.py`) are the driver.

**Sources:**
- `current_session.md` line 11132 ("All per-layer k/v match to cos≥0.91 (layer 0 v is 0.91, others 0.99+). Yet final action cos=0.08. That means the expert_stack.onnx is diverging or there's something interleaved wrong.")
- `sessions_md.md` line 125 ("Per-layer vlm_kv ONNX export: task #25, in progress. Means exporting each transformer layer's K/V separately so the VLM prefix KV-cache matches the training path.")
- `modal_apps_and_pm_docs.md` stage-diff logs ("layer_0_v cos=+0.9117 outlier persists → this is the reproducible structural discrepancy")
- `modal_scripts.md` `modal_stage_diff.py` section (per-layer KV validation)

---

## Bug 20: Cross-attention layer renormalization of `position_ids`

**Symptom:** Single SELF-attn layer (layer 0) matches to 1e-5 precision, cos=1.0000, but COMPOSITION of multiple layers diverges. Probably cross-attention layers.

**Root cause:** Open at session close. The cross-attention layers receive `position_ids` that may need re-normalization per layer (some models renormalize at each cross-attn based on the VLM prefix's own position encoding). Our composition may be using the wrong position_ids pattern across layers.

**Fix:** Open. Likely fix is to copy lerobot's `forward_cross_attn_layer` verbatim into `reflex.runtime.smolvla_native` and swap only `RMSNorm → DecomposedRMSNorm` for TRT compat.

**Sources:**
- `current_session.md` line 11468 ("Single SELF-attn layer (layer 0) matches to 1e-5 precision, cos=1.0000. The bug is somewhere in COMPOSITION — probably cross-attention layers. Let me test a cross layer.")
- `current_session.md` line 11524 ("Copy lerobot's `SmolVLAPolicy.sample_actions` + `embed_prefix` + `embed_suffix` + `forward_cross_attn_layer` into `reflex/runtime/smolvla_native.py`. Swap only `RMSNorm → DecomposedRMSNorm` for TRT compat.")
- `modal_apps_and_pm_docs.md` stage-diff runs — layer_0_v cos=0.9117 persists, reproducible

---

## Cross-cutting observations

The 12 pipeline bugs classify naturally into two buckets (from `current_session.md` line 11574):

**Pipeline orchestration bugs (our CLI's fault, NOT in lerobot):**
- Bug 1 state_proj random weights
- Bug 2 AutoModel vs AutoModelForImageTextToText
- Bug 3 Fine-tuned VLM not overlaid
- Bug 4 5D pixel_values
- Bug 5 Missing √hidden scaling
- Bug 6 SigLIP [-1, 1] range
- Bug 7 Missing newline on task
- Bug 8 8D vs 6D state

**Our reimplementation bugs (would NOT happen with lerobot code):**
- Bug 9 Sinusoidal timestep
- Bug 10 RoPE base 10k vs 100k
- Bug 11 self-attn position_ids offset
- Bug 12 KV mask for cross-attn

**"8 of 12 bugs disappear if we use lerobot's actual code."** — the eventual correct remediation path is the hybrid pivot (current_session.md line 11574): copy lerobot's modeling files into `reflex/exporters/smolvla_native.py` verbatim, override only the ops that don't export (RMSNorm), let torch.onnx.export with `dynamo=True` handle everything else.

## The cos_sim progression — numerical fingerprint of the hunt

| Stage | cos_sim | Cause addressed |
|---|---|---|
| First baseline | 0.28 | Nothing; just noisy noise comparison |
| State_proj real weights + manual preprocessing | 0.498 | Bug 1 partial |
| Shared noise injected | 0.305 | Methodology fix (not a bug) |
| AutoModelForImageTextToText + unwrap | 0.08 | Bug 2; vision now perfect but composition broke |
| √hidden scaling | -0.21 | Bug 5, but overcorrected |
| Per-layer kv + RoPE base | -0.27 → -0.24 | Bugs 10, 11 |
| Per-step expert velocity | 0.977 per step | Bug 9 partial |
| 10-step flow accumulation | -0.24 final | Bug 17 sign flips |

**Required budget:** per-step cos > 0.999 to survive 10 Euler integration steps. We never got there in-session.

## Files
- `src/reflex/runtime/vlm_orchestrator.py` — state encoder, VLM prefix assembly, bugs 1, 4
- `src/reflex/exporters/vlm_prefix_exporter.py` — VLM components export, bugs 2, 3, 5, 7
- `src/reflex/exporters/vlm_components.py` — VisionEncoderForONNX, bug 6
- `src/reflex/exporters/smolvla_exporter.py` — expert stack, bugs 9, 10, 11, 12, 17, 19, 20
- `src/reflex/runtime/adapters/vla_eval.py` — LIBERO adapter, bugs 13, 14, 8
- `src/reflex/_onnx_backend.py`, `_pytorch_backend.py` — bug 18
- `scripts/modal_pytorch_vs_onnx.py` — the decisive cos_sim test
- `scripts/modal_stage_diff.py` — per-stage localization

