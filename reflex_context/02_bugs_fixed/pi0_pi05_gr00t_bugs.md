# pi0, pi0.5, GR00T, OpenVLA — Model-Specific Bugs

Bugs specific to VLA models other than SmolVLA, found while building per-family exporters (pi0, pi0.5, GR00T N1.6, OpenVLA). Each model has its own architectural quirks (AdaRMSNorm variants, GEGLU vs GELU, diffusers-style attention, tokenized action head). These bugs are family-specific and would not apply to the SmolVLA pipeline.

Sources: `sessions_md.md`, `modal_scripts.md`, `git_history.md`, `current_session.md`, `modal_apps_and_pm_docs.md`.

---

## Bug 1: GR00T `action_encoder` / `action_decoder` missing (bias gotcha)

**Symptom:** GR00T serve fails in denoising loop: expert emits velocity in 1024-dim action-token space while input is 1536-dim action tokens. `noisy + velocity * dt` can't cross dimensions. ONNX export itself passes validation; only serve fails.

**Root cause:** The initial GR00T exporter (`68119b7`) exported only the DiT expert (1091.7M params, 32 blocks). But GR00T's actual inference loop wraps the DiT with per-embodiment:
1. `action_encoder`: 3 linears that encode raw actions (128-dim) into the DiT's input space (1536-dim).
2. `action_decoder`: 2 linears that decode DiT velocity output (1024-dim) back to raw action space (128-dim).

Without wrapping these, the denoise loop breaks because dimensions don't match. Additionally, the action_encoder has a specific residual + gating pattern with silu activations that must be replicated exactly. `action_encoder.bias` and related bias tensors had to be correctly sourced from the per-embodiment weight tensor.

**Fix:** Commit `ff9fc3a` ("GR00T full-stack export: raw actions in, raw actions out"):
- Wrap DiT expert with GR00T's `action_encoder` (3 linears) + `action_decoder` (2 linears) pinned to `embodiment_id=0` by default.
- Encoder: `action(128) → W1(1536) → silu → cat(h1, time_emb) → W2(1536) → silu → (h2 + h1) → W3(1536)`. Residual + gating pattern.
- Decoder: `velocity_tokens(1024) → L1(1024) → silu → L2(128)`.
- Weight shape is `[embodiment, in, out]` with leading dim of 32 — slice at `embodiment_id` and transpose for F.linear compat.
- Full stack: 1091.7M + 10M buffers. Export 80.5s, max_diff 3.77e-06. Serve ready 8s. POST /act 2.35s (CPU provider).

**Sources:**
- `git_history.md` commit `ff9fc3a` (full-stack export)
- `git_history.md` commit `9087da3` ("GR00T serve fails in denoising loop because expert emits velocity in 1024-dim action-token space while input is 1536-dim action tokens — `noisy + velocity*dt` can't cross dimensions.")
- `current_session.md` line 4009 ("GR00T's expert emits 1024-dim velocity but input is 1536-dim action tokens. Generic denoising loop can't noisy + velocity * dt across dims.")
- `modal_scripts.md` `modal_probe_gr00t.py` section ("The existing Reflex GR00T exporter skips [action_encoder/action_decoder] — adding them lets reflex serve do full-loop denoising")

---

## Bug 2: pi0 — GELU vs GEGLU in FF layer

**Symptom:** pi0 expert export numerically passes (max_diff small) but downstream actions look wrong when integrating.

**Root cause:** pi0's feed-forward layer was initially assumed to be GEGLU (gated with 2× inner dim: `gate_proj` + `up_proj` multiplicative). Actually, pi0's `ff` layer is a **plain GELU-activated MLP**. The `ff.net.0.proj` outputs `ff_inner` directly, NOT `2*ff_inner`. No gating. Mistaking GEGLU for GELU doubles the expected weight dimensionality and picks the wrong half of the weights.

**Fix:** Reshape exporter to plain GELU path. `ff.net.0.proj(x) → gelu → ff.net.2.proj`. No gating. See `src/reflex/exporters/pi0_exporter.py` and current_session.md line 3846.

```python
# pi0 FF (correct):
h = gelu(ff.net[0].proj(x))
y = ff.net[2].proj(h)
# NOT: h = gelu(gate(x)) * up(x)
```

**Sources:**
- `current_session.md` line 3846 ("Found the bug. The FF layer is NOT GEGLU — it's a plain GELU-activated MLP. `ff.net.0.proj` outputs `ff_inner` directly (not `2*ff_inner`).")
- `git_history.md` commit `45794b0` ("Add pi0 support + fix serve startup (full E2E passes)")

---

## Bug 3: pi0.5 AdaRMSNorm variant — time-conditioned 3-chunk

**Symptom:** pi0.5 export fails with shape mismatch during the RMSNorm application step. Weight dimensions unexpected.

**Root cause:** pi0.5 uses **AdaRMSNorm** (Adaptive RMS Normalization) — time-conditioned RMSNorm that mixes in time embeddings:
```
time_emb → dense → chunk(3) → x * rsqrt(var + eps) * (1 + scale) + shift
```

The dense layer outputs 3 chunks: scale, shift, gate. This is DISTINCT from:
- Standard RMSNorm (no time conditioning).
- pi0's plain RMSNorm (time concatenated with action instead).
- GR00T's AdaLN 2-chunk (only scale + shift, no gate).

Identifying pi0.5 requires detecting both `input_layernorm.dense*` keys (the scale projection) AND `time_mlp*` keys (time conditioning MLP).

**Fix:** Implement `DecomposedAdaRMSNorm` in `src/reflex/decompose.py` with 3-chunk output. Add `ExpertAdaRMSLayer` + `Pi05ExpertStack` classes in `src/reflex/exporters/pi0_exporter.py`. Detect pi0.5 via key-name pattern matching. Verified `lerobot/pi05_base` (3.62B checkpoint, 18 layers, 426.9M expert; pi0 was 314.6M — AdaRMS `dense` layers add ~112M). ONNX max_diff 2.37e-06.

Key-wise: time MLP runs at STACK level (separate from action), UNLIKE pi0 where time is CONCATENATED with action in the suffix encoder.

**Sources:**
- `git_history.md` commit `c0a3a7b` ("pi0.5 support: AdaRMSNorm expert stack + CLI auto-dispatch")
- `modal_scripts.md` `modal_test_pi05.py` section
- `current_session.md` line 2619+ (GQA dim probing pattern)

---

## Bug 4: OpenVLA — no action expert; tokenized head via LM vocab

**Symptom:** Building an OpenVLA exporter "just like pi0" produces nonsensical output. OpenVLA's "action head" isn't flow-matching.

**Root cause:** OpenVLA is NOT a flow-matching VLA. Its action head is literally `argmax(lm_logits[:, -7:]) + 256-bin lookup` on top of the Llama-2 vocab. There is no dedicated action expert to reconstruct. The "action tokens" are just language model tokens at the last 7 positions, decoded through a discrete bin-lookup table.

The full model is standard Llama-2-7B + DINOv2 + SigLIP + projector — which HF's `optimum-onnx` already handles. Building a full Reflex exporter here would duplicate optimum-onnx for zero architectural insight.

**Fix:** Don't build a full OpenVLA exporter. Ship a postprocess HELPER instead:
- `src/reflex/exporters/openvla_exporter.py` — raises `NotImplementedError` with clear message pointing to optimum-onnx.
- `src/reflex/postprocess/openvla.py` — `decode_actions(lm_logits, bin_edges) -> actions`. ~129 LOC.
- `reflex models` shows OpenVLA in yellow ("partial support").
- `reflex export` with openvla dispatches to optimum-onnx under the hood + applies `decode_actions` as a post-hook.

Commit `c00ca82` lands this.

**Sources:**
- `git_history.md` commit `c00ca82` ("OpenVLA: postprocess helper instead of redundant exporter")
- `current_session.md` line 4009 ("OpenVLA's 'action head' is just `argmax(lm_logits[:, -7:]) + bin-lookup` on Llama vocab. No custom expert to reconstruct; full model is standard Llama-2-7B + DINOv2 + SigLIP that optimum-onnx already handles.")
- `current_session.md` bug-classification catalog ("OpenVLA postprocess strategy")

---

## Bug 5: GR00T serve dim mismatch — 1024 velocity vs 1536 tokens

**Symptom:** GR00T export PASSES validation with max_diff=3.5e-05. But `reflex serve` immediately errors on first `/act` with dimension mismatch. Export artifact is "valid ONNX" but not serve-compatible.

**Root cause:** Related to Bug 1 but deserves its own entry because it caught the team twice. The initial GR00T export returned velocity tokens in the action-token space (1024-dim, the DiT's output projection). The denoise loop does `noisy_actions + velocity * dt` — but input is 1536-dim action tokens, not 1024. Cannot add arrays of different shapes.

The root confusion: in training, the DiT output (1024) is fed through `action_decoder` to produce raw actions (128); but during flow-matching INTEGRATION, we're operating in the action-TOKEN space (1536) — not raw actions — because that's where the noise schedule lives. The token space and the DiT output space differ by dim.

**Fix:** Use Bug 1's fix (full-stack wrap). Commit `9087da3` documents the diagnosis; `ff9fc3a` applies the fix. The proper serve path wraps action_encoder on the input side and action_decoder on the output side so the denoise loop operates in raw action space (128-dim) consistently.

**Sources:**
- `git_history.md` commit `9087da3` (diagnosis)
- `git_history.md` commit `ff9fc3a` (fix)
- `current_session.md` line 4009 (same explanation)

---

## Bug 6: GR00T — DiT not GQA; MHA with head_dim=48

**Symptom:** Auto-probing for GR00T attention structure using the SmolVLA / pi0 GQA pattern returns wrong head configuration.

**Root cause:** GR00T is **DiT-based** (diffusion transformer), NOT a decoder-style expert like pi0 or SmolVLA. Key differences:
- **32-block DiT** (not decoder LM layers).
- **32-head MHA, not GQA** — no KV-head split. head_dim=48. hidden=1536.
- **Diffusers attention API**: `to_q / to_k / to_v / to_out.0`, bias=True (unlike lerobot's pattern).
- **AdaLN 2-chunk** (scale + shift, NO gate — unlike pi0.5's AdaRMSNorm 3-chunk).
- **Alternating cross/self-attn**: cross-attn on EVEN blocks (KV from VLM at 2048-dim), self-attn on ODD blocks (1536). Reverse of SmolVLA's even-self/odd-cross.
- **Plain GELU-approx MLP**: `ff.net.0.proj` outputs `ff_inner` directly, NOT `2×`. Same pattern as pi0.
- **Non-affine LayerNorms**: learnable bias but no learnable scale.
- **Output via final AdaLN + proj_out_2** → 1024 tokens.
- **BFloat16 storage**: must cast to fp32 for ONNX export.
- **Uses Cosmos-Reason-2B VLM**, NOT Qwen3 (earlier pi0_exporter research had this wrong).

**Fix:** Build separate `src/reflex/exporters/gr00t_exporter.py` that probes `action_head.model.transformer_blocks.*` keys and constructs a DiT block with diffusers-style API. Commit `68119b7` lands this (+537 LOC). Commit `e887598` adds the stub first for documentation.

Actual dimensions:
```
3.29B params total, 448 DiT keys.
Expert stack: 1091.7M, 32 blocks.
Export: 63.5s, max_diff=2.18e-05.
```

**Sources:**
- `git_history.md` commit `68119b7` ("GR00T N1.6 support: DiT expert with AdaLN + alternating cross/self-attn")
- `git_history.md` commit `e887598` (GR00T arch documented in stub)
- `current_session.md` line 5756 ("GR00T N1.6 uses Cosmos-Reason-2B, NOT Qwen3. I had this wrong in the earlier pi0_exporter research.")

---

## Bug 7: pi0 `head_dim=128` vs SmolVLA `head_dim=64`

**Symptom:** Reusing SmolVLA's `ExpertGQALayer` for pi0 without adjusting head_dim causes silent numerical drift.

**Root cause:** pi0's expert has head_dim=128 (not 64 like SmolVLA). 16 Q heads / 2 KV heads, head_dim=128. Total expert: 314.6M. Prefix: `paligemma_with_expert.gemma_expert.model.*`.

**Fix:** Parameterize head_dim in the shared `ExpertGQALayer`. Pass `head_dim=128` in pi0 exporter, `head_dim=64` in SmolVLA. The shared layer code handles either as long as dims are explicit. Commit `45794b0`.

**Sources:**
- `git_history.md` commit `45794b0` ("pi0 arch = 18 layers, 16Q/2KV GQA, head_dim=128. Prefix: paligemma_with_expert.gemma_expert.model.*. pi0 full export max_diff 3.73e-08.")
- `modal_scripts.md` `modal_test_pi0.py` section ("head_dim=128. Export timeout 600s.")

---

## Bug 8: Adaptive denoising only works on pi0 (not smolvla/pi0.5/gr00t)

**Symptom:** `reflex turbo --strategy adaptive` was pitched as a universal win (early-stop the denoise loop when velocity norm deltas drop below threshold). Modal validation on real VLAs showed it fails for 3 of 4 models.

**Root cause:** The 0.01 velocity-norm-delta threshold was validated only on a synthetic 16-hidden toy model (`scripts/modal_sim_test.py`). Real VLAs have different convergence behaviors:
- **smolvla**: 0/25 triggered. 0% savings. Velocities never converge under the threshold — straight-line trajectories, no flatlining.
- **pi0**: 25/25 triggered. Mean trigger step 4.2. 58.4% savings. action_diff=0.073 (small, OK). Real win.
- **pi0.5**: 3/25 triggered. Step 9.4, 5.6% savings, action_diff=0.762 (LARGE — bad drift even when triggered).
- **gr00t**: 25/25 triggered. Step 3.0, 70% savings, action_diff=0.674 (LARGE — clearly wrong).

**Fix:** Commit `091074c` makes per-model threshold the gate:
- pi0: adaptive active, ship with `--adaptive-steps`.
- smolvla/pi0.5/gr00t: warn with `"adaptive-steps only validated on pi0, results on <model> may degrade"`.
- Docs quote only the pi0 number, not a universal one.
- Per-model threshold tuning deferred to v0.2.

GOALS.yaml `adaptive-denoise-fix` (weight 5) codifies: "Adaptive denoising works on pi0 (supported), is gated behind --experimental for smolvla/pi0.5/gr00t (unsafe)".

**Sources:**
- `git_history.md` theme "Phase IV — adaptive denoising validation on real VLAs"
- `git_history.md` commit `091074c` (verdict + per-model warnings)
- `modal_scripts.md` `modal_verify_adaptive_real.py` section

---

## Bug 9: pi0 normalizer missing — 0% on LIBERO

**Symptom:** pi0 on LIBERO, task success = 0%. Actions look numerically sensible but robot doesn't progress. Same pattern as SmolVLA (see `libero_integration_bugs.md` Bug 18).

**Root cause:** Same pattern as SmolVLA — pi0 / pi0.5 / GR00T checkpoints ship dataset normalizers. Running LIBERO without them means action magnitudes are off; actions are interpreted as scaled wrong by the environment.

**Fix:** Add normalizer support to the vla-eval adapter. Confirmed via `norm=on` log at adapter startup + "4 stats loaded" (action_mean, action_std, state_mean, state_std). Task #24.

Still 0% after this — normalizer alone doesn't close the LIBERO gap; other bugs in the SmolVLA pipeline dominate.

**Sources:**
- `sessions_md.md` line 18 ("pi0 / pi0.5 / GR00T normalizer missing — running LIBERO without the dataset normalizer meant action magnitudes were off; task-success = 0%. Added normalizer support, confirmed `norm=on` in adapter startup log (4 stats loaded). Still 0% — something else wrong.")

---

## Bug 10: GR00T embodiment routing — per-embodiment weights leading dim=32

**Symptom:** Weight shape errors when building DiT encoders/decoders. Attempts to `F.linear(x, W)` fail because `W.shape = [32, in, out]` but `F.linear` expects `[in, out]`.

**Root cause:** GR00T has multi-embodiment support. Its action_encoder and action_decoder weights are stored with leading dim 32 (one slot per embodiment type) — `W.shape = [num_embodiments, in_features, out_features]`. At inference time, pick the embodiment slot based on `embodiment_id` in the request, then transpose for `F.linear` compatibility (PyTorch expects `W.shape = [out_features, in_features]`).

**Fix:** Slice at `embodiment_id` (default 0 for first embodiment in the header table) and transpose:
```python
W_sliced = weights[embodiment_id]  # [in, out]
W_linear = W_sliced.t()  # [out, in] for F.linear
```

Set default `embodiment_id=0` (first embodiment in the per-embodiment table — typically GR1 humanoid or OXE_DROID single-arm). Decoder still available if users want custom. Full-stack commit `ff9fc3a`.

**Sources:**
- `sessions_md.md` line 111 ("GR00T embodiment quirks: Multiple modes per embodiment tag. Humanoids (GR1) use absolute joint positions. Single-arm (OXE_DROID) uses end-effector control. EE-based actions go through a decoder. Per-embodiment weights (leading dim 32) sliced at embodiment_id=0 by default.")
- `git_history.md` commit `ff9fc3a`

---

## Bug 11: GR00T BF16 storage — cast to fp32 for ONNX

**Symptom:** ONNX export fails with dtype mismatch on Gather / MatMul nodes.

**Root cause:** GR00T's checkpoint is stored in BF16. Default ONNX export paths don't always handle BF16 cleanly — some ops expect fp32. Casting errors or silent precision loss.

**Fix:** Cast weights to fp32 before export:
```python
gr00t_state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
```

Then export at fp32 in ONNX; the TRT FP16 engine build re-quantizes as needed.

**Sources:**
- `git_history.md` commit `68119b7` ("BFloat16 storage (cast to fp32 for export)")

---

## Bug 12: pi0.5 time MLP at STACK level, not layer level

**Symptom:** Attempting to apply time conditioning per-layer in pi0.5 produces wrong dimensionality (1×hidden × num_layers instead of just 1×hidden).

**Root cause:** pi0.5's time MLP runs ONCE at the stack level (before layers), producing a time embedding that's threaded into each layer's AdaRMSNorm. UNLIKE pi0, where time is CONCATENATED with action in the suffix encoder (per-step), pi0.5 applies time conditioning globally via the AdaRMSNorm's `dense` projection per-layer.

**Fix:** Time MLP: `time_emb` computed once per forward pass, fed to each layer's AdaRMSNorm. Per-layer: `dense(time_emb) → chunk(3) → scale/shift/gate`. Key structural difference from pi0 that tripped initial pi0-inherited design.

**Sources:**
- `git_history.md` commit `c0a3a7b` ("pi0.5 = time-conditioned RMSNorm (AdaRMSNorm)... Time MLP runs at stack level (separate from action, UNLIKE pi0 where time is concatenated with action).")

---

## Bug 13: pi0 checkpoint memory pressure — delete state_dict between steps

**Symptom:** OOM on A100-40GB when loading pi0 checkpoint (3.5GB) + running the build pipeline.

**Root cause:** pi0 checkpoints are ~3.5GB. Holding the full state_dict in memory while also running model.cuda() and the export pipeline can push past 40GB. GR00T at 6.6GB is even tighter.

**Fix:** Delete `state_dict` between build and CLI export to free memory. `scripts/modal_test_pi0.py`:
```python
state_dict = load_pi0(...)
expert_stack = build_pi0_expert_stack(state_dict, head_dim=128)
del state_dict  # Free ~3.5 GB before the export pass
# ... now run reflex export
```

**Sources:**
- `modal_scripts.md` `modal_test_pi0.py` section ("Deletes state_dict between build and CLI export to free memory (~3.5GB checkpoints are tight on A100-40GB).")

---

## Bug 14: GR00T chunk_size is model-internal, not user-chosen

**Symptom:** Users passing `--chunk-size 50` on GR00T got the wrong internal chunk shape.

**Root cause:** GR00T's expert has a built-in chunk_size that's different from SmolVLA / pi0 family where chunk_size=50 is standard. GR00T's meta returns `chunk_size` as a model-internal value; attempting to override at serve time breaks shape compatibility with the action_encoder input.

**Fix:** `build_gr00t_expert_stack` returns metadata including chunk_size from the model internals. CLI respects this; `reflex serve` reads it from `reflex_config.json`. Don't let user override.

**Sources:**
- `modal_scripts.md` `modal_test_gr00t.py` section ("The meta returned by `build_gr00t_expert_stack` advertises `chunk_size` as a model-internal value, not a user-chosen one.")

---

## Summary Table — VLA model architectural gotchas

| Model | Head attn | head_dim | Layers | Normalization | FF | Time conditioning | VLM-KV dim |
|---|---|---|---|---|---|---|---|
| SmolVLA | 15Q/5KV GQA | 64 | 16 | RMSNorm | GEGLU (gate + up) | In suffix encoder | 320 |
| pi0 | 16Q/2KV GQA | 128 | 18 | RMSNorm | Plain GELU (no gate) | Concatenated w/ action | — |
| pi0.5 | 16Q/2KV GQA | 128 | 18 | AdaRMSNorm (3-chunk) | Plain GELU | Stack-level time MLP | — |
| GR00T N1.6 | 32-head MHA (no GQA) | 48 | 32 blocks DiT | AdaLN (2-chunk) | Plain GELU approx | Diffusers time_embed | 2048 |
| OpenVLA | (standard Llama-2) | — | — | — | — | N/A (tokenized head) | N/A |

| Model | Checkpoint size | Total params | Expert params | VLM | max_diff (export) |
|---|---|---|---|---|---|
| SmolVLA | ~1.8 GB | 450M | 99.8M | SmolVLM2-500M | 4.77e-06 |
| pi0 | 3.5 GB | — | 314.6M | PaliGemma2 | 3.73e-08 |
| pi0.5 | 3.62 GB | — | 426.9M | PaliGemma2 + AdaRMS | 2.37e-06 |
| GR00T N1.6 | 6.6 GB | 3.29B | 1091.7M | Cosmos-Reason-2B | 2.18e-05 |
| OpenVLA | ~14 GB | 7B | — | DINOv2 + SigLIP + Llama-2 | via optimum-onnx |

## Files
- `src/reflex/exporters/pi0_exporter.py` — pi0 + pi0.5 (shared stack)
- `src/reflex/exporters/gr00t_exporter.py` — GR00T full-stack
- `src/reflex/exporters/openvla_exporter.py` — NotImplementedError stub
- `src/reflex/postprocess/openvla.py` — decode_actions helper
- `src/reflex/decompose.py` — DecomposedAdaRMSNorm added for pi0.5
- `scripts/modal_test_pi0.py`, `modal_test_pi05.py`, `modal_test_gr00t.py`, `modal_test_gr00t_full.py`, `modal_probe_gr00t.py`
- `scripts/modal_verify_adaptive_real.py` — adaptive validation

