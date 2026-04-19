# 2026-04-19 — All Four Open VLAs at Machine Precision

**Session theme:** close the "every major open VLA verified at cos=1.0" story. Enter the session with SmolVLA + pi0 verified at num_steps=1, pi0 at num_steps=10 stuck at cos=0.977 (the `create_causal_mask → None` shim loses prefix-pad info on PaliGemma). Exit the session with all four — SmolVLA, pi0, pi0.5, GR00T N1.6 — at cos=+1.000000 machine precision, every artifact reproducible via a single `modal run` command.

This was the technical-work finale of v0.2. Next step is tagging a new release (v0.2.1 or v0.3.0) and launching.

---

## Arc of the day

Three milestones, in chronological order:

1. **Fix #1: pi0 num_steps=10 cos=1.0** (morning, ~3h including a false-start). The hardest bug of v0.2. Commit `bac658a`.
2. **Fix #2: pi0.5 support** (~1h, ported the fix #1 stack). Commit `c604962`.
3. **GR00T N1.6 monolithic + parity** (~1h). Commit `4090e76`.

Six commits in total when counting doc updates. All pushed to `main` on `github.com/rylinjames/reflex-vla`.

---

## Fix #1: pi0 cos=1.0 at num_steps=10 — the three-patch stack

### Why it was stuck

The v0.2 pi0 export used a `create_causal_mask → None` shim to dodge a 835→886 shape-expand bug under `torch.export` FakeTensor tracing. The shim worked for SmolVLA (SmolLM2 attention doesn't need the mask for correctness) but silently corrupted pi0's PaliGemma attention (prefix-pad masking got skipped). Result: cos=0.977, max_abs≈1.31e-01 — an **approximation** we shipped as a known limitation.

Enter today: fix it, or live with the disclaimer.

### What was tried (in order)

**Attempt 1: 4D-mask dispatch `ccm_shim`.** Replace `create_causal_mask → None` with one that checks if a 4D mask was passed and returns it; otherwise None. The theory: let the caller pass an explicit mask so we don't lose prefix-pad info. Result: cos=0.977 unchanged. Debug revealed the mask arriving at attention was `[1,1,51,835]` not `[1,1,51,886]` — the cat of prefix+suffix masks was losing the suffix dim under FakeTensor tracing.

**Attempt 2: F.pad instead of `torch.cat` for the mask.** Replace `torch.cat([prefix_mask, suffix_mask], dim=2)` with two `F.pad` calls + logical AND. `F.pad` has concrete output size per step, so the mask survives tracing. Result: export ran cleanly, cos still 0.977. Something else was off.

**Attempt 3: diagnostic on `past_kv.get_seq_length()`.** Added a print inside the patched denoise_step. Discovery: `past_kv.get_seq_length()` grows 784 → 835 → 886 across the 10 Euler iterations — cache was mutating across unrolled iterations even with `use_cache=False`. Root cause identified.

**Attempt 4 (the fix): freeze `DynamicLayer.update` during denoise phase.** Wrap `transformers.cache_utils.DynamicLayer.update` so that when a denoise-phase flag is set, it returns `cat(past, current)` without mutating `self.keys`/`self.values`. The prefix forward runs with the flag off (builds the cache normally); the Euler loop runs with the flag on (cache stays at prefix size). This matches canonical eager PyTorch semantics (`use_cache=False` doesn't mutate). Result: **cos=+1.000000, first-action max_abs=2.09e-07**.

### The three patches (they only work together)

1. **F.pad mask** — replaces `torch.cat([prefix_mask, suffix_mask], dim=2)` inside `denoise_step`. Cat loses the suffix dim under FakeTensor; pad+AND has concrete output size.
2. **Frozen `DynamicLayer.update`** — prevents cache growth across the unrolled 10 iterations. Without this, iteration N's attention sees a past of size `(orig + N·suffix_len)`.
3. **`past_kv.get_seq_length()` for mask assembly** — not `prefix_pad_masks.shape[1]`. Those diverge under tracing; using the wrong one → mask's K dim is smaller than attention scores' K → 835→886 broadcast failure.

Details: [02_bugs_fixed/pi0_num_steps_10_three_patch_stack.md](../02_bugs_fixed/pi0_num_steps_10_three_patch_stack.md).

### Verified numbers (Modal A10G, 2026-04-19, commit `bac658a`)

```
====== PARITY (num_steps=10) ======
  first-action max_abs: 2.0862e-07
  first-action cos:     +1.000000
  full-chunk max_abs:   4.1723e-07
  full-chunk cos:       +1.000000
  VERDICT: PASS
```

Reproducer: `modal run scripts/modal_pi0_monolithic_export.py --parity --num-steps 10`.

---

## Fix #2: pi0.5 port

Mirror the pi0 export for `lerobot/pi05_base`. pi0.5 differs from pi0 in two places that matter to the exporter:

1. **No `state` input.** pi0.5 tokenizes robot state into language tokens upstream; `PI05Pytorch.denoise_step` is `(prefix_pad_masks, past_key_values, x_t, timestep)` — 4 args, not 5.
2. **Wrapper forward** — no `state` tensor input. The wrapper calls `sample_actions(images, img_masks, lang_tokens, lang_masks, noise=noise, num_steps=10)`.

The 3-patch stack ported verbatim (same F.pad, same frozen `DynamicLayer.update`, same `get_seq_length`) — it's a PaliGemma property, not a pi0-specific one. First-try PASS.

### Verified numbers (Modal A10G, 2026-04-19, commit `c604962`)

```
====== PARITY (num_steps=10) ======
  first-action max_abs: 2.3842e-07
  first-action cos:     +1.000000
  full-chunk max_abs:   6.5565e-07
  full-chunk cos:       +1.000000
  VERDICT: PASS
```

Reproducer: `modal run scripts/modal_pi05_monolithic_export.py --parity --num-steps 10`. ONNX size: 13GB on disk, 12996.0MB.

---

## GR00T N1.6 — a different architecture, a cleaner path

GR00T is not flow matching. It's DDPM-style discrete-timestep diffusion over a DiT (Diffusion Transformer) action head with AdaLN conditioning. Canonical num_steps=4, not 10. No DynamicCache, no PaliGemma attention, no prefix-mask surgery.

### What's exported

Not the full denoise loop — just the **per-step velocity function**: `(noisy_actions, timestep, position_ids) → velocity`. The loop lives in `reflex serve` (or in a Python host-loop for testing). This matches Isaac-GR00T's own inference pattern: the denoise loop is part of the runtime, not the model graph.

Why this structure works for the exporter:
- GR00T's DiT graph is simpler than pi0's decoder-only flow matching — plain `torch.onnx.export(opset=19)` traces it cleanly. No torch.export patches needed.
- The existing `src/reflex/exporters/gr00t_exporter.py` (757 lines, shipped pre-session) already had `build_gr00t_full_stack` producing a validated nn.Module. This session just wrapped it in a Modal script and measured parity.
- VLM conditioning is zero-stubbed (same convention as pi0/SmolVLA's prefix=None). Full multimodal control needs Eagle VLM export — deferred to v0.3.

### Verified numbers (Modal A100-40GB, 2026-04-19, commit `4090e76`)

Two separate measurements:

**Single-step:** ONNX(noise, timestep, position_ids) vs `GR00TFullStack.forward` with identical inputs.

```
====== PARITY ======
  first-action max_abs: 8.3447e-07
  first-action cos:     +1.000000
  full-chunk max_abs:   3.6955e-06
  full-chunk cos:       +1.000000
  VERDICT: PASS
```

**4-step denoise loop (end-to-end):** Python loop over ONNX vs same loop over PyTorch. Uniform schedule t∈[1→0], shared seeded initial noise, same dt.

```
====== LOOP PARITY (num_steps=4) ======
  first-action max_abs: 4.7684e-07
  first-action cos:     +1.000000
  full-chunk max_abs:   1.9073e-06
  full-chunk cos:       +1.000000
  VERDICT: PASS
```

Per-step machine precision composes to end-of-chunk machine precision. Analogous to pi0's num_steps=10 unrolled parity, just with the loop external.

Reproducers:
```
modal run scripts/modal_gr00t_monolithic_export.py --parity
modal run scripts/modal_gr00t_monolithic_export.py --loop
```

ONNX size: 4.4GB total (2.1MB `model.onnx` + 4407MB external data), 1.1B params.

---

## The unified verified table

| Model | Type | Comparison | first-action cos | first-action max_abs |
|---|---|---|---|---|
| SmolVLA | flow matching | `sample_actions(num_steps=10)` | +1.000000 | 5.96e-07 |
| pi0 | flow matching | `sample_actions(num_steps=10)` | +1.000000 | 2.09e-07 |
| pi0.5 | flow matching | `sample_actions(num_steps=10)` | +1.000000 | 2.38e-07 |
| GR00T N1.6 | DDPM DiT | `GR00TFullStack.forward` (single-step) | +1.000000 | 8.34e-07 |
| GR00T N1.6 | DDPM DiT | 4-step denoise loop | +1.000000 | 4.77e-07 |

All Modal, A10G or A100-40GB, shared seeded inputs, documented in `measured_numbers.md` Verified section.

---

## Incidents during the session

**Modal HF secret name ambiguity.** The pi0/pi05 scripts use `modal.Secret.from_name("hf-token")`; my account has a secret named `huggingface`. First GR00T run failed after 4 min of image build. Tried `from_name(...) except` fallback — doesn't work because `from_name` is lazy, doesn't resolve until the function runs. Fixed by hardcoding `"huggingface"` for GR00T. Added to `modal_deployment_gotchas.md`.

**Transient Modal image build terminations.** One GR00T run died with "Image build terminated due to external shut-down. Please try again." No retry logic on my end — second run succeeded. Probably a Modal-side transient; not worth wiring in auto-retry given how rare it is.

**Cache warmup.** First Modal parity test after an export sometimes sees truncated external data (volume sync lag). Mitigation in pi0/pi05 scripts: `sleep 60` before parity. Worked fine today; no truncation issues observed on GR00T (different volume, `gr00t-onnx-outputs`, fresh write).

---

## What shipped to git

Six commits pushed to `main` this session (oldest first):

- `eedf078` — fix #3 + #4: real HTTP `/act` roundtrip + CI docker-run smoke (pre-session, landed just before)
- `0c09f80` — fix #1 investigation: v0.3 deferral
- `21043a6` — fix #1 deep dive: partial progress + revert
- `036f98d` — launch: master `ANNOUNCEMENT.md` for v0.2 shipping state
- `bac658a` — **FIX #1 SOLVED: pi0 cos=1.0 at num_steps=10** (the three-patch stack)
- `c604962` — **FIX #2 SOLVED: pi0.5 num_steps=10 cos=1.0**
- `4090e76` — **GR00T N1.6 cos=1.0 machine precision (all 4 major open VLAs verified)**

New files:
- `scripts/modal_pi05_monolithic_export.py`
- `scripts/modal_gr00t_monolithic_export.py`

Updates to docs: `README.md`, `reflex_context/measured_numbers.md`, `launch/ANNOUNCEMENT.md`, `launch/lerobot_3146_draft.md`, `launch/show_hn_draft.md`, `launch/reddit_robotics_draft.md`.

Updates to library: `src/reflex/exporters/monolithic.py` got `export_pi05_monolithic` + `export_gr00t_monolithic` + dispatcher entries, so the CLI `reflex export --monolithic <hf_id>` now routes all four model families correctly.

---

## What's next

v0.2 is now "all four open VLAs verified at machine precision" — the cross-framework moat claim is load-bearing. Next session's candidate work, in priority order:

1. **Tag v0.2.1 (or v0.3.0) and launch.** LeRobot #3146 → Show HN → r/robotics. The verified numbers are now a coherent four-row story.
2. **GR00T VLM conditioning (Eagle backbone export).** The DiT + AdaLN stack is machine precision but VLM conditioning is zero-stubbed. Full multimodal control is v0.3.
3. **Jetson latency numbers** — blocks on hardware access (CloudJetson Orin Nano is waitlisted).
4. **FP16 engine rebuild for Orin Nano 8GB fit** — pi0/pi0.5 don't fit at FP32; SmolVLA and GR00T do.

---

## Artifacts index

| Artifact | Location |
|---|---|
| pi0 monolithic ONNX (num_steps=10, verified) | Modal Volume `pi0-onnx-outputs`, `/onnx_out/monolithic/model.onnx` (12.5GB) |
| pi0.5 monolithic ONNX (num_steps=10, verified) | Modal Volume `pi0-onnx-outputs`, `/onnx_out/monolithic/model.onnx` (13GB) |
| GR00T monolithic ONNX (per-step, verified) | Modal Volume `gr00t-onnx-outputs`, `/onnx_out/monolithic/model.onnx` (4.4GB) |
| SmolVLA monolithic ONNX (num_steps=10, verified) | Modal Volume `smolvla-onnx-outputs`, `/onnx_out/monolithic/model.onnx` (1.6GB) |

All ONNX artifacts are persistent across Modal container restarts; parity tests run without re-exporting.
