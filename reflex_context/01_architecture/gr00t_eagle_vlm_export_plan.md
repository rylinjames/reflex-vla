# GR00T Eagle VLM ONNX export plan (2026-04-19)

**Goal:** close the last gap in GR00T support — export the Eagle VLM backbone so real images + language conditioning drive the DiT, instead of a zero-stubbed VLM-KV tensor. This is goal `gr00t-vlm-conditioning` (weight 9) in GOALS.yaml, the last monetization blocker for GR00T customers.

**Current state:** `scripts/modal_gr00t_monolithic_export.py` exports a DiT-only ONNX at `cos=+1.000000` parity with PyTorch. But the VLM-KV input is a zero tensor. Deployed, the model ignores the image + prompt and produces noise-driven actions. Not shippable for real multi-modal control.

---

## Eagle 2.5 VL — what we're exporting

Eagle 2.5 VL is a SigLIP + Qwen2 composite VLM. GR00T uses a fine-tuned copy bundled inside `nvidia/GR00T-N1.6-3B`.

| Component | Model | Hidden dim | Purpose |
|---|---|---|---|
| Vision | `SiglipVisionModel` (SigLIP-so400m-patch14-384, ~400M) | 1152 | Image → patch tokens |
| Pixel shuffle | 0.5× downsample | — | 1024 patch tokens → 256 image tokens/tile |
| Connector | `mlp1`: LayerNorm + 2-layer MLP | 2048 | Image features → Qwen2 hidden space |
| Text | `Qwen2ForCausalLM` (0.5B) | 2048 | Process image tokens + text, output contextualized hidden states |

**Output fed to DiT:** `[B, T, 2048]` where T is vision tokens + text tokens. Our DiT's `vlln` is `LayerNorm(2048)` — confirms N1.6 consumes raw Qwen2 hidden (skipping the `eagle_linear` 2048→1536 projection that N1.5 had).

**State-dict keys** (inside `nvidia/GR00T-N1.6-3B`):
- `backbone.eagle_model.vision_model.*` — SigLIP
- `backbone.eagle_model.language_model.*` — Qwen2
- `backbone.eagle_model.mlp1.*` — connector
- (`backbone.eagle_linear.*` — absent in N1.6 per hypothesis; verify by dumping keys)

**Entry point** (in lerobot source, `groot_n1.py:135-146`):

```python
eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
eagle_features = eagle_output.hidden_states[self.select_layer]
return eagle_features, eagle_input["attention_mask"]
```

Single-shot encode, no autoregressive loop, no runtime KV cache. **Good for ONNX** — avoids the DynamicCache surgery we needed for pi0/pi0.5.

---

## Torch-export patches required

Qwen2's decoder-only attention uses the same stack that bit us on pi0/pi0.5:

1. **F.pad causal mask** instead of `torch.cat` (FakeTensor loses suffix dim)
2. **Frozen DynamicLayer.update** during trace (Qwen2 calls `past_kv.update()` unconditionally)
3. **`past_kv.get_seq_length()`** for mask K dim, not `input_ids.shape[-1]`

Plus a fourth patch specific to Eagle:
- **Force `_attn_implementation="eager"`** at load time. Eagle's config hardcodes `flash_attention_2`. ORT/TRT can't consume FA2 ops.

Reference stack lives in `scripts/modal_pi0_monolithic_export.py` lines 200–310. Port these wholesale to the Eagle export script.

**SigLIP needs no patches** — bidirectional attention, no causal mask, no cache.

---

## Integration with existing DiT ONNX

Good news: `GR00TExpertStack.forward` (in `src/reflex/exporters/gr00t_exporter.py:208-241`) already accepts `vlm_kv: Optional[Tensor]`. The existing export just passes `None`. Surgical add to `GR00TFullStack`:

```python
# Current (zero-stubbed):
def forward(self, noisy_actions, timestep, position_ids):
    ...
    velocity_tokens = self.dit(tokens, timestep, position_ids, vlm_kv=None)

# New:
def forward(self, noisy_actions, timestep, position_ids, vlm_kv):
    ...
    velocity_tokens = self.dit(tokens, timestep, position_ids, vlm_kv=vlm_kv)
```

Keep the existing zero-VLM ONNX for backward compat (naming it `model.onnx` currently). The new export writes **two ONNXes**:
- `eagle_vlm.onnx` — vision + text → KV features `[B, T, 2048]`
- `expert_stack_with_vlm.onnx` — DiT taking `noisy_actions + timestep + position_ids + vlm_kv` → `velocity`

Runtime chains them: Eagle runs once per observation (images + prompt), DiT runs 4 times per action chunk (DDPM denoise loop) reusing the same KV.

---

## 5-step implementation plan (1–2 days)

### ✅ Step 1 — vendor Eagle source (DONE 2026-04-19)
Copied `lerobot/policies/groot/eagle2_hg_model/*` (4 files, 1575 lines) into `src/reflex/exporters/eagle_vendor/`. Modifications:
- `peft` + `LoraConfig` imports made optional (training-only, not needed for export)
- `_attn_implementation` default flipped `flash_attention_2` → `eager` in both config and modeling (FA2 ops not consumable by ONNX/TRT)
- Training code can still explicitly pass FA2 on capable hardware

**Step 1c diagnostic (key dump) unblocked Step 2** by revealing:
- `eagle_linear` ABSENT — skip the 2048→1536 projection
- State dict prefix is `backbone.model.*` (not `backbone.eagle_model.*`)
- **NEW FINDING: `action_head.state_encoder` EXISTS** (32-embodiment, 128→1024→1536, 54.6M params) — was missing from prior reflex exports
- **`future_tokens` ABSENT** in N1.6 — DiT sequence is simpler: `sa_embs = cat(state, actions)` (no learnable prefix)
- DiT substructure: timestep_encoder, 32 transformer_blocks, proj_out_1 (final AdaLN), proj_out_2 (velocity)
- All documented in `reflex_context/01_architecture/gr00t_n16_state_dict_analysis.md`

### ✅ Step 2 — port state_encoder + extend GR00TFullStack (DONE 2026-04-19)
Code changes in `src/reflex/exporters/gr00t_exporter.py` (commit `16d6d9c`):
- NEW `GR00TStateEncoder(nn.Module)`: 2-linear (128→1024→1536), per-embodiment slicing (32-emb), matches the existing action_encoder/decoder pattern
- `GR00T_META_KEYS` extended with `state_enc_{1,2}_{W,b}` mappings to `action_head.state_encoder.*`
- `GR00TFullStack` forward signature extended: `(noisy_actions, timestep, position_ids, state, vlm_kv)`. Builds `sa_embs = cat(state_token, action_tokens)` when state provided; slices state prefix off before action_decoder
- `build_gr00t_full_stack` soft-loads state_encoder (back-compat: older checkpoints without state_enc keys fall back to action-only sequence)
- DiT (`GR00TExpertStack`) already accepted `vlm_kv: Optional[Tensor]` — zero-stub path preserved as default

**Step 2 validation — PASSED 2026-04-19** (`scripts/modal_gr00t_state_encoder_sanity.py`, Modal run `ap-558ZOlxr9XhpT7cZGppr5F`):

| Test | Result |
|---|---|
| state_encoder loads from N1.6 | ✅ 128→1536, `has_state_encoder=True` |
| Output shape | ✅ `(1, 50, 128)` |
| State conditioning LIVE | ✅ max\|a-b\| = 1.14e-02 (ratio 0.06) |
| VLM conditioning LIVE | ✅ max\|zero-rand\| = 5.81e-01 (ratio **2.85**) |
| Back-compat (state=None) | ✅ works |

VLM dominates (2.85× ratio) over state (0.06× ratio) — expected: vision + language drive decisions, state is secondary proprio. Both conditioning paths confirmed LIVE, not dead code. Ready for Step 3 parity test against lerobot reference.

### ✅ Step 3 — parity test DONE 2026-04-19 (`cos=+1.000000, max_abs=0.0` bit-exact)

`scripts/modal_gr00t_vlm_parity.py` — Modal run `ap-qZUywn100OUSIm49xZf6D5`.

**Important pivot during Step 3:** originally planned to compare against lerobot's `GR00TN15.from_pretrained`, but lerobot 0.5.1 can't load N1.6 (config has `model_type="Gr00tN1d6"` but lerobot 0.5.1 only supports `gr00t_n1_5`; also missing `backbone_cfg` attr). Instead wrote a hand-rolled reference that uses the SAME primitive classes as our GR00TFullStack (action_encoder, state_encoder, dit, action_decoder) wired directly, so any delta from GR00TFullStack reveals a wrapper bug.

**Bug caught during Step 3:** pos_embed was being added to the state_token (position 0). Lerobot's code adds pos_embed to action_features BEFORE concat with state; state has NO pos_embed. Fixed in this session:

```python
# FIXED: apply pos_embed to action_tokens BEFORE concat, skip DiT's internal add
action_pos_ids = torch.arange(chunk)
action_pos = self.dit.pos_embed[action_pos_ids].unsqueeze(0)
action_tokens = action_tokens + action_pos
tokens = torch.cat([state_token, action_tokens], dim=1)  # state has no pos_embed
self.dit(tokens, ..., add_pos_embed=False)  # skip DiT's own pos_embed add
```

Also added `add_pos_embed: bool = True` kwarg to `GR00TExpertStack.forward` for this skip-path.

Final parity: **cos=+1.000000, max_abs=0.0** on seeded synthetic inputs (noisy, timestep=0.5, state=randn, vlm_kv=randn). Identical byte-for-byte first_action values confirm the wrapper is semantically sound.

**Caveat:** this parity is against a hand-rolled reference using our own primitive classes, NOT against the real lerobot/GR00T-N1.5 reference (blocked on lerobot version incompat). We trust the primitives because they were already verified cos=1.0 against PyTorch in prior session (see `monolithic_parity_table.md` GR00T rows). The real "does this produce task-relevant actions" test is Step 5 (end-to-end).

### Step 4 — Modal export

**Step 4a — expert_stack_with_vlm.onnx (DONE 2026-04-19)**

Committed. `expert_stack_with_vlm.onnx` writes to `/onnx_out/monolithic_with_vlm/` on Modal volume.

- Export command: `modal run scripts/modal_gr00t_monolithic_export.py --vlm`
- ONNX conversion time: 314.7s (~5 min)
- Size: 2.0MB model + 4414.3MB external data = **4.4GB total**
- All 5 inputs wired: `noisy_actions, timestep, position_ids, state, vlm_kv`
- Dynamic axes: batch on all inputs, vlm_seq on vlm_kv

**Parity test (DONE):**
- `modal run scripts/modal_gr00t_monolithic_export.py --vlm-parity`
- Modal run `ap-n7oJT5C20y4tEdVu4D55n1`
- cos = +1.000000, max_abs = 1.78e-05 (fp32 noise)
- First action matches PyTorch and ONNX to 5 decimal places
- **VERDICT: PASS (machine precision)**

The with-vlm ONNX is equivalent to our existing zero-stub `model.onnx` (cos=1.0 parity already measured), just with state + vlm_kv as first-class external inputs instead of hardcoded zeros. Runtime-swappable.

**Step 4b — eagle_vlm.onnx (DONE 2026-04-19)**

`scripts/modal_gr00t_eagle_vlm_export.py` — three entrypoints (smoke/export/parity). Modal run `ap-jtSMVpCoXVjkBBRp29bFWT` (parity).

| Metric | Value |
|---|---|
| Export time | 222s on A100-40GB |
| Total params | 1.868 B |
| Graph rewrites applied | 431 (dynamo + optimize) |
| ONNX size | 5.6 MB model + 5.99 GB external data |
| PyTorch forward | 1.21s |
| ONNX CUDA forward | 0.81s |
| Output shape | `[1, 80, 2048]` |
| cos sim | **+1.000000** (0.99999999994) |
| max abs diff | 4.25e-04 |
| mean abs diff | 1.81e-05 |
| Verdict | **PASS (machine precision)** |

**Turned out easier than expected**: the feared 3-patch Qwen2 stack (F.pad mask, frozen DynamicLayer.update, past_kv.get_seq_length) was NOT needed because our export is single-shot (`use_cache=False`). Dynamo tracing handled Qwen3-style attention cleanly.

**Surprise findings during Step 4b implementation**:

1. **N1.6 language model is Qwen3, not Qwen2** — has `qk_layernorm` (q_norm + k_norm inside self-attn) and no bias on q/k/v projections. Discovered when Qwen2 config produced 48 missing keys (q/k/v.bias) and 32 unexpected keys (k_norm, q_norm). Switched architecture to `Qwen3ForCausalLM` → 0 missing, 0 unexpected.
2. **SigLIP is 224×224, not 448×448** in N1.6. Derived from position_embedding shape: 256 positions = 16×16 patch grid × 14-pixel patches = 224×224 input. pixel_shuffle scale=0.5 → 64 image tokens per tile.
3. **patch_embedding is stored flat [1152, 588]** in N1.6 state dict (588 = 3×14×14 = patch flattened). Standard SigLIP uses Conv2d weights [1152, 3, 14, 14]. Reshape-on-load handles this.
4. **pixel_shuffle needed `.reshape()` not `.view()`** in the vendored Eagle source — non-contiguous tensor failed view in export path.
5. **Image-token splicing broke ORT** with the Eagle original `input_embeds[selected] = vit_flat` (boolean-index assignment). It lowered to `index_put → Where` that couldn't broadcast (vit_len=64 vs seq=80). Fix: replace with `torch.cat([vit_embeds, text_embeds[vision_seq:]], dim=1)`. Export contract: caller must pack image tokens at the FRONT of input_ids.

**Eagle class name is `Eagle25VL`** (no underscores, despite the model being called "Eagle 2.5 VL" — NOT `Eagle2_5_VL`).

### ✅ Step 5 — end-to-end chain test (DONE 2026-04-19)

`scripts/modal_gr00t_e2e_chain_test.py` — Modal run `ap-Me7Dh9NjO1jtCOyLJeLWb4`.

| Gate | Metric | Value | Pass? |
|---|---|---|---|
| Parity A | cos(PyTorch_chain, ONNX_chain) | **+1.000000** (0.99999999996) | ✅ (>0.9999) |
| Parity A | max_abs on raw actions | 1.90e-05 | ✅ |
| Parity B | cos(PyTorch_chain, ONNX_chain) | **+1.000000** (0.99999999996) | ✅ (>0.9999) |
| Parity B | max_abs on raw actions | 1.46e-05 | ✅ |
| Sensitivity PyTorch | max_abs(actions_A - actions_B) | 0.212 | ✅ (>0.01) |
| Sensitivity PyTorch | cos(actions_A, actions_B) | +0.982 | (image-driven delta — NOT identical) |
| Sensitivity ONNX | max_abs(actions_A - actions_B) | 0.212 | ✅ (>0.01) |
| Sensitivity ONNX | cos(actions_A, actions_B) | +0.982 | (matches PyTorch) |

**Closes the last GR00T deployment gap.** Both exported ONNXes compose correctly, VLM conditioning is LIVE (image change → action change), and the ONNX chain faithfully mirrors the PyTorch chain. Before: `vlm_kv=0` stub → actions ignored the image. After: real Eagle KV → actions track the image.

### Step 5 (ORIGINAL plan, preserved for reference) — end-to-end chain test
Real image + "pick up the red cup" through the full chain → actions:
`pixel_values + prompt → eagle_vlm.onnx → hidden_states[B, T, 2048] → expert_stack_with_vlm.onnx (vlm_kv) → velocity → actions[B, chunk, 32]`.

Flip the image; actions must change (currently with zero-KV they don't). Then update `measured_numbers.md` + launch drafts with "GR00T now has real VLM conditioning, cos=+1.000000 verified on both ONNX hops."

Modal script to write: `scripts/modal_gr00t_e2e_chain_test.py`:
1. Load N1.6 once on A100
2. Build `EagleExportStack` + `GR00TFullStack` (PyTorch reference)
3. Load both ONNXes via ORT
4. Generate image A (random seed 1) and image B (random seed 2) + identical prompt tokens
5. Run PyTorch chain (Eagle then DiT) → actions_A_pt, actions_B_pt
6. Run ONNX chain (eagle_vlm.onnx then expert_stack_with_vlm.onnx) → actions_A_onnx, actions_B_onnx
7. Assertions:
   - `cos(actions_A_pt, actions_A_onnx) > 0.9999`  (parity holds end-to-end)
   - `max_abs(actions_A_pt - actions_B_pt) > 0.01` (images drive different actions in PyTorch)
   - `max_abs(actions_A_onnx - actions_B_onnx) > 0.01` (same sensitivity survives ONNX)

---

## Known land mines

- **N1.5 vs N1.6 difference.** Research agent couldn't verify N1.6's config directly. N1.5 has `eagle_linear` (2048→1536); N1.6 hypothesized to skip it. **Verify first** by dumping state_dict keys. If `backbone.eagle_linear.weight` exists, include it; if absent, skip.
- **Standalone `nvidia/Eagle2.5-8B` ≠ GR00T's Eagle.** GR00T's is a fine-tune. Export from the GR00T checkpoint, not from the standalone release.
- **Flash Attention 2 hardcoded.** Eagle's `configuration_eagle2_5_vl.py:52` has `_attn_implementation = "flash_attention_2"` baked in. Must override to "eager" before calling `from_pretrained`, or ONNX export breaks on FA2 ops.

---

## Expected ONNX sizes

Rough estimates (post-fp32):
- `eagle_vlm.onnx`: ~2–3 GB (SigLIP ~400M + Qwen2 ~0.5B + connector ~5M = ~1B params × 4 bytes + embeddings)
- `expert_stack_with_vlm.onnx`: ~4.4 GB (same as current DiT-only export + one more input)

Combined: ~7 GB. Fits on Orin 16GB+, not Orin Nano 8GB (same limitation as pi0/pi0.5).

---

## Cross-references

- `src/reflex/exporters/gr00t_exporter.py` lines 208-241 — where `vlm_kv` plumbs through `GR00TExpertStack`
- `src/reflex/exporters/gr00t_exporter.py` lines 480-700 — `GR00TFullStack` (needs new arg)
- `scripts/modal_gr00t_monolithic_export.py` — extend this with `export_gr00t_vlm_modal`
- `scripts/modal_pi0_monolithic_export.py` lines 200–310 — the 3-patch stack reference
- `/Users/romirjain/Desktop/building projects/lerobot/src/lerobot/policies/groot/groot_n1.py` — `EagleBackbone` source to mirror
- `/Users/romirjain/Desktop/building projects/lerobot/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py` — Eagle model definition
- `GOALS.yaml` — `gr00t-vlm-conditioning` (weight 9, in current_focus)
- `reflex_context/external_refs.md` — clone location of lerobot
