# pi0 ExpertStackWithPrefix vs real pi0 — 12 divergences

**Context:** our custom `src/reflex/runtime/pi0_onnx_server.py` + `Pi0ExpertStackWithPrefix` got cos=-0.19 vs pi0 reference on end-to-end parity. Research agent audited the code vs `lerobot/policies/pi0/modeling_pi0.py` and found 12 specific divergences, ranked by likely impact.

**Status:** we pivoted to monolithic wrap (Path C) instead of fixing all 12. Keeping this list for future debugging — if Path C ever fails and we return to decomposed export, these are the gaps to close.

**Source references:**
- `mpi0` = `.venv/lib/python3.13/site-packages/lerobot/policies/pi0/modeling_pi0.py`
- `pg` = `.venv/lib/python3.13/site-packages/lerobot/policies/pi_gemma.py`

## Divergences ranked by impact

### 1. MISSING state token prepend (mpi0:699-706)
**pi0:** suffix is 51 tokens. Token 0 = `state_proj(state)[:, None, :]` shape `[B,1,1024]`. Tokens 1..50 = `action_time_emb`.

**Our stack:** only 50 tokens (action_time only). Missing the state token entirely.

**Fix if returning to Path A:** take `state` as input to forward, compute `state_emb = state_proj(state)[:, None, :]`, concat `[state_emb, action_time_emb]` along seq dim.

### 2. WRONG output slice (mpi0:929)
**pi0:** `action_out_proj(suffix_out[:, -chunk_size:])` — drops the state token before projection.

**Our stack:** `action_out_proj(final_norm(x))` — projects all 51 tokens (if state were prepended) or all 50 tokens.

**Fix:** `return self.action_out_proj(final_norm(x)[:, -50:])`.

### 3. SINUSOIDAL params likely wrong (mpi0:83-100)
**pi0:** `create_sinusoidal_pos_embedding(timestep, 1024, min_period=4e-3, max_period=4.0)`. Compute in **float64**, `period = min * (max/min)**fraction`, scale = `1/period * 2pi`, concat `[sin, cos]` (not interleaved).

**Our stack:** `_sinusoidal_pos_embedding(timestep, expert_hidden)` — may use different min/max_period, dtype, or sin/cos order.

**Fix:** verify our `_sinusoidal_pos_embedding` in `src/reflex/exporters/smolvla_exporter.py` matches pi0's exact computation. If not, port pi0's version for the pi0-specific path.

### 4. MISSING attention mask (mpi0:908-911)
**pi0:** per layer, attention mask is `[B, 1, 51, prefix_len+51]` 4D tensor with:
- Prefix rows = prefix_pad_masks expanded (suffix attends to all valid prefix)
- Suffix rows = `make_att_2d_masks` result (state + action block pattern)

**Our stack:** no mask. Concat `[prefix_k, action_k]` then dense softmax over all.

**Impact:** probably minor if prefix has no padding (which is our case), but state-token masking is non-trivial — state can only attend backward, actions all attend as one block.

**Fix:** build 4D attention mask per layer, apply in `ExpertGQALayer.forward(prefix_k_concat=..., mask=...)`.

### 5. WRONG position_ids (mpi0:912-913)
**pi0:** `position_ids = prefix_offsets + cumsum(suffix_pad_masks, 1) - 1` → `[prefix_len, prefix_len+1, ..., prefix_len+50]`.

**Our stack:** `np.arange(chunk_size)` → `[0..49]`. **Fixed in Pi0OnnxServer after our Apr-18 bug hunt** but the ONNX still received the wrong positions.

**Fix:** already applied (Pi0OnnxServer.predict now uses `prefix_len + np.arange(chunk)`). Verify the ONNX graph actually uses position_ids (see bug #8 — RoPE may not respond to position_ids changes).

### 6. NO sqrt(hidden_size) scaling (pg:270-277)
**pi0:** PiGemmaModel **overrides** HF Gemma's forward and DROPS the `inputs_embeds * sqrt(hidden_size)` multiplier. Confirmed by reading pg:270-277 vs transformers/models/gemma/modeling_gemma.py:400-401.

**Our stack:** no sqrt scaling applied to suffix_embs.

**Decision:** **do NOT add the scaling.** Our current behavior matches pi0. This is NOT a bug to fix — it's a trap for future session not to accidentally add it.

### 7. RMSNorm weight convention (pg:117-120)
**pi0:** PiGemmaRMSNorm `y = x_normed * (1 + weight.float())`. Weights trained centered at 0.

**Our stack:** DecomposedRMSNorm `y = x_normed * weight`. Fixed at commit 066f68a by pre-transforming weights: `weight_stored = 1 + weight_pi0`.

**Status:** FIXED. Verify the +1 transform is applied to both input_layernorm and post_attention_layernorm AND the final norm.

### 8. RoPE not responding to position_ids
**Observed:** direct ORT test of our exported expert showed `pos_ids=[0..49]` vs `pos_ids=[16..65]` produced max_diff=1.5e-4 (effectively no change).

**Cause:** either _DecomposedRoPE ONNX graph has position_ids constant-folded, or our rope_theta was 100000 (SmolVLA) instead of 10000 (Gemma). Fixed rope_theta → still minimal response.

**Unknown:** why RoPE doesn't strongly respond. Could be attention soft masking + random prefix dominating.

**Test:** feed IDENTICAL position_ids but different prefix_k/v — verify output changes. If not, attention is broken. If yes, RoPE is just being masked by softmax smoothing.

### 9. Head geometry (mpi0:317-323)
**pi0:** `num_heads=8, num_kv_heads=1, head_dim=256` (gemma-300m config).

**Our stack:** was `nq=16, nkv=2, head_dim=128` due to hardcoded head_dim=128 in pi0_exporter.build_pi0_expert_stack. **Fixed at commit 33a890d.**

**Status:** FIXED.

### 10. PaliGemma vs expert RoPE cos/sin source (mpi0:260, pg:277)
**pi0:** compute_layer_complete uses `paligemma.language_model.rotary_emb(dummy, position_ids)` for combined forward. `gemma_expert.model.forward` uses its OWN `rotary_emb`.

**Check:** both are built from the same GemmaConfig (theta=10000, base, dim=256) so should produce identical cos/sin. Verify.

### 11. deepcopy(past_key_values) per step (mpi0:918)
**pi0:** `past_key_values = copy.deepcopy(past_key_values)` — defensive copy so cache isn't mutated across denoise steps.

**Our stack:** prefix_kv passed as ONNX inputs, effectively read-only. Safe.

**Status:** OK.

### 12. dtype precision
**pi0:** variance computed in float32 inside PiGemmaRMSNorm regardless of input dtype (pg:117-120). Training is bf16; inference in our setup is fp32.

**Our stack:** DecomposedRMSNorm promotes to float32 for variance. Matches.

**Status:** OK.

## Summary

| # | Divergence | Status | Must-fix for Path A | Relevant to Path C |
|---|---|---|---|---|
| 1 | Missing state token | OPEN | YES | N/A (Path C uses pi0's own suffix) |
| 2 | Wrong output slice | OPEN | YES | N/A |
| 3 | Sinusoidal params | OPEN (verify) | MAYBE | N/A |
| 4 | Missing attention mask | OPEN | YES | N/A |
| 5 | Wrong position_ids | FIXED | — | — |
| 6 | sqrt(hidden) scaling | N/A — pi0 drops it | Don't add | — |
| 7 | RMSNorm (1+w) | FIXED | — | — |
| 8 | RoPE not responding | UNKNOWN | Maybe symptom | — |
| 9 | Head geometry | FIXED | — | — |
| 10 | Separate rotary_emb | LIKELY OK | Verify | — |
| 11 | deepcopy cache | OK | — | — |
| 12 | dtype precision | OK | — | — |

**Path A remaining work (if resumed):** ~3-5 of the 12 items need fixing (1, 2, 3, 4, plus verify 8, 10). Estimated 4-8 hours + parity iterations.

**Path C (monolithic wrap):** bypasses ALL of these by using pi0's own tested sample_actions. Preferred for this reason.
