# pi0-onnx-parity: empirical de-risk findings

**Session:** 2026-04-17 (Day 1 of 2-day plan)
**Status:** Day 1 partial — blocked on HF auth for full PaliGemma test, but toolchain signal captured.

## Key finding — Optimum support map is better than research indicated

Before downloading anything, we confirmed via `TasksManager.get_supported_tasks_for_model_type`:

| Component | Optimum support | Implication for pi0 |
|---|---|---|
| **Gemma** | ✅ `text-generation-with-past` (KV cache output built-in) | Pi0 backbone prefix-KV extraction = 1 CLI command |
| **Gemma2** | ✅ same tasks | Future-proofs pi0 migrations |
| **SigLIP** | ✅ `feature-extraction`, `zero-shot-image-classification` | Pi0 vision encoder = 1 CLI command |
| BERT / Llama | ✅ supported (reference) | Sanity check |
| **PaliGemma / PaliGemma2** | ❌ NOT supported | Confirms research — need custom glue |
| **SmolVLM** | ❌ NOT supported | Explains why our Apr-17 SmolVLA path hand-rolled everything |

## Strategic implication

Pi0 export = SigLIP + Gemma + pi0 glue. Two of three components are Optimum-native.

**Revised pi0-onnx-parity work breakdown:**

| Component | Path | Estimated LoC | Time |
|---|---|---|---|
| SigLIP vision encoder | `optimum-cli export onnx --task feature-extraction google/siglip-so400m-patch14-224` | ~0 (CLI) | 0.5 day |
| Gemma language (prefix-KV) | `optimum-cli export onnx --task text-generation-with-past google/gemma-2b` then apply pi0 fine-tuned weights | ~100 (weight overlay) | 1 day |
| Expert stack (18 layers, GQA, RoPE) | Hand-written; fork existing `pi0_exporter.py` + our SmolVLA `ExpertStack` | ~300–500 | 3–5 days |
| Flow matching loop glue | Host Python (per Isaac-GR00T pattern); N trt engine calls | ~50 | 0.5 day |
| Composition: vision + KV + expert + loop | New `pi0_prefix_exporter.py` + `pi0_runtime/server.py` | ~300 | 2–3 days |
| Parity verification + bug hunt | Polygraphy diff ladder + our `local_*_diff.py` port | — | 3–5 days |
| **Total** | | **~750–950 LoC of our custom code** | **~10–15 engineer-days (2–3 weeks)** |

**Previous estimate:** 2–3 weeks (matches). **Confidence:** significantly higher now because the biggest question (how to handle PaliGemma monolith) is resolved into "handle it as two Optimum-native components + glue."

## Revised mess-up probability: 25–30%

Down from 25–35% based on:
- **Optimum covers 2/3 of the major components** — confirmed via dry-run TasksManager query
- **KV extraction via CLI flag** — eliminates the riskiest custom code (prefix-KV Python-object serialization)
- **We already have SigLIP export patterns** from our SmolVLA work — pattern transfer direct
- Tooling installed and working (Polygraphy, graphsurgeon, onnx-diagnostic, onnxsim)

Residual risk:
- HF auth blocker (solvable) — need HF_TOKEN or `hf auth login`
- Custom glue between Optimum-exported components + our expert stack + flow-matching loop — this is where bugs will live
- Orin TRT 10.3 numerical bugs (~5% residual, can't de-risk without real hardware)

## Day 1 tooling install — completed

```
✓ polygraphy 0.49.26
✓ onnx-graphsurgeon 0.6.1
✓ onnx-diagnostic 0.9.2
✓ onnxsim 0.6.2
✓ optimum 2.1.0 + optimum-onnx 0.1.0 + transformers 4.57.6 (downgraded from 5.5.4 automatically)
```

Side effect: transformers downgrade from 5.5.4 → 4.57.6. lerobot still imports OK. Verified.

## Day 1 blockers

### Blocker 1: HF rate limit on Tacoin download

Attempted: `hf download Tacoin/openpi-pi0.5-libero-onnx --local-dir /tmp/oracle_pi05_libero_onnx`

Error: `HfHubHTTPError: 429 Client Error: Too Many Requests. We had to rate limit your IP (14.195.187.228). To continue using our service, create a HF account or login to your existing account, and make sure you pass a HF_TOKEN if you're using the API.`

**Resolution required:** user sets `HF_TOKEN` or runs `hf auth login`.

### Blocker 2: same rate limit blocks PaliGemma download

Attempted: `optimum-cli export onnx --model google/paligemma-3b-pt-224 ...`

Error: `OSError: We couldn't connect to 'https://huggingface.co' to load the files`.

**Resolution required:** same as Blocker 1.

### Partial workaround attempted: Optimum on cached SmolVLM2

Optimum does NOT support `smolvlm` model type (confirmed via empirical export attempt). Cannot use SmolVLM2 as a proxy sanity check for Optimum toolchain.

**But:** we learned Optimum DOES support `gemma` and `siglip` as separate architectures. That's the unlock.

## Next action: user unblock + Day 2 tasks

### User: unblock HF auth

```bash
# Option 1: set token
export HF_TOKEN=hf_xxx

# Option 2: interactive login
hf auth login
```

### Day 2: proceed with Optimum-per-component tests

Once HF auth works:

1. **SigLIP sanity**: `optimum-cli export onnx --model google/siglip-so400m-patch14-224 --task feature-extraction /tmp/siglip_sanity/`
   - Verifies Optimum SigLIP path works end-to-end
   - Size check: should be ~400MB for so400m; watch for 2GB protobuf limit
   - Polygraphy-diff vs `transformers.SiglipVisionModel` PyTorch output

2. **Gemma backbone sanity**: `optimum-cli export onnx --model google/gemma-2b --task text-generation-with-past /tmp/gemma_sanity/`
   - Verifies Gemma ONNX with KV outputs works
   - Inspect `decoder_model_merged.onnx` and `decoder_with_past.onnx` structure
   - Polygraphy-diff vs `transformers.GemmaModel` PyTorch output on a fixed prompt

3. **Apply pi0 fine-tuned weights to Gemma ONNX**
   - Extract Gemma subset from pi0_base state_dict (`paligemma_with_expert.paligemma.language_model.*`)
   - Overlay onto Optimum's Gemma ONNX via onnx-graphsurgeon weight replacement
   - Verify replaced-weight Gemma ONNX still produces sensible outputs

4. **Verdict + revised plan update**
   - If SigLIP + Gemma both clean with pi0 weights overlaid → commit to 2-week pi0-onnx-parity
   - If Gemma weight overlay fails → fallback: export pi0's Gemma subset directly via our own `torch.onnx.export` pipeline

## Output artifacts expected

By end of Day 2:
- `/tmp/siglip_sanity/siglip.onnx` + parity diff result
- `/tmp/gemma_sanity/decoder_*.onnx` + parity diff result
- `/tmp/oracle_pi05_libero_onnx/` (Tacoin — for pi0.5 future work)
- Updated `reflex_context/mvp_queue.md` with empirical ETA
- Draft `src/reflex/exporters/pi0_prefix_exporter.py` skeleton (new file)
