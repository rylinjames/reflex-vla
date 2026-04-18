# Measured Numbers

Single source of truth for what reflex-vla can credibly claim today. Three buckets: **verified** (reproducible script + commit), **unverified** (ran but don't trust), **unmeasured** (vapor).

Any number not in this file is not a claim. Any number in this file without a reproducer is a bug in this file — fix or move it.

Last updated: 2026-04-17.

---

## Verified

Reproducible on a fresh clone with the linked script at the linked commit. These are the only numbers safe to put in pitch decks, README, tweets.

| Date | Metric | Value | Reproducer | Commit |
|---|---|---|---|---|
| 2026-04-17 | Native path (lerobot `SmolVLAPolicy` + `DecomposedRMSNorm` swap) vs PyTorch reference, end-to-end first-action | **cos = 1.0000, L2 = 0.000** | `pip install 'reflex-vla[native]' && REFLEX_NATIVE=1 python scripts/local_full_diff.py` | `9de9f64` (re-verified post-rebalance; also holds on `0616265`) |
| 2026-04-17 | Vision encoder (SigLIP) vs PyTorch | cos = 1.0000, L2 = 2.2e-03, max_abs = 1.07e-04 | `scripts/local_stage_diff.py` | `0616265` |
| 2026-04-17 | Text embedder vs PyTorch | cos = 1.0000 | `scripts/local_stage_diff.py` | `0616265` |
| 2026-04-17 | State projection vs PyTorch | cos = 1.0000 | `scripts/local_stage_diff.py` | `0616265` |
| 2026-04-17 | Per-layer vlm_k (decomposed path) | cos = 1.0000 | `scripts/local_stage_diff.py` | `0616265` |
| 2026-04-17 | Single self-attn layer (layer 0, isolated weight copy) | cos = 1.0000 to 1e-5 | `scripts/local_single_layer_diff.py` | `0616265` |
| 2026-04-17 | Optimum-exported tiny Gemma ONNX vs PyTorch (`text-generation-with-past`) | cos = +1.00000000, max_abs_diff = 7.15e-07, KV diff = 1.79e-07 | `scripts/local_tiny_gemma_sanity.py` (per-layer K/V exposed as `present.N.key/value` outputs) | `85c24c0` |
| 2026-04-17 | **pi0's actual 2.51B Gemma backbone** (extracted from pi0_base) exported via Optimum ONNX vs PyTorch | **cos = +0.99999994**, logits max_abs_diff = 3.86e-05, present.17.key (deepest layer) max_diff = 3.02e-05 | Pi0's `paligemma_with_expert.paligemma.model.language_model.*` state-dict subset loaded into `GemmaForCausalLM`, exported via `optimum-cli export onnx --task text-generation-with-past`, 10GB on-disk, 9.3GB external-data ONNX | `f67a012` (Day 2 empirical) |
| 2026-04-17 | **pi0's SigLIP vision tower** (427.7M, 27 layers, so400m-patch14-224) exported via Optimum ONNX vs PyTorch | **cos = +0.99999994**, last_hidden_state max_abs_diff = 3.59e-04, mean_abs_diff = 2.45e-06 | Pi0's `paligemma_with_expert.paligemma.model.vision_tower.*` state-dict subset loaded into `SiglipVisionModel`, exported via `optimum-cli export onnx --task feature-extraction` | `a32bcf0` (Day 2 empirical SigLIP) |
| 2026-04-17 | pi0's multi_modal_projector (Linear 1152→2048) via our own `torch.onnx.export` | cos = +1.00000012, max_abs = 5.25e-06 | `scripts/local_pi0_prefix_smoke.py` | `099c9da` (Week-1 scaffold) |
| 2026-04-17 | pi0's text_embedder (Embedding 257152×2048, tied lm_head) via our own `torch.onnx.export` | cos = +1.00000000, max_abs = 0.00e+00 (bit-exact) | `scripts/local_pi0_prefix_smoke.py` | `099c9da` (Week-1 scaffold) |
| 2026-04-17 | **End-to-end pi0 prefix export pipeline** — all 4 ONNX components produced in one run: vision_encoder (1.7GB), multi_modal_projector (9MB), text_embedder (2.0GB), decoder_prefill (10GB external data) | Exit code 0; Optimum internal validation per-layer KV max_diff 2–4e-4 (matches standalone tests) | `export_pi0_prefix(output_dir="/tmp/pi0_full_prefix_export")` from `src/reflex/exporters/pi0_prefix_exporter.py` | `39ffc6d` (Week-1 end-to-end) |
| 2026-04-18 | **pi0's action expert as HF GemmaForCausalLM (gemma-300m, 0.57B)** exported via Optimum ONNX with `text-generation-with-past` — all 18 layers of past_kv as inputs | cos = +0.99999994, max_abs = 1.24e-05 (standalone, no prefix) | Pi0's `paligemma_with_expert.gemma_expert.model.*` state-dict subset loaded into `GemmaForCausalLM` (gemma-300m config), exported via `optimum-cli export onnx --task text-generation-with-past` | `bdbfef9` + subsequent (expert redesign day 2) |
| 2026-04-18 | **END-TO-END pi0 monolithic ONNX** (12.5GB, 6MB model.onnx + 12GB external data) vs PyTorch `PI0Pytorch.sample_actions(num_steps=1)` on shared inputs + seeded noise | **first-action cos = +1.0000000, max_abs = 1.43e-06; full-chunk cos = +1.0000000, max_abs = 2.98e-06** | `scripts/modal_pi0_monolithic_export.py` via Modal A10G + onnx-diagnostic `torch_export_patches(patch_transformers=True)` + `torch.export.export` + `torch.onnx.export(ExportedProgram)` (dynamo-based, opset 19). num_steps=10 hits 835→886 shape expand bug; num_steps=1 exports cleanly. Reflex's production plan uses num_steps=1 ONNX in a host-Python flow-matching loop (per Path C architectural decision — see `01_architecture/pi0_monolithic_wrap_pattern.md`). | Modal runs 1-11 (this session); parity run 12 verified |
| 2026-04-18 | **pi0 native path** (`PI0Policy.predict_action_chunk` wrapper) vs raw `PI0Pytorch.sample_actions` on shared seeded inputs | **cos = 1.0000000000, max_abs = 0.000e+00 (bit-exact)** | `modal run scripts/modal_pi0_monolithic_export.py --native`. Verifies the PI0Policy wrapper's preprocessing (image norm + state pad + lang tokens) introduces no drift vs the raw forward that ONNX matches at cos=+1.0000000. Closes `multi-model-native-parity` for pi0 (pi0.5 + GR00T deferred to v0.3). | Modal run via `parity_native_pi0` function |
| 2026-04-18 | **END-TO-END SmolVLA monolithic ONNX** (1.6GB total) vs PyTorch `SmolVLAPolicy.model.sample_actions(num_steps=1)` on shared inputs + seeded noise | **first-action cos = +1.0000000, max_abs = 1.55e-06; full-chunk cos = +1.0000000, max_abs = 3.34e-06** | `scripts/modal_smolvla_monolithic_export.py` via Modal A10G. Same onnx-diagnostic `torch_export_patches` pattern as pi0, with three additional fixes: (1) pin `transformers==5.3.0` exactly (5.4+ has q_length scalar regression in masking_utils that breaks onnx-diagnostic patches); (2) monkey-patch `torch.where` to explicit-cast mismatched-dtype branches (SmolVLM2 vision embeds do `torch.where(bool_mask, torch.full(fill_value=0), float_tensor)` which torch.export traces as mismatched int64/float32 inputs); (3) post-export ONNX pass that finds Where nodes with mismatched branches and inserts Cast nodes to match the declared output dtype — needed because torch.onnx sometimes lowers `index_put` to Where(bool, int64, float) even when the aten graph is clean. Output verified at cos=+1.0000000 on parity run 6. | Modal runs 1-13 (this session); parity run 14 verified |

**Headline claim:** "Reflex's native + ONNX export paths match the reference PyTorch policy to cos = 1.0000 end-to-end on BOTH SmolVLA and pi0." The cross-framework moat claim is now load-bearing-ready.

---

## Unverified (ran, don't trust)

Numbers that came out of an actual run but aren't stable, aren't reproducible, or are known to be from a buggy code path. Do not cite externally.

| Date | Metric | Value | Why unverified |
|---|---|---|---|
| 2026-04-17 | LIBERO-10 success rate, decomposed ONNX path | 0% | Path had 12 known bugs at time of run; composition issue unresolved (layer_0_v cos = 0.9117 outlier, per-step velocity cos = 0.977 compounds to -0.24 over 10 Euler steps) |
| 2026-04-17 | LIBERO-10 success rate, native path | never completed | Modal container never finished — dep conflict between LIBERO-2023 stack and lerobot-2026 stack |
| 2026-04-13 | TRT FP16 latency headlines (2.6–3.3× FP32) | archived | Apples-to-oranges: FP16 TRT vs FP32 torch, not FP16 vs FP16; also never re-verified on native path |
| 2026-04-17 | Per-layer vlm_v cos (decomposed path, layers 1–15) | 0.91–1.00 | layer_0_v = 0.9117 is the outlier; unresolved in decomposed path. Native path bypasses this. |
| 2026-04-17 | Per-step velocity (decomposed expert) | cos = 0.977 per step | Compounds catastrophically across 10 flow-matching steps → final cos = -0.24. Native path supersedes. |

---

## Unmeasured (vapor)

Claims we'd want to make but have zero evidence for. If someone asks for one of these in a customer call or pitch, the honest answer is "we haven't measured it yet."

- **Jetson Orin Nano latency** — ms/step, Hz, memory. Never run.
- **Jetson Orin (32GB / 64GB) latency** — never run.
- **Jetson Thor (FP8) latency** — never run; no hardware access.
- **Desktop A10G latency on native path** — the benchmarks we have are from the decomposed path, pre-native pivot. Need re-run.
- **Calibration (ECE / Brier / NLL)** on any benchmark — never computed. This is the *one* non-sim metric with a monotonic link to real-world task success (Zollo 2025), and we have nothing.
- **Memory footprint** — VRAM per export stage on any target. Never measured.
- **Task success on any benchmark** (LIBERO, SimplerEnv, ManiSkill, real robot) — never completed.
- **pi0 / pi0.5 / GR00T native path parity** — only SmolVLA has a native path today. Decomposed paths for these were exported but never cos-compared.
- **Cold-start time** — first-action latency after load. Not measured.

---

## Methodology notes

**Shared noise.** All cos comparisons feed *the same noise tensor* to both PyTorch and ONNX paths. Flow matching is deterministic given noise, so a non-zero L2 with shared noise is a real discrepancy, not RNG. Without shared noise, the comparison is meaningless (see `04_iteration_lessons/shared_noise_discipline.md`).

**Cos_sim threshold.** cos = 0.977 per-step is *catastrophic* after 10 integration steps (observed: final cos = -0.24). Per-step parity must be > 0.999 to survive composition. This is why the decomposed path died and the native path is the production target.

**Reproducer discipline.** Every row in Verified has: script path, env flags, commit SHA. If you can't paste the command and get the number back, the number doesn't belong in Verified.

---

## Update protocol

When a new number gets measured:
1. If reproducible → add a row to **Verified** with reproducer + commit.
2. If ran but suspect → add to **Unverified** with the reason.
3. If superseded → move to Unverified with `archived` status, link to the replacement.
4. If targeted but not yet run → add to **Unmeasured** as a tracked gap.

When citing a number externally (README, tweet, pitch, blog): pull it from **Verified** only. Nothing else.
