# Research: VLM Prefix Encoder for reflex export/serve

**Date:** 2026-04-16
**Backend:** claude-native-teams (Explore agent)
**Scope:** Export VLM backbone as separate ONNX with KV-cache output; wire into serve so /act returns task-conditioned actions

## Summary

Current v0.1 exports ONLY the action expert stack with dummy-zero VLM conditioning. SmolVLA (line 132) and GR00T (line 227) allocate zeros for cross-attention KV; pi0/pi0.5 have no cross-attention layers at all (cross_indices=[]). The server's predict() ignores image+instruction inputs entirely. To fix: export VLM backbone as separate ONNX, add vlm_kv as input to expert ONNX, wire server to run VLM then expert per request.

## Key files

| File | Finding |
|------|---------|
| smolvla_exporter.py:122-141 | dummy_kv = torch.zeros(...) for cross-attn; VLM never exported |
| smolvla_exporter.py:231 | vlm_kv_dim=512, cross_attn_layers=[1,5,9,13] (4 of 16 layers) |
| pi0_exporter.py:121-126 | cross_indices=[] — NO cross-attn; VLM prefix consumed via prefix-KV at export time |
| gr00t_exporter.py:220-227 | vlm_kv_normed zeros for cross-attn; vlm_kv_dim=2048 |
| runtime/server.py:316-380 | predict() ignores image/instruction; runs expert with random noisy_actions only |
| inference.py:61-87 | flow_matching_denoise takes conditioning param but never uses it |

## SmolVLA architecture (where to cut)

1. Image → SigLIP (384px) → [batch, seq, 512]
2. Text → SmolLM2 tokenizer → embeddings
3. Concatenate → SmolLM2 transformer (16 layers) → [batch, seq, 512]
4. **CUT HERE** → export as vlm_prefix.onnx
5. VLM output → projected → feeds cross-attn in action expert (layers 1,5,9,13)
6. Expert denoises noisy_actions → velocity → Euler step → action chunk

## Per-model VLM status

| Model | Cross-attn? | VLM dims | Current conditioning | Needs prefix export? |
|-------|------------|----------|---------------------|---------------------|
| SmolVLA | Yes (4 layers) | 512 | Dummy zeros | YES (priority) |
| pi0 | No | N/A | N/A | No — different mechanism |
| pi0.5 | No | N/A | N/A | No — different mechanism |
| GR00T | Yes (16/32 blocks) | 2048 | Dummy zeros | YES (v2) |

## KV-cache feasibility

ONNX opset 19 + ORT 1.18+ supports KV-cache I/O via optional tensors. SmolLM2's 16 layers each output k_cache + v_cache. Server caches these keyed by (instruction_hash, image_hash) and reuses across chunks. Feasible but adds memory management complexity.

## What must change

1. **New:** src/reflex/exporters/vlm_prefix_exporter.py — exports SigLIP + SmolLM2 as separate ONNX
2. **Update:** smolvla_exporter.py — add vlm_kv as dynamic ONNX input to expert stack (replacing dummy zeros)
3. **Update:** runtime/server.py — add _vlm_session, run VLM prefix per request, pass vlm_kv to expert
4. **Update:** reflex_config.json schema — add vlm_prefix_onnx field
5. **Later:** gr00t_exporter.py — same pattern but vlm_kv_dim=2048
6. **Later:** _pytorch_backend.py + _onnx_backend.py — wire VLM into validate harness

## Risks

- Dynamic shapes in VLM (variable seq length) may break TRT
- Batching: each request needs separate VLM encode, negating batch throughput gains
- Backward compat: old expert ONNX files lack vlm_kv input; need config versioning
- Memory: caching SmolLM2 KV across 16 layers at FP16 = ~32MB per cached prefix
- pi0/pi0.5 consume VLM via a different mechanism entirely (prefix-KV baked at export, not runtime cross-attn)

## Scoping decision for v1

SmolVLA only. No KV-cache persistence in v1 (recompute VLM per call). KV caching deferred to v1.1. GR00T deferred to v2. pi0/pi0.5 don't need this (no cross-attn).
