# RPI Phase 1 Summary — VLM prefix encoder research

**Date:** 2026-04-16
**Research:** .agents/research/2026-04-16-vlm-prefix-encoder.md

## Key findings

- v0.1 exports action expert only with dummy-zero VLM conditioning (smolvla_exporter.py:132, gr00t_exporter.py:227)
- SmolVLA has cross-attention at layers [1,5,9,13] with vlm_kv_dim=512; needs real VLM prefix
- pi0/pi0.5 have NO cross-attention (cross_indices=[]) — different mechanism, skip in v1
- GR00T has cross-attention with vlm_kv_dim=2048 — defer to v2
- Server predict() ignores image/instruction entirely (server.py:342-380)
- KV-cache in opset 19 is feasible via ORT 1.18+

## Scope for this /rpi cycle

SmolVLA only. 4 deliverables:
1. VLM prefix exporter (SigLIP + SmolLM2 → vlm_prefix.onnx)
2. Expert ONNX updated to accept vlm_kv input (replacing zeros)
3. Server wired to run VLM then expert per request
4. Config schema updated

No KV-cache persistence (recompute per call). No GR00T. No pi0/pi0.5 (not needed).

## Ready for /plan
