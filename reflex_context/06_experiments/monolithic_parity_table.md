# Monolithic ONNX parity — all four VLAs side-by-side (2026-04-19)

Single-source comparison of what we shipped in v0.2 for each supported VLA. Every number here also appears in `measured_numbers.md` Verified section; this doc is the "what does success look like, at a glance" view.

---

## Canonical parity table

All runs: Modal A10G or A100-40GB, shared seeded inputs (`torch.manual_seed(42)`), CPU ONNX Runtime. Compared against PyTorch eager with identical noise/inputs.

| Model | HF ID | Type | Reference forward | first-action cos | first-action max_abs | full-chunk cos | full-chunk max_abs | Reproducer |
|---|---|---|---|---|---|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | flow matching | `sample_actions(num_steps=10)` | +1.0000000 | **5.96e-07** | +1.0000000 | 3.70e-06 | `modal run scripts/modal_smolvla_monolithic_export.py --parity --num-steps 10` |
| pi0 | `lerobot/pi0_base` | flow matching | `sample_actions(num_steps=10)` | +1.0000000 | **2.09e-07** | +1.0000000 | 4.17e-07 | `modal run scripts/modal_pi0_monolithic_export.py --parity --num-steps 10` |
| pi0.5 | `lerobot/pi05_base` | flow matching | `sample_actions(num_steps=10)` | +1.0000000 | **2.38e-07** | +1.0000000 | 6.56e-07 | `modal run scripts/modal_pi05_monolithic_export.py --parity --num-steps 10` |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | DDPM DiT (single-step) | `GR00TFullStack.forward` | +1.0000000 | **8.34e-07** | +1.0000000 | 3.70e-06 | `modal run scripts/modal_gr00t_monolithic_export.py --parity` |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | DDPM DiT (4-step loop) | Python loop over PyTorch ref | +1.0000000 | **4.77e-07** | +1.0000000 | 1.91e-06 | `modal run scripts/modal_gr00t_monolithic_export.py --loop` |

All five rows at cos=+1.0000000 — machine precision. max_abs values span 2.09e-07 to 8.34e-07 on first-action, mostly determined by model size and operator variance in the fp32 path.

---

## Model size comparison

| Model | Params | Monolithic ONNX | External data | Total on disk |
|---|---|---|---|---|
| SmolVLA | 450M | 1.6 MB | 1.6 GB | 1.6 GB |
| pi0 | 3.5B | 6 MB | 12.5 GB | 12.5 GB |
| pi0.5 | 3.6B | 6 MB | 13.0 GB | 13.0 GB |
| GR00T N1.6 (DiT + encoder/decoder, embodiment=0) | 1.1B (of 3.29B) | 2.1 MB | 4.4 GB | 4.4 GB |

**Why GR00T is smaller than params suggest.** The 3.29B model size includes all 32 per-embodiment action encoders/decoders + the Eagle VLM backbone. Our export pins embodiment=0 and stubs VLM conditioning as zero, so only the shared DiT (32 blocks × 1536 hidden × 32 heads) + one embodiment's encoder/decoder (~300M total) is shipped.

**Why pi-family models are larger than their stated params.** The monolithic ONNX includes PaliGemma (2.5B vision+language) + the 500M-700M action expert + embeddings + norms stored in fp32. Actual VLA-only inference has ~6GB fp16 footprint; fp32 ONNX is 2× that.

---

## Architecture type → export pattern

| Architecture | Example | Export pattern | Wrap complexity |
|---|---|---|---|
| Flow-matching decoder-only (Gemma-family + action expert) | pi0, pi0.5 | Monolithic `sample_actions(num_steps=N)` unrolled | **3-patch stack required** (F.pad mask + frozen DynamicLayer.update + past_kv.get_seq_length) |
| Flow-matching decoder-only (SmolLM2 + action expert) | SmolVLA | Monolithic `sample_actions(num_steps=N)` unrolled | SmolVLM2 `torch.where` dtype fix + post-export Cast insertion (simpler than pi0) |
| DDPM DiT (standalone DiT + AdaLN + separate VLM) | GR00T N1.6 | Per-step `GR00TFullStack.forward` + runtime loop | None — plain `torch.onnx.export(opset=19)` |

Full diagnostic for each pattern in the architecture docs; 3-patch stack details in `02_bugs_fixed/pi0_num_steps_10_three_patch_stack.md`.

---

## Host-loop vs baked-loop parity

GR00T is the only VLA where the denoise loop lives external to the ONNX. Two separate parity tests verify both levels:

1. **Single-step:** ONNX(noise, t, pos) vs `GR00TFullStack.forward(noise, t, pos)` → cos=+1.0 max_abs=8.34e-07
2. **4-step loop:** Python `for _ in range(4): x = x + dt * ORT.run(x, t, pos)` vs same loop over PyTorch → cos=+1.0 max_abs=4.77e-07

The per-step machine precision composes to end-of-chunk machine precision across the 4-step DDIM integration. Confirms empirically: if per-step is machine precision, the loop is machine precision (no accumulated drift).

For pi0 / pi0.5 / SmolVLA the loop is baked into the ONNX graph (10-step Euler unrolled), so the single parity test IS the end-of-chunk test.

---

## Known limitations (documented honestly in launch materials)

**GR00T VLM conditioning stubbed.** The current GR00T export feeds a zero tensor as VLM-KV input to the cross-attn blocks. This makes PyTorch and ONNX behave identically (both see zeros), so the parity test is valid. But full multi-modal control requires exporting the Eagle-2-HG VLM backbone + chaining its KV output into the DiT — v0.3 item.

**Orin Nano 8GB fit.** pi0 (12.5 GB) and pi0.5 (13 GB) don't fit on Orin Nano 8GB at FP32 or FP16 once activations + OS are counted. SmolVLA (1.6 GB) and GR00T (4.4 GB) are small enough, but FP16 engine rebuild + Orin Nano validation is v0.3.

**Jetson latency numbers.** None. CloudJetson Orin Nano is waitlisted; no hardware access yet. `reflex bench` reproduces on any GPU, but the launch table currently has no Jetson row.

---

## Reproducibility checklist

Everything in this table reproduces from a fresh Modal workspace:

```bash
# 1. Clone repo, set HF_TOKEN (for pi0/pi05/GR00T gated models)
export HF_TOKEN=hf_xxxxx

# 2. Export + parity for each model (each ~5-10 min on A10G, ~3 min on A100)
modal run scripts/modal_smolvla_monolithic_export.py --parity --num-steps 10
modal run scripts/modal_pi0_monolithic_export.py     --parity --num-steps 10
modal run scripts/modal_pi05_monolithic_export.py    --parity --num-steps 10
modal run scripts/modal_gr00t_monolithic_export.py   --parity
modal run scripts/modal_gr00t_monolithic_export.py   --loop
```

Each command prints a PASS/FAIL verdict + exact cos + max_abs numbers. The numbers above are from the runs that landed this table (2026-04-19).

---

## References

- `measured_numbers.md` — full claim ledger (Verified / Unverified / Unmeasured)
- `01_architecture/pi0_monolithic_wrap_pattern.md` — flow-matching wrap pattern
- `01_architecture/gr00t_ddpm_dit_vs_flow_matching.md` — DDPM DiT pattern
- `02_bugs_fixed/pi0_num_steps_10_three_patch_stack.md` — the 3-patch stack, each patch explained
- `05_sessions/2026-04-19_all_four_vlas.md` — session narrative of landing all four
