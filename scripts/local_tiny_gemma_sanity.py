"""Tiny-Gemma Optimum ONNX sanity test — end-to-end parity check.

Builds a minimal (0.4M param) Gemma, runs Optimum's text-generation-with-past
ONNX export, and verifies PyTorch vs ONNX output parity. Proves the Optimum
+ Gemma + per-layer KV extraction path works correctly before spending ~1
week on pi0-specific integration.

Result on 2026-04-17 first run:
  logits cos_sim = +1.00000000
  logits max_abs_diff = 7.15e-07
  present.0.key max_abs_diff = 1.79e-07
  VERDICT: PASS

This is load-bearing evidence for the pi0-onnx-parity plan — see
reflex_context/03_research/pi0_empirical_derisk_findings.md

Prerequisites:
  pip install 'optimum[onnxruntime]>=1.22'
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import GemmaConfig, GemmaForCausalLM


SAVE_DIR = Path("/tmp/tiny_gemma_sanity")


def build_and_save():
    """Create a minimal Gemma (2 layers, 128 hidden, 4/2 GQA, head_dim=32)."""
    cfg = GemmaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        vocab_size=1000,
    )
    model = GemmaForCausalLM(cfg).eval()

    pt_dir = SAVE_DIR / "tiny_gemma_pt"
    pt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(pt_dir)
    (pt_dir / "tokenizer_config.json").write_text(
        json.dumps({
            "tokenizer_class": "GemmaTokenizer",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        })
    )
    return pt_dir, model


def export_onnx(pt_dir: Path) -> Path:
    """Run optimum-cli export onnx with text-generation-with-past."""
    import subprocess
    onnx_dir = SAVE_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/optimum-cli", "export", "onnx",
        "--model", str(pt_dir),
        "--task", "text-generation-with-past",
        "--framework", "pt",
        str(onnx_dir),
    ]
    subprocess.check_call(cmd)
    return onnx_dir / "model.onnx"


def verify_parity(pt_dir: Path, onnx_path: Path):
    """Run PyTorch and ORT on the same inputs, compute cos + abs diff."""
    pt_model = GemmaForCausalLM.from_pretrained(pt_dir).eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (1, 8), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        pt_out = pt_model(input_ids, attention_mask=attention_mask, use_cache=True)
    pt_logits = pt_out.logits.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    # Empty past_key_values for a prefill pass
    for i in range(pt_model.config.num_hidden_layers):
        ort_inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, pt_model.config.num_key_value_heads, 0, pt_model.config.head_dim),
            dtype=np.float32,
        )
        ort_inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, pt_model.config.num_key_value_heads, 0, pt_model.config.head_dim),
            dtype=np.float32,
        )

    ort_outs = sess.run(None, ort_inputs)
    ort_logits = ort_outs[0]

    abs_diff = np.abs(pt_logits - ort_logits)
    cos = float(
        np.dot(pt_logits.flatten(), ort_logits.flatten())
        / (np.linalg.norm(pt_logits) * np.linalg.norm(ort_logits) + 1e-8)
    )

    print(f"logits max_abs_diff:  {abs_diff.max():.2e}")
    print(f"logits mean_abs_diff: {abs_diff.mean():.2e}")
    print(f"logits cos_sim:       {cos:+.8f}")

    # Compare present.0.key if available
    pt_kv = pt_out.past_key_values
    if hasattr(pt_kv, "layers"):
        pt_k0 = pt_kv.layers[0].keys.numpy()
        ort_k0 = ort_outs[1]
        k_diff = np.abs(pt_k0 - ort_k0).max()
        print(f"present.0.key diff:   {k_diff:.2e}")

    passed = abs_diff.max() < 1e-3 and cos > 0.9999
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("Step 1: building tiny Gemma...")
    pt_dir, _ = build_and_save()

    print("Step 2: exporting to ONNX via Optimum...")
    onnx_path = export_onnx(pt_dir)
    print(f"  exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    print("\nStep 3: verifying parity (PyTorch vs ORT)...")
    ok = verify_parity(pt_dir, onnx_path)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
