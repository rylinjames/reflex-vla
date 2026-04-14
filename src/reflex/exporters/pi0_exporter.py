"""pi0 / pi0.5 export pipeline: checkpoint → ONNX → TRT engines.

Parallels smolvla_exporter.py but for the PaliGemma + Gemma300M architecture
from Physical Intelligence (lerobot/pi0_base, lerobot/pi05_base).

Key differences from SmolVLA:
- State-dict prefix: `paligemma_with_expert.gemma_expert.model.layers.N.*`
- Top-level action projections (no `model.` prefix): `action_in_proj.weight`, etc.
- GQA config: 16 Q heads, 2 KV heads, head_dim=128 (vs SmolVLA's 15/5/64)
- Expert hidden = 1024 (vs 720)
- 18 expert layers (vs 16)
- pi0 has `state_proj`; pi0.5 uses AdaRMSNorm (not yet supported by this exporter)

Only pi0 is fully supported in v0.1 of this exporter.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from reflex.config import ExportConfig, get_hardware_profile
from reflex.checkpoint import load_checkpoint
from reflex.exporters.onnx_export import export_module_to_onnx, optimize_onnx
from reflex.exporters.smolvla_exporter import ExpertGQALayer, ExpertStack
from reflex.exporters.trt_build import build_engine, check_trtexec
from reflex.decompose import DecomposedRMSNorm

logger = logging.getLogger(__name__)


PI0_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
PI0_ACTION_KEYS = {
    # Top-level (no `model.` prefix in pi0 checkpoints)
    "in_w": "action_in_proj.weight",
    "in_b": "action_in_proj.bias",
    "t_in_w": "action_time_mlp_in.weight",
    "t_in_b": "action_time_mlp_in.bias",
    "t_out_w": "action_time_mlp_out.weight",
    "t_out_b": "action_time_mlp_out.bias",
    "out_w": "action_out_proj.weight",
    "out_b": "action_out_proj.bias",
}


def build_pi0_expert_stack(
    state_dict: dict[str, torch.Tensor],
    head_dim: int = 128,
) -> tuple[ExpertStack, dict]:
    """Build the full pi0 expert stack from state_dict.

    Reuses ExpertStack/ExpertGQALayer from smolvla_exporter — the module is
    general enough, only the state-dict key paths differ.
    """
    # 1. Verify this is pi0 (not pi0.5)
    sample_layernorm_keys = [k for k in state_dict.keys() if "input_layernorm" in k]
    if sample_layernorm_keys and any("dense" in k for k in sample_layernorm_keys):
        raise NotImplementedError(
            "pi0.5 uses AdaRMSNorm (time-conditioned). Not yet supported — "
            "use pi0_base (lerobot/pi0_base) for now."
        )

    # 2. Extract suffix (action head) weights — top-level keys
    missing = [name for name in PI0_ACTION_KEYS.values() if name not in state_dict]
    if missing:
        raise ValueError(f"Missing expected pi0 keys: {missing[:3]}...")

    expert_hidden = state_dict[PI0_ACTION_KEYS["in_w"]].shape[0]
    action_dim = state_dict[PI0_ACTION_KEYS["in_w"]].shape[1]

    # 3. Find expert layers
    base_prefix = PI0_EXPERT_PREFIX
    layer_indices = set()
    for k in state_dict.keys():
        if not k.startswith(base_prefix + "layers."):
            continue
        parts = k[len(base_prefix):].split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_indices.add(int(parts[1]))
    if not layer_indices:
        raise ValueError(f"No expert layers found with prefix {base_prefix}")
    num_layers = max(layer_indices) + 1

    # 4. Detect GQA shapes from layer 0
    q_shape = state_dict[f"{base_prefix}layers.0.self_attn.q_proj.weight"].shape
    k_shape = state_dict[f"{base_prefix}layers.0.self_attn.k_proj.weight"].shape
    gate_shape = state_dict[f"{base_prefix}layers.0.mlp.gate_proj.weight"].shape
    nq = q_shape[0] // head_dim
    nkv = k_shape[0] // head_dim
    inter = gate_shape[0]

    logger.info(
        "pi0 expert: %d layers, hidden=%d, nq=%d, nkv=%d, head_dim=%d, inter=%d, action_dim=%d",
        num_layers, expert_hidden, nq, nkv, head_dim, inter, action_dim,
    )

    # 5. Build layers — pi0 expert does NOT use cross-attention (unlike SmolVLA)
    # All layers are self-attention over action tokens; VLM prefix is consumed
    # via prefix-KV cache, which we skip for the expert-only export.
    layers = []
    cross_indices = []  # empty for pi0
    vlm_kv_dim = 0
    for i in range(num_layers):
        prefix = f"{base_prefix}layers.{i}"
        kv_in = state_dict[f"{prefix}.self_attn.k_proj.weight"].shape[1]
        # pi0 expert self-attn has kv_in == expert_hidden (no cross-attn)
        is_cross = kv_in != expert_hidden
        if is_cross:
            cross_indices.append(i)
            vlm_kv_dim = kv_in

        layer = ExpertGQALayer(
            expert_hidden, nq, nkv, head_dim, inter,
            kv_in=kv_in if is_cross else None,
        )
        layer_sd = {
            "input_layernorm.weight": state_dict[f"{prefix}.input_layernorm.weight"],
            "post_attention_layernorm.weight": state_dict[f"{prefix}.post_attention_layernorm.weight"],
            "q_proj.weight": state_dict[f"{prefix}.self_attn.q_proj.weight"],
            "k_proj.weight": state_dict[f"{prefix}.self_attn.k_proj.weight"],
            "v_proj.weight": state_dict[f"{prefix}.self_attn.v_proj.weight"],
            "o_proj.weight": state_dict[f"{prefix}.self_attn.o_proj.weight"],
            "gate_proj.weight": state_dict[f"{prefix}.mlp.gate_proj.weight"],
            "up_proj.weight": state_dict[f"{prefix}.mlp.up_proj.weight"],
            "down_proj.weight": state_dict[f"{prefix}.mlp.down_proj.weight"],
        }
        layer.load_state_dict(layer_sd, strict=False)
        layers.append(layer)

    # 6. Final RMSNorm
    final_norm_key = f"{base_prefix}norm.weight"
    final_norm_w = state_dict.get(final_norm_key, torch.ones(expert_hidden))

    # 7. Assemble stack
    stack = ExpertStack(
        layers=layers,
        expert_hidden=expert_hidden,
        action_dim=action_dim,
        cross_indices=cross_indices,
        vlm_kv_dim=vlm_kv_dim,
        suffix_weights={
            "in_w": state_dict[PI0_ACTION_KEYS["in_w"]],
            "in_b": state_dict[PI0_ACTION_KEYS["in_b"]],
            "t_in_w": state_dict[PI0_ACTION_KEYS["t_in_w"]],
            "t_in_b": state_dict[PI0_ACTION_KEYS["t_in_b"]],
            "t_out_w": state_dict[PI0_ACTION_KEYS["t_out_w"]],
            "t_out_b": state_dict[PI0_ACTION_KEYS["t_out_b"]],
        },
        action_proj_weights={
            "w": state_dict[PI0_ACTION_KEYS["out_w"]],
            "b": state_dict[PI0_ACTION_KEYS["out_b"]],
        },
        final_norm_weight=final_norm_w,
    )
    stack.eval()

    metadata = {
        "expert_hidden": expert_hidden,
        "action_dim": action_dim,
        "num_layers": num_layers,
        "n_q_heads": nq,
        "n_kv_heads": nkv,
        "head_dim": head_dim,
        "intermediate": inter,
        "cross_attn_layers": cross_indices,
        "vlm_kv_dim": vlm_kv_dim,
        "total_params_m": sum(p.numel() for p in stack.parameters()) / 1e6,
    }
    return stack, metadata


def export_pi0(
    config: ExportConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Full pi0 export pipeline.

    Args:
        config: export config
        state_dict: optionally pre-loaded state dict (avoids re-downloading 3.5GB)
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = get_hardware_profile(config.target)
    result = {"status": "ok", "files": {}, "metadata": {}}

    # 1. Load checkpoint (if not provided)
    if state_dict is None:
        logger.info("Loading pi0 checkpoint: %s", config.model_id)
        state_dict, model_config = load_checkpoint(config.model_id)
    total_params = sum(v.numel() for v in state_dict.values())
    logger.info("Loaded %d tensors, %.1fM params", len(state_dict), total_params / 1e6)

    # 2. Build expert stack
    logger.info("Building pi0 expert stack...")
    expert_stack, expert_meta = build_pi0_expert_stack(state_dict, head_dim=128)
    result["metadata"]["expert"] = expert_meta

    # 3. Export to ONNX
    logger.info("Exporting expert stack to ONNX...")
    action_dim = expert_meta["action_dim"]
    chunk_size = config.action_chunk_size

    dummy_actions = torch.randn(1, chunk_size, action_dim)
    dummy_time = torch.tensor([0.5])
    dummy_pos = torch.arange(chunk_size).unsqueeze(0)

    expert_onnx = output_dir / "expert_stack.onnx"
    export_module_to_onnx(
        expert_stack,
        (dummy_actions, dummy_time, dummy_pos),
        expert_onnx,
        input_names=["noisy_actions", "timestep", "position_ids"],
        output_names=["velocity"],
        dynamic_axes={"noisy_actions": {0: "batch"}, "timestep": {0: "batch"}, "position_ids": {0: "batch"}},
        opset_version=config.opset,
    )
    optimize_onnx(expert_onnx)
    result["files"]["expert_onnx"] = str(expert_onnx)

    # 4. Validate ONNX
    if config.validate:
        logger.info("Validating ONNX export...")
        try:
            import onnxruntime as ort
            import numpy as np

            sess = ort.InferenceSession(str(expert_onnx))
            ort_out = sess.run(None, {
                "noisy_actions": dummy_actions.numpy(),
                "timestep": dummy_time.numpy(),
                "position_ids": dummy_pos.numpy().astype(np.int64),
            })[0]
            torch_out = expert_stack(dummy_actions, dummy_time, dummy_pos).detach().numpy()
            max_diff = float(np.abs(ort_out - torch_out).max())
            result["metadata"]["onnx_validation"] = {"max_diff": max_diff, "passed": max_diff < 0.01}
            logger.info("ONNX validation: max_diff=%.2e (%s)", max_diff, "PASS" if max_diff < 0.01 else "FAIL")
        except ImportError:
            logger.warning("onnxruntime not installed, skipping validation")

    # 5. Build TRT engine if available
    if check_trtexec():
        logger.info("Building TensorRT engine...")
        expert_trt = output_dir / "expert_stack.trt"
        try:
            build_engine(expert_onnx, expert_trt, hardware)
            result["files"]["expert_trt"] = str(expert_trt)
        except RuntimeError as e:
            logger.warning("TRT build failed: %s", e)

    # 6. Save config
    export_config = {
        "model_id": config.model_id,
        "model_type": "pi0",
        "target": config.target,
        "precision": config.precision,
        "opset": config.opset,
        "num_denoising_steps": config.num_denoising_steps,
        "action_chunk_size": config.action_chunk_size,
        "action_dim": action_dim,
        "hardware": {
            "name": hardware.name,
            "memory_gb": hardware.memory_gb,
            "fp8": hardware.fp8_support,
            "precision": hardware.trt_precision,
        },
        "expert": expert_meta,
    }
    config_path = output_dir / "reflex_config.json"
    config_path.write_text(json.dumps(export_config, indent=2))
    result["files"]["config"] = str(config_path)
    return result
