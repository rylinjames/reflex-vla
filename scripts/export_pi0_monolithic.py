"""Monolithic pi0 ONNX export — wraps sample_actions end-to-end.

Path C, per research consensus (openpi_on_thor, Tacoin, Dexmal all do this).
Past_key_values stays internal to the graph; the 10-step Euler denoise loop
is baked in at trace time. Uses dynamo=False (legacy tracer, mandatory for
Gemma-family in 2026 per pytorch #160761, #170172, #172903).

Output: single monolithic ONNX (~14GB with external data).
Inputs:  (images, img_masks, lang_tokens, lang_masks, state, noise)
Output:  actions [B, chunk_size, action_dim]
"""
import sys
import types

for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def _patch_pi0_for_transformers_457():
    """Match local_full_diff_pi0.py patches for lerobot 0.5.1 vs transformers 4.57.6."""
    from lerobot.policies.pi0 import modeling_pi0

    def patched_embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        if hasattr(image_outputs, "pooler_output"):
            features = image_outputs.pooler_output
        else:
            features = image_outputs
        features = features * self.paligemma.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    modeling_pi0.PaliGemmaWithExpertModel.embed_image = patched_embed_image


def _patch_create_causal_mask_kwarg():
    from transformers import masking_utils

    original = masking_utils.create_causal_mask

    def shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original(*args, **kwargs)

    masking_utils.create_causal_mask = shim
    from lerobot.policies import pi_gemma
    if hasattr(pi_gemma, "create_causal_mask"):
        pi_gemma.create_causal_mask = shim


_patch_pi0_for_transformers_457()
_patch_create_causal_mask_kwarg()


def _patch_gemma_attention_reshape():
    """Thor patch reimplemented: GemmaAttention.forward uses reshape(*, -1) which
    produces a rank-ambiguous tensor during torch.onnx.export legacy trace,
    breaking the subsequent cat/concat in later ops (AssertionError in
    symbolic_opset9.cat). Replace -1 with explicit num_heads * head_dim."""
    from transformers.models.gemma import modeling_gemma

    original_forward = modeling_gemma.GemmaAttention.forward

    def patched_forward(self, hidden_states, position_embeddings, attention_mask=None,
                         past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )

        # The fix: explicit num_heads * head_dim instead of -1
        attn_output = attn_output.reshape(*input_shape, self.config.num_attention_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_gemma.GemmaAttention.forward = patched_forward


_patch_gemma_attention_reshape()


class Pi0MonolithicWrapper(nn.Module):
    """Wraps PI0Pytorch.sample_actions for single-graph ONNX export.

    The sample_actions method internally computes prefix past_kv and runs
    a 10-step Euler denoise loop. With dynamo=False legacy tracer, the loop
    is unrolled and past_kv lives as Python variables through the trace
    (becoming tensor concat ops in the final graph).
    """

    def __init__(self, pi0_pytorch_model: nn.Module, num_steps: int = 10):
        super().__init__()
        self.model = pi0_pytorch_model
        self.num_steps = num_steps

    def forward(
        self,
        img_base: torch.Tensor,
        img_wrist_l: torch.Tensor,
        img_wrist_r: torch.Tensor,
        mask_base: torch.Tensor,
        mask_wrist_l: torch.Tensor,
        mask_wrist_r: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        # sample_actions expects images/img_masks as lists
        images = [img_base, img_wrist_l, img_wrist_r]
        img_masks = [mask_base, mask_wrist_l, mask_wrist_r]
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=self.num_steps,
        )
        return actions


def export_pi0_monolithic(output_path: Path, model_id: str = "lerobot/pi0_base"):
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    print(f"Loading PI0Policy: {model_id}")
    policy = PI0Policy.from_pretrained(model_id).eval().to(dtype=torch.float32).to("cpu")

    wrapper = Pi0MonolithicWrapper(policy.model, num_steps=10).eval()

    # Build dummy inputs matching pi0's expected tensor shapes
    # After _preprocess_images: images are [B, 3, 224, 224], img_masks are [B, 1] or similar
    B = 1
    cfg = policy.config
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = cfg.max_state_dim if hasattr(cfg, "max_state_dim") else 32

    # Use preprocessor to understand actual shapes
    img = torch.randn(B, 3, 224, 224, dtype=torch.float32)
    mask = torch.ones(B, dtype=torch.bool)
    lang_tokens = torch.randint(0, 257152, (B, 16), dtype=torch.long)
    lang_masks = torch.ones(B, 16, dtype=torch.bool)
    state = torch.randn(B, state_dim, dtype=torch.float32)
    noise = torch.randn(B, chunk, action_dim, dtype=torch.float32)

    dummy_inputs = (img, img, img, mask, mask, mask, lang_tokens, lang_masks, state, noise)

    print(f"Exporting to {output_path} (dynamo=False, opset=19)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=[
            "img_base", "img_wrist_l", "img_wrist_r",
            "mask_base", "mask_wrist_l", "mask_wrist_r",
            "lang_tokens", "lang_masks",
            "state", "noise",
        ],
        output_names=["actions"],
        dynamic_axes={
            "img_base": {0: "batch"},
            "img_wrist_l": {0: "batch"},
            "img_wrist_r": {0: "batch"},
            "mask_base": {0: "batch"},
            "mask_wrist_l": {0: "batch"},
            "mask_wrist_r": {0: "batch"},
            "lang_tokens": {0: "batch", 1: "seq"},
            "lang_masks": {0: "batch", 1: "seq"},
            "state": {0: "batch"},
            "noise": {0: "batch"},
            "actions": {0: "batch"},
        },
        opset_version=19,
        do_constant_folding=True,
        dynamo=False,  # MANDATORY for Gemma-family in 2026
    )
    print(f"Export complete: {output_path}")
    return output_path


if __name__ == "__main__":
    out = Path("/tmp/pi0_monolithic_onnx/model.onnx")
    export_pi0_monolithic(out)
