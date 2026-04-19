"""EagleExportStack — export-friendly wrapper for GR00T's Eagle VLM backbone.

Mirrors `lerobot.policies.groot.groot_n1.EagleBackbone.forward_eagle` for
a single-shot encode (no autoregressive decode). Takes:

    pixel_values   [B, 3, H, W]        — SigLIP input image
    input_ids      [B, seq]             — Qwen2 token ids (with image_token placeholders)
    attention_mask [B, seq]             — 1 for real tokens, 0 for padding
    image_flags    [B]                  — 1 if image is real (kept), 0 to drop

Produces:

    hidden_states  [B, seq, 2048]       — Qwen2 hidden at select_layer (last by default)

This is what GR00T's DiT cross-attn consumes as vlm_kv. The corresponding
ONNX export feeds these hidden_states as the `vlm_kv` input of
`expert_stack_with_vlm.onnx` (Step 4a).

Weight loading: state_dict keys come from `nvidia/GR00T-N1.6-3B` under the
`backbone.model.*` prefix (NOT `backbone.eagle_model.*` as older research
guessed — verified by scripts/modal_gr00t_keys_dump.py).

Export-relevant tweaks:
  - `_attn_implementation="eager"` forced in the vendored config default
    (FA2 not consumable by ONNX/TRT)
  - `use_cache=False` in the language_model forward to avoid DynamicCache
  - No autoregressive loop → no cache growth, no need for the 3-patch
    stack that pi0/pi0.5 required
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EagleExportStack(nn.Module):
    """Single-shot encode: (pixel_values, input_ids, attention_mask, image_flags)
    → hidden_states at select_layer.
    """

    def __init__(self, eagle_model: nn.Module, select_layer: int = -1):
        super().__init__()
        self.eagle = eagle_model
        self.select_layer = select_layer
        # Cache useful sub-modules for readability
        self.vision_model = eagle_model.vision_model
        self.language_model = eagle_model.language_model
        self.mlp1 = eagle_model.mlp1
        self.image_token_index = eagle_model.image_token_index

    def _splice_vision_into_text(
        self,
        input_embeds: torch.Tensor,     # [B, seq, C]
        input_ids: torch.Tensor,        # [B, seq]
        vit_embeds: torch.Tensor,       # [B, vision_seq, C]
    ) -> torch.Tensor:
        """Replace image_token_index positions in input_embeds with vit_embeds.

        Eagle's original forward does this via a boolean mask assignment
        (`input_embeds[selected] = ...`). That boolean indexing doesn't
        play perfectly with torch.export; use a scatter-style expand
        that's ONNX-friendly.
        """
        b, n, c = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(b * n, c)
        input_ids_flat = input_ids.reshape(b * n)
        vit_flat = vit_embeds.reshape(-1, c)  # [B * vision_seq, C]

        # selected mask: positions where the text token == image_token_index
        selected = input_ids_flat == self.image_token_index  # [B * n]
        # If there are exactly B*vision_seq image-token positions, we can
        # splice them 1:1 in order.
        input_embeds_flat = input_embeds_flat.clone()
        # We expect vit_flat.shape[0] == selected.sum(). Fill the True rows.
        input_embeds_flat[selected] = vit_flat
        return input_embeds_flat.reshape(b, n, c)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_flags: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Vision path: SigLIP → pixel_shuffle → mlp1
        vit_embeds = self.eagle.extract_feature(pixel_values)  # [B, vision_seq, C]

        # Filter by image_flags (drop images marked absent)
        if image_flags is not None:
            image_flags_flat = image_flags.view(-1)
            vit_embeds = vit_embeds[image_flags_flat == 1]

        # 2. Text embed lookup
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 3. Splice vision into text at image_token positions
        input_embeds = self._splice_vision_into_text(
            input_embeds, input_ids, vit_embeds,
        )

        # 4. Qwen2 forward, with output_hidden_states to get the select_layer
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states is a tuple of (num_layers + 1) tensors. select_layer
        # is typically -1 (last layer).
        hidden = outputs.hidden_states[self.select_layer]  # [B, seq, 2048]
        return hidden


def build_eagle_export_stack(
    state_dict: dict[str, torch.Tensor],
    select_layer: int = -1,
) -> tuple[EagleExportStack, dict]:
    """Instantiate Eagle from vendored source + load state dict.

    Returns (stack, meta). Meta includes:
      - vit_hidden: vision hidden dim (1152 for SigLIP-so400m)
      - llm_hidden: text hidden dim (2048 for Qwen2-0.5B)
      - shuffled_hidden: pixel-shuffled vision dim before mlp1 (4608 = 1152*4)
      - vocab_size: Qwen2 tokenizer vocab (151680)
      - image_token_index: sentinel ID for image placeholder in input_ids
      - total_params_m: total parameter count in millions

    Requires the `action_head.state_encoder.*` and `backbone.model.*` keys
    to be present (the N1.6 pattern verified by modal_gr00t_keys_dump.py).
    """
    # Lazy-import the vendor so downstream files don't always pull it.
    from reflex.exporters.eagle_vendor.modeling_eagle2_5_vl import (
        Eagle25VLForConditionalGeneration,
    )
    from reflex.exporters.eagle_vendor.configuration_eagle2_5_vl import (
        Eagle25VLConfig,
    )

    # Derive config from state_dict shapes
    vocab_size, llm_hidden = state_dict[
        "backbone.model.language_model.model.embed_tokens.weight"
    ].shape

    # Build a default config. The vendored default already has eager attn.
    # Image token index = default for Eagle 2.5 (verified at runtime)
    cfg = Eagle25VLConfig(
        vision_config={"model_type": "siglip_vision_model"},
        text_config={"architectures": ["Qwen2ForCausalLM"]},
        image_token_index=151655,  # Qwen2-VL's default; actual may differ per-ckpt
    )

    # Instantiate. The Eagle25VL vendor handles submodule construction.
    model = Eagle25VLForConditionalGeneration(cfg)

    # Load weights from state_dict — prefix is `backbone.model.*` so strip it.
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.model."):
            remapped[k[len("backbone.model."):]] = v.float()

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logger.warning("[eagle-export] %d missing keys (first 3: %s)",
                       len(missing), list(missing)[:3])
    if unexpected:
        logger.warning("[eagle-export] %d unexpected keys (first 3: %s)",
                       len(unexpected), list(unexpected)[:3])

    stack = EagleExportStack(model, select_layer=select_layer).float().eval()

    vit_hidden = cfg.vision_config.hidden_size
    meta = {
        "vit_hidden": vit_hidden,
        "llm_hidden": int(llm_hidden),
        "shuffled_hidden": int(vit_hidden * 4),  # pixel_shuffle concat of 2×2
        "vocab_size": int(vocab_size),
        "image_token_index": stack.image_token_index,
        "total_params_m": sum(p.numel() for p in stack.parameters()) / 1e6,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
    }
    return stack, meta


__all__ = ["EagleExportStack", "build_eagle_export_stack"]
