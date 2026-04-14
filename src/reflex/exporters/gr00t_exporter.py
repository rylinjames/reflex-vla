"""GR00T N1.6 export pipeline — STUB.

GR00T N1.6 (nvidia/GR00T-N1.6-3B, 3.29B params) is architecturally
meaningfully different from SmolVLA / pi0 / pi0.5 and needs dedicated
modules rather than sharing the ExpertGQALayer class.

Repo: nvidia/GR00T-N1.6-3B
Size: 6.57 GB (bf16, 2 safetensors shards)

# Architecture reference (do not delete — needed for future implementation)

## Action expert (DiT) — from config.json `diffusion_model_cfg`
- num_layers: 32 (N1.5 was 16)
- num_attention_heads: 32, head_dim: 48 → hidden=1536
- action_horizon: 50, max_action_dim: 128, max_state_dim: 128
- norm_type: "ada_norm" (AdaLN, NOT AdaRMSNorm — only 2 chunks)
- activation_fn: "gelu-approximate" → GEGLU
- interleave_self_attention: true (even idx = cross-attn to VLM, odd = self-attn)
- use_alternate_vl_dit: true (AlternateVLDiT — state_dict identical to DiT parent)
- Flow matching: Beta(1.5, 1.0), **4 inference steps** (vs 10 for SmolVLA/pi0)

## State-dict key roots (NO `model.` wrapper — unlike SmolVLA/pi0)
```
action_head.model.transformer_blocks.{0..31}.
  attn1.to_q.{weight,bias}              # diffusers Attention, NOT q_proj/k_proj
  attn1.to_k.{weight,bias}              # on even blocks, to_k.weight = [1536, 2048] (cross-attn KV dim)
  attn1.to_v.{weight,bias}
  attn1.to_out.0.{weight,bias}
  ff.net.0.proj.{weight,bias}            # GEGLU: [2*ff_inner, 1536] — split along dim 0
  ff.net.2.{weight,bias}                 # final FF linear
  norm1.linear.{weight,bias}             # AdaLN: Linear(1536, 2*1536) for scale+shift (NOT 3*)
  (norm3 is non-affine LN, no params)
action_head.model.timestep_encoder.timestep_embedder.linear_{1,2}.{weight,bias}
action_head.model.proj_out_1.{weight,bias}     # Final AdaLN scale+shift: [3072, 1536]
action_head.model.proj_out_2.{weight,bias}     # Output head: [1024, 1536]
action_head.vlln.{weight,bias}                 # LayerNorm(2048) on backbone features
action_head.position_embedding.weight          # Embedding(1024, 1536)
action_head.state_encoder.layer{1,2}.{W,b}     # Per-embodiment MLP (32 embodiments)
action_head.action_encoder.W{1,2,3}.{W,b}
action_head.action_decoder.layer{1,2}.{W,b}
backbone.model.language_model.model.layers.{0..15}.*   # Qwen3, 16 layers
  self_attn.{q,k,v,o}_proj.weight + q_norm.weight + k_norm.weight
  input_layernorm.weight, post_attention_layernorm.weight
  mlp.{gate,up,down}_proj.weight
backbone.model.vision_model.vision_model.encoder.layers.{0..21}.*  # SigLIP2, 22 layers
backbone.model.mlp1.{0,1,3}.{weight,bias}      # connector (LN + Linear + GELU + Linear)
```

## What needs to be built to support GR00T

1. **DiffusersGQALayer / DiffusersMHALayer module**
   - `attn1.to_q/to_k/to_v/to_out.0` naming
   - Variable K/V input dim (cross-attn blocks have kv_in=2048, self-attn have kv_in=1536)
   - BIAS ENABLED on projections (unlike SmolVLA/pi0 which have bias=False)

2. **GEGLU MLP**: `ff.net.0.proj.weight[: ff_inner]` + `ff.net.0.proj.weight[ff_inner:]` →
   `gelu_approx(gate) * value`, then `ff.net.2`.

3. **AdaLN variant** (2-chunk, not 3-chunk): `norm1.linear(temb) → chunk(2) → scale, shift`.
   Apply as: `x = (1 + scale) * LayerNorm(x) + shift` where LayerNorm has no learnable params.

4. **Alternating cross-attn with VLM tokens at 2048-dim**. Even blocks need a KV placeholder
   of shape [b, seq_kv, 2048]; odd blocks are pure self-attn.

5. **Final AdaLN + proj_out_2**: `proj_out_1(SiLU(temb)) → chunk(2)` modulates pre-LayerNorm,
   then `proj_out_2` projects 1536 → 1024 action logits (then action_decoder goes 1024→action_dim).

6. **Embodiment selection**: state_encoder / action_encoder / action_decoder weights carry a
   leading dim of 32 (one slot per embodiment). At export time, select the target embodiment id
   and slice the weights down to a single-embodiment module.

## Estimate
~600-800 LoC of new code for GR00T-specific layers + stack + builder.
Budget: 4-6 hours for v1 (no embodiment selection polish, single target embodiment only).
"""

from __future__ import annotations

from typing import Any

import torch

from reflex.config import ExportConfig


def build_gr00t_expert_stack(
    state_dict: dict[str, torch.Tensor],
    embodiment_id: int = 0,
) -> tuple[Any, dict]:
    """Placeholder — raises until GR00T layers are implemented."""
    raise NotImplementedError(
        "GR00T N1.6 export is not yet implemented. See gr00t_exporter.py docstring "
        "for the checklist — diffusers-style attn, GEGLU, AdaLN (2-chunk), and "
        "alternating cross/self-attn with VLM-KV at 2048-dim. Tracking target: v0.3."
    )


def export_gr00t(
    config: ExportConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    raise NotImplementedError(
        "GR00T N1.6 export is not yet implemented. Supported today: smolvla, pi0, pi05."
    )
