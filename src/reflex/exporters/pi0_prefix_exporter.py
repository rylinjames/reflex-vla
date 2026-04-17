"""pi0 prefix-KV export pipeline — orchestrates Optimum + our custom expert.

Goal: take `lerobot/pi0_base` (PaliGemma + Gemma expert + flow matching)
and produce ONNX files that TensorRT can compile on Jetson, preserving
cos >= 0.999 end-to-end parity vs the reference PyTorch policy.

Architecture (pi0 forward):
    image -> SigLIP vision_tower -> 256 vision features (1152-dim)
                                        |
                                        v
                               multi_modal_projector
                                     (1152->2048)
                                        |
    text -> embed_tokens (257152x2048)  |
                  |                     |
                  v                     v
                  +---------------------+
                            |
                            v
                  Gemma language_model (18 layers)
                  -> per-layer past_key_values (prefix KV)
                            |
                            v
                  gemma_expert (18 layers, separate)
                  + flow matching 10-step Euler
                            |
                            v
                        actions

Export strategy (after 2026-04-17 empirical de-risk):
    1. vision_encoder.onnx     -- Optimum SigLIP (feature-extraction)
    2. multi_modal_projector.onnx -- tiny Linear (1152->2048)
    3. text_embedder.onnx      -- extracted embed_tokens (tied w/ lm_head)
    4. decoder_prefill.onnx    -- Optimum Gemma (text-generation-with-past)
    5. expert_stack.onnx       -- reuse pi0_exporter.py

Each stage verified independently at cos >= 0.9999:
- vision_encoder: Verified 2026-04-17 on pi0_base subset (cos=+0.99999994)
- decoder_prefill: Verified 2026-04-17 on pi0_base subset (cos=+0.99999994)

See reflex_context/03_research/pi0_empirical_derisk_findings.md.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open

logger = logging.getLogger(__name__)


# Pi0 state-dict prefixes (confirmed 2026-04-17 on lerobot/pi0_base)
PI0_VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."
PI0_LANGUAGE_PREFIX = "paligemma_with_expert.paligemma.model.language_model."
PI0_PROJECTOR_KEYS = {
    "weight": "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
    "bias": "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias",
}
PI0_LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"  # tied with embed_tokens


# Default architecture params (confirmed for pi0_base, matches google/paligemma-3b-pt-224)
DEFAULT_SIGLIP_CFG = dict(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=27,
    num_attention_heads=16,
    num_channels=3,
    image_size=224,
    patch_size=14,
    hidden_act="gelu_pytorch_tanh",
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
)
DEFAULT_GEMMA_CFG = dict(
    vocab_size=257152,  # PaliGemma tokenizer
    hidden_size=2048,
    intermediate_size=16384,
    num_hidden_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    max_position_embeddings=8192,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
)


def load_pi0_state_dict(model_id: str = "lerobot/pi0_base") -> dict[str, torch.Tensor]:
    """Load the full pi0 state dict from HF cache (or download)."""
    repo_dir = Path(snapshot_download(model_id))
    sf_files = sorted(repo_dir.glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"No safetensors in {repo_dir}")

    state = {}
    for f in sf_files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                state[k] = sf.get_tensor(k)
    logger.info("Loaded pi0 state dict: %d tensors from %s", len(state), model_id)
    return state


def build_siglip_dir(state: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Extract SigLIP vision tower → standard HF SiglipVisionModel dir.

    Verified: produces ONNX with cos=+0.99999994 via Optimum (see
    scripts/local_pi0_siglip_parity.py).
    """
    from transformers import SiglipVisionConfig, SiglipVisionModel

    cfg = SiglipVisionConfig(**DEFAULT_SIGLIP_CFG)
    model = SiglipVisionModel(cfg).eval()

    sd = {}
    for k, v in state.items():
        if k.startswith(PI0_VISION_PREFIX):
            sd[k[len(PI0_VISION_PREFIX):]] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(
        "SigLIP: loaded %d keys, missing=%d unexpected=%d",
        len(sd), len(missing), len(unexpected),
    )
    # Missing keys expected: vision_model.head.* (attention-pool head,
    # unused by PaliGemma)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    (out_dir / "preprocessor_config.json").write_text(
        json.dumps({
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1 / 255.0,
            "resample": 3,
            "size": {"height": 224, "width": 224},
            "image_processor_type": "SiglipImageProcessor",
        })
    )
    return out_dir


def build_gemma_dir(state: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Extract Gemma language_model backbone → standard HF GemmaForCausalLM dir.

    Verified: produces ONNX with cos=+0.99999994 via Optimum (see
    scripts/local_pi0_gemma_parity.py).
    """
    from transformers import GemmaConfig, GemmaForCausalLM

    cfg = GemmaConfig(**DEFAULT_GEMMA_CFG)
    model = GemmaForCausalLM(cfg).eval()

    sd = {}
    for k, v in state.items():
        if k.startswith(PI0_LANGUAGE_PREFIX):
            sd[k[len(PI0_LANGUAGE_PREFIX):]] = v
    # Also load embed_tokens from lm_head (tied weights in Gemma)
    if PI0_LM_HEAD_KEY in state:
        sd["embed_tokens.weight"] = state[PI0_LM_HEAD_KEY].clone()
        logger.info(
            "Using tied lm_head (%s) as embed_tokens, shape %s",
            PI0_LM_HEAD_KEY, tuple(state[PI0_LM_HEAD_KEY].shape),
        )

    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    # Also tie lm_head explicitly
    if PI0_LM_HEAD_KEY in state:
        model.lm_head.weight = nn.Parameter(state[PI0_LM_HEAD_KEY].clone())
    logger.info(
        "Gemma: loaded %d keys, missing=%d unexpected=%d",
        len(sd), len(missing), len(unexpected),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps({
            "tokenizer_class": "GemmaTokenizer",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        })
    )
    return out_dir


class MultiModalProjector(nn.Module):
    """PaliGemma's SigLIP→Gemma linear projection (1152→2048)."""

    def __init__(self, in_dim: int = 1152, out_dim: int = 2048):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_projector(state: dict[str, torch.Tensor]) -> MultiModalProjector:
    """Extract PaliGemma's multi_modal_projector (Linear layer)."""
    w = state[PI0_PROJECTOR_KEYS["weight"]]
    b = state[PI0_PROJECTOR_KEYS["bias"]]
    in_dim, out_dim = w.shape[1], w.shape[0]
    proj = MultiModalProjector(in_dim=in_dim, out_dim=out_dim)
    proj.linear.weight.data.copy_(w)
    proj.linear.bias.data.copy_(b)
    proj.eval()
    logger.info("Projector: %d -> %d", in_dim, out_dim)
    return proj


def export_projector_onnx(projector: MultiModalProjector, out_path: Path) -> Path:
    """Export multi_modal_projector to ONNX (trivial — single Linear)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 256, projector.linear.in_features, dtype=torch.float32)
    torch.onnx.export(
        projector,
        dummy,
        out_path,
        input_names=["vision_features"],
        output_names=["projected_features"],
        dynamic_axes={"vision_features": {0: "batch", 1: "seq"}, "projected_features": {0: "batch", 1: "seq"}},
        opset_version=19,
    )
    logger.info("Exported projector ONNX: %s", out_path)
    return out_path


class EmbedTokens(nn.Module):
    """Standalone text embedding layer (tied with lm_head, vocab × hidden)."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


def build_embed_tokens(state: dict[str, torch.Tensor]) -> EmbedTokens:
    """Extract PaliGemma's tied lm_head as standalone embed_tokens."""
    w = state[PI0_LM_HEAD_KEY]
    vocab_size, hidden_size = w.shape
    emb = EmbedTokens(vocab_size, hidden_size)
    emb.embed.weight.data.copy_(w)
    emb.eval()
    logger.info("EmbedTokens: vocab=%d, hidden=%d", vocab_size, hidden_size)
    return emb


def export_embed_tokens_onnx(emb: EmbedTokens, out_path: Path) -> Path:
    """Export embed_tokens as ONNX."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randint(0, emb.embed.num_embeddings, (1, 16), dtype=torch.long)
    torch.onnx.export(
        emb,
        dummy,
        out_path,
        input_names=["input_ids"],
        output_names=["text_features"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "text_features": {0: "batch", 1: "seq"}},
        opset_version=19,
    )
    logger.info("Exported embed_tokens ONNX: %s", out_path)
    return out_path


def optimum_export_onnx(model_dir: Path, task: str, out_dir: Path) -> Path:
    """Run `optimum-cli export onnx` on a pre-built HF model dir.

    Uses legacy tracer (dynamo=False internally, which is Optimum's default)
    per the 2026 Gemma ONNX gotchas (see reflex_context/03_research/
    pi0_onnx_importable_sources.md Critical risk #2).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Find the venv optimum-cli (prefer repo-local venv)
    venv_cli = Path(".venv/bin/optimum-cli")
    cli = str(venv_cli) if venv_cli.exists() else "optimum-cli"
    cmd = [
        cli, "export", "onnx",
        "--model", str(model_dir),
        "--task", task,
        "--framework", "pt",
        str(out_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    onnx_path = out_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Optimum export did not produce {onnx_path}")
    return onnx_path


def export_pi0_prefix(
    output_dir: Path,
    model_id: str = "lerobot/pi0_base",
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    skip_gemma: bool = False,
    skip_siglip: bool = False,
) -> dict[str, Any]:
    """Full pi0 prefix-KV export pipeline.

    Args:
        output_dir: where to write the ONNX bundle.
        model_id: HF model id for pi0 weights.
        state_dict: optionally pre-loaded state (saves re-reading 14GB).
        skip_gemma / skip_siglip: skip the Optimum exports (useful during
            iteration when they're already cached).

    Returns:
        Dict with 'files' (mapping of component → ONNX path) and 'metadata'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if state_dict is None:
        state_dict = load_pi0_state_dict(model_id)

    result: dict[str, Any] = {"status": "ok", "files": {}, "metadata": {}}

    # 1. SigLIP vision encoder
    siglip_pt_dir = output_dir / "_pt" / "siglip"
    if not skip_siglip or not siglip_pt_dir.exists():
        logger.info("Building SigLIP vision tower dir...")
        build_siglip_dir(state_dict, siglip_pt_dir)
    siglip_onnx_dir = output_dir / "vision_encoder"
    if not (siglip_onnx_dir / "model.onnx").exists():
        logger.info("Exporting SigLIP via Optimum (feature-extraction)...")
        optimum_export_onnx(siglip_pt_dir, "feature-extraction", siglip_onnx_dir)
    result["files"]["vision_encoder"] = str(siglip_onnx_dir / "model.onnx")

    # 2. Multi-modal projector (SigLIP features → Gemma hidden)
    logger.info("Extracting multi_modal_projector...")
    projector = build_projector(state_dict)
    projector_onnx = output_dir / "multi_modal_projector.onnx"
    export_projector_onnx(projector, projector_onnx)
    result["files"]["multi_modal_projector"] = str(projector_onnx)
    result["metadata"]["projector"] = {
        "in_dim": projector.linear.in_features,
        "out_dim": projector.linear.out_features,
    }

    # 3. Text embed_tokens (tied with lm_head, shared vocab)
    logger.info("Extracting embed_tokens...")
    emb = build_embed_tokens(state_dict)
    emb_onnx = output_dir / "text_embedder.onnx"
    export_embed_tokens_onnx(emb, emb_onnx)
    result["files"]["text_embedder"] = str(emb_onnx)
    result["metadata"]["text_embedder"] = {
        "vocab_size": emb.embed.num_embeddings,
        "hidden_size": emb.embed.embedding_dim,
    }

    # 4. Gemma backbone decoder (with prefix-KV output)
    gemma_pt_dir = output_dir / "_pt" / "gemma"
    if not skip_gemma or not gemma_pt_dir.exists():
        logger.info("Building Gemma backbone dir...")
        build_gemma_dir(state_dict, gemma_pt_dir)
    gemma_onnx_dir = output_dir / "decoder_prefill"
    if not (gemma_onnx_dir / "model.onnx").exists():
        logger.info("Exporting Gemma backbone via Optimum (text-generation-with-past)...")
        optimum_export_onnx(gemma_pt_dir, "text-generation-with-past", gemma_onnx_dir)
    result["files"]["decoder_prefill"] = str(gemma_onnx_dir / "model.onnx")

    # 5. Expert stack — TODO: reuse pi0_exporter.build_pi0_expert_stack + export
    #    (expert is already handled by pi0_exporter.export_pi0; this new exporter
    #    replaces only the prefix pieces, keeps the expert path identical)
    logger.info(
        "Expert stack — delegate to pi0_exporter.export_pi0 (not yet composed here)"
    )

    # 6. Save config manifest
    config = {
        "model_id": model_id,
        "model_type": "pi0",
        "pipeline": "prefix_optimum + expert_custom",
        "components": result["files"],
        "metadata": result["metadata"],
    }
    (output_dir / "reflex_config.json").write_text(json.dumps(config, indent=2))
    result["files"]["config"] = str(output_dir / "reflex_config.json")

    logger.info("pi0 prefix export complete: %d components", len(result["files"]))
    return result


# TODO(v0.3): compose with pi0_exporter.build_pi0_expert_stack + flow matching
# host-side loop to produce end-to-end parity test.
# TODO(v0.3): write scripts/local_full_diff_pi0.py (analog of
# scripts/local_full_diff.py) that runs shared-noise PyTorch pi0 vs this
# composed ONNX pipeline end-to-end and reports cos >= 0.9999.
