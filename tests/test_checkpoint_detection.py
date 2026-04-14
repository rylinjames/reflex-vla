"""Tests for model type auto-detection from state_dict keys."""

import torch

from reflex.checkpoint import detect_model_type


def _dummy_tensor():
    return torch.zeros(1)


class TestDetectModelType:
    def test_detects_smolvla_via_vision_encoder(self):
        sd = {"model.vision_encoder.patch_embed.weight": _dummy_tensor()}
        assert detect_model_type(sd) == "smolvla"

    def test_detects_smolvla_via_vlm_with_expert(self):
        sd = {"model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight": _dummy_tensor()}
        assert detect_model_type(sd) == "smolvla"

    def test_detects_pi0(self):
        sd = {
            "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight": _dummy_tensor(),
            "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight": _dummy_tensor(),
            "action_in_proj.weight": _dummy_tensor(),
        }
        assert detect_model_type(sd) == "pi0"

    def test_detects_pi05_via_adarmsnorm(self):
        sd = {
            "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight": _dummy_tensor(),
            "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense.weight": _dummy_tensor(),
            "action_in_proj.weight": _dummy_tensor(),
        }
        assert detect_model_type(sd) == "pi05"

    def test_returns_none_for_unknown(self):
        sd = {"some.random.model.weight": _dummy_tensor()}
        assert detect_model_type(sd) is None


class TestPi0ExpectedStructure:
    """Smoke tests that pi0 exporter's expected keys are in the agreed layout."""

    def test_pi0_action_keys_are_top_level(self):
        """pi0 has action projections at the top level (no 'model.' prefix)."""
        from reflex.exporters.pi0_exporter import PI0_ACTION_KEYS

        for key_path in PI0_ACTION_KEYS.values():
            # pi0 keys are top-level, e.g. "action_in_proj.weight", not "model.action_in_proj.weight"
            assert not key_path.startswith("model."), f"pi0 key should be top-level: {key_path}"

    def test_pi0_expert_prefix(self):
        """pi0 expert layers are under 'paligemma_with_expert.gemma_expert.model.'."""
        from reflex.exporters.pi0_exporter import PI0_EXPERT_PREFIX

        assert PI0_EXPERT_PREFIX.startswith("paligemma_with_expert")
        assert "gemma_expert" in PI0_EXPERT_PREFIX
