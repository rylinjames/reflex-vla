"""pi0 monolithic ONNX parity: single-graph ONNX vs PyTorch sample_actions.

Uses the monolithic ONNX produced by scripts/export_pi0_monolithic.py.
Same inputs fed to both PyTorch policy.predict_action_chunk and the
ONNX graph; compare first action by cos, max_abs, L2.

Target: cos >= 0.999.
"""
import sys
import types

for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub

import numpy as np
import torch


def _patch_pi0_for_transformers_457():
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


MONOLITHIC_ONNX = "/tmp/pi0_monolithic_onnx/model.onnx"


def main():
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from huggingface_hub import snapshot_download
    import onnxruntime as ort

    print("Loading PyTorch pi0_base...")
    policy = PI0Policy.from_pretrained("lerobot/pi0_base").eval().to(dtype=torch.float32).to("cpu")

    # Build the same input as the monolithic export used as dummy
    B = 1
    cfg = policy.config
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim

    rng = np.random.RandomState(42)
    img_np = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img_t = img_t * 2.0 - 1.0
    state = torch.from_numpy(rng.randn(14).astype(np.float32) * 0.1)
    task = "pick up the red bowl"

    batch_raw = {
        "observation.images.base_0_rgb": img_t.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": [task],
    }
    repo = snapshot_download("lerobot/pi0_base")
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    batch_pp = pre(batch_raw)

    noise = np.random.RandomState(99).randn(1, chunk, action_dim).astype(np.float32)

    print("\n--- PyTorch reference ---")
    with torch.no_grad():
        pt_actions = policy.predict_action_chunk(batch_pp, noise=torch.from_numpy(noise))
    pt_actions = pt_actions.cpu().numpy() if hasattr(pt_actions, "cpu") else np.asarray(pt_actions)
    print(f"  shape: {pt_actions.shape}, first: {pt_actions[0, 0, :5]}")

    # Inspect what pi0 fed into sample_actions
    images, img_masks = policy._preprocess_images(batch_pp)
    lang_tokens = batch_pp["observation.language.tokens"]
    lang_masks = batch_pp["observation.language.attention_mask"]
    state_tensor = policy.prepare_state(batch_pp)

    print(f"\n  images: {[img.shape for img in images]}")
    print(f"  lang_tokens: {lang_tokens.shape}")

    print("\n--- ONNX monolithic ---")
    sess = ort.InferenceSession(MONOLITHIC_ONNX, providers=["CPUExecutionProvider"])
    # Feed inputs matching the wrapper signature
    ort_inputs = {
        "img_base": images[0].numpy().astype(np.float32),
        "img_wrist_l": images[1].numpy().astype(np.float32),
        "img_wrist_r": images[2].numpy().astype(np.float32),
        "mask_base": img_masks[0].numpy(),
        "mask_wrist_l": img_masks[1].numpy(),
        "mask_wrist_r": img_masks[2].numpy(),
        "lang_tokens": lang_tokens.numpy().astype(np.int64),
        "lang_masks": lang_masks.numpy(),
        "state": state_tensor.numpy().astype(np.float32),
        "noise": noise,
    }
    ort_out = sess.run(None, ort_inputs)
    onnx_actions = ort_out[0]
    print(f"  shape: {onnx_actions.shape}, first: {onnx_actions[0, 0, :5]}")

    # Compare
    pt0 = pt_actions[0, 0]
    on0 = onnx_actions[0, 0]
    diff = pt0 - on0
    max_abs = float(np.abs(diff).max())
    l2 = float(np.linalg.norm(diff))
    cos = float(np.dot(pt0, on0) / (np.linalg.norm(pt0) * np.linalg.norm(on0) + 1e-8))

    print(f"\n====== MONOLITHIC PARITY ======")
    print(f"  first-action max_abs = {max_abs:.4e}")
    print(f"  first-action L2      = {l2:.4e}")
    print(f"  first-action cos     = {cos:+.6f}")

    full_cos = float(
        np.dot(pt_actions.flatten(), onnx_actions.flatten())
        / (np.linalg.norm(pt_actions) * np.linalg.norm(onnx_actions) + 1e-8)
    )
    full_max = float(np.abs(pt_actions - onnx_actions).max())
    print(f"  full chunk max_abs = {full_max:.4e}, cos = {full_cos:+.6f}")

    passed = cos >= 0.999 and max_abs < 0.1
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
