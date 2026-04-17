# diagnostic_ladder — the progression from "black box fails" to "localized bug"

## The four rungs

When a VLA export diverges from its PyTorch reference, the naive "run end-to-end and compare final action" has near-zero diagnostic value — a cos_sim of 0.08 tells you "something is wrong" but nothing about *where*. The Apr-17 session pushed on this for ~6h before articulating a proper ladder. The ladder, rung-by-rung:

```
Rung 4: Single-layer weight-copy diff  ←  disambiguates implementation vs composition bugs
Rung 3: Per-layer diff within a stage  ←  layer 0 k/v, layer 8 k/v, layer 15 k/v
Rung 2: Per-stage diff                 ←  vision / text / state / decoder / expert
Rung 1: Full end-to-end pipeline diff  ←  crude L2 / cos_sim on final action
```

Climb from Rung 1 downward only if Rung 1 says "broken." Each rung costs more to set up but localizes the bug further. The first-passing rung is where the bug lives *above*; the first-failing rung is where the bug lives *at*.

---

## Rung 1: Full end-to-end pipeline diff

**Purpose:** Triage. Is the exporter broken at all? If cos_sim > 0.95, ship. If < 0.5, something is structurally wrong.

**Sketch** (`scripts/local_full_diff.py`):

```python
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from reflex.runtime.server import ReflexServer

EXPORT_DIR = "/tmp/reflex_export"
CHUNK = 50
MAX_ACTION_DIM = 32

# ---- shared noise (see shared_noise_discipline.md) ----
rng = np.random.RandomState(99)
noise = rng.randn(1, CHUNK, MAX_ACTION_DIM).astype(np.float32)

# ---- common inputs ----
image_256 = (rng.rand(1, 3, 256, 256) * 255).astype(np.uint8)
state_8d = rng.randn(1, 8).astype(np.float32)
task = "put the red bowl on the plate"

# ---- torch reference ----
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero").to("cpu").to(torch.float32)
policy.eval()
with torch.no_grad():
    torch_actions = policy.predict_action_chunk(
        image=image_256, state=state_8d, task=task, noise=torch.from_numpy(noise)
    )

# ---- onnx pipeline ----
server = ReflexServer(EXPORT_DIR, device="cuda", strict_providers=False)
onnx_actions = server.predict(image=image_256, state=state_8d, task=task, noise=noise)

# ---- verdict ----
t = torch_actions.cpu().numpy().reshape(-1)
o = np.asarray(onnx_actions).reshape(-1)
l2 = float(np.linalg.norm(t - o))
cos = float(t @ o / (np.linalg.norm(t) * np.linalg.norm(o) + 1e-9))
print(f"L2 = {l2:.3f}   cos_sim = {cos:+.3f}")
print("VERDICT:", "correct" if cos > 0.95 else "minor drift" if cos > 0.5 else "structural bug")
```

**Typical results from Apr-17 session:**
- cos_sim = 0.28 → "something wrong, no idea where" (line 10799)
- After state_proj fix: 0.498 (line 10809)
- After noise seeding fix: 0.305 (line 10912)
- After AutoModel class fix: 0.08 → actually worse than before (line 11089)
- Final: -0.24 with all per-stage cos ≈ 1.0 (line 11175)

This rung is a necessary but insufficient tool. It justifies going deeper; it does not pinpoint.

---

## Rung 2: Per-stage diff

**Purpose:** Localize divergence to one pipeline stage. The SmolVLA forward pass has five stages. Diff each independently.

**Stages:**
1. Vision encoder (SigLIP + connector) — `[B, 3, 512, 512]` → `[B, 64, 960]`
2. Text embedder (SmolLM2 `embed_tokens`) — token_ids → `[B, T, 960]`
3. State projection — `[B, 8]` → `[B, 1, 960]`
4. Decoder prefill (32-layer SmolLM2 → truncated 16 layers) — produces per-layer k/v
5. Action expert one-step velocity — `[B, 50, 32]` + VLM k/v → velocity

**Sketch** (`scripts/local_stage_diff.py`):

```python
import torch, numpy as np, onnxruntime as ort
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def cos(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero").to(torch.float32)
policy.eval()

# Inputs
img = torch.randn(1, 3, 512, 512)
img_f = img.clone() * 2.0 - 1.0  # SigLIP expects [-1,1]
task_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
state = torch.randn(1, 8)

# ---- Stage 1: vision ----
with torch.no_grad():
    vlm = policy.model.vlm_with_expert.vlm
    torch_vision = vlm.embed_image(img_f.to(torch.float32))  # [1, 64, 960]

onnx_vision = ort.InferenceSession("/tmp/reflex_export/vision_encoder.onnx").run(
    None, {"pixel_values": img_f.numpy()}
)[0]

print(f"Stage 1 (vision)   cos={cos(torch_vision.numpy(), onnx_vision):+.4f}   "
      f"torch_norm={np.linalg.norm(torch_vision.numpy()):.1f}   "
      f"onnx_norm={np.linalg.norm(onnx_vision):.1f}")

# ---- Stage 2: text embedder ----
with torch.no_grad():
    text_tokens = vlm.model.text_model.embed_tokens(task_ids)
onnx_text = ort.InferenceSession("/tmp/reflex_export/text_embedder.onnx").run(
    None, {"input_ids": task_ids.numpy()}
)[0]
print(f"Stage 2 (text)     cos={cos(text_tokens.numpy(), onnx_text):+.4f}")

# ---- Stage 3: state proj ----
# (state_proj is a Linear(32, 960), pad state to 32 if needed)
state_padded = torch.zeros(1, 32); state_padded[:, :8] = state
torch_state = policy.model.state_proj(state_padded)[:, None, :]
# ONNX state is inlined in VLMPrefixOrchestrator as a numpy linear, compare directly
# (see src/reflex/runtime/vlm_orchestrator.py)
# ...
```

**Typical results from Apr-17 session** (lines 10969, 11089, 11132):

| Stage | Cos_sim | Notes |
|---|---|---|
| 1. Vision (before fix) | 0.6983 | ❌ norms 1644 vs 1104 — wrong class used in export |
| 1. Vision (after unwrap fix) | 1.0000 | ✓ max_abs 1e-4 |
| 2. Text embedder | 1.0000 | ✓ |
| 3. State projection | 1.0000 | ✓ |
| 4. Per-layer decoder k/v | 0.91-1.00 | mostly ✓; layer 0 v is 0.91 outlier |
| 5. Expert velocity (one step) | 0.977 | tiny per-step error |
| End-to-end (integrated) | -0.24 | catastrophic after 10 Euler steps |

The Rung 2 test is how the Apr-17 session found that vision encoder was failing silently: stages 2/3 were perfect, stage 1 was at 0.6983. Without Rung 2, vision would have stayed hidden for another week.

---

## Rung 3: Per-layer diff within a stage

**Purpose:** When one stage is wrong, localize to a specific layer.

For the decoder prefill stage (Rung 2 stage 4), the SmolLM2 decoder has 16 layers. Each emits a k (key) and v (value) tensor after RoPE. Diff each layer's k and v separately.

**Sketch** (`scripts/local_single_layer_diff.py` — per-layer k/v diff across decoder):

```python
import torch, numpy as np, onnxruntime as ort
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero").to(torch.float32).eval()
vlm = policy.model.vlm_with_expert.vlm

# Prepare a batched input that exercises the prefix
# image_embeds + text_embeds + state_embed -> [1, 64+T+1, 960]
inputs_embeds = torch.randn(1, 70, 960)

# Hook every layer to capture k and v after RoPE
captured = {}
def make_hook(i):
    def hook(module, args, output):
        # LlamaAttention returns (attn_output, attn_weights, past_key_value)
        # Past KV is the (k, v) post-RoPE we want
        past_kv = output[2] if len(output) > 2 else None
        if past_kv is not None:
            captured[i] = (past_kv[0].detach().clone(), past_kv[1].detach().clone())
    return hook

text_model = vlm.model.text_model
for i, layer in enumerate(text_model.layers[:16]):
    layer.self_attn.register_forward_hook(make_hook(i))

# Run torch forward
with torch.no_grad():
    text_model(inputs_embeds=inputs_embeds, use_cache=True)

# Run our decoder_prefill.onnx
sess = ort.InferenceSession("/tmp/reflex_export/decoder_prefill.onnx")
# decoder_prefill produces per-layer k,v stacked: shape [num_layers, 2, B, num_kv_heads, T, head_dim]
onnx_kv = sess.run(None, {"inputs_embeds": inputs_embeds.numpy()})[0]

# Compare layer by layer
for i in range(16):
    torch_k, torch_v = captured[i]
    onnx_k = onnx_kv[i, 0]
    onnx_v = onnx_kv[i, 1]
    ck = cos(torch_k.numpy(), onnx_k)
    cv = cos(torch_v.numpy(), onnx_v)
    print(f"layer {i:2d}  k cos={ck:+.4f}  v cos={cv:+.4f}")
```

**Typical result from Apr-17** (app `ap-oXrqhfnQFJLuuY4A9GbPSv`, 11:08 IST):
```
layer 0  k cos=+1.0000  v cos=+0.9117    ← outlier
layer 8  k cos=+0.9997  v cos=+0.9967
layer 15 k cos=+0.9994  v cos=+0.9954
```

The layer 0 v cos=0.9117 outlier is reproducible across three distinct stage-diff Modal runs. That told us the first-layer value projection or first-layer RoPE is wrong; it is NOT an attention-mask bug (those affect deep layers more).

---

## Rung 4: Single-layer weight-copy diff

**Purpose:** Disambiguate "my layer implementation is wrong" from "the composition of layers is wrong."

The trick: take one layer of the real model. Copy its weights into our implementation of that layer. Feed both the same input (noise + k/v + position_ids). Compare outputs.

- If single-layer cos_sim ≈ 1.0 → **the layer implementation is correct**; the bug is in how layers compose (attention mask, between-layer norm, prefix offset, RoPE frequency table).
- If single-layer cos_sim < 0.99 → **the layer has a bug** (RMSNorm formula, RoPE base, MLP ordering, GQA reshape, attention scale).

**Sketch** (`scripts/local_single_layer_diff.py` — single layer copy-and-compare):

```python
import torch, numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from reflex.exporters.smolvla_exporter import ExpertGQALayer

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero").to(torch.float32).eval()
real_layer = policy.model.vlm_with_expert.lm_expert.layers[0]

# Our reconstruction
our_layer = ExpertGQALayer(
    hidden=720, num_q_heads=15, num_kv_heads=5, head_dim=64, intermediate=2048
)

# Copy weights: real layer state_dict → our layer, matching keys
our_sd = our_layer.state_dict()
real_sd = real_layer.state_dict()
for k in our_sd:
    # may need a small remap table; self-attn vs cross-attn differs in v0.1
    if k in real_sd:
        our_sd[k] = real_sd[k]
    else:
        print(f"MISSING in real: {k}")
our_layer.load_state_dict(our_sd, strict=False)

# Same input through both
x = torch.randn(1, 50, 720)
kv = torch.randn(1, 65, 320)  # VLM k/v prefix, 5 KV heads x 64 head_dim
pos_ids = torch.arange(50)[None]

with torch.no_grad():
    real_out = real_layer(x, vlm_k=kv, vlm_v=kv, position_ids=pos_ids)[0]
    our_out = our_layer(x, vlm_k=kv, vlm_v=kv, position_ids=pos_ids)

c = cos(real_out.numpy(), our_out.numpy())
print(f"Single-layer cos = {c:+.5f}")
print("VERDICT:", "layer impl correct" if c > 0.9999 else "layer impl buggy")
```

**Actual Apr-17 result** (line 11468):
> "Single SELF-attn layer (layer 0) matches to **1e-5 precision, cos=1.0000**. The bug is somewhere in COMPOSITION — probably cross-attention layers."

That finding eliminated 5 candidate bugs (RMSNorm formula, MLP ordering, GQA reshape, RoPE, attention scale) in a single check and redirected the investigation to composition-level issues (prefix_offset for self-attn, kv_mask for cross-attn, attention mask on padded prefix positions, etc.).

Without Rung 4, the Apr-17 session would have spent another day re-checking the layer implementation.

---

## The right escalation order

1. Start at Rung 1. If pass, ship.
2. Fail → Rung 2. Find the failing stage.
3. That stage fails → Rung 3. Find the failing layer.
4. That layer fails at Rung 3 but passes at Rung 4 → it's a **composition bug** (attention mask, prefix offset, RoPE frequency table between layers).
5. That layer fails at both Rung 3 and Rung 4 → it's an **implementation bug** inside that layer (RMSNorm, RoPE, GQA reshape, MLP ordering).

The escalation direction should always be "narrower." Never climb back up in a single debug session without fixing something.

---

## Supporting disciplines

### Shared noise (critical)

All four rungs assume shared noise between torch and ONNX paths. Without it, every cos_sim is dominated by random drift in the noise. See `reflex_context/04_iteration_lessons/shared_noise_discipline.md`.

### Deterministic inputs

Use a fixed seed for everything: numpy random, torch random, image generation, state vector. Reproducibility across runs is non-negotiable for this type of debugging.

```python
rng = np.random.RandomState(7)  # or 99, or 42 — pick one and stick
```

### Per-step cos > 0.999 rule (line 11435)

For a 10-step Euler denoise to survive integration, per-step cos_sim must be > 0.999. A per-step error of 2% compounds catastrophically: `0.98^10 ≈ 0.82` and that's multiplicative with the attention pointing slightly wrong, which gave the -0.24 end-to-end cos in Apr-17.

**Numerical budget:** each stage in the pipeline contributes. Final end-to-end cos < prod(stage cos). If you have 5 stages each at 0.99, expect final 0.95. If any stage drops below 0.99 you will blow the budget.

---

## Files in this repo

- `scripts/local_full_diff.py` — Rung 1
- `scripts/local_stage_diff.py` — Rung 2
- `scripts/local_single_layer_diff.py` — Rung 3 + Rung 4 (per-layer k/v + single-layer copy)
- `scripts/local_expert_diff.py` — Rung 2/3 hybrid for the expert stack alone

For Modal variants (needed only for benchmarks or LIBERO integration), see `scripts/modal_stage_diff.py`, `scripts/modal_pytorch_vs_onnx.py`.

## Forgotten discipline

The Apr-17 session shows that **Rung 1 alone wasted 4+ hours of Modal spend and ~30 fix attempts** because cos_sim = 0.08 said "bug somewhere" with no further information. The *first* thing to do when an end-to-end diff fails is drop to Rung 2. The scripts were only written mid-session as a reactive measure; they should be written **upfront, once**, and lived in before the first exporter ships.
