# ONNX Export — What Reflex Decomposes and Why

**TL;DR:** Reflex's exporters manually decompose four PyTorch ops that would otherwise break TensorRT on Jetson, even though stock `torch.onnx.export` emits them correctly for ORT CPU/GPU. The decompositions are: **RMSNorm** (TRT opset 23 gap), **RoPE** (static-shape cos/sin caches), **GQA attention** (explicit repeat_interleave before matmul), and **image-position IDs** (pre-computed instead of runtime `bucketize`). Each decomposition is forced by a specific downstream-tool bug; stripping any of them reopens a specific known failure mode.

---

## Decomposition catalog

### 1. RMSNorm — decomposed

**What we do:** Replace `torch.nn.RMSNorm` (or HF `LlamaRMSNorm`) with elementwise math:

```python
class DecomposedRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(self.weight.dtype) * self.weight
```

Present in `src/reflex/decompose.py` (since v0.1.0, commit `e8fe39f`). Reused by every per-family exporter (`smolvla_exporter.py`, `pi0_exporter.py`, `gr00t_exporter.py`).

**Why:** The canonical post-mortem is in current_session line 11574:

> *"TRT's ONNX parser doesn't support opset 23 `RMSNormalization` op yet (issue #4639). Modern torch export emits this op when it sees `nn.RMSNorm`. So if we use `torch.onnx.export` with default settings, RMSNorm won't compile on Jetson. For Jetson TRT, we STILL need decomposed RMSNorm. But only RMSNorm — torch.onnx.export handles attention, RoPE, GQA, etc. cleanly."*

**Specific bugs avoided:**
- TensorRT issue #4639 — opset 23 `RMSNormalization` parser gap. Blocks Jetson TRT engine build at ONNX-load time, not at runtime.
- TRT fp16 accuracy issues on SigLIP (issues #3908, #4373) — independent of RMSNorm, but amplified when `RMSNormalization` is parsed vs our elementwise decomposition.

**Upcast rationale:** `x.to(torch.float32).pow(2)` is deliberate. The squared value is large enough that fp16 can overflow in the sum; upcasting to fp32 is what HuggingFace does in `LlamaRMSNorm.forward`. Every exporter preserves this.

**pi0.5 / GR00T variants:**
- **pi0.5 AdaRMSNorm (`DecomposedAdaRMSNorm`, commit `c0a3a7b`):** time-conditioned RMSNorm. `time_emb → dense → chunk(3) → x*rsqrt(var+eps)*(1+scale)+shift`. 3-chunk (scale/shift/gate), distinct from GR00T's 2-chunk AdaLN. Pi0.5 adds ~112M params vs pi0 purely from these `dense` layers.
- **GR00T AdaLN (inlined in `gr00t_exporter.py`, commit `68119b7`):** 2-chunk `adaLN_modulation: time → [scale, shift]`, applied pre-attention and pre-MLP. Non-affine base LayerNorm (not RMSNorm) — GR00T inherits DiT conventions, not Llama.

**Validated parity (per commit `e8fe39f`):** RMSNorm eps=1e-6, numerical parity to PyTorch native `nn.RMSNorm` < 1e-6 on A100.

---

### 2. RoPE — decomposed with cos/sin caches

**What we do:** Pre-compute `cos_cache, sin_cache` as tensors sized `[max_seq_len, head_dim]`. Pass them into the attention layer (and into the ONNX graph as buffers). `apply_rope` becomes an explicit elementwise op with constant-size gathers.

```python
class DecomposedRoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=512, rope_theta=100000.0):
        super().__init__()
        # rope_theta = 100000 for SmolLM2, NOT 10000 default (bug #8)
        freqs_base = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2) / head_dim))
        pos = torch.arange(max_seq_len)
        freqs = torch.outer(pos, freqs_base)                    # [max_seq_len, head_dim // 2]
        # IMPORTANT: concat to full head_dim (not half) — half-dim was a pre-landing bug
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)     # [max_seq_len, head_dim]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, q, k, position_ids):
        cos = self.cos_cache[position_ids]
        sin = self.sin_cache[position_ids]
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)
```

Present in `src/reflex/decompose.py`. Used in every per-family exporter.

**Why:** Multiple compounding reasons:
1. **Static shapes for TRT:** TRT engines compile against fixed input shapes. RoPE's canonical implementation computes `cos(m * theta_i)` at inference time via `sin/cos` of dynamic position indices. Pre-computing a cache converts this to a `Gather` op over a fixed 2D buffer — TRT handles that natively.
2. **opset 19 for numerical stability:** We target opset 19 (not the default modern 21+) so all ops have long-standing TRT parser support. ComputeGraph ops added in opset 20+ have patchy TRT coverage.
3. **Half-dim concat bug:** A pre-landing bug had `cos_cache` at `head_dim // 2`. Attention silently degraded. Fix is `cat([freqs.cos(), freqs.cos()], -1)` to give a **full-dim** cache.
4. **`rope_theta = 100000` not 10000:** SmolLM2 uses `rope_theta=100000`. Default `DecomposedRoPE` was 10000 (10× wrong) and bug #8 in current_session line 11424 caught this.

**Specific bug avoided:** Without the `Gather`-over-cache form, TRT would need to fuse a dynamic `cos(arange)` pattern that it can't emit optimally, leading to per-step CPU dispatch.

**Per commit `6fedff3` (GQA spike):** PyTorch 2.11's new exporter (`torch.export.export(strict=False)`) handles HF `LlamaRotaryEmbedding` cleanly without our decomposition when targeting ORT. But **for Jetson TRT we still force the decomposition** because the parser + engine-build path is more fragile than the ORT path.

---

### 3. Attention — GQA expanded with `repeat_interleave` before matmul

**What we do:** For Grouped-Query Attention, we expand K/V heads explicitly in the ONNX graph instead of letting torch emit `scaled_dot_product_attention` with GQA broadcasting. SmolVLA: 15 Q heads, 5 KV heads, `repeat_factor = 3`.

```python
# Inside ExpertGQALayer.forward, after RoPE
# q: [B, nq=15, S, head_dim=64]; k,v: [B, nkv=5, S, head_dim=64]
k = k.repeat_interleave(nq // nkv, dim=1)   # → [B, 15, S, 64]
v = v.repeat_interleave(nq // nkv, dim=1)   # → [B, 15, S, 64]

scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
# softmax upcast to fp32 for stability — important for fp16 serve
attn = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
attn = torch.matmul(attn, v)
attn = attn.transpose(1, 2).reshape(B, S, nq * head_dim)  # [B, S, 960]
out = attn @ o_proj.T                                      # 960 → 720 back to expert hidden
```

Implemented in `src/reflex/exporters/smolvla_exporter.py::ExpertGQALayer` (patterns from `modal_expert_export.py` commit `74d24c3`).

**Why:**
1. **`scaled_dot_product_attention` fuses to MHA in ONNX, not MHA+GQA.** Torch emits the op assuming K/V match Q heads. With GQA (Q=15, KV=5), the op either silently broadcasts (fragile — varies by opset + backend) or errors. Explicit `repeat_interleave` makes the graph unambiguous.
2. **TRT LayerFusion failure on GeLU + MHA:** Per sessions_md.md line 15, "GeLU + TRT LayerFusion failure: GeLU lands as elementwise Mul+Tanh. ONNX export works, TRT LayerFusion fails; kept as generic MHA." Decomposed GQA doesn't help with GeLU, but the combined pattern (decomposed RoPE + explicit KV expansion + fp32 softmax + standard MatMul) gives TRT a clean pattern to fuse into a TensorRT MHA plugin.
3. **Head dim verification:** SmolVLA `k_proj.shape = [320, 720]` (bug 2619 in current_session): nkv × head_dim = 5 × 64 = 320. This is distinct from VLM-scale 960. The GQA expansion is what gets the 320-dim K/V back up to the 960-dim attention space so the matmul works.

**Specific bug avoided:** Silent shape mismatches during TRT engine build. With implicit GQA, the builder may pick an incompatible kernel that fails at runtime; explicit repeat makes the path deterministic.

**fp32 softmax upcast:** The 12-bugs list calls this out as a remaining suspect for the cos=-0.24 final divergence. Our decomposition explicitly upcasts to fp32 for softmax, then downcasts — matches HF convention.

**Cross-attention variant:** For cross-attn layers, K/V don't come from the hidden state — they come from the VLM prefix KV cache at this layer index. Same `repeat_interleave` applies; the difference is the K/V source (external input `vlm_k[layer]`, `vlm_v[layer]`) not a self-projection.

---

### 4. Image-position IDs — pre-computed instead of runtime `bucketize`

**What we do:** Pre-compute the image token position IDs as a static buffer. Bypass SmolVLMVisionEmbeddings' dynamic `index_put` / `bucketize` loop.

From commit `5869a3e`:
> *"VisionEncoderForONNX wraps SigLIP vision encoder + SmolVLM connector. Pre-computes position IDs to avoid the dynamic `index_put` / `bucketize` loop in SmolVLMVisionEmbeddings.forward() that produces ONNX nodes with int64/float type mismatches ORT cannot load. Input `pixel_values [B, 3, 512, 512]`; output `image_embeds [B, 64, 960]`. `patch_onnx_type_mismatches` post-fixes any remaining type mismatches in the ONNX graph."*

**Why:**
1. **`bucketize` with float boundaries emits int64/float mismatched Gather ops** that ORT refuses to load. Per current_session line 7252 (ETARS research summary): "Must use `do_constant_folding=False` (folding corrupts the graph). Must run `patch_gather_indices_once()` post-export (vision encoder produces float Gather indices that ORT rejects)."
2. **Static image size (512×512) means token grid (64 tokens) is fixed** at export time. SigLIP SO400M native is 512×512 with 16-px patches producing 32×32 = 1024 patches, then pixel_shuffle 4×4 = 64 tokens. No reason to compute boundaries at runtime.
3. **Numerical threshold (`ORT_MAX_DIFF_THRESHOLD = 5e-4`):** SigLIP's 27 transformer layers accumulate fp32 rounding. Max_diff ~2-4e-4 is expected. Commit `5869a3e` constant.

**Specific bug avoided:** ORT session creation fails when Gather op has int64 indices but the indexed tensor is float. The vanilla `torch.onnx.export` of SmolVLM produces these mismatches. Pre-computing eliminates the op entirely.

**Wrapper pattern:**
```python
class VisionEncoderForONNX(nn.Module):
    def __init__(self, vlm):
        super().__init__()
        self.vision_model = vlm.model.vision_model
        self.connector = vlm.model.connector
        # Pre-compute position IDs for 512x512 / 16px patches / 4x4 pixel_shuffle
        self.register_buffer("precomputed_position_ids", _build_pos_ids(), persistent=False)

    def forward(self, pixel_values):
        # pixel_values: [B, 3, 512, 512], already scaled to [-1, 1] by caller
        vision_out = self.vision_model(pixel_values, position_ids=self.precomputed_position_ids)
        image_embeds = self.connector(vision_out)  # pixel_shuffle + linear projection
        return image_embeds  # [B, 64, 960]
```

Plus post-export ONNX graph patching:
```python
def patch_onnx_type_mismatches(onnx_path):
    # Scan for Gather ops where indices dtype != int64; insert Cast nodes.
    # Scan for float indices in IndexPut/Gather; patch with static int64 equivalents.
```

---

## What we do NOT decompose (and why)

**Stock `torch.onnx.export` handles these cleanly:**
- **Standard attention MatMul + softmax** (when GQA is explicit per above)
- **SwiGLU MLP** (`silu(gate_proj(x)) * up_proj(x)` → `down_proj`)
- **Token embedding** (`embed_tokens` lookup table; our `text_embedder.onnx` is exactly this, no decomposition needed)
- **SigLIP vision transformer** (ViT blocks with standard LayerNorm, MHA, MLP — all long-supported in ONNX)

**Per current_session line 11524 (the "copy lerobot code" pivot):** We write our own `ExpertGQALayer`, `DecomposedRMSNorm`, `_DecomposedRoPE` **optimized for ONNX/TensorRT op coverage and edge hardware**. Every tiny mismatch (missing √hidden scaling, wrong `rope_theta`, missing pad mask, RoPE half-dim concat) becomes a bug we have to discover and fix.

The hybrid strategy from line 11574 is: copy lerobot's `SmolVLAPolicy.sample_actions` + `embed_prefix` + `embed_suffix` + `forward_cross_attn_layer` into `reflex/runtime/smolvla_native.py`. Swap only `RMSNorm → DecomposedRMSNorm` for TRT compat. Let `torch.onnx.export` handle the rest. "Hours of work, correct by construction, Jetson compatible."

---

## The opset choice: why 19

- **Opset 19** is the floor that every TRT version Reflex cares about supports cleanly.
- **Opset 21+** adds fused ops (RMSNormalization at 23) that the Jetson TRT parser doesn't support yet.
- **Opset 13/14** (older) lack `ScatterND` improvements we rely on for the GQA K/V expansion.
- All Modal test scripts use `opset_version=19` (commit `e8fe39f` onwards).

**Per commit `6fedff3`:** PyTorch 2.11's new exporter (`torch.export.export(strict=False)`) respects the target opset and produces cleaner graphs than legacy `torch.onnx.export(dynamo=False)`. The GQA spike validated this: "**torch.onnx.export at opset 19, single decoder layer wrapped with RoPE computation included in wrapper. PyTorch 2.11 new exporter (torch.export.export strict=False) under the hood.**"

---

## The "TRT × static shapes" sharp edge (related)

Reflex's exporters bake **static** shapes into the ONNX graph (batch=1, chunk=50, action_dim=32). This has consequences:

- **`trtexec --minShapes/--optShapes/--maxShapes` error out** when the ONNX is fully static (sessions_md.md line 19). Must omit those flags.
- **TRT engine can't handle batch=N > 1 at inference time** without rebuilding (sessions_md.md line 13, commit `e76678c`). First batched request = 34s rebuild, every subsequent = rebuild again (200× pessimization).
- **Fix (commit `e76678c`):** `reflex serve --max-batch > 1` drops TRT EP and falls through to CUDA EP, which handles dynamic batch shapes natively and gives 2.88× throughput at batch=16 on pi0.
- **Long-term fix:** dynamic batch shape export + TRT shape profiles (batch=1/4/8/16). Deferred to v0.2.

This isn't a decomposition decision but it's a constraint our exporter shape choices create.

---

## Validation thresholds (what we measure)

Per-exporter expected `max_diff` (PyTorch fp32 vs ONNX fp32 on same inputs):

| Component | Expected max_diff | Source |
|-----------|-------------------|--------|
| DecomposedRMSNorm vs nn.RMSNorm | < 1e-6 | commit `e8fe39f` |
| Suffix encoder (action_in_proj + action_time_mlp) | 2.15e-06 | commit `1ed46ab` |
| Action projection | 1.07e-06 | commit `1ed46ab` |
| Single expert layer | 5.36e-07 | commit `74d24c3` (current_session line 2660) |
| Full 16-layer expert stack | 4.77e-06 | commit `47f3d5d` |
| SmolVLA full export (expert_stack.onnx) | 3.81e-06 | commit `c1726e7` |
| pi0 full export | 3.73e-08 | commit `45794b0` |
| pi0.5 full export | 2.37e-06 | commit `c0a3a7b` |
| GR00T DiT block | 2.18e-05 | commit `68119b7` |
| GR00T full-stack | 3.77e-06 | commit `ff9fc3a` |
| SigLIP + connector (vision_encoder.onnx) | 2-4e-04 | commit `5869a3e` (27 fp32 layers accumulate) |
| GQA+RoPE single decoder layer | 4.01e-05 | commit `6fedff3` spike |

**The validate wedge threshold** (Unreleased, `ValidateRoundTrip` default): `--threshold 1e-4`. This is a BREAKING change from v0.1 stub's 0.02 placeholder. Pass `--threshold 0.02` explicitly for old behavior.

---

## References

- `src/reflex/decompose.py` — `DecomposedRMSNorm`, `DecomposedAdaRMSNorm`, `DecomposedRoPE`
- `src/reflex/exporters/smolvla_exporter.py::ExpertGQALayer` — GQA attention decomposition
- `src/reflex/exporters/vlm_components.py` — `VisionEncoderForONNX`, pre-computed image position IDs
- `src/reflex/exporters/vlm_prefix_exporter.py::patch_onnx_type_mismatches` — post-export ONNX graph patcher
- **Commits:** `e8fe39f` (baseline decompose.py), `74d24c3` (ExpertGQALayer + expert decomp), `c0a3a7b` (AdaRMSNorm for pi0.5), `68119b7` (GR00T AdaLN), `5869a3e` (vision_components with pre-computed pos IDs + patch_onnx_type_mismatches), `6fedff3` (GQA+RoPE spike, PyTorch 2.11 new exporter validation)
- **Reference bugs:** TensorRT #4639 (RMSNormalization opset 23), TensorRT #3908 and #4373 (SigLIP fp16 accuracy)
- **Post-mortem:** current_session line 11574 (the "only RMSNorm needs decomposition for Jetson TRT" finding)
