# shared_noise_discipline — why shared noise matters for flow-matching diffs

## The rule

> When comparing torch vs ONNX actions (or any two flow-matching implementations), **seed the same noise into both paths**. Without this, cos_sim is dominated by random variation, not export correctness. The comparison is useless.

## Why it matters

SmolVLA, pi0, pi0.5, and GR00T all use **flow-matching** (Euler-integrated denoising loops) to produce action chunks. The loop starts from a random noise tensor `noise ~ N(0, I)` of shape `[B, chunk, max_action_dim]` and integrates over N steps (typically 10).

```
actions_0 = noise                                  # random start
for step in range(1, 11):
    t = step / 10
    velocity = expert(actions_{step-1}, t, vlm_kv, state, pos_ids)
    actions_{step} = actions_{step-1} + velocity * dt    # Euler step
```

If two paths (torch reference and our ONNX pipeline) start from **different** noise tensors, the final actions diverge by an amount that depends on the noise, not on whether the paths are computing the same function. You can have a perfectly correct ONNX export that produces wildly different final actions simply because each call draws fresh noise.

Concretely, an N-dim noise tensor drawn from `N(0, 1)` has norm ≈ √N. For a `[1, 50, 32]` chunk that's `√1600 ≈ 40`. The integrated trajectory norm is in the same magnitude range (~9e1 on Apr-17 probe). Subtracting two independently-random noise-dominated trajectories gives an L2 in the same range, and cos_sim ≈ 0.

**Before noise sharing, cos_sim is noise-bounded.** After noise sharing, cos_sim measures actual export correctness.

## The canonical noise pattern

From `scripts/modal_pytorch_vs_onnx.py` and `scripts/local_full_diff.py`:

```python
import numpy as np

CHUNK = 50
MAX_ACTION_DIM = 32

# Shared noise — SAME seed, SAME tensor fed to BOTH paths.
# RandomState so torch's global RNG doesn't corrupt it.
shared_noise = np.random.RandomState(99).randn(1, CHUNK, MAX_ACTION_DIM).astype(np.float32)

# ---- Path 1: torch ----
with torch.no_grad():
    torch_actions = policy.predict_action_chunk(
        image=image, state=state, task=task,
        noise=torch.from_numpy(shared_noise),   # feed shared noise
    )

# ---- Path 2: onnx ----
onnx_actions = server.predict(
    image=image, state=state, task=task,
    noise=shared_noise,                          # feed SAME shared noise
)
```

### What breaks without it

Apr-17 Modal run `ap-oBhVQcQnjsd4uMK6lSy98D` (reflex-pytorch-vs-onnx, second iteration):

| Path | First 7 dims of first action | Note |
|---|---|---|
| Torch (deterministic) | `[-0.112, 0.113, -0.173, 0.020, 0.023, 0.009, 1.025]` | same every call |
| ONNX run 1 | `[-0.314, -0.474, 0.200, 0.120, 0.373, 0.171, -0.188]` | L2=1.494, cos=+0.082 |
| ONNX run 2 (same export, no change) | `[-0.541, -0.207, 0.066, 0.093, 0.441, 0.294, -0.723]` | L2=1.890, cos=-0.209 |

The ONNX output **changed between runs** despite no code change. Root cause (session line 10846):
> "The cos_sim numbers vary because flow-matching uses fresh random noise each call — BOTH paths get different noise, so we're comparing noisy vs noisy. Need to inject the SAME noise into both paths."

The sign of cos_sim literally flipped across two otherwise-identical runs because each draw of the initial noise happened to align differently with the final trajectory.

## Seed choice

Both `99` and `7` have been used. Pick one per script and don't change it (it enables reproducibility across sessions):

- `np.random.RandomState(99)` — `modal_pytorch_vs_onnx.py`, `scripts/local_full_diff.py`
- `np.random.RandomState(7)` — `modal_stage_diff.py`, `scripts/local_stage_diff.py`

Seed doesn't matter for correctness; consistency does. Changing seed across debugging iterations is a common mistake — you end up chasing phantom bugs as different initial conditions produce different trajectories.

## Where to inject noise (all three VLAs)

### SmolVLA (flow-matching with 10 Euler steps)

`policy.model.predict_action_chunk(...)` internally calls `policy.model.sample_actions(...)` which draws `noise = torch.randn(...)`. To override:

```python
# Option 1: monkey-patch the noise draw
def predict_with_noise(policy, image, state, task, noise_np):
    policy.reset()
    # Build batch
    batch = policy._prepare_batch(image, state, task)
    noise_t = torch.from_numpy(noise_np).to(policy.device).to(policy.dtype)
    # sample_actions accepts a `noise` kwarg in lerobot 0.5+
    return policy.model.sample_actions(batch, noise=noise_t)

# Option 2: override torch.randn in the closure
_orig_randn = torch.randn
def fixed_randn(*args, **kwargs):
    if args == (1, 50, 32):  # the action noise draw
        return torch.from_numpy(shared_noise)
    return _orig_randn(*args, **kwargs)
torch.randn = fixed_randn
try:
    result = policy.predict_action_chunk(...)
finally:
    torch.randn = _orig_randn
```

For our ONNX path (`src/reflex/runtime/server.py:_run_denoise`), add a `noise` kwarg to `predict()` and thread it through. Already done as of Apr-17.

### pi0 / pi0.5 (flow-matching)

Same pattern as SmolVLA. Noise tensor shape `[1, chunk, action_dim]` where pi0 uses `chunk=50, action_dim=32` (but native DoF varies by embodiment).

### GR00T (full-stack, action_encoder + DiT + action_decoder)

Noise applies to the DiT expert's input `action_tokens` (dim 1024, not 32 — GR00T works in token space and projects to native DoF via `action_decoder`). Shape `[1, chunk, 1024]`.

Injection point: inside `build_gr00t_expert_stack` wrapper, replace the `torch.randn(1, chunk, 1024)` call with a passed tensor.

## What shared noise proves (and doesn't)

**Proves**: if cos_sim > 0.95 with shared noise, the ONNX pipeline computes (close to) the same function as torch. Minor numerical drift only.

**Doesn't prove**: that task success on a real benchmark matches. You can have shared-noise cos_sim=1.0 and still get 0% LIBERO because the non-noise conditioning (image, text, state) is fed incorrectly (12 separate bugs identified Apr-17; see `reflex_context/02_bugs_fixed/smolvla_inference_bugs.md`).

So: shared noise is a **necessary** but **not sufficient** condition to trust an end-to-end diff.

## Extended discipline: lock ALL sources of randomness

Flow-matching initial noise is the main one, but not the only one. Also lock:

- `torch.manual_seed(0)` at the top of every diagnostic script
- `np.random.seed(0)` (legacy API; `RandomState` is preferred but some code uses the global)
- Dropout / BatchNorm — put model in `.eval()` mode before any diff
- Task input — fix the task string; don't let a changing timestamp pollute it
- Image input — fix the image (generate once from a seeded RNG and keep it across runs)
- State input — fix the state tensor
- Tokenizer — same text input → same token IDs (this is actually deterministic for the same model but easy to break if you change models mid-session)

The goal: given (image, state, task, shared_noise), the ONNX output should be bit-for-bit identical on every re-run. If it's not, there's remaining non-determinism in the ONNX path.

Before the Apr-17 noise fix, text embedding was also non-deterministic (line 7534): `_encode_text()` produced max_diff=34.7 on consecutive calls with the same input because the fallback path used `np.random.randn()` without a seed. Fix: seed by token IDs so the same instruction always produces the same embedding. After fix: max_diff=0.0.

## Checklist for every new diagnostic script

Before claiming a cos_sim number:

- [ ] Both paths run `eval()` / `torch.no_grad()`.
- [ ] Both paths load the same weights.
- [ ] Both paths preprocess the image the same way (SigLIP expects [-1, 1]; raw PIL is [0, 255]).
- [ ] Both paths preprocess the task string the same way (lerobot prepends a newline; our earlier ONNX path didn't; bug #11).
- [ ] Both paths use the same state dimension (8D for LIBERO; 6D for SO-100).
- [ ] Both paths run on the same dtype (fp32 usually; see `_pytorch_backend.py`).
- [ ] `shared_noise` is a numpy array of shape `[1, chunk, max_action_dim]`, seeded, passed to both paths.
- [ ] `torch.manual_seed(0)` at top of script.

If any of the above is off, the cos_sim number is **meaningless**.

## Forgotten discipline

Shared noise should have been the first thing built into the diagnostic harness. Instead, it surfaced as a fix mid-session (line 10846 in the Apr-17 transcript) after several iterations of chasing phantom bugs. The cost of having the discipline from day one is zero; the cost of discovering its absence is ~2h of attack-surface confusion per diff session.
