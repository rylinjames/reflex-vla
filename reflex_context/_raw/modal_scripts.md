# Modal Scripts Knowledge Base — reflex-vla

Generated from mining ~32 Modal verification/benchmark scripts in `/Users/romirjain/Desktop/building projects/reflex-vla/scripts/modal_*.py`.

Each entry captures purpose, targets, hardware, image/dependency pins, embedded gotchas, status inference, and related scripts.

---

## scripts/modal_test_export.py
**Purpose:** Earliest smoke test of SmolVLA export. Downloads `lerobot/smolvla_base`, loads the safetensors, inspects key structure, exercises the RMSNorm decomposition trick and a mini-transformer ONNX export on Modal A100. Designed to surface whether ONNX export is viable before writing a full exporter.
**Model(s):** SmolVLA (weights-only; does NOT load via lerobot — explicit "lerobot not installed on Modal" skip).
**Hardware:** A100-40GB, timeout=1200s, scaledown_window=60s.
**Key params:** dim=576 (SmolVLA hidden), opset_version=19, dynamic_axes={batch, seq_len}, test tensor `[1, 50, 576]`. RMSNorm eps=1e-6. Mini-transformer uses 8 heads.
**Dependencies:**
- torch, safetensors, transformers>=4.40, huggingface_hub, onnx, onnxruntime, onnxscript, numpy
- python=3.11, debian_slim
**Gotchas / comments:**
- Hardcodes attempt to import `lerobot.common.policies.smolvla.modeling_smolvla.SmolVLAPolicy` and logs skip when absent — indicates this predates fixing the lerobot dependency.
- Uses `torch.cuda.max_memory_allocated()` to size GPU weight residency.
- RMSNorm decomposition pattern: `variance = x.to(float32).pow(2).mean(-1, keepdim=True); x * rsqrt(variance + eps) * weight` — upcast matters for ONNX numerics.
**Status:** superseded — later scripts (modal_full_export, modal_real_export) actually load via lerobot and extract real submodules. Kept as the step-0 bake-off.
**Related:** modal_full_export.py (adds lerobot path), modal_real_export.py (loads real weights).

---

## scripts/modal_full_export.py
**Purpose:** Second iteration: load SmolVLA via `lerobot` git-installed into the image (`python=3.12`). Falls back through 3 loading strategies (lerobot module paths → transformers AutoModel → raw state_dict). Maps model structure (3-level key prefix buckets), counts VLM vs expert vs projection params, detects RMSNorm/RoPE modules by type-name matching.
**Model(s):** SmolVLA (primary), attempts full policy object load.
**Hardware:** A100-40GB, 1800s timeout.
**Key params:** Uses `snapshot_download("lerobot/smolvla_base")`, `opset_version=19`, tries vision encoder with `torch.randn(1, 3, 512, 512)`, benchmarks policy `select_action` over 20 trials with 3 warmups. Looks up VLM via attr names `vlm_with_expert|vlm|backbone|language_model`; vision via `vision_encoder|vision_tower|image_encoder`.
**Dependencies:**
- torch, safetensors, transformers>=4.51, huggingface_hub, onnx, onnxruntime, onnxscript, numpy
- `lerobot @ git+https://github.com/huggingface/lerobot.git`
- apt: git
**Gotchas / comments:**
- Has three-strategy fallback with verbose module discovery loop — shows uncertainty about lerobot's internal module layout at the time.
- The "VLM backbone: Xm, Action expert: Ym, Action projections: Zm" breakdown is the first systematic parameter accounting.
- Vision-encoder export has a broad try/except and will skip if modules not exposed separately.
**Status:** superseded by modal_vlm_export + modal_expert_export which target components directly.
**Related:** modal_test_export.py, modal_vlm_export.py (vision-focus), modal_expert_export.py.

---

## scripts/modal_real_export.py
**Purpose:** First *working* SmolVLA component extraction. Extracts suffix encoder (action_in_proj + action_time_mlp_in/out) and action_out_projection, rebuilds them as standalone `nn.Module`s, loads real weights, exports both to ONNX, validates PyTorch ↔ ONNX numerically, benchmarks a simplified 10-step denoising loop (no expert transformer, just suffix+proj).
**Model(s):** SmolVLA.
**Hardware:** A100-40GB, 600s timeout.
**Key params:** `opset_version=19`, dynamic batch axis. Infers `expert_hidden` and `action_dim` from `model.action_in_proj.weight` shape. Bench: 100 trials, 10 warmups. `chunk_size=50`. Time embedding: sinusoidal via `min_period=4e-3, max_period=4.0`.
**Dependencies:**
- torch, safetensors, huggingface_hub, onnx, onnxruntime, onnxscript, numpy
- apt: git
- python=3.12, debian_slim
**Gotchas / comments:**
- Note in Step 9 benchmark: "(suffix+proj only, no expert transformer)" — the headline Hz is misleading; bench omits the 98M expert.
- Introduces `create_sinusoidal_pos_embedding(time_val, dimension, min_period=4e-3, max_period=4.0)` helper used across later scripts.
- Skips transformers install entirely — proves SmolVLA extraction can work without it.
**Status:** superseded by modal_expert_export.py + modal_full_pipeline.py, which handle the transformer.
**Related:** modal_full_pipeline.py (full-stack version), modal_expert_export.py (transformer half).

---

## scripts/modal_expert_export.py
**Purpose:** The hardest export: the 98M action-expert transformer with RMSNorm, RoPE, and GQA cross-attention to VLM KV cache. Reconstructs expert architecture by probing state_dict keys, builds an `ExpertGQALayer` with decomposed ops, auto-detects key prefix, exports layer 0 to ONNX as a proof of concept, validates and benchmarks.
**Model(s):** SmolVLA (action expert only).
**Hardware:** A100-40GB, 1200s timeout.
**Key params:** Derives from HuggingFaceTB/SmolVLM2-500M-Video-Instruct config: `head_dim = vlm_hidden // num_attention_heads`, `expert_hidden = int(vlm_hidden * 0.75)`, `expert_intermediate = round_to_256(2/3 * 4 * expert_hidden)`. Detects actual head counts from `q_proj` / `k_proj` shapes. `max_seq_len=512` for RoPE cache. RMSNorm eps=1e-6.
**Dependencies:**
- torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy
- python=3.12
**Gotchas / comments:**
- Contains decomposed RMSNorm + decomposed RoPE modules with explicit cos/sin caching — these patterns flow into every subsequent expert exporter.
- Cross-attn detection: compares `q_shape[1]` (input dim) to `k_shape[1]`. Self-attn → same hidden; cross-attn → smaller (720 vs 320 numbers mentioned in later scripts).
- Prefix auto-detection block handles "layers.0" variants with/without `.model.` embedded — indicates multiple checkpoint layouts exist.
- The last log line estimates the full-loop cost: `estimated_full_denoise = mean_ms * 16 * 10` (16 layers × 10 steps). That estimation later becomes the target to beat.
**Status:** superseded by `src/reflex/exporters/smolvla_exporter.py`; the class patterns here survive in production.
**Related:** modal_full_pipeline.py (uses the same GQA layer), modal_real_export.py (companion suffix/proj piece).

---

## scripts/modal_full_pipeline.py
**Purpose:** First fully-composed pipeline: suffix encoder + 16-layer expert stack (self-attn + cross-attn) + final norm + action projection → 10-step flow-matching denoise. Exports the whole stack to ONNX and validates. No VLM involvement (`cross_attn_kv = zeros` placeholder).
**Model(s):** SmolVLA (expert stack, no VLM backbone yet).
**Hardware:** A100-40GB, 1800s timeout.
**Key params:** `chunk_size=50`, `num_steps=10`, `dt = -1.0/10`. Warmup 5, measure 50 latencies for the full denoise. `max_seq_len=512` for RoPE.
**Dependencies:** Same as modal_expert_export (torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy), python=3.12.
**Gotchas / comments:**
- Explains per-layer KV dimension detection: "720 for self-attn, 320 for cross-attn". The same pattern is reused in runtime server.
- Self-attn vs cross-attn classification pattern: `layer_kv_in != expert_hidden` ⇒ cross-attn. Prints the pattern once per distinct layer.
- Final norm lookup falls back to `torch.ones(expert_hidden)` with a warning if not found.
- Exports full stack with dynamic batch axis; validates max_diff for the full chunk.
**Status:** superseded by shared exporter module; still a useful reference for the denoise orchestration pattern.
**Related:** modal_expert_export.py, modal_e2e_pipeline.py (adds VLM).

---

## scripts/modal_vlm_export.py
**Purpose:** Isolate and export the 350M SmolVLM2 backbone (SigLIP vision tower + SmolLM2 16 text layers). Load from HF directly, probe top-level structure, find and export vision encoder, truncate text model from its original layer count to 16 (matching what SmolVLA uses). Runs full VLM forward to sanity-check.
**Model(s):** SmolVLM2-500M-Video-Instruct (the VLM half of SmolVLA).
**Hardware:** A100-40GB, 1200s timeout.
**Key params:** Input image=`1×3×384×384` (SigLIP default). Truncates text model `layers` to first 16. Extracts dims: `text_hidden_size`, `text_num_heads`, `text_num_kv_heads`, `text_num_layers`, `text_intermediate`, `text_head_dim`, `vocab_size`, optional vision dims.
**Dependencies:** torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy. Python=3.12. No `git` apt (since no git+ installs). No lerobot.
**Gotchas / comments:**
- Attempts vision encoder access via 4 attribute paths (`vision_model`, `vision_encoder`, `model.vision_model`, `model.vision_tower`) then falls back to scanning `named_modules` for "vision" + "encoder" in name.
- Has parallel attempts for text-model layer truncation at `model.text_model`, `text_model`, `model.language_model`, `language_model`.
- Benchmarks "text-only" forward (only 5 input ids, no images) — noting: in prod, VLM runs once per inference.
- Prints dims as JSON — the dims block is what exporter needs to wire expert dims against.
**Status:** patterns migrated into `src/reflex/exporters/smolvla_exporter.py::export_vlm_prefix`; this script is a historical probe.
**Related:** modal_e2e_pipeline.py (consumes VLM), modal_e2e_demo.py (end-to-end).

---

## scripts/modal_e2e_pipeline.py
**Purpose:** First complete end-to-end SmolVLA forward: image → VLM prefix + KV → action expert 10-step denoise → actions. Assembles VLM (truncated to 16 layers), expert layers (with cross-attn projections), and a `SmolVLAFull` wrapper. Tests dummy `1×1×3×384×384` pixels, 5 token IDs, 6-dim state.
**Model(s):** SmolVLA (VLM + expert, full stack).
**Hardware:** A100-40GB, 1800s timeout.
**Key params:** `chunk_size=50`, `num_steps=10`, image shape `[1, 1, 3, 384, 384]` (5D for SmolVLM which takes `[batch, num_images, C, H, W]`). State projection: 6→hidden_size. Truncation logic tries both `vlm.model.text_model.layers` and `vlm.language_model.model.layers`.
**Dependencies:** torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy, Pillow. Python=3.12.
**Gotchas / comments:**
- Projects VLM hidden (960) down to cross-attn KV dim (320) via learned `vlm_to_kv_proj = nn.Linear(vlm_hidden_size, vlm_kv_dim, bias=False)` — note: this creates a randomly-initialized projection, meaning this pipeline's outputs aren't numerically faithful to the true model; it's structural validation only.
- `encode_suffix` shares dims with `action_time_mlp_in`: `hidden*2 → hidden` via `silu(cat([action, t_emb]))`.
- Bench: 3 warmup, 20 measure, `mean/p50/p95/Hz` reported. All inference under `@torch.no_grad()`.
**Status:** superseded by per-layer VLM-KV approach in the current exporter; this script still useful as the first compilable E2E model.
**Related:** modal_vlm_export.py, modal_full_pipeline.py.

---

## scripts/modal_cli_export.py
**Purpose:** Validate the `reflex` CLI's export wedge actually works on Modal. Pip installs the local `src/reflex` package, runs `reflex --version`, `reflex export --dry-run`, full `reflex export lerobot/smolvla_base --target desktop`, checks outputs (files + `reflex_config.json`), and `reflex targets` command.
**Model(s):** SmolVLA via the CLI (not direct).
**Hardware:** A100-40GB, 600s timeout.
**Key params:** `--target orin-nano` for dry run (verbose), `--target desktop` for full export. Passes `/tmp/reflex_cli_export` as output dir. Timeouts: 120s dry-run, 300s full export.
**Dependencies:** torch, safetensors, transformers>=4.51, huggingface_hub, onnx, onnxruntime, onnxscript, numpy, typer, rich, pydantic>=2.0, pyyaml. Adds local dir `src/reflex`, `pyproject.toml`, `README.md`, runs `pip install -e .`.
**Gotchas / comments:**
- Expects `reflex_config.json` in output; reads `target` and `expert.num_layers` to validate config content.
- Checks `reflex targets` prints both `orin-nano` and `Jetson Thor` text — confirms target table completeness.
- The `add_local_dir + pip install -e .` pattern becomes template for later verify scripts.
**Status:** current — superseded pieces feed into larger e2e scripts but this is the canonical "does the CLI work" smoke test.
**Related:** modal_e2e_demo.py, modal_verify_cli.py, modal_verify_install_path.py.

---

## scripts/modal_sim_test.py
**Purpose:** Simulate SmolVLA acting in a robot env (without actual sim backend) — just run denoise loops from noise, measure episode latency, compare fixed 10-step vs early-stopping adaptive denoising, apply dummy joint-limit safety clamping. Episode = 1 denoise call per call, 10 episodes.
**Model(s):** SmolVLA (expert stack only; cross-attn fed with zeros).
**Hardware:** A100-40GB, 1200s timeout.
**Key params:** `chunk_size=50`, `num_steps=10` fixed, adaptive threshold `abs(v_norm - prev_norm) < 0.01` after step ≥ 3. Joint limits: `(-3.14, 3.14)` per joint, 6 joints. Apt: `git libgl1-mesa-glx libglib2.0-0`; pip: adds `gymnasium` (unused).
**Dependencies:** torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy, Pillow, gymnasium.
**Gotchas / comments:**
- Fixed vs adaptive comparison returns `speedup: Xx` — the number later drove the original turbo wedge design.
- Safety simulation counts violations out of `50*6=300` action values — gives a baseline for the guard wedge's expected violation rate on random actions.
- Gymnasium is imported but never used — legacy stub.
**Status:** superseded by modal_verify_adaptive_real.py (real VLAs) and guard-wedge tests. The synthetic adaptive test was the "only validated on 16-hidden toy" noted in modal_verify_adaptive_real.
**Related:** modal_verify_adaptive_real.py (real-model validation, explicitly references this script's weakness).

---

## scripts/modal_e2e_demo.py
**Purpose:** The canonical user story: run `reflex export` then `reflex serve` as a background process, POST to `/act`, benchmark 10 requests, hit `/config`. Tests the full CLI + server path on Modal.
**Model(s):** SmolVLA via CLI.
**Hardware:** A100-40GB, 900s timeout.
**Key params:** Port 8765. 90s server boot timeout (polled /health every 1s). Serves with `--device cpu`. 10 benchmark requests, timeout=30s each.
**Dependencies:** torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy, Pillow, typer, rich, pydantic>=2.0, pyyaml, fastapi, uvicorn, httpx. Local src/reflex install.
**Gotchas / comments:**
- **Post-mortem note in code**: server stdout is redirected to a FILE (`/tmp/reflex_server.log`), not a pipe, "to avoid pipe-buffer deadlock". Documented explicitly. This pattern propagates to all later verify scripts with `stdout_fh = open(log_path, "wb")`.
- Poll interval prints log tail every 5s if `model_loaded` flag not set, every 10s if /health not responding.
- Sends dummy instruction "pick up the red cup" + state `[0.1,0.2,...,0.6]`.
- Response is expected to include `actions`, `latency_ms`, `hz`, `inference_mode`.
**Status:** current (user-story canonical test).
**Related:** modal_e2e_all_models.py (same but across 4 VLAs), modal_cli_export.py.

---

## scripts/modal_test_pi0.py
**Purpose:** Verify the pi0 exporter pipeline. Downloads `lerobot/pi0_base` (3.5GB), calls `reflex.checkpoint.load_checkpoint` + `detect_model_type`, calls `reflex.exporters.pi0_exporter.build_pi0_expert_stack(state_dict, head_dim=128)`, then runs the full `reflex export` CLI and inspects `reflex_config.json`.
**Model(s):** pi0 (π₀ base, Physical Intelligence, 3.5GB).
**Hardware:** A100-40GB, 1200s timeout.
**Key params:** `head_dim=128`. Export timeout 600s. Expected detection: `model_type == "pi0"` else fail.
**Dependencies:** Same base as CLI tests + local src/reflex install.
**Gotchas / comments:**
- Deletes `state_dict` between build and CLI export to free memory (~3.5GB checkpoints are tight on A100-40GB).
- Prints sample keys filtered by `"expert"`/`"action"` for manual inspection.
- Asserts `config["expert"]["action_dim"]` and `num_layers` are populated.
**Status:** current smoke test for pi0 support.
**Related:** modal_test_pi05.py (sibling), modal_e2e_all_models.py.

---

## scripts/modal_test_pi05.py
**Purpose:** Verify the pi0.5 exporter (AdaRMSNorm variant of pi0). Downloads `lerobot/pi05_base` (3.62GB), detects AdaRMSNorm markers (`input_layernorm.dense` keys) and time_mlp keys, builds the pi05 expert stack, runs a forward pass, then runs `reflex export` and captures validation lines from stdout.
**Model(s):** pi0.5.
**Hardware:** A100-40GB, 1200s timeout.
**Key params:** `head_dim=128`, `chunk_size=50` for forward test, timestep=0.5. Expected detection: `model_type == "pi05"`, `uses_adarms=True`.
**Dependencies:** same as modal_test_pi0 (torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy, Pillow, typer, rich, pydantic>=2.0, pyyaml).
**Gotchas / comments:**
- Explicitly looks for two types of keys to confirm AdaRMSNorm: `input_layernorm.dense*` (scale projection) and `time_mlp*` (time conditioning MLP).
- Greps stdout for "Validation" and "max_diff" lines rather than parsing JSON — signals exporter writes unstructured stdout for validation.
**Status:** current.
**Related:** modal_test_pi0.py (base arch), modal_e2e_all_models.py.

---

## scripts/modal_test_gr00t.py
**Purpose:** Verify GR00T N1.6 (NVIDIA, 3B) export. Loads `nvidia/GR00T-N1.6-3B` (6.6GB, 2 shards), detects `action_head.model.transformer_blocks.*` DiT keys, calls `build_gr00t_expert_stack(state_dict, embodiment_id=0)`, tests forward pass, runs `reflex export`.
**Model(s):** GR00T-N1.6-3B.
**Hardware:** A100-40GB, 1800s timeout.
**Key params:** `embodiment_id=0`. Prints `meta`: blocks, hidden, num_heads × head_dim, ff_inner, vlm_kv_dim, chunk_size, output_dim, total_params_m. Export timeout 900s.
**Dependencies:** Same as other CLI tests.
**Gotchas / comments:**
- GR00T is DiT-based (not a decoder-style expert like pi0 / SmolVLA) — different attn layout.
- The meta returned by `build_gr00t_expert_stack` advertises `chunk_size` as a model-internal value, not a user-chosen one.
- The per-embodiment nature of GR00T means `embodiment_id=0` is pinned (first embodiment in the header table).
**Status:** current (initial GR00T integration test).
**Related:** modal_probe_gr00t.py (weight-shape exploration), modal_test_gr00t_full.py (adds action_encoder/decoder for full-loop).

---

## scripts/modal_probe_gr00t.py
**Purpose:** Tiny, surgical shape-dump script: load GR00T, list all `action_encoder`/`action_decoder`/`state_encoder`/`position_embedding`/`vlln`/`timestep_encoder`/`proj_out` weight shapes, plus block-0 attention shapes. Used to design the "full-stack" GR00T exporter without guessing.
**Model(s):** GR00T-N1.6-3B.
**Hardware:** A100-40GB, 600s timeout (no GPU needed, but GPU'd for HF download parallelism).
**Key params:** none (pure shape probe).
**Dependencies:** minimal — torch, safetensors, huggingface_hub + local reflex install.
**Gotchas / comments:**
- Header docstring explicitly says: "The existing Reflex GR00T exporter skips [action_encoder/action_decoder] — adding them lets `reflex serve` do full-loop denoising (raw actions in, raw actions out)". Clear flag that the base exporter was incomplete.
- Emits "block 0 sample shapes" for sanity — DiT blocks have different projections than LM expert blocks.
**Status:** current (diagnostic / design aid).
**Related:** modal_test_gr00t_full.py (uses the probed shapes).

---

## scripts/modal_test_gr00t_full.py
**Purpose:** Verify the GR00T *full-stack* export: `action_encoder → DiT → action_decoder`, pinned to embodiment 0. Validates that the ONNX emitted by `reflex export` can be driven by `reflex serve`'s standard flow-matching loop (raw actions in, raw actions out — no per-embodiment plumbing at serve time).
**Model(s):** GR00T-N1.6-3B (full stack incl. encoders/decoders).
**Hardware:** A100-40GB, 1800s timeout.
**Key params:** `embodiment_id=0`. Meta adds `raw_action_dim`, `full_stack_params_m`, `full_stack_buffers_m` on top of standard meta. Bench uses 7-dim state (`[0.0]*7`) — GR00T embodiment 0 has 7 DoF.
**Dependencies:** torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy, Pillow, typer, rich, pydantic>=2.0, pyyaml, fastapi, uvicorn, httpx. Local reflex install.
**Gotchas / comments:**
- Runs `reflex serve` and POSTs `/act`; confirms the full-stack export is serve-compatible.
- Port 8799. 60s server boot timeout (lower than e2e_demo's 90s — GR00T loads faster once the model is in place).
**Status:** current (adds the encoder/decoder full-stack flow that modal_test_gr00t left as a gap).
**Related:** modal_test_gr00t.py, modal_probe_gr00t.py.

---

## scripts/modal_e2e_all_models.py
**Purpose:** Publishable benchmark across all 4 supported VLA models (SmolVLA, pi0, pi0.5, GR00T). For each: `reflex export` → `reflex serve` (CPU) → 10 POST /act requests → record mean/p50/p95/Hz. Emits a summary table.
**Model(s):** SmolVLA + pi0 + pi0.5 + GR00T.
**Hardware:** A100-40GB, 2400s timeout.
**Key params:** export timeout=600s; serve boot timeout=60s; 10 benchmark requests; ports 8765..8768. Device=cpu. `rm -rf export_dir` between models to free disk (checkpoints are big).
**Dependencies:** standard CLI test deps; local reflex install.
**Gotchas / comments:**
- Comment in the models list: `("gr00t", "nvidia/GR00T-N1.6-3B")` — header says "3 supported" in some log lines but the list has 4 entries. The "3" is stale.
- Parses exporter stdout for `max_diff` to record per-model `validation_max_diff` alongside benchmark numbers.
- Prints summary table: `Model | Export | max_diff | Mean ms | Hz`.
**Status:** current publishable bench target.
**Related:** modal_test_pi0/pi05/gr00t/smolvla (per-model variants), modal_bench_onnx_vs_torch.py (pure-GPU perf).

---

## scripts/modal_verify_cli.py
**Purpose:** Confirm `reflex --help` lists all 7 wedges and exercise 4 wedge CLIs: `reflex turbo`, `reflex split`, `reflex adapt`, `reflex check` — all against small/fake inputs so nothing big downloads.
**Model(s):** N/A (CLI surface check); `reflex check lerobot/smolvla_base` for the `check` wedge smoke.
**Hardware:** cpu=2, timeout=300s (no GPU).
**Key params:** `turbo --trials 2 --action-dim 6 --chunk-size 5`, `split --prefer edge --output /tmp/split.json`, `adapt --num-joints 6 --framework lerobot --output /tmp/embodiment.json`, `check lerobot/smolvla_base --target desktop`.
**Dependencies:** torch, safetensors, huggingface_hub, numpy, Pillow, typer, rich, pydantic>=2.0, pyyaml. No onnxruntime. Local reflex install.
**Gotchas / comments:**
- Wedge list hardcoded as `["export", "serve", "guard", "turbo", "split", "adapt", "check"]` — note `check` is still here (before the task #15 merge of check → `validate --quick`).
- `turbo`, `split`, `adapt` are the commands later deprecated (per memory project_reflex_vla notes: "Remove split, adapt, turbo commands with deprecation warnings", task #14 completed).
**Status:** superseded post-deprecation; the wedge list and `turbo/split/adapt` invocations would fail now.
**Related:** modal_verify_strict_providers.py (newer wedge-related verify).

---

## scripts/modal_bench_onnx_vs_torch.py
**Purpose:** Answers two questions: (1) Does exported ONNX actually beat `torch.compile` eager PyTorch? If not, reflex export has no reason to exist. (2) What's the weights + peak-forward memory per model — determines Jetson SKU feasibility (Nano 8GB / Orin 32GB / Thor 128GB). For each of the 4 VLAs: PyTorch eager → `torch.compile(mode="reduce-overhead")` → ONNX Runtime CUDAExecutionProvider. 10 warmup, 100 trial timing.
**Model(s):** SmolVLA, pi0, pi0.5, GR00T.
**Hardware:** A100-40GB, 2400s timeout.
**Key params:** `chunk_size=50`, dynamic `action_dim` from meta. ORT providers: `[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]`. FP32 weights. `opset_version=19`.
**Dependencies:** torch, safetensors, huggingface_hub, transformers>=4.51, onnx, `onnxruntime-gpu` (not onnxruntime), onnxscript, numpy, Pillow, typer, rich, pydantic>=2.0, pyyaml.
**Gotchas / comments:**
- Comment: "onnxruntime-gpu ≥1.17 bundles its own CUDA libs, so we can pip install cleanly." — Early version believed this; the later modal_bench_path_b.py script pins torch==2.5.1 + onnxruntime-gpu==1.20.1 + installs explicit CUDA / cuDNN wheels because this assumption broke.
- `bench_callable` helper used consistently thereafter.
- Emits Jetson-fit table: FP16 × 2 overhead, checks against 8/32/128 GB SKUs.
**Status:** superseded by modal_bench_path_b.py (fixes ORT CUDA provider loading), but still a useful first-pass diagnostic.
**Related:** modal_bench_path_b.py (fixed CUDA loading + adds CUDA graph), modal_bench_trt_fp16.py (adds TensorRT).

---

## scripts/modal_bench_path_b.py
**Purpose:** "Path B benchmark: can Reflex actually beat torch.compile on GPU?" Tests 4 execution paths per VLA: PyTorch eager, `torch.compile(reduce-overhead)` single step + full 10-step Python-driver loop, ORT-GPU (fixed provider loading), Reflex turbo CUDA graph capture of the full denoise. Has very explicit version pinning to address the silent-CPU-fallback bug from the previous bench.
**Model(s):** SmolVLA + pi0 + pi0.5 + GR00T.
**Hardware:** A100-40GB, 2700s timeout.
**Key params:** `chunk_size=50`, `num_steps=10`. 10 warmup, 50 trials. Full-loop warmup 3, trials 30. CUDA graph: `TurboOptimizer(TurboConfig(strategy="cuda_graph"))`, captures the full 10-step loop; replays measured 100 trials.
**Dependencies:**
- `torch==2.5.1` (CUDA 12.4, not 13 like torch 2.11) — inline comment: "Torch 2.5 uses CUDA 12.4 (NOT 13 like torch 2.11)"
- `onnxruntime-gpu==1.20.1` — inline comment: "ORT 1.20.x uses cuDNN 9 + CUDA 12.x"
- transformers>=4.40,<5.0, safetensors, huggingface_hub, onnx, onnxscript, numpy<2.0, Pillow, typer, rich, pydantic>=2.0, pyyaml
- `pip install -e . --no-deps` for reflex (to avoid re-pulling pinned packages)
**Gotchas / comments:**
- **Extensive post-mortem in image construction comments**: "FIXED image: pin torch to a version using CUDA 12 (matches ORT 1.20), install matching cuDNN and cuBLAS."
- Prints torch + CUDA + ORT version and `ort.get_available_providers()` on startup.
- Summary table: `Model | Params | Compile×10 | ORT×10 | Reflex CUDA-graph` — the "×10" columns extrapolate single-step bench to the 10-step loop (or use the actual 10-step loop number when available).
**Status:** current (the canonical "does reflex beat compile" bench after CUDA-provider fixes).
**Related:** modal_bench_onnx_vs_torch.py (preceded), modal_bench_trt_fp16.py (adds TRT).

---

## scripts/modal_verify_strict_providers.py
**Purpose:** Phase I.1 verification: `reflex serve` now fails loudly instead of silently falling back to CPU when `onnxruntime-gpu` isn't available. Four scenarios: (1) GPU box + `onnxruntime-gpu` + `--device cuda` → starts, (2) GPU box + `--device cpu` → starts, (3) CPU-only `onnxruntime` + `--device cuda` default → exits 1 with clear error, (4) scenario 3 + `--no-strict-providers` → starts with fallback.
**Model(s):** N/A (fake-export ONNX Identity graph stands in).
**Hardware:** A10G on all scenarios.
**Key params:** Uses a tiny fake ONNX with `Identity` op for `noisy_actions → velocity`. Custom `_make_fake_export` writes a 3-input `reflex_config.json` (`model_type=smolvla`, chunk=50, action_dim=32). Ports 9001..9004.
**Dependencies:**
- `image_gpu`: torch==2.5.1, onnxruntime-gpu==1.20.1, transformers>=4.40,<5.0, numpy<2.0, fastapi, uvicorn, httpx, typer, rich, pydantic>=2.0, pyyaml.
- `image_cpu`: onnxruntime==1.20.1 (CPU-only — intentionally wrong variant).
- Both use `pip install -e . --no-deps`.
**Gotchas / comments:**
- Comment: "CPU-only image — installs `onnxruntime` (NOT the -gpu variant). Used to verify the CLI exits with a helpful error when a user pip-installed the wrong package."
- Explicit mock ONNX generation via `onnx.helper.make_node("Identity", ...)` — this reusable helper pattern appears in modal_verify_wedge_compose / modal_verify_batching.
- Scenario 3 verdict logic: "if exit_code == 1 → PASS" (refused to silently fall back).
**Status:** current (I.1 is landed; script documents the exact expected behavior).
**Related:** modal_verify_wedge_compose.py, modal_verify_install_path.py.

---

## scripts/modal_verify_wedge_compose.py
**Purpose:** Phase I.2 verification: `reflex serve` composes wedges via flags. Runs `reflex serve` with combinations of `--safety-config`, `--adaptive-steps`, `--deadline-ms`, `--cloud-fallback`, and POSTs /act to verify the response surfaces telemetry keys from each wedge (`safety_violations`, `adaptive_enabled`, `deadline_exceeded`, `split_enabled`).
**Model(s):** Fake SmolVLA ONNX (`Identity` op that produces velocity ≈ noise, which integrates to unsafe actions → guaranteed safety violations).
**Hardware:** `cpu=2`, timeout 600s (no GPU needed).
**Key params:** 4 scenarios on ports 9100..9103. Scenario 2: `--safety-config /tmp/safety.json` with position_min/max crushed to `[-0.01, 0.01]` per joint. Scenario 3: `--adaptive-steps`. Scenario 4: all wedges + `--deadline-ms 1000000` (so deadline never fires) + fake `--cloud-fallback` URL.
**Dependencies:** Same as strict-providers image (torch==2.5.1, onnxruntime-gpu==1.20.1 — uses GPU image but runs on CPU). Adds `yourdfpy`, `trimesh` for URDF pipeline.
**Gotchas / comments:**
- Inline comment on Identity choice: "With random input that has values around ~N(0,1), the integration loop produces actions around ~N(0, 10), which puts them outside typical joint limits. Good for testing the guard wedge actually fires."
- Safety limits are deliberately `[-0.01, 0.01]` to force violations on every action.
- Scenario 1 assertion: "no wedge keys leaking" when no flags enabled (negative test).
**Status:** current.
**Related:** modal_verify_strict_providers.py, modal_verify_adaptive_real.py.

---

## scripts/modal_bench_trt_fp16.py
**Purpose:** Phase II benchmark: TensorRT FP16 engines vs `torch.compile` on **A10G**. A10G = Ampere = closest cloud GPU to Jetson Orin; proxy for "what happens on Jetson with TRT FP16". 4 paths per model: PyTorch eager, `torch.compile(reduce-overhead)`, ORT-GPU FP32, TensorRT FP16 engine (built via trtexec).
**Model(s):** SmolVLA + pi0 + pi0.5 + GR00T.
**Hardware:** **A10G**, timeout 3600s.
**Key params:** `chunk_size=50`. 10 warmup, 50 trial. `trtexec --fp16 --memPoolSize=workspace:4096MiB`. Bench cmd: `trtexec --loadEngine=... --warmUp=200 --iterations=500 --avgRuns=100 --useCudaGraph`. Parses `GPU Compute Time: min=, max=, mean=, median=, percentile(99%)=` line.
**Dependencies:**
- Base image: `nvcr.io/nvidia/tensorrt:24.10-py3` (TRT 10.5, CUDA 12.6, cuDNN 9, trtexec included)
- torch==2.5.1, onnxruntime-gpu>=1.20,<1.24, etc.
- apt: git. `pip install -e . --no-deps`.
- Validates `trtexec --help` at image build time.
**Gotchas / comments:**
- **Crucial post-mortem comment**: "Our ONNX export has static shapes baked in (no dynamic_axes), so we MUST NOT pass --minShapes/--optShapes/--maxShapes — trtexec rejects them for static models. The engine is fixed at the export shape (batch=1, chunk=50, action_dim from model)." This is the reason for the `modal_verify_trt_with_batch.py` caveats.
- "trtexec already measures GPU compute latency precisely" — justifies not using the Python TRT bindings (which `nvcr.io/nvidia/tensorrt` doesn't pre-install; would require `/opt/tensorrt/python/python_setup.sh`).
- Summary picks `winner = min(numeric, key=numeric.get)` across `{compile, ORT-GPU, TRT-FP16}` per model.
**Status:** current (Phase II Jetson-proxy bench).
**Related:** modal_bench_path_b.py (GPU-general), modal_verify_trt_with_batch.py (TRT + batching interaction).

---

## scripts/modal_verify_batching.py
**Purpose:** Phase III verification: `reflex serve --max-batch N` actually batches. 3 configs (batch=1, 4, 8), each fires 16 concurrent POST /act requests (asyncio + httpx.AsyncClient), measures throughput multiplier vs baseline. Asserts `batch_size` key surfaces in response JSON.
**Model(s):** Fake SmolVLA (Identity ONNX, dynamic batch axis this time).
**Hardware:** `cpu=4`, timeout 600s.
**Key params:** `--max-batch N --batch-timeout-ms 20`. N=1|4|8, 16 concurrent. Ports 9200..9202. `--no-strict-providers` because this is CPU-only and the ONNX is fake.
**Dependencies:** Same base as strict-providers + extra: `nvidia-cudnn-cu12>=9.0,<10.0`, `nvidia-cublas-cu12>=12.0,<13.0`, yourdfpy, trimesh. (Inherited — not actually used, script runs on CPU.)
**Gotchas / comments:**
- Dynamic batch axis fix: `TensorProto.FLOAT, ["batch", 50, action_dim]` — critical — previous fake-export scripts had static shapes that would fail batching.
- Comment: "the fake-Identity-op test only measured queue overhead" — acknowledges this is a synthetic test, and why modal_verify_batching_real.py exists.
- `throughput_qps = n_concurrent / elapsed` — simple throughput metric.
**Status:** current (synthetic batch test — queue-only).
**Related:** modal_verify_batching_real.py (uses real pi0 model).

---

## scripts/modal_verify_batching_real.py
**Purpose:** Phase III.2: real-model batching. Same as modal_verify_batching but uses a real exported pi0 (~50ms/chunk on GPU) so batching should give 2-3× throughput. Tests batch=1/4/8/16 with 32 concurrent requests on A10G GPU.
**Model(s):** pi0 (real, via `reflex export lerobot/pi0_base --target desktop`).
**Hardware:** A10G, timeout 2400s.
**Key params:** 32 concurrent, batch sizes 1/4/8/16. Ports 9300..9303. Device=cuda (not cpu). 180s server boot timeout (pi0 ONNX ~1.7GB; first ORT-GPU session creation is slow).
**Dependencies:**
- Base: `nvcr.io/nvidia/tensorrt:24.10-py3` (same as TRT bench).
- onnxruntime-gpu>=1.20,<1.24, torch==2.5.1, etc.
- **Post-mortem comment in image**: "Use NVIDIA's TRT container so cuDNN 9 (including libcudnn_adv) is already on the system path. The pip-installed nvidia-cudnn-cu12 wheel is missing libcudnn_adv.so.9 which ORT 1.20+ requires."
**Gotchas / comments:**
- Comment on pipe handling: "Use a real file for stdout — subprocess.PIPE deadlocks if the process logs more than one OS pipe buffer (~64KB) before we read." Same lesson as modal_e2e_demo.
- 5 sequential warmups before concurrent bench — keeps GPU hot.
- Records per-request `amortized_latency_ms` alongside wall-clock throughput.
**Status:** current.
**Related:** modal_verify_batching.py, modal_verify_trt_with_batch.py.

---

## scripts/modal_verify_adaptive_real.py
**Purpose:** Phase IV: validate that `reflex turbo --strategy adaptive` actually saves time on **real** VLAs. The current adaptive-stop heuristic in `src/reflex/kernels/turbo.py` and the inline adaptive-stop in `src/reflex/runtime/server.py:_run_denoise` stops the denoise loop when consecutive velocity-norm deltas drop below 0.01. That heuristic was only validated on a synthetic 16-hidden toy model.
**Model(s):** SmolVLA + pi0 + pi0.5 + GR00T.
**Hardware:** A10G, timeout 2400s.
**Key params:** `NUM_STEPS=10`, `NUM_TRIALS=25`, `THRESHOLD=0.01`. Measures per-step velocity norms, records step at which `abs(delta) < THRESHOLD` (after step ≥ 2). Also records max-abs action diff at trigger-step vs full-10-step output.
**Dependencies:** Base `nvcr.io/nvidia/tensorrt:24.10-py3` + torch==2.5.1 + onnxruntime-gpu.
**Gotchas / comments:**
- **Explicit post-mortem in docstring**: the heuristic was validated on a "synthetic 16-hidden toy model" (= modal_sim_test.py), and this script exists to test whether it survives on real VLAs. If velocities never converge OR action diff is large, adaptive "needs to be removed or rewritten".
- VERDICT table columns: `Model | triggered | mean_step | savings | action_diff`.
- `trials_that_triggered` counts 25-trial subset that actually hit the threshold before step 10.
**Status:** current (decisive phase-IV diagnostic; outcome determines whether adaptive stays in the product).
**Related:** modal_sim_test.py (the toy-model bench this disproves/validates).

---

## scripts/modal_verify_install_path.py
**Purpose:** Test that the README quickstart install path actually works on a clean box. Spins up a clean image (no reflex preinstalled), runs the EXACT README commands: `pip install 'reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla'`, `reflex --help`, `reflex models`, `reflex export ...`, `reflex serve ... --device cuda`, POST /act via stdlib `urllib.request` (no pip-installed test client to avoid polluting the test).
**Model(s):** SmolVLA (smallest for fastest test).
**Hardware:** A10G, timeout 900s.
**Key params:** `reflex export lerobot/smolvla_base --target desktop --output /tmp/sv`; serve on `http://127.0.0.1:8765 --device cuda`. 240s server boot timeout — comment: "server now does a warmup pass during lifespan startup. With TRT EP enabled, the first denoising loop also builds + caches a TRT engine, which can take 30-90s for our smallest model and longer for pi0/gr00t."
**Dependencies:** Image starts from `nvcr.io/nvidia/tensorrt:24.10-py3`, adds `git curl`. **No other pip installs** — the test is the install command itself.
**Gotchas / comments:**
- Uses stdlib `urllib.request` for GET/POST: "the whole point of this test is to verify what users get with the advertised install path" — don't pollute with httpx.
- Step 5 is a pure check for `expert_stack.onnx` + `reflex_config.json`; no command, just `os.listdir()`.
- Captures server log to `/tmp/serve.log` and prints the last 1500 chars on failure.
**Status:** current (must-pass gate before every public release / README change).
**Related:** modal_verify_bench_all.py (bench equivalent), modal_verify_trt_with_batch.py.

---

## scripts/modal_verify_bench_all.py
**Purpose:** Verify `reflex bench` works for all 4 supported VLAs with auto-TRT FP16. smolvla install-path test confirmed auto-TRT for that model; this extends to pi0, pi0.5, GR00T where longer engine builds surface edge cases. For each: `reflex export <hf_id> --target desktop`, then `reflex bench <export_dir> --iterations 50 --warmup 10 --device cuda`; parse `Inference mode:` and `mean/p50/p95/p99` lines from stdout.
**Model(s):** SmolVLA + pi0 + pi0.5 + GR00T.
**Hardware:** A10G, timeout 3600s.
**Key params:** `--iterations 50 --warmup 10 --device cuda`. Install reflex at the start via the same README git+pip command as modal_verify_install_path. Export timeout 900s per model.
**Dependencies:** Base `nvcr.io/nvidia/tensorrt:24.10-py3`; reflex installed at runtime via pip.
**Gotchas / comments:**
- Parses stdout with explicit regexes: `mean|p50|p95|p99  Xms` and `Inference mode: \w+` — stable format from `reflex bench`.
- `rm -rf export_dir` between models: "Free disk between models — checkpoints are huge".
- Summary table: `Model | mean_ms | p95_ms | mode | export_s`.
**Status:** current.
**Related:** modal_verify_install_path.py.

---

## scripts/modal_verify_trt_with_batch.py
**Purpose:** Verify the TRT FP16 + multi-batch interaction. Known sharp edge: TRT engines compile against a specific input shape. Our exporters bake static shapes (batch=1). When `--max-batch N` fires a batched request, TRT engine has to handle batch=N. ORT's TRT EP can (a) fall back to CUDA EP for unknown shapes, or (b) build+cache a new engine per distinct shape (slow first hit). Tests both paths and reports whether batched mode silently falls back or fails.
**Model(s):** pi0 (real).
**Hardware:** A10G, timeout 2400s.
**Key params:** 3 scenarios (batch=1, 4, 8), 8 concurrent requests each. Ports 9400..9402. 300s server boot timeout (TRT engine cold-start).
**Dependencies:** Base `nvcr.io/nvidia/tensorrt:24.10-py3`; reflex via git+pip at runtime.
**Gotchas / comments:**
- Docstring lists 3 resolutions if batching fails with auto-TRT: "(a) Disable TRT EP when --max-batch > 1, (b) Export with dynamic batch shape, (c) Pre-build engines for common batch sizes". This script is the diagnostic that decides which approach.
- Captures both `startup_inference_mode` (from `/health`) and `sample_response_mode` (from `/act`) — compares them; a mismatch means the first /act request forced a fallback.
**Status:** current (decisive diagnostic).
**Related:** modal_bench_trt_fp16.py, modal_verify_batching_real.py.

---

## scripts/modal_trajectory_replay.py
**Purpose:** Trajectory replay smoke test. Exports SmolVLA, starts serve with `--device cpu --no-strict-providers`, feeds synthetic 224×224 images with 5 different instructions, verifies (1) shape is `[50, action_dim]`, (2) no NaN/Inf, (3) actions bounded `< 50`, (4) different instructions produce different action chunks (semantic diversity: min L2 > 1e-6).
**Model(s):** SmolVLA via CLI.
**Hardware:** A10G, 1800s timeout.
**Key params:** `NUM_EPISODES=5` (instructions), `L2_THRESHOLD=2.0` (dead constant — not used in final code). Port 8321. Serve boot polls every 2s for up to 120s. Request timeout 60s.
**Dependencies:** adds `datasets` (unused), `ffmpeg` apt (for LeRobot video-loader, but synthetic images don't need it).
**Gotchas / comments:**
- Comment: "Instead of downloading a dataset (LeRobot v2 stores images as video files that need their custom loader), we test the server directly with synthetic images." — doc acknowledges dataset replay path was skipped.
- Semantic diversity assertion: after generating action chunks for 5 instructions, pairwise L2-diff, assert `min_diversity >= 1e-6` — "some instruction pairs produce identical actions" ⇒ FAIL.
- `vlm_conditioning` key surfaced in response — confirms serve is plumbing the VLM prefix through.
**Status:** current (cheap shape+bounds+semantics smoke).
**Related:** modal_libero10.py (real-sim replay).

---

## scripts/modal_libero10.py
**Purpose:** LIBERO-10 evaluation via `vla-eval` on Modal A10G. Thin runner: `reflex export lerobot/smolvla_libero`, launch `reflex.runtime.adapters.vla_eval` as a background server, run `vla-eval run` against it on LIBERO-10. All inference logic lives in `reflex.runtime.ReflexServer`; this script is only Modal image + LIBERO sim plumbing.
**Model(s):** SmolVLA LIBERO fine-tune (`lerobot/smolvla_libero`).
**Hardware:** A10G, 7200s (2h) timeout.
**Key params:** `episodes_per_task=1`, `max_steps=150`, `suite="libero_10"`, `seed=7`, `num_steps_wait=10`. `send_state=True`, `send_wrist_image=True`. `REFLEX_ACTION_DIM_OUT=7` (LIBERO = 6 joints + gripper). MUJOCO_GL=osmesa.
**Dependencies:** huge image:
- apt: git, libgl1, libglib2, libegl1, libglvnd0, ffmpeg, cmake, build-essential, libosmesa6, libosmesa6-dev
- pip stack: torch, safetensors, huggingface_hub, transformers>=4.51, onnx, onnxruntime, onnxscript, numpy, Pillow, pydantic>=2.0, fastapi>=0.100.0, uvicorn>=0.23.0, typer, rich, pyyaml, mujoco>=3.0, gymnasium, **vla-eval**, **robosuite==1.4.1** (pinned), h5py, **bddl==1.0.1** (pinned — LIBERO imports bddl.parsing), future, robomimic, hydra-core>=1.1, easydict, einops, opencv-python-headless, gym (old gym — LIBERO's venv.py uses gym not gymnasium)
- git clone + `pip install -e .` of LIBERO from GitHub; applies `patch_libero.py` to fix interactive input prompts that hang on import.
**Gotchas / comments:**
- **Multi-paragraph post-mortem in image comment**: "LIBERO's setup.py is install_requires=[]; its envs import bddl, robomimic, hydra-core at reset() time. Installing the full requirements.txt would downgrade transformers/numpy/etc. and nuke the ONNX export stack. Install only the runtime-required deps with flexible versions."
- Inline comment on pinning: "robosuite 1.5+ moved module paths — pin 1.4.1 which LIBERO expects."
- Uses **both** `osmesa` and `egl` for MuJoCo rendering: "osmesa for MuJoCo software rendering (EGL hangs silently on some debian_slim+NVIDIA combos with LIBERO; osmesa is reliable but slow)."
- Step 3b: LIBERO `env.reset()` **smoke test** in a subprocess before running vla-eval: "If this hangs, vla-eval would hang too — fail fast here with a clear message." — big post-mortem lesson baked into the script.
- `send_state=True, send_wrist_image=True` has a pointed comment: "CRITICAL: without these, vla-eval sends only images + task_description. Our first-predict dump showed state=none, which means the model was predicting actions from zero state vectors — garbage trajectories no matter what the VLM pipeline looked like."
- Idle-timeout guard: kills vla-eval if no stdout for 600s (10 min). Comment: "osmesa first-scene compilation can be slow". Overall timeout 3600s.
- Streams vla-eval stdout line-by-line via select: "subprocess.run(capture_output=True) ... buffers until the subprocess exits, which meant we couldn't tell if a run was hung vs mid-episode for 50+ minutes."
- Dumps adapter log tail for key markers: "First predict", "First predict actions", "VLM orchestrator failed", "dummy conditioning", "ERROR", "Traceback".
- Checks for normalizer files: `policy_preprocessor_step_5_normalizer_processor.safetensors`, `policy_postprocessor_step_0_unnormalizer_processor.safetensors`. Note about ongoing task (task #24 "Add normalizer support to adapter" in memory): "Verify the normalizer/unnormalizer files landed (the thing that's broken now)".
**Status:** current and being iterated (task #23 "Ship LIBERO-10 Modal run and capture task-success number" is in_progress).
**Related:** modal_trajectory_replay.py (synthetic replay), modal_pytorch_vs_onnx.py (decides whether 0% task success is export bug or sim bug), modal_stage_diff.py.

---

## scripts/modal_pytorch_vs_onnx.py
**Purpose:** "The decisive test". Runs PyTorch reference policy and our ONNX pipeline on IDENTICAL preprocessed inputs with SHARED flow-matching noise, reports L2/cos_sim on the first action of the chunk. Goal: if cos_sim > 0.95, LIBERO 0% is a sim/env issue; if 0.5 < cos ≤ 0.95, minor numeric drift; if cos ≤ 0.5, structural bug in export.
**Model(s):** SmolVLA LIBERO fine-tune (`lerobot/smolvla_libero`).
**Hardware:** A10G, 1200s timeout.
**Key params:** 256×256 synthetic image, 3 cameras of same content, 8-dim state (LIBERO: eef_pos(3)+axis_angle(3)+gripper_qpos(2)). Task: "put the red bowl on the plate". Shared noise from `np.random.RandomState(99).randn(1, chunk, max_action_dim)` ensures both paths hit the same flow-matching initial condition.
**Dependencies:** lerobot (from pypi), num2words, transformers>=4.51, standard ONNX stack.
**Gotchas / comments:**
- **Very explicit post-mortem in code**: "Generate SHARED noise so the flow-matching initial condition is identical in both paths. Without this, cos_sim is dominated by random noise drift, not export correctness."
- Uses `PolicyProcessorPipeline.from_pretrained` with explicit `to_transition`, `to_output`, `to_policy_action` converter args — LeRobot 0.x API.
- Runs torch on GPU (`device = cuda if available`), our ONNX pipeline on CPU via `ReflexServer(export_dir, device="cuda", strict_providers=False)` (note: server says cuda, but this is the ORT-GPU fallback path).
- Final verdict logic explicitly articulates the 3-way split (correct / minor drift / major divergence).
- Comment on 8D state: "LeRobot's LIBERO dataset uses 8D state: eef_pos(3) + axis_angle(3) + gripper_qpos(2). The preprocessor's normalizer stats are shape (8,) so we need 8D here."
**Status:** current and critical (directly informs whether to ship current export or keep debugging).
**Related:** modal_stage_diff.py (per-stage breakdown), modal_libero10.py.

---

## scripts/modal_stage_diff.py
**Purpose:** Per-stage PyTorch-vs-ONNX diff to localize divergence. Builds deterministic inputs, compares intermediate tensors at each pipeline stage: (1) vision encoder output per camera, (2) text embeddings, (3) state projection, (4) per-layer VLM k (post-RoPE) and v from `decoder_prefill.onnx`, (5) expert one-step velocity with identical vlm_k/vlm_v. "First stage where L2 diverges is where the bug lives."
**Model(s):** SmolVLA LIBERO fine-tune.
**Hardware:** A10G, 1200s timeout.
**Key params:** 512×512 synthetic image (matches vision ONNX export size), 8-dim state, task string. `num_keep = min(16, len(tm.layers))` — SmolVLA truncates VLM to 16 layers. Chunk and max_action_dim from policy config. Noise via `RandomState(7)`.
**Dependencies:** Same stack as modal_pytorch_vs_onnx (lerobot + standard ONNX).
**Gotchas / comments:**
- Reconstructs per-layer k/v with RoPE in PyTorch as the reference (re-implements what our `decoder_prefill.onnx` should do). Uses `rotate_half(x) = cat([-x[..., h:], x[..., :h]])` pattern consistently.
- Comment on dtype: "Our ONNX is fp32 on CPU — force the policy to match or the input-dtype checks inside torch blow up with 'Input FloatTensor vs weight BFloat16'." → explicit `policy.to(dtype=torch.float32); policy.to("cpu")`.
- Vision normalization: `img_f = img_f * 2.0 - 1.0` — SigLIP expects `[-1, 1]`. Note: only 1 camera probed (same image for all 3 in modal_pytorch_vs_onnx).
- Pos ID derivation: `pos_ids = mask.long().cumsum(-1) - 1; pos_ids = pos_ids.clamp(min=0)`.
- Stage 5 has a caveat: "For now skip this [full one-step velocity diff] — the decoder diffs already show per-layer kv is fine, so the divergence must be in expert_stack.onnx itself." — partial.
**Status:** current (companion to modal_pytorch_vs_onnx.py; iterates on task #25 "Per-layer vlm_kv ONNX export").
**Related:** modal_pytorch_vs_onnx.py, modal_libero10.py.

---

# Summary

## Working/PASS-clear scripts (clear numerical result, integration PASS, or passes all smoke asserts)

Scripts with well-defined numerical/PASS results:

1. **modal_test_export.py** — RMSNorm + mini-transformer ONNX export validated (max_diff readings).
2. **modal_real_export.py** — SuffixEncoder + ActionProj export: valid ONNX + numerical match + denoise bench.
3. **modal_expert_export.py** — single expert layer export + valid ONNX + max_diff log.
4. **modal_full_pipeline.py** — 16-layer expert stack export + full 10-step denoise bench (mean/p50/p95/Hz).
5. **modal_e2e_pipeline.py** — E2E VLM + expert forward + bench, structural (not numerical).
6. **modal_cli_export.py** — CLI wedge smoke (5-step PASS/FAIL table).
7. **modal_sim_test.py** — 10 sim episodes + adaptive-vs-fixed comparison (speedup ratio).
8. **modal_e2e_demo.py** — CLI+serve+bench, 6-step PASS/FAIL table.
9. **modal_test_pi0.py** — pi0 export 4-step PASS/FAIL.
10. **modal_test_pi05.py** — pi0.5 export 4-step PASS/FAIL.
11. **modal_test_gr00t.py** — GR00T export 4-step PASS/FAIL.
12. **modal_test_gr00t_full.py** — GR00T full-stack 4-step PASS/FAIL.
13. **modal_e2e_all_models.py** — 4-model benchmark table (Export/max_diff/Mean ms/Hz).
14. **modal_verify_cli.py** — 7-wedge CLI check (but references deprecated wedges; see "Superseded" below).
15. **modal_bench_onnx_vs_torch.py** — 4-model eager/compile/ORT bench.
16. **modal_bench_path_b.py** — 4-model + CUDA-graph bench, fixed ORT-CUDA.
17. **modal_verify_strict_providers.py** — 4 scenarios, verdict 4/4.
18. **modal_verify_wedge_compose.py** — 4 scenarios, verdict 4/4.
19. **modal_bench_trt_fp16.py** — 4-model TRT FP16 bench table.
20. **modal_verify_batching.py** — batching scenarios table.
21. **modal_verify_install_path.py** — 6-step quickstart PASS gate.
22. **modal_verify_bench_all.py** — 4-model `reflex bench` auto-TRT verify.
23. **modal_trajectory_replay.py** — 5-instruction semantic-diversity PASS.
24. **modal_probe_gr00t.py** — pure shape dump (can't fail; utility).

## Experimental / diagnostic (deliberately measuring whether a feature works; no clear "PASS" verdict — the number itself is the result)

25. **modal_verify_batching_real.py** — real-pi0 batching; result is the qps-vs-speedup table.
26. **modal_verify_adaptive_real.py** — disproves/validates adaptive-stop on real VLAs; result is the `trigger-rate | mean_step | savings | action_diff` table; outcome decides whether adaptive wedge ships or gets removed.
27. **modal_verify_trt_with_batch.py** — TRT + batching interaction diagnostic; result picks one of 3 documented resolutions.
28. **modal_pytorch_vs_onnx.py** — decisive cos_sim test to classify LIBERO-0%-cause into {export-correct / minor-drift / structural-bug}.
29. **modal_stage_diff.py** — per-stage L2 localization; partial (Stage 5 has a "skipped for now" note).
30. **modal_libero10.py** — benchmark runner; task_success % is the result; currently in-progress (task #23 in memory).

## Superseded / legacy

31. **modal_test_export.py** — superseded by modal_real_export.
32. **modal_full_export.py** — superseded by modal_vlm_export + modal_expert_export.
33. **modal_verify_cli.py** — references deprecated wedges (turbo/split/adapt), post-task-#14 this would fail.
34. **modal_sim_test.py** — the "synthetic 16-hidden toy" that modal_verify_adaptive_real is built to disprove/validate; kept for history.

## Counts

- **Clear PASS/working-result scripts (numerical or PASS-tabled):** 24
- **Experimental/diagnostic (result is a number, not PASS):** 6
- **Superseded (retained for history):** 4 (including overlaps with the first category — e.g., modal_test_export passes its assertions but is nonetheless superseded architecturally)

Total unique Modal scripts: 32 (24 clear-working + 6 diagnostic + 2 purely superseded that weren't re-counted).

## Cross-cutting patterns

- **Decomposed ops** (`DecomposedRMSNorm`, `DecomposedRoPE`) with `variance = x.to(float32).pow(2).mean(-1, keepdim=True)` pattern and RoPE cos/sin buffers — reused in every SmolVLA-style script.
- **Pipe-buffer deadlock lesson**: several scripts log "Use a real file for stdout — subprocess.PIPE deadlocks if the process logs more than one OS pipe buffer (~64KB) before we read." Canonical pattern: `log_fh = open(path, "wb"); Popen(..., stdout=log_fh, stderr=subprocess.STDOUT)`.
- **ORT-GPU version pinning post-mortem**: `torch==2.5.1 + onnxruntime-gpu==1.20.1 + numpy<2.0 + transformers>=4.40,<5.0` is the compatible stack once fake "onnxruntime-gpu bundles its own CUDA libs" assumption died.
- **TRT image base**: `nvcr.io/nvidia/tensorrt:24.10-py3` chosen because "cuDNN 9 (including libcudnn_adv) is already on the system path" — the pip `nvidia-cudnn-cu12` wheel is missing `libcudnn_adv.so.9`.
- **LIBERO install minefield**: `bddl==1.0.1`, `robosuite==1.4.1`, `gym` (NOT gymnasium), `osmesa` over EGL, patch_libero.py for interactive `input()` calls. Full LIBERO requirements.txt would downgrade the ONNX stack — install narrowly.
- **Shared noise for numerical diff**: when comparing two flow-matching pipelines, seed and share the initial noise, else cos_sim is dominated by noise drift.
- **Jetson fit table**: `fp16_gb × 2` (overhead) compared to {8, 32, 128} GB for Nano / Orin / Thor SKUs — repeated in bench scripts.
