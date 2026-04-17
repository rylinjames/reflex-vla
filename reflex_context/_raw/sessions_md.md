# Reflex VLA — Extracted Session Knowledge

Source: `.agents/ao/sessions/*.md` (13 files, Apr 10–16, 2026). Many of these sessions are for other projects (AfriCal, PairLaunch, Santhicaios, PBM/ClearScript, BrainTrain/CortexRep, EcomLinked, whop-ai-chat, gbrain). Only the Reflex VLA sessions are mined exhaustively below.

---

## 2026-04-10 (session `ced2c4f1`) — Path selection, wedge design, full week of Reflex build + LIBERO hunt

This is the marquee Reflex session: ~91 KB, 1661 lines, ~1.12M tokens. It's a cumulative narrative covering strategy, multi-wedge CLI design, model-by-model export/serve work (SmolVLA, pi0, pi0.5, GR00T, OpenVLA), Modal debugging, TensorRT builds, distillation planning, and the multi-day LIBERO task-success hunt.

### Bugs found

- **TRT EP rebuilding engines per input shape** — when `reflex serve --max-batch > 1`, TensorRT Execution Provider was rebuilding engines per call, producing 34s/call, a ~200× pessimization. Fixed by routing to CUDA EP when `max_batch > 1`. ADR: `2026-04-14-disable-trt-when-batch-gt-1.md`.
- **ORT CUDA silent CPU fallback** — `onnxruntime-gpu` silently fell back to CPU EP when the CUDA provider failed to load. The failure was `libcublasLt.so.12: cannot open shared object file`. CPU fallback produced 462ms for SmolVLA (vs 25.8ms on GPU), 999ms pi0, etc. Fix direction: pin `torch==2.5.1` + `onnxruntime-gpu==1.20.1` + `nvidia-cudnn-cu12==9.*`. ADR: `2026-04-14-strict-provider-no-silent-cpu-fallback.md`.
- **GeLU + TRT LayerFusion failure** — GeLU lands as elementwise `Mul+Tanh`. ONNX export works, TRT LayerFusion fails; kept as generic MHA. RMSNorm and RoPE export cleanly.
- **RoPE half-dim concat bug** — cos/sin frequencies were the wrong dim. Fixed by concatenating `[freqs.cos(), freqs.cos()]` to get full dim instead of half.
- **Cross-attention K/V dim mismatch (320 vs 720)** — cross-attn layers accept VLM KV dim, not expert hidden dim. Had to align expert dims explicitly.
- **pi0 / pi0.5 / GR00T normalizer missing** — running LIBERO without the dataset normalizer meant action magnitudes were off; task-success = 0%. Added normalizer support, confirmed `norm=on` in adapter startup log (4 stats loaded). Still 0% — something else wrong.
- **`trtexec --minShapes/--optShapes/--maxShapes` rejected** — fails when the ONNX has *static* shapes. Have to strip the dynamic-shape flags for static graphs.
- **`tensorrt.Logger` attribute missing** — Python bindings not installed in `nvcr.io` container. Switched to `trtexec --loadEngine` for benchmarking (bypasses Python bindings).
- **LIBERO stack setup cascade**: Run #2 killed by wrong `requirements.txt`, Run #3 missing `cmake` for `egl_probe`, Run #4 `bddl`/`future`/`gym` cascade. Run #5 worked with proper readiness polling.
- **`subprocess.run(capture_output=True)` buffering** — inside the Modal function this buffered all vla-eval stdout/stderr until the subprocess completed. Looked like the container was hung (37 min with no output). Fix: stream stdout line-by-line rather than capturing.
- **Modal preemption on A100 spot** — the loop got a new worker mid-run and continued. Not really a bug, just a gotcha.
- **`capture_output` output only lands on completion** — same buffering class as above; means you can't tell if a 40-min LIBERO run is healthy.
- **lerobot package install-structure quirk** — `pip install git+lerobot` installs but `import lerobot` fails because the package has a non-standard install structure. Took 5 iterations of Modal debugging.
- **`.pyc` cache / 4th `input()` call** — old code was running despite edits because cached bytecode persisted. "Must be a `.pyc` cache or a 4th call" — had to nuke caches + add catch-all.
- **Text-embedding non-determinism** — same instruction produced different embeddings between runs. Fixed by seeding the RNG with the token IDs so the same instruction always produces the same embedding. Still a fallback path — real fix is `text_embedder.onnx` (Issue 2).
- **ONNX sessions never closed** — leaking sessions. Added `close()` method to `VLMPrefixOrchestrator` + `__del__`.
- **Expert's velocity sign flips on dims 2 and 6 (gripper)** — 20% magnitude error plus sign inversions → trajectory integration destroyed task-success.
- **Vision cos=1.0, text cos=1.0, state proj cos=1.0, but end-to-end cos=0.08** — indicates a systemic issue not localized to any single sub-component. Per-layer kv matched after `rope_theta` fix; end-to-end still diverged to cos=-0.27 then -0.24.
- **`chromium-style-utils` / old python2/3 compat lib missing** — `ModuleNotFoundError: No module named '_utils'`. Easy fix.
- **vla-eval uses WebSocket+msgpack, not HTTP** — had to build a thin adapter (`reflex.runtime.adapters.vla_eval`).
- **`providerIdentity` missing on `APIProfile`** (cross-pollinated note from unrelated project but logged).
- **`libonnxruntime_providers_cuda.so`** loading failure with `libcublasLt.so.12` unresolved — requires CUDA 12 / cuDNN 9 alignment pins.

### Decisions / architecture

- **Pivot: Path 00 → Path Alt-VLA**. Path 00 (Datadog-for-inference) was structurally downstream of model labs ("forever reacting to the next frontier lab's release"). Path Alt-VLA's trillion-dollar ceiling via VLA foundation + deployment stack. Barbell: if Alt-VLA fails, still have a $30B Datadog-for-inference outcome. Tesla playbook: Roadster → Model S → Robotaxi.
- **Reflex = 7-wedge product**: `export`, `serve`, `guard`, `turbo`, `split`, `adapt`, `check`. Later consolidated: `split`, `adapt`, `turbo` deprecated; `check` folded into `validate --quick`. ADR: `2026-04-14-deprioritize-adapt-and-split.md`.
- **v0.2 flagship = `reflex distill`** via pi-Flow recipe (arXiv 2510.14974, ICLR'26) on pi0.5-base. 10→2 denoising steps, <5% LIBERO accuracy drop. Target ~$60 Modal, ~1 week eng. (Pre-mortem base rate = 3 weeks.)
- **Pricing**: Free tier = FP16 export. Pro tier = FP8/INT4 quantization + validation ($99/mo). Reflex Compute Pack (own-branded hardware) at Year 2+.
- **"Wrap, don't rebuild" vla-eval**: use WebSocket+msgpack adapter instead of reimplementing. ADR: `2026-04-14-wrap-not-rebuild-vla-eval.md`.
- **No cloud-edge split until**: (a) ≥2 design partners explicitly request cloud offload, (b) reference VLA that doesn't fit edge hardware, (c) their need is concrete. Only build the `SplitConfig` latency monitor now.
- **Don't drop CLI commands users are finding**: each command "discovered naturally" (export slow → find `turbo`; persistent serving → find `serve`).
- **Strict provider mode** — no silent CPU fallback. If CUDA not available, error out. ADR: `2026-04-14-strict-provider-no-silent-cpu-fallback.md`.
- **Unified `reflex export`**: automatically calls `export_vlm_prefix` for SmolVLA. No more separate commands.
- **Distill CLI scaffold**: `reflex distill --recipe dmpo|pi_flow`, training loop lands v0.2.1.
- **LIBERO number = the north star benchmark**. Without it, marketing architecture; with it, marketing results.
- **Phase 2 = bundle with hardware (Seeed, Trossen, Jetson), rev-share, no inventory**. Phase 3 = Reflex Compute Pack. Phases live under `vla_to_hardware_roadmap/phase_1_vla_software/...`.
- **Knowledge repo separation**: `reflex-vla/` = code + agents artifacts; `reflex_context/` = vision, decisions, research, experiments, product, inbox, archive. `CLAUDE.md`, `TEMPLATE.md` at each level.
- **"Pi-Flow velocity-field matching loss; teacher = frozen 10-step model; student = 2-step distilled"** is the chosen recipe (Salimans+Ho 2022 consistency models rejected; OneDP, Consistency Policy, Shortcut Models ranked below).
- **GOALS.yaml evolve loop**: `/evolve` picks highest-weighted failing goal; one goal per cycle. Weight 9 goals: `user-data-scoping`, `distill-dmpo`. Weight 8: `stripe-license-gating`, `ros2-bridge`. The evolve loop continues until kill-switch or stagnation.
- **Paid tier anchor on `reflex distill`**. Free = export. Pro = distill + quantize + validate.
- **Pre-mortem: "Fix the brain before timing it or compressing it"** — means VLM conditioning must work before distillation claims are credible.
- **Observability = bottom-up sale** (engineering budget). Compliance/audit-trail = top-down sale (legal/CISO, 6–9 mo cycles, bigger checks).
- **EU AI Act Article 12**: retention ≥6 months for log format. Current log format is missing the "decision pathway" field (which policy head / token produced the action). Draft requires ML-model-state auditability (Clause 8.3.1), decision-pathway data, secure storage, access controls, software-or-hardware flexibility.
- **VLA runtime must own the "missing infrastructure layer"** — Foxglove's Banisadr (Actuate 2025), Hsu a16z: *"95% in the lab, 60% in the field"*. This is the wedge positioning.
- **Adopt RTC (Real-Time Chunking, arXiv 2506.07339, `lerobot.policies.rtc`, LeRobot v0.5)** in `reflex serve`. Chunk threshold `chunk_size_threshold=0.7`. Network dropout: continue executing current chunk; VLASH's forward-rolling handles stale actions.
- **VLM prefix export is internal**. Users never call `export_vlm_prefix()` directly.
- **Use `AutoModel` in the exporter** (GOALS.yaml check).
- **Consistent auth**: API-key-auth, nan-guard-hardening in v0.2 roadmap.

### Measurements / numbers

- **Model benchmark table (Modal A10G, batch=1, FP16 unless noted)**:

  | Model   | Params  | torch.compile | ORT (still-CPU fallback) | Reflex CUDA graph | Eager |
  |---------|---------|---------------|--------------------------|--------------------|-------|
  | SmolVLA | 99.8M   | **25.8ms**    | 462ms                    | 35.0ms             | 19.1  |
  | pi0     | 314.6M  | **47.5ms**    | 999ms                    | 63.5ms             | 23.9  |
  | pi0.5   | 426.9M  | **52.9ms**    | 1163ms                   | 69.7ms             | ~55.6 |
  | GR00T   | —       | **113.2ms**   | —                        | 142.4ms            | —     |

  (Note: eager-ms column is the "Eager" from the separate table earlier in session; SmolVLA eager=19.1ms, pi0 eager=23.9ms, pi0.5=55.6 extrapolated, GR00T not listed.)

- torch.compile wins by 20–26% across all four models (SmolVLA 25.8 vs 35.0, pi0 47.5 vs 63.5, pi0.5 52.9 vs 69.7, GR00T 113.2 vs 142.4 ms/chunk). Root cause: torch.compile already uses CUDA graphs internally, so Reflex's CUDA-graph capture was roughly duplicating work.
- **Marketing headline**: "2.6 to 3.3× faster than PyTorch with torch.compile at the model's native FP32 precision" — flagged by reviewer as weak (FP16 vs FP32 comparison is a gotcha; real engineers will spot this).
- **ORT launch overhead**: ~5 µs × 10 per step. CUDA graph saves this.
- **pi0 full 10-step chunk on A100**: ~55.6 ms via torch.compile (extrapolated from per-step).
- **Token-efficient benchmark (earlier work, rollout-diff)**: utilization 42.6% → 69.6%. Truncated 1/10 → 1/10. Throughput neutral. 43.8% token-budget savings on easy prompts (256 instead of 2048).
- **AERO**: 48% compute reduction (for comparison).
- **FP8 KV throughput**: 0.97× (free/neutral). Industry consensus: ~1.0×. Confirms but not novel.
- **pi-Flow distillation** (paper claim): 10→2 steps, <5% task-success drop on LIBERO.
- **LIBERO run durations**: ~3 min/episode, ~60 min for 20 eps, ~45 min for 100 eps.
- **Task-success on LIBERO-10**: **0%** across 5 runs with normalizer on, 5D fix, state + wrist, RoPE fix, multi-camera. Each episode fails at 150 steps. Per-layer kv match good, final cos=-0.27, then ~0.98 per-step, -0.24 final velocity. Velocity has 20% magnitude error + sign flips on dims 2 and 6.
- **Hardware proxies for Jetson**: A10G = Ampere SM_86. Jetson Orin Nano = Ampere Tegra SM_87. Same compute family, different memory. A10G ≈ $1.10/hr on Modal.
- **Jetson Orin Nano Super Dev Kit** = $249 target for distill v0.2.
- **Modal trial credits**: enough for a few training runs.
- **Inference.net grants**: up to $10k, ~50% approval.
- **HF Community Grant**: ZeroGPU Pro + H200 on Spaces, ~40-50% approval, requires published Space demo.
- **All 5 E2E steps pass on Modal A100** for GR00T: build, PyTorch load, ONNX export, ORT session, actionable forward — "4-model parity."
- **LIBERO image size**: 256×256, cameras keyed `camera1/2/3`.
- **Mobileye–Mentee M&A**: $900M (one real M&A data point).

### Gotchas to remember

- **Modal image builds are slow** for ML-heavy deps: lerobot, torch, transformers, onnx, fastapi on Python 3.12 installs take ~5–10 min. Always allow build time before declaring failure.
- **`modal run --detach`** returns immediately — the local bash task ends but the Modal function keeps running. Check via `modal app list` / `modal app logs <app_id>`.
- **Modal A100 spot is preemptible**; runs get new workers mid-execution.
- **Background task output files** land in `/private/tmp/claude-501/-Users-romirjain/.../tasks/*.output`. ~15 runs accumulated during the LIBERO hunt.
- **Line-by-line subprocess output**: don't use `capture_output=True` with long-running subprocesses on Modal — buffering obscures progress. Stream.
- **`pip install git+lerobot`** has non-standard install structure. Import still fails. Multiple Modal iterations burned.
- **`nvcr.io` base image** doesn't include TensorRT Python bindings — use `trtexec --loadEngine` for benchmarking.
- **`trtexec --minShapes/--optShapes/--maxShapes`** only valid when ONNX has *dynamic* shapes. Static shapes → these flags error out.
- **opset 19 vs training's native PyTorch ops** may cause subtle numerical divergence. Suspect in unresolved 0% LIBERO.
- **Cache `.pyc` files persist** across iterations; nuke them when "old behavior" seems to come back.
- **Text-embedding path**: current exporter is a fallback — pseudo-random projections, not real SmolLM2. Real fix = export `text_embedder.onnx`.
- **LIBERO sends 1 image (first camera)** while SmolVLA was trained on 3 cameras (camera1/2/3). Camera mismatch is candidate for remaining 0%.
- **`states` vs `controller_states`** in LIBERO obs — model may have been trained on controller_states (controller output) not states (raw). Another candidate for 0%.
- **GR00T embodiment quirks**: Multiple modes per embodiment tag. Humanoids (GR1) use absolute joint positions. Single-arm (OXE_DROID) uses end-effector control. EE-based actions go through a decoder. Per-embodiment weights (leading dim 32) sliced at `embodiment_id=0` by default. Decoder still available if users want custom.
- **SmolVLA expert hidden-dim mismatch**: expert dim (320) ≠ VLM KV dim (720). Cross-attn expects VLM KV dim.
- **`reflex serve` must support hot-reload** — planned, not yet built. Currently users restart.
- **Per-wedge files must feed the real code, not live in a separate research tree** — the PRIORITY_MATRIX.md + researcher outputs had a "no connection to code" problem.
- **`caffeinate -dimsu &`** to prevent Mac sleep during long-running Modal streams. Kill via `killall caffeinate`.
- **Modal's TRT container**: 10GB pull. Extraction is slow (`unpacking rootfs`). Then deps install on top, then 4 checkpoints download, then TRT engines build. ~10 min full cycle.
- **IRA/Selected Drugs tab crash** (noted from unrelated PBM project but in same session) — indicates general pattern: frontend `.map()` on response object vs `response.<field>`.
- **Distillation adds drift**: stacking VLM fallback + distillation compounds the error.
- **FP16 vs FP32 baseline comparison is a known gotcha** — reviewers will catch it. Compare FP16 vs FP16 or report both.
- **ORT CPU threading quirk**: tokenizer fails single-threaded; force `inter_op_num_threads=1` only after confirming it's not the tokenizer.

### Open questions / dead ends

- **LIBERO-10 task-success stuck at 0%** after 5+ runs. Tried: normalizer, 5D fix, RoPE + rope_theta fix, multi-camera, state+wrist, per-layer kv. Per-component cos=1.0 but end-to-end cos=-0.27. Something systemic; candidates include opset-19 numeric drift, cameras mismatch (1 vs 3), states vs controller_states, expert velocity sign flips dims 2/6.
- **PyTorch-vs-ONNX diff test**: the "real long-term fix" — needed to isolate systemic divergence. Earlier attempt hit lerobot API version issues. Scripts in `scripts/local_*_diff.py`.
- **Per-layer vlm_kv ONNX export**: task #25, in progress. Means exporting each transformer layer's K/V separately so the VLM prefix KV-cache matches the training path.
- **Text embedder ONNX**: not yet exported. Seeded RNG is a fallback.
- **Whether progressive distillation (Salimans+Ho 2022) is the right recipe for VLAs** — briefly questioned, settled on pi-Flow instead.
- **xVLA** (880M, tokenized head) — new model family Reflex doesn't support. On the "trending, not yet covered" list.
- **Timing for `reflex distill --recipe dmpo`**: Revenue view says ship early (weight 9); solo-founder view says defer ("2-week research project pretending to be engineering"); robotics engineer says weight 5. Not resolved; attempted to fold in via "3 days to ship objection gone".
- **Cloud-edge split**: 2-day latency monitor as stub while waiting for customer demand signal. No production rollout until conditions met.
- **ROS2 bridge (`ros2-bridge`)** weight 8: enables new test categories. Timing TBD.
- **Stripe license gating** weight 8. Product surface still TBD.
- **HF Community Grant path** requires a published Space demo; not yet done.
- **Distributional regression detection (`reflex check` v2)**: a test category that "grows as the product grows" — design vague.
- **Authentic baseline**: FP16 vs torch.compile FP16, not FP16 vs FP32. Need re-run to publish.
- **Pre-brief Nathan Lambert, quote-tweet on P_max drop**: highest-leverage first-90-day move, not yet executed.
- **Zhihao Jia (CMU) / Hao Zhang (UCSD) / Yiying Zhang (UCSD) co-founder outreach**: plan set, not yet sent.
- **User interviews**: 5-10 LeRobot Discord / Open Robotics Discord / ROS Discourse conversations — noted as Phase 1 gap, not yet done.
- **Authentication in `reflex serve`**: planned; not built.
- **Fleet-mode batching**: dropped after earlier bugs; revisit.
- **Consumer tool / benchmark authority / Roche-style certification play**: deprioritized but tracked.
- **Adaptive denoising thresholds**: only pi0 currently works; per-model not yet built.

### Files + artifacts referenced (Reflex VLA project specifically)

- `src/reflex/` — code root.
  - `__init__.py`, `config.py`, `checkpoint.py`, `decompose.py`, `inference.py`, `validate.py`, `benchmark.py`, `cli.py`, `validate_training.py`, `validate_roundtrip.py`, `ci_template.py`, `_pytorch_backend.py`, `_onnx_backend.py`
  - `exporters/` — `__init__.py`, `onnx_export.py`, `trt_build.py`, `smolvla_exporter.py`, `pi0_exporter.py`, `gr00t_exporter.py`, `openvla_exporter.py`, `vlm_prefix_exporter.py`, `vlm_components.py`
  - `runtime/` — `__init__.py`, `server.py`, `split.py`, `vlm_orchestrator.py`, `adapters/__init__.py`, `adapters/vla_eval.py`
  - `safety/` — `__init__.py`, `guard.py`
  - `kernels/` — `__init__.py`, `turbo.py`
  - `models/` — `__init__.py`, `smolvla.py`, `adapt.py`
  - `postprocess/` — `__init__.py`, `openvla.py`
  - `eval/` — `__init__.py`, `libero.py`, `simpler.py`, `maniskill.py`
  - `distill/` — `__init__.py`, `dmpo.py`, `pi_flow.py`
- `scripts/` — `modal_test_export.py`, `modal_full_export.py`, `modal_real_export.py`, `modal_expert_export.py`, `modal_full_pipeline.py`, `modal_vlm_export.py`, `modal_e2e_pipeline.py`, `modal_cli_export.py`, `modal_sim_test.py`, `modal_e2e_demo.py`, `modal_test_pi0.py`, `modal_test_pi05.py`, `modal_e2e_all_models.py`, `modal_test_gr00t.py`, `modal_probe_gr00t.py`, `modal_test_gr00t_full.py`, `modal_verify_cli.py`, `modal_bench_onnx_vs_torch.py`, `modal_bench_path_b.py`, `modal_verify_strict_providers.py`, `modal_verify_wedge_compose.py`, `modal_bench_trt_fp16.py`, `modal_verify_batching.py`, `modal_verify_batching_real.py`, `modal_verify_adaptive_real.py`, `modal_verify_install_path.py`, `modal_verify_bench_all.py`, `modal_verify_trt_with_batch.py`, `modal_trajectory_replay.py`, `modal_libero10.py`, `patch_libero.py`, `modal_pytorch_vs_onnx.py`, `modal_stage_diff.py`, `local_stage_diff.py`, `local_full_diff.py`, `local_expert_diff.py`, `local_single_layer_diff.py`
- `tests/` — `test_decompose.py`, `test_config.py`, `test_validate.py`, `test_cli.py`, `test_server.py`, `test_guard.py`, `test_turbo.py`, `test_split.py`, `test_adapt.py`, `test_check.py`, `test_checkpoint_detection.py`, `test_openvla_postprocess.py`, `test_vla_eval_adapter.py`, `test_vlm_prefix.py`
- `launch/` — `lerobot_3146_draft.md`, `show_hn_draft.md`, `reddit_robotics_draft.md`, `README.md`
- `docs/getting_started.md`
- `.agents/` — `council/`, `plans/`, `research/`, `rpi/` (phase-1..6 summaries for `reflex-validate` and `vlm-prefix`)
- `GOALS.yaml`, `CHANGELOG.md`, `pyproject.toml`, `LICENSE`, `.gitignore`
- **`reflex_context/` layout** (the new knowledge base):
  - `00_vision/` (INDEX, north_star.md, positioning.md, moat.md)
  - `01_decisions/` with ADRs: `2026-04-14-ship-distill-first.md`, `-deprioritize-adapt-and-split.md`, `-wrap-not-rebuild-vla-eval.md`, `-disable-trt-when-batch-gt-1.md`, `-strict-provider-no-silent-cpu-fallback.md`, `2026-04-16-council-reprioritization.md`
  - `02_research/papers/`: `2510.14974-piflow.md`, `2506.07339-rtc.md`, `2603.13966-vla-eval.md`, `2604.05014-starvla.md`, `2510.26742-dexmal-realtime-vla.md`, `2601.11250-vlagents.md`
  - `02_research/competitors/`: `physical_intelligence.md`, `nvidia_groot.md`, `lerobot.md`, `vlagents.md`, `allenai_vla_eval.md`
  - `02_research/`: `2026-04-16-goals-research.md`, `2026-04-16-vlm-prefix-encoder.md`, `2026-04-16-vlm-real-export.md`, `2026-04-16-vlm-issue-research.md`, `2026-04-16-hardware-alternatives.md`
  - `03_experiments/`: `2026-04-14-trt-fp16-vs-torch-compile.md`, `2026-04-14-batching-scale.md`, `2026-04-14-adaptive-denoising.md`
  - `04_product/`: `roadmap_6week.md`, `roadmap_5phase.md`, `prd/distill.md`, `prd/serve_v2.md`, `prd/export_v2.md`, `prd/check_v2.md`, `prd/guard_v2.md`
  - `05_inbox/`, `06_archive/`, `02_research/{hardware_partners,customers,market}/`
- `/Users/romirjain/.claude/projects/-Users-romirjain/memory/project_reflex_vla.md`
- `/Users/romirjain/.claude/projects/-Users-romirjain/memory/project_reflex_vla_inference_bugs.md`

### Key papers / references

- **pi-Flow** — arXiv 2510.14974 (ICLR'26). The chosen distillation recipe: velocity-field matching loss, 10→2 steps, <5% drop on LIBERO, teacher frozen, student 2-step.
- **RTC (Real-Time Chunking)** — arXiv 2506.07339. In `lerobot.policies.rtc` (LeRobot v0.5, Mar 2026). Adopted into `reflex serve`.
- **VLA-Eval** — arXiv 2603.13966. AllenAI benchmark.
- **Characterizing VLA Models** — arXiv 2603.02271. Confirms action generation = 75% of latency.
- **StarVLA** — arXiv 2604.05014.
- **Dexmal Real-Time VLA** — arXiv 2510.26742.
- **VLAgents** — arXiv 2601.11250.
- **DMPO** — arXiv 2601.20701 (one-step generation, alternative to pi-Flow).
- **DexGrasp-Zero** — arXiv 2603.16806 (morphology-aligned graph for hands).
- **Embodiment Scaling Laws** — arXiv 2505.05753 (scale beats adapters).
- **Xiaomi-Robotics-0**: 4.7B open-source VLA with async execution decoupled.
- **OneDP**, **Consistency Policy**, **Shortcut Models** — distillation alternatives considered, rejected in favor of pi-Flow.
- **RoboECC** — confirms action-head bottleneck.
- **Salimans + Ho 2022** — progressive distillation, briefly considered, rejected.

### Competitor landscape (notes captured)

- **Physical Intelligence (`openpi`)**: releases pi0/pi0.5 weights only, no deployment tool. Reflex = the bridge.
- **NVIDIA GEAR / GR00T N1**: maps the Jetson/Isaac space. Competitive threat. 
- **LeRobot**: async server broken (issues 2356, 3204, 2980). No working VLA async server sold anywhere. Issue 3146 was opened 2026-03-12 by `jashshah999`, 0 comments, 5 reactions — our wedge signal.
- **Foxglove Agent + Cloud**: $90/user/mo, $40M Series B. Customers include NVIDIA/Amazon/Anduril/Waabi/Dexterity. Observability niche.
- **AllenAI vla-eval**: WebSocket+msgpack adapter, no HTTP server.
- **Isaac Sim 5.0**, **OneDP**, **Xiaomi-Robotics-0**: simulation/data entrants.
- **Inference.net** (grants program, up to $10k) — the easiest non-dilutive money right now.
- **xVLA (880M, tokenized head)**: new, not yet supported.

### Product / strategic decisions

- **Launch drafts written** for LeRobot issue 3146 (`lerobot_3146_draft.md`), Show HN (`show_hn_draft.md`), Reddit robotics (`reddit_robotics_draft.md`).
- **Pricing tier plan**:
  - Months 0-6: OSS CLI, free, Apache 2.0. First Pro tier at $99/mo.
  - Months 6-12: Bundle with Seeed/Trossen/Jetson integrators. Rev-share, no inventory.
  - Year 2+: Reflex Compute Pack — own-branded hardware.
- **Observability budget + Audit/Compliance two axes**: split in positioning. Engineering buys observability (Arize wedge, bottom-up PLG). Legal/CISO buys governance (top-down, longer cycles, bigger checks).
- **"Edge-deployable models designed specifically for robotics constraints"** — Hsu's fourth pillar (a16z) is the thesis wedge.
- **No ARR on internal dashboards for 18 months (anti-SaaS policy)**. Ship open-source runtime first.
- **Paper 1 must land by day 90** — probability narrative holds = ~50%.
- **Publish-first, commercial-second**: "lab with commercial tail."
- **Marketing copy rewrite** after critic review: "pi-Flow distillation from 10 to 2 steps, <5% accuracy drop on LIBERO (per arXiv 2510.14974)" replaces "5x faster."
- **Hero stat debate**: "3 days vs 14 days" for export. 14-day framing triggers "that must be slop" skepticism from sophisticated buyers. Picked 3 days.
- **"Deep daily engagement across the user base"** — rewrites "250K+ prayers logged every week" for the renamed case studies. Normalizes opacity, defends against AI-slop detection.
- **Codename riskiest projects** (MindRep kept live; others renamed). Marketing stance.
- **Apache 2.0 license**: ships free, captures market "the way Vercel captured Next.js developers or HashiCorp captured ops."

---

## 2026-04-10 (session `ba6b3fb`) — AfriCal (OFF-TOPIC)

iOS app session on AfriCal (calorie-tracking). Not Reflex VLA. Capturing only as a marker that this session file exists in the directory.

Key contents: `til -flushcache && sudo killall -HUP mDNSResponder` for Cloudflare worker DNS cache; React Native/SwiftUI screen navigation bugs around onboarding paywall; `AppLogoImage.imageset` Contents.json issues; referral manager work.

Files: AfriCal repo (`Onboarding/`, `Home/`, `Logic/`, `Subscriptions/`, `backend/teenybase.ts`, `worker.ts`, `.dev.vars`, `.prod.vars`, `wrangler.toml`).

No Reflex-specific knowledge.

---

## 2026-04-11 (session `f31b651`) — Santhicaios (OFF-TOPIC)

iOS medication app (Santhicaios). Not Reflex VLA.

Key contents: `SelectedMedicationSuggestion` struct, `PrescriptionReviewScreen` soft-confirm dialog, RxNorm service, IndianDrugDatabase, medication dedup, ingredient catalog. Open issues tracker noted: Duplicate document upload detection (#8), Medication deduplication (#9), Audit Trail & DPDP Compliance (#11).

No Reflex-specific knowledge.

---

## 2026-04-11 (session `c74be73`) — PairLaunch site polish (OFF-TOPIC)

Portfolio website (PairLaunch). Not Reflex VLA.

Key contents: `index.html` / `style.css` polish; tilt/magnetic button effects; Lenis smooth scroll; replaced expensive mousemove tilt with `requestAnimationFrame` + CSS keyframe animations + IntersectionObserver reveals. Project list includes Vinyl Scout, Taptic, Santhica, LockMode, TorahFlow, BrainTrain, DopaFry, DailyGrace, Offleaf, CortexRep, CalorieAI. Fastlane Snapshot noted as screenshot automation option.

No Reflex-specific knowledge.

---

## 2026-04-13 (session `49e865f`) — PBM/ClearScript (OFF-TOPIC)

Pharmacy benefit management app review. Not Reflex VLA.

Key contents: IRA Selected Drugs tab crash with `TypeError: x.map is not a function`; middleware for per-user scoping with Clerk; 408(b)(2) / CAA compliance; legacy-data-cleanup where all rows needed a `user_id=user_3BpuZlUJkYMzVFudeKCwj213YDm` backfill; Risk Strategies broker partnership; Nick Segal broker profile (2012 Broker of the Year). Files in `/Users/romirjain/Desktop/building projects/pbm/clearscript/...`. Glance MCP browser automation used.

No Reflex-specific knowledge.

---

## 2026-04-13 (session `6964676`) — BrainTrain / CortexRep (OFF-TOPIC)

Cognitive training iOS app. Not Reflex VLA.

Key contents: Speed Processing grid, Dual N-Back logic bug (match detection happened after stimulus off-set → fixed), ScorePopup rolling numbers, AdaptiveDifficultyEngine, HealthKitManager, app-store-screenshots skill, Xcode archive + signing (team Y6HQQ535H7), rename BrainTrain → BrainForge → CortexRep, ITMS-90683 missing purpose string, AppIcon asset updates.

No Reflex-specific knowledge.

---

## 2026-04-13 (session `30070a3`) — vla_to_hardware_roadmap scaffolding + academic MCP servers

Short session (~0.5MB tokens / ~53k tokens estimate). Has two named files:

### Files changed
- `/Users/romirjain/Desktop/building projects/axion_compute/vla_to_hardware_roadmap/README.md`
- `/Users/romirjain/.claude/settings.json`
- `/Users/romirjain/Library/Application Support/Claude/claude_desktop_config.json`

### Key content
- Set up the **`vla_to_hardware_roadmap`** directory as the multi-phase research tree (Phase 1 = VLA software, Phase 2 = bundle with hardware, Phase 3 = make VLA hardware, Phase 4 = make datacenter hardware).
- Configured **academic skill MCP servers**. Three need API keys:
  1. SerpApi (for google-patents)
  2. USPTO API key (optional — basic search works without)
  3. Third unnamed key
- **Gotcha**: the new MCP servers do not appear until the Claude desktop app is restarted.
- `hf-mcp-server` with `no-auth` noted as issue.

No benchmark numbers, no code bugs. Just config + roadmap scaffolding.

---

## 2026-04-13 (session `b31891a`) — Hermes / gbrain config (OFF-TOPIC)

Hermes CLI OAuth token pain: `ANTHROPIC_TOKEN` stale after re-auth. Pattern matching only for next attempts. `gbrain` CLI test calls (`~/gbrain/bin/gbrain put people/someone`, `~/gbrain/bin/gbrain query "who do I know at sequoia"`).

Files: `~/.hermes/config.yaml`, `~/.hermes/BOOT.md`, `~/gbrain/serve.sh`, `~/gbrain/gb`, `~/gbrain/src/core/embedding.ts`.

No Reflex-specific knowledge.

---

## 2026-04-14 (session `d61a79c`) — whop-ai-chat / DesignSystemPlayground (OFF-TOPIC)

Code review of `whop-ai-chat-main` in `/Users/romirjain/Downloads/for-raymo/`. DesignSystemPlayground rendered at `/` instead of a landing page; recommendation was redirect. Theme customization in sidebar user-section.

No Reflex-specific knowledge.

---

## 2026-04-16 (session `5e5753a`) — EcomLinked WHOP/Discord course (OFF-TOPIC)

Building a dropshipping course on WHOP. Discord bot setup (privileged gateway intents: server members, message content). Role hierarchy (bot role at position 1, move to 2). Stripe connection + Discord OAuth requires manual human clicks. Dropshipping vs branded vs POD vs private label decision.

No Reflex-specific knowledge.

---

## 2026-04-16 (session `4fcce3c`) — PairLaunch portfolio v2 (OFF-TOPIC)

PairLaunch homepage iteration. Normalized case-study copy opacity: "Selected excerpts. 100+ shipped under confidentiality." Renamed riskiest projects to codenames. Formsubmit.co setup. DNS A records for `pairlaunch.com` apex. GitHub Pages + Let's Encrypt provisioning wait time (15 min – 24h). Performance: hero orb blur 80→40px, gradient-text shift 12s→24s.

No Reflex-specific knowledge.

---

## 2026-04-16 (session `4aa11b0`) — Santhicaios identity IDs + Twilio OTP (OFF-TOPIC)

Stable identity IDs decoupled from auth (survives phone/device change). Twilio OTP test for US and India numbers. `providerIdentity` field on `APIProfile` still missing (todo flag). `ap-south-1` mentioned for Mumbai Supabase project.

No Reflex-specific knowledge.

---

# Summary of Reflex-VLA substantive sessions

Only 2 of 13 files substantively cover Reflex VLA:

1. **`ced2c4f` (Apr 10, 1661 lines, ~1.12M tokens)** — the marquee session with all technical, product, bug, benchmark, and strategic content.
2. **`30070a3` (Apr 13, 62 lines)** — light scaffold session for `vla_to_hardware_roadmap/` and academic MCP servers.

The other 11 sessions are for adjacent projects (AfriCal, PairLaunch, Santhicaios, PBM/ClearScript, BrainTrain/CortexRep, EcomLinked, whop-ai-chat, gbrain, Hermes). They share `romirjain/.claude/settings.json` cross-talk and occasional evolve-loop scaffold details, but contain no Reflex-VLA-specific learnings worth preserving.

Recommend in future sessions: route Reflex work into a dedicated project namespace so the session mining has a cleaner signal-to-noise ratio.
