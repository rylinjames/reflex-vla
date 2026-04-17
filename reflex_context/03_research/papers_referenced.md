# Papers Referenced

Every academic paper and arXiv identifier surfaced across the Reflex-VLA research corpus (sessions, commits, Modal logs, PM docs, `.agents/council/`, `vla_to_hardware_roadmap/`). For each: arXiv ID, title (if known), relevance, and how it has influenced Reflex.

Papers the repository already has a per-paper note for live in `reflex_context/02_research/papers/<arxiv-id>-<slug>.md`. This file is the index and cross-reference.

---

## pi-Flow — arXiv 2510.14974 (ICLR 2026)

- **Status in corpus:** chosen distillation recipe for `reflex distill --recipe pi_flow`.
- **Title:** (pi-Flow — exact title not quoted in corpus; paper identified by ID and "velocity-field matching" framing).
- **Why relevant:** velocity-field matching loss; teacher = frozen 10-step flow-matching model; student = 2-step distilled model. Paper claim: 10 → 2 denoising steps, **<5% task-success drop on LIBERO**.
- **Reflex influence:**
  - v0.2 flagship feature is `reflex distill --recipe pi_flow` (ADR: `2026-04-14-ship-distill-first.md`).
  - Rejected alternatives: Salimans+Ho 2022 progressive distillation, OneDP, Consistency Policy, Shortcut Models — all ranked below pi-Flow.
  - Pre-mortem correction: "no data needed, function-match on random noise" was wrong; action head is conditioned on VLM features, so real `(image, state, language)` triples from LeRobot/LIBERO/DROID are required. Compute budget revised from "~$10 Modal" to ~200–500 A10G-hours (~$200–$500).
  - Target step count revised from 2 to 4 after the pre-mortem (line 5756 of `current_session`).
  - Scaffold shipped: `src/reflex/distill/pi_flow.py` (commit `ed8157c 2026-04-16`).
  - Marketing copy rewrite: "pi-Flow distillation from 10 to 2 steps, <5% accuracy drop on LIBERO (per arXiv 2510.14974)" replaced the prior "5× faster" claim.

---

## RTC — Real-Time Chunking — arXiv 2506.07339

- **Status in corpus:** adopted into `reflex serve`; paper note in `02_research/papers/2506.07339-rtc.md`.
- **Why relevant:** defines the chunk-scheduling semantics that let a VLA server execute one chunk while the next is being inferred; handles network dropout by continuing current chunk and tolerating stale actions via forward-rolling (VLASH pattern).
- **Reflex influence:**
  - `reflex serve` carries `chunk_size_threshold=0.7` — request the next chunk when 70% of the current chunk has been consumed.
  - Lives in `lerobot.policies.rtc` (LeRobot v0.5, March 2026) — so the dependency was already tractable via `pip install lerobot`.
  - Framing borrowed for launch drafts: *"Open-loop while actions available, idle while waiting for next chunk"* (phrase steal from this paper / Hsu a16z).

---

## VLA-Eval — arXiv 2603.13966

- **Status in corpus:** reflex integrates it via `reflex.runtime.adapters.vla_eval`; paper note in `02_research/papers/2603.13966-vla-eval.md`.
- **Title / authors:** AllenAI VLA evaluation harness.
- **Why relevant:** the wire protocol + benchmark-suite contract for LIBERO / SimplerEnv / ManiSkill. WebSocket + msgpack, not HTTP.
- **Reflex influence:**
  - ADR `2026-04-14-wrap-not-rebuild-vla-eval.md`: adopted "wrap, don't rebuild" — built a thin WebSocket adapter rather than reimplementing the suite.
  - `src/reflex/runtime/adapters/vla_eval.py` implements `PredictModelServer`; `run_server` auto-injects kwargs from `__init__` signature — see `vla_eval_integration.md` in this folder.
  - `scripts/modal_libero10.py` uses `vla-eval` as the benchmark runner on Modal A10G.
  - Drove the discovery that LIBERO / vla-eval sends 1 camera while SmolVLA was trained on 3; `send_state` / `send_wrist_image` config knobs chosen after a diagnostic dump (line 10117 of current_session.md).

---

## DMPO — arXiv 2601.20701

- **Status in corpus:** competing distillation recipe; scaffolded as `reflex distill --recipe dmpo`.
- **Why relevant:** paper's claim — **one-step generation at 1770 Hz without a teacher**. This is the "no data needed" endpoint of the distillation spectrum. Supersedes pi-Flow for true one-step generation.
- **Reflex influence:**
  - GOALS.yaml `distill-dmpo` weight 9 — the Pro-tier flagship depending on whether we want fastest inference (DMPO, one-step) or best accuracy retention (pi-Flow, 2-4 step).
  - Scaffold shipped: `src/reflex/distill/dmpo.py` (commit `ed8157c`). Training loop deferred to v0.2.1.
  - Strategic tension: robotics-engineer critic called it "researcher catnip"; solo-founder critic called it the biggest time-waster; VC critic says ship distill early (weight 9). Resolution: distill is the right eventual paid wedge, wrong *now* before we have users.

---

## SmolVLA — arXiv 2506.01844

- **Status in corpus:** primary shipping model for Reflex; paper ID referenced in 7-parallel-agent research batch and in `vla_to_hardware_roadmap`.
- **Why relevant:** the 450M-param VLA Reflex ships export, serve, validate support for. Architecture: VLM backbone = SmolVLM2-500M (350.2M truncated to 16 layers) + action expert (98.2M) + projections (1.6M). Flow-matching denoise, 10 steps.
- **Reflex influence:**
  - `src/reflex/exporters/smolvla_exporter.py` is the canonical exporter pattern that pi0 and pi0.5 exporters imitate.
  - Expert geometry verified from checkpoint inspection: `expert_hidden=720`, `action_dim=32`, `num_layers=16`, `15 Q heads / 5 KV heads GQA`, `head_dim=64`, `intermediate=2048`, cross-attn on odd indices.
  - Published on HuggingFace at `lerobot/smolvla_base` (foundation) and `lerobot/smolvla_libero` (LIBERO fine-tune). 907MB single safetensors file — "the most export-friendly VLA available" per `ced2c4f1` session framing.
  - GQA+RoPE ONNX spike (commit `6fedff3 2026-04-16`): **SmolLM2's 15Q/5KV GQA decoder exports to ONNX cleanly first try** — no patches, no custom ops, opset 19 sufficient. Full VLM has 32 layers; SmolVLA truncates to 16.

---

## StarVLA — arXiv 2604.05014

- **Status in corpus:** paper note exists at `02_research/papers/2604.05014-starvla.md`.
- **Why relevant:** newer-generation VLA architecture. Positioned as "on the list but not yet supported" next to xVLA.
- **Reflex influence:** currently tracked for scope planning. `xvla-exporter` has GOALS weight 7 as a parallel "new model family" line item; StarVLA is one of the candidates behind whether the next family integration is xVLA or StarVLA.

---

## Dexmal Real-Time VLA — arXiv 2510.26742

- **Status in corpus:** paper note at `02_research/papers/2510.26742-dexmal-realtime-vla.md`; quoted in launch drafts.
- **Why relevant:** coined the "3–5 FPS against a need for 20–30 Hz" gap for VLAs on edge. Supports the "prefix KV-cache" deployment pattern now used in `reflex serve`.
- **Reflex influence:**
  - Quote "3-5 FPS against a need for 20-30 Hz" appears in launch drafts and in the Post-Mortem item 3 of `2026-04-16-post-mortem-reflex-validate.md`:  "VLM prefix encoder + KV-cache export (critical) — per Dexmal realtime-vla (arXiv 2510.26742). Estimate 2 weeks."
  - Direct architectural influence on `src/reflex/runtime/vlm_orchestrator.py` — the VLM prefill / expert-denoise KV-cache separation follows Dexmal's pattern.

---

## VLAgents — arXiv 2601.11250

- **Status in corpus:** paper note at `02_research/papers/2601.11250-vlagents.md`.
- **Why relevant:** newer-generation framework / survey on VLAs. Competitor track in `02_research/competitors/vlagents.md`.
- **Reflex influence:** mapped in competitor-landscape; referenced when reflex's "deploy any VLA to edge" wedge was being differentiated from frameworks that want to own training + deployment end-to-end.

---

## Characterizing VLA Models — arXiv 2603.02271

- **Why relevant:** **"confirms action generation = 75% of latency"** (sessions_md line 180). Quantitative evidence for the action-head-first optimization strategy.
- **Reflex influence:**
  - Justifies why Reflex export focuses on the action expert stack (phrase 75% is the engineering budget).
  - The adaptive-denoising wedge (`src/reflex/kernels/turbo.py`) rests on "action generation dominates latency, so early-stopping the denoise loop dominates the win."

---

## VLA-Perf — arXiv 2602.18397

- **Why relevant:** claimed *"first VLA inference benchmark"* framing; also coined **"memory-bound on Thor"** which is quoted in launch drafts.
- **Reflex influence:**
  - Flank in positioning: we deliberately scope narrower (4 VLA families × 4 hardware targets) rather than going head-to-head with their benchmark-authority framing.
  - Phrase "memory-bound on Thor" is on the "phrases to steal" list (line 5791 of current_session.md).

---

## DexGrasp-Zero — arXiv 2603.16806

- **Why relevant:** morphology-aligned graph for hands. Research adjacency; not in shipped path.
- **Reflex influence:** cited as prior art for the eventual cross-embodiment / hand-morphology story. Not influencing code today.

---

## Embodiment Scaling Laws — arXiv 2505.05753

- **Why relevant:** argues "scale beats adapters" for multi-embodiment VLAs. Directly informs the `reflex adapt` deprecation decision.
- **Reflex influence:**
  - Background justification for deprecating `reflex adapt` (task #14 complete): if scale beats adapters, an adapter wedge is low-ROI.
  - Cross-embodiment pain still exists as a customer problem — but the answer is fine-tune the VLA on more data, not provide an adapter CLI.

---

## Xiaomi Robotics-0

- **Why relevant:** 4.7B open-source VLA with **async execution decoupled**. Competitor signal — confirms others have shipped async runtime.
- **Reflex influence:** benchmarked against our async-chunking claim (RTC) when council reviewed the 7-wedge composition.

---

## RoboECC

- **Why relevant:** independent confirmation of the action-head bottleneck. No arXiv ID captured in corpus.
- **Reflex influence:** secondary citation supporting the "75% latency is action generation" framing.

---

## OneDP

- **Why relevant:** one-step diffusion policy. Distillation-adjacent prior art.
- **Reflex influence:** considered and **rejected** in favor of pi-Flow for v0.2 distill recipe (noted in council batch). Revisit if pi-Flow compute cost balloons.

---

## Consistency Policy

- **Why relevant:** Heun-style consistency distillation applied to diffusion policies.
- **Reflex influence:** rejected in favor of pi-Flow. Noted in the distillation-alternatives ranking.

---

## Shortcut Models

- **Why relevant:** one-step generative models via shortcut sampling; distillation alternative.
- **Reflex influence:** rejected in favor of pi-Flow. Noted in same council ranking.

---

## Salimans + Ho 2022 — Progressive Distillation

- **Why relevant:** original progressive-distillation recipe for diffusion models.
- **Reflex influence:** briefly considered as the pi-Flow alternative; rejected because pi-Flow's velocity-field matching loss is simpler and LIBERO-validated.

---

## PaliGemma2 ONNX prior-art

- **Why relevant:** `onnx-community/paligemma2-3b-pt-224` ships a 3-file split (vision_encoder + embed_tokens + decoder_model_merged). **Already a solved recipe.**
- **Reflex influence:**
  - Template for our 4-file split (vision_encoder / text_embedder / decoder_prefill / expert_stack) in `src/reflex/exporters/vlm_prefix_exporter.py`.
  - Correction from line 7252 of current_session.md: our earlier plan said 3 files (vision + embed_tokens + decoder); reality needs 4 because SmolVLA has a special prefill step producing KV cache consumed by the expert.

---

## ETARS — SmolVLA ONNX export notebook

- **Source:** `aifoundry-org/ETARS` public repo, `smolVLA_libero_export.ipynb`.
- **Why relevant:** an independent working SmolVLA → ONNX export. Reflex directly studied it to unblock the per-layer vlm_kv export.
- **Reflex influence (line 7252):**
  1. Wrong plan-number of ONNX files — ETARS shows 4 files, not 3.
  2. Wrong model loading — use `SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')` (16 truncated layers) not `AutoModel.from_pretrained('SmolVLM2-500M')` (full 32 layers).
  3. Vision export was over-engineered — `model.embed_image()` already does vision + pixel_shuffle + connector in one call; just wrap that.
  4. Export gotchas surfaced: `do_constant_folding=False` (folding corrupts the graph), `patch_gather_indices_once()` post-export (ORT rejects float Gather indices).
  5. State token: `nn.Linear(32, 960)` → exactly 1 token.
  6. Correct vlm_kv_dim = 960 (not 512 from our stub).

---

## LiteVLA-Edge

- **Why relevant:** already ships GGUF quantization on Jetson at 6.6 Hz in a ROS2 pipeline. **Direct competitor** on the edge-deployment wedge.
- **Reflex influence:** heightens urgency of shipping LIBERO numbers and Jetson validation. Referenced in Batch-1 research findings aggregation.

---

## veRL log-prob divergence study

- **Why relevant:** **benchmarked upstream of Reflex**; concrete data on BF16 (training) vs FP16 (serving) divergence on Qwen2.5-7B on A100. 10/10 requests divergent, 5 critical, 5 drift, 419/1209 tokens (35%) above 0.01 threshold, max delta 2.05, P95 1.17.
- **Reflex influence:** not a VLA paper, but the foundational data point for "training precision != serving precision causes silent semantic drift" — motivates `reflex validate` round-trip parity at 1e-4 (the CHANGELOG BREAKING change from 0.02 → 1e-4).

---

## Papers referenced but under-documented in corpus (gap items)

- **SimplerEnv** — named as a benchmark target (`reflex bench --benchmark simpler_env`) and `src/reflex/eval/simpler.py` stub exists, but **no paper citation or details** captured anywhere in the raw corpus. Gap.
- **ManiSkill** — same: named as a benchmark target (`src/reflex/eval/maniskill.py`), but no citation or architectural notes.
- **AERO** — only appears as "AERO: 48% compute reduction (for comparison)" (sessions_md line 82) — no arXiv ID, no title, no reference path. Gap.
- **Token-efficient benchmark** ("utilization 42.6% → 69.6%") — referenced without a paper ID. Unclear whether it's earlier Reflex work or external.
- **`StarVLA`** has a paper note file but no synthesis of the paper's technical claims in raw sources — the file may be a stub.

---

## Cross-cutting research threads

Three recurring threads span these papers:

1. **Denoising efficiency (action-head ~75% of latency)** — drives pi-Flow, DMPO, OneDP, Consistency Policy, Shortcut Models, adaptive denoising. The flagship research story.
2. **Edge deployment gap (3–5 FPS vs 20–30 Hz needed)** — Dexmal, VLA-Perf, LiteVLA-Edge, Characterizing-VLA. The positioning story.
3. **ONNX / export mechanics** — PaliGemma2 prior art, ETARS, the decomposition lessons. The engineering story, not on arXiv but living in GitHub issues (LeRobot #819, #1923, #3146; OpenPI #386, #826).
