# 2026-04-10 — The Mega Session

**Source:** Transcript `ced2c4f1-a341-45bf-ae1b-ba9f6ab0931c.jsonl` (55MB, 11,664 lines, 4,530 assistant messages, ~1.12M tokens).
**Duration:** Cumulative narrative spanning Apr 10 through Apr 17 — the original day-zero session that never quite ended. This file captures the opening Apr-10 decisions; the later days have their own entries.
**Arc:** From "should we even build this" to "what's the CLI look like" to "does SmolVLA even export" — a day of strategy collapsing into engineering.

---

## Goal

Pick a company-shape. Pick a product name. Pick the first wedge. Get SmolVLA from HuggingFace onto something that resembles a runtime. By end-of-day, have enough scaffolding that the claim "we are building a deployment layer for VLAs" is defensible with code, not slides.

The session opens with a pre-project strategic sweep inherited from InferScope / Axion — 31 candidate wedges, 7 company-shape paths. The user is three weeks into strategy-mode and wants to stop.

---

## The seven paths and why six of them lose

Before a line of code, the session worked through the seven company-shape candidates:

- **Path 01 — Datadog-for-Inference.** Observability spine. Pays bills, structurally downstream of model labs. Ceiling $30B.
- **Path 02 — Benchmark Authority.** Roche-style certification play. Downgrade to free asset, not primary company.
- **Path 03 — Reasoning-aware serving.** Demoted to SKU.
- **Path 05 — DiT/video serving.** 2027 second act, not 2026.
- **Path 06 — Physical AI Inference Stack.** The $1T shot if it hits. Stage via the "Stack H" narrow wedge.
- **Path 07 — NVFP4/Blackwell calibration.** Demoted to SKU.
- **Path 00 — Inference Research Lab.** The academic umbrella synthesis.
- **Path Alt-VLA** — month-18 pivot option into full VLA foundation-model lab.

The decision crystallized as a barbell: **Path 01 for cashflow, Path 06 narrowed to VLA-deployment as the moonshot, Path Alt-VLA reserved as the month-18 option if triggers hit.**

The core discomfort with Path 00 was articulated in the transcript: *"forever reacting to the next frontier lab's release."* That framing killed it. Observability is structurally downstream — the model labs set the cadence, we chase it. VLA deployment is upstream of every lab's go-to-market because deployment friction kills adoption even when the model is excellent. Tesla playbook: Roadster (SmolVLA export) → Model S (pi0/GR00T runtime) → Robotaxi (own silicon + own model).

---

## Naming: Reflex, not Forge, not Actuate

The product needed a name that works as a CLI verb prefix, a company name, and eventually a hardware brand.

- **Forge** — too generic. Every AI tool is a forge.
- **Actuate** — sounds like "actually" when spoken. Dies in phone calls.
- **Reflex** — one word, technical but not jargon, works as `reflex export`, `reflex serve`, `reflex guard`. Nobody in robotics AI owns it.

Settled: **Reflex**. The CLI grammar shipped with this name locks in every subsequent surface. Memory file: `/Users/romirjain/.claude/projects/-Users-romirjain/memory/project_reflex_vla.md`.

---

## The no-VLA-exporter gap

The session's most load-bearing discovery was that *there is no general-purpose VLA export tool*. Specifically — no tool exists today that takes a trained VLA checkpoint (from LeRobot, OpenVLA, OpenPI, SmolVLA) and produces an optimized TensorRT / ONNX engine for Jetson deployment. Teams do this manually, painfully, or not at all.

Three concrete GitHub issues were cited as evidence:

- **LeRobot #819** — torchvision >0.21 needs torch >2.6, latest Jetson torch is 2.5. Blocks JetPack GPU acceleration entirely. Closed "not planned."
- **LeRobot #1923** — "Deploying SmolVLA with a simulator" — users cannot get SmolVLA predictions into sim or real robots.
- **OpenPI #826** — "Jetson Thor support for Pi05 - Invalid key(s) in state_dict" — pi0.5 literally will not load on Thor.

Further pain inventory: **OpenPI alone has 6+ zero-response issues about cross-embodiment adaptation** (872, 740, 714, 580, 449, 591). This became the "after the exporter ships, the next wedge" note.

The wedge thesis crystallized: *Every VLA team writes their own export pipeline and most of them break.* Reflex is the bridge. Every future scope-creep decision was later measured against one question: *does this close the exporter gap for real users posting on these issue threads?*

---

## The 7-wedge CLI surface

With the naming locked and the gap identified, the CLI grammar emerged:

```
reflex export   # checkpoint → ONNX + TensorRT
reflex serve    # HTTP inference server
reflex guard    # URDF safety limits + EU AI Act logging
reflex turbo    # action-head optimization (adaptive denoising)
reflex split    # cloud-edge orchestration
reflex adapt    # cross-embodiment mapping
reflex check    # pre-deployment checks
```

Each wedge was a naturally-discovered pain point:
- **export** — the gap.
- **serve** — you exported, now what? HTTP server.
- **guard** — you served, but it could hurt a human. EU AI Act Article 12 audit log.
- **turbo** — you served, but it's slow. Early-stop denoising.
- **split** — you need cloud and edge. Orchestrator with fallback.
- **adapt** — you have Franka data but SO-100 hardware. Cross-embodiment.
- **check** — 5 pre-deploy gates before the robot gets the ONNX.

The principle articulated: *"Don't drop CLI commands users are finding."* Each command must be discoverable through natural workflow. If `turbo` is the 2-of-10 use case, keep it; removing it forces users to rebuild it themselves.

### Consolidation rationale (foreshadowed)

Even on Apr-10, the transcript showed awareness that the 7 wedges were too many. Three were marked for eventual deprecation: `split`, `adapt`, `turbo`. `check` was marked for eventual merge into `validate --quick`. ADR: `2026-04-14-deprioritize-adapt-and-split.md`. This consolidation actually landed Apr-16 (steps 1-5 in `fdd9bb3..ed8157c`), but the rationale — *"three commands are dead weight (split <10% use, adapt no users, turbo broken on 3/4 models)"* — was already in the record.

---

## First SmolVLA export bring-up

Armed with the CLI grammar, the session started cutting code. The initial scaffold landed as commit **`e8fe39f`** (2026-04-13 19:59) — technically after Apr-10 midnight, but work-session-continuous:

- 19 files, +1094 LOC.
- `cli.py`, `config.py`, `checkpoint.py`, `decompose.py`, `exporters/onnx_export.py`, `exporters/trt_build.py`, `inference.py`, `validate.py`, `benchmark.py`.
- Initial abstraction: *"split VLA into vision / backbone / denoising"* — superseded within 24 hours once real SmolVLA architecture was discovered.

The `decompose.py` file — RMSNorm and RoPE decompositions for ONNX opset 19 — **survived every subsequent rewrite**. The core insight that RMSNorm lands as `variance = x.to(float32).pow(2).mean(-1, keepdim=True); x * rsqrt(variance + eps) * weight` with explicit fp32 upcast is load-bearing for every model added later.

Real SmolVLA export followed quickly in **`1ed46ab`** (2026-04-14 00:48):
- Suffix encoder (1.58M params): max_diff 2.15e-06 PASS.
- Action projection: max_diff 1.07e-06 PASS.
- Bench: 5.07 ms mean / 197 Hz — but this was *without* the expert transformer.
- SmolVLA structural discovery: `expert_hidden=720`, `action_dim=32`, 450M total params (350M VLM + 98M expert + 1.6M projections).

Expert transformer landed in **`74d24c3`** (2026-04-14 01:35):
- GQA architecture: 15 Q heads, 5 KV heads (3:1), `head_dim=64`, intermediate 2048, 16 layers, 98.2M params.
- Decomposed RMSNorm + decomposed RoPE exports cleanly at opset 19.
- Per-layer max_diff 5.36e-07.

Full 16-layer stack: **`47f3d5d`** (01:46): Full 10-step Euler denoise at 202.1 ms / 4.9 Hz. Alternating self-attn (even indices) and cross-attn (odd indices [1,3,5,7,9,11,13,15]). ONNX 1.1MB, max_diff 4.77e-06.

VLM backbone: **`da237f5`** (01:55): SmolVLM2-500M truncated to 16 layers = 350.2M (86.4M vision + 263.8M decoder). Key number: VLM prefix cost 48.3 ms on A100. 2.01 GB peak GPU.

E2E pipeline: **`fb9a317`** (02:06): image → VLM prefix → expert stack → 50 actions at 253.3 ms / 3.9 Hz on A100 PyTorch eager. 1.82 GB peak — fits Orin Nano 8GB.

---

## TRT FP16 benchmark — the right numbers

The Apr-14 TRT FP16 benchmark (commit `fce8a6f`, run on Modal A10G — closest cloud GPU to Jetson Orin) produced the table that flipped the strategic narrative:

| Model | Params | torch.compile | ORT-GPU FP32 | TRT FP16 | Speedup |
|-------|--------|---------------|--------------|----------|---------|
| SmolVLA | 99.8M | 3.06 ms | 3.26 ms | **0.95 ms** | 3.2× |
| pi0 | 314.6M | 6.23 ms | 5.53 ms | **1.94 ms** | 3.2× |
| pi0.5 | 426.9M | 7.34 ms | 7.37 ms | **2.24 ms** | 3.3× |
| GR00T | 1091.7M | 14.61 ms | 14.45 ms | **5.59 ms** | 2.6× |

Per-chunk wall-clock (10-step denoise): SmolVLA 9.5 ms = 105 Hz; pi0 19.4 ms = 52 Hz; pi0.5 22.4 ms = 45 Hz; GR00T 55.9 ms = 18 Hz. **All four meet or exceed the 20-30 Hz target for real-time robot control.**

Earlier in the day — before this benchmark — the Apr-14 narrative was the opposite: *"Reflex cannot win faster inference on cloud GPU."* That claim (L1 of the post-mortem) was based on a benchmark where `onnxruntime-gpu` was silently falling back to CPU (see `2026-04-14_benchmark_postmortem.md`). Once the CUDA 12 + cuDNN 9 alignment pins landed, TRT FP16 was shown to **dominate** torch.compile 2.6–3.3× on cloud GPU. The marketing narrative flipped from "edge-only moat" to "cloud AND edge, same TRT toolchain."

---

## Wedge consolidation rationale

The transcript surfaced three pain points driving eventual CLI consolidation:

1. Three commands pretend to "validate" something: `check` (static pre-flight), `validate` (ONNX-vs-PyTorch parity), `guard` (URDF safety). Users will guess wrong.
2. Two commands pretend to "benchmark": `bench` (latency) and `eval` (task success).
3. Three commands are dead weight: `split` (<10% use), `adapt` (no users), `turbo` (broken on 3/4 models).

Proposed redesign — landed Apr-16:
- MERGE `check` → `validate --quick`.
- MERGE `eval` → `bench --benchmark`.
- DELETE `split` (replace with `--cloud-fallback` flag).
- DELETE `adapt`.
- DELETE `turbo` (move to `serve --adaptive-steps`).

The October number: **13 → 9 commands**. Smaller surface, tighter story.

---

## Competitor landscape captured

- **Physical Intelligence ($11B, pi0/pi0.5, Levine / Hausman / Finn)** — customer OR existential competitor depending on whether they ship "pi-serve."
- **Figure ($39B, Helix, dual-system arch — System 1 on robot + System 2 in cloud)** — vertically integrated, will not sell model, cannot become "Windows of robotics."
- **1X (OpenAI-backed, Neo consumer robot, world-models + teleop supervision).**
- **Skild ($300M, Pathak)** — post-Sergey Levine graduate student network.
- **NVIDIA GR00T / Isaac / Jetson Thor / Cosmos** — the full-stack threat.
- **LeRobot** — async server broken (issues 2356, 3204, 2980). No working VLA async server sold anywhere. **Issue 3146 opened 2026-03-12 by `jashshah999`, 0 comments, 5 reactions** — the wedge signal.
- **Foxglove Agent + Cloud** — $90/user/mo, $40M Series B. Customers include NVIDIA / Amazon / Anduril / Waabi / Dexterity. Observability niche.
- **AllenAI vla-eval** — WebSocket+msgpack adapter, no HTTP server.
- **VLA-Perf (arxiv 2602.18397)** already claimed "first VLA inference benchmark" framing — flank it with narrower paper.
- **LiteVLA-Edge** — ships GGUF quantization on Jetson at 6.6 Hz in ROS2 pipeline. Direct competitor.
- **Inference.net** (grants up to $10k, ~50% approval) — the easiest non-dilutive money right now.

---

## Pricing ladder

The session settled on a tiered ladder that has held across every subsequent artifact:

- **Free** — FP16 export (slow but works).
- **Pro — $99/mo** — FP8 / INT4 quantization + validation.
- **Team — $499/mo** — custom kernel optimization.
- **Enterprise — $2–5k/mo per robot** — full runtime + ROS2 + support.

Revenue curve: $5–10K/mo month 3 → $15–25K month 6 → $30–50K month 10 → $80–150K month 14 → $200K+ month 20 → acquisition or Series A at month 24.

Later council review (line 6497 of transcript) flagged `$99/mo anchors you cheap. Real comp set is Roboflow ($249-999), W&B enterprise ($30-80k)`. The restructure proposed: **Pro $299/mo, Team $1500/mo, Enterprise $30-80k/yr.** Unresolved.

v0.2 flagship target: **`reflex distill`** via pi-Flow recipe (arXiv 2510.14974, ICLR'26) on pi0.5-base. 10→2 denoising steps, <5% LIBERO accuracy drop. Target ~$60 Modal cost, ~1 week eng. (Pre-mortem base rate: 3 weeks.)

---

## Safety guard placeholder

The observation: *"Nothing stops a VLA from outputting a motor command that damages the robot or hurts a human. There's no provable safety layer."*

`reflex guard` was scaffolded as a placeholder CLI:
- URDF → SafetyLimits JSON.
- Clamp at request time, sub-millisecond check.
- EU AI Act Article 12 audit log — timestamp, input hash, raw and safe actions, violations, model version.

Design gap flagged: *"needs a provable control-theoretic safety shim (CBF / MPC layer) before Thor-tier enterprise customers sign."* Kept as v0.2+ work.

EU AI Act Article 12 retention requirement: **≥6 months for log format.** Current log format missing the "decision pathway" field (which policy head / token produced the action). Draft requires ML-model-state auditability (Clause 8.3.1).

---

## OSS / GTM motion

The GTM motion captured as a narrative: *"Buy Jetson Orin Nano ($299); Clone LeRobot → attempt SmolVLA export to TensorRT → it fails; Fix it → PR to LeRobot (solves #3146); Clone OpenPI → reproduce pi0.5 crash on Orin (#826) → fix state_dict loading."*

Literally: post fixes to the open issue threads from the gap list. Every exporter ship should drop a link / PR into LeRobot #819, #1923, #3146 or OpenPI #826 to generate inbound.

Three launch drafts written (all in `/launch/`, unpublished pending user approval):
- `lerobot_3146_draft.md` — direct reply to the wedge-signal issue.
- `show_hn_draft.md` — "Show HN: Reflex – ONNX/TensorRT export for VLA models, runs on Jetson."
- `reddit_robotics_draft.md` — "Open-source tool to deploy SmolVLA / pi0 / pi0.5 / GR00T to Jetson via ONNX + TensorRT."

Sequencing: **LeRobot #3146 first → 48–72h later, Show HN → same/next day, r/robotics.** Rationale: reduces signal in each, forces serial attention.

Tagline: ***"Deploy any VLA model to any edge hardware. One command."*** README snippet: `pip install reflex-vla; reflex export --model lerobot/smolvla_base --target orin-nano`. The one-command demo was declared the only marketing asset that matters.

License: **Apache 2.0**. Ships free, captures market "the way Vercel captured Next.js developers or HashiCorp captured ops."

Research flagship paper: *"Action-Chunk Scheduling: A Serving Contract for Robot Foundation Models"* — MLSys 2027 target.

---

## Knowledge repo separation

Late in the session, the decision to separate code from knowledge:
- `reflex-vla/` — code + agents artifacts.
- `reflex_context/` — vision, decisions, research, experiments, product, inbox, archive. `CLAUDE.md`, `TEMPLATE.md` at each level.

Per-wedge files must feed the real code, not live in a separate research tree — the `PRIORITY_MATRIX.md` + researcher outputs had a *"no connection to code"* problem on earlier projects, solved here by the `reflex_context/` layout.

---

## Outcome

By the time the session paused, the project had:

- **A name** (Reflex) that works everywhere.
- **A path** (01 cashflow, 06 moonshot, Alt-VLA optional pivot).
- **A wedge** (VLA deployment — because no exporter exists).
- **A CLI surface** (7 wedges with a known consolidation to 4-5).
- **Working SmolVLA export** (max_diff <5e-6 across every component).
- **A benchmark table** (TRT FP16 dominates torch.compile 2.6–3.3×).
- **A pricing ladder** ($0 / $99 / $499 / $2–5k).
- **Three launch drafts** ready to post on user approval.
- **A knowledge repo** separated from code.

The Apr-10 session set the direction that held across every subsequent day. No subsequent session overturned the core decisions; they iterated on bugs, benchmarks, and the one unfilled gap — VLM prefix encoder for real task-conditioned actions — which became the multi-day chase documented in the Apr-16 and Apr-17 sessions.

---

## Carry-over

Work from this session that continued into later sessions:

- **VLM prefix encoder** — identified as the single biggest unfilled thing ("v0.1 runs with random conditioning, action values look random / nonsensical") — chased through the `2026-04-16_libero_infra_hunt.md` and `2026-04-17_libero_correctness_hunt.md` sessions.
- **CLI consolidation (split / adapt / turbo → deprecated; check → validate --quick)** — designed Apr-10, landed Apr-16 (`fdd9bb3..ed8157c`).
- **TRT × batching sharp edge** — noted Apr-10, diagnostically verified Apr-13, `skip TRT EP when --max-batch > 1` fix landed as `e76678c` on Apr-14.
- **`reflex distill`** — target as v0.2 flagship via pi-Flow; scaffolded Apr-16 as `ed8157c`, training loop deferred to v0.2.1.
- **Launch drafts** — written Apr-10, still unpublished pending user approval as of Apr-17.
- **pre-brief Nathan Lambert / Zhihao Jia / Hao Zhang / Yiying Zhang outreach** — planned, not executed.
- **HF Community Grant / Inference.net grants** — plans set, not yet filed.
- **User interviews with LeRobot / Open Robotics / ROS Discourse** — noted as Phase 1 gap, not yet done.
- **Continuous batching (Phase III)** and **adaptive denoising validation (Phase IV)** — designed Apr-10, validated on real VLAs Apr-13, documented in `2026-04-13_phase_iii_batching_adaptive.md`.
- **LIBERO task-success number** — the north-star benchmark, attempted Apr-16 (infrastructure), still 0% as of Apr-17 (correctness).
