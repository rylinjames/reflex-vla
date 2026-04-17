# Competitor Landscape

Positioning: **Reflex is the deployment layer for VLAs** — take a Vision-Language-Action model off the training cluster and onto a robot. Cross-framework ONNX + TRT, edge-first.

Per-competitor files live at `reflex_context/02_research/competitors/{physical_intelligence, nvidia_groot, lerobot, vlagents, allenai_vla_eval}.md`. This file is the synthesis.

Each entry: who they are, what they ship, where Reflex differentiates, existential-risk call.

---

## LeRobot (HuggingFace)

### What they ship

- Open-source VLA training + policy-eval framework from HuggingFace.
- Supports pi0, pi0.5, SmolVLA, ACT, Diffusion Policy out of the box.
- `lerobot/smolvla_base`, `lerobot/pi0_base`, `lerobot/pi05_base`, `lerobot/smolvla_libero` — the primary HF hub presence for open VLA checkpoints.
- RTC real-time chunking (`lerobot.policies.rtc`, v0.5 March 2026).

### The VLA exporter gap

From `prior_sessions.md` — explicit list of LeRobot GitHub issues:

- **#819** — torchvision >0.21 needs torch >2.6, latest Jetson torch is 2.5. Blocks JetPack GPU accel entirely. **Closed "not planned."** The gap is officially abandoned upstream.
- **#1923** — "Deploying SmolVLA with a simulator" — users can't get SmolVLA predictions into sim/real robots.
- **#3146** — opened `2026-03-12` by `@jashshah999`, **0 comments, 5 reactions**, untouched since March. Asks for ONNX/TRT export. **This is Reflex's wedge signal** — silent demand, zero coverage.
- **#2061** — torch.compile for policies.
- **#2356** — "AsyncInference only runs one chunk." One of three broken async-server issues (#2356, #3204, #2980).
- **#1899** — "smolvla ONNX" companion request.

Counting the opens:
- **ONNX issues: 4 open / 0 closed**
- **TensorRT: 3 open / 0 closed**

Async-server status: **no working VLA async server sold anywhere** (sessions_md line 195). LeRobot's async is broken. OpenPI's async is unshipped.

### Reflex differentiation

- Reflex ships the export → TRT engine → serve pipeline LeRobot lacks.
- `reflex serve` composes the RTC chunking semantics + guard + split on top — not in LeRobot.
- GTM: drop a link in #3146 on launch. Every exporter ship drops a link/PR into LeRobot #819, #1923, #3146 or OpenPI #826 to generate inbound (prior_sessions.md OSS/GTM section).

### People to @-mention (from current_session.md line 5783)

- `@jashshah999` — author of #3146, 19 PRs to LeRobot. First ally.
- `@imstevenpmwork` — most-active LeRobot staff (95 PRs).
- `@fracapuano` — commentator-in-chief, engages on deployment threads.
- `@not-heavychevy` — torch.compile champion.

### Existential risk

- HuggingFace could ship `lerobot export` tomorrow and fold the whole wedge.
- Probability: medium. They're under-resourced on deployment; LeRobot staff are training-focused.
- **Mitigation:** ship first, own the `reflex` name, position as "the one place that covers 4 VLA families + OpenVLA" before HF plays catch-up on a single model.

---

## Physical Intelligence (`openpi`)

### What they ship

- pi0, pi0.5 flow-matching VLAs. Apache-2.0 weights.
- $11B valuation, Sergey Levine / Karol Hausman / Chelsea Finn.
- Flow-matching head, 50 Hz control target.
- **Zero deployment runtime.** Checkpoint only.

### The OpenPI exporter gap

- **Issue #826** — "Jetson Thor support for Pi05 - Invalid key(s) in state_dict". Pi0.5 **literally won't load on Thor** out of the box.
- **Issue #386** — "Deploying Pi0 on Jetson Orin errors."
- **6+ issues with zero responses about cross-embodiment adaptation** (prior_sessions.md): #872, #740, #714, #580, #449, #591.

### Reflex differentiation

- Reflex is "the bridge" between PI's checkpoints and Jetson (sessions_md line 193: *"Physical Intelligence: releases pi0/pi0.5 weights only, no deployment tool. Reflex = the bridge."*).
- Show HN draft: **"OpenPI doesn't ship a serve layer"** — the tagline for the LeRobot post (current_session.md line 5774).
- Pi0 on Reflex: 23.6ms per 10-step chunk / 42 Hz on A10G. Pi0.5: 27.1ms / 37 Hz. With TRT FP16. Apple-to-apples beats any reasonable PyTorch baseline 3.2×.

### Existential risk — the clear and present danger

From current_session.md "Verbalized-only insights" #7:

> "When Physical Intelligence ships pi-1.5 with a first-party Jetson runtime (they will; deployment friction kills their adoption), Reflex's reason to exist evaporates."

- Probability: **high within 18 months**. PI has the compute, talent, and motivation.
- **Mitigation:** Reflex must be multi-VLA (which it is — 4 families + OpenVLA) and own the positioning before PI ships pi-serve.

---

## Figure

### What they ship

- $39B valuation. Helix — dual-system architecture: System 1 (fast policy) on-robot + System 2 (slow planner) in cloud.
- **Vertically integrated** — owns robot + model + deployment.

### Reflex differentiation

- Figure **won't sell their model** — so they can't become "Windows of robotics."
- Reflex lives in the ecosystem of **open VLAs** (SmolVLA, pi0, OpenVLA, GR00T). Figure lives in its own silo.
- Non-overlap by construction.

### Existential risk

- Low. Competitors for developer mindshare, not for the deployment-runtime wedge.

---

## 1X

### What they ship

- OpenAI-backed. Neo consumer robot.
- World-models + teleop supervision approach.

### Reflex differentiation

- Like Figure: vertically integrated, won't sell model or deployment stack.
- Non-overlapping customer segment.

---

## Skild

### What they ship

- $300M. Pathak lab spinout. Post-Sergey-Levine grad-student network (same people who built OpenVLA).

### Reflex differentiation

- Foundation-model company, not a deployment-stack company.
- **Potential future customer** if Skild needs a deployment partner for their models.

---

## NVIDIA GEAR / GR00T / Isaac / Jetson Thor / Cosmos

### What they ship

- **GR00T N1.6-3B** — 3.29B-param VLA, bundled with Jetson Thor SDK.
- **Isaac Sim 5.0** — sim for VLA training / eval.
- **Cosmos-Reason-2B** — new backbone for GR00T N1.6 (NOT Qwen3 as Reflex's early research batch incorrectly assumed — current_session.md line 5756).
- **Jetson hardware** — the target we deploy to.
- **GR00T issue #517** — GR00T TRT export, NVIDIA's in-progress work. **They know the gap exists.**

### Reflex differentiation

- Reflex handles GR00T N1.6 today: `src/reflex/exporters/gr00t_exporter.py`. 1091.7M expert + 10M buffers, full-stack export with action_encoder/decoder wrapped at `embodiment_id=0`.
- GR00T serve fix (commit `ff9fc3a`): wrapped DiT expert so the denoise loop works raw-in / raw-out.
- Same TRT engine that runs on A10G (via trtexec) runs on Jetson via Jetson TRT — no separate cloud/edge model variants.

### The "NVIDIA GR00T Serve" extinction event

From `prior_sessions.md`:

> "If NVIDIA ships a bundled VLA serving runtime alongside Jetson Thor SDK, it kills Path Alt-VLA AND commoditizes the exporter wedge. 30-55% probability through 2026."

- This is the **single hardest kill criterion** for the standalone-exporter pitch.
- **Watch signal:** any Isaac / GR00T release notes mentioning "Serve" or runtime bundling. When that fires, Reflex accelerates to Phase 2 (bundle with hardware providers) or pivots to Phase 3 (make VLA hardware) faster.

### Issue to cross-link on launch

- **gr00t #517** (GR00T TRT export) — our cross-link.

---

## AllenAI vla-eval

### What they ship

- LIBERO / SimplerEnv / ManiSkill evaluation harness. WebSocket+msgpack wire format.
- arXiv 2603.13966.

### Reflex differentiation

- **Not a direct competitor** — we consume them.
- ADR `2026-04-14-wrap-not-rebuild-vla-eval.md`: wrap, don't rebuild.
- `reflex.runtime.adapters.vla_eval.ReflexVlaEvalAdapter` implements their `PredictModelServer` interface.
- Our `reflex bench --benchmark libero_10` delegates to their harness.

### Future positioning

- If AllenAI extends vla-eval to own "deployment timing" (latency + Hz + memory in addition to task-success), the benchmark surface could absorb Reflex's pitch.
- **Mitigation:** publish Reflex as the reference backend on the vla-eval leaderboard. Be the one that ships the numbers.

---

## Foxglove

### What they ship

- Observability + teleop for robotics.
- **$40M Series B**, $90/user/mo. Customers include NVIDIA, Amazon, Anduril, Waabi, Dexterity.
- Agent + Cloud architecture.

### Reflex differentiation

- Foxglove is **observability** (the log / visualize layer). Reflex is **inference** (the run layer).
- Their CEO Banisadr at Actuate 2025 coined *"the missing infrastructure layer for real-world robotics"* — exactly the positioning Reflex is claiming. Phrase on the "steal" list (current_session.md line 5791).
- **Potential integration partner** — they visualize what Reflex serves.

### Reflex's observability play

- Engineering budget = bottom-up PLG (Foxglove wedge).
- Legal/CISO budget = top-down EU AI Act audit (where `reflex guard` logs).
- Split across two axes, not head-to-head with Foxglove.

---

## LiteVLA-Edge

### What they ship

- GGUF quantization on Jetson at 6.6 Hz in a ROS2 pipeline.

### Reflex differentiation

- **Direct competitor on the Jetson-edge wedge.**
- Reflex targets 20–30 Hz (SmolVLA 86 Hz, pi0 42 Hz on A10G; Jetson proxy pending). LiteVLA-Edge at 6.6 Hz is slower but may be a stronger ROS2 story.
- Reflex does not support GGUF quantization — paths: FP16 default, FP8 (pro tier, paid), INT4 (pro tier, paid).

### Signal

> "No serious deployment-tool competitor yet (all under 5 stars), but the window is closing" — current_session.md Batch 1 findings.

---

## OpenPI / OpenVLA projects

### What they ship

- Open-source VLA checkpoints (pi0, pi0.5 under openpi; OpenVLA under `openvla/openvla-7b`).
- Training + research code, no deployment stack.

### Reflex differentiation

- Reflex handles both families. pi0/pi0.5 via flow-matching exporter; OpenVLA via postprocess helper + `optimum-cli export onnx`.
- `openvla_exporter.py` deliberately raises NotImplementedError — optimum-onnx already handles Llama-2 + DINOv2 + SigLIP + projector. Reflex ships only the 256-bin lookup (`reflex.postprocess.openvla.decode_actions`).

---

## xVLA

### What they ship

- 880M, **tokenized action head**. New model family as of 2026 (current_session.md line 5756, 6961).

### Reflex differentiation

- **Reflex does NOT support xVLA today.** GOALS.yaml weight 7: `xvla-exporter` — "reflex export auto-detects and exports xVLA (880M, tokenized action head) to ONNX".
- Priority: ship LIBERO number first, then plumb xVLA.

---

## Xiaomi Robotics-0

### What they ship

- 4.7B-param open-source VLA with **async execution decoupled** (on-device rollout while planner thinks).

### Reflex differentiation

- Confirms the RTC / async-chunking pattern reflex adopted. Not a direct competitor but validates our design direction.
- Could become a supported model family if users ask.

---

## VLAgents

### What they ship

- Framework-survey paper (arXiv 2601.11250). Mostly research, not deployed tooling.

---

## VLA-Perf

### What they ship

- arXiv 2602.18397. **Claimed "first VLA inference benchmark"** framing.

### Reflex differentiation

- Reflex deliberately flanks narrower: 4 VLAs × 4 hardware targets. Not head-to-head with VLA-Perf's benchmark-authority framing.
- Quoted phrase "memory-bound on Thor" on the launch-copy steal list.

---

## Inference.net / Crusoe / others

### What they ship

- Grant / compute programs. Not competitors for Reflex's product surface.
- **Inference.net grants up to $10k**, ~50% approval — the easiest non-dilutive money available (sessions_md line 89).

---

## Observability competitors (Arize, Weights & Biases)

### What they ship

- ML observability platforms.
- Arize $249–999/mo; W&B enterprise $30–80k/yr.

### Reflex pricing calibration

From current_session.md line 6497: **"$99/mo anchors you cheap. Real comp set is Roboflow ($249-999), W&B enterprise ($30-80k)."** Council restructured pricing: Pro $299/mo, Team $1500/mo, Enterprise $30-80k/yr.

---

## Consolidated differentiation statements

### "What Reflex is" (README)

> The deployment layer between a trained VLA and a real robot. Cross-framework export (4 VLA families covered), composable runtime (serve + safety + turbo + split), Jetson-first.

### "What Reflex isn't" (README)

> A training framework (PyTorch/JAX own that) or a cloud inference provider (vLLM/Baseten own that). Reflex's moat is the deployment toolchain: cross-framework ONNX, TensorRT FP16 engines that beat `torch.compile` on cloud GPU by 2.6-3.3× *and* run on Jetson, deterministic deploy graph, and the wedge composition for production robot deployments.

### "Edge-deployable models designed specifically for robotics constraints"

Hsu a16z's **fourth pillar** — the thesis wedge. (sessions_md line 210)

### "Deploy any VLA to edge"

The one-line positioning. Everything else is dressing.

---

## Issues to cross-link on launch (one place, canonical list)

- `lerobot#3146` — the wedge signal; `@jashshah999` is the ally.
- `lerobot#1899` — smolvla ONNX companion.
- `lerobot#2061` — torch.compile for policies.
- `lerobot#2356` — AsyncInference broken.
- `openpi#386` — Pi0 Jetson Orin errors.
- `openpi#826` — Pi0.5 Jetson Thor key mismatch.
- `gr00t#517` — GR00T TRT export (NVIDIA's in-progress track).

---

## Barbell positioning (from ced2c4f1 session)

The 7 paths:
- **Path 00** = Inference Research Lab (the synthesis — where Reflex sits today).
- **Path 01** = Datadog-for-Inference (revenue spine).
- **Path 02** = Benchmark Authority (kill as primary, keep as free asset).
- **Path 03** = Reasoning-aware serving (demote to SKU).
- **Path 05** = DiT/video serving (2027 second act).
- **Path 06** = Physical AI Inference Stack (staged via "Stack H" — only $1T shot).
- **Path 07** = NVFP4/Blackwell calibration (demote to SKU).
- **Path Alt-VLA** = month-18 pivot into VLA foundation-model lab (requires 5-of-7 triggers hit).

Reflex-VLA = Path 06's narrowest shippable slice. If it fails, still have Path 01 outcome ($30B Datadog-for-inference).

---

## Phases of competition

- **Phase 1** (Months 0–6) — OSS CLI, Apache 2.0, free. First Pro tier at $99/mo (later revised to $299/mo). Core competitors: LeRobot async gap, OpenPI deployment gap, LiteVLA-Edge.
- **Phase 2** (Months 6–12) — Bundle with Seeed / Trossen / Jetson integrators. Rev-share, no inventory. Competitors: NVIDIA GR00T Serve bundle.
- **Phase 3** (Year 2+) — Reflex Compute Pack. Own-branded hardware. Competitors: Physical Intelligence pi-serve, Figure Helix.
- **Phase 4** — Make datacenter hardware (Blackwell-successor silicon). (`axion_compute/vla_to_hardware_roadmap/README.md`)

Reflex-VLA repo is **Phase 1 execution**. Everything else is downstream leverage.

---

## Related files

- `reflex_context/02_research/competitors/physical_intelligence.md`
- `reflex_context/02_research/competitors/nvidia_groot.md`
- `reflex_context/02_research/competitors/lerobot.md`
- `reflex_context/02_research/competitors/vlagents.md`
- `reflex_context/02_research/competitors/allenai_vla_eval.md`
- `launch/lerobot_3146_draft.md`, `launch/show_hn_draft.md`, `launch/reddit_robotics_draft.md`
