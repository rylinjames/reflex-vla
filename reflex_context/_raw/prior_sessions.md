# Prior Sessions Knowledge Base — Reflex VLA

## IMPORTANT FINDING: File mismatch on request

The four transcripts originally specified by the caller do NOT contain reflex-vla content. Verified via `cwd` inspection of each JSONL:

| Requested file | Actual project (cwd) |
|---|---|
| `35017f59-56ac-43fb-be05-d8c7ec39fa4e.jsonl` (10MB) | opalclone iOS |
| `aca7560b-e466-4da8-a550-f7ee6521144e.jsonl` (17MB) | Santhica / santhica-backend |
| `799fe46d-1b35-4935-abbd-8587fbbe704c.jsonl` (22MB) | whop-ai-chat / for-raymo (EcomLinked) |
| `c7962eb2-9030-4a11-b35a-5b3bc1b7b000.jsonl` (22MB) | Taptic iOS |

Zero mentions of `reflex`, `SmolVLA`, `OpenVLA`, `pi0`, or `LeRobot` in these four files. They are other projects that happened to be the largest files on disk at the time.

The ACTUAL reflex-vla sessions on the machine were auto-detected by `cwd` scan + keyword hit-rate:

| Real file | Size | Content |
|---|---|---|
| `ced2c4f1-a341-45bf-ae1b-ba9f6ab0931c.jsonl` | 55MB | The mega reflex-vla session. 7265 `reflex-vla` hits, 1827 `smolvla` hits, 876 `LIBERO` hits. Contains the entire project-naming, wedge-selection, and v0.1 build. |
| `30070a3d-79cb-4fca-8a63-f18261fd445c.jsonl` | 450KB | VLA-to-hardware roadmap exploration + MCP tooling list. |
| `5e5753a2-826f-42aa-a991-7d239c605838.jsonl` | 8MB | Mostly EcomLinked/WHOP content, minor reflex reference. |
| `ba6b3fbc-31dd-4b71-abfc-bf61bfb285c7.jsonl` | 4.5MB | Minor reflex reference only. |

The rest of this document extracts insights from `ced2c4f1` (the primary source) and `30070a3d` (the roadmap session). The four requested files contain no reflex-vla-relevant content and are not summarized further.

---

## File: ced2c4f1-a341-45bf-ae1b-ba9f6ab0931c (the mega session)

### Themes covered in this session
- Pre-project InferScope / Axion strategic wedge sweep — 31 wedges, 7-path company-shape analysis
- Path 00 / Path Alt-VLA framing: VLA as "trillion-dollar tail" vs Datadog-for-inference as core
- Naming the tool: **Reflex** chosen over Forge, Actuate
- Market landscape: PI (pi0/pi0.5), Figure (Helix), 1X, Skild, NVIDIA GR00T, OpenVLA, SmolVLA
- Gap-finding exercise across LeRobot, OpenPI, GR00T — "no general-purpose VLA export tool exists"
- Concrete v0.1 scaffold: `src/reflex`, CLI verbs (`export`, `validate`, `benchmark`, `targets`)
- Hardware targets defined: Orin Nano, Orin AGX, Thor, desktop
- SmolVLA export pipeline implementation: ONNX decomposition, TensorRT engine build
- Modal runtime used for Jetson-approximating builds on A100
- Competitive pressure tracking: NVIDIA GR00T Serve as "kills everything" scenario

### Theme: Product naming — why "Reflex"
- Finding: Settled on `Reflex` after rejecting `Forge` (too generic) and `Actuate` (sounds like "actually").
- Context: Quote — *"Reflex... one word, memorable, technical but not jargon, works as a CLI name (`reflex export`, `reflex serve`, `reflex guard`), works as a company name, works as a hardware brand, and nobody in robotics AI has it."*
- Carry-over: Name locks in the CLI grammar we ship today (`reflex export/validate/benchmark/targets`). Company / hardware branding is already aligned.

### Theme: 7-wedge / 7-path strategic taxonomy
- Finding: The 7 "paths" (Path 00–07) were company-shape candidates, not product wedges. Each got a pre-mortem-style critique.
  - Path 01 = Datadog-for-Inference (revenue spine)
  - Path 02 = Benchmark Authority (kill-as-primary, keep as free asset)
  - Path 03 = Reasoning-aware serving (DEMOTE to SKU)
  - Path 05 = DiT/video serving (2027 second act)
  - Path 06 = Physical AI Inference Stack (STAGED via "Stack H" — only $1T shot)
  - Path 07 = NVFP4/Blackwell calibration (DEMOTE to SKU)
  - Path 00 = Inference Research Lab (the synthesis)
  - Path Alt-VLA = month-18 pivot option into full VLA foundation-model lab
- Context: Barbell strategy — Path 01 pays bills, Path 06 is the moonshot; pivot to Path Alt-VLA only if 5-of-7 triggers hit by month 18.
- Carry-over: The reflex-vla repo exists because Path 06 (VLA Inference) plus wedge #17 became the narrowest, shippable slice. The "7-wedge" framing in current Reflex docs inherits from this vocabulary.

### Theme: VLA model landscape (the fixed fact-base)
- Finding: Only a handful of VLA checkpoints matter today.
  - **SmolVLA** (HuggingFace / LeRobot, 450M params, 907 MB single file) — *"the most export-friendly VLA available"*
  - **pi0 / pi0.5** (Physical Intelligence, flow-matching head, 50 Hz control)
  - **OpenVLA** (Open X-Embodiment baseline, 7B, discrete action tokens via RT-2 style)
  - **GR00T N1 / N1.6** (NVIDIA, bundled with Jetson Thor)
  - **Octo** — older RT-2 cousin
- Context: *"Every VLA is trained on specific robots. Pi0 was trained on 7 platforms. GR00T was trained on GR1 humanoid. SmolVLA was trained on SO-100 arms."*
- Carry-over: Exporter priority order = SmolVLA (easiest, ship first) → pi0 → OpenVLA → GR00T. Cross-embodiment retraining is an unsolved pain customers will pay for.

### Theme: The "there is no VLA exporter" gap (the actual wedge)
- Finding: *"There is NO general-purpose VLA export tool that takes a trained VLA checkpoint (from LeRobot, OpenVLA, OpenPI, SmolVLA, etc.) and produces an optimized TensorRT/ONNX engine for Jetson deployment. Everyone is doing this manually, painfully, or not at all. NVIDIA's own GR00T pipeline is buggy. LeRobot has zero export support. OpenPI assumes a fat GPU server."*
- Evidence — concrete GitHub issues cited:
  - LeRobot #819 — torchvision >0.21 needs torch >2.6, latest Jetson torch is 2.5. Blocks JetPack GPU accel entirely. Closed "not planned."
  - LeRobot #1923 — "Deploying SmolVLA with a simulator" — users can't get SmolVLA predictions into sim/real robots.
  - OpenPI #826 — "Jetson Thor support for Pi05 - Invalid key(s) in state_dict" — Pi0.5 literally won't load on Thor.
- Carry-over: This is the ORIGINAL rationale for the repo. Any feature scope creep should return to: "does this close the exporter gap for real users posting on these issue threads?"

### Theme: SmolVLA export plan (ONNX + TRT)
- Finding: ONNX export works if you manually decompose RMSNorm/RoPE and externalize the denoising loop. TensorRT conversion is proven (GR00T does it; OpenPI community has working TRT pipeline). No published SmolVLA export existed — v0.1 fills that.
- Context: The initial build validated per-layer numerical parity:
  - Suffix encoder max_diff = 2.15e-06
  - Action projection max_diff = 1.07e-06
  - Expert layer max_diff = 5.36e-07
  - Full expert stack max_diff = 4.77e-06
- Carry-over: Parity thresholds are established. Any future model adds should reproduce <1e-5 per-layer max diff before shipping.

### Theme: CLI surface & consolidation (today's shape)
- Finding: The original scaffold exposed `reflex export/validate/benchmark/targets` (+ in-development `serve`, `guard`, `split`, `adapt`, `turbo`). Current session tasks reflect a consolidation pass — `split`/`adapt`/`turbo` getting deprecation warnings; `check` merged into `validate --quick`; `export_vlm_prefix` auto-invoked inside `reflex export` for SmolVLA.
- Context (from the current Task list): Steps 1-5 done. `reflex bench` now has `--benchmark` wrapping LIBERO/SimplerEnv. `reflex distill` scaffolded for DMPO recipe. `reflex.runtime.adapters.vla_eval` exists.
- Carry-over: The public API is now `export | validate | benchmark | distill | targets`. Deprecation deadlines for `split/adapt/turbo` should still be communicated in CHANGELOG for any existing users.

### Theme: Hardware target matrix (baked in)
- Finding: Hardware profiles registered at build time:
  - `orin-nano`  — Jetson Orin Nano, 8 GB, fp16
  - `orin-agx`   — Jetson Orin AGX, 64 GB, fp16/fp8
  - `thor`       — Jetson Thor, 128 GB, fp8
  - `desktop`    — 4090 / A100 / H100, 24+ GB, fp16
- Carry-over: These are the only profiles exposed via `reflex targets`. Any Blackwell / Rubin / ARM server target additions should follow this table's schema.

### Theme: Modal as the Jetson-substitute cloud
- Finding: Dev loop runs on Modal A100. Scripts like `scripts/modal_cli_export.py` and `scripts/modal_libero10.py` wrap the exporter + LIBERO eval in a Modal function so the team never needs a physical Jetson to iterate.
- Context: Running `modal run scripts/modal_cli_export.py` validated the end-to-end `reflex export --model lerobot/smolvla_base --target orin-nano` pipeline before any Jetson hardware arrived.
- Carry-over: Pattern to keep — every new model or target gets a Modal smoke-test script in `scripts/` that reproduces the full pipeline on rented GPU. Don't gate PRs on physical hardware.

### Theme: LIBERO as the go/no-go benchmark
- Finding: LIBERO-10 + SimplerEnv are the two benchmark harnesses wired in. The open in-progress Task is "Ship LIBERO-10 Modal run and capture task-success number" (#23). The SmolVLA Libero checkpoint is `lerobot/smolvla_libero`.
- Context: `reflex bench --benchmark` wraps LIBERO / SimplerEnv. `reflex.runtime.adapters.vla_eval` is the shared adapter. Normalizer support is an open gap (task #24).
- Carry-over: Do not ship an exporter claim without a LIBERO task-success regression number. Headline metric for v0.1 is LIBERO-10 success rate before vs after the export.

### Theme: "NVIDIA GR00T Serve" as the extinction event
- Finding: If NVIDIA ships a bundled VLA serving runtime alongside Jetson Thor SDK, it kills Path Alt-VLA AND commoditizes the exporter wedge. 30-55% probability through 2026.
- Carry-over: Watch for any signal in Isaac / GR00T release notes mentioning "Serve" or runtime bundling. That's the hard kill-criterion for the standalone-exporter pitch; pivot accelerator if it fires.

### Theme: Action-chunk scheduling as the systems research flagship
- Finding: Planned flagship paper — *"Action-Chunk Scheduling: A Serving Contract for Robot Foundation Models"* — MLSys 2027 target. Core thesis: action chunks (pi0 convention, 8–50 consecutive motor commands) should be the atomic scheduling unit, not tokens.
- Context: Tied to *"Edge-Cloud Hybrid VLA Inference Under Intermittent Connectivity"* (CoRL 2026 / RSS 2027 target) and hard-real-time scheduler design.
- Carry-over: If we want research credibility behind the runtime, the scheduler code needs to be the published artifact, not an implementation detail.

### Theme: Competitor map (active)
- Physical Intelligence ($11B, pi0/pi0.5, Levine/Hausman/Finn) — *customer OR existential competitor* depending on whether they ship "pi-serve".
- Figure ($39B, Helix, dual-system arch — System 1 on robot + System 2 in cloud) — vertically integrated, won't sell model, can't become "Windows of robotics."
- 1X (OpenAI-backed, Neo consumer robot, world-models + teleop supervision).
- Skild ($300M, Pathak) — post-Sergey Levine graduate student network.
- NVIDIA GR00T / Isaac / Jetson Thor / Cosmos — the full-stack threat.
- VLA-Perf (arxiv 2602.18397) already claimed "first VLA inference benchmark" framing — flank it with narrower paper.

### Theme: Cost model / economics (if / when the company shape crystallizes)
- Y1 budget reference: $8M seed. $4M team-comp, $1.5M compute, $0.5M robots, rest G&A.
- Comp: VLA research lead ~$550k cash + 3–5% equity; senior robot engineer ~$400k + 0.3%; Path 00 levels ~$350k + 0.15%.
- Series B trigger: $200–400M at $1–2B post, comparable to PI ($2B post) and Skild ($1.5B post).
- Carry-over: Not directly actionable in code, but sets the magnitude for "how much of this matters" when scoping product moves.

### Theme: OSS / GTM motion (what the exporter does to the funnel)
- Month 1–2 plan quote: *"Buy Jetson Orin Nano ($299); Clone LeRobot → attempt SmolVLA export to TensorRT → it fails; Fix it → PR to LeRobot (solves #3146); Clone OpenPI → reproduce pi0.5 crash on Orin (#826) → fix state_dict loading."*
- Carry-over: The GTM is literally "post fixes to the open issue threads from the gap list." Every exporter ship should drop a link/PR into LeRobot #819, #1923, #3146 or OpenPI #826 to generate inbound.

### Theme: Pricing ladder (early hypothesis)
- From `vla_to_hardware_roadmap/README.md`:
  - Free tier: FP16 export (slow but works)
  - Pro tier: FP8/INT4 quant + validation — $99/mo
  - Team tier: custom kernel optimization — $499/mo
  - Enterprise tier: full runtime + ROS2 + support — $2–5K/mo per robot
- Revenue curve: Month 3 $5–10K → Month 24 $200K+ → acquisition or Series A.
- Carry-over: The paid surface is kernel optimization + runtime+ROS2 support, not export itself. Export stays free forever to drive funnel; only stop supporting once you can charge for serving SLAs.

### Theme: Cross-embodiment as the "next wedge"
- Finding: Cross-embodiment retraining is DIY config surgery today. OpenPI alone has 6+ issues with zero responses about adapting to new robots (issues 872, 740, 714, 580, 449, 591).
- Carry-over: After the exporter ships, the next wedge is a "cross-embodiment adapter" that takes a VLA trained on Franka and re-projects it onto SO-100 / UR5 / humanoid torsos without full re-training. Start scoping after LIBERO-10 number lands.

### Theme: Safety guard (shipping concern)
- Finding: *"Nothing stops a VLA from outputting a motor command that damages the robot or hurts a human. There's no provable safety layer."*
- Carry-over: `reflex guard` is a placeholder CLI today. Design needs a provable control-theoretic safety shim (CBF / MPC layer) before Thor-tier enterprise customers sign.

### Theme: Per-layer vlm_kv ONNX export (open engineering)
- Status: in-progress per current tasks (#25).
- Background finding: SmolVLA ONNX decomposition required manually splitting RMSNorm/RoPE and externalizing the denoising loop. Per-layer KV is the harder case.
- Carry-over: When it lands, parity regression must stay <1e-5 max diff per layer; TRT engine build should still hit the 2–3x speedup target (10Hz+) on Orin Nano.

### Theme: DMPO / reflex distill (research tie-in)
- Status: `reflex distill` scaffolded (#17). Intended to wrap DMPO / RL post-train recipe.
- Carry-over: Don't let `distill` become a sidecar — the story is "export + distill + serve are one pipeline." If distill ships as its own tool it loses the narrative.

### Theme: Brand narrative (one-liner)
- Working tagline: *"Deploy any VLA model to any edge hardware. One command."*
- README snippet: `pip install reflex-vla; reflex export --model lerobot/smolvla_base --target orin-nano`.
- Carry-over: The one-command demo is the only marketing asset that matters. Don't let docs grow past what breaks it.

---

## File: 30070a3d-79cb-4fca-8a63-f18261fd445c (VLA-to-hardware roadmap + tooling)

### Themes covered in this session
- Locating the `vla_to_hardware_roadmap` folder inside `axion_compute` (not inside reflex-vla itself)
- 4-phase hardware roadmap (software → bundle → make VLA hardware → make datacenter hardware)
- Ranking of MCP servers + Claude skills for VLA research
- Pricing / tier ladder for `vladeployer`

### Theme: 4-phase hardware roadmap
- Phase 1 — VLA software: break in, amass users, telemetry, optimize, convert to paid.
- Phase 2 — Bundle software with VLA hardware providers (Advantech, ADLINK, NVIDIA Jetson ecosystem).
- Phase 3 — Make VLA hardware (co-design inference ASIC with Taiwan design houses).
- Phase 4 — Make datacenter hardware (ODM AI server optimization → silicon).
- Carry-over: The repo you're in NOW is Phase 1 execution. Anything that goes beyond "free CLI + paid runtime" leans toward Phase 2.

### Theme: MCP + skill tier list
- Tier 1 (install first):
  1. Semiconductor Supply Chain MCP (IP cores, ASIC design services)
  2. Patents MCP (USPTO + Google Patents) — for ASIC/chip IP mapping in Phase 3–4
  3. Nexar/Octopart MCP (component sourcing / BOM in Phase 2–3)
  4. ROS MCP Server — because `vladeployer` targets ROS2 integration
  5. Deep Research skill
- Tier 2: Microchip MCP, EDA Tools MCP (MCP4EDA, Yosys), Academic Research skills, DigiKey MCP, Materials Project MCP.
- Already connected: paper-search, hf-mcp-server, crunchbase, github, youtube-transcript.
- Carry-over: When exploration on Phase 2/3 restarts, this ranked list short-circuits "which MCP do I need?" questions.

### Theme: Product tier ladder (copy from README)
- vladeployer one-liner: "checkpoint in, Jetson engine out."
- Free: FP16 export
- Pro ($99/mo): FP8/INT4 quant + validation
- Team ($499/mo): custom kernel optimization
- Enterprise ($2–5K/mo per robot): full runtime + ROS2 + support
- Revenue curve: $5–10K/mo (m3) → $15–25K (m6) → $30–50K (m10) → $80–150K (m14) → $200K+ (m20) → acquisition or Series A at m24.
- Carry-over: Confirms the pricing seen in the ced2c4f1 extracts; consistent across sessions.

---

## Notes on source mismatch & dedup posture

- The four files originally named in the task contain nothing reflex-vla relevant; they are separate project histories (opalclone, Santhica, whop-ai-chat / EcomLinked, Taptic). I did not waste output space on their content.
- For the real reflex-vla work, `ced2c4f1` is the single authoritative prior session. All other sessions are either derivative (repeat points) or off-topic.
- Dedup across files is minimal because the other sessions had only incidental mentions. All duplication within `ced2c4f1` (e.g., multiple verdict paragraphs repeating "Path 06 is the only $1T shot") has been collapsed into the single Theme lines above rather than quoted twice.
- Paths referenced (absolute) for follow-up:
  - `/Users/romirjain/.claude/projects/-Users-romirjain/ced2c4f1-a341-45bf-ae1b-ba9f6ab0931c.jsonl`
  - `/Users/romirjain/.claude/projects/-Users-romirjain/30070a3d-79cb-4fca-8a63-f18261fd445c.jsonl`
  - `/Users/romirjain/Desktop/building projects/axion_compute/vla_to_hardware_roadmap/README.md`
  - `/Users/romirjain/Desktop/building projects/reflex-vla/` (the active repo)
