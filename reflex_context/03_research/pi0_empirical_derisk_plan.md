# pi0-onnx-parity: 2-day empirical de-risk plan

**Why:** Research reduced mess-up probability to 25–35% but further reduction requires empirical signal. The single biggest unknown is whether `optimum-cli export onnx` actually produces a usable PaliGemma base we can fork. This plan resolves that in ~2 days of real testing before committing to the full ~2-3w pi0-onnx-parity implementation.

**Goal outcomes:**
- HIGH: Optimum produces a clean PaliGemma ONNX we can diff against PyTorch with Polygraphy → commit to option A at ~2w estimate
- MEDIUM: Optimum partial success (produces components but KeyError on task) → proceed to fork-our-own-exporter path at ~3w estimate
- LOW: Optimum fully blocked / parity gap huge / TRT-side bugs dominate → pivot to Torch-TensorRT direct path or reduce MVP scope to SmolVLA-only

## Day 1 — Tooling + ground truth (~4 hours)

### Task 1.1: Install the Polygraphy toolkit (30 min)

```bash
cd /Users/romirjain/Desktop/building\ projects/reflex-vla
.venv/bin/python -m pip install \
  polygraphy \
  onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com \
  onnx-diagnostic \
  onnxsim
```

Verify:
```bash
.venv/bin/polygraphy --help
.venv/bin/python -c "import onnx_graphsurgeon; print(onnx_graphsurgeon.__version__)"
.venv/bin/python -c "from onnx_diagnostic import validate_onnx_model; print('ok')"
```

**Success criterion:** all three imports succeed.

### Task 1.2: Pull Tacoin parity oracle (15 min)

```bash
.venv/bin/huggingface-cli download Tacoin/openpi-pi0.5-libero-onnx \
  --local-dir /tmp/oracle_pi05_libero_onnx \
  --local-dir-use-symlinks False
```

Inspect structure:
```bash
ls -lh /tmp/oracle_pi05_libero_onnx/
# Expect: single monolithic ONNX + tokenizer files
```

**Success criterion:** Tacoin ONNX loads in onnxruntime without error.
**Caveat:** this is pi0.5, not pi0. Use for architecture reference + as a diff target for our pi0.5 work later, NOT as the pi0 oracle.

### Task 1.3: Install optimum-onnx + run PaliGemma sanity (30 min + wait)

```bash
.venv/bin/python -m pip install 'optimum[onnxruntime]>=1.22'
.venv/bin/optimum-cli --help

# The canonical export command — if this works, pi0-onnx-parity path is viable
mkdir -p /tmp/optimum_paligemma_sanity
.venv/bin/optimum-cli export onnx \
  --model google/paligemma-3b-pt-224 \
  --task image-to-text \
  --framework pt \
  /tmp/optimum_paligemma_sanity \
  2>&1 | tee /tmp/optimum_export.log
```

**Expected:** KeyError on `image-text-to-text` task OR partial success with components.

Try alternate tasks if image-text-to-text fails:
- `--task text-generation-with-past` (exports Gemma decoder with KV outputs)
- `--task feature-extraction` (raw embeddings)
- No task flag (Optimum guesses)

**Success criterion:** produce at least ONE of: `embed_tokens.onnx`, `vision_encoder.onnx`, `decoder_model_merged.onnx`, `decoder_with_past.onnx`.

### Task 1.4: Alternate path — NSTiwari notebook (1 hour if Optimum blocked)

If Optimum is fully blocked, adapt `NSTiwari/PaliGemma2-ONNX-Transformers.js/Convert_PaliGemma2_to_ONNX.ipynb` locally:

```bash
mkdir -p /tmp/nstiwari_adapt
cd /tmp/nstiwari_adapt
curl -O https://raw.githubusercontent.com/NSTiwari/PaliGemma2-ONNX-Transformers.js/main/Convert_PaliGemma2_to_ONNX.ipynb
# Adapt for PaliGemma 1 (3b-pt-224) vs PaliGemma 2
jupyter nbconvert --to python Convert_PaliGemma2_to_ONNX.ipynb
# Edit, run, see what comes out
```

**Success criterion:** produce the 4-file split manually via the notebook pattern.

### Task 1.5: Inspect outputs with Netron desktop + size check (30 min)

- Netron desktop (not web): https://github.com/lutzroeder/netron/releases
- Inspect vision_encoder.onnx, decoder_model_merged.onnx structure
- Check file sizes — if any single file > 2GB, must use external-data format

**Success criterion:** identify the graph-level structure of each component.

### Task 1.6: Produce PyTorch reference outputs for diff target (1 hour)

Write `scripts/local_paligemma_sanity.py`:
- Load `google/paligemma-3b-pt-224` via transformers
- Run with deterministic dummy input (seeded image + seeded text)
- Dump outputs: vision embed, text embed, final logits
- Store as `.npz` for comparison against Optimum's ONNX

**Success criterion:** `.npz` file with reference outputs, reproducible with seed.

## Day 2 — Polygraphy diff + verdict (~4 hours)

### Task 2.1: Polygraphy end-to-end diff — ORT vs PyTorch (1 hour)

```bash
# If Optimum produced a unified graph:
.venv/bin/polygraphy run /tmp/optimum_paligemma_sanity/model.onnx \
  --onnxrt \
  --input-shapes "pixel_values:[1,3,224,224]" \
  --load-inputs /tmp/sanity_inputs.json \
  --save-outputs /tmp/ort_outputs.json \
  --atol 1e-4 --rtol 1e-4

# Compare vs our PyTorch outputs.npz via custom Python script
.venv/bin/python scripts/local_paligemma_diff.py \
  --ort /tmp/ort_outputs.json \
  --pt /tmp/paligemma_pt_outputs.npz
```

**Success criterion:** Mean absolute difference per output tensor < 1e-4.

### Task 2.2: Polygraphy per-node marking — where does parity break? (1 hour)

```bash
# Mark all intermediate outputs and compare
.venv/bin/polygraphy run /tmp/optimum_paligemma_sanity/model.onnx \
  --onnxrt \
  --onnx-outputs mark all \
  --save-outputs /tmp/all_node_outputs.json
```

Then script a per-node diff vs PyTorch hooks. Identifies which layer / op introduces drift.

**Success criterion:** produce a list of nodes where diff exceeds threshold.

### Task 2.3: TensorRT parse test (if trtexec available, else skip) (30 min)

```bash
# If we have trtexec locally:
trtexec --onnx=/tmp/optimum_paligemma_sanity/model.onnx \
  --saveEngine=/tmp/paligemma.engine \
  --fp16 \
  --verbose 2>&1 | tee /tmp/trt_parse.log

# Or use Polygraphy:
.venv/bin/polygraphy run /tmp/optimum_paligemma_sanity/model.onnx \
  --trt --onnxrt \
  --atol 1e-3 --rtol 1e-3
```

**Success criterion:** TRT parses without errors, engine builds, FP16 numerical diff < 1e-3.
**Failure mode to expect:** FP16 overflow in Gemma attention per the gotcha list.

### Task 2.4: Tacoin oracle cross-check (30 min, pi0.5-specific)

```bash
# If the Tacoin ONNX loads, do a smoke diff against their expected outputs
.venv/bin/polygraphy run /tmp/oracle_pi05_libero_onnx/*.onnx \
  --onnxrt \
  --input-shapes "..."  # per their README
```

Confirms: can we run their MIT-licensed oracle to use as a pi0.5 parity target later? This is not directly about pi0 but unlocks pi0.5 verification faster.

**Success criterion:** Tacoin ONNX produces sensible output shapes.

### Task 2.5: Verdict + revised estimate (1 hour)

Based on Day 1 + 2.1 + 2.2 + 2.3 results, fill in `reflex_context/03_research/pi0_empirical_derisk_findings.md` with:

- Did Optimum export cleanly? If yes, which task flag worked?
- What's the PyTorch vs Optimum ONNX parity (mean abs diff)?
- Did TRT parse the Optimum output? Did FP16 work, or did Gemma attention overflow as predicted?
- What's the first concrete bug we need to fix (if any)?
- Revised mess-up probability after empirical evidence
- Revised ETA for pi0-onnx-parity: {confirmed 2w / pushed to 3w / pivot to Torch-TRT / deeper research needed}

**Decision matrix:**

| Day 2 outcome | Next step |
|---|---|
| ONNX parity < 1e-4, TRT builds, FP16 works | Commit to option A, 2-week implementation |
| ONNX parity OK, TRT fails FP16 | Option A with BF16 path, 2.5-week implementation |
| ONNX parity > 1e-3, root cause localizable | Option A with manual fixes, 3-week implementation |
| ONNX export fully blocked | Pivot to Torch-TensorRT direct path, revised research needed |
| Tacoin oracle unusable | pi0.5 verification deferred; pi0 still viable |

---

## Risk checkpoints

Each task has a fail-fast gate. If stuck > 1 hour on any task:

| Stuck on | First move |
|---|---|
| Polygraphy install (CUDA deps) | Install `polygraphy` alone without `nvidia-pyindex`, skip TRT-side for now |
| Optimum export KeyError | Try `NSTiwari` notebook path (Task 1.4) |
| PaliGemma download too slow | Use smaller proxy: `google/paligemma-1b-pt-224` if exists, or `google/siglip-base` |
| TRT not available locally | Skip Task 2.3 entirely, use onnxruntime as TRT proxy |

## Dependencies + prereqs

- Disk space: ~10 GB free (PaliGemma 3B weights ~7 GB + ONNX output ~6 GB)
- Time: ~8 engineer-hours total across 2 days
- No external services needed (all local on Mac or Modal if preferred)

## Output artifacts

By end of day 2:
1. `/tmp/optimum_paligemma_sanity/` — Optimum ONNX output
2. `/tmp/oracle_pi05_libero_onnx/` — Tacoin MIT oracle
3. `scripts/local_paligemma_sanity.py` — PyTorch reference harness
4. `scripts/local_paligemma_diff.py` — Polygraphy diff harness
5. `reflex_context/03_research/pi0_empirical_derisk_findings.md` — the verdict
6. Revised `reflex_context/mvp_queue.md` pi0-onnx-parity row with empirical ETA
