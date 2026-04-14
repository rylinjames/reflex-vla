"""Verify `reflex bench` works for all 4 supported VLAs with auto-TRT FP16.

The smolvla install-path test confirmed auto-TRT works for that model. This
script extends to pi0, pi0.5, gr00t — bigger models with longer engine builds
that are more likely to surface edge cases.

For each model:
  reflex export <hf_id> --target desktop
  reflex bench <export_dir> --iterations 50 --warmup 10
Capture inference_mode and per-chunk latency from the bench output.

Usage:
    modal run scripts/modal_verify_bench_all.py
"""

import modal

app = modal.App("reflex-bench-all-verify")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt:24.10-py3",
        add_python="3.12",
    )
    .apt_install("git")
)


@app.function(image=image, gpu="A10G", timeout=3600)
def run_all():
    import os
    import re
    import subprocess
    import time

    # Install reflex from git (clean install path — same as users get)
    print("=== Installing reflex-vla[serve,gpu] from git ===", flush=True)
    r = subprocess.run(
        ["pip", "install",
         "reflex-vla[serve,gpu] @ git+https://github.com/rylinjames/reflex-vla"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        return {"error": "pip install failed", "stderr": r.stderr[-1000:]}
    print("  install ok", flush=True)

    models = [
        ("smolvla", "lerobot/smolvla_base"),
        ("pi0", "lerobot/pi0_base"),
        ("pi05", "lerobot/pi05_base"),
        ("gr00t", "nvidia/GR00T-N1.6-3B"),
    ]

    results = {}
    for tag, hf_id in models:
        print(f"\n{'='*60}\n{tag} ({hf_id})\n{'='*60}", flush=True)
        export_dir = f"/tmp/{tag}"

        # Export
        t0 = time.time()
        r = subprocess.run(
            ["reflex", "export", hf_id, "--target", "desktop", "--output", export_dir],
            capture_output=True, text=True, timeout=900,
        )
        export_s = time.time() - t0
        if r.returncode != 0:
            results[tag] = {"export_error": r.stderr[-500:]}
            print(f"  EXPORT FAIL ({export_s:.1f}s): {r.stderr[-300:]}", flush=True)
            continue
        files = os.listdir(export_dir)
        print(f"  export ok ({export_s:.1f}s, files={files})", flush=True)

        # Bench
        t0 = time.time()
        r = subprocess.run(
            ["reflex", "bench", export_dir, "--iterations", "50", "--warmup", "10",
             "--device", "cuda"],
            capture_output=True, text=True, timeout=600,
        )
        bench_s = time.time() - t0
        if r.returncode != 0:
            results[tag] = {
                "export_s": round(export_s, 1),
                "bench_error": r.stderr[-500:] or r.stdout[-500:],
            }
            print(f"  BENCH FAIL ({bench_s:.1f}s): {r.stdout[-400:]}", flush=True)
            continue

        # Parse the bench output. The CLI prints lines like:
        #   mean    11.52 ms
        #   p50     11.50 ms
        #   p95     11.85 ms
        #   p99     12.01 ms
        #   hz       86.8
        #   Inference mode: onnx_trt_fp16
        out = r.stdout
        def _parse(label):
            m = re.search(rf"^\s*{re.escape(label)}\s+([\d.]+)\s*ms?", out, re.MULTILINE)
            return float(m.group(1)) if m else None

        mode_match = re.search(r"Inference mode:\s*(\S+)", out)
        mode = mode_match.group(1) if mode_match else "?"

        results[tag] = {
            "export_s": round(export_s, 1),
            "bench_s": round(bench_s, 1),
            "mean_ms": _parse("mean"),
            "p50_ms": _parse("p50"),
            "p95_ms": _parse("p95"),
            "p99_ms": _parse("p99"),
            "min_ms": _parse("min"),
            "inference_mode": mode,
        }
        print(f"  bench ok ({bench_s:.1f}s): mean={results[tag]['mean_ms']}ms "
              f"p95={results[tag]['p95_ms']}ms mode={mode}", flush=True)

        # Free disk between models — checkpoints are huge
        subprocess.run(["rm", "-rf", export_dir], check=False)

    print(f"\n{'='*80}", flush=True)
    print("VERDICT — `reflex bench` works for all 4 VLAs", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<10} {'mean_ms':>10} {'p95_ms':>10} {'mode':>20} {'export_s':>10}", flush=True)
    for tag, r in results.items():
        if "export_error" in r:
            print(f"{tag:<10} EXPORT FAILED", flush=True)
        elif "bench_error" in r:
            print(f"{tag:<10} BENCH FAILED ({r.get('export_s', 0)}s export)", flush=True)
        else:
            mean = r.get("mean_ms", "—")
            p95 = r.get("p95_ms", "—")
            mode = r.get("inference_mode", "?")
            exp = f"{r.get('export_s', 0)}s"
            print(f"{tag:<10} {str(mean):>10} {str(p95):>10} {mode:>20} {exp:>10}", flush=True)
    return results


@app.local_entrypoint()
def main():
    print("Verifying `reflex bench` works for all 4 VLAs (auto-TRT)\n")
    r = run_all.remote()
    import json
    print("\n=== JSON ===")
    print(json.dumps(r, indent=2, default=str))
