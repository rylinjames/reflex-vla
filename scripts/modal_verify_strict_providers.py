"""Phase I.1 verification: reflex serve now fails loudly on silent CPU fallback.

Three scenarios tested:
  1. `onnxruntime-gpu` installed + CUDA 12 libs present + --device cuda → starts
  2. `onnxruntime-gpu` installed + CUDA 12 libs present + --device cpu → starts
  3. `onnxruntime` (CPU-only) installed + --device cuda (default) → exits with
     a clear error instead of silently running on CPU

Usage:
    modal run scripts/modal_verify_strict_providers.py
"""

import modal

app = modal.App("reflex-strict-providers-verify")

# GPU-capable image with correctly pinned CUDA 12 / cuDNN 9 stack
image_gpu = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",  # torch 2.5.x uses CUDA 12.4
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime-gpu==1.20.1",
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
        "fastapi", "uvicorn", "httpx",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e . --no-deps")
)

# CPU-only image — installs `onnxruntime` (NOT the -gpu variant). Used to
# verify the CLI exits with a helpful error when a user pip-installed the
# wrong package.
image_cpu = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime==1.20.1",  # CPU-only!
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
        "fastapi", "uvicorn", "httpx",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e . --no-deps")
)


def _make_fake_export(export_dir: str):
    """Write a minimal reflex_config.json + empty ONNX to let serve try to start."""
    import json
    import os

    os.makedirs(export_dir, exist_ok=True)
    # Minimal valid ONNX — just the magic bytes + a tiny trivial graph.
    # We don't need it to actually run; we just need `_load_onnx` to be reached.
    import onnx
    from onnx import helper, TensorProto
    input_tensor = helper.make_tensor_value_info(
        "noisy_actions", TensorProto.FLOAT, [1, 50, 32]
    )
    time_tensor = helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1])
    pos_tensor = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, 50])
    output_tensor = helper.make_tensor_value_info(
        "velocity", TensorProto.FLOAT, [1, 50, 32]
    )
    identity = helper.make_node(
        "Identity", inputs=["noisy_actions"], outputs=["velocity"], name="identity"
    )
    graph = helper.make_graph(
        [identity],
        "fake",
        [input_tensor, time_tensor, pos_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.save(model, f"{export_dir}/expert_stack.onnx")
    with open(f"{export_dir}/reflex_config.json", "w") as f:
        json.dump({
            "model_type": "smolvla",
            "action_chunk_size": 50,
            "action_dim": 32,
            "expert": {"action_dim": 32, "expert_hidden": 720},
        }, f)


@app.function(image=image_gpu, gpu="A10G", timeout=300)
def scenario_gpu_ok():
    """onnxruntime-gpu present, CUDA libs present → server should load on CUDA."""
    import subprocess
    _make_fake_export("/tmp/fake_export")

    import onnxruntime as ort
    print(f"ORT version: {ort.__version__}", flush=True)
    print(f"Available providers: {ort.get_available_providers()}", flush=True)

    # Launch server, wait briefly, check health, kill
    import time
    proc = subprocess.Popen(
        ["reflex", "serve", "/tmp/fake_export", "--port", "9001",
         "--host", "127.0.0.1", "--device", "cuda"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(6)
    proc.terminate()
    out, _ = proc.communicate(timeout=5)
    return {
        "scenario": "gpu_ok",
        "exit_code": proc.returncode,
        "stdout_tail": out[-1500:],
        "expected": "server starts successfully on CUDA",
    }


@app.function(image=image_gpu, gpu="A10G", timeout=300)
def scenario_gpu_cpu_flag():
    """--device cpu should start cleanly even on a GPU box."""
    import subprocess
    import time
    _make_fake_export("/tmp/fake_export")

    proc = subprocess.Popen(
        ["reflex", "serve", "/tmp/fake_export", "--port", "9002",
         "--host", "127.0.0.1", "--device", "cpu"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(6)
    proc.terminate()
    out, _ = proc.communicate(timeout=5)
    return {
        "scenario": "gpu_cpu_flag",
        "exit_code": proc.returncode,
        "stdout_tail": out[-1500:],
        "expected": "server starts on CPU without error",
    }


@app.function(image=image_cpu, gpu="A10G", timeout=300)
def scenario_cpu_only_silent_fallback_blocked():
    """CPU-only onnxruntime + --device cuda → should exit non-zero with helpful error."""
    import subprocess
    import time
    _make_fake_export("/tmp/fake_export")

    import onnxruntime as ort
    print(f"ORT version: {ort.__version__}", flush=True)
    print(f"Available providers: {ort.get_available_providers()}", flush=True)
    print(f"Expected: CUDAExecutionProvider NOT in the list.", flush=True)

    # Default device=cuda should now exit with code 1 before even starting
    proc = subprocess.run(
        ["reflex", "serve", "/tmp/fake_export", "--port", "9003",
         "--host", "127.0.0.1"],  # default --device cuda
        capture_output=True, text=True, timeout=20,
    )
    return {
        "scenario": "cpu_only_silent_fallback_blocked",
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
        "expected": "exit code 1 with message about CUDAExecutionProvider not available",
    }


@app.function(image=image_cpu, gpu="A10G", timeout=300)
def scenario_cpu_only_with_no_strict_flag():
    """--no-strict-providers on CPU-only box should allow fallback."""
    import subprocess
    import time
    _make_fake_export("/tmp/fake_export")

    proc = subprocess.Popen(
        ["reflex", "serve", "/tmp/fake_export", "--port", "9004",
         "--host", "127.0.0.1", "--device", "cuda", "--no-strict-providers"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(6)
    proc.terminate()
    out, _ = proc.communicate(timeout=5)
    return {
        "scenario": "cpu_only_with_no_strict_flag",
        "exit_code": proc.returncode,
        "stdout_tail": out[-1500:],
        "expected": "server starts despite CUDA unavailable (best-effort fallback)",
    }


@app.local_entrypoint()
def main():
    print("Phase I.1 verification — reflex serve strict provider mode\n")
    print("=" * 70)

    print("\n[1/4] GPU box + onnxruntime-gpu + --device cuda (should START)")
    r1 = scenario_gpu_ok.remote()
    print(f"  exit={r1['exit_code']}")
    print(f"  expected: {r1['expected']}")

    print("\n[2/4] GPU box + --device cpu (should START)")
    r2 = scenario_gpu_cpu_flag.remote()
    print(f"  exit={r2['exit_code']}")
    print(f"  expected: {r2['expected']}")

    print("\n[3/4] CPU-only ORT + --device cuda default (should EXIT 1)")
    r3 = scenario_cpu_only_silent_fallback_blocked.remote()
    print(f"  exit={r3['exit_code']}")
    print(f"  expected: {r3['expected']}")
    print(f"  stdout tail: {r3['stdout_tail'][-600:]}")

    print("\n[4/4] CPU-only ORT + --device cuda --no-strict-providers (should START)")
    r4 = scenario_cpu_only_with_no_strict_flag.remote()
    print(f"  exit={r4['exit_code']}")
    print(f"  expected: {r4['expected']}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    passes = 0
    total = 4
    # Scenario 1: server ran until SIGTERM — exit code may be -15 or 0 depending on shell
    if r1["exit_code"] in (0, -15, 143, None):
        print(f"  1 PASS: server started cleanly on GPU")
        passes += 1
    else:
        print(f"  1 FAIL: {r1['exit_code']}")
    if r2["exit_code"] in (0, -15, 143, None):
        print(f"  2 PASS: server started cleanly on CPU when requested")
        passes += 1
    else:
        print(f"  2 FAIL: {r2['exit_code']}")
    if r3["exit_code"] == 1:
        print(f"  3 PASS: refused to silently fall back to CPU")
        passes += 1
    else:
        print(f"  3 FAIL (expected exit 1, got {r3['exit_code']})")
    if r4["exit_code"] in (0, -15, 143, None):
        print(f"  4 PASS: --no-strict-providers allowed fallback")
        passes += 1
    else:
        print(f"  4 FAIL: {r4['exit_code']}")

    print(f"\nOverall: {passes}/{total}")
