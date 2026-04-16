"""Trajectory replay smoke test for Reflex VLA on Modal.

Downloads episodes from a LeRobot dataset using LeRobot's native loader
(handles video-encoded images), feeds each observation frame through a
reflex serve endpoint, and compares predicted actions to expert ground truth.

Usage:
    modal run scripts/modal_trajectory_replay.py
"""

import json
import sys
import time

import modal

app = modal.App("reflex-trajectory-replay")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch",
        "safetensors",
        "huggingface_hub",
        "transformers>=4.51",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "httpx>=0.24.0",
        "typer",
        "rich",
        "pyyaml",
        "lerobot @ git+https://github.com/huggingface/lerobot.git",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e .")
)

L2_THRESHOLD = 2.0
NUM_EPISODES = 5


@app.function(gpu="A10G", image=image, timeout=1800, scaledown_window=60)
def run_trajectory_replay():
    """Full trajectory replay: export -> serve -> replay episodes -> report."""
    import base64
    import io
    import subprocess
    import threading

    import httpx
    import numpy as np
    from PIL import Image

    results = {
        "steps": [],
        "episodes": [],
        "pass": False,
        "mean_l2": None,
        "max_l2": None,
    }

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL" if status == "fail" else "INFO"
        print(f"{tag}: {name} -- {detail}")

    # ── Step 1: Export SmolVLA ─────────────────────────────────────
    print("\n=== Step 1: Export SmolVLA ===")
    export_dir = "/tmp/reflex_traj_export"
    t0 = time.time()
    r = subprocess.run(
        ["reflex", "export", "lerobot/smolvla_base",
         "--target", "desktop", "--output", export_dir, "--verbose"],
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0
    if r.returncode == 0:
        log("export", "pass", f"SmolVLA exported in {elapsed:.1f}s")
    else:
        log("export", "fail", (r.stdout + r.stderr)[-500:])
        return results

    from pathlib import Path
    files = list(Path(export_dir).iterdir())
    log("export_files", "pass", f"{len(files)} files: {[f.name for f in files]}")

    # ── Step 2: Start reflex serve ─────────────────────────────────
    print("\n=== Step 2: Start reflex serve ===")
    serve_port = 8321
    serve_proc = subprocess.Popen(
        ["reflex", "serve", export_dir, "--port", str(serve_port),
         "--device", "cpu", "--no-strict-providers"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    server_log_lines = []
    def _drain():
        for line in serve_proc.stdout:
            server_log_lines.append(line.rstrip())
    threading.Thread(target=_drain, daemon=True).start()

    server_ready = False
    for attempt in range(60):
        time.sleep(2)
        try:
            resp = httpx.get(f"http://127.0.0.1:{serve_port}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("model_loaded"):
                server_ready = True
                break
        except Exception:
            pass

    if server_ready:
        log("serve_ready", "pass", f"Server ready after {(attempt+1)*2}s")
    else:
        for line in server_log_lines[-20:]:
            print(f"  [server] {line}")
        log("serve_ready", "fail", "Server not ready in 120s")
        serve_proc.terminate()
        return results

    # ── Step 3: Load dataset via LeRobot native loader ─────────────
    print("\n=== Step 3: Load dataset via LeRobotDataset ===")
    episodes = {}
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset_name = "lerobot/pusht"
        print(f"  Loading {dataset_name}...")
        lr_dataset = LeRobotDataset(dataset_name)
        print(f"  Total frames: {len(lr_dataset)}")

        sample0 = lr_dataset[0]
        print(f"  Sample keys: {list(sample0.keys())}")
        for k, v in sample0.items():
            info = f" shape={v.shape} dtype={v.dtype}" if hasattr(v, 'shape') else ""
            print(f"    {k}: {type(v).__name__}{info}")

        # Collect frames
        frames_seen = 0
        max_frames = 500

        for idx in range(min(len(lr_dataset), max_frames)):
            sample = lr_dataset[idx]
            ep_idx = sample.get("episode_index", 0)
            if hasattr(ep_idx, 'item'):
                ep_idx = ep_idx.item()

            if len(episodes) >= NUM_EPISODES and ep_idx not in episodes:
                continue
            if ep_idx not in episodes:
                episodes[ep_idx] = {"frames": [], "actions": []}

            # Find image
            img = None
            for img_key in ["observation.images.top", "observation.image",
                            "observation.images.front", "observation.images.wrist"]:
                if img_key in sample and sample[img_key] is not None:
                    img = sample[img_key]
                    break
            if img is None:
                for k, v in sample.items():
                    if "image" in k.lower() and v is not None:
                        img = v
                        break

            # Find action
            action = sample.get("action", None)
            if action is None:
                continue
            if hasattr(action, "tolist"):
                action = action.tolist()

            episodes[ep_idx]["frames"].append(img)
            episodes[ep_idx]["actions"].append(action)
            frames_seen += 1

        log("dataset", "pass", f"Loaded {dataset_name}: {len(episodes)} episodes, {frames_seen} frames")

    except Exception as e:
        log("dataset", "fail", str(e)[:500])
        serve_proc.terminate()
        return results

    if not episodes:
        log("replay", "fail", "No episodes collected")
        serve_proc.terminate()
        return results

    # Check if we actually got images
    first_ep = list(episodes.values())[0]
    first_frame = first_ep["frames"][0] if first_ep["frames"] else None
    if first_frame is None:
        log("replay", "fail", "Frames are all None — dataset has no image observations")
        serve_proc.terminate()
        return results

    print(f"  First frame type: {type(first_frame).__name__}")
    if hasattr(first_frame, 'shape'):
        print(f"  First frame shape: {first_frame.shape}")

    # ── Step 4: Replay episodes through /act ───────────────────────
    print("\n=== Step 4: Replay through /act ===")
    all_l2_errors = []
    instruction = "push the T block"

    for ep_idx in sorted(episodes.keys())[:NUM_EPISODES]:
        ep = episodes[ep_idx]
        ep_l2_errors = []
        num_frames = len(ep["frames"])
        step_indices = list(range(0, num_frames, max(1, num_frames // 10)))[:10]

        for step_i in step_indices:
            img = ep["frames"][step_i]
            expert_action = np.array(ep["actions"][step_i], dtype=np.float32)

            # Convert to PIL
            try:
                if isinstance(img, Image.Image):
                    pil_img = img
                elif isinstance(img, np.ndarray):
                    if img.dtype in (np.float32, np.float64):
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                elif hasattr(img, 'numpy'):
                    arr = img.numpy()
                    if arr.ndim == 3 and arr.shape[0] in (1, 3):
                        arr = arr.transpose(1, 2, 0)
                    if arr.dtype in (np.float32, np.float64):
                        arr = (arr * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr)
                elif isinstance(img, dict) and "bytes" in img:
                    pil_img = Image.open(io.BytesIO(img["bytes"]))
                else:
                    continue
            except Exception:
                continue

            buf = io.BytesIO()
            pil_img.convert("RGB").resize((224, 224)).save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            try:
                resp = httpx.post(
                    f"http://127.0.0.1:{serve_port}/act",
                    json={"image": img_b64, "instruction": instruction, "state": [0.0] * 6},
                    timeout=30,
                )
                result = resp.json()
                if "error" in result:
                    print(f"  /act error ep={ep_idx} step={step_i}: {result['error']}")
                    continue

                pred_actions = np.array(result["actions"], dtype=np.float32)
                pred_first = pred_actions[0]
                min_dim = min(len(pred_first), len(expert_action))
                l2 = float(np.linalg.norm(pred_first[:min_dim] - expert_action[:min_dim]))
                ep_l2_errors.append(l2)
            except Exception as e:
                print(f"  /act failed ep={ep_idx} step={step_i}: {e}")

        if ep_l2_errors:
            ep_mean = float(np.mean(ep_l2_errors))
            ep_max = float(np.max(ep_l2_errors))
            results["episodes"].append({
                "episode_index": int(ep_idx),
                "num_steps": len(ep_l2_errors),
                "mean_l2": round(ep_mean, 4),
                "max_l2": round(ep_max, 4),
            })
            all_l2_errors.extend(ep_l2_errors)
            print(f"  Episode {ep_idx}: {len(ep_l2_errors)} steps, mean_L2={ep_mean:.4f}, max_L2={ep_max:.4f}")

    # ── Step 5: Report ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAJECTORY REPLAY RESULTS")
    print("=" * 60)

    if all_l2_errors:
        mean_l2 = float(np.mean(all_l2_errors))
        max_l2 = float(np.max(all_l2_errors))
        results["mean_l2"] = round(mean_l2, 4)
        results["max_l2"] = round(max_l2, 4)

        print(f"\n{'Episode':<12} {'Steps':<8} {'Mean L2':<12} {'Max L2':<12}")
        print("-" * 44)
        for ep in results["episodes"]:
            print(f"{ep['episode_index']:<12} {ep['num_steps']:<8} {ep['mean_l2']:<12.4f} {ep['max_l2']:<12.4f}")
        print("-" * 44)
        print(f"{'TOTAL':<12} {len(all_l2_errors):<8} {mean_l2:<12.4f} {max_l2:<12.4f}")

        if mean_l2 < L2_THRESHOLD:
            print(f"\nRESULT: PASS (mean L2 {mean_l2:.4f} < {L2_THRESHOLD})")
            results["pass"] = True
            log("trajectory_replay", "pass", f"mean_L2={mean_l2:.4f}")
        else:
            print(f"\nRESULT: FAIL (mean L2 {mean_l2:.4f} >= {L2_THRESHOLD})")
            log("trajectory_replay", "fail", f"mean_L2={mean_l2:.4f}")
    else:
        print("No frames replayed successfully")
        log("trajectory_replay", "fail", "No frames replayed")

    serve_proc.terminate()
    serve_proc.wait(timeout=10)
    return results


@app.local_entrypoint()
def main():
    print("Starting trajectory replay on Modal (A10G)...")
    result = run_trajectory_replay.remote()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    if result.get("pass"):
        print("\nTRAJECTORY REPLAY: PASS")
        sys.exit(0)
    else:
        print("\nTRAJECTORY REPLAY: FAIL")
        sys.exit(1)
