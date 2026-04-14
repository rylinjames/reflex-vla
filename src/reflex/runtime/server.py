"""Reflex VLA inference server.

Persistent HTTP server that loads an exported VLA model and serves
action predictions. Camera image in, robot actions out.

Usage:
    reflex serve ./reflex_export/ --port 8000

Then from robot:
    POST http://localhost:8000/act
    {
        "image": "<base64 encoded image>",
        "instruction": "pick up the cup",
        "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
    → {"actions": [[...], [...], ...], "latency_ms": 253.1, "hz": 3.9}
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ReflexServer:
    """VLA inference server that loads exported models and serves predictions."""

    def __init__(
        self,
        export_dir: str | Path,
        device: str = "cuda",
        num_denoising_steps: int = 10,
        providers: list[str] | None = None,
        strict_providers: bool = True,
    ):
        """Create the server.

        Args:
            export_dir: directory with exported ONNX + reflex_config.json
            device: "cuda" or "cpu" — selects default ONNX execution provider
            num_denoising_steps: Euler flow matching steps per inference
            providers: explicit list of ORT execution providers to request, e.g.
                ["CUDAExecutionProvider", "CPUExecutionProvider"]. If omitted,
                derived from `device`. Useful for explicit control in production.
            strict_providers: if True (default), raise a loud RuntimeError when
                the requested provider fails to load instead of silently falling
                back to CPU. Set False only if you explicitly want best-effort
                fallback (almost always wrong for GPU deployments).
        """
        self.export_dir = Path(export_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_denoising_steps = num_denoising_steps
        self._requested_device = device
        self._requested_providers = providers
        self._strict_providers = strict_providers
        self.config = self._load_config()
        self.model = None
        self._ready = False

    def _load_config(self) -> dict[str, Any]:
        config_path = self.export_dir / "reflex_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}

    def load(self) -> None:
        """Load the model from exported directory."""
        logger.info("Loading model from %s", self.export_dir)
        start = time.perf_counter()

        expert_meta = self.config.get("expert", {})
        self.action_dim = expert_meta.get("action_dim", 32)
        self.chunk_size = self.config.get("action_chunk_size", 50)
        self.expert_hidden = expert_meta.get("expert_hidden", 720)

        # Try ONNX runtime first, fall back to PyTorch
        onnx_path = self.export_dir / "expert_stack.onnx"
        if onnx_path.exists():
            self._load_onnx(onnx_path)
        else:
            logger.warning("No ONNX model found, inference not available")
            return

        elapsed = time.perf_counter() - start
        self._ready = True
        logger.info("Model loaded in %.1fs, ready to serve", elapsed)

    def _load_onnx(self, onnx_path: Path) -> None:
        """Load ONNX model via onnxruntime.

        Honors `self._requested_device` and `self._requested_providers`. Raises
        if the requested provider fails to load AND `self._strict_providers` is
        set (the default). Silent CPU fallback was causing users to publish
        "GPU" benchmarks that were actually CPU — Apr 14 post-mortem. Never
        again.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is not installed. For GPU inference, install "
                "`onnxruntime-gpu` (not `onnxruntime`). For CPU only, "
                "install `onnxruntime`."
            ) from e

        # Decide which providers to request
        if self._requested_providers is not None:
            providers = list(self._requested_providers)
        elif self._requested_device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # What the installed ORT actually supports on this machine
        available = set(ort.get_available_providers())
        logger.info("Requested providers: %s; available: %s", providers, sorted(available))

        # Create session
        self._ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = self._ort_session.get_providers()
        logger.info("Loaded ONNX model: %s — active providers: %s", onnx_path.name, active)

        # Strict check: if caller asked for CUDA but we ended up on CPU, fail loudly.
        cuda_requested = "CUDAExecutionProvider" in providers
        cuda_active = "CUDAExecutionProvider" in active
        if cuda_requested and not cuda_active and self._strict_providers:
            install_hint = ""
            if "CUDAExecutionProvider" not in available:
                install_hint = (
                    "\n\nCUDAExecutionProvider is not available in this ORT install. "
                    "Likely causes:\n"
                    "  - You installed `onnxruntime` (CPU-only). Replace it with:\n"
                    "      pip uninstall onnxruntime && pip install onnxruntime-gpu\n"
                    "  - CUDA 12 + cuDNN 9 libraries are not on the library path. "
                    "ORT 1.20+ requires CUDA 12.x and cuDNN 9.x. "
                    "See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements\n"
                    "  - You are on a machine without an NVIDIA GPU."
                )
            raise RuntimeError(
                f"reflex serve was started with --device cuda (or CUDAExecutionProvider "
                f"in --providers) but ONNX Runtime fell back to CPU. "
                f"Active providers: {active}. "
                f"Refusing to continue under strict mode — use --no-strict-providers or "
                f"--device cpu to explicitly request CPU execution.{install_hint}"
            )

        self._inference_mode = "onnx_gpu" if cuda_active else "onnx_cpu"

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(
        self,
        image: np.ndarray | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run inference: image + instruction + state → action chunk.

        For v0.1, runs the expert stack only (no VLM prefix encoding).
        VLM integration comes in a future version.

        Args:
            image: RGB image array [H, W, 3] or None
            instruction: text instruction (unused in v0.1 expert-only mode)
            state: robot state vector [N] or None

        Returns:
            dict with "actions" (list of action vectors), "latency_ms", "hz"
        """
        if not self._ready:
            return {"error": "Model not loaded. Call load() first."}

        start = time.perf_counter()

        # Prepare inputs for expert stack denoising
        noisy_actions = np.random.randn(1, self.chunk_size, self.action_dim).astype(np.float32)
        position_ids = np.arange(self.chunk_size, dtype=np.int64).reshape(1, -1)

        # Euler denoise loop
        dt = -1.0 / self.num_denoising_steps
        for step in range(self.num_denoising_steps):
            t = 1.0 + step * dt
            timestep = np.array([t], dtype=np.float32)

            if self._inference_mode.startswith("onnx"):
                velocity = self._ort_session.run(
                    None,
                    {
                        "noisy_actions": noisy_actions,
                        "timestep": timestep,
                        "position_ids": position_ids,
                    },
                )[0]
            else:
                return {"error": f"Unknown inference mode: {self._inference_mode}"}

            noisy_actions = noisy_actions + velocity * dt

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert to list for JSON serialization
        actions = noisy_actions[0].tolist()

        return {
            "actions": actions,
            "num_actions": len(actions),
            "action_dim": self.action_dim,
            "latency_ms": round(elapsed_ms, 1),
            "hz": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
            "denoising_steps": self.num_denoising_steps,
            "inference_mode": self._inference_mode,
        }

    def predict_from_base64(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        """Predict from base64-encoded image (for HTTP API)."""
        image = None
        if image_b64:
            try:
                from PIL import Image

                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = np.array(img)
            except Exception as e:
                return {"error": f"Failed to decode image: {e}"}

        return self.predict(image=image, instruction=instruction, state=state)


try:
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        image: str | None = None  # base64 encoded
        instruction: str = ""
        state: list[float] | None = None

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        inference_mode: str = ""
        export_dir: str = ""

except ImportError:
    PredictRequest = None  # type: ignore
    HealthResponse = None  # type: ignore


def create_app(
    export_dir: str,
    device: str = "cuda",
    providers: list[str] | None = None,
    strict_providers: bool = True,
) -> Any:
    """Create a FastAPI app for serving VLA predictions."""
    try:
        from contextlib import asynccontextmanager
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("Install fastapi: pip install 'reflex-vla[serve]'")

    server = ReflexServer(
        export_dir,
        device=device,
        providers=providers,
        strict_providers=strict_providers,
    )

    @asynccontextmanager
    async def lifespan(app):
        server.load()
        yield

    app = FastAPI(
        title="Reflex VLA Server",
        description="Deploy any VLA model to any edge hardware.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok" if server.ready else "not_ready",
            model_loaded=server.ready,
            inference_mode=getattr(server, "_inference_mode", ""),
            export_dir=str(server.export_dir),
        )

    @app.post("/act")
    async def act(request: PredictRequest):
        result = server.predict_from_base64(
            image_b64=request.image,
            instruction=request.instruction,
            state=request.state,
        )
        return JSONResponse(content=result)

    @app.get("/config")
    async def config():
        return JSONResponse(content=server.config)

    return app
