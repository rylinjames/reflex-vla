"""Round-trip validation orchestrator for `reflex validate`.

This module defines :class:`ValidateRoundTrip`, the orchestrator that loads an
export directory, drives identical (image, prompt, state) fixtures through both
the exported ONNX graph and a PyTorch reference, then aggregates per-fixture
numerical differences into a single pass / fail decision.

Reference semantics (v1)
------------------------
"PyTorch reference" here means the *exporter's decomposed PyTorch model* — the
same architecture used to produce the ONNX export. This proves ONNX export
correctness, which is the primary goal of `reflex validate`. Comparing against
the true upstream reference (e.g., LeRobot's `SmolVLAPolicy` end-to-end) is a
separate v2 concern and is intentionally out of scope here.

Seed bridge
-----------
`torch.manual_seed()` does not seed numpy. To guarantee that the PyTorch and
ONNX paths consume *byte-identical* initial noise, this orchestrator generates
the noise tensor exactly once via a seeded `torch.Generator`, converts it to a
numpy array, and hands the same numpy array to both backends. The backends are
responsible for re-wrapping it as needed for their runtime.

Exit code convention
--------------------
The CLI handler that wraps this class translates `run()`'s ``status`` field
into the following process exit codes:

* ``0`` — pass (every fixture's max abs diff < threshold)
* ``1`` — fail (at least one fixture exceeds threshold)
* ``2`` — error (missing ONNX, malformed export dir, unsupported model type)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reflex.checkpoint import detect_model_type, load_checkpoint
from reflex.fixtures.vla_fixtures import load_fixtures
from reflex.validate import ValidationResult, validate_outputs

# Backends from Issues 4/5 — required at import time now.
from reflex._pytorch_backend import load_pytorch_backend
from reflex._onnx_backend import load_onnx_backend

# Per-model action_dim fallback when reflex_config.json doesn't specify it.
_ACTION_DIM_FALLBACK: dict[str, int] = {
    "smolvla": 6,
    "pi0": 14,
    "gr00t": 64,
}

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES: tuple[str, ...] = ("smolvla", "pi0", "gr00t")
UNSUPPORTED_MODEL_MESSAGE = (
    "reflex validate v1 supports smolvla, pi0, gr00t. "
    "For pi0.5 / openvla see roadmap."
)


class ValidateRoundTrip:
    """Orchestrate ONNX-vs-PyTorch round-trip validation for an export directory.

    The class loads the export config, builds both backends, generates seeded
    fixtures + initial noise, runs every fixture through both runtimes, and
    aggregates per-fixture L2/abs-diff statistics into a single result dict.

    Parameters
    ----------
    export_dir:
        Path to a `reflex export` output directory containing
        ``reflex_config.json`` plus the ONNX graph(s).
    model_id:
        Optional HuggingFace ID (e.g., ``lerobot/pi0_base``) for the PyTorch
        reference checkpoint. If ``None``, the value from ``reflex_config.json``
        is used.
    threshold:
        Per-element max-abs-diff threshold (default ``1e-4``). A fixture
        passes when ``max_abs_diff < threshold``.
    num_test_cases:
        Number of seeded fixtures to evaluate (default ``5``).
    seed:
        RNG seed shared by fixture generation and the initial-noise bridge
        (default ``0``).
    device:
        Device for the PyTorch reference (default ``"cpu"``). pi0 and GR00T are
        3B+ parameter models — keep CPU unless explicitly overridden.

    Raises
    ------
    FileNotFoundError
        If ``export_dir`` does not exist.
    ValueError
        If ``reflex_config.json`` is missing, or the detected model type is not
        in :data:`SUPPORTED_MODEL_TYPES`.
    """

    def __init__(
        self,
        export_dir: Path,
        model_id: str | None = None,
        threshold: float = 1e-4,
        num_test_cases: int = 5,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.export_dir: Path = Path(export_dir)
        self.model_id: str | None = model_id
        self.threshold: float = float(threshold)
        self.num_test_cases: int = int(num_test_cases)
        self.seed: int = int(seed)
        self.device: str = device

        if not self.export_dir.exists():
            raise FileNotFoundError(
                f"Export directory does not exist: {self.export_dir}"
            )
        if not self.export_dir.is_dir():
            raise ValueError(
                f"Export path is not a directory: {self.export_dir}"
            )

        config_path = self.export_dir / "reflex_config.json"
        if not config_path.exists():
            raise ValueError(
                f"Missing reflex_config.json in export dir: {self.export_dir}"
            )
        self.config: dict[str, Any] = json.loads(config_path.read_text())

        model_type = self.config.get("model_type")
        if not model_type:
            logger.debug("model_type missing from reflex_config.json; detecting from checkpoint")
            ref_id = self.model_id or self.config.get("model_id")
            if ref_id is None:
                raise ValueError(
                    "Cannot detect model_type: reflex_config.json has no "
                    "model_type or model_id field, and --model was not passed."
                )
            state_dict, _ = load_checkpoint(ref_id, device="cpu")
            model_type = detect_model_type(state_dict)

        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(UNSUPPORTED_MODEL_MESSAGE)

        self.model_type: str = model_type
        if self.model_id is None:
            self.model_id = self.config.get("model_id")

        logger.info(
            "ValidateRoundTrip ready: export_dir=%s model_type=%s threshold=%s "
            "num_test_cases=%d seed=%d device=%s",
            self.export_dir,
            self.model_type,
            self.threshold,
            self.num_test_cases,
            self.seed,
            self.device,
        )

    # ------------------------------------------------------------------ public

    def run(self) -> dict[str, Any]:
        """Execute the full round-trip validation and return a result dict.

        Loads both backends, generates ``num_test_cases`` seeded fixtures plus
        their shared initial-noise tensors, runs each fixture through both
        runtimes, compares outputs, and aggregates the results.
        """
        self.pytorch = self._load_pytorch()
        self.onnx = self._load_onnx()

        fixtures = load_fixtures(self.model_type, self.num_test_cases, self.seed)
        rng = torch.Generator(device="cpu").manual_seed(self.seed)

        per_fixture: list[dict[str, Any]] = []
        for idx, (image, prompt, state) in enumerate(fixtures):
            noise = self._generate_initial_noise(rng)
            pt_out = self.pytorch.forward(image, prompt, state, noise)
            onnx_out = self.onnx.forward(image, prompt, state, noise)
            result = self._compare(pt_out, onnx_out)
            result["fixture_idx"] = idx
            per_fixture.append(result)

        return self._aggregate(per_fixture)

    # ----------------------------------------------------------------- private

    def _load_pytorch(self) -> Any:
        """Build and return the PyTorch reference backend for this export."""
        return load_pytorch_backend(self.export_dir, self.model_id, self.device)

    def _load_onnx(self) -> Any:
        """Build and return the ONNX runtime backend for this export."""
        return load_onnx_backend(self.export_dir, self.device)

    def _generate_initial_noise(self, rng: torch.Generator) -> np.ndarray:
        """Generate the initial-noise tensor shared by both backends."""
        chunk_size = int(
            self.config.get("action_chunk_size")
            or self.config.get("chunk_size")
            or 50
        )
        action_dim = int(
            self.config.get("action_dim")
            or _ACTION_DIM_FALLBACK.get(self.model_type, 6)
        )
        noise = torch.randn((chunk_size, action_dim), generator=rng).numpy().astype(np.float32)
        return noise

    def _compare(
        self,
        pytorch_out: np.ndarray,
        onnx_out: np.ndarray,
    ) -> dict[str, Any]:
        """Compare a single fixture's PyTorch and ONNX outputs."""
        result = validate_outputs(
            pytorch_out, onnx_out, threshold=self.threshold, name="roundtrip"
        )
        return result.to_dict()

    def _aggregate(
        self,
        per_fixture_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate per-fixture results into a single run summary."""
        if per_fixture_results:
            max_abs = max(float(r["max_abs_diff"]) for r in per_fixture_results)
        else:
            max_abs = 0.0
        passed = all(bool(r["passed"]) for r in per_fixture_results) if per_fixture_results else False

        return {
            "model_type": self.model_type,
            "threshold": self.threshold,
            "num_test_cases": self.num_test_cases,
            "seed": self.seed,
            "results": per_fixture_results,
            "summary": {
                "max_abs_diff_across_all": max_abs,
                "passed": passed,
            },
        }


__all__ = [
    "SUPPORTED_MODEL_TYPES",
    "UNSUPPORTED_MODEL_MESSAGE",
    "ValidateRoundTrip",
    "ValidationResult",
    "validate_outputs",
]
