"""Regression marker for `cuda-runtime-verified` GOALS.yaml item.

Actual GPU parity runs on Modal via the `parity_cuda_{pi0,smolvla}`
Modal functions in `scripts/modal_{pi0,smolvla}_monolithic_export.py`.
Receipt is written to `reflex_context/cuda_runtime_last_run.json`.

Passes when the last recorded run had cos>=0.999 and actually used
CUDAExecutionProvider (not silent CPU fallback).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPT = Path(__file__).parent.parent / "reflex_context" / "cuda_runtime_last_run.json"


def test_cuda_runtime_receipt_passes():
    if not RECEIPT.exists():
        pytest.skip(
            "No CUDA runtime receipt. Run the `--cuda` entrypoints on "
            "the Modal export scripts and drop results at "
            f"{RECEIPT}."
        )
    data = json.loads(RECEIPT.read_text())
    for model, entry in data.get("models", {}).items():
        assert entry.get("cos", 0) >= 0.999, f"{model} cos below threshold: {entry}"
        assert entry.get("used_provider") == "CUDAExecutionProvider", (
            f"{model} fell back to CPU: {entry}"
        )
