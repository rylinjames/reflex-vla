"""Regression marker for `serve-act-roundtrip` GOALS.yaml gate.

Actual roundtrip runs on Modal via `scripts/modal_serve_roundtrip_test.py`
— loads the monolithic ONNX through Pi0OnnxServer (or raw onnxruntime
for SmolVLA) and verifies a predict call returns a valid action chunk.

Receipt: `reflex_context/serve_roundtrip_last_run.json`.

Passes when the last recorded run had every model marked passed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPT = Path(__file__).parent.parent / "reflex_context" / "serve_roundtrip_last_run.json"


def test_serve_roundtrip_receipt_passes():
    if not RECEIPT.exists():
        pytest.skip(
            "No serve-roundtrip receipt. Run "
            "`modal run scripts/modal_serve_roundtrip_test.py` and drop the "
            f"receipt at {RECEIPT}."
        )
    data = json.loads(RECEIPT.read_text())
    for model, entry in data.get("models", {}).items():
        assert entry.get("passed") is True, f"{model} did not pass: {entry}"
