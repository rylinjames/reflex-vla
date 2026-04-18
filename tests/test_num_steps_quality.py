"""Regression marker for `num-steps-quality-gate` GOALS.yaml item.

Characterizes drift between num_steps=1 (what our monolithic ONNX bakes
in) and num_steps=10 (PyTorch canonical). Actual measurement runs on
Modal via the `--quality` entrypoints; receipt at
`reflex_context/num_steps_quality_last_run.json`.

Passes when the last measurement was recorded. We do NOT enforce a
cos threshold here — documenting the drift is the point of the gate,
not constraining it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPT = Path(__file__).parent.parent / "reflex_context" / "num_steps_quality_last_run.json"


def test_num_steps_quality_receipt_present():
    if not RECEIPT.exists():
        pytest.skip(
            "No num_steps quality receipt. Run "
            "`modal run scripts/modal_{pi0,smolvla}_monolithic_export.py "
            f"--quality` and drop results at {RECEIPT}."
        )
    data = json.loads(RECEIPT.read_text())
    for model, entry in data.get("models", {}).items():
        assert "first_cos" in entry
        assert "full_cos" in entry
        assert "relative_max_abs" in entry
