"""Regression marker for the `fresh-install-verified` GOALS.yaml item.

The actual fresh-install verification runs on Modal via
`scripts/modal_fresh_install_test.py` (clean container + pip install
from GitHub + CLI smoke). This test documents the receipt: when a
successful Modal run completes, drop a JSON receipt at
`reflex_context/fresh_install_last_run.json` and this test passes.

Until that receipt exists we skip — keeps CI green on clones without
Modal access.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPT = Path(__file__).parent.parent / "reflex_context" / "fresh_install_last_run.json"


def test_fresh_install_receipt_exists():
    if not RECEIPT.exists():
        pytest.skip(
            "No fresh-install receipt. Run "
            "`modal run scripts/modal_fresh_install_test.py` and drop the "
            f"result dict as JSON at {RECEIPT}."
        )
    data = json.loads(RECEIPT.read_text())
    assert data.get("passed") is True, f"Last fresh-install run did not pass: {data}"
