"""pi-Flow: 10→2 step velocity-field distillation (arXiv 2510.14974).

Legacy recipe. DMPO supersedes it — see reflex.distill.dmpo.

v0.2 status: scaffold only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["PiFlowConfig", "PiFlowTrainer"]


@dataclass
class PiFlowConfig:
    teacher_export_dir: str = ""
    student_output_dir: str = "./distilled_student"
    dataset: str = "libero_10"
    target_steps: int = 2  # vs teacher's 10


class PiFlowTrainer:
    """pi-Flow distillation (legacy recipe; DMPO is preferred)."""

    def __init__(self, config: PiFlowConfig):
        self.config = config

    def train(self) -> dict[str, Any]:
        raise NotImplementedError(
            "pi-Flow scaffold only. Prefer DMPO (reflex.distill.dmpo) — "
            "it's strictly better per arXiv 2601.20701 analysis."
        )
