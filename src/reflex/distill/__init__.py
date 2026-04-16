"""VLA distillation (DMPO one-step generation recipe).

Full training pipeline ships in v0.2.1+. This module scaffolds the
architectural surface so `reflex distill` is wired into the CLI and the
recipe-loading pattern is established.

DMPO (arXiv 2601.20701): One-step MeanFlow Policy with dispersive
regularization. Strictly better than pi-Flow for robot action distillation
because it eliminates the teacher-model dependency while achieving
single-step inference at 1770 Hz on consumer GPUs.

Pipeline:
    reflex distill <teacher_export> --output <student_export>
        -> train DMPO student on LIBERO/DROID trajectories
        -> export student to ONNX (reuses existing exporters)
        -> validate student vs teacher (uses existing validate harness)
"""
from __future__ import annotations

__all__ = ["get_recipe"]


def get_recipe(name: str):
    """Load a distillation recipe by name.

    Available recipes:
        dmpo - MeanFlow one-step distillation (arXiv 2601.20701)
        pi_flow - 10→2 step velocity-field matching (arXiv 2510.14974)
    """
    if name == "dmpo":
        from reflex.distill.dmpo import DMPOTrainer
        return DMPOTrainer
    if name == "pi_flow":
        from reflex.distill.pi_flow import PiFlowTrainer
        return PiFlowTrainer
    raise ValueError(f"Unknown distillation recipe: {name!r}. Try 'dmpo' or 'pi_flow'.")
