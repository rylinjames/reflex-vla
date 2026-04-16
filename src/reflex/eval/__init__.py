"""Task-success evaluation plugins for `reflex bench --benchmark`.

Each plugin wraps a simulation benchmark (LIBERO, SimplerEnv, ManiSkill).
Heavy sim dependencies (MuJoCo, robosuite, LIBERO) live behind the `eval` extra:
    pip install 'reflex-vla[eval]'

The plugin framework is deliberately thin — each benchmark is one function
that takes an export_dir and returns a standardized result dict.
"""
from __future__ import annotations

from typing import Any

__all__ = ["run_task_benchmark"]


def run_task_benchmark(
    benchmark: str,
    *,
    export_dir: str,
    episodes_per_task: int = 10,
    device: str = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch to the named benchmark plugin.

    Returns a dict with:
        benchmark: name
        success_rate: 0.0-1.0 (fraction of episodes that succeeded)
        episodes_completed: total episodes run
        per_task: [{task, success_rate, num_episodes}, ...]
        duration_s: wall-clock seconds
    """
    if benchmark.startswith("libero"):
        from reflex.eval.libero import run_libero
        return run_libero(
            suite=benchmark,
            export_dir=export_dir,
            episodes_per_task=episodes_per_task,
            device=device,
            **kwargs,
        )
    if benchmark == "simpler":
        from reflex.eval.simpler import run_simpler
        return run_simpler(export_dir=export_dir, **kwargs)
    if benchmark == "maniskill":
        from reflex.eval.maniskill import run_maniskill
        return run_maniskill(export_dir=export_dir, **kwargs)
    raise ValueError(f"Unknown benchmark: {benchmark!r}")
