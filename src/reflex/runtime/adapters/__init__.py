"""Adapters that expose Reflex's VLA inference to external evaluation harnesses.

Each adapter is a thin wrapper around :class:`reflex.runtime.ReflexServer` —
the real inference (VLM prefix + expert denoising + safety + deadline) lives
in ReflexServer, and adapters only translate between observation/action
schemas. When a bug exists in the denoising loop or VLM wiring, it is fixed
in ReflexServer, not in the adapters.

Available adapters:
    vla_eval — AllenAI's vla-evaluation-harness (LIBERO, SimplerEnv, ManiSkill)
"""
from __future__ import annotations

__all__: list[str] = []
