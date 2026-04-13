"""Tests for safety guardrails."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from reflex.safety.guard import ActionGuard, SafetyLimits, SafetyCheckResult


class TestSafetyLimits:
    def test_default_creates_6dof(self):
        limits = SafetyLimits.default(6)
        assert len(limits.joint_names) == 6
        assert len(limits.position_min) == 6
        assert len(limits.position_max) == 6

    def test_save_and_load(self, tmp_path):
        limits = SafetyLimits.default(4)
        path = tmp_path / "limits.json"
        limits.save(path)
        loaded = SafetyLimits.from_json(path)
        assert loaded.joint_names == limits.joint_names
        assert loaded.position_min == limits.position_min

    def test_custom_limits(self):
        limits = SafetyLimits(
            joint_names=["j1", "j2"],
            position_min=[-1.0, -2.0],
            position_max=[1.0, 2.0],
            velocity_max=[1.5, 1.5],
            effort_max=[50.0, 50.0],
        )
        assert limits.position_min[0] == -1.0
        assert limits.position_max[1] == 2.0


class TestActionGuard:
    def test_safe_action_passes(self):
        guard = ActionGuard.default(num_joints=3)
        action = np.array([0.1, 0.2, 0.3])
        result = guard.check_single(action)
        assert result.safe
        assert len(result.violations) == 0
        assert not result.clamped

    def test_out_of_bounds_clamped(self):
        limits = SafetyLimits(
            joint_names=["j1", "j2"],
            position_min=[-1.0, -1.0],
            position_max=[1.0, 1.0],
            velocity_max=[2.0, 2.0],
            effort_max=[50.0, 50.0],
        )
        guard = ActionGuard(limits=limits, mode="clamp")
        action = np.array([5.0, -3.0])
        result = guard.check_single(action)
        assert not result.safe
        assert result.clamped
        assert result.safe_action[0] == 1.0
        assert result.safe_action[1] == -1.0

    def test_reject_mode_zeros(self):
        limits = SafetyLimits(
            joint_names=["j1"],
            position_min=[-1.0],
            position_max=[1.0],
            velocity_max=[2.0],
            effort_max=[50.0],
        )
        guard = ActionGuard(limits=limits, mode="reject")
        action = np.array([5.0])
        result = guard.check_single(action)
        assert not result.safe
        assert result.safe_action[0] == 0.0

    def test_batch_check(self):
        guard = ActionGuard.default(num_joints=3)
        actions = np.array([
            [0.1, 0.2, 0.3],
            [10.0, -10.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        safe_actions, results = guard.check(actions)
        assert len(results) == 3
        assert results[0].safe
        assert not results[1].safe
        assert results[2].safe
        assert safe_actions.shape == actions.shape

    def test_inference_count(self):
        guard = ActionGuard.default(num_joints=2)
        actions = np.array([[0.1, 0.2]])
        guard.check(actions)
        guard.check(actions)
        assert guard.inference_count == 2

    def test_logging(self, tmp_path):
        guard = ActionGuard.default(num_joints=2, log_dir=tmp_path, model_version="test-v1")
        actions = np.array([[0.1, 0.2], [5.0, -5.0]])
        guard.check(actions)

        log_files = list(tmp_path.glob("inference_log_*.jsonl"))
        assert len(log_files) == 1

        lines = log_files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["model_version"] == "test-v1"
        assert "timestamp" in entry
        assert len(entry["actions_raw"]) == 2

    def test_check_time_is_fast(self):
        guard = ActionGuard.default(num_joints=6)
        action = np.random.randn(6).astype(np.float32)
        result = guard.check_single(action)
        assert result.check_time_ms < 1.0  # Sub-millisecond
