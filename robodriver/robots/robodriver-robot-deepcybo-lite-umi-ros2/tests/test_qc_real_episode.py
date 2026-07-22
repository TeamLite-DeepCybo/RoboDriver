# tests/test_qc_real_episode.py
"""The 2026-07-15 recording is known-bad: constant-0.0 grippers, 82.1% right
raw tracked, a 0.70 s left dropout. The checker must reject it for exactly
those reasons — this is the regression guard that the gates work on real data.
"""
import json
from pathlib import Path

import pytest

from robodriver_robot_deepcybo_lite_umi_ros2.collection_qc import GRIPPER_COL
from robodriver_robot_deepcybo_lite_umi_ros2.qc_episode import (
    check_episode, load_episode_inputs,
)

REAL = Path(__file__).resolve().parents[4].parent / "umi_real_rec_2026-07-15"

pytestmark = pytest.mark.skipif(
    not REAL.is_dir(), reason=f"real recording not found at {REAL}"
)


def test_gripper_columns_match_recorded_feature_names():
    """Pin GRIPPER_COL against the real dataset's declared feature names, so a
    layout change upstream fails here instead of silently gating the wrong
    column."""
    info = json.loads((REAL / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"]["observation.state"]["names"]
    assert names[GRIPPER_COL["left"]] == "left_gripper.pos"
    assert names[GRIPPER_COL["right"]] == "right_gripper.pos"


def test_real_episode_is_rejected_for_the_right_reasons():
    idx, qc = check_episode(REAL)
    assert idx == 0
    assert not qc.passed
    names = {f.name for f in qc.failures}
    # constant-0.0 gripper stub
    assert "gripper_moved" in names
    # right arm raw tracked 82.1%, below the 90% floor
    assert "picking_raw_tracked" in names


def test_real_episode_inputs_match_known_values():
    kw = load_episode_inputs(REAL, 0)
    assert kw["n_frames"] == 240
    assert kw["raw_tracked_frac"]["right"] == pytest.approx(197 / 240, abs=1e-3)
    assert kw["raw_tracked_frac"]["left"] == pytest.approx(178 / 240, abs=1e-3)
    assert kw["gripper_range"]["left"] == pytest.approx(0.0, abs=1e-9)
    assert kw["gripper_range"]["right"] == pytest.approx(0.0, abs=1e-9)
    # left arm carries the 0.70 s unfillable dropout
    assert kw["coverage"]["left"].unfillable == 29
    assert kw["coverage"]["right"].unfillable == 0
