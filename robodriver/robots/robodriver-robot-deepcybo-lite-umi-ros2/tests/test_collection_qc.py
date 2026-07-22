import pytest

from robodriver_robot_deepcybo_lite_umi_ros2.collection_qc import (
    GRIPPER_COL, EpisodeQC, QCThresholds, evaluate_gates, format_qc,
    usable_fraction,
)
from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import ArmCoverage


def _cov(n=300, measured=290, interpolated=10, unfillable=0):
    return ArmCoverage(
        n=n, measured=measured, interpolated=interpolated, unfillable=unfillable,
        filled_gap_hist={}, unfilled_gap_hist={},
        longest_filled_gap_s=0.0, longest_unfilled_gap_s=0.0,
    )


def _kwargs(**over):
    base = dict(
        coverage={"left": _cov(), "right": _cov()},
        raw_tracked_frac={"left": 0.95, "right": 0.95},
        gripper_range={"left": 0.0, "right": 0.4},
        camera_frame_counts={"image_head": 300, "image_wrist_left": 300,
                             "image_wrist_right": 300},
        n_frames=300,
        duration_s=10.0,
    )
    base.update(over)
    return base


def test_gripper_column_layout():
    assert GRIPPER_COL == {"left": 7, "right": 15}


def test_usable_fraction():
    assert usable_fraction(90, 5, 100) == pytest.approx(0.95)
    assert usable_fraction(0, 0, 0) == 0.0          # no divide-by-zero


def test_clean_episode_passes():
    qc = evaluate_gates(**_kwargs())
    assert qc.passed
    assert qc.failures == ()


def test_steadying_gripper_may_be_constant():
    # left steadies the container; its gripper need not move
    qc = evaluate_gates(**_kwargs(gripper_range={"left": 0.0, "right": 0.4}))
    assert qc.passed


def test_picking_gripper_constant_fails():
    # this is the 2026-07-15 stub failure, and a failed grasp
    qc = evaluate_gates(**_kwargs(gripper_range={"left": 0.0, "right": 0.0}))
    assert not qc.passed
    assert [f.name for f in qc.failures] == ["gripper_moved"]


def test_picking_usable_below_bar_fails():
    qc = evaluate_gates(**_kwargs(
        coverage={"left": _cov(), "right": _cov(measured=200, interpolated=80,
                                                unfillable=20)}))
    assert not qc.passed
    names = [f.name for f in qc.failures]
    assert "picking_usable" in names and "picking_unfillable" in names


def test_any_unfillable_on_picking_arm_fails():
    qc = evaluate_gates(**_kwargs(
        coverage={"left": _cov(), "right": _cov(measured=289, interpolated=10,
                                                unfillable=1)}))
    assert not qc.passed
    assert "picking_unfillable" in [f.name for f in qc.failures]


def test_raw_tracked_floor_catches_smoothing_crutch():
    # usable is fine because smoothing recovered it, but raw tracking has rotted
    qc = evaluate_gates(**_kwargs(
        coverage={"left": _cov(), "right": _cov(measured=200, interpolated=100,
                                                unfillable=0)},
        raw_tracked_frac={"left": 0.95, "right": 0.667}))
    assert not qc.passed
    assert [f.name for f in qc.failures] == ["picking_raw_tracked"]


def test_steadying_arm_has_its_own_looser_bar():
    qc = evaluate_gates(**_kwargs(
        coverage={"left": _cov(measured=200, interpolated=60, unfillable=40),
                  "right": _cov()}))
    assert not qc.passed
    assert [f.name for f in qc.failures] == ["steadying_usable"]


def test_missing_camera_stream_fails():
    qc = evaluate_gates(**_kwargs(
        camera_frame_counts={"image_head": 300, "image_wrist_left": 300}))
    assert not qc.passed
    assert "cameras" in [f.name for f in qc.failures]


def test_camera_frame_count_mismatch_fails():
    qc = evaluate_gates(**_kwargs(
        camera_frame_counts={"image_head": 300, "image_wrist_left": 188,
                             "image_wrist_right": 300}))
    assert not qc.passed
    assert "cameras" in [f.name for f in qc.failures]


@pytest.mark.parametrize("dur", [4.9, 20.1])
def test_duration_out_of_range_fails(dur):
    qc = evaluate_gates(**_kwargs(duration_s=dur))
    assert not qc.passed
    assert "duration" in [f.name for f in qc.failures]


def test_picking_arm_can_be_left():
    # roles swapped: left picks, right steadies
    qc = evaluate_gates(**_kwargs(
        gripper_range={"left": 0.0, "right": 0.4}, picking_arm="left"))
    assert not qc.passed
    assert "gripper_moved" in [f.name for f in qc.failures]


def test_thresholds_are_overridable():
    qc = evaluate_gates(**_kwargs(duration_s=1.0),
                        thresholds=QCThresholds(duration_min_s=0.5))
    assert qc.passed


def test_format_qc_shows_pass_and_failures():
    ok = format_qc(evaluate_gates(**_kwargs()), episode_index=7)
    assert "episode_000007" in ok and "PASS" in ok
    bad = format_qc(
        evaluate_gates(**_kwargs(gripper_range={"left": 0.0, "right": 0.0})),
        episode_index=7)
    assert "FAIL" in bad and "gripper_moved" in bad
    assert "REDO" in bad
