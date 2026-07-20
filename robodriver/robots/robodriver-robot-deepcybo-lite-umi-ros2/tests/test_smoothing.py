import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import (
    ARM_LAYOUT, INTERPOLATED, MEASURED, UNFILLABLE,
    bracketed_runs, smooth_arm,
)


def _traj(n=30, fps=30.0):
    """Smooth known trajectory: line in pos, constant-rate z-rotation."""
    t = np.arange(n) / fps
    pos = np.stack([t * 0.3, np.sin(t), np.full(n, 0.5)], axis=1)
    quat = Rotation.from_euler("z", 60.0 * t, degrees=True).as_quat()
    return t, pos.astype(np.float64), quat.astype(np.float64)


def test_layout_matches_feature_name_contract():
    L, R = ARM_LAYOUT["left"], ARM_LAYOUT["right"]
    assert (L.pos, L.quat, L.tracked) == (slice(0, 3), slice(3, 7), 16)
    assert (R.pos, R.quat, R.tracked) == (slice(8, 11), slice(11, 15), 19)


def test_bracketed_runs_finds_interior_gaps_only():
    #        0  1  2  3  4  5  6
    anchors = np.array([0, 1, 0, 0, 1, 1, 0], dtype=bool)
    assert list(bracketed_runs(anchors)) == [(1, 4)]  # leading 0 / trailing 6 excluded


def test_interpolation_recovers_knocked_out_frames():
    t, pos, quat = _traj()
    anchors = np.ones(len(t), dtype=bool)
    anchors[10:14] = False              # 4-frame gap, anchors t[9]..t[14] = 0.166s
    p_in, q_in = pos.copy(), quat.copy()
    p_in[10:14] = p_in[9]               # simulate hold-last corruption
    q_in[10:14] = q_in[9]
    p_out, q_out, prov = smooth_arm(t, p_in, q_in, anchors, max_gap_s=0.25)
    np.testing.assert_allclose(p_out[10:14], pos[10:14], atol=5e-3)
    for k in range(10, 14):             # orientation within 1 degree of truth
        err = (Rotation.from_quat(q_out[k]) * Rotation.from_quat(quat[k]).inv()).magnitude()
        assert np.degrees(err) < 1.0
    assert (prov[10:14] == INTERPOLATED).all()


def test_anchor_frames_bit_exact():
    t, pos, quat = _traj()
    anchors = np.ones(len(t), dtype=bool)
    anchors[5] = False
    p_out, q_out, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=0.25)
    assert (p_out[anchors] == pos[anchors]).all()
    assert (q_out[anchors] == quat[anchors]).all()
    assert (prov[anchors] == MEASURED).all()


def test_over_long_gap_left_unfillable():
    t, pos, quat = _traj()
    anchors = np.ones(len(t), dtype=bool)
    anchors[5:20] = False               # anchors t[4]..t[20] = 0.533s > 0.25
    p_out, q_out, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=0.25)
    assert (prov[5:20] == UNFILLABLE).all()
    assert (p_out[5:20] == pos[5:20]).all()   # held input passes through


def test_gap_exactly_at_limit_is_filled():
    t, pos, quat = _traj(fps=30.0)
    anchors = np.ones(len(t), dtype=bool)
    # anchors at 9 and 16: span 7/30 s ≈ 0.2333; use max_gap_s exactly equal
    anchors[10:16] = False
    span = t[16] - t[9]
    _, _, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=span)
    assert (prov[10:16] == INTERPOLATED).all()


def test_leading_and_trailing_gaps_unfillable():
    t, pos, quat = _traj()
    anchors = np.ones(len(t), dtype=bool)
    anchors[:3] = False
    anchors[-2:] = False
    _, _, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=10.0)
    assert (prov[:3] == UNFILLABLE).all()
    assert (prov[-2:] == UNFILLABLE).all()


def test_fewer_than_two_anchors_all_unfillable_but_one():
    t, pos, quat = _traj(n=5)
    anchors = np.zeros(5, dtype=bool)
    anchors[2] = True
    _, _, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=1.0)
    assert prov[2] == MEASURED
    assert (prov[[0, 1, 3, 4]] == UNFILLABLE).all()


def test_sign_flipped_anchor_takes_short_arc():
    # 40 deg apart; second anchor hemisphere-flipped (same rotation)
    q0 = Rotation.from_euler("z", 0, degrees=True).as_quat()
    q1 = -Rotation.from_euler("z", 40, degrees=True).as_quat()
    t = np.array([0.0, 0.05, 0.1])
    pos = np.zeros((3, 3))
    quat = np.stack([q0, q0, q1])       # middle frame is a gap
    anchors = np.array([True, False, True])
    _, q_out, _ = smooth_arm(t, pos, quat, anchors, max_gap_s=0.25)
    mid = Rotation.from_quat(q_out[1])
    err = (mid * Rotation.from_euler("z", 20, degrees=True).inv()).magnitude()
    assert np.degrees(err) < 1e-6       # 20 deg = short arc midpoint
    assert abs(np.linalg.norm(q_out[1]) - 1.0) < 1e-9
