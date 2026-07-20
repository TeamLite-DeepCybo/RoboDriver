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


from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import (
    ArmCoverage, arm_coverage, regen_action, smooth_state,
)


def _state(n=30, fps=30.0):
    """Full 23-dim state with both arms following distinct trajectories."""
    t = np.arange(n) / fps
    s = np.zeros((n, 23), dtype=np.float32)
    _, lp, lq = _traj(n, fps)
    s[:, 0:3] = lp
    s[:, 3:7] = lq
    s[:, 7] = np.linspace(0, 1, n)                     # L gripper ramp
    s[:, 8:11] = lp[:, [1, 0, 2]] * -1.0               # different R traj
    s[:, 11:15] = Rotation.from_euler(
        "x", 30.0 * t, degrees=True).as_quat()
    s[:, 15] = np.linspace(1, 0, n)                    # R gripper ramp
    s[:, 16] = 1.0; s[:, 17] = 1.0; s[:, 18] = 0.1     # L quality
    s[:, 19] = 1.0; s[:, 20] = 1.0; s[:, 21] = 0.1     # R quality
    s[:, 22] = 1.0
    return t, s


def test_smooth_state_arms_independent():
    t, s = _state()
    s[10:13, 16] = 0.0                 # left dropout only
    s_in = s.copy()
    out, prov = smooth_state(t, s_in, max_gap_s=0.25)
    # right arm bit-exact everywhere
    assert (out[:, 8:15] == s_in[:, 8:15]).all()
    assert (prov[:, 1] == MEASURED).all()
    # left gap interpolated
    assert (prov[10:13, 0] == INTERPOLATED).all()
    assert not (out[10:13, 0:7] == s_in[10:13, 0:7]).all()


def test_smooth_state_passthrough_columns_untouched():
    t, s = _state()
    s[10:13, 16] = 0.0
    s_in = s.copy()
    out, _ = smooth_state(t, s_in, max_gap_s=0.25)
    assert (out[:, 7] == s_in[:, 7]).all()      # L gripper
    assert (out[:, 15] == s_in[:, 15]).all()    # R gripper
    assert (out[:, 16:23] == s_in[:, 16:23]).all()  # quality dims byte-identical


def test_smooth_state_rejects_bad_input():
    t, s = _state()
    with pytest.raises(ValueError):
        smooth_state(t, s[:, :22], max_gap_s=0.25)   # wrong dim
    t2 = t.copy(); t2[5] = t2[4]                      # non-monotonic
    with pytest.raises(ValueError, match="monotonic"):
        smooth_state(t2, s, max_gap_s=0.25)


def test_regen_action_mirrors_first_16():
    t, s = _state()
    out, _ = smooth_state(t, s, max_gap_s=0.25)
    a = regen_action(out)
    assert a.shape == (len(s), 16)
    assert (a == out[:, :16]).all()
    a[0, 0] = 99.0                                    # must be a copy
    assert out[0, 0] != 99.0


def test_arm_coverage_counts_and_histogram():
    t, s = _state()
    s[5:7, 16] = 0.0      # 2-frame gap (fillable)
    s[20:21, 16] = 0.0    # 1-frame gap (fillable)
    anchors = s[:, 16] > 0.5
    _, prov = smooth_state(t, s, max_gap_s=0.25)
    cov = arm_coverage(t, anchors, prov[:, 0])
    assert cov.n == 30
    assert cov.measured == 27
    assert cov.interpolated == 3
    assert cov.unfillable == 0
    assert cov.gap_hist == {2: 1, 1: 1}
    assert cov.longest_gap_s == pytest.approx(t[7] - t[4])
