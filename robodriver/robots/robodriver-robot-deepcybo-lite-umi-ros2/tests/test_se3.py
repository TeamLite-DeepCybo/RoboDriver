import numpy as np
import pytest

from robodriver_robot_deepcybo_lite_umi_ros2 import se3

SQ2 = np.sqrt(2.0) / 2.0


def test_identity_round_trip():
    T = se3.pos_quat_to_T([0, 0, 0], [0, 0, 0, 1])
    assert np.allclose(T, np.eye(4))
    pos, quat = se3.T_to_pos_quat(T)
    assert np.allclose(pos, [0, 0, 0])
    assert np.allclose(quat, [0, 0, 0, 1])


def test_translation_only():
    T = se3.pos_quat_to_T([0.1, -0.2, 0.5], [0, 0, 0, 1])
    assert np.allclose(T[:3, 3], [0.1, -0.2, 0.5])
    assert np.allclose(T[:3, :3], np.eye(3))


def test_90deg_about_z():
    # quat (0,0,sin45,cos45) == +90 deg about z: x-axis -> y-axis
    T = se3.pos_quat_to_T([0, 0, 0], [0, 0, SQ2, SQ2])
    expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(T[:3, :3], expected_R, atol=1e-12)


def test_round_trip_pose():
    pos = [0.3, -0.1, 0.9]
    quat = [0.18257419, 0.36514837, 0.54772256, 0.73029674]  # normalized (1,2,3,4)
    T = se3.pos_quat_to_T(pos, quat)
    pos2, quat2 = se3.T_to_pos_quat(T)
    assert np.allclose(pos2, pos, atol=1e-9)
    assert np.allclose(quat2, quat, atol=1e-9)  # w>0 branch preserved


def test_quat_normalized_on_input():
    T = se3.pos_quat_to_T([0, 0, 0], [0, 0, 2 * SQ2, 2 * SQ2])  # unnormalized
    expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(T[:3, :3], expected_R, atol=1e-12)


def test_compose_world_head_times_head_tcp():
    # T_world_head: +1m z translation, 180 deg about x  (quat (1,0,0,0))
    T_wh = se3.pos_quat_to_T([0.0, 0.0, 1.0], [1, 0, 0, 0])
    # T_head_tcp: pure translation
    T_ht = se3.pos_quat_to_T([0.1, 0.2, 0.5], [0, 0, 0, 1])
    T_wt = T_wh @ T_ht
    pos, quat = se3.T_to_pos_quat(T_wt)
    # R(180x) @ (0.1,0.2,0.5) = (0.1,-0.2,-0.5); + (0,0,1) => (0.1,-0.2,0.5)
    assert np.allclose(pos, [0.1, -0.2, 0.5], atol=1e-12)
    assert np.allclose(np.abs(quat), [1, 0, 0, 0], atol=1e-12)
