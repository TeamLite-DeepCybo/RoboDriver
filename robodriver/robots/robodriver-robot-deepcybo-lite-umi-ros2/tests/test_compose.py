# tests/test_compose.py
import numpy as np
import pytest

from robodriver_robot_deepcybo_lite_umi_ros2 import se3
from robodriver_robot_deepcybo_lite_umi_ros2.compose import (
    REPROJ_INVALID,
    STAMP_TOLERANCE_NS,
    EefComposer,
    WorldBuffer,
    build_quality_vector,
    build_state_vector,
    stamp_to_ns,
)


def test_stamp_to_ns():
    assert stamp_to_ns(0, 0) == 0
    assert stamp_to_ns(1, 500_000_000) == 1_500_000_000
    assert stamp_to_ns(1700000000, 123) == 1_700_000_000_000_000_123


def test_world_buffer_exact_hit():
    buf = WorldBuffer()
    T = se3.pos_quat_to_T([0, 0, 1], [1, 0, 0, 0])
    buf.add(1_000_000_000, T)
    got = buf.lookup(1_000_000_000)
    assert got is not None
    assert np.allclose(got, T)


def test_world_buffer_nearest_within_tolerance():
    buf = WorldBuffer()
    T = se3.pos_quat_to_T([0, 0, 1], [0, 0, 0, 1])
    buf.add(1_000_000_000, T)
    assert buf.lookup(1_000_000_000 + STAMP_TOLERANCE_NS) is not None
    assert buf.lookup(1_000_000_000 - STAMP_TOLERANCE_NS) is not None


def test_world_buffer_miss_beyond_tolerance():
    buf = WorldBuffer()
    buf.add(1_000_000_000, np.eye(4))
    assert buf.lookup(1_000_000_000 + STAMP_TOLERANCE_NS + 1) is None
    assert buf.lookup(2_000_000_000) is None


def test_world_buffer_picks_nearest_of_several():
    buf = WorldBuffer()
    T_near = se3.pos_quat_to_T([1, 0, 0], [0, 0, 0, 1])
    buf.add(1_000_000_000, np.eye(4))
    buf.add(1_000_004_000, T_near)  # 4 us closer to query
    got = buf.lookup(1_000_003_000)
    assert np.allclose(got, T_near)


def test_world_buffer_evicts_oldest():
    buf = WorldBuffer(maxlen=3)
    for i in range(5):
        buf.add(i * 1_000_000_000, np.eye(4))
    assert buf.lookup(0) is None          # evicted
    assert buf.lookup(1_000_000_000) is None  # evicted
    assert buf.lookup(4_000_000_000) is not None


T_WH = se3.pos_quat_to_T([0.0, 0.0, 1.0], [1, 0, 0, 0])  # 180deg about x, 1m up
T_HT = se3.pos_quat_to_T([0.1, 0.2, 0.5], [0, 0, 0, 1])  # pure translation
STAMP = 1_000_000_000


def _fresh_world() -> WorldBuffer:
    buf = WorldBuffer()
    buf.add(STAMP, T_WH)
    return buf


def test_composer_fresh_compose():
    comp = EefComposer()
    st = comp.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                     world=_fresh_world())
    assert st.valid
    assert st.tracked == 1.0 and st.present == 1.0 and st.world_fresh == 1.0
    assert st.reproj == pytest.approx(0.4)
    assert np.allclose(st.pose7[:3], [0.1, -0.2, 0.5], atol=1e-6)
    assert np.allclose(np.abs(st.pose7[3:]), [1, 0, 0, 0], atol=1e-6)


def test_composer_holds_pose_on_tracking_loss():
    comp = EefComposer()
    comp.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                world=_fresh_world())
    held = comp.update(STAMP + 33_000_000, None, tracked=False, present=False,
                       reproj=float("inf"), world=_fresh_world())
    assert held.valid                       # still have a usable pose
    assert held.tracked == 0.0 and held.present == 0.0
    assert held.world_fresh == 0.0          # no fresh compose this frame
    assert held.reproj == REPROJ_INVALID    # inf clamped
    assert np.allclose(held.pose7[:3], [0.1, -0.2, 0.5], atol=1e-6)  # held


def test_composer_world_miss_holds_and_flags():
    comp = EefComposer()
    comp.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                world=_fresh_world())
    empty = WorldBuffer()  # no world pose at all
    st = comp.update(STAMP + 33_000_000, T_HT, tracked=True, present=True,
                     reproj=0.5, world=empty)
    assert st.tracked == 1.0 and st.present == 1.0
    assert st.world_fresh == 0.0
    assert np.allclose(st.pose7[:3], [0.1, -0.2, 0.5], atol=1e-6)  # held


def test_composer_recovers_after_dropout():
    comp = EefComposer()
    comp.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                world=_fresh_world())
    comp.update(STAMP + 1, None, tracked=False, present=False,
                reproj=float("nan"), world=_fresh_world())
    buf = WorldBuffer()
    buf.add(STAMP + 2, T_WH)
    T_HT2 = se3.pos_quat_to_T([0.2, 0.0, 0.5], [0, 0, 0, 1])
    st = comp.update(STAMP + 2, T_HT2, tracked=True, present=True, reproj=0.3,
                     world=buf)
    assert st.world_fresh == 1.0
    assert np.allclose(st.pose7[:3], [0.2, 0.0, 0.5], atol=1e-6)


def test_composer_invalid_before_first_compose():
    comp = EefComposer()
    st = comp.update(STAMP, None, tracked=False, present=False, reproj=0.0,
                     world=_fresh_world())
    assert not st.valid


def test_state_vector_ordering():
    left, right = EefComposer(), EefComposer()
    ls = left.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                     world=_fresh_world())
    T_HT_R = se3.pos_quat_to_T([-0.1, 0.2, 0.5], [0, 0, 0, 1])
    rs = right.update(STAMP, T_HT_R, tracked=True, present=True, reproj=0.6,
                      world=_fresh_world())
    vec = build_state_vector(ls, rs, left_grip=0.25, right_grip=0.75)
    assert vec.shape == (16,) and vec.dtype == np.float32
    assert np.allclose(vec[0:7], ls.pose7)
    assert vec[7] == pytest.approx(0.25)
    assert np.allclose(vec[8:15], rs.pose7)
    assert vec[15] == pytest.approx(0.75)


def test_quality_vector_ordering_and_world_fresh_max():
    left, right = EefComposer(), EefComposer()
    ls = left.update(STAMP, T_HT, tracked=True, present=True, reproj=0.4,
                     world=_fresh_world())
    rs = right.update(STAMP, None, tracked=False, present=False, reproj=2.0,
                      world=_fresh_world())
    q = build_quality_vector(ls, rs)
    assert q.shape == (7,) and q.dtype == np.float32
    assert list(q[:3]) == [1.0, 1.0, pytest.approx(0.4)]
    assert list(q[3:6]) == [0.0, 0.0, pytest.approx(2.0)]
    assert q[6] == 1.0  # left composed fresh => world was available
