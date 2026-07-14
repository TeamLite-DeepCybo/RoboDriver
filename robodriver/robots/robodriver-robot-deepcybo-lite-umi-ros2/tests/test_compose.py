# tests/test_compose.py
import numpy as np
import pytest

from robodriver_robot_deepcybo_lite_umi_ros2 import se3
from robodriver_robot_deepcybo_lite_umi_ros2.compose import (
    STAMP_TOLERANCE_NS,
    WorldBuffer,
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
