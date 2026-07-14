"""Minimal SE(3) helpers (pure numpy, no ROS imports).

Quaternions are (x, y, z, w). `T_to_pos_quat` returns w >= 0 so round trips
are stable (q and -q are the same rotation).
"""
from __future__ import annotations

import numpy as np


def pos_quat_to_T(pos, quat_xyzw) -> np.ndarray:
    """Position (3,) + quaternion xyzw (4,) -> 4x4 homogeneous transform."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = q / n
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def T_to_pos_quat(T) -> tuple[np.ndarray, np.ndarray]:
    """4x4 transform -> (pos (3,), quat xyzw (4,) with w >= 0)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    if quat[3] < 0.0:
        quat = -quat
    quat /= np.linalg.norm(quat)
    return T[:3, 3].copy(), quat
