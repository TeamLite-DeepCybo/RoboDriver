"""Pure composition logic: stamp pairing, world-pose buffer, per-arm hold-last.

No ROS imports so tests/test_compose.py runs anywhere.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np

from . import se3

STAMP_TOLERANCE_NS = 5_000_000  # 5 ms — spec §6


def stamp_to_ns(sec: int, nanosec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nanosec)


class WorldBuffer:
    """Last-N T_world_head poses keyed by stamp (ns), nearest-lookup."""

    def __init__(self, maxlen: int = 30) -> None:
        self._maxlen = maxlen
        self._buf: OrderedDict[int, np.ndarray] = OrderedDict()

    def add(self, stamp_ns: int, T_world_head: np.ndarray) -> None:
        self._buf[int(stamp_ns)] = np.asarray(T_world_head, dtype=np.float64)
        while len(self._buf) > self._maxlen:
            self._buf.popitem(last=False)

    def lookup(
        self, stamp_ns: int, tol_ns: int = STAMP_TOLERANCE_NS
    ) -> np.ndarray | None:
        if not self._buf:
            return None
        stamp_ns = int(stamp_ns)
        exact = self._buf.get(stamp_ns)
        if exact is not None:
            return exact
        best = min(self._buf, key=lambda s: abs(s - stamp_ns))
        if abs(best - stamp_ns) <= tol_ns:
            return self._buf[best]
        return None


REPROJ_INVALID = 999.0


def _finite_or_invalid(v: float) -> float:
    v = float(v)
    return v if np.isfinite(v) else REPROJ_INVALID


@dataclass
class EefState:
    """Per-arm composed eef state with quality flags (spec §5)."""

    pose7: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    valid: bool = False        # a pose has been composed at least once
    tracked: float = 0.0
    present: float = 0.0
    reproj: float = REPROJ_INVALID
    world_fresh: float = 0.0   # 1.0 only when composed fresh this frame


class EefComposer:
    """Hold-last composer for one arm (spec §6–7)."""

    def __init__(self) -> None:
        self._state = EefState()

    def update(
        self,
        stamp_ns: int,
        T_head_tcp: np.ndarray | None,
        *,
        tracked: bool,
        present: bool,
        reproj: float,
        world: WorldBuffer,
    ) -> EefState:
        world_fresh = 0.0
        if present and T_head_tcp is not None:
            T_world_head = world.lookup(stamp_ns)
            if T_world_head is not None:
                T_world_tcp = T_world_head @ np.asarray(T_head_tcp, np.float64)
                pos, quat = se3.T_to_pos_quat(T_world_tcp)
                self._state.pose7 = np.concatenate([pos, quat]).astype(np.float32)
                self._state.valid = True
                world_fresh = 1.0
        # tracking loss or world miss: pose7 held from last success
        self._state.tracked = 1.0 if tracked else 0.0
        self._state.present = 1.0 if present else 0.0
        self._state.reproj = _finite_or_invalid(reproj)
        self._state.world_fresh = world_fresh
        return replace(self._state, pose7=self._state.pose7.copy())


def build_state_vector(
    left: EefState, right: EefState, left_grip: float, right_grip: float
) -> np.ndarray:
    """16-dim [L eepose7, L grip, R eepose7, R grip] (spec §5)."""
    return np.concatenate(
        [
            left.pose7,
            np.array([left_grip], dtype=np.float32),
            right.pose7,
            np.array([right_grip], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_quality_vector(left: EefState, right: EefState) -> np.ndarray:
    """[L_tracked, L_present, L_reproj, R_tracked, R_present, R_reproj, world_fresh]."""
    return np.array(
        [
            left.tracked,
            left.present,
            left.reproj,
            right.tracked,
            right.present,
            right.reproj,
            max(left.world_fresh, right.world_fresh),
        ],
        dtype=np.float32,
    )
