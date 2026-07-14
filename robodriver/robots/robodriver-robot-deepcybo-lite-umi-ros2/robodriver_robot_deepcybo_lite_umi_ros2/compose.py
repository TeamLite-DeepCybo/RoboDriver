"""Pure composition logic: stamp pairing, world-pose buffer, per-arm hold-last.

No ROS imports so tests/test_compose.py runs anywhere.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

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
