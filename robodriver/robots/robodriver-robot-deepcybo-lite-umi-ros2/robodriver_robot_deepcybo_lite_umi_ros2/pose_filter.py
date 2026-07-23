"""Causal pose filters for teleop (spec 2026-07-22).

Smooths the LIVE ArUco gripper pose stream so it can drive a robot arm.
Pure maths — stdlib + numpy + scipy only, no ROS and no I/O, so the filter is
validated against recorded data with zero hardware.

The measured need is JITTER rejection (2.67/3.15 mm median on the real rig),
not prediction through dropouts: extrapolating a constant velocity across an
occlusion during which the operator reversed direction produced 17.8 cm of
error in simulation, roughly 2x worse than simply freezing. The gap policy
below therefore predicts at most `max_predict_frames` and then hard-freezes,
making that regime unreachable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FilterOutput:
    """One filter step.

    `pos`/`quat` are None ONLY before the first tracked measurement, so a
    consumer that ignores `stale` fails loudly rather than driving an arm to a
    default pose.
    """
    pos: np.ndarray | None
    quat: np.ndarray | None
    stale: bool
    n_predicted: int


@runtime_checkable
class PoseFilter(Protocol):
    def update(self, t: float, pos, quat, tracked: bool) -> FilterOutput: ...
    def reset(self) -> None: ...


class BasePoseFilter(ABC):
    """Owns initialization and the shared gap/freeze policy.

    Subclasses supply only their smoothing maths via the three hooks. Keeping
    the policy here means both filters are governed by ONE implementation of
    the safety-critical behaviour.
    """

    def __init__(self, max_predict_frames: int = 3):
        if max_predict_frames < 0:
            raise ValueError(
                f"max_predict_frames must be >= 0, got {max_predict_frames}"
            )
        self.max_predict_frames = int(max_predict_frames)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._initialized = False
        self._n_predicted = 0
        self._frozen: tuple[np.ndarray, np.ndarray] | None = None
        self._last: tuple[np.ndarray, np.ndarray] | None = None

    def _emit(self, pos: np.ndarray, quat: np.ndarray, stale: bool,
              n_predicted: int) -> FilterOutput:
        """Record and return one output.

        `_last` holds the exact arrays most recently emitted, so freezing can
        repeat them bit-identically rather than recomputing (which would drift).

        Uses a genuine copy (not `np.asarray`, which returns the SAME object
        when the input is already a float64 ndarray) so `_last`/`_frozen` can
        never alias a subclass's mutable internal state -- a subclass hook
        that mutates its state array in place (e.g. a Kalman filter) must not
        be able to retroactively change an already-frozen, already-emitted
        pose.
        """
        p = np.array(pos, dtype=float, copy=True)
        q = np.array(quat, dtype=float, copy=True)
        self._last = (p, q)
        return FilterOutput(p, q, stale, n_predicted)

    @property
    def initialized(self) -> bool:
        return self._initialized

    # -- subclass hooks ----------------------------------------------------
    @abstractmethod
    def _on_first(self, t: float, pos: np.ndarray, quat: np.ndarray) -> None:
        """Adopt the first measurement as initial state."""

    @abstractmethod
    def _on_measurement(self, t: float, pos: np.ndarray, quat: np.ndarray):
        """Fuse a measurement; return the filtered (pos, quat)."""

    @abstractmethod
    def _on_predict(self, t: float):
        """No measurement available; return the predicted (pos, quat)."""

    # -- the policy --------------------------------------------------------
    def update(self, t: float, pos, quat, tracked: bool) -> FilterOutput:
        if tracked:
            p = np.asarray(pos, dtype=float)
            q = np.asarray(quat, dtype=float)
            if not self._initialized:
                # Verbatim adoption: a warm-up ramp would command the arm to
                # drift from an arbitrary origin toward the true pose.
                self._on_first(t, p, q)
                self._initialized = True
                self._n_predicted = 0
                self._frozen = None
                return self._emit(p.copy(), q.copy(), False, 0)
            out_p, out_q = self._on_measurement(t, p, q)
            self._n_predicted = 0
            self._frozen = None
            return self._emit(out_p, out_q, False, 0)

        if not self._initialized:
            return FilterOutput(None, None, True, 0)

        self._n_predicted += 1
        if self._n_predicted > self.max_predict_frames:
            if self._frozen is None:
                # Freeze on the LAST value emitted while still within budget,
                # then repeat it bit-identically: the arm holds still instead
                # of creeping.
                self._frozen = self._last
            p, q = self._frozen
            return FilterOutput(p, q, True, self._n_predicted)

        out_p, out_q = self._on_predict(t)
        return self._emit(out_p, out_q, False, self._n_predicted)
