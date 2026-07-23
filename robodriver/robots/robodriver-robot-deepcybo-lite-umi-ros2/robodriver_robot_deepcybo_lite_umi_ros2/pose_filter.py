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
from scipy.spatial.transform import Rotation


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

        The `FilterOutput` handed back is built from a SEPARATE copy (via
        `_pack`), never the arrays stored in `_last`: a caller that mutates a
        returned `.pos`/`.quat` in place must not be able to corrupt this
        filter's retained state, nor any other output -- including a later
        frozen tick -- that happens to share the same recorded pose.
        """
        p = np.array(pos, dtype=float, copy=True)
        q = np.array(quat, dtype=float, copy=True)
        self._last = (p, q)
        return self._pack(p, q, stale, n_predicted)

    def _pack(self, p: np.ndarray, q: np.ndarray, stale: bool,
              n_predicted: int) -> FilterOutput:
        """Build a `FilterOutput` from fresh copies of `p`/`q`.

        Shared by `_emit` (packaging a just-recorded pose) and the freeze
        branch of `update` (repeating `self._frozen`), so every returned
        `FilterOutput` -- including every tick of a freeze -- owns its own
        arrays. No array object is ever shared between the filter's internal
        state (`_last`/`_frozen`) and a returned output, or between two
        returned outputs.
        """
        return FilterOutput(p.copy(), q.copy(), stale, n_predicted)

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
                return self._emit(p, q, False, 0)
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
            return self._pack(p, q, True, self._n_predicted)

        out_p, out_q = self._on_predict(t)
        return self._emit(out_p, out_q, False, self._n_predicted)


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _OneEuroChannel:
    """Streaming 1e filter over a flat D-vector (Casiez et al.).

    Cutoff rises with speed: smooth hard when slow (jitter is visible), get
    out of the way when fast (lag is what you notice).
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    def reset(self, x0: np.ndarray) -> None:
        self.x_prev = np.array(x0, dtype=float)
        self.dx_prev = np.zeros_like(self.x_prev)

    def update(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dt = max(dt, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class OneEuroPoseFilter(BasePoseFilter):
    """1e filter on position, and on orientation in the tangent space.

    Orientation is filtered as the rotation VECTOR of the delta from the last
    filtered orientation, which stays on the manifold by construction — no
    renormalisation and no hemisphere handling.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.4,
                 d_cutoff: float = 1.0, max_predict_frames: int = 3):
        self._mc, self._beta, self._dc = min_cutoff, beta, d_cutoff
        super().__init__(max_predict_frames=max_predict_frames)

    def reset(self) -> None:
        super().reset()
        self._pos_f = _OneEuroChannel(self._mc, self._beta, self._dc)
        self._rot_f = _OneEuroChannel(self._mc, self._beta, self._dc)
        self._t = None
        self._p = None
        self._R = None
        self._v = np.zeros(3)

    def _on_first(self, t, pos, quat):
        self._t = t
        self._p = np.array(pos, dtype=float)
        self._R = Rotation.from_quat(np.asarray(quat, dtype=float))
        self._pos_f.reset(self._p)
        self._rot_f.reset(np.zeros(3))
        self._v = np.zeros(3)

    def _on_measurement(self, t, pos, quat):
        dt = max(t - self._t, 1e-6)
        p_new = self._pos_f.update(pos, dt)
        self._v = (p_new - self._p) / dt
        self._p, self._t = p_new, t

        R_meas = Rotation.from_quat(np.asarray(quat, dtype=float))
        # `as_rotvec()` returns an angle in [0, pi], so a per-frame rotation
        # exceeding 180 degrees would wrap discontinuously here (branch cut).
        # Unreachable at teleop speeds: >90 deg/frame at 30 fps is >2700 deg/s,
        # far beyond any human hand or the tracker's own measurement range.
        delta = (R_meas * self._R.inv()).as_rotvec()
        delta_f = self._rot_f.update(delta, dt)
        self._R = Rotation.from_rotvec(delta_f) * self._R
        return self._p, self._R.as_quat()

    def _on_predict(self, t):
        dt = max(t - self._t, 0.0)
        return self._p + self._v * dt, self._R.as_quat()


class _CVChannel:
    """Scalar constant-velocity Kalman filter. State [value, rate]."""

    def __init__(self, sigma_meas: float, sigma_accel: float):
        self.r = float(sigma_meas) ** 2
        self.sa2 = float(sigma_accel) ** 2
        self.x = np.zeros(2)
        self.P = np.eye(2)

    def reset(self, x0: float) -> None:
        self.x = np.array([float(x0), 0.0])
        self.P = np.diag([self.r, 1.0])

    def predict(self, dt: float) -> float:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.sa2 * np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                                 [dt ** 3 / 2.0, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[0]

    def update(self, z: float, dt: float) -> float:
        self.predict(dt)
        H = np.array([[1.0, 0.0]])
        S = float(H @ self.P @ H.T) + self.r
        K = (self.P @ H.T / S).ravel()
        y = float(z) - self.x[0]
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        return self.x[0]


class EkfPoseFilter(BasePoseFilter):
    """Constant-velocity KF per position axis and per orientation tangent axis.

    A plain KF on the tangent space rather than a full 6-DoF error-state EKF:
    the tangent-space formulation already keeps orientation on the manifold,
    and the extra machinery would not change the jitter/lag trade this filter
    is judged on.
    """

    def __init__(self, sigma_meas: float = 0.004, sigma_accel: float = 1.0,
                 sigma_meas_rot: float = 0.02, sigma_alpha: float = 5.0,
                 max_predict_frames: int = 3):
        self._sm, self._sa = sigma_meas, sigma_accel
        self._smr, self._salpha = sigma_meas_rot, sigma_alpha
        super().__init__(max_predict_frames=max_predict_frames)

    def reset(self) -> None:
        super().reset()
        self._pos = [_CVChannel(self._sm, self._sa) for _ in range(3)]
        self._rot = [_CVChannel(self._smr, self._salpha) for _ in range(3)]
        self._t = None
        self._R = None

    @property
    def velocity(self) -> np.ndarray:
        """Inferred linear velocity — never measured directly."""
        return np.array([c.x[1] for c in self._pos])

    def _on_first(self, t, pos, quat):
        self._t = t
        for c, v in zip(self._pos, np.asarray(pos, dtype=float)):
            c.reset(v)
        for c in self._rot:
            c.reset(0.0)
        self._R = Rotation.from_quat(np.asarray(quat, dtype=float))

    def _on_measurement(self, t, pos, quat):
        dt = max(t - self._t, 1e-6)
        self._t = t
        p = np.array([c.update(z, dt)
                      for c, z in zip(self._pos, np.asarray(pos, float))])

        R_meas = Rotation.from_quat(np.asarray(quat, dtype=float))
        delta = (R_meas * self._R.inv()).as_rotvec()
        d = np.array([c.update(z, dt) for c, z in zip(self._rot, delta)])
        self._R = Rotation.from_rotvec(d) * self._R
        for c in self._rot:            # the delta is consumed; re-centre
            c.x[0] = 0.0
        return p, self._R.as_quat()

    def _on_predict(self, t):
        dt = max(t - self._t, 0.0)
        p = np.array([c.x[0] + c.x[1] * dt for c in self._pos])
        return p, self._R.as_quat()
