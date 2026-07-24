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

That frame-count cap alone is not enough: the reversal error is `2*v*n/fps`,
which grows without bound as hand speed rises (measured: 2.1 cm vs freezing's
1.2 cm at the rig's 0.124 m/s median, but 16.7 cm vs freezing's 10.0 cm at
1.0 m/s -- still reporting `stale=False`). `max_predict_displacement_m` caps
the predicted DISPLACEMENT itself, so the worst-case reversal error is
bounded at roughly 2x the cap regardless of speed -- see `BasePoseFilter` for
the full rationale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

#: Predict through at most this many consecutive untracked frames before
#: hard-freezing. Defined once here and referenced everywhere else (both
#: filter subclasses, the node's CLI default, and the README) so the
#: safety-critical default is never repeated out of sync with itself.
DEFAULT_MAX_PREDICT_FRAMES = 3

#: Freeze instead of predicting once a predicted pose would travel more than
#: this far (in metres) from the pose that would otherwise be frozen. See
#: `BasePoseFilter` for the rationale.
DEFAULT_MAX_PREDICT_DISPLACEMENT_M = 0.015


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

    `max_predict_displacement_m` caps how far a predicted pose may travel
    from the pose that would otherwise be frozen (the last emitted pose from
    BEFORE the current gap began). Freezing's worst-case error equals how far
    the hand travelled during the gap; prediction's error is that PLUS
    however far the filter predicted -- so capping the predicted displacement
    at D bounds prediction's worst-case reversal error at roughly 2*D
    REGARDLESS OF SPEED, instead of the uncapped `2*v*n/fps` ceiling, which
    grows without bound as hand speed rises. At the rig's median speed
    (~0.124 m/s), `max_predict_frames` (3) frames of prediction cover only
    ~12.4 mm -- under the 15 mm default cap -- so the anti-stutter benefit for
    common single-frame dropouts is preserved; at 1.0 m/s the cap engages
    almost immediately, bounding the error near 3 cm instead of ~20 cm.
    """

    def __init__(self, max_predict_frames: int = DEFAULT_MAX_PREDICT_FRAMES,
                 max_predict_displacement_m: float =
                 DEFAULT_MAX_PREDICT_DISPLACEMENT_M):
        if max_predict_frames < 0:
            raise ValueError(
                f"max_predict_frames must be >= 0, got {max_predict_frames}"
            )
        if max_predict_displacement_m < 0:
            raise ValueError(
                "max_predict_displacement_m must be >= 0, got "
                f"{max_predict_displacement_m}"
            )
        self.max_predict_frames = int(max_predict_frames)
        # 0.0 is valid and means "never predict": every untracked frame
        # freezes immediately, same as max_predict_frames=0.
        self.max_predict_displacement_m = float(max_predict_displacement_m)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._initialized = False
        self._n_predicted = 0
        self._frozen: tuple[np.ndarray, np.ndarray] | None = None
        self._last: tuple[np.ndarray, np.ndarray] | None = None
        # The pose that would be frozen right now, captured at the moment
        # the CURRENT gap began -- i.e. before any predicted output for this
        # gap was emitted. Fixed for the duration of one gap so the
        # displacement cap bounds CUMULATIVE predicted travel, not the
        # frame-to-frame delta.
        self._gap_anchor: tuple[np.ndarray, np.ndarray] | None = None

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
        in_p = np.asarray(pos, dtype=float)
        in_q = np.asarray(quat, dtype=float)
        # A NaN/inf in a "tracked" measurement must never enter the filter
        # state: every filter here is linear, so one non-finite value would
        # poison every subsequent output forever, with `stale` staying False
        # and no reset path from the node. Route it through the exact same
        # gap/predict/freeze policy as a genuinely untracked frame instead --
        # it can then never be fused by `_on_first`/`_on_measurement`.
        usable = tracked and bool(
            np.all(np.isfinite(in_p)) and np.all(np.isfinite(in_q)))

        if usable:
            if not self._initialized:
                # Verbatim adoption: a warm-up ramp would command the arm to
                # drift from an arbitrary origin toward the true pose.
                self._on_first(t, in_p, in_q)
                self._initialized = True
                self._n_predicted = 0
                self._frozen = None
                self._gap_anchor = None
                return self._emit(in_p, in_q, False, 0)
            out_p, out_q = self._on_measurement(t, in_p, in_q)
            self._n_predicted = 0
            self._frozen = None
            self._gap_anchor = None
            return self._emit(out_p, out_q, False, 0)

        if not self._initialized:
            return FilterOutput(None, None, True, 0)

        if self._n_predicted == 0:
            # Entering a new gap: anchor the displacement cap on the pose
            # that would be frozen right now (see `_gap_anchor` above).
            self._gap_anchor = self._last
        self._n_predicted += 1

        over_frame_budget = self._n_predicted > self.max_predict_frames
        if over_frame_budget or self.max_predict_displacement_m <= 0.0:
            # `max_predict_displacement_m == 0` means "never predict":
            # freeze immediately without ever calling `_on_predict`, exactly
            # like an immediate frame-budget breach.
            exceeded = True
        else:
            cand_p, cand_q = self._on_predict(t)
            anchor_p, _ = self._gap_anchor
            displacement = float(
                np.linalg.norm(np.asarray(cand_p, dtype=float) - anchor_p))
            exceeded = displacement > self.max_predict_displacement_m

        if exceeded:
            if self._frozen is None:
                # Freeze on the LAST value emitted while still within
                # budget, then repeat it bit-identically: the arm holds
                # still instead of creeping.
                self._frozen = self._last
            p, q = self._frozen
            return self._pack(p, q, True, self._n_predicted)

        return self._emit(cand_p, cand_q, False, self._n_predicted)


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

    `beta=1.0` was selected by `filter_bench`'s own parameter sweep: at
    `min_cutoff=1.0`, `beta=1.0` gives the same 0.44 mm jitter as
    `beta=0.4` (the prior default) but at 79 ms of lag instead of 106 ms, at
    equal (0 %) overshoot -- strictly better on every axis the benchmark
    measures, so shipping `beta=0.4` was leaving a free improvement on the
    table.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 1.0,
                 d_cutoff: float = 1.0,
                 max_predict_frames: int = DEFAULT_MAX_PREDICT_FRAMES,
                 max_predict_displacement_m: float =
                 DEFAULT_MAX_PREDICT_DISPLACEMENT_M):
        self._mc, self._beta, self._dc = min_cutoff, beta, d_cutoff
        super().__init__(max_predict_frames=max_predict_frames,
                          max_predict_displacement_m=max_predict_displacement_m)

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
        S = (H @ self.P @ H.T).item() + self.r
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
                 max_predict_frames: int = DEFAULT_MAX_PREDICT_FRAMES,
                 max_predict_displacement_m: float =
                 DEFAULT_MAX_PREDICT_DISPLACEMENT_M):
        self._sm, self._sa = sigma_meas, sigma_accel
        self._smr, self._salpha = sigma_meas_rot, sigma_alpha
        super().__init__(max_predict_frames=max_predict_frames,
                          max_predict_displacement_m=max_predict_displacement_m)

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
        # `as_rotvec()` returns an angle in [0, pi], so a per-frame rotation
        # exceeding 180 degrees would wrap discontinuously here (branch cut).
        # Unreachable at teleop speeds: >90 deg/frame at 30 fps is >2700 deg/s,
        # far beyond any human hand or the tracker's own measurement range.
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
