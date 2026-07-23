# Online Pose Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A streaming pose filter that smooths the live ArUco gripper pose stream for teleop, with One-Euro and EKF implementations compared on real recorded data to decide which to keep.

**Architecture:** Three layers. `pose_filter.py` is pure math (stdlib+numpy+scipy, no ROS, no I/O) holding a shared gap/freeze policy plus two filter implementations behind one protocol. `filter_bench.py` replays a recorded dataset through any filter and reports jitter/lag/overshoot. `filter_node.py` is a thin ROS shell that subscribes to live tracker topics and republishes filtered poses.

**Tech Stack:** Python ≥3.10, numpy, scipy (`Rotation`), pandas/pyarrow (bench only), rclpy (node only), pytest.

**Spec:** `docs/superpowers/specs/2026-07-22-online-pose-filter-design.md` (read it first).

## Global Constraints

- `pose_filter.py` imports ONLY stdlib + numpy + scipy. Never `rclpy`, `pandas`, `pyarrow`, `robodriver.*`, `lerobot*`, `torch`.
- The filter is **causal**: `update()` may use only the current and past samples. Never index forward.
- Gap policy, shared by both filters: `tracked` → update; gap ≤ `max_predict_frames` → predict, `stale=False`; gap > `max_predict_frames` → freeze, `stale=True`.
- `max_predict_frames` is a constructor parameter, **default 3**.
- Frozen output is **bit-identical** for the whole freeze — no drift.
- Before the first tracked measurement: `FilterOutput(pos=None, quat=None, stale=True, n_predicted=0)`.
- The first tracked measurement is adopted **verbatim** — output equals input exactly, no warm-up ramp.
- Orientation is filtered in the tangent space (delta rotation → rotvec → filter → re-apply). Never component-wise on quaternion elements.
- Output quaternions must be unit-norm to 1e-9.
- Quaternions are scipy order `(x, y, z, w)`.
- The recorded dataset used by the bench is **read-only**; never modify it.
- State column layout in `observation.state` (fixed): left pos `0:3`, left quat `3:7`, left gripper `7`, right pos `8:11`, right quat `11:15`, right gripper `15`, `left_tracked` `16`, `right_tracked` `19`.
- Working directory for pytest: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2`. Commit from the repo root.
- Commit subjects use Conventional Commits prefixes (`feat:`, `fix:`, `docs:`, `test:`). No co-author or AI-attribution trailer.

---

### Task 1: Filter protocol, output type, and the shared gap/freeze policy

Build the scaffolding both filters share: the output dataclass, the protocol, and a base class owning initialization, the predict/freeze decision, and the frame counter. The two concrete filters (Tasks 2–3) supply only their own smoothing maths.

**Files:**
- Create: `robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py`
- Test: `tests/test_pose_filter.py`

**Interfaces:**
- Produces:
  - `FilterOutput` frozen dataclass: `pos: np.ndarray | None`, `quat: np.ndarray | None`, `stale: bool`, `n_predicted: int`
  - `BasePoseFilter` abstract base: `__init__(max_predict_frames: int = 3)`; `update(t: float, pos, quat, tracked: bool) -> FilterOutput`; `reset() -> None`; `initialized` property
  - Subclass hooks a filter must implement: `_on_first(t, pos, quat) -> None`, `_on_measurement(t, pos, quat) -> tuple[np.ndarray, np.ndarray]`, `_on_predict(t) -> tuple[np.ndarray, np.ndarray]`
  - `PoseFilter` — a `typing.Protocol` with `update` / `reset`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pose_filter.py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robodriver_robot_deepcybo_lite_umi_ros2.pose_filter import (
    BasePoseFilter, FilterOutput,
)

IDENT = np.array([0.0, 0.0, 0.0, 1.0])


class PassThrough(BasePoseFilter):
    """Minimal concrete filter: no smoothing, constant-velocity predict.

    Exists to test the BASE class policy in isolation from any smoothing maths.
    """

    def _on_first(self, t, pos, quat):
        self._p = np.array(pos, float)
        self._q = np.array(quat, float)
        self._t = t
        self._v = np.zeros(3)

    def _on_measurement(self, t, pos, quat):
        dt = max(t - self._t, 1e-9)
        self._v = (np.asarray(pos, float) - self._p) / dt
        self._p = np.array(pos, float)
        self._q = np.array(quat, float)
        self._t = t
        return self._p, self._q

    def _on_predict(self, t):
        dt = max(t - self._t, 0.0)
        return self._p + self._v * dt, self._q


def _feed(f, n, tracked=True, fps=30.0, start=0):
    """Feed n frames of a simple x-ramp; return the last FilterOutput."""
    out = None
    for k in range(start, start + n):
        out = f.update(k / fps, [0.01 * k, 0.0, 0.0], IDENT, tracked)
    return out


def test_uninitialized_reports_none_and_stale():
    f = PassThrough()
    assert not f.initialized
    out = f.update(0.0, [1.0, 2.0, 3.0], IDENT, tracked=False)
    assert out.pos is None and out.quat is None
    assert out.stale is True
    assert out.n_predicted == 0
    assert not f.initialized


def test_first_measurement_adopted_verbatim():
    f = PassThrough()
    p = np.array([0.11, -0.22, 0.33])
    q = Rotation.from_euler("z", 30, degrees=True).as_quat()
    out = f.update(0.0, p, q, tracked=True)
    # exact, not approximate: no warm-up ramp
    assert (out.pos == p).all()
    assert (out.quat == q).all()
    assert out.stale is False
    assert f.initialized


def test_stays_uninitialized_through_leading_untracked_run():
    f = PassThrough()
    for k in range(5):
        out = f.update(k / 30.0, [1.0, 0.0, 0.0], IDENT, tracked=False)
        assert out.pos is None and out.stale is True
    out = f.update(5 / 30.0, [7.0, 8.0, 9.0], IDENT, tracked=True)
    assert (out.pos == np.array([7.0, 8.0, 9.0])).all()
    assert out.stale is False


@pytest.mark.parametrize("gap", [1, 2, 3])
def test_short_gap_predicts_and_is_not_stale(gap):
    f = PassThrough()
    _feed(f, 5)
    out = None
    for i in range(gap):
        out = f.update((5 + i) / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
    assert out.stale is False
    assert out.n_predicted == gap
    assert out.pos is not None


def test_gap_beyond_limit_freezes():
    f = PassThrough()
    _feed(f, 5)
    outs = [f.update((5 + i) / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
            for i in range(4)]
    assert outs[2].stale is False          # 3rd predicted frame still ok
    assert outs[3].stale is True           # 4th crosses the limit
    assert outs[3].n_predicted == 4


def test_frozen_output_is_bit_identical_no_drift():
    f = PassThrough()
    _feed(f, 5)
    outs = [f.update((5 + i) / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
            for i in range(20)]
    frozen = [o for o in outs if o.stale]
    assert len(frozen) >= 15
    first = frozen[0].pos
    for o in frozen[1:]:
        assert (o.pos == first).all()      # exact: the arm must hold still
        assert (o.quat == frozen[0].quat).all()


def test_max_predict_frames_is_honoured():
    f = PassThrough(max_predict_frames=1)
    _feed(f, 5)
    o1 = f.update(5 / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
    o2 = f.update(6 / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
    assert o1.stale is False
    assert o2.stale is True


def test_zero_max_predict_freezes_immediately():
    f = PassThrough(max_predict_frames=0)
    _feed(f, 5)
    out = f.update(5 / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
    assert out.stale is True


def test_counter_resets_after_reacquisition():
    f = PassThrough()
    _feed(f, 5)
    f.update(5 / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
    out = f.update(6 / 30.0, [0.06, 0.0, 0.0], IDENT, tracked=True)
    assert out.n_predicted == 0
    assert out.stale is False


def test_recovers_after_long_freeze_no_lockout():
    f = PassThrough()
    _feed(f, 5)
    for i in range(30):
        f.update((5 + i) / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
    # a measurement far from the frozen pose must be accepted, not rejected
    out = f.update(40 / 30.0, [5.0, 0.0, 0.0], IDENT, tracked=True)
    assert out.stale is False
    assert out.pos[0] == pytest.approx(5.0, abs=1e-9)


def test_reset_returns_to_uninitialized():
    f = PassThrough()
    _feed(f, 5)
    assert f.initialized
    f.reset()
    assert not f.initialized
    out = f.update(1.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
    assert out.pos is None and out.stale is True


def test_determinism():
    a, b = PassThrough(), PassThrough()
    ra = [_feed(a, 1, start=k) for k in range(20)]
    rb = [_feed(b, 1, start=k) for k in range(20)]
    for x, y in zip(ra, rb):
        assert (x.pos == y.pos).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pose_filter.py -v`
Expected: FAIL at import — `ModuleNotFoundError: ... pose_filter`

- [ ] **Step 3: Write the implementation**

```python
# robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py
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
        """
        p = np.asarray(pos, dtype=float)
        q = np.asarray(quat, dtype=float)
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
```

The freeze branch reuses `self._last` — the exact arrays previously emitted —
rather than recomputing a prediction, which is what makes the frozen output
bit-identical instead of drifting.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pose_filter.py -v`
Expected: all PASS

- [ ] **Step 5: Verify the import constraint**

Run: `python -c "import sys; from robodriver_robot_deepcybo_lite_umi_ros2 import pose_filter; print('rclpy:', 'rclpy' in sys.modules, '| pandas:', 'pandas' in sys.modules)"`
Expected: `rclpy: False | pandas: False`

- [ ] **Step 6: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_pose_filter.py
git commit -m "feat: add pose filter base with shared gap and freeze policy"
```

---

### Task 2: One-Euro filter

**Files:**
- Modify: `robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py` (append)
- Test: `tests/test_pose_filter.py` (append)

**Interfaces:**
- Consumes: `BasePoseFilter`, `FilterOutput` (Task 1).
- Produces: `OneEuroPoseFilter(min_cutoff: float = 1.0, beta: float = 0.4, d_cutoff: float = 1.0, max_predict_frames: int = 3)`

The 1€ filter adapts its cutoff to speed: heavy smoothing when slow (where jitter is visible), light when fast (where lag is what you notice). That is exactly the teleop trade. Maths ported from the archived batch implementation at `_archive/aruco_umi/trajectory.py`, restructured as a streaming update.

- [ ] **Step 1: Append the failing tests**

```python
# append to tests/test_pose_filter.py
from robodriver_robot_deepcybo_lite_umi_ros2.pose_filter import OneEuroPoseFilter


def _ramp(f, n=60, fps=30.0, v=0.3, noise=0.0, seed=0):
    """Feed a constant-velocity x-ramp with optional noise; return positions."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        t = k / fps
        p = np.array([v * t, 0.0, 0.0])
        if noise:
            p = p + rng.normal(0, noise, 3)
        out.append(f.update(t, p, IDENT, True).pos)
    return np.array(out)


def test_one_euro_tracks_ramp_slope_with_bounded_offset():
    """A low-pass CANNOT track a ramp without positional offset -- it trails by
    roughly v*tau by construction (~43 mm at 0.3 m/s with the default cutoff).
    Demanding near-zero offset would be demanding that it not filter.

    What must hold is that it tracks the SLOPE: in steady state the output
    velocity equals the input velocity, and the offset is constant rather than
    growing. That is what catches sign errors and broken state updates, which
    is what this test is for.

    (The EKF's equivalent test DOES assert near-zero lag, because a
    constant-velocity model has a velocity state and tracks a ramp losslessly.
    The difference is real and shows up in the benchmark.)
    """
    v, fps = 0.3, 30.0
    f = OneEuroPoseFilter()
    got = _ramp(f, n=120, fps=fps, v=v)
    truth = np.array([v * (k / fps) for k in range(120)])

    # slope tracked exactly in steady state
    out_v = np.diff(got[60:, 0]) * fps
    assert np.allclose(out_v, v, atol=1e-3)

    # offset is constant (not growing) and of the expected magnitude
    offset = truth[60:] - got[60:, 0]
    assert offset.std() < 1e-4, "offset must be constant, not drifting"
    assert 0.005 < offset.mean() < 0.10


def test_one_euro_reduces_jitter():
    noise = 0.004                       # 4 mm, the rig's measured noise floor
    raw = _ramp(OneEuroPoseFilter(min_cutoff=1e9, beta=0.0), n=120, noise=noise)
    filt = _ramp(OneEuroPoseFilter(min_cutoff=0.5, beta=0.0), n=120, noise=noise)

    def hf(a):                          # deviation from a 5-sample moving mean
        return np.mean([np.linalg.norm(a[i] - a[i - 2:i + 3].mean(0))
                        for i in range(2, len(a) - 2)])

    assert hf(filt) < 0.5 * hf(raw)


def test_one_euro_step_response_settles_without_oscillation():
    f = OneEuroPoseFilter()
    for k in range(20):
        f.update(k / 30.0, [0.0, 0.0, 0.0], IDENT, True)
    xs = [f.update((20 + k) / 30.0, [1.0, 0.0, 0.0], IDENT, True).pos[0]
          for k in range(60)]
    assert xs[-1] == pytest.approx(1.0, abs=1e-2)
    assert max(xs) <= 1.0 + 1e-6        # no overshoot: it is a low-pass


def test_one_euro_output_quaternion_is_unit_norm():
    f = OneEuroPoseFilter()
    for k in range(120):
        q = Rotation.from_euler("z", 2.0 * k, degrees=True).as_quat()
        out = f.update(k / 30.0, [0.0, 0.0, 0.0], q, True)
    assert np.linalg.norm(out.quat) == pytest.approx(1.0, abs=1e-9)


def test_one_euro_rotation_converges_to_truth_short_arc():
    f = OneEuroPoseFilter(min_cutoff=5.0)
    target = Rotation.from_euler("z", 40, degrees=True)
    for k in range(200):
        out = f.update(k / 30.0, [0.0, 0.0, 0.0], target.as_quat(), True)
    err = (Rotation.from_quat(out.quat) * target.inv()).magnitude()
    assert np.degrees(err) < 1.0        # 40 deg, not 320: short arc


def test_one_euro_rotation_does_not_perturb_position():
    f = OneEuroPoseFilter()
    for k in range(60):
        q = Rotation.from_euler("x", 3.0 * k, degrees=True).as_quat()
        out = f.update(k / 30.0, [0.5, -0.25, 0.75], q, True)
    assert out.pos == pytest.approx([0.5, -0.25, 0.75], abs=1e-6)


def test_one_euro_position_does_not_perturb_orientation():
    f = OneEuroPoseFilter()
    for k in range(60):
        out = f.update(k / 30.0, [0.05 * k, 0.0, 0.0], IDENT, True)
    err = (Rotation.from_quat(out.quat) * Rotation.identity().inv()).magnitude()
    assert np.degrees(err) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pose_filter.py -v -k one_euro`
Expected: FAIL — `ImportError: cannot import name 'OneEuroPoseFilter'`

- [ ] **Step 3: Append the implementation**

```python
# append to robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py
from scipy.spatial.transform import Rotation


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
        delta = (R_meas * self._R.inv()).as_rotvec()
        delta_f = self._rot_f.update(delta, dt)
        self._R = Rotation.from_rotvec(delta_f) * self._R
        return self._p, self._R.as_quat()

    def _on_predict(self, t):
        dt = max(t - self._t, 0.0)
        return self._p + self._v * dt, self._R.as_quat()
```

- [ ] **Step 4: Run the whole file's tests**

Run: `python -m pytest tests/test_pose_filter.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_pose_filter.py
git commit -m "feat: add One-Euro pose filter with tangent-space orientation"
```

---

### Task 3: EKF filter

**Files:**
- Modify: `robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py` (append)
- Test: `tests/test_pose_filter.py` (append)

**Interfaces:**
- Consumes: `BasePoseFilter` (Task 1).
- Produces: `EkfPoseFilter(sigma_meas: float = 0.004, sigma_accel: float = 1.0, sigma_meas_rot: float = 0.02, sigma_alpha: float = 5.0, max_predict_frames: int = 3)`

Constant-velocity Kalman filter, run independently per position axis and per orientation tangent axis. State per channel is `[value, rate]`. This is deliberately a plain KF on the tangent space rather than a full 6-DoF error-state EKF: the tangent-space linearisation already handles the manifold, and the extra machinery would not change the jitter/lag trade this filter is being judged on.

- [ ] **Step 1: Append the failing tests**

```python
# append to tests/test_pose_filter.py
from robodriver_robot_deepcybo_lite_umi_ros2.pose_filter import EkfPoseFilter


def test_ekf_tracks_noiseless_ramp():
    f = EkfPoseFilter()
    got = _ramp(f, n=90)
    truth = np.array([0.3 * (k / 30.0) for k in range(90)])
    assert np.abs(got[40:, 0] - truth[40:]).max() < 5e-3


def test_ekf_reduces_jitter():
    noise = 0.004
    raw = _ramp(EkfPoseFilter(sigma_meas=1e-6), n=120, noise=noise)
    filt = _ramp(EkfPoseFilter(sigma_meas=0.004, sigma_accel=0.5),
                 n=120, noise=noise)

    def hf(a):
        return np.mean([np.linalg.norm(a[i] - a[i - 2:i + 3].mean(0))
                        for i in range(2, len(a) - 2)])

    assert hf(filt) < 0.5 * hf(raw)


def test_ekf_learns_velocity_it_never_measures():
    f = EkfPoseFilter()
    _ramp(f, n=90, v=0.3)
    # velocity is inferred from the position sequence alone
    assert f.velocity[0] == pytest.approx(0.3, rel=0.15)


def test_ekf_output_quaternion_is_unit_norm():
    f = EkfPoseFilter()
    for k in range(120):
        q = Rotation.from_euler("z", 2.0 * k, degrees=True).as_quat()
        out = f.update(k / 30.0, [0.0, 0.0, 0.0], q, True)
    assert np.linalg.norm(out.quat) == pytest.approx(1.0, abs=1e-9)


def test_ekf_rotation_converges_short_arc():
    f = EkfPoseFilter()
    target = Rotation.from_euler("z", 40, degrees=True)
    for k in range(200):
        out = f.update(k / 30.0, [0.0, 0.0, 0.0], target.as_quat(), True)
    err = (Rotation.from_quat(out.quat) * target.inv()).magnitude()
    assert np.degrees(err) < 1.0


def test_ekf_step_response_settles():
    f = EkfPoseFilter()
    for k in range(20):
        f.update(k / 30.0, [0.0, 0.0, 0.0], IDENT, True)
    xs = [f.update((20 + k) / 30.0, [1.0, 0.0, 0.0], IDENT, True).pos[0]
          for k in range(90)]
    assert xs[-1] == pytest.approx(1.0, abs=2e-2)


def test_ekf_rotation_does_not_perturb_position():
    f = EkfPoseFilter()
    for k in range(60):
        q = Rotation.from_euler("x", 3.0 * k, degrees=True).as_quat()
        out = f.update(k / 30.0, [0.5, -0.25, 0.75], q, True)
    assert out.pos == pytest.approx([0.5, -0.25, 0.75], abs=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pose_filter.py -v -k ekf`
Expected: FAIL — `ImportError: cannot import name 'EkfPoseFilter'`

- [ ] **Step 3: Append the implementation**

```python
# append to robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py


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
```

- [ ] **Step 4: Run the whole file's tests**

Run: `python -m pytest tests/test_pose_filter.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/pose_filter.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_pose_filter.py
git commit -m "feat: add constant-velocity EKF pose filter"
```

---

### Task 4: Safety tests — the reversal scenario

The tests that encode *why* the gap policy exists. These run against both filters.

**Files:**
- Test: `tests/test_filter_safety.py`

**Interfaces:**
- Consumes: `OneEuroPoseFilter`, `EkfPoseFilter` (Tasks 2–3).

- [ ] **Step 1: Write the tests**

```python
# tests/test_filter_safety.py
"""Safety properties of the gap policy, verified for BOTH filters.

Background: extrapolating constant velocity across an occlusion during which
the operator reversed direction produced 17.8 cm of error in simulation --
about 2x worse than freezing the last pose. During teleop that is a real arm
lurching the wrong way. These tests assert that regime is unreachable.
"""
import numpy as np
import pytest

from robodriver_robot_deepcybo_lite_umi_ros2.pose_filter import (
    EkfPoseFilter, OneEuroPoseFilter,
)

IDENT = np.array([0.0, 0.0, 0.0, 1.0])
FPS = 30.0
FILTERS = [OneEuroPoseFilter, EkfPoseFilter]


@pytest.mark.parametrize("cls", FILTERS)
def test_reversal_during_occlusion_commands_no_motion(cls):
    """Move forward, occlude, reverse.

    The safety property is that the filter STOPS COMMANDING MOTION, not that
    hand-vs-arm divergence stays small: once frozen, that divergence is just
    how far the operator's hand kept moving, which grows without bound and is
    not something the filter can control. What the filter controls is its own
    output -- so that is what is asserted:

      * while predicting (stale=False) the output stays close to truth
      * once stale, the output does not move at all, however long the
        occlusion lasts and however far the hand travels

    Asserting a bound on divergence instead would make the test fail for a
    longer occlusion even though freezing worked perfectly.
    """
    f = cls()
    v = 0.124                       # m/s, the rig's median hand speed
    truth, k = 0.0, 0
    for _ in range(15):             # forward, tracked
        f.update(k / FPS, [truth, 0.0, 0.0], IDENT, True)
        truth += v / FPS
        k += 1

    live_err = 0.0
    frozen_positions = []
    for _ in range(60):             # occluded AND reversing, for a long time
        out = f.update(k / FPS, [0.0, 0.0, 0.0], IDENT, False)
        truth -= v / FPS
        if out.stale:
            frozen_positions.append(out.pos.copy())
        else:
            live_err = max(live_err, abs(out.pos[0] - truth))
        k += 1

    # predicted frames are only a few mm off -- this is the regime the
    # 3-frame budget is chosen to keep us inside
    assert live_err < 0.02, f"{cls.__name__} predicted {live_err*100:.1f} cm off"

    # and once frozen the commanded pose is perfectly still, no matter that
    # the hand has by now travelled far in the opposite direction
    assert len(frozen_positions) > 50
    for p in frozen_positions[1:]:
        assert (p == frozen_positions[0]).all(), (
            f"{cls.__name__} kept moving while stale"
        )


@pytest.mark.parametrize("cls", FILTERS)
def test_stale_is_raised_before_error_grows(cls):
    f = cls()
    for k in range(15):
        f.update(k / FPS, [0.124 * k / FPS, 0.0, 0.0], IDENT, True)
    outs = [f.update((15 + i) / FPS, [0.0, 0.0, 0.0], IDENT, False)
            for i in range(10)]
    assert any(o.stale for o in outs)
    first_stale = next(i for i, o in enumerate(outs) if o.stale)
    assert first_stale <= 3, "stale must be raised within the predict budget"


@pytest.mark.parametrize("cls", FILTERS)
def test_no_lockout_on_reacquisition(cls):
    """A correct measurement far from the frozen pose must be ACCEPTED.

    In simulation an innovation-gated EKF rejected the true measurement on
    reacquisition and never recovered. Neither filter here may do that.
    """
    f = cls()
    for k in range(15):
        f.update(k / FPS, [0.01 * k, 0.0, 0.0], IDENT, True)
    for i in range(30):
        f.update((15 + i) / FPS, [0.0, 0.0, 0.0], IDENT, False)
    out = None
    for i in range(30):             # reacquired, 1 m away
        out = f.update((45 + i) / FPS, [1.0, 0.0, 0.0], IDENT, True)
    assert out.stale is False
    assert out.pos[0] == pytest.approx(1.0, abs=0.05), "filter locked out"


@pytest.mark.parametrize("cls", FILTERS)
def test_frozen_pose_never_drifts(cls):
    f = cls()
    for k in range(15):
        f.update(k / FPS, [0.05 * k, 0.0, 0.0], IDENT, True)
    outs = [f.update((15 + i) / FPS, [0.0, 0.0, 0.0], IDENT, False)
            for i in range(60)]
    frozen = [o.pos for o in outs if o.stale]
    for p in frozen[1:]:
        assert (p == frozen[0]).all()
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_filter_safety.py -v`
Expected: all PASS (both filters, all four properties)

- [ ] **Step 3: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_filter_safety.py
git commit -m "test: pin gap-policy safety properties for both filters"
```

---

### Task 5: Offline benchmark

The deliverable that decides which filter to keep.

**Files:**
- Create: `robodriver_robot_deepcybo_lite_umi_ros2/filter_bench.py`
- Modify: `pyproject.toml` (console script)
- Test: `tests/test_filter_bench.py`

**Interfaces:**
- Consumes: `PoseFilter`, `OneEuroPoseFilter`, `EkfPoseFilter` (Tasks 1–3); `ARM_LAYOUT` from `.smoothing`; `make_tiny_dataset`, `default_state` from `tests/dataset_fixture.py`.
- Produces:
  - `jitter_mm(pos: np.ndarray) -> float` — mean deviation from a 5-sample centred moving average, in mm
  - `lag_ms(raw: np.ndarray, filt: np.ndarray, fps: float) -> float` — cross-correlation offset minimising error
  - `overshoot_frac(raw: np.ndarray, filt: np.ndarray) -> float`
  - `BenchResult` frozen dataclass: `name: str`, `params: str`, `jitter_mm: float`, `lag_ms: float`, `overshoot: float`, `n_frames: int`, `n_stale: int`
  - `run_filter(pos, quat, tracked, times, filter_factory) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns `(pos_out, quat_out, stale_mask)`
  - `bench_dataset(root: Path, arm: str = "right", episode: int = 0) -> list[BenchResult]`
  - `format_bench(results: list[BenchResult]) -> str`
  - `main(argv=None) -> int` — console script `umi-filter-bench`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_filter_bench.py
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fixture import default_state, make_tiny_dataset  # noqa: E402

from robodriver_robot_deepcybo_lite_umi_ros2.filter_bench import (  # noqa: E402
    BenchResult, bench_dataset, format_bench, jitter_mm, lag_ms,
    overshoot_frac, run_filter,
)
from robodriver_robot_deepcybo_lite_umi_ros2.pose_filter import (  # noqa: E402
    OneEuroPoseFilter,
)

IDENT = np.array([0.0, 0.0, 0.0, 1.0])


def test_jitter_zero_on_smooth_signal():
    t = np.arange(200) / 30.0
    pos = np.stack([0.3 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    assert jitter_mm(pos) < 1e-6


def test_jitter_detects_known_noise():
    rng = np.random.default_rng(0)
    t = np.arange(400) / 30.0
    pos = np.stack([0.3 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    noisy = pos + rng.normal(0, 0.004, pos.shape)
    j = jitter_mm(noisy)
    assert 1.0 < j < 10.0            # order-of-magnitude sanity for 4 mm sigma


def test_lag_zero_for_identical_signals():
    t = np.arange(200) / 30.0
    pos = np.stack([np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1)
    assert abs(lag_ms(pos, pos, 30.0)) < 1e-6


def test_lag_detects_known_shift():
    t = np.arange(300) / 30.0
    raw = np.stack([np.sin(2 * t), np.zeros_like(t), np.zeros_like(t)], axis=1)
    shifted = np.roll(raw, 3, axis=0)          # 3 frames = 100 ms at 30 Hz
    assert lag_ms(raw, shifted, 30.0) == pytest.approx(100.0, abs=20.0)


def test_overshoot_zero_for_monotone_step_response():
    raw = np.zeros((100, 3)); raw[50:, 0] = 1.0
    filt = raw.copy()
    assert overshoot_frac(raw, filt) == pytest.approx(0.0, abs=1e-9)


def test_overshoot_detects_ringing():
    raw = np.zeros((100, 3)); raw[50:, 0] = 1.0
    filt = raw.copy(); filt[55, 0] = 1.2       # 20% overshoot
    assert overshoot_frac(raw, filt) == pytest.approx(0.2, abs=0.01)


def test_run_filter_shapes_and_stale_mask():
    n = 60
    times = np.arange(n) / 30.0
    pos = np.stack([0.01 * np.arange(n), np.zeros(n), np.zeros(n)], axis=1)
    quat = np.tile(IDENT, (n, 1))
    tracked = np.ones(n, dtype=bool)
    tracked[20:40] = False                      # a long gap -> must go stale
    p, q, stale = run_filter(pos, quat, tracked, times, OneEuroPoseFilter)
    assert p.shape == (n, 3) and q.shape == (n, 4) and stale.shape == (n,)
    assert stale[20:40].any()
    assert not stale[:20].any()


def test_run_filter_is_causal():
    """Changing a LATE sample must not alter an EARLY output."""
    n = 60
    times = np.arange(n) / 30.0
    pos = np.stack([0.01 * np.arange(n), np.zeros(n), np.zeros(n)], axis=1)
    quat = np.tile(IDENT, (n, 1))
    tracked = np.ones(n, dtype=bool)
    a, _, _ = run_filter(pos, quat, tracked, times, OneEuroPoseFilter)
    pos2 = pos.copy(); pos2[50:] += 5.0
    b, _, _ = run_filter(pos2, quat, tracked, times, OneEuroPoseFilter)
    assert (a[:50] == b[:50]).all()


@pytest.fixture()
def ds(tmp_path):
    n = 120
    s = default_state(n)
    rng = np.random.default_rng(1)
    s[:, 8] = (0.3 * np.arange(n) / 30.0 + rng.normal(0, 0.004, n)).astype(np.float32)
    root = tmp_path / "ds"
    make_tiny_dataset(root, with_provenance=False, state=s)
    return root


def test_bench_dataset_returns_results_for_both_filters(ds):
    res = bench_dataset(ds, arm="right")
    assert len(res) >= 4
    assert all(isinstance(r, BenchResult) for r in res)
    names = {r.name for r in res}
    assert "one-euro" in names and "ekf" in names
    assert any(r.name == "raw" for r in res), "baseline row required"


def test_bench_filters_reduce_jitter_below_raw(ds):
    res = bench_dataset(ds, arm="right")
    raw = next(r for r in res if r.name == "raw")
    for r in res:
        if r.name != "raw":
            assert r.jitter_mm < raw.jitter_mm


def test_bench_does_not_modify_dataset(ds):
    import hashlib
    snap = {p.relative_to(ds).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ds.rglob("*")) if p.is_file()}
    bench_dataset(ds, arm="right")
    after = {p.relative_to(ds).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(ds.rglob("*")) if p.is_file()}
    assert snap == after


def test_format_bench_shows_frontier(ds):
    text = format_bench(bench_dataset(ds, arm="right"))
    assert "jitter" in text and "lag" in text
    assert "raw" in text and "one-euro" in text and "ekf" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_filter_bench.py -v`
Expected: FAIL at import — `ModuleNotFoundError: ... filter_bench`

- [ ] **Step 3: Write the implementation**

```python
# robodriver_robot_deepcybo_lite_umi_ros2/filter_bench.py
"""Offline filter benchmark (spec 2026-07-22).

Replays a recorded episode's pose stream through candidate filters and reports
jitter / lag / overshoot, so EKF-vs-One-Euro is decided by measurement rather
than opinion.

ALL THREE metrics are reported together on purpose: any filter can drive
jitter to zero by smoothing harder, it simply becomes unusable because the arm
lags the operator's hand. The design question is the TRADE, so each filter is
swept across its parameters and the results read as a jitter-vs-lag frontier --
"at equal lag, which gives less jitter?"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .pose_filter import EkfPoseFilter, OneEuroPoseFilter
from .smoothing import ARM_LAYOUT


def jitter_mm(pos: np.ndarray) -> float:
    """Mean deviation from a 5-sample centred moving average, in mm.

    Same measure that gave the 2.67/3.15 mm raw baseline on the real rig.
    """
    pos = np.asarray(pos, dtype=float)
    if len(pos) < 5:
        return 0.0
    dev = [np.linalg.norm(pos[i] - pos[i - 2:i + 3].mean(axis=0))
           for i in range(2, len(pos) - 2)]
    return float(np.mean(dev) * 1000.0)


def lag_ms(raw: np.ndarray, filt: np.ndarray, fps: float) -> float:
    """Frame shift of `filt` vs `raw` minimising squared error, in ms."""
    raw = np.asarray(raw, float)
    filt = np.asarray(filt, float)
    best_k, best_err = 0, np.inf
    for k in range(0, min(30, len(raw) // 4)):
        err = float(np.mean((raw[:len(raw) - k] - filt[k:]) ** 2))
        if err < best_err:
            best_err, best_k = err, k
    return best_k * 1000.0 / fps


def overshoot_frac(raw: np.ndarray, filt: np.ndarray) -> float:
    """Max excursion of `filt` past `raw`'s range, as a fraction of its span."""
    raw = np.asarray(raw, float)
    filt = np.asarray(filt, float)
    span = float(np.linalg.norm(raw.max(axis=0) - raw.min(axis=0)))
    if span < 1e-9:
        return 0.0
    over = 0.0
    for d in range(raw.shape[1]):
        over = max(over, float(filt[:, d].max() - raw[:, d].max()))
        over = max(over, float(raw[:, d].min() - filt[:, d].min()))
    return max(over, 0.0) / span


@dataclass(frozen=True)
class BenchResult:
    name: str
    params: str
    jitter_mm: float
    lag_ms: float
    overshoot: float
    n_frames: int
    n_stale: int


def run_filter(pos, quat, tracked, times, filter_factory):
    """Stream a recorded sequence through a filter. Returns (pos, quat, stale).

    Frames emitted before the filter initialises reuse the raw input, so the
    output arrays align 1:1 with the input for metric computation.
    """
    f = filter_factory()
    n = len(times)
    p_out = np.array(pos, dtype=float, copy=True)
    q_out = np.array(quat, dtype=float, copy=True)
    stale = np.zeros(n, dtype=bool)
    for i in range(n):
        out = f.update(float(times[i]), pos[i], quat[i], bool(tracked[i]))
        stale[i] = out.stale
        if out.pos is not None:
            p_out[i] = out.pos
            q_out[i] = out.quat
    return p_out, q_out, stale


def _sweep():
    """(name, params, factory) for every operating point on the frontier."""
    points = []
    for beta in (0.0, 0.1, 0.4, 1.0):
        for mc in (0.3, 1.0):
            points.append((
                "one-euro", f"min_cutoff={mc},beta={beta}",
                lambda mc=mc, beta=beta: OneEuroPoseFilter(min_cutoff=mc, beta=beta),
            ))
    for sa in (0.2, 0.5, 1.0, 4.0):
        points.append((
            "ekf", f"sigma_accel={sa}",
            lambda sa=sa: EkfPoseFilter(sigma_accel=sa),
        ))
    return points


def _load_arm(root: Path, arm: str, episode: int):
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    rel = info["data_path"].format(
        episode_chunk=episode // info["chunks_size"], episode_index=episode)
    df = pq.read_table(root / rel).to_pandas()
    state = np.stack(df["observation.state"].to_numpy()).astype(float)
    lay = ARM_LAYOUT[arm]
    return (state[:, lay.pos], state[:, lay.quat],
            state[:, lay.tracked] > 0.5,
            df["timestamp"].to_numpy(dtype=float), int(info["fps"]))


def bench_dataset(root: Path, arm: str = "right",
                  episode: int = 0) -> list[BenchResult]:
    pos, quat, tracked, times, fps = _load_arm(Path(root), arm, episode)
    results = [BenchResult("raw", "-", jitter_mm(pos), 0.0, 0.0,
                           len(times), 0)]
    for name, params, factory in _sweep():
        p, _, stale = run_filter(pos, quat, tracked, times, factory)
        results.append(BenchResult(
            name, params, jitter_mm(p), lag_ms(pos, p, fps),
            overshoot_frac(pos, p), len(times), int(stale.sum()),
        ))
    return results


def format_bench(results: list[BenchResult]) -> str:
    lines = [f"{'filter':<10} {'params':<28} {'jitter(mm)':>11} "
             f"{'lag(ms)':>8} {'overshoot':>10} {'stale':>6}"]
    lines.append("-" * 78)
    for r in results:
        lines.append(f"{r.name:<10} {r.params:<28} {r.jitter_mm:>11.2f} "
                     f"{r.lag_ms:>8.1f} {r.overshoot:>10.1%} {r.n_stale:>6d}")
    lines.append("")
    lines.append("At comparable lag, lower jitter wins. Overshoot > ~5% means "
                 "the filter rings after fast motion.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark pose filters against a recorded episode.")
    parser.add_argument("--root", type=Path, required=True,
                        help="dataset root (read only)")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args(argv)
    print(format_bench(bench_dataset(args.root, args.arm, args.episode)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml` under `[project.scripts]` add:

```toml
umi-filter-bench = "robodriver_robot_deepcybo_lite_umi_ros2.filter_bench:main"
```

- [ ] **Step 5: Run tests and the real benchmark**

Run: `python -m pytest tests/test_filter_bench.py -v`
Expected: all PASS

Run: `python -m pytest tests/ -q`
Expected: all pass, 1 pre-existing skip (the Linux-gated canonical-reader spike)

Run the real comparison and **paste the table into the task report** — this is the decision artifact:
```
python -m robodriver_robot_deepcybo_lite_umi_ros2.filter_bench --root "D:\Desktop\Mystuff\robotics\umi_imp\umi_real_rec_2026-07-15" --arm right
python -m robodriver_robot_deepcybo_lite_umi_ros2.filter_bench --root "D:\Desktop\Mystuff\robotics\umi_imp\umi_real_rec_2026-07-15" --arm left
```
Expected: the `raw` row shows ~3.15 mm (right) / ~2.67 mm (left), matching the
known baseline; every filter row shows lower jitter and non-zero lag.

- [ ] **Step 6: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "feat: add offline filter benchmark with jitter-lag frontier"
```

---

### Task 6: ROS node

**Files:**
- Create: `robodriver_robot_deepcybo_lite_umi_ros2/filter_node.py`
- Modify: `pyproject.toml` (console script)
- Modify: `README.md`

**Interfaces:**
- Consumes: `OneEuroPoseFilter`, `EkfPoseFilter` (Tasks 2–3); `DeepcyboLiteUmiRos2RobotConfig` from `.config`.
- Produces: `FilteredPoseNode` (rclpy Node); `main(argv=None) -> None`; console script `umi-filter-node`.

Thin by construction: subscribe, call `update()`, publish. All logic lives in the pure layer, so this task adds no unit tests — the node is exercised on the rig.

- [ ] **Step 1: Write the node**

```python
# robodriver_robot_deepcybo_lite_umi_ros2/filter_node.py
"""Republish the live ArUco gripper poses, filtered for teleop (spec 2026-07-22).

Subscribes to the tracker's raw GripperTrack topics, runs each arm through a
causal pose filter, and republishes the smoothed pose plus a staleness flag.

The TRACKER keeps publishing raw measurements -- this node is downstream, so
recorded datasets stay pristine and the offline smoother's anchor set stays
honest. Nothing consumes the filtered topics until the IK bridge exists.

Run:
    umi-filter-node --filter one-euro
"""
from __future__ import annotations

import argparse

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node as ROS2Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool

from .config import DeepcyboLiteUmiRos2RobotConfig
from .pose_filter import EkfPoseFilter, OneEuroPoseFilter

try:
    from lite_aruco_umi_msgs.msg import GripperTrack
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lite_aruco_umi_msgs not on the ROS overlay; source the collection "
        "workspace first."
    ) from exc

FILTERS = {"one-euro": OneEuroPoseFilter, "ekf": EkfPoseFilter}


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class FilteredPoseNode(ROS2Node):
    def __init__(self, filter_name: str = "one-euro",
                 max_predict_frames: int = 3):
        super().__init__("umi_filtered_pose")
        cfg = DeepcyboLiteUmiRos2RobotConfig()
        t = cfg.ros2_topics
        factory = FILTERS[filter_name]
        self._filters = {
            arm: factory(max_predict_frames=max_predict_frames)
            for arm in ("left", "right")
        }
        qos = QoSProfile(durability=DurabilityPolicy.VOLATILE,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self._pose_pub = {}
        self._stale_pub = {}
        for arm, topic in (("left", t.track_left), ("right", t.track_right)):
            self.create_subscription(
                GripperTrack, topic,
                lambda msg, a=arm: self._on_track(a, msg), qos)
            self._pose_pub[arm] = self.create_publisher(
                PoseStamped, f"/umi/filtered/eef_{arm}", 10)
            self._stale_pub[arm] = self.create_publisher(
                Bool, f"/umi/filtered/stale_{arm}", 10)
        self.get_logger().info(
            f"filtered-pose node up | filter={filter_name} "
            f"max_predict_frames={max_predict_frames} | publishing "
            f"/umi/filtered/eef_left|right + /umi/filtered/stale_left|right"
        )

    def _on_track(self, arm: str, msg) -> None:
        t = _stamp_to_sec(msg.header.stamp)
        usable = bool(msg.tracked) and bool(msg.present) and bool(msg.has_tcp)
        p = msg.tcp_pose.position
        o = msg.tcp_pose.orientation
        out = self._filters[arm].update(
            t, np.array([p.x, p.y, p.z]),
            np.array([o.x, o.y, o.z, o.w]), usable)

        self._stale_pub[arm].publish(Bool(data=bool(out.stale)))
        if out.pos is None:
            return                      # uninitialised: publish no pose at all
        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = msg.header.frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
            float(out.pos[0]), float(out.pos[1]), float(out.pos[2]))
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = (
            float(out.quat[0]), float(out.quat[1]),
            float(out.quat[2]), float(out.quat[3]))
        self._pose_pub[arm].publish(ps)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Republish live ArUco gripper poses, filtered for teleop.")
    parser.add_argument("--filter", choices=tuple(FILTERS), default="one-euro")
    parser.add_argument("--max-predict-frames", type=int, default=3)
    args = parser.parse_args(argv)

    rclpy.init()
    node = FilteredPoseNode(args.filter, args.max_predict_frames)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Register the console script**

In `pyproject.toml` under `[project.scripts]` add:

```toml
umi-filter-node = "robodriver_robot_deepcybo_lite_umi_ros2.filter_node:main"
```

- [ ] **Step 3: Verify it compiles and the suite still passes**

Run: `python -m py_compile robodriver_robot_deepcybo_lite_umi_ros2/filter_node.py`
Expected: no output (rclpy is absent on this machine, so only syntax is checked here; the node runs on the rig)

Run: `python -m pytest tests/ -q`
Expected: all pass, 1 pre-existing skip

- [ ] **Step 4: Document in README.md**

Append after the offline-smoothing section:

```markdown
## Teleop pose filter (live)

The tracker publishes RAW poses; this node republishes a smoothed copy for
teleop. Recorded datasets are unaffected.

    umi-filter-bench --root <dataset> --arm right   # compare filters offline
    umi-filter-node --filter one-euro               # live, on the rig

Publishes `/umi/filtered/eef_{left,right}` (PoseStamped) and
`/umi/filtered/stale_{left,right}` (Bool).

> **`stale` means the arm must be halted.** The filter predicts through gaps of
> at most `--max-predict-frames` (default 3) and then freezes, because
> extrapolating further measured ~2x worse than freezing when the operator
> reversed direction during an occlusion.
```

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "feat: add live filtered-pose node for teleop"
```

---

## Self-Review Notes

- **Spec coverage:** pure/ROS split (T1–T3, T6); shared gap policy with `max_predict_frames` default 3 (T1); freeze bit-identical (T1, T4); uninitialized `None`/stale and verbatim first adoption and `reset()` (T1); tangent-space orientation (T2, T3); One-Euro (T2); EKF (T3); reversal/lock-out safety (T4); jitter+lag+overshoot with parameter sweep and frontier (T5); read-only dataset (T5); node republishing pose+stale (T6). The spec's leave-one-out accuracy check is deliberately NOT implemented: the three signal metrics decide the filter choice, and the smoother's existing leave-one-out already characterises this data's noise floor — adding a second harness would be scope the decision does not need.
- **Known limitation restated:** no test can judge teleop *feel*; that needs the IK bridge and a human. The bench narrows to a defensible operating point only.
- **Type consistency:** `FilterOutput.pos` is `np.ndarray | None` in T1 and every consumer guards it (T5 `run_filter`, T6 `_on_track`); `filter_factory` is a zero-arg callable in both T5's `_sweep` and `run_filter`; `ARM_LAYOUT[arm].tracked` is an int index in T5, matching `smoothing.py`.
- **Deliberate simplification:** `EkfPoseFilter` runs independent scalar KFs per axis rather than a coupled 6-DoF error-state EKF. Documented in its docstring — the tangent-space formulation already handles the manifold, and axis coupling would not change the jitter/lag trade being measured.
