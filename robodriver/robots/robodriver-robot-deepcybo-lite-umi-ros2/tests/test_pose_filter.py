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


class MutatingInPlace(BasePoseFilter):
    """Subclass whose hooks return a REFERENCE to persistent internal state
    that is mutated in place on each call, rather than reallocating a fresh
    array -- the natural (and desirable, for performance) way to implement a
    real Kalman filter's state vector.

    Exists solely to expose the `_emit` aliasing hazard: `np.asarray` does
    NOT copy an input that is already a float64 ndarray, so if `_emit` used
    it directly, `self._last` (and therefore `self._frozen`, since freezing
    does `self._frozen = self._last`) would hold the SAME array object the
    subclass keeps mutating. A later in-place update to that internal state
    would then silently corrupt a pose already frozen and emitted to drive
    the arm -- exactly the regression this test guards against.
    """

    def _on_first(self, t, pos, quat):
        self._p = np.array(pos, dtype=float)
        self._q = np.array(quat, dtype=float)
        self._t = t

    def _on_measurement(self, t, pos, quat):
        self._p[:] = pos      # in-place mutation, no reallocation
        self._q[:] = quat
        self._t = t
        return self._p, self._q

    def _on_predict(self, t):
        return self._p, self._q   # live reference to internal state


def test_frozen_pose_immune_to_subclass_inplace_mutation():
    f = MutatingInPlace(max_predict_frames=1)
    f.update(0.0, [0.0, 0.0, 0.0], IDENT, tracked=True)         # _on_first
    f.update(1 / 30.0, [1.0, 0.0, 0.0], IDENT, tracked=True)    # _on_measurement
    f.update(2 / 30.0, [1.0, 0.0, 0.0], IDENT, tracked=False)   # predict, within budget
    frozen_out = f.update(3 / 30.0, [1.0, 0.0, 0.0], IDENT, tracked=False)  # crosses budget -> freeze
    assert frozen_out.stale is True
    frozen_pos = frozen_out.pos
    assert (frozen_pos == np.array([1.0, 0.0, 0.0])).all()

    # A brand-new tracked measurement drives the subclass to mutate its
    # internal array in place. The pose we already captured as "frozen" and
    # handed to the (hypothetical) arm controller must NOT change as a
    # side effect.
    f.update(4 / 30.0, [9.0, 9.0, 9.0], IDENT, tracked=True)

    assert (frozen_pos == np.array([1.0, 0.0, 0.0])).all(), (
        "frozen pose mutated via aliasing with subclass internal state"
    )


def test_frozen_outputs_are_independent_objects():
    """Guards the CALLER-side aliasing hazard (as opposed to the subclass-side
    hazard covered by `test_frozen_pose_immune_to_subclass_inplace_mutation`
    above): `_emit`/the freeze branch used to hand back the exact array
    objects retained in `self._last`/`self._frozen`, so every frozen tick's
    `.pos`/`.quat` was literally the SAME ndarray. A downstream consumer that
    does an in-place transform on one returned output (e.g. normalising it,
    or reusing a buffer) would then silently corrupt every other frozen tick,
    past and future -- the same class of silent pose corruption the
    subclass-side fix claims to have eliminated, just approached from the
    other side of the interface.
    """
    f = PassThrough(max_predict_frames=1)
    _feed(f, 5)
    f.update(5 / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)  # predict, within budget
    out1 = f.update(6 / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)  # crosses budget -> freeze
    assert out1.stale is True

    # Snapshot the true frozen value BEFORE mutating out1, so the assertions
    # below don't depend on hardcoding PassThrough's extrapolated number.
    expected_pos = out1.pos.copy()
    expected_quat = out1.quat.copy()

    # Simulate a downstream consumer mutating a returned pose in place.
    out1.pos[:] = [42.0, 42.0, 42.0]
    out1.quat[:] = [42.0, 42.0, 42.0, 42.0]

    out2 = f.update(7 / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)  # another frozen tick
    assert out2.stale is True

    # The mutation of out1 must NOT leak into out2.
    assert (out2.pos == expected_pos).all(), (
        "frozen pose corrupted by caller-side in-place mutation of a prior output"
    )
    assert (out2.quat == expected_quat).all(), (
        "frozen quat corrupted by caller-side in-place mutation of a prior output"
    )

    # Equal in value, but never the same object -- neither to each other nor
    # to whatever the filter retains internally.
    assert out1.pos is not out2.pos
    assert out1.quat is not out2.quat
    fresh = f.update(8 / 30.0, [0.0, 0.0, 0.0], IDENT, tracked=False)
    assert (fresh.pos == out2.pos).all()
    assert fresh.pos is not out2.pos


def test_negative_max_predict_frames_raises():
    with pytest.raises(ValueError):
        PassThrough(max_predict_frames=-1)


def test_zero_max_predict_frames_does_not_raise():
    f = PassThrough(max_predict_frames=0)
    assert f.max_predict_frames == 0


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
