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
