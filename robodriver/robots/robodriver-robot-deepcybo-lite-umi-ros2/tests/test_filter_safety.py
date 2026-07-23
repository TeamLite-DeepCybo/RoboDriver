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

    # Predicted-frame error during a reversal is bounded by PHYSICS, not by
    # filter quality: the filter extrapolates forward at +v while truth moves
    # at -v, so after n predicted frames the gap is at most 2*v*n/fps. That
    # ceiling IS what the predict budget costs in the worst case, so assert
    # against it rather than a magic number -- and it stays correct if
    # max_predict_frames or the test speed changes.
    #
    # A filter with a lagged velocity estimate (One-Euro) comes in under this
    # by accident, not by design; an accurate one (EKF) approaches it.
    ceiling = 2.0 * v * f.max_predict_frames / FPS
    assert live_err <= ceiling * 1.05, (
        f"{cls.__name__} predicted {live_err*100:.2f} cm off, "
        f"above the {ceiling*100:.2f} cm physical ceiling"
    )

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
