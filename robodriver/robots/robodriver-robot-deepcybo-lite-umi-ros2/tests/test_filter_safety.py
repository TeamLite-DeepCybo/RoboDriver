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


# ---------------------------------------------------------------------------
# Fix 1: the frame-count budget alone lets the reversal error (2*v*n/fps)
# grow without bound as hand speed rises. `max_predict_displacement_m` caps
# the predicted DISPLACEMENT itself so the worst case stays near 2x the cap
# regardless of speed. Parametrized over both filters, same as the rest of
# this file.
# ---------------------------------------------------------------------------
CAP_M = 0.015


def _reversal_run(f, v, n_gap=30):
    """Forward for 15 tracked frames, then occlude AND reverse for `n_gap`
    frames. Returns the list of FilterOutput from the occluded portion.
    """
    truth, k = 0.0, 0
    for _ in range(15):
        f.update(k / FPS, [truth, 0.0, 0.0], IDENT, True)
        truth += v / FPS
        k += 1
    outs = []
    for _ in range(n_gap):
        out = f.update(k / FPS, [0.0, 0.0, 0.0], IDENT, False)
        truth -= v / FPS
        outs.append((out, truth))
        k += 1
    return outs


@pytest.mark.parametrize("cls", FILTERS)
def test_displacement_cap_bounds_error_at_high_speed(cls):
    """At v = 1.0 m/s the UNCAPPED ceiling (2*v*max_predict_frames/fps) is
    ~20 cm and grows further with speed. With the cap active, the
    predicted-frame error (measured, like `test_reversal_during_occlusion_
    commands_no_motion`, only while `stale` is False -- once frozen the
    divergence is hand travel, not filter error, and is unbounded by design)
    must stay near 2x the 15 mm cap instead, and be materially better than
    the uncapped ceiling.
    """
    v = 1.0
    f = cls(max_predict_displacement_m=CAP_M)
    outs = _reversal_run(f, v)

    live_err = max((abs(out.pos[0] - truth) for out, truth in outs
                    if not out.stale), default=0.0)

    uncapped_ceiling = 2.0 * v * f.max_predict_frames / FPS
    assert live_err <= 2.0 * CAP_M * 1.5, (
        f"{cls.__name__} predicted-frame error {live_err*100:.2f} cm is not "
        f"bounded near 2x the {CAP_M*100:.1f} cm displacement cap"
    )
    assert live_err < 0.5 * uncapped_ceiling, (
        f"{cls.__name__} capped error {live_err*100:.2f} cm is not "
        f"materially better than the uncapped 2*v*n/fps ceiling of "
        f"{uncapped_ceiling*100:.2f} cm"
    )


@pytest.mark.parametrize("cls", FILTERS)
def test_displacement_cap_engages_before_frame_budget_at_high_speed(cls):
    """At v = 1.0 m/s, one frame of extrapolation alone (~33 mm) already
    exceeds the 15 mm cap, so the filter must freeze in FEWER predicted
    frames than `max_predict_frames` -- the cap, not the frame count, is
    what stops it.
    """
    f = cls(max_predict_displacement_m=CAP_M)
    outs = _reversal_run(f, v=1.0)

    n_predicted_before_freeze = 0
    for out, _truth in outs:
        if out.stale:
            break
        n_predicted_before_freeze += 1

    assert n_predicted_before_freeze < f.max_predict_frames, (
        f"{cls.__name__} predicted {n_predicted_before_freeze} frames "
        f"before freezing at v=1.0 m/s; expected the displacement cap to "
        f"engage before the {f.max_predict_frames}-frame budget"
    )


@pytest.mark.parametrize("cls", FILTERS)
def test_displacement_cap_does_not_engage_at_median_speed(cls):
    """At the rig's median speed (0.124 m/s), `max_predict_frames` frames of
    prediction cover ~12.4 mm -- under the 15 mm default cap -- so the
    anti-stutter behaviour for ordinary single/double-frame dropouts must be
    unchanged: all `max_predict_frames` predicted frames still happen before
    freezing.
    """
    f = cls()  # default cap
    outs = _reversal_run(f, v=0.124)

    n_predicted_before_freeze = 0
    for out, _truth in outs:
        if out.stale:
            break
        n_predicted_before_freeze += 1

    assert n_predicted_before_freeze == f.max_predict_frames, (
        f"{cls.__name__} predicted only {n_predicted_before_freeze} frames "
        f"at the rig's median speed; expected all {f.max_predict_frames} "
        f"(the anti-stutter property regressed)"
    )


@pytest.mark.parametrize("cls", FILTERS)
def test_zero_displacement_cap_freezes_immediately(cls):
    """`max_predict_displacement_m=0.0` means "never predict": the very
    first untracked frame must freeze, same as `max_predict_frames=0`.
    """
    f = cls(max_predict_displacement_m=0.0)
    for k in range(15):
        f.update(k / FPS, [0.01 * k, 0.0, 0.0], IDENT, True)
    out = f.update(15 / FPS, [0.0, 0.0, 0.0], IDENT, False)
    assert out.stale is True
    assert out.n_predicted == 1
