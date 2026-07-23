# tests/test_filter_bench.py
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fixture import default_state, make_tiny_dataset  # noqa: E402

from robodriver_robot_deepcybo_lite_umi_ros2.filter_bench import (  # noqa: E402
    BenchResult, bench_dataset, format_bench, jitter_mm, lag_ms,
    overshoot_frac, run_filter, synthetic_ramp_lag, synthetic_step_overshoot,
)
from robodriver_robot_deepcybo_lite_umi_ros2.pose_filter import (  # noqa: E402
    EkfPoseFilter, OneEuroPoseFilter,
)

IDENT = np.array([0.0, 0.0, 0.0, 1.0])


def test_jitter_zero_on_smooth_signal():
    t = np.arange(200) / 30.0
    pos = np.stack([0.3 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    tracked = np.ones(len(t), dtype=bool)
    assert jitter_mm(pos, tracked) < 1e-6


def test_jitter_detects_known_noise():
    rng = np.random.default_rng(0)
    t = np.arange(400) / 30.0
    pos = np.stack([0.3 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    noisy = pos + rng.normal(0, 0.004, pos.shape)
    tracked = np.ones(len(t), dtype=bool)
    j = jitter_mm(noisy, tracked)
    assert 1.0 < j < 10.0            # order-of-magnitude sanity for 4 mm sigma


def test_jitter_uses_median_so_a_rare_outlier_does_not_dominate():
    """A single one-off jump must not drag a mean-like statistic up.

    41 static frames except one 50 mm spike at the centre: the 5-frame
    window only picks up the spike for the ~5 indices nearest it, so the
    MEDIAN of all (mostly-zero) deviations stays at 0 -- a mean would not.
    """
    n = 41
    pos = np.zeros((n, 3))
    pos[20, 0] = 0.05
    tracked = np.ones(n, dtype=bool)
    assert jitter_mm(pos, tracked) == pytest.approx(0.0, abs=1e-9)


def test_jitter_only_counts_consecutively_tracked_runs():
    """A held/untracked block between two tracked runs must not leak in.

    Without per-run splitting, the moving-average window straddling the
    boundary between a near-zero run and a near-10 m run would register a
    multi-metre "deviation" -- nothing like sensor jitter. With per-run
    splitting, only the tiny in-run noise (~1 mm sigma) can contribute.
    """
    rng = np.random.default_rng(2)
    n = 60
    pos = np.zeros((n, 3))
    pos[:20] = rng.normal(0, 0.001, (20, 3))
    pos[20:40] = 5.0                      # held/duplicate frames, huge offset
    pos[40:] = 10.0 + rng.normal(0, 0.001, (20, 3))
    tracked = np.zeros(n, dtype=bool)
    tracked[:20] = True
    tracked[40:] = True
    j = jitter_mm(pos, tracked)
    assert j < 5.0   # order-of-magnitude looser than the ~1mm noise floor,
                     # far below what any cross-run contamination would give


def test_jitter_returns_zero_when_no_run_is_long_enough():
    n = 10
    pos = np.random.default_rng(3).normal(0, 0.01, (n, 3))
    tracked = np.zeros(n, dtype=bool)
    tracked[3:6] = True   # a 3-frame run: shorter than the 5-frame window
    assert jitter_mm(pos, tracked) == 0.0


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


def test_synthetic_lag_near_zero_for_a_pass_through_ish_filter():
    lag_mm, lag_ms_v = synthetic_ramp_lag(
        lambda: OneEuroPoseFilter(min_cutoff=100.0, beta=0.0))
    assert lag_mm < 1.0
    assert lag_ms_v < 5.0


def test_synthetic_lag_clearly_nonzero_for_heavy_smoothing():
    lag_mm, lag_ms_v = synthetic_ramp_lag(
        lambda: OneEuroPoseFilter(min_cutoff=0.2, beta=0.0), n=1000)
    assert lag_mm > 100.0
    assert lag_ms_v > 300.0


def test_synthetic_lag_is_monotonic_with_smoothing():
    """Heavier smoothing (a lower cutoff) must give strictly more lag."""
    cutoffs = (0.2, 0.5, 1.0, 3.0)
    lags = [synthetic_ramp_lag(
                lambda mc=mc: OneEuroPoseFilter(min_cutoff=mc, beta=0.0),
                n=1000)[0]
            for mc in cutoffs]
    assert all(a > b for a, b in zip(lags, lags[1:]))


def test_synthetic_lag_matches_independently_measured_reference():
    """Pins the exact number this task's brief root-caused against: a
    One-Euro configuration independently measured at 31.8 mm (~106 ms) of
    lag on a moving signal."""
    lag_mm, lag_ms_v = synthetic_ramp_lag(
        lambda: OneEuroPoseFilter(min_cutoff=1.0, beta=0.4))
    assert lag_mm == pytest.approx(31.8, abs=0.5)
    assert lag_ms_v == pytest.approx(106.0, abs=2.0)


def test_synthetic_overshoot_zero_for_one_euro():
    """A single-pole low-pass (One-Euro) never overshoots a step."""
    ov = synthetic_step_overshoot(
        lambda: OneEuroPoseFilter(min_cutoff=0.3, beta=0.4))
    assert ov == pytest.approx(0.0, abs=1e-9)


def test_synthetic_overshoot_detects_ekf_ringing():
    """The EKF's velocity state overshoots a step at a high sigma_accel."""
    ov = synthetic_step_overshoot(lambda: EkfPoseFilter(sigma_accel=4.0))
    assert ov > 0.05


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
    assert "jitter" in text and "lag" in text and "overshoot" in text
    assert "raw" in text and "one-euro" in text and "ekf" in text
    # a reader must not mistake real-data jitter for synthetic-motion lag
    assert "REAL" in text and "SYNTH" in text
