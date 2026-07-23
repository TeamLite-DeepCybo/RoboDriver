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
