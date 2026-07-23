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

Jitter, lag and overshoot are measured on DIFFERENT inputs, on purpose:
  - jitter is a property of the SENSOR NOISE, so it is measured on the real
    recording (median deviation from a local moving average, over
    consecutively-tracked runs -- see `jitter_mm`).
  - lag and overshoot are properties of the FILTER'S DYNAMICS, so they are
    measured on controlled SYNTHETIC motion instead (`synthetic_ramp_lag`,
    `synthetic_step_overshoot`). The available recording covers only ~3.3-3.5
    cm of total travel over 8 s -- the hand jiggled in place rather than
    reaching -- so lag is not observable on it at all: the frame-quantised
    cross-correlation `lag_ms` computes lands at 0 frames for nearly every
    configuration on that recording, even for settings independently measured
    at 31.8 mm (~106 ms) of lag on a moving signal. `lag_ms`/`overshoot_frac`
    remain here, correct, for a future recording with real reaching motion;
    the decision table below relies on the synthetic measurements instead.
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


def jitter_mm(pos: np.ndarray, tracked: np.ndarray) -> float:
    """Median deviation from a 5-sample centred moving average, in mm.

    Computed only within runs of consecutively TRACKED frames: held/frozen
    duplicate frames, and the position jump at reacquisition, are not sensor
    jitter, so they must not contribute to it. Each run supplies its own
    local moving average (never spanning outside the run), and the 5-mm
    baseline is the MEDIAN over all runs pooled together, not a per-array
    mean -- this is the design spec's measurement method (docs/superpowers/
    specs/2026-07-22-online-pose-filter-design.md), and is what reproduces
    its quoted 2.67 mm (left) / 3.15 mm (right) raw baseline; a mean over the
    whole array including untracked frames does not.
    """
    pos = np.asarray(pos, dtype=float)
    tracked = np.asarray(tracked, dtype=bool)
    if len(pos) != len(tracked):
        raise ValueError("pos and tracked must have the same length")
    devs: list[float] = []
    n = len(pos)
    i = 0
    while i < n:
        if not tracked[i]:
            i += 1
            continue
        j = i
        while j < n and tracked[j]:
            j += 1
        run = pos[i:j]
        if len(run) >= 5:
            devs.extend(
                float(np.linalg.norm(run[k] - run[k - 2:k + 3].mean(axis=0)))
                for k in range(2, len(run) - 2)
            )
        i = j
    return float(np.median(devs) * 1000.0) if devs else 0.0


def lag_ms(raw: np.ndarray, filt: np.ndarray, fps: float) -> float:
    """Frame shift of `filt` vs `raw` minimising squared error, in ms.

    Correct, but only informative when `raw` actually moves enough for the
    cross-correlation optimum to be non-flat -- see the module docstring.
    Not used for the decision table; kept for a future recording with real
    reaching motion. `synthetic_ramp_lag` below is what the table uses.
    """
    raw = np.asarray(raw, float)
    filt = np.asarray(filt, float)
    best_k, best_err = 0, np.inf
    for k in range(0, min(30, len(raw) // 4)):
        err = float(np.mean((raw[:len(raw) - k] - filt[k:]) ** 2))
        if err < best_err:
            best_err, best_k = err, k
    return best_k * 1000.0 / fps


def overshoot_frac(raw: np.ndarray, filt: np.ndarray) -> float:
    """Max excursion of `filt` past `raw`'s range, as a fraction of its span.

    Needs a clean excursion to measure against; the real recording has none
    (see the module docstring), so this is not used on it for the decision
    table. Kept for a future recording with real reaching motion, and reused
    directly by `synthetic_step_overshoot` below on synthetic step data.
    """
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
    jitter_mm: float        # REAL data: median dev., consecutively-tracked runs
    lag_mm_synth: float      # SYNTHETIC 0.3 m/s ramp: steady-state positional lag
    lag_ms_synth: float      # same, as time: lag_mm_synth / v
    overshoot_synth: float   # SYNTHETIC step: overshoot fraction
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


_IDENT_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def synthetic_ramp_lag(filter_factory, v: float = 0.3, fps: float = 30.0,
                       n: int = 200) -> tuple[float, float]:
    """Steady-state lag of `filter_factory` on a noiseless constant-velocity ramp.

    Lag is a property of the filter's DYNAMICS, not the sensor noise, so it
    is measured on controlled synthetic motion rather than on the real
    recording -- see the module docstring for why lag is not observable
    there at all.

    Measures the steady-state POSITIONAL offset directly: the mean of
    (raw - filtered) position over the back half of the run, once the
    initial transient has settled. This is deliberately NOT `lag_ms`'s
    frame-quantised cross-correlation -- a positional difference has
    sub-frame precision, so it resolves lag differences smaller than one
    frame interval (33 ms at 30 Hz), which cross-correlation cannot.

    `v=0.3` m/s and `n=200` (frames at 30 Hz => ~6.7 s) are a representative
    teleop speed and enough frames to reach steady state for the filter
    settings this benchmark sweeps; a caller characterising much heavier
    smoothing (a much lower cutoff / longer time constant) should pass a
    larger `n` so the back-half tail window starts after the transient has
    actually settled.

    Returns (lag_mm, lag_ms), with lag_ms = lag_mm / v.
    """
    dt = 1.0 / fps
    times = np.arange(n) * dt
    pos = np.stack([v * times, np.zeros(n), np.zeros(n)], axis=1)
    quat = np.tile(_IDENT_QUAT, (n, 1))
    tracked = np.ones(n, dtype=bool)
    filt, _, _ = run_filter(pos, quat, tracked, times, filter_factory)
    tail = slice(n // 2, n)
    lag_mm = float(np.mean(pos[tail, 0] - filt[tail, 0]) * 1000.0)
    return lag_mm, lag_mm / v


def synthetic_step_overshoot(filter_factory, fps: float = 30.0,
                            n: int = 200, step_m: float = 0.05) -> float:
    """Overshoot of `filter_factory` on a synthetic step input.

    Overshoot, like lag, is a property of the filter's dynamics, so it is
    measured on controlled synthetic motion instead of the real recording,
    which has no clean step to measure against (see the module docstring).
    Reuses `overshoot_frac` on a synthetic raw/filtered pair rather than
    duplicating its logic.
    """
    dt = 1.0 / fps
    times = np.arange(n) * dt
    raw = np.zeros((n, 3))
    raw[n // 4:, 0] = step_m
    quat = np.tile(_IDENT_QUAT, (n, 1))
    tracked = np.ones(n, dtype=bool)
    filt, _, _ = run_filter(raw, quat, tracked, times, filter_factory)
    return overshoot_frac(raw, filt)


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
    results = [BenchResult("raw", "-", jitter_mm(pos, tracked),
                           0.0, 0.0, 0.0, len(times), 0)]
    for name, params, factory in _sweep():
        p, _, stale = run_filter(pos, quat, tracked, times, factory)
        lag_mm, lag_ms_synth = synthetic_ramp_lag(factory)
        results.append(BenchResult(
            name, params, jitter_mm(p, tracked), lag_mm, lag_ms_synth,
            synthetic_step_overshoot(factory), len(times), int(stale.sum()),
        ))
    return results


def format_bench(results: list[BenchResult]) -> str:
    header = (f"{'filter':<10} {'params':<28} {'jitter_REAL(mm)':>16} "
              f"{'lag_SYNTH(mm)':>14} {'lag_SYNTH(ms)':>14} "
              f"{'overshoot_SYNTH':>16} {'stale':>6}")
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.name:<10} {r.params:<28} {r.jitter_mm:>16.2f} "
            f"{r.lag_mm_synth:>14.2f} {r.lag_ms_synth:>14.1f} "
            f"{r.overshoot_synth:>16.1%} {r.n_stale:>6d}")
    lines.append("")
    lines.append(
        "jitter_REAL = median deviation from a local 5-frame moving average, "
        "over consecutively-tracked runs of the REAL recording. "
        "lag_SYNTH / overshoot_SYNTH = measured on a SYNTHETIC 0.3 m/s ramp "
        "/ step, not the real recording: the real recording covers only "
        "~3.3-3.5 cm of travel over 8 s, too little motion for lag to be "
        "observable at all (the hand jiggled in place rather than reaching). "
        "At comparable lag, lower jitter wins. Overshoot > ~5% means the "
        "filter rings after fast motion.")
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
