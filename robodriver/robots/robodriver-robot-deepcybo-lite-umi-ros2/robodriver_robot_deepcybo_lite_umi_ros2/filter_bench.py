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
