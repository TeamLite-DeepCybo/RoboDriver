"""Per-episode collection quality gates (spec 2026-07-20, stage 2).

Pure logic: thresholds in, pass/fail out. No I/O and no pandas/pyarrow, so the
gate policy can be tested without building datasets. Dataset loading lives in
qc_episode.py.

The gates exist to be run AT THE RIG between episodes. An episode found bad
now costs ~30 s to redo; found next week it is simply lost, because the object
placement, lighting and hand motion cannot be recreated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoids a runtime scipy import via .smoothing
    from .smoothing import ArmCoverage

# Gripper opening columns inside the 23-dim observation.state vector.
# Restated here rather than imported (config.py needs the lerobot env); pinned
# against the recorded feature names by the real-dataset test.
GRIPPER_COL: dict[str, int] = {"left": 7, "right": 15}


@dataclass(frozen=True)
class QCThresholds:
    """Gate bars. Defaults are the spec's; all overridable per session."""
    gripper_range_min: float = 0.05
    picking_usable_min: float = 0.95
    picking_max_unfillable: int = 0
    picking_raw_tracked_min: float = 0.90
    steadying_usable_min: float = 0.90
    duration_min_s: float = 5.0
    duration_max_s: float = 20.0
    n_cameras: int = 3


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class EpisodeQC:
    passed: bool
    results: tuple[GateResult, ...]

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if not r.passed)


def usable_fraction(measured: int, interpolated: int, n: int) -> float:
    """Fraction of frames carrying a real or reconstructed pose."""
    if n <= 0:
        return 0.0
    return (measured + interpolated) / n


def evaluate_gates(
    *,
    coverage: dict[str, "ArmCoverage"],
    raw_tracked_frac: dict[str, float],
    gripper_range: dict[str, float],
    camera_frame_counts: dict[str, int],
    n_frames: int,
    duration_s: float,
    picking_arm: str = "right",
    thresholds: QCThresholds = QCThresholds(),
) -> EpisodeQC:
    """Apply every stage-2 gate. `coverage` maps arm -> ArmCoverage."""
    if picking_arm not in ("left", "right"):
        raise ValueError(f"picking_arm must be 'left' or 'right', got {picking_arm!r}")
    steadying_arm = "right" if picking_arm == "left" else "left"
    t = thresholds
    pick, steady = coverage[picking_arm], coverage[steadying_arm]
    pick_usable = usable_fraction(pick.measured, pick.interpolated, pick.n)
    steady_usable = usable_fraction(steady.measured, steady.interpolated, steady.n)
    pick_raw = raw_tracked_frac[picking_arm]
    grip = gripper_range[picking_arm]

    results = [
        # The single highest-value check: catches a constant-0.0 encoder stub
        # AND a demonstration where the grasp simply never happened.
        GateResult(
            "gripper_moved", grip > t.gripper_range_min,
            f"{picking_arm} gripper range {grip:.3f} "
            f"(need > {t.gripper_range_min})",
        ),
        GateResult(
            "picking_usable", pick_usable >= t.picking_usable_min,
            f"{picking_arm} usable {pick_usable:.1%} "
            f"(need >= {t.picking_usable_min:.0%})",
        ),
        GateResult(
            "picking_unfillable", pick.unfillable <= t.picking_max_unfillable,
            f"{picking_arm} unfillable {pick.unfillable} "
            f"(need <= {t.picking_max_unfillable})",
        ),
        # Raw floor: without it, smoothing masks a degrading rig — usable stays
        # green while real tracking rots, because the smoother keeps recovering.
        GateResult(
            "picking_raw_tracked", pick_raw >= t.picking_raw_tracked_min,
            f"{picking_arm} raw tracked {pick_raw:.1%} "
            f"(need >= {t.picking_raw_tracked_min:.0%})",
        ),
        GateResult(
            "steadying_usable", steady_usable >= t.steadying_usable_min,
            f"{steadying_arm} usable {steady_usable:.1%} "
            f"(need >= {t.steadying_usable_min:.0%})",
        ),
        GateResult(
            "cameras",
            len(camera_frame_counts) == t.n_cameras
            and all(c == n_frames for c in camera_frame_counts.values()),
            f"{len(camera_frame_counts)}/{t.n_cameras} streams, counts "
            f"{dict(sorted(camera_frame_counts.items()))} vs {n_frames} frames",
        ),
        GateResult(
            "duration", t.duration_min_s <= duration_s <= t.duration_max_s,
            f"{duration_s:.1f}s (need {t.duration_min_s}-{t.duration_max_s}s)",
        ),
    ]
    return EpisodeQC(passed=all(r.passed for r in results), results=tuple(results))


def format_qc(qc: EpisodeQC, episode_index: int) -> str:
    """Operator-facing one-screen verdict."""
    head = "PASS" if qc.passed else "FAIL"
    lines = [f"episode_{episode_index:06d}   {head}"]
    for r in qc.results:
        lines.append(f"  {'ok  ' if r.passed else 'FAIL'} {r.name:20s} {r.detail}")
    if not qc.passed:
        lines.append("")
        lines.append("  -> REDO THIS EPISODE NOW (the setup still exists)")
    return "\n".join(lines)
