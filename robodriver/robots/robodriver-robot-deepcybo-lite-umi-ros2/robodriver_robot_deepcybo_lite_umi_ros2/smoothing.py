"""Offline gap interpolation for recorded UMI eef episodes.

Pure pose math — no dataset I/O, no ROS, no robodriver imports (it must run
with numpy+scipy alone). Spec: docs/superpowers/specs/
2026-07-20-umi-offline-smoother-design.md.

Provenance values (float, stored in the observation.provenance feature):
  MEASURED     — tracked==1 anchor frame, copied bit-exact
  INTERPOLATED — synthesized by lerp(pos) + Slerp(quat) between anchors
  UNFILLABLE   — gap too long or unbracketed; held input pose retained
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

MEASURED = 0.0
INTERPOLATED = 1.0
UNFILLABLE = 2.0

# Full observation.state width (16 eef/gripper dims + 7 quality dims). NOT the
# same thing as config.STATE_DIM, which is 16 (eef/gripper dims only, the
# adapter's policy-input width) -- deliberately distinct names so the two
# meanings can never be confused within this package.
FULL_STATE_DIM = 23
ACTION_DIM = 16


@dataclass(frozen=True)
class ArmLayout:
    """Column layout of one arm inside the 23-dim observation.state vector.

    Mirrors the adapter's feature-name contract (config.EEF_FEATURE_NAMES +
    QUALITY_FEATURE_NAMES); config.py itself needs the lerobot env, so the
    indices are restated here. The real guard against layout drift is
    tests/test_real_episode_e2e.py::test_layout_matches_recorded_feature_names,
    which reads the actual recorded meta/info.json feature names and checks
    ARM_LAYOUT against them. tests/test_smoothing.py::
    test_layout_matches_feature_name_contract only pins these hardcoded
    indices against hardcoded literals -- it never touches real data, so it
    cannot detect a genuine recorder layout change.
    """
    pos: slice      # x, y, z
    quat: slice     # qx, qy, qz, qw (scipy order)
    tracked: int    # quality column: 1.0 = genuinely measured by PnP


ARM_LAYOUT: dict[str, ArmLayout] = {
    "left": ArmLayout(pos=slice(0, 3), quat=slice(3, 7), tracked=16),
    "right": ArmLayout(pos=slice(8, 11), quat=slice(11, 15), tracked=19),
}


def bracketed_runs(anchors: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield (a, b) pairs of consecutive anchor indices with a gap between.

    Leading/trailing non-anchor runs have no bracket and are not yielded.
    """
    idx = np.flatnonzero(anchors)
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a + 1:
            yield int(a), int(b)


def smooth_arm(
    times: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    anchors: np.ndarray,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill non-anchor frames between anchors; anchors pass through bit-exact.

    A gap is fillable iff its anchor-to-anchor span t[b]-t[a] <= max_gap_s
    (spec: the span covers the missing frames PLUS one frame interval on each
    side). scipy Slerp takes the short arc regardless of quaternion hemisphere
    (double cover) — pinned by test_sign_flipped_anchor_takes_short_arc.

    Returns (pos_out, quat_out, provenance), provenance per MEASURED/
    INTERPOLATED/UNFILLABLE.
    """
    times = np.asarray(times, dtype=float)
    anchors = np.asarray(anchors, dtype=bool)
    pos_out = np.array(pos, copy=True)
    quat_out = np.array(quat, copy=True)
    prov = np.full(len(times), UNFILLABLE, dtype=np.float32)
    prov[anchors] = MEASURED
    n_anchors = int(anchors.sum())
    if n_anchors < 2:
        warnings.warn(
            f"smooth_arm: only {n_anchors} anchor(s) in {len(times)} frames -- "
            "nothing to bracket, all non-anchor frames left UNFILLABLE",
            stacklevel=2,
        )
        return pos_out, quat_out, prov

    for a, b in bracketed_runs(anchors):
        if times[b] - times[a] > max_gap_s:
            continue
        slerp = Slerp(
            [times[a], times[b]], Rotation.from_quat([quat[a], quat[b]])
        )
        span = times[b] - times[a]
        for k in range(a + 1, b):
            w = (times[k] - times[a]) / span
            pos_out[k] = (1.0 - w) * pos[a] + w * pos[b]
            quat_out[k] = slerp([times[k]]).as_quat()[0]
            prov[k] = INTERPOLATED
    return pos_out, quat_out, prov


GRIPPER_COLS = (7, 15)
QUALITY_COLS = slice(16, 23)


def smooth_state(
    times: np.ndarray, state: np.ndarray, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth both arms of an (N, 23) state matrix independently.

    Returns (state_out, provenance) with provenance (N, 2) ordered
    [left, right]. Gripper and quality columns pass through untouched.
    Raises ValueError on wrong state shape or non-monotonic timestamps.
    """
    state = np.asarray(state)
    times = np.asarray(times, dtype=float)
    if state.ndim != 2 or state.shape[1] != FULL_STATE_DIM:
        raise ValueError(
            f"state must be (N, {FULL_STATE_DIM}), got {state.shape}"
        )
    if len(times) != len(state):
        raise ValueError("times and state length mismatch")
    dt = np.diff(times)
    if len(dt) and dt.min() <= 0:
        bad = int(np.argmin(dt)) + 1
        raise ValueError(
            f"timestamps not strictly monotonic at index {bad} "
            f"(t[{bad - 1}]={times[bad - 1]}, t[{bad}]={times[bad]})"
        )

    out = np.array(state, copy=True)
    prov = np.zeros((len(state), 2), dtype=np.float32)
    for col, arm in enumerate(("left", "right")):
        lay = ARM_LAYOUT[arm]
        anchors = state[:, lay.tracked] > 0.5
        pos_out, quat_out, arm_prov = smooth_arm(
            times, state[:, lay.pos], state[:, lay.quat], anchors, max_gap_s
        )
        out[:, lay.pos] = pos_out
        out[:, lay.quat] = quat_out
        prov[:, col] = arm_prov
    return out, prov


def regen_action(state_out: np.ndarray) -> np.ndarray:
    """Rebuild the action matrix from smoothed state (adapter invariant
    action == state[:, :16], spec §Architecture)."""
    return np.array(state_out[:, :ACTION_DIM], copy=True)


@dataclass(frozen=True)
class ArmCoverage:
    n: int
    measured: int
    interpolated: int
    unfillable: int
    filled_gap_hist: dict[int, int]     # filled-gap length in frames -> count
    unfilled_gap_hist: dict[int, int]   # unfilled (rejected) gap length -> count
    longest_filled_gap_s: float         # longest anchor-to-anchor span that WAS filled
    longest_unfilled_gap_s: float       # longest bracketed span that was REJECTED
    # (e.g. by --max-gap-s) and left UNFILLABLE; this is the number that
    # tells a user how much to raise --max-gap-s by. Leading/trailing
    # unbracketed runs are not bracketed spans and are not counted in either
    # histogram or longest value, only in `unfillable`.


def arm_coverage(
    times: np.ndarray, anchors: np.ndarray, provenance: np.ndarray
) -> ArmCoverage:
    times = np.asarray(times, dtype=float)
    anchors = np.asarray(anchors, dtype=bool)
    filled_hist: dict[int, int] = {}
    unfilled_hist: dict[int, int] = {}
    longest_filled = 0.0
    longest_unfilled = 0.0
    for a, b in bracketed_runs(anchors):
        span = float(times[b] - times[a])
        n_frames = b - a - 1
        if (provenance[a + 1:b] == INTERPOLATED).all():
            filled_hist[n_frames] = filled_hist.get(n_frames, 0) + 1
            longest_filled = max(longest_filled, span)
        else:
            unfilled_hist[n_frames] = unfilled_hist.get(n_frames, 0) + 1
            longest_unfilled = max(longest_unfilled, span)
    return ArmCoverage(
        n=len(times),
        measured=int((provenance == MEASURED).sum()),
        interpolated=int((provenance == INTERPOLATED).sum()),
        unfillable=int((provenance == UNFILLABLE).sum()),
        filled_gap_hist=filled_hist,
        unfilled_gap_hist=unfilled_hist,
        longest_filled_gap_s=longest_filled,
        longest_unfilled_gap_s=longest_unfilled,
    )
