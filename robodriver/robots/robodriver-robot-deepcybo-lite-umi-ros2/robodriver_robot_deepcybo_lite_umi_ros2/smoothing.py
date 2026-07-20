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

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

MEASURED = 0.0
INTERPOLATED = 1.0
UNFILLABLE = 2.0

STATE_DIM = 23
ACTION_DIM = 16


@dataclass(frozen=True)
class ArmLayout:
    """Column layout of one arm inside the 23-dim observation.state vector.

    Mirrors the adapter's feature-name contract (config.EEF_FEATURE_NAMES +
    QUALITY_FEATURE_NAMES); config.py itself needs the lerobot env, so the
    indices are restated here and pinned by test_layout_matches_feature_name_contract.
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
    if int(anchors.sum()) < 2:
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
