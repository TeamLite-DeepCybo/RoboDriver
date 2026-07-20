# tests/test_real_episode_e2e.py
"""End-to-end check against the real 2026-07-15 rig recording, when present.

Validates the smoothed output with the same assertions used to verify the
raw recording (2026-07-17 review): shapes, finiteness, unit quats, action
mirror — plus the smoother's own guarantees.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes import main
from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import (
    INTERPOLATED, MEASURED, UNFILLABLE,
)

REAL = Path(__file__).resolve().parents[4].parent / "umi_real_rec_2026-07-15"

pytestmark = pytest.mark.skipif(
    not REAL.is_dir(), reason=f"real recording not found at {REAL}"
)


def test_real_episode_smooths_clean(tmp_path):
    out = tmp_path / "smoothed"
    assert main(["--root", str(REAL), "--out", str(out)]) == 0

    df = pd.read_parquet(out / "data/chunk-000/episode_000000.parquet")
    S = np.stack(df["observation.state"]).astype(np.float64)
    A = np.stack(df["action"]).astype(np.float64)
    P = np.stack(df["observation.provenance"])

    assert S.shape == (240, 23) and A.shape == (240, 16) and P.shape == (240, 2)
    assert np.isfinite(S).all() and np.isfinite(A).all()
    assert (A == S[:, :16]).all()
    for q in (S[:, 3:7], S[:, 11:15]):
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)

    raw = pd.read_parquet(REAL / "data/chunk-000/episode_000000.parquet")
    S_raw = np.stack(raw["observation.state"]).astype(np.float64)
    # measured frames bit-exact per arm
    for prov_col, cols in ((0, slice(0, 7)), (1, slice(8, 15))):
        measured = P[:, prov_col] == MEASURED
        assert (S[measured, cols] == S_raw[measured, cols]).all()
    # quality dims byte-identical
    assert (S[:, 16:23] == S_raw[:, 16:23]).all()
    # known counts from the 2026-07-17 review: left tracked 178, right 197
    assert int((P[:, 0] == MEASURED).sum()) == 178
    assert int((P[:, 1] == MEASURED).sum()) == 197
    # everything not measured was either interpolated or explicitly unfillable
    assert set(np.unique(P)) <= {MEASURED, INTERPOLATED, UNFILLABLE}
