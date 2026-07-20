# tests/test_smooth_episodes.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fixture import default_state, make_tiny_dataset  # noqa: E402

from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import (  # noqa: E402
    INTERPOLATED, MEASURED, ArmCoverage,
)
from robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes import (  # noqa: E402
    process_episode_parquet,
)
import robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes as smooth_episodes_mod  # noqa: E402


@pytest.fixture()
def raw_ds(tmp_path):
    """Raw tiny dataset with a 2-frame left dropout injected."""
    state = default_state(6)
    state[2:4, 16] = 0.0          # left not tracked
    state[2:4, 0:7] = state[1, 0:7]  # hold-last corruption
    root = tmp_path / "raw"
    make_tiny_dataset(root, with_provenance=False, state=state)
    return root


def test_process_episode_adds_provenance_and_interpolates(raw_ds, tmp_path):
    src = raw_ds / "data/chunk-000/episode_000000.parquet"
    dst = tmp_path / "out.parquet"
    result = process_episode_parquet(src, dst, max_gap_s=0.25)

    df = pd.read_parquet(dst)
    assert "observation.provenance" in df.columns
    prov = np.stack(df["observation.provenance"])
    assert prov.shape == (6, 2)
    assert (prov[2:4, 0] == INTERPOLATED).all()
    assert (prov[:, 1] == MEASURED).all()

    raw = pd.read_parquet(src)
    s_in = np.stack(raw["observation.state"])
    s_out = np.stack(df["observation.state"])
    # measured rows bit-exact; corrupted rows changed
    assert (s_out[[0, 1, 4, 5]] == s_in[[0, 1, 4, 5]]).all()
    assert not (s_out[2:4, 0:7] == s_in[2:4, 0:7]).all()
    # action regenerated to mirror
    a_out = np.stack(df["action"])
    assert (a_out == s_out[:, :16]).all()
    # coverage reported
    assert result.coverage["left"].interpolated == 2
    assert result.coverage["right"].interpolated == 0


def test_output_schema_and_hf_metadata(raw_ds, tmp_path):
    src = raw_ds / "data/chunk-000/episode_000000.parquet"
    dst = tmp_path / "out.parquet"
    process_episode_parquet(src, dst, max_gap_s=0.25)

    schema = pq.read_schema(dst)
    prov_field = schema.field("observation.provenance")
    assert str(prov_field.type) == "fixed_size_list<element: float>[2]"
    # column order: original columns first, provenance appended last
    assert schema.names[-1] == "observation.provenance"
    hf = json.loads(schema.metadata[b"huggingface"])
    entry = hf["info"]["features"]["observation.provenance"]
    assert entry == {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": 2, "_type": "List",
    }
    # original entries preserved
    assert hf["info"]["features"]["observation.state"]["length"] == 23


def test_non_pose_columns_pass_through(raw_ds, tmp_path):
    src = raw_ds / "data/chunk-000/episode_000000.parquet"
    dst = tmp_path / "out.parquet"
    process_episode_parquet(src, dst, max_gap_s=0.25)
    raw, out = pd.read_parquet(src), pd.read_parquet(dst)
    for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert (out[col].to_numpy() == raw[col].to_numpy()).all()
    s_in, s_out = np.stack(raw["observation.state"]), np.stack(out["observation.state"])
    assert (s_out[:, 16:23] == s_in[:, 16:23]).all()


def test_measured_anchor_mismatch_raises(raw_ds, tmp_path, monkeypatch):
    """If arm_coverage's `measured` disagrees with the anchor count that
    process_episode_parquet computed itself, that's smooth_state and
    process_episode_parquet disagreeing on the anchor mask -- must raise."""
    src = raw_ds / "data/chunk-000/episode_000000.parquet"
    dst = tmp_path / "out.parquet"

    real_arm_coverage = smooth_episodes_mod.arm_coverage

    def fake_arm_coverage(times, anchors, provenance):
        cov = real_arm_coverage(times, anchors, provenance)
        return ArmCoverage(
            n=cov.n,
            measured=cov.measured + 1,  # disagrees with true anchor count
            interpolated=cov.interpolated,
            unfillable=cov.unfillable,
            gap_hist=cov.gap_hist,
            longest_gap_s=cov.longest_gap_s,
        )

    monkeypatch.setattr(smooth_episodes_mod, "arm_coverage", fake_arm_coverage)

    with pytest.raises(AssertionError, match="disagree on the anchor mask"):
        process_episode_parquet(src, dst, max_gap_s=0.25)
