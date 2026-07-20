# tests/test_canonical_reader.py
"""Spike (plan Task 1): a dataset carrying an extra observation.provenance
column must be readable by the canonical DoRobotDataset reader.

Skips where the RoboDriver env is absent (Windows dev box); the Linux rig run
is the actual gate. If this FAILS on Linux, stop and switch to the spec's
sidecar meta/provenance.jsonl fallback before building Tasks 4-6.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fixture import N, make_tiny_dataset  # noqa: E402

robodriver = pytest.importorskip(
    "robodriver.dataset.dorobot_dataset",
    reason="canonical-reader spike needs the RoboDriver (Linux) env",
)


def test_dorobot_dataset_reads_provenance_column(tmp_path):
    make_tiny_dataset(tmp_path / "ds", with_provenance=True)
    ds = robodriver.DoRobotDataset("spike/tiny", root=tmp_path / "ds")
    assert ds.meta.total_frames == N
    assert "observation.provenance" in ds.features
    item = ds[0]
    prov = np.asarray(item["observation.provenance"])
    assert prov.shape == (2,)
    assert prov.tolist() == [0.0, 0.0]
    state = np.asarray(item["observation.state"])
    assert state.shape == (23,)
