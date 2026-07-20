# robodriver_robot_deepcybo_lite_umi_ros2/smooth_episodes.py
"""Offline episode smoother: dataset I/O + CLI (spec 2026-07-20).

Reads a recorded UMI eef LeRobot v2.1 dataset, interpolates dropout frames
between tracked anchors (smoothing.py), and writes a NEW dataset with an
appended observation.provenance feature. Direct file manipulation only —
pandas/pyarrow, never DoRobotDataset (see spec §Architecture for why).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .smoothing import (
    ARM_LAYOUT, ArmCoverage, arm_coverage, regen_action, smooth_state,
)

PROVENANCE_KEY = "observation.provenance"
PROVENANCE_FEATURE = {
    "dtype": "float32",
    "names": ["left_provenance", "right_provenance"],
    "shape": [2],
}


def _hf_provenance_entry() -> dict:
    return {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": 2,
        "_type": "List",
    }


def _fsl(arr2d: np.ndarray) -> pa.FixedSizeListArray:
    a = np.ascontiguousarray(arr2d, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(a.reshape(-1), type=pa.float32()), a.shape[1]
    )


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    coverage: dict[str, ArmCoverage]


def process_episode_parquet(
    src: Path, dst: Path, max_gap_s: float
) -> EpisodeResult:
    """Smooth one episode parquet from src into dst (dst parent must exist)."""
    table_in = pq.read_table(src)
    df = table_in.to_pandas()
    times = df["timestamp"].to_numpy(dtype=float)
    state_in = np.stack(df["observation.state"].to_numpy())

    state_out, prov = smooth_state(times, state_in, max_gap_s)
    action_out = regen_action(state_out)

    coverage = {}
    for col, arm in enumerate(("left", "right")):
        anchors = state_in[:, ARM_LAYOUT[arm].tracked] > 0.5
        coverage[arm] = arm_coverage(times, anchors, prov[:, col])
        held_pre = coverage[arm].n - coverage[arm].measured
        # Post-condition (spec §Error handling): smoothing must never
        # increase the number of bad frames.
        assert coverage[arm].unfillable <= held_pre, (
            f"{arm}: unfillable {coverage[arm].unfillable} > held {held_pre}"
        )

    # Rebuild the table: original column order, pose columns replaced,
    # provenance appended last.
    columns: dict[str, pa.Array] = {}
    for name in table_in.schema.names:
        if name == "observation.state":
            columns[name] = _fsl(state_out)
        elif name == "action":
            columns[name] = _fsl(action_out)
        else:
            columns[name] = table_in.column(name).combine_chunks()
    columns[PROVENANCE_KEY] = _fsl(prov)

    table_out = pa.table(columns)
    meta = dict(table_in.schema.metadata or {})
    if b"huggingface" in meta:
        hf = json.loads(meta[b"huggingface"])
        hf["info"]["features"][PROVENANCE_KEY] = _hf_provenance_entry()
        meta[b"huggingface"] = json.dumps(hf).encode()
    table_out = table_out.replace_schema_metadata(meta)
    pq.write_table(table_out, dst)

    ep_idx = int(df["episode_index"].iloc[0]) if len(df) else 0
    return EpisodeResult(episode_index=ep_idx, coverage=coverage)
