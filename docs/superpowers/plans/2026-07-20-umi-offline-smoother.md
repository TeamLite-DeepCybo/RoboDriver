# UMI Offline Episode Smoother Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI (`umi-smooth-episodes`) that reads a recorded UMI eef LeRobot dataset, rebuilds every non-measured pose frame by lerp+Slerp between `tracked==1` anchors, and writes a new dataset with per-frame provenance — raw dataset untouched.

**Architecture:** Two new modules in the existing adapter package: `smoothing.py` (pure numpy/scipy pose math, no I/O) and `smooth_episodes.py` (dataset I/O via pandas/pyarrow + CLI). Dataset files are manipulated directly — parquet written with pyarrow `fixed_size_list` columns plus the `huggingface` schema-metadata JSON, `meta/` copied and surgically patched, images hardlinked. `DoRobotDataset` is never imported at runtime; canonical-reader compatibility is enforced by a Linux-gated round-trip test (Task 1).

**Tech Stack:** Python ≥3.10, numpy, scipy (`Rotation`/`Slerp`), pandas, pyarrow, pytest. No torch/lerobot/ROS imports in the new modules.

**Spec:** `docs/superpowers/specs/2026-07-20-umi-offline-smoother-design.md` (read it first).

## Global Constraints

- New modules import ONLY stdlib + numpy + scipy + pandas + pyarrow. Never `robodriver.*`, `lerobot*`, `torch`, `rclpy`. (Tests MAY import `robodriver` behind a skipif guard.)
- Measured frames (`tracked==1`) pass through **bit-exact** — output arrays equal input at those rows.
- `observation.state` stays shape (23,); provenance is a NEW feature `observation.provenance`, float32, shape (2,), names `["left_provenance", "right_provenance"]`, values `0.0=MEASURED, 1.0=INTERPOLATED, 2.0=UNFILLABLE`.
- Gap limit is **anchor-to-anchor**: a gap is fillable iff `t[b] - t[a] <= max_gap_s` for bracketing anchors `a, b`. Default `max_gap_s = 0.25`.
- `action` in the output is regenerated as `state_out[:, :16]` exactly.
- Gripper columns (7, 15) and quality columns (16–22) pass through unmodified.
- Left/right arms are processed fully independently.
- The 7 quality dims in the output are byte-identical to the input.
- Raw dataset directory is never written to.
- State column layout (from the recorded `info.json`, do not reorder):
  `0-6` L eef (x,y,z,qx,qy,qz,qw), `7` L grip, `8-14` R eef, `15` R grip, `16` L_tracked, `17` L_present, `18` L_reproj, `19` R_tracked, `20` R_present, `21` R_reproj, `22` world_fresh.
- Working directory for all commands: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2` (the package root). Commit from the repo root.

---

### Task 1: Canonical-reader spike (Linux-gated)

The one unverified risk: can `DoRobotDataset` open a dataset whose parquet we wrote ourselves with an extra `observation.provenance` column? This task builds a tiny synthetic dataset with plain pyarrow (no smoother code — it doesn't exist yet) and a test that opens it with `DoRobotDataset`. The test SKIPs where `robodriver` deps are missing (Windows) and runs on the Linux rig. **If this test fails on Linux, STOP and switch the design to the sidecar `meta/provenance.jsonl` fallback (spec §Output schema) before doing Tasks 4–6.**

**Files:**
- Create: `tests/dataset_fixture.py` (plain helper module — NOT a test module)
- Create: `tests/test_canonical_reader.py`

**Interfaces:**
- Produces (in `tests/dataset_fixture.py`): `make_tiny_dataset(root: Path, with_provenance: bool = True, state: np.ndarray | None = None) -> None` — builds a minimal 1-episode, 6-frame, 1-camera v2.1 dataset on disk; plus `default_state(n) -> np.ndarray`, `STATE_NAMES`, `ACTION_NAMES`, `FPS`, `N`. Tasks 4–6 tests import from this module.

> **Why a separate module:** `pytest.importorskip` at module level raises
> `Skipped` on import. If the fixture lived in `test_canonical_reader.py`,
> importing it from Tasks 4–6 tests would skip THOSE suites entirely on any
> machine without the RoboDriver env. The guard must stay confined to the
> spike test.

- [ ] **Step 1a: Write the fixture module**

```python
# tests/dataset_fixture.py
"""Builds a minimal LeRobot v2.1 UMI dataset on disk for tests.

Writes the v2.1 files directly with pyarrow/json, so it is independent of the
smoother implementation and can produce BOTH the raw input (with_provenance=
False) and a provenance-carrying dataset (for the canonical-reader spike).

NOT a test module — no test collection, and deliberately free of any
pytest.importorskip guard so importing it never skips the caller's suite.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
N = 6  # frames

STATE_NAMES = [
    "left_eef_x.pos", "left_eef_y.pos", "left_eef_z.pos",
    "left_eef_qx.pos", "left_eef_qy.pos", "left_eef_qz.pos", "left_eef_qw.pos",
    "left_gripper.pos",
    "right_eef_x.pos", "right_eef_y.pos", "right_eef_z.pos",
    "right_eef_qx.pos", "right_eef_qy.pos", "right_eef_qz.pos", "right_eef_qw.pos",
    "right_gripper.pos",
    "left_tracked.flag", "left_present.flag", "left_reproj.flag",
    "right_tracked.flag", "right_present.flag", "right_reproj.flag",
    "world_fresh.flag",
]
ACTION_NAMES = STATE_NAMES[:16]


def _fsl(arr2d: np.ndarray) -> pa.FixedSizeListArray:
    """(N, D) float array -> arrow fixed_size_list<float32>[D]."""
    a = np.ascontiguousarray(arr2d, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(a.reshape(-1), type=pa.float32()), a.shape[1]
    )


def _hf_meta(features: dict) -> bytes:
    """The 'huggingface' parquet schema metadata the datasets lib embeds."""
    info = {}
    for key, ft in features.items():
        if ft["dtype"] == "image":
            continue
        if ft["shape"] == [1] or key in (
            "timestamp", "frame_index", "episode_index", "index", "task_index"
        ):
            info[key] = {"dtype": ft["dtype"], "_type": "Value"}
        else:
            info[key] = {
                "feature": {"dtype": ft["dtype"], "_type": "Value"},
                "length": ft["shape"][0],
                "_type": "List",
            }
    return json.dumps({"info": {"features": info}}).encode()


def default_state(n: int = N) -> np.ndarray:
    """A benign all-tracked state matrix (unit quats, flags good)."""
    s = np.zeros((n, 23), dtype=np.float32)
    s[:, 6] = 1.0    # L qw
    s[:, 14] = 1.0   # R qw
    s[:, 0] = np.linspace(0.0, 0.5, n)   # L x moves
    s[:, 8] = np.linspace(0.0, -0.5, n)  # R x moves
    s[:, 16] = 1.0; s[:, 17] = 1.0       # L tracked/present
    s[:, 19] = 1.0; s[:, 20] = 1.0       # R tracked/present
    s[:, 22] = 1.0                        # world_fresh
    s[:, 18] = 0.1; s[:, 21] = 0.1       # reproj
    return s


def make_tiny_dataset(
    root: Path, with_provenance: bool = True, state: np.ndarray | None = None
) -> None:
    n = N if state is None else len(state)
    state = default_state(n) if state is None else np.asarray(state, np.float32)
    action = state[:, :16].copy()
    ts = (np.arange(n) / FPS).astype(np.float32)

    features = {
        "action": {"dtype": "float32", "names": ACTION_NAMES, "shape": [16]},
        "observation.state": {"dtype": "float32", "names": STATE_NAMES, "shape": [23]},
        "observation.images.image_head": {
            "dtype": "image", "names": ["height", "width", "channels"],
            "shape": [480, 640, 3],
        },
        "timestamp": {"dtype": "float32", "names": None, "shape": [1]},
        "frame_index": {"dtype": "int64", "names": None, "shape": [1]},
        "episode_index": {"dtype": "int64", "names": None, "shape": [1]},
        "index": {"dtype": "int64", "names": None, "shape": [1]},
        "task_index": {"dtype": "int64", "names": None, "shape": [1]},
    }
    cols = {
        "action": _fsl(action),
        "observation.state": _fsl(state),
        "timestamp": pa.array(ts, type=pa.float32()),
        "frame_index": pa.array(np.arange(n), type=pa.int64()),
        "episode_index": pa.array(np.zeros(n, np.int64)),
        "index": pa.array(np.arange(n), type=pa.int64()),
        "task_index": pa.array(np.zeros(n, np.int64)),
    }
    if with_provenance:
        features["observation.provenance"] = {
            "dtype": "float32",
            "names": ["left_provenance", "right_provenance"],
            "shape": [2],
        }
        cols["observation.provenance"] = _fsl(np.zeros((n, 2), np.float32))

    table = pa.table(cols)
    table = table.replace_schema_metadata({b"huggingface": _hf_meta(features)})
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")

    img_dir = root / "images" / "observation.images.image_head" / "episode_000000"
    img_dir.mkdir(parents=True)
    # tiny valid JPEG (1x1 white) so image paths exist without a cv2 dependency
    jpg = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
        "07090908080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c23"
        "1c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100"
        "ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc4"
        "00b5100002010303020403050504040000017d01020300041105122131410613516107"
        "227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a34"
        "35363738393a434445464748494a535455565758595a636465666768696a7374757677"
        "78797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7"
        "b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4"
        "f5f6f7f8f9faffda0008010100003f00fbfe8a28a2803fffd9"
    )
    for i in range(n):
        (img_dir / f"frame_{i:06d}.jpg").write_bytes(jpg)

    meta = root / "meta"
    meta.mkdir()
    info = {
        "codebase_version": "v2.1",
        "dorobot_dataset_version": "v1.0",
        "robot_type": None,
        "total_episodes": 1, "total_frames": n, "total_tasks": 1,
        "total_videos": 0, "total_chunks": 1, "chunks_size": 10000,
        "fps": FPS, "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "image_path": "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg",
        "video_path": None, "audio_path": None,
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["tiny"], "length": n}) + "\n",
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "tiny"}) + "\n", encoding="utf-8"
    )
    zeros23 = [0.0] * 23
    stats = {
        "episode_index": 0,
        "stats": {
            key: {
                "min": list(map(float, arr.min(0))),
                "max": list(map(float, arr.max(0))),
                "mean": list(map(float, arr.mean(0))),
                "std": list(map(float, arr.std(0))),
                "count": [n],
            }
            for key, arr in {
                "action": action, "observation.state": state,
                **({"observation.provenance": np.zeros((n, 2), np.float32)}
                   if with_provenance else {}),
            }.items()
        },
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        col = np.asarray(cols[key])
        stats["stats"][key] = {
            "min": [float(col.min())], "max": [float(col.max())],
            "mean": [float(col.mean())], "std": [float(col.std())], "count": [n],
        }
    (meta / "episodes_stats.jsonl").write_text(
        json.dumps(stats) + "\n", encoding="utf-8"
    )
```

- [ ] **Step 1b: Write the gated spike test**

```python
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
```

- [ ] **Step 2: Run — expect SKIP on Windows, PASS on Linux**

Run: `python -m pytest tests/test_canonical_reader.py -v`
Expected (Windows): `SKIPPED ... needs the RoboDriver (Linux) env`
Expected (Linux rig, RoboDriver venv active): `1 passed`

The Linux run is the actual gate. If it fails there: STOP, report the error, and switch Tasks 4–6 to the sidecar fallback per spec.

- [ ] **Step 3: Sanity-check the fixture parquet locally (Windows-runnable)**

Run: `python -c "import sys,tempfile,pathlib; sys.path.insert(0,'tests'); import pandas as pd; from dataset_fixture import make_tiny_dataset; d=pathlib.Path(tempfile.mkdtemp()); make_tiny_dataset(d); df=pd.read_parquet(d/'data/chunk-000/episode_000000.parquet'); print(df.shape, list(df.columns))"`
Expected: `(6, 8) ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'observation.provenance']`

- [ ] **Step 4: Verify the fixture import does NOT skip a plain suite**

Create a scratch check that the guard is properly confined:

Run: `python -c "import sys; sys.path.insert(0,'tests'); import dataset_fixture; print('fixture imports cleanly without robodriver:', dataset_fixture.N)"`
Expected: `fixture imports cleanly without robodriver: 6` (no Skipped exception)

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/dataset_fixture.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_canonical_reader.py
git commit -m "Add canonical-reader spike and reusable tiny-dataset fixture"
```

---

### Task 2: `smoothing.py` — layout constants and per-arm gap interpolation

**Files:**
- Create: `robodriver_robot_deepcybo_lite_umi_ros2/smoothing.py`
- Test: `tests/test_smoothing.py`

**Interfaces:**
- Produces:
  - `MEASURED = 0.0`, `INTERPOLATED = 1.0`, `UNFILLABLE = 2.0` (module floats)
  - `STATE_DIM = 23`, `ACTION_DIM = 16`
  - `ArmLayout(pos: slice, quat: slice, tracked: int)` frozen dataclass; `ARM_LAYOUT: dict[str, ArmLayout]` with keys `"left"`, `"right"`
  - `bracketed_runs(anchors: np.ndarray) -> Iterator[tuple[int, int]]` — consecutive-anchor index pairs `(a, b)` with at least one non-anchor frame between
  - `smooth_arm(times, pos, quat, anchors, max_gap_s) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(pos_out (N,3), quat_out (N,4), provenance (N,))`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_smoothing.py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import (
    ARM_LAYOUT, INTERPOLATED, MEASURED, UNFILLABLE,
    bracketed_runs, smooth_arm,
)


def _traj(n=30, fps=30.0):
    """Smooth known trajectory: line in pos, constant-rate z-rotation."""
    t = np.arange(n) / fps
    pos = np.stack([t * 0.3, np.sin(t), np.full(n, 0.5)], axis=1)
    quat = Rotation.from_euler("z", 60.0 * t, degrees=True).as_quat()
    return t, pos.astype(np.float64), quat.astype(np.float64)


def test_layout_matches_feature_name_contract():
    L, R = ARM_LAYOUT["left"], ARM_LAYOUT["right"]
    assert (L.pos, L.quat, L.tracked) == (slice(0, 3), slice(3, 7), 16)
    assert (R.pos, R.quat, R.tracked) == (slice(8, 11), slice(11, 15), 19)


def test_bracketed_runs_finds_interior_gaps_only():
    #        0  1  2  3  4  5  6
    anchors = np.array([0, 1, 0, 0, 1, 1, 0], dtype=bool)
    assert list(bracketed_runs(anchors)) == [(1, 4)]  # leading 0 / trailing 6 excluded


def test_interpolation_recovers_knocked_out_frames():
    t, pos, quat = _traj()
    anchors = np.ones(len(t), dtype=bool)
    anchors[10:14] = False              # 4-frame gap, anchors t[9]..t[14] = 0.166s
    p_in, q_in = pos.copy(), quat.copy()
    p_in[10:14] = p_in[9]               # simulate hold-last corruption
    q_in[10:14] = q_in[9]
    p_out, q_out, prov = smooth_arm(t, p_in, q_in, anchors, max_gap_s=0.25)
    np.testing.assert_allclose(p_out[10:14], pos[10:14], atol=5e-3)
    for k in range(10, 14):             # orientation within 1 degree of truth
        err = (Rotation.from_quat(q_out[k]) * Rotation.from_quat(quat[k]).inv()).magnitude()
        assert np.degrees(err) < 1.0
    assert (prov[10:14] == INTERPOLATED).all()


def test_anchor_frames_bit_exact():
    t, pos, quat = _traj()
    anchors = np.ones(len(t), dtype=bool)
    anchors[5] = False
    p_out, q_out, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=0.25)
    assert (p_out[anchors] == pos[anchors]).all()
    assert (q_out[anchors] == quat[anchors]).all()
    assert (prov[anchors] == MEASURED).all()


def test_over_long_gap_left_unfillable():
    t, pos, quat = _traj()
    anchors = np.ones(len(t), dtype=bool)
    anchors[5:20] = False               # anchors t[4]..t[20] = 0.533s > 0.25
    p_out, q_out, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=0.25)
    assert (prov[5:20] == UNFILLABLE).all()
    assert (p_out[5:20] == pos[5:20]).all()   # held input passes through


def test_gap_exactly_at_limit_is_filled():
    t, pos, quat = _traj(fps=30.0)
    anchors = np.ones(len(t), dtype=bool)
    # anchors at 9 and 16: span 7/30 s ≈ 0.2333; use max_gap_s exactly equal
    anchors[10:16] = False
    span = t[16] - t[9]
    _, _, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=span)
    assert (prov[10:16] == INTERPOLATED).all()


def test_leading_and_trailing_gaps_unfillable():
    t, pos, quat = _traj()
    anchors = np.ones(len(t), dtype=bool)
    anchors[:3] = False
    anchors[-2:] = False
    _, _, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=10.0)
    assert (prov[:3] == UNFILLABLE).all()
    assert (prov[-2:] == UNFILLABLE).all()


def test_fewer_than_two_anchors_all_unfillable_but_one():
    t, pos, quat = _traj(n=5)
    anchors = np.zeros(5, dtype=bool)
    anchors[2] = True
    _, _, prov = smooth_arm(t, pos, quat, anchors, max_gap_s=1.0)
    assert prov[2] == MEASURED
    assert (prov[[0, 1, 3, 4]] == UNFILLABLE).all()


def test_sign_flipped_anchor_takes_short_arc():
    # 40 deg apart; second anchor hemisphere-flipped (same rotation)
    q0 = Rotation.from_euler("z", 0, degrees=True).as_quat()
    q1 = -Rotation.from_euler("z", 40, degrees=True).as_quat()
    t = np.array([0.0, 0.05, 0.1])
    pos = np.zeros((3, 3))
    quat = np.stack([q0, q0, q1])       # middle frame is a gap
    anchors = np.array([True, False, True])
    _, q_out, _ = smooth_arm(t, pos, quat, anchors, max_gap_s=0.25)
    mid = Rotation.from_quat(q_out[1])
    err = (mid * Rotation.from_euler("z", 20, degrees=True).inv()).magnitude()
    assert np.degrees(err) < 1e-6       # 20 deg = short arc midpoint
    assert abs(np.linalg.norm(q_out[1]) - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_smoothing.py -v`
Expected: FAIL at import — `ModuleNotFoundError: ... smoothing`

- [ ] **Step 3: Write the implementation**

```python
# robodriver_robot_deepcybo_lite_umi_ros2/smoothing.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_smoothing.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/smoothing.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_smoothing.py
git commit -m "Add per-arm offline gap interpolation (lerp + Slerp between tracked anchors)"
```

---

### Task 3: `smoothing.py` — whole-state smoothing, action regen, coverage

**Files:**
- Modify: `robodriver_robot_deepcybo_lite_umi_ros2/smoothing.py` (append)
- Test: `tests/test_smoothing.py` (append)

**Interfaces:**
- Consumes: `smooth_arm`, `ARM_LAYOUT`, constants (Task 2).
- Produces:
  - `smooth_state(times, state, max_gap_s) -> tuple[np.ndarray, np.ndarray]` — `(state_out (N,23), provenance (N,2) [left, right])`; raises `ValueError` on bad shape or non-monotonic timestamps
  - `regen_action(state_out) -> np.ndarray` — `(N,16)` copy of the first 16 columns
  - `ArmCoverage` dataclass: `n, measured, interpolated, unfillable: int`, `gap_hist: dict[int, int]` (filled-gap length in frames → count), `longest_gap_s: float` (longest anchor-to-anchor span containing missing frames, 0.0 if none)
  - `arm_coverage(times, anchors, provenance) -> ArmCoverage`

- [ ] **Step 1: Append the failing tests**

```python
# append to tests/test_smoothing.py
from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import (
    ArmCoverage, arm_coverage, regen_action, smooth_state,
)


def _state(n=30, fps=30.0):
    """Full 23-dim state with both arms following distinct trajectories."""
    t = np.arange(n) / fps
    s = np.zeros((n, 23), dtype=np.float32)
    _, lp, lq = _traj(n, fps)
    s[:, 0:3] = lp
    s[:, 3:7] = lq
    s[:, 7] = np.linspace(0, 1, n)                     # L gripper ramp
    s[:, 8:11] = lp[:, [1, 0, 2]] * -1.0               # different R traj
    s[:, 11:15] = Rotation.from_euler(
        "x", 30.0 * t, degrees=True).as_quat()
    s[:, 15] = np.linspace(1, 0, n)                    # R gripper ramp
    s[:, 16] = 1.0; s[:, 17] = 1.0; s[:, 18] = 0.1     # L quality
    s[:, 19] = 1.0; s[:, 20] = 1.0; s[:, 21] = 0.1     # R quality
    s[:, 22] = 1.0
    return t, s


def test_smooth_state_arms_independent():
    t, s = _state()
    s[10:13, 16] = 0.0                 # left dropout only
    s_in = s.copy()
    out, prov = smooth_state(t, s_in, max_gap_s=0.25)
    # right arm bit-exact everywhere
    assert (out[:, 8:15] == s_in[:, 8:15]).all()
    assert (prov[:, 1] == MEASURED).all()
    # left gap interpolated
    assert (prov[10:13, 0] == INTERPOLATED).all()
    assert not (out[10:13, 0:7] == s_in[10:13, 0:7]).all()


def test_smooth_state_passthrough_columns_untouched():
    t, s = _state()
    s[10:13, 16] = 0.0
    s_in = s.copy()
    out, _ = smooth_state(t, s_in, max_gap_s=0.25)
    assert (out[:, 7] == s_in[:, 7]).all()      # L gripper
    assert (out[:, 15] == s_in[:, 15]).all()    # R gripper
    assert (out[:, 16:23] == s_in[:, 16:23]).all()  # quality dims byte-identical


def test_smooth_state_rejects_bad_input():
    t, s = _state()
    with pytest.raises(ValueError):
        smooth_state(t, s[:, :22], max_gap_s=0.25)   # wrong dim
    t2 = t.copy(); t2[5] = t2[4]                      # non-monotonic
    with pytest.raises(ValueError, match="monotonic"):
        smooth_state(t2, s, max_gap_s=0.25)


def test_regen_action_mirrors_first_16():
    t, s = _state()
    out, _ = smooth_state(t, s, max_gap_s=0.25)
    a = regen_action(out)
    assert a.shape == (len(s), 16)
    assert (a == out[:, :16]).all()
    a[0, 0] = 99.0                                    # must be a copy
    assert out[0, 0] != 99.0


def test_arm_coverage_counts_and_histogram():
    t, s = _state()
    s[5:7, 16] = 0.0      # 2-frame gap (fillable)
    s[20:21, 16] = 0.0    # 1-frame gap (fillable)
    anchors = s[:, 16] > 0.5
    _, prov = smooth_state(t, s, max_gap_s=0.25)
    cov = arm_coverage(t, anchors, prov[:, 0])
    assert cov.n == 30
    assert cov.measured == 27
    assert cov.interpolated == 3
    assert cov.unfillable == 0
    assert cov.gap_hist == {2: 1, 1: 1}
    assert cov.longest_gap_s == pytest.approx(t[7] - t[4])
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_smoothing.py -v -k "smooth_state or regen or coverage"`
Expected: FAIL — `ImportError: cannot import name 'smooth_state'`

- [ ] **Step 3: Append the implementation**

```python
# append to robodriver_robot_deepcybo_lite_umi_ros2/smoothing.py

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
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(
            f"state must be (N, {STATE_DIM}), got {state.shape}"
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
    gap_hist: dict[int, int]     # filled-gap length in frames -> count
    longest_gap_s: float         # longest anchor-to-anchor span with a gap


def arm_coverage(
    times: np.ndarray, anchors: np.ndarray, provenance: np.ndarray
) -> ArmCoverage:
    times = np.asarray(times, dtype=float)
    anchors = np.asarray(anchors, dtype=bool)
    hist: dict[int, int] = {}
    longest = 0.0
    for a, b in bracketed_runs(anchors):
        longest = max(longest, float(times[b] - times[a]))
        if (provenance[a + 1:b] == INTERPOLATED).all():
            n_frames = b - a - 1
            hist[n_frames] = hist.get(n_frames, 0) + 1
    return ArmCoverage(
        n=len(times),
        measured=int((provenance == MEASURED).sum()),
        interpolated=int((provenance == INTERPOLATED).sum()),
        unfillable=int((provenance == UNFILLABLE).sum()),
        gap_hist=hist,
        longest_gap_s=longest,
    )
```

- [ ] **Step 4: Run the full smoothing suite**

Run: `python -m pytest tests/test_smoothing.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/smoothing.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_smoothing.py
git commit -m "Add whole-state smoothing, action regen, and per-arm coverage stats"
```

---

### Task 4: `smooth_episodes.py` — episode parquet read/transform/write

**Files:**
- Create: `robodriver_robot_deepcybo_lite_umi_ros2/smooth_episodes.py`
- Test: `tests/test_smooth_episodes.py`

**Interfaces:**
- Consumes: `smooth_state`, `regen_action`, `arm_coverage`, `ARM_LAYOUT`, `ArmCoverage` (Tasks 2–3); `make_tiny_dataset` (Task 1) as the test fixture.
- Produces:
  - `PROVENANCE_FEATURE = {"dtype": "float32", "names": ["left_provenance", "right_provenance"], "shape": [2]}`
  - `EpisodeResult` dataclass: `episode_index: int`, `coverage: dict[str, ArmCoverage]`
  - `process_episode_parquet(src: Path, dst: Path, max_gap_s: float) -> EpisodeResult` — reads one episode parquet, smooths, writes `dst` with the appended `observation.provenance` column and patched `huggingface` schema metadata; regenerates `action`
  - `_hf_provenance_entry() -> dict` — the metadata entry `{"feature": {"dtype": "float32", "_type": "Value"}, "length": 2, "_type": "List"}`

- [ ] **Step 1: Write the failing tests**

```python
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
    INTERPOLATED, MEASURED,
)
from robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes import (  # noqa: E402
    process_episode_parquet,
)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_smooth_episodes.py -v`
Expected: FAIL at import — `ModuleNotFoundError: ... smooth_episodes`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_smooth_episodes.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/smooth_episodes.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_smooth_episodes.py
git commit -m "Add episode parquet smoothing with appended provenance column"
```

---

### Task 5: `smooth_episodes.py` — dataset-level walk, meta patch, image links

**Files:**
- Modify: `robodriver_robot_deepcybo_lite_umi_ros2/smooth_episodes.py` (append)
- Test: `tests/test_smooth_episodes.py` (append)

**Interfaces:**
- Consumes: `process_episode_parquet`, `PROVENANCE_KEY`, `PROVENANCE_FEATURE` (Task 4); `make_tiny_dataset` fixture (Task 1).
- Produces:
  - `smooth_dataset(root: Path, out: Path, max_gap_s: float, link_images: str = "hard", overwrite: bool = False, dry_run: bool = False) -> list[EpisodeResult]`
  - Raises `FileExistsError` if `out` exists without `overwrite`; `NotImplementedError` if the dataset uses videos (`info["video_path"]` set and `total_videos > 0`); `ValueError` if `root` lacks `meta/info.json`.

- [ ] **Step 1: Append the failing tests**

```python
# append to tests/test_smooth_episodes.py
from robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes import (
    smooth_dataset,
)


def test_smooth_dataset_end_to_end(raw_ds, tmp_path):
    out = tmp_path / "smoothed"
    results = smooth_dataset(raw_ds, out, max_gap_s=0.25)
    assert len(results) == 1

    # info.json: provenance feature added, everything else preserved
    info_in = json.loads((raw_ds / "meta/info.json").read_text(encoding="utf-8"))
    info_out = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))
    assert info_out["features"]["observation.provenance"] == {
        "dtype": "float32",
        "names": ["left_provenance", "right_provenance"],
        "shape": [2],
    }
    for k, v in info_in.items():
        if k != "features":
            assert info_out[k] == v
    for k, v in info_in["features"].items():
        assert info_out["features"][k] == v

    # episodes/tasks copied verbatim
    for name in ("episodes.jsonl", "tasks.jsonl"):
        assert (out / "meta" / name).read_bytes() == (raw_ds / "meta" / name).read_bytes()

    # stats: recomputed for state/action, added for provenance, others verbatim
    stats_in = json.loads((raw_ds / "meta/episodes_stats.jsonl").read_text(encoding="utf-8"))
    stats_out = json.loads((out / "meta/episodes_stats.jsonl").read_text(encoding="utf-8"))
    assert "observation.provenance" in stats_out["stats"]
    assert stats_out["stats"]["timestamp"] == stats_in["stats"]["timestamp"]
    df = pd.read_parquet(out / "data/chunk-000/episode_000000.parquet")
    s = np.stack(df["observation.state"])
    assert stats_out["stats"]["observation.state"]["min"] == pytest.approx(
        s.min(0).tolist()
    )

    # images exist in the output tree
    img = out / "images/observation.images.image_head/episode_000000/frame_000000.jpg"
    assert img.is_file()
    assert img.read_bytes() == (
        raw_ds / "images/observation.images.image_head/episode_000000/frame_000000.jpg"
    ).read_bytes()

    # raw dataset untouched (no provenance in the input parquet)
    raw_df = pd.read_parquet(raw_ds / "data/chunk-000/episode_000000.parquet")
    assert "observation.provenance" not in raw_df.columns


def test_smooth_dataset_refuses_existing_out(raw_ds, tmp_path):
    out = tmp_path / "smoothed"
    out.mkdir()
    with pytest.raises(FileExistsError):
        smooth_dataset(raw_ds, out, max_gap_s=0.25)
    smooth_dataset(raw_ds, out, max_gap_s=0.25, overwrite=True)  # ok


def test_smooth_dataset_dry_run_writes_nothing(raw_ds, tmp_path):
    out = tmp_path / "smoothed"
    results = smooth_dataset(raw_ds, out, max_gap_s=0.25, dry_run=True)
    assert len(results) == 1
    assert results[0].coverage["left"].interpolated == 2
    assert not out.exists()


def test_smooth_dataset_copy_mode(raw_ds, tmp_path):
    out = tmp_path / "smoothed"
    smooth_dataset(raw_ds, out, max_gap_s=0.25, link_images="copy")
    img = out / "images/observation.images.image_head/episode_000000/frame_000001.jpg"
    assert img.is_file()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_smooth_episodes.py -v -k smooth_dataset`
Expected: FAIL — `ImportError: cannot import name 'smooth_dataset'`

- [ ] **Step 3: Append the implementation**

```python
# append to robodriver_robot_deepcybo_lite_umi_ros2/smooth_episodes.py
import os
import shutil


def _episode_parquet_relpath(info: dict, episode_index: int) -> Path:
    return Path(
        info["data_path"].format(
            episode_chunk=episode_index // info["chunks_size"],
            episode_index=episode_index,
        )
    )


def _stats_for(arr2d: np.ndarray) -> dict:
    a = np.asarray(arr2d, dtype=np.float64)
    return {
        "min": a.min(0).tolist(),
        "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
        "count": [int(len(a))],
    }


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hard":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # cross-device or unsupported FS -> fall back to copy
    shutil.copy2(src, dst)


def smooth_dataset(
    root: Path,
    out: Path,
    max_gap_s: float,
    link_images: str = "hard",
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[EpisodeResult]:
    """Smooth every episode of the dataset at root into a new dataset at out.

    dry_run computes coverage without writing anything. link_images is
    "hard" (default; falls back to copy per-file on OSError) or "copy".
    """
    root, out = Path(root), Path(out)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"not a LeRobot dataset root (no meta/info.json): {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("video_path") and info.get("total_videos", 0) > 0:
        raise NotImplementedError(
            "video-backed datasets are out of scope; this recorder writes images"
        )
    if link_images not in ("hard", "copy"):
        raise ValueError(f"link_images must be 'hard' or 'copy', got {link_images!r}")
    if not dry_run:
        if out.exists():
            if not overwrite:
                raise FileExistsError(
                    f"output exists: {out} (pass overwrite=True / --overwrite)"
                )
            shutil.rmtree(out)
        (out / "data").mkdir(parents=True)
        (out / "meta").mkdir()

    results: list[EpisodeResult] = []
    for ep in range(info["total_episodes"]):
        rel = _episode_parquet_relpath(info, ep)
        src = root / rel
        if dry_run:
            # Reuse the transform without writing: process into a throwaway
            # in-memory location is not possible with pq.write_table, so
            # compute coverage directly.
            table = pq.read_table(src)
            df = table.to_pandas()
            times = df["timestamp"].to_numpy(dtype=float)
            state_in = np.stack(df["observation.state"].to_numpy())
            _, prov = smooth_state(times, state_in, max_gap_s)
            coverage = {}
            for col, arm in enumerate(("left", "right")):
                anchors = state_in[:, ARM_LAYOUT[arm].tracked] > 0.5
                coverage[arm] = arm_coverage(times, anchors, prov[:, col])
            results.append(EpisodeResult(episode_index=ep, coverage=coverage))
            continue

        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        results.append(process_episode_parquet(src, dst, max_gap_s))

    if dry_run:
        return results

    # ---- meta/ ----
    info_out = json.loads(json.dumps(info))  # deep copy
    info_out["features"][PROVENANCE_KEY] = dict(PROVENANCE_FEATURE)
    (out / "meta" / "info.json").write_text(
        json.dumps(info_out, indent=4), encoding="utf-8"
    )
    for name in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copy2(root / "meta" / name, out / "meta" / name)

    stats_lines = []
    with open(root / "meta" / "episodes_stats.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                stats_lines.append(json.loads(line))
    with open(out / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for rec in stats_lines:
            ep = rec["episode_index"]
            df = pd.read_parquet(out / _episode_parquet_relpath(info, ep))
            rec["stats"]["observation.state"] = _stats_for(
                np.stack(df["observation.state"])
            )
            rec["stats"]["action"] = _stats_for(np.stack(df["action"]))
            rec["stats"][PROVENANCE_KEY] = _stats_for(
                np.stack(df[PROVENANCE_KEY])
            )
            f.write(json.dumps(rec) + "\n")

    # ---- images ----
    images_root = root / "images"
    if images_root.is_dir():
        for src_img in images_root.rglob("*"):
            if src_img.is_file():
                _link_or_copy(
                    src_img, out / "images" / src_img.relative_to(images_root),
                    link_images,
                )
    return results
```

- [ ] **Step 4: Run the whole file's tests**

Run: `python -m pytest tests/test_smooth_episodes.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/smooth_episodes.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_smooth_episodes.py
git commit -m "Add dataset-level smoothing: meta patch, stats recompute, image links"
```

---

### Task 6: CLI, coverage report, console script, README

**Files:**
- Modify: `robodriver_robot_deepcybo_lite_umi_ros2/smooth_episodes.py` (append)
- Modify: `pyproject.toml` (console script + deps)
- Modify: `README.md` (usage section)
- Test: `tests/test_smooth_episodes.py` (append)

**Interfaces:**
- Consumes: `smooth_dataset`, `EpisodeResult` (Task 5).
- Produces:
  - `format_report(results: list[EpisodeResult], fps: int) -> str` — the spec's per-episode/per-arm report text
  - `main(argv: list[str] | None = None) -> int` — argparse CLI; console script `umi-smooth-episodes`

- [ ] **Step 1: Append the failing tests**

```python
# append to tests/test_smooth_episodes.py
from robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes import (
    format_report, main,
)


def test_format_report_shape():
    from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import ArmCoverage
    from robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes import EpisodeResult
    res = [EpisodeResult(episode_index=0, coverage={
        "left": ArmCoverage(n=240, measured=178, interpolated=62, unfillable=0,
                            gap_hist={1: 3, 4: 2, 7: 1}, longest_gap_s=0.2333),
        "right": ArmCoverage(n=240, measured=197, interpolated=43, unfillable=0,
                             gap_hist={1: 5}, longest_gap_s=0.0667),
    })]
    text = format_report(res, fps=30)
    assert "episode_000000" in text
    assert "measured 178/240 (74.2%)" in text
    assert "interpolated 62" in text
    assert "3x1f, 2x4f, 1x7f" in text
    assert "longest 0.233s" in text
    assert "usable 240/240 (100.0%)" in text
    assert "KEEP" in text


def test_format_report_flags_low_usable():
    from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import ArmCoverage
    from robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes import EpisodeResult
    res = [EpisodeResult(episode_index=0, coverage={
        "left": ArmCoverage(n=100, measured=50, interpolated=10, unfillable=40,
                            gap_hist={}, longest_gap_s=2.0),
        "right": ArmCoverage(n=100, measured=100, interpolated=0, unfillable=0,
                             gap_hist={}, longest_gap_s=0.0),
    })]
    text = format_report(res, fps=30)
    assert "usable 60/100 (60.0%)" in text
    assert "REVIEW" in text


def test_cli_end_to_end(raw_ds, tmp_path, capsys):
    out = tmp_path / "smoothed"
    rc = main(["--root", str(raw_ds), "--out", str(out), "--max-gap-s", "0.25"])
    assert rc == 0
    assert (out / "meta" / "info.json").is_file()
    assert "episode_000000" in capsys.readouterr().out


def test_cli_dry_run(raw_ds, tmp_path, capsys):
    out = tmp_path / "smoothed"
    rc = main(["--root", str(raw_ds), "--out", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()
    assert "interpolated 2" in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_smooth_episodes.py -v -k "report or cli"`
Expected: FAIL — `ImportError: cannot import name 'format_report'`

- [ ] **Step 3: Append the implementation**

```python
# append to robodriver_robot_deepcybo_lite_umi_ros2/smooth_episodes.py
import argparse

USABLE_KEEP_THRESHOLD = 0.90  # matches the README's >90% coverage target


def format_report(results: list[EpisodeResult], fps: int) -> str:
    """Per-episode/per-arm coverage report (spec §CLI).

    'usable' is a lower bound: n minus the WORST arm's unfillable count.
    (Exact per-frame overlap would need provenance masks in EpisodeResult;
    the worst-arm bound is sufficient for the KEEP/REVIEW decision.)
    """
    lines: list[str] = []
    for res in results:
        lines.append(f"episode_{res.episode_index:06d}")
        n = next(iter(res.coverage.values())).n
        for arm in ("left", "right"):
            c = res.coverage[arm]
            hist = ", ".join(
                f"{cnt}x{ln}f" for ln, cnt in sorted(c.gap_hist.items())
            )
            pct = 100.0 * c.measured / max(c.n, 1)
            lines.append(
                f"  {arm:<7} measured {c.measured}/{c.n} ({pct:.1f}%)"
                f"  interpolated {c.interpolated}  unfillable {c.unfillable}"
            )
            if hist:
                lines.append(
                    f"          gaps: {hist}    longest {c.longest_gap_s:.3f}s"
                )
        worst_unfillable = max(c.unfillable for c in res.coverage.values())
        usable = n - worst_unfillable  # lower bound (arms' gaps may overlap)
        frac = usable / max(n, 1)
        verdict = "KEEP" if frac >= USABLE_KEEP_THRESHOLD else "REVIEW"
        lines.append(
            f"  -> usable {usable}/{n} ({100.0 * frac:.1f}%)   {verdict}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline UMI episode smoother: interpolate dropout frames "
        "between tracked anchors into a NEW dataset (raw left untouched)."
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="input dataset root (never modified)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output dataset root (must not exist)")
    parser.add_argument("--max-gap-s", type=float, default=0.25,
                        help="max anchor-to-anchor gap span to fill (default 0.25)")
    parser.add_argument("--link-images", choices=("hard", "copy"), default="hard")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the coverage report without writing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    info = json.loads(
        (args.root / "meta" / "info.json").read_text(encoding="utf-8")
    )
    results = smooth_dataset(
        args.root, args.out, args.max_gap_s,
        link_images=args.link_images,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(format_report(results, fps=info["fps"]))
    if args.dry_run:
        print("\n(dry run: nothing written)")
    else:
        print(f"\nwrote {len(results)} episode(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Register the console script and deps in `pyproject.toml`**

In `[project.scripts]` add:

```toml
umi-smooth-episodes = "robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes:main"
```

In `[project] dependencies`, add any of `"pandas"`, `"pyarrow"`, `"scipy"` not already present (check first — the lerobot dep likely already pulls some in).

- [ ] **Step 5: Run the full test file + CLI smoke**

Run: `python -m pytest tests/test_smooth_episodes.py -v`
Expected: all PASS
Run: `python -m robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes --help`
Expected: usage text with `--root`, `--out`, `--max-gap-s`, `--dry-run`

- [ ] **Step 6: Add the README section**

Append to `README.md` after the Commands section:

```markdown
## Offline smoothing (post-collection)

Recorded episodes contain hold-last dropout frames (tracker + adapter tiers,
flagged by the quality dims). `umi-smooth-episodes` rebuilds every
non-measured frame by lerp+Slerp between `tracked==1` anchors, writing a NEW
dataset with an `observation.provenance` feature (0=measured, 1=interpolated,
2=unfillable) — the raw dataset is never modified:

​```bash
umi-smooth-episodes --root <dataset> --out <dataset>_smoothed [--max-gap-s 0.25]
umi-smooth-episodes --root <dataset> --out /dev/null --dry-run   # report only
​```

> **WARNING — provenance is for filtering, not for the policy.** Like the 7
> quality dims, `observation.provenance` must be excluded from policy inputs
> in training configs.

Design: `docs/superpowers/specs/2026-07-20-umi-offline-smoother-design.md`.
```

(Remove the zero-width characters around the inner code fence when pasting.)

- [ ] **Step 7: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add umi-smooth-episodes CLI with coverage report and README docs"
```

---

### Task 7: End-to-end verification on the real episode + Linux runbook

**Files:**
- Create: `tests/test_real_episode_e2e.py` (skips if the real dataset is absent)

**Interfaces:**
- Consumes: `main` (Task 6). The real dataset lives OUTSIDE the repo at `../umi_real_rec_2026-07-15` relative to the RoboDriver repo root (Windows: `D:\Desktop\Mystuff\robotics\umi_imp\umi_real_rec_2026-07-15`).

- [ ] **Step 1: Write the gated end-to-end test**

```python
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
```

- [ ] **Step 2: Run it (Windows — real dataset is on this machine)**

Run: `python -m pytest tests/test_real_episode_e2e.py -v`
Expected: PASS (or SKIP on machines without the recording)

Also eyeball the report once:
Run: `python -m robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes --root "D:\Desktop\Mystuff\robotics\umi_imp\umi_real_rec_2026-07-15" --out "D:\Desktop\Mystuff\robotics\umi_imp\_scratch_smoothed" --dry-run`
Expected: left `measured 178/240 (74.2%)`, right `measured 197/240 (82.1%)`, gap histogram printed; delete nothing (dry run).

- [ ] **Step 3: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_real_episode_e2e.py
git commit -m "Add gated end-to-end smoothing test against the real rig recording"
```

- [ ] **Step 4: Linux runbook (manual, on the rig — record results in the session)**

```bash
# in the RoboDriver venv, repo synced to this branch
python -m pytest robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_canonical_reader.py -v   # Task 1 gate: must PASS here
umi-smooth-episodes --root ~/umi_real_rec_2026-07-15 --out ~/umi_real_rec_2026-07-15_smoothed
deepcybo-lite-umi-visualize-episode --root ~/umi_real_rec_2026-07-15_smoothed   # canonical reader opens the output
```

Expected: spike test PASSES on Linux; visualize-episode opens and plays the smoothed episode. If the spike FAILS: stop, capture the traceback, and revisit per the spec's sidecar fallback.

---

## Self-Review Notes

- **Spec coverage:** offline CLI/new dataset (T5–6), anchors `tracked==1` (T2–3), keep+mark+report (T2–6), fill-only bit-exact (T2, pinned in T2/T4/T7 tests), adapter-package home (all), Slerp short-arc pin (T2), provenance schema + hf metadata (T4), declared⇒mandatory consequence honored by always writing the column (T4), stats/meta patch (T5), images hardlink (T5), dry-run/report/`max_gap_s` anchor-to-anchor default 0.25 (T2/T6), error cases (T3 monotonic, T5 exists/videos, T4 post-condition assert), spike + canonical-reader gate (T1, T7 step 4), real-episode e2e (T7). Sidecar fallback is a documented stop-condition in T1/T7, not built (YAGNI).
- **Known simplification:** `usable` in the report is the worst-arm lower bound (per-frame overlap not tracked); exact per-frame usable requires provenance masks in `EpisodeResult` — not needed for the KEEP/REVIEW decision at current data volumes.
- **Type consistency check:** `EpisodeResult.coverage: dict[str, ArmCoverage]` used identically in T4/T5/T6/T7; `smooth_arm`/`smooth_state` signatures match between T2/T3 definitions and T4 call sites; fixture lives in `tests/dataset_fixture.py` and is imported via `sys.path` insert in T1's spike test and T4's test header.
- **Pre-flight fix (2026-07-20):** the fixture was originally specified inside `test_canonical_reader.py`, whose module-level `pytest.importorskip` would have raised `Skipped` on import and silently skipped the entire Task 4/5/6 suites on any machine without the RoboDriver env. Fixture extracted to `tests/dataset_fixture.py`; the guard is now confined to the spike test, verified by Task 1 Step 4.
