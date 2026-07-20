# tests/dataset_fixture.py
"""Builds a minimal LeRobot v2.1 UMI dataset on disk for tests.

Writes the v2.1 files directly with pyarrow/json, so it is independent of the
smoother implementation and can produce BOTH the raw input (with_provenance=
False) and a provenance-carrying dataset (for the canonical-reader spike).

NOT a test module -- no test collection, and deliberately free of any
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
