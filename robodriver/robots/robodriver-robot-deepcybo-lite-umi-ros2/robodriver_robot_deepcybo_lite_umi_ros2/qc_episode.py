# robodriver_robot_deepcybo_lite_umi_ros2/qc_episode.py
"""At-the-rig per-episode QC (spec 2026-07-20, stage 2).

Reads the episode just written by the recorder, applies the collection gates,
and prints a PASS/FAIL verdict in about two seconds so a bad episode is redone
while the setup still exists.

Coverage comes from smooth_episodes._smooth_episode_frame -- the same code the
offline smoother uses -- so the QC verdict and the eventual smoothed output can
never disagree about how much of the episode is usable.

Usage at the rig:
    umi-qc-episode --root <dataset> --cell B3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .collection_qc import (
    GRIPPER_COL, EpisodeQC, QCThresholds, evaluate_gates, format_qc,
)
from .smooth_episodes import _smooth_episode_frame
from .smoothing import ARM_LAYOUT

CAMERA_KEYS = ("image_head", "image_wrist_left", "image_wrist_right")


def _read_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"not a LeRobot dataset root (no meta/info.json): {root}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def latest_episode_index(root: Path) -> int:
    """Index of the most recently written episode (what the operator just did)."""
    total = int(_read_info(Path(root))["total_episodes"])
    if total <= 0:
        raise ValueError(f"dataset has no episodes: {root}")
    return total - 1


def _episode_parquet(root: Path, info: dict, episode_index: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=episode_index // info["chunks_size"],
        episode_index=episode_index,
    )


def _camera_frame_counts(root: Path, episode_index: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in CAMERA_KEYS:
        d = root / "images" / f"observation.images.{key}" / f"episode_{episode_index:06d}"
        if d.is_dir():
            counts[key] = sum(1 for p in d.iterdir() if p.is_file())
    return counts


def load_episode_inputs(
    root: Path, episode_index: int, max_gap_s: float = 0.25
) -> dict:
    """Gather everything evaluate_gates needs from one recorded episode."""
    root = Path(root)
    info = _read_info(root)
    df = pq.read_table(_episode_parquet(root, info, episode_index)).to_pandas()
    state = np.stack(df["observation.state"].to_numpy()).astype(float)
    times = df["timestamp"].to_numpy(dtype=float)

    _, _, _, coverage = _smooth_episode_frame(df, max_gap_s)

    raw_tracked = {
        arm: float((state[:, lay.tracked] > 0.5).mean())
        for arm, lay in ARM_LAYOUT.items()
    }
    grip_range = {
        arm: float(state[:, col].max() - state[:, col].min())
        for arm, col in GRIPPER_COL.items()
    }
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    return dict(
        coverage=coverage,
        raw_tracked_frac=raw_tracked,
        gripper_range=grip_range,
        camera_frame_counts=_camera_frame_counts(root, episode_index),
        n_frames=int(len(df)),
        duration_s=duration,
    )


def check_episode(
    root: Path,
    episode_index: int | None = None,
    picking_arm: str = "right",
    max_gap_s: float = 0.25,
    thresholds: QCThresholds = QCThresholds(),
) -> tuple[int, EpisodeQC]:
    """Run every gate against one episode. Never modifies the dataset."""
    root = Path(root)
    idx = latest_episode_index(root) if episode_index is None else episode_index
    kwargs = load_episode_inputs(root, idx, max_gap_s)
    return idx, evaluate_gates(
        picking_arm=picking_arm, thresholds=thresholds, **kwargs
    )


def append_session_log(log_path: Path, entry: dict) -> None:
    """Append one JSON-lines record. Collection metadata lives here, NOT in the
    dataset -- the dataset schema is already settled."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="At-the-rig QC for the episode just recorded. "
        "Exit code 0 = keep, 1 = redo."
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="dataset root (never modified)")
    parser.add_argument("--episode", type=int, default=None,
                        help="episode index (default: the most recent)")
    parser.add_argument("--picking-arm", choices=("left", "right"), default="right")
    parser.add_argument("--max-gap-s", type=float, default=0.25)
    parser.add_argument("--session-log", type=Path, default=None,
                        help="JSONL log to append the verdict to")
    parser.add_argument("--cell", default=None,
                        help="object placement cell for this episode, e.g. B3")
    parser.add_argument("--no-prompt", action="store_true",
                        help="skip the manual good/bad review prompt")
    args = parser.parse_args(argv)

    idx, qc = check_episode(
        args.root, args.episode, args.picking_arm, args.max_gap_s
    )
    print(format_qc(qc, idx))

    # Manual review: every automated gate measures TRACKING quality; none can
    # tell whether the demonstration itself was any good (dropped object,
    # botched grasp). UMI drops episodes whose check_result.txt != true.
    manual_ok: bool | None = None
    if not args.no_prompt and qc.passed:
        reply = input("\n  demonstration itself good? [Y/n] ").strip().lower()
        manual_ok = reply in ("", "y", "yes")
        if not manual_ok:
            print("  -> marked bad by operator; REDO")

    keep = qc.passed and (manual_ok is not False)
    if args.session_log is not None:
        append_session_log(args.session_log, {
            "episode_index": idx,
            "passed": qc.passed,
            "manual_ok": manual_ok,
            "keep": keep,
            "cell": args.cell,
            "picking_arm": args.picking_arm,
            "failures": [f.name for f in qc.failures],
            "details": {r.name: r.detail for r in qc.results},
        })
    return 0 if keep else 1


if __name__ == "__main__":
    raise SystemExit(main())
