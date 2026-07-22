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
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .collection_qc import (
    GRIPPER_COL, EpisodeQC, QCThresholds, evaluate_gates, format_qc,
)
from .smooth_episodes import _smooth_episode_frame
from .smoothing import ARM_LAYOUT

CAMERA_KEYS = ("image_head", "image_wrist_left", "image_wrist_right")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# QCThresholds field names, one per overridable CLI flag (--gripper-range-min
# -> gripper_range_min, etc). Kept as a tuple so main() and the threshold
# builder can't drift apart.
_THRESHOLD_FIELDS = (
    "gripper_range_min",
    "picking_usable_min",
    "picking_max_unfillable",
    "picking_raw_tracked_min",
    "steadying_usable_min",
    "steadying_raw_tracked_min",
    "duration_min_s",
    "duration_max_s",
)


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
    """Count only image files per camera directory.

    A stray non-image file (Thumbs.db, .DS_Store) must not perturb the count
    against a good episode's actual frame total.
    """
    counts: dict[str, int] = {}
    for key in CAMERA_KEYS:
        d = root / "images" / f"observation.images.{key}" / f"episode_{episode_index:06d}"
        if d.is_dir():
            counts[key] = sum(
                1 for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
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
    # NOTE: gripper_range is max - min over the episode, which confirms SOME
    # travel occurred but not that it was an actual open->close->open cycle --
    # a monotonic drift or a single noise spike would satisfy it just as well.
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


def _thresholds_from_args(args: argparse.Namespace) -> QCThresholds:
    """Build QCThresholds from the CLI, overriding only the flags supplied.

    The gripper-range bar in particular cannot be set a priori (the encoder
    units are unverified) and must be calibrated during the pilot; the other
    bars may also need loosening for a pilot session before real data has
    ever cleared them. Without this, an operator hitting an unreachable
    default is stuck editing source at the rig.
    """
    overrides = {
        field: value
        for field in _THRESHOLD_FIELDS
        if (value := getattr(args, field)) is not None
    }
    return QCThresholds(**overrides)


def _add_threshold_args(parser: argparse.ArgumentParser) -> None:
    defaults = QCThresholds()
    parser.add_argument("--gripper-range-min", type=float, default=None,
                         help=f"override (default {defaults.gripper_range_min})")
    parser.add_argument("--picking-usable-min", type=float, default=None,
                         help=f"override (default {defaults.picking_usable_min})")
    parser.add_argument("--picking-max-unfillable", type=int, default=None,
                         help=f"override (default {defaults.picking_max_unfillable})")
    parser.add_argument("--picking-raw-tracked-min", type=float, default=None,
                         help=f"override (default {defaults.picking_raw_tracked_min})")
    parser.add_argument("--steadying-usable-min", type=float, default=None,
                         help=f"override (default {defaults.steadying_usable_min})")
    parser.add_argument("--steadying-raw-tracked-min", type=float, default=None,
                         help=f"override (default {defaults.steadying_raw_tracked_min})")
    parser.add_argument("--duration-min-s", type=float, default=None,
                         help=f"override (default {defaults.duration_min_s})")
    parser.add_argument("--duration-max-s", type=float, default=None,
                         help=f"override (default {defaults.duration_max_s})")


def _default_session_log(resolved_root: Path) -> Path:
    """`<root>.qc_log.jsonl`, a SIBLING of the dataset directory -- never
    inside it, so the dataset is never modified by QC runs."""
    return resolved_root.with_name(resolved_root.name + ".qc_log.jsonl")


def _run(args: argparse.Namespace) -> int:
    thresholds = _thresholds_from_args(args)
    idx, qc = check_episode(
        args.root, args.episode, args.picking_arm, args.max_gap_s, thresholds
    )
    print(format_qc(qc, idx))

    # Manual review: every automated gate measures TRACKING quality; none can
    # tell whether the demonstration itself was any good (dropped object,
    # botched grasp). UMI drops episodes whose check_result.txt != true.
    manual_ok: bool | None = None
    if not args.no_prompt and qc.passed:
        try:
            reply = input("\n  demonstration itself good? [Y/n] ").strip().lower()
            manual_ok = reply in ("", "y", "yes")
            if not manual_ok:
                print("  -> marked bad by operator; REDO")
        except EOFError:
            # Non-interactive stdin (e.g. a wrapper script run without
            # --no-prompt): treat as "not asked" rather than crashing a
            # passing episode. manual_ok stays None -- it cannot fail qc.
            pass
        except KeyboardInterrupt:
            print("\n  interrupted", file=sys.stderr)
            return 2

    keep = qc.passed and (manual_ok is not False)

    resolved_root = Path(args.root).resolve()
    log_path = args.session_log if args.session_log is not None \
        else _default_session_log(resolved_root)
    append_session_log(log_path, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(resolved_root),
        "max_gap_s": args.max_gap_s,
        "thresholds": dataclasses.asdict(thresholds),
        "operator": args.operator,
        "note": args.note,
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


def main(argv: list[str] | None = None) -> int:
    """At-the-rig QC CLI entry point.

    Exit codes: 0 = keep the episode, 1 = redo it (a gate or the operator's
    manual review failed), 2 = tool error (bad --root, unreadable dataset,
    or the review prompt was interrupted) -- kept distinct from 1 so a
    wrapper script can tell "this episode is bad" apart from "the tool
    itself broke".
    """
    parser = argparse.ArgumentParser(
        description="At-the-rig QC for the episode just recorded. "
        "Exit codes: 0 = keep, 1 = redo, 2 = tool error."
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="dataset root (never modified)")
    parser.add_argument("--episode", type=int, default=None,
                        help="episode index (default: the most recent)")
    parser.add_argument("--picking-arm", choices=("left", "right"), default="right")
    parser.add_argument("--max-gap-s", type=float, default=0.25)
    parser.add_argument("--session-log", type=Path, default=None,
                        help="JSONL log to append the verdict to "
                        "(default: <root>.qc_log.jsonl, a sibling of --root)")
    parser.add_argument("--cell", default=None,
                        help="object placement cell for this episode, e.g. B3")
    parser.add_argument("--operator", default=None,
                        help="operator name/id, recorded in the session log")
    parser.add_argument("--note", default=None,
                        help="free-text session note (object, container "
                        "position, lighting, rig changes), recorded in the "
                        "session log")
    parser.add_argument("--no-prompt", action="store_true",
                        help="skip the manual good/bad review prompt")
    _add_threshold_args(parser)
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
