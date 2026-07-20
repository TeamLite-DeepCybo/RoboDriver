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


def _smooth_episode_frame(
    df: pd.DataFrame, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, ArmCoverage]]:
    """Shared read -> smooth -> coverage transform for one episode.

    Takes the episode's raw dataframe (as read from its source parquet) and
    returns (state_out, action_out, provenance, coverage):
      - state_out:  (N, 23) smoothed observation.state
      - action_out: (N, 16) regenerated action (mirrors state_out[:, :16])
      - provenance: (N, 2) [left, right] MEASURED/INTERPOLATED/UNFILLABLE
      - coverage:   {"left": ArmCoverage, "right": ArmCoverage}

    This is the single source of truth for the smoothing + coverage
    computation, called by BOTH process_episode_parquet (which additionally
    writes the result to a parquet file) and smooth_dataset's dry_run branch
    (which writes nothing). Keeping one implementation means the two paths
    can never drift or disagree on reported coverage, and dry_run inherits
    the cross-check assert below for free.
    """
    times = df["timestamp"].to_numpy(dtype=float)
    state_in = np.stack(df["observation.state"].to_numpy())

    state_out, prov = smooth_state(times, state_in, max_gap_s)
    action_out = regen_action(state_out)

    coverage: dict[str, ArmCoverage] = {}
    for col, arm in enumerate(("left", "right")):
        anchors = state_in[:, ARM_LAYOUT[arm].tracked] > 0.5
        coverage[arm] = arm_coverage(times, anchors, prov[:, col])
        # Cross-check (spec §Error handling): this function computes
        # `anchors` itself while smooth_state computes its own anchor mask
        # internally. If those two independently-derived masks ever diverge
        # (e.g. a changed threshold in one place but not the other),
        # provenance and coverage would silently desync.
        assert coverage[arm].measured == int(anchors.sum()), (
            f"{arm}: coverage measured {coverage[arm].measured} != "
            f"anchor count {int(anchors.sum())} — smooth_state and "
            f"process_episode_parquet disagree on the anchor mask"
        )
    return state_out, action_out, prov, coverage


def process_episode_parquet(
    src: Path, dst: Path, max_gap_s: float
) -> EpisodeResult:
    """Smooth one episode parquet from src into dst (dst parent must exist)."""
    table_in = pq.read_table(src)
    df = table_in.to_pandas()
    state_out, action_out, prov, coverage = _smooth_episode_frame(df, max_gap_s)

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
    root_r, out_r = root.resolve(), out.resolve()
    if out_r == root_r or out_r in root_r.parents:
        raise ValueError(
            f"out ({out}) equals or is an ancestor of root ({root}); "
            "overwrite=True would shutil.rmtree(out) and destroy the input "
            "dataset -- refusing to let out alias or contain root"
        )
    if root_r in out_r.parents:
        raise ValueError(
            f"out ({out}) is nested inside root ({root}); the output "
            "cannot be placed inside the input dataset -- writing episodes "
            "and images there would mutate root's own directory tree"
        )
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
            # Reuse the exact same read->smooth->coverage transform as the
            # real path (_smooth_episode_frame), just without the write, so
            # dry-run coverage can never drift from what a real run reports.
            table = pq.read_table(src)
            df = table.to_pandas()
            _, _, _, coverage = _smooth_episode_frame(df, max_gap_s)
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


import argparse
import tempfile

USABLE_KEEP_THRESHOLD = 0.90  # matches the README's >90% coverage target


def format_report(results: list[EpisodeResult]) -> str:
    """Per-episode/per-arm coverage report (spec §CLI).

    'usable' is a lower bound: n minus the WORST arm's unfillable count.
    (Exact per-frame overlap would need provenance masks in EpisodeResult;
    the worst-arm bound is sufficient for the KEEP/REVIEW decision.)

    Filled and unfilled gaps are reported on separate lines so the two
    "longest" figures can never be conflated: `longest filled` is the
    biggest gap the smoother actually closed; `longest unfilled` (only
    printed when there are unfilled gaps) is the biggest bracketed span it
    REJECTED -- the number that tells a user how much to raise --max-gap-s.
    """
    lines: list[str] = []
    for res in results:
        lines.append(f"episode_{res.episode_index:06d}")
        n = next(iter(res.coverage.values())).n
        for arm in ("left", "right"):
            c = res.coverage[arm]
            filled_hist = ", ".join(
                f"{cnt}x{ln}f" for ln, cnt in sorted(c.filled_gap_hist.items())
            )
            pct = 100.0 * c.measured / max(c.n, 1)
            lines.append(
                f"  {arm:<7} measured {c.measured}/{c.n} ({pct:.1f}%)"
                f"  interpolated {c.interpolated}  unfillable {c.unfillable}"
            )
            if filled_hist:
                lines.append(
                    f"          gaps: {filled_hist}    "
                    f"longest filled {c.longest_filled_gap_s:.3f}s"
                )
            if c.unfilled_gap_hist:
                unfilled_hist = ", ".join(
                    f"{cnt}x{ln}f" for ln, cnt in sorted(c.unfilled_gap_hist.items())
                )
                lines.append(
                    f"          unfilled gaps: {unfilled_hist}    "
                    f"longest unfilled {c.longest_unfilled_gap_s:.3f}s"
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
    parser.add_argument("--out", type=Path, default=None,
                        help="output dataset root; must not already exist "
                        "unless --overwrite is given; required unless "
                        "--dry-run is given (unused in dry-run mode)")
    parser.add_argument("--max-gap-s", type=float, default=0.25,
                        help="max anchor-to-anchor gap span to fill (default 0.25)")
    parser.add_argument("--link-images", choices=("hard", "copy"), default="hard")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the coverage report without writing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run is given")

    # In dry-run mode --out is optional and unused (smooth_dataset never
    # writes anything when dry_run=True); when omitted, pass a placeholder
    # that cannot alias, contain, or be contained by --root so the
    # unconditional alias/containment guards in smooth_dataset never trip.
    out_arg = args.out
    if out_arg is None:
        out_arg = Path(tempfile.gettempdir()) / "smooth_episodes_dry_run_unused"

    results = smooth_dataset(
        args.root, out_arg, args.max_gap_s,
        link_images=args.link_images,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(format_report(results))
    if args.dry_run:
        print("\n(dry run: nothing written)")
    else:
        print(f"\nwrote {len(results)} episode(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
