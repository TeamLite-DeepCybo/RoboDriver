# UMI Collection QC Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An at-the-rig per-episode QC checker (plus two small collection helpers) that tells the operator PASS/FAIL in ~2 seconds, so a bad episode is redone immediately instead of discovered a week later.

**Architecture:** Two new modules in the existing adapter package. `collection_qc.py` holds pure gate logic with no I/O — thresholds in, results out. `qc_episode.py` does the dataset loading and CLI, reusing `_smooth_episode_frame` from `smooth_episodes.py` as the single source of coverage truth rather than recomputing it. Two further small tools (placement prompter, latency measurement) follow.

**Tech Stack:** Python ≥3.10, numpy, pandas, pyarrow, pytest. No torch/lerobot/ROS in the gate or loader modules.

**Spec:** `docs/superpowers/specs/2026-07-20-umi-collection-pipeline-design.md` (read it first).

## Global Constraints

- `collection_qc.py` imports ONLY stdlib + numpy. No pandas, no pyarrow, no I/O of any kind — it is pure logic so the thresholds can be tested without building datasets.
- `qc_episode.py` may import stdlib + numpy + pandas + pyarrow, plus the package's own `.smoothing` and `.smooth_episodes`. Never `robodriver.*`, `lerobot*`, `torch`, `rclpy`, and never `DoRobotDataset`.
- Coverage MUST come from `smooth_episodes._smooth_episode_frame` — do not reimplement smoothing or coverage.
- The dataset being checked is **never modified**.
- State column layout, fixed: left pos `0:3`, left quat `3:7`, **left gripper `7`**, right pos `8:11`, right quat `11:15`, **right gripper `15`**, `left_tracked` `16`, `right_tracked` `19`.
- Default thresholds, verbatim from the spec: picking-arm usable ≥ **0.95** with **zero** unfillable; picking-arm **raw tracked ≥ 0.90**; steadying-arm usable ≥ **0.90**; episode length **5–20 s**; all **3** camera streams present with counts matching the frame count.
- Default picking arm is **`"right"`**; the steadying arm is the other one.
- "usable" = `(measured + interpolated) / n`. "raw tracked" = fraction of frames whose `<arm>_tracked` column is > 0.5.
- Working directory for all pytest commands: `robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2`. Commit from the repo root.
- Do NOT add any co-author, "Co-Authored-By", or AI-attribution trailer to commit messages.

---

### Task 1: Pure gate logic

**Files:**
- Create: `robodriver_robot_deepcybo_lite_umi_ros2/collection_qc.py`
- Test: `tests/test_collection_qc.py`

**Interfaces:**
- Produces:
  - `QCThresholds` frozen dataclass with fields `gripper_range_min: float = 0.05`, `picking_usable_min: float = 0.95`, `picking_max_unfillable: int = 0`, `picking_raw_tracked_min: float = 0.90`, `steadying_usable_min: float = 0.90`, `duration_min_s: float = 5.0`, `duration_max_s: float = 20.0`, `n_cameras: int = 3`
  - `GateResult` frozen dataclass: `name: str`, `passed: bool`, `detail: str`
  - `EpisodeQC` frozen dataclass: `passed: bool`, `results: tuple[GateResult, ...]`, with property `failures -> tuple[GateResult, ...]`
  - `GRIPPER_COL: dict[str, int]` = `{"left": 7, "right": 15}`
  - `usable_fraction(measured: int, interpolated: int, n: int) -> float`
  - `evaluate_gates(*, coverage, raw_tracked_frac, gripper_range, camera_frame_counts, n_frames, duration_s, picking_arm="right", thresholds=QCThresholds()) -> EpisodeQC`
  - `format_qc(qc: EpisodeQC, episode_index: int) -> str`

`coverage` is `dict[str, ArmCoverage]` from `.smoothing`; only its `n`, `measured`, `interpolated`, `unfillable` fields are read here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_collection_qc.py
import pytest

from robodriver_robot_deepcybo_lite_umi_ros2.collection_qc import (
    GRIPPER_COL, EpisodeQC, QCThresholds, evaluate_gates, format_qc,
    usable_fraction,
)
from robodriver_robot_deepcybo_lite_umi_ros2.smoothing import ArmCoverage


def _cov(n=300, measured=290, interpolated=10, unfillable=0):
    return ArmCoverage(
        n=n, measured=measured, interpolated=interpolated, unfillable=unfillable,
        filled_gap_hist={}, unfilled_gap_hist={},
        longest_filled_gap_s=0.0, longest_unfilled_gap_s=0.0,
    )


def _kwargs(**over):
    base = dict(
        coverage={"left": _cov(), "right": _cov()},
        raw_tracked_frac={"left": 0.95, "right": 0.95},
        gripper_range={"left": 0.0, "right": 0.4},
        camera_frame_counts={"image_head": 300, "image_wrist_left": 300,
                             "image_wrist_right": 300},
        n_frames=300,
        duration_s=10.0,
    )
    base.update(over)
    return base


def test_gripper_column_layout():
    assert GRIPPER_COL == {"left": 7, "right": 15}


def test_usable_fraction():
    assert usable_fraction(90, 5, 100) == pytest.approx(0.95)
    assert usable_fraction(0, 0, 0) == 0.0          # no divide-by-zero


def test_clean_episode_passes():
    qc = evaluate_gates(**_kwargs())
    assert qc.passed
    assert qc.failures == ()


def test_steadying_gripper_may_be_constant():
    # left steadies the container; its gripper need not move
    qc = evaluate_gates(**_kwargs(gripper_range={"left": 0.0, "right": 0.4}))
    assert qc.passed


def test_picking_gripper_constant_fails():
    # this is the 2026-07-15 stub failure, and a failed grasp
    qc = evaluate_gates(**_kwargs(gripper_range={"left": 0.0, "right": 0.0}))
    assert not qc.passed
    assert [f.name for f in qc.failures] == ["gripper_moved"]


def test_picking_usable_below_bar_fails():
    qc = evaluate_gates(**_kwargs(
        coverage={"left": _cov(), "right": _cov(measured=200, interpolated=80,
                                                unfillable=20)}))
    assert not qc.passed
    names = [f.name for f in qc.failures]
    assert "picking_usable" in names and "picking_unfillable" in names


def test_any_unfillable_on_picking_arm_fails():
    qc = evaluate_gates(**_kwargs(
        coverage={"left": _cov(), "right": _cov(measured=289, interpolated=10,
                                                unfillable=1)}))
    assert not qc.passed
    assert "picking_unfillable" in [f.name for f in qc.failures]


def test_raw_tracked_floor_catches_smoothing_crutch():
    # usable is fine because smoothing recovered it, but raw tracking has rotted
    qc = evaluate_gates(**_kwargs(
        coverage={"left": _cov(), "right": _cov(measured=200, interpolated=100,
                                                unfillable=0)},
        raw_tracked_frac={"left": 0.95, "right": 0.667}))
    assert not qc.passed
    assert [f.name for f in qc.failures] == ["picking_raw_tracked"]


def test_steadying_arm_has_its_own_looser_bar():
    qc = evaluate_gates(**_kwargs(
        coverage={"left": _cov(measured=200, interpolated=60, unfillable=40),
                  "right": _cov()}))
    assert not qc.passed
    assert [f.name for f in qc.failures] == ["steadying_usable"]


def test_missing_camera_stream_fails():
    qc = evaluate_gates(**_kwargs(
        camera_frame_counts={"image_head": 300, "image_wrist_left": 300}))
    assert not qc.passed
    assert "cameras" in [f.name for f in qc.failures]


def test_camera_frame_count_mismatch_fails():
    qc = evaluate_gates(**_kwargs(
        camera_frame_counts={"image_head": 300, "image_wrist_left": 188,
                             "image_wrist_right": 300}))
    assert not qc.passed
    assert "cameras" in [f.name for f in qc.failures]


@pytest.mark.parametrize("dur", [4.9, 20.1])
def test_duration_out_of_range_fails(dur):
    qc = evaluate_gates(**_kwargs(duration_s=dur))
    assert not qc.passed
    assert "duration" in [f.name for f in qc.failures]


def test_picking_arm_can_be_left():
    # roles swapped: left picks, right steadies
    qc = evaluate_gates(**_kwargs(
        gripper_range={"left": 0.0, "right": 0.4}, picking_arm="left"))
    assert not qc.passed
    assert "gripper_moved" in [f.name for f in qc.failures]


def test_thresholds_are_overridable():
    qc = evaluate_gates(**_kwargs(duration_s=1.0),
                        thresholds=QCThresholds(duration_min_s=0.5))
    assert qc.passed


def test_format_qc_shows_pass_and_failures():
    ok = format_qc(evaluate_gates(**_kwargs()), episode_index=7)
    assert "episode_000007" in ok and "PASS" in ok
    bad = format_qc(
        evaluate_gates(**_kwargs(gripper_range={"left": 0.0, "right": 0.0})),
        episode_index=7)
    assert "FAIL" in bad and "gripper_moved" in bad
    assert "REDO" in bad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_collection_qc.py -v`
Expected: FAIL at import — `ModuleNotFoundError: ... collection_qc`

- [ ] **Step 3: Write the implementation**

```python
# robodriver_robot_deepcybo_lite_umi_ros2/collection_qc.py
"""Per-episode collection quality gates (spec 2026-07-20, stage 2).

Pure logic: thresholds in, pass/fail out. No I/O and no pandas/pyarrow, so the
gate policy can be tested without building datasets. Dataset loading lives in
qc_episode.py.

The gates exist to be run AT THE RIG between episodes. An episode found bad
now costs ~30 s to redo; found next week it is simply lost, because the object
placement, lighting and hand motion cannot be recreated.
"""
from __future__ import annotations

from dataclasses import dataclass

# Gripper opening columns inside the 23-dim observation.state vector.
# Restated here rather than imported (config.py needs the lerobot env); pinned
# against the recorded feature names by the real-dataset test.
GRIPPER_COL: dict[str, int] = {"left": 7, "right": 15}


@dataclass(frozen=True)
class QCThresholds:
    """Gate bars. Defaults are the spec's; all overridable per session."""
    gripper_range_min: float = 0.05
    picking_usable_min: float = 0.95
    picking_max_unfillable: int = 0
    picking_raw_tracked_min: float = 0.90
    steadying_usable_min: float = 0.90
    duration_min_s: float = 5.0
    duration_max_s: float = 20.0
    n_cameras: int = 3


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class EpisodeQC:
    passed: bool
    results: tuple[GateResult, ...]

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if not r.passed)


def usable_fraction(measured: int, interpolated: int, n: int) -> float:
    """Fraction of frames carrying a real or reconstructed pose."""
    if n <= 0:
        return 0.0
    return (measured + interpolated) / n


def evaluate_gates(
    *,
    coverage: dict,
    raw_tracked_frac: dict[str, float],
    gripper_range: dict[str, float],
    camera_frame_counts: dict[str, int],
    n_frames: int,
    duration_s: float,
    picking_arm: str = "right",
    thresholds: QCThresholds = QCThresholds(),
) -> EpisodeQC:
    """Apply every stage-2 gate. `coverage` maps arm -> ArmCoverage."""
    if picking_arm not in ("left", "right"):
        raise ValueError(f"picking_arm must be 'left' or 'right', got {picking_arm!r}")
    steadying_arm = "right" if picking_arm == "left" else "left"
    t = thresholds
    pick, steady = coverage[picking_arm], coverage[steadying_arm]
    pick_usable = usable_fraction(pick.measured, pick.interpolated, pick.n)
    steady_usable = usable_fraction(steady.measured, steady.interpolated, steady.n)
    pick_raw = raw_tracked_frac[picking_arm]
    grip = gripper_range[picking_arm]

    results = [
        # The single highest-value check: catches a constant-0.0 encoder stub
        # AND a demonstration where the grasp simply never happened.
        GateResult(
            "gripper_moved", grip > t.gripper_range_min,
            f"{picking_arm} gripper range {grip:.3f} "
            f"(need > {t.gripper_range_min})",
        ),
        GateResult(
            "picking_usable", pick_usable >= t.picking_usable_min,
            f"{picking_arm} usable {pick_usable:.1%} "
            f"(need >= {t.picking_usable_min:.0%})",
        ),
        GateResult(
            "picking_unfillable", pick.unfillable <= t.picking_max_unfillable,
            f"{picking_arm} unfillable {pick.unfillable} "
            f"(need <= {t.picking_max_unfillable})",
        ),
        # Raw floor: without it, smoothing masks a degrading rig — usable stays
        # green while real tracking rots, because the smoother keeps recovering.
        GateResult(
            "picking_raw_tracked", pick_raw >= t.picking_raw_tracked_min,
            f"{picking_arm} raw tracked {pick_raw:.1%} "
            f"(need >= {t.picking_raw_tracked_min:.0%})",
        ),
        GateResult(
            "steadying_usable", steady_usable >= t.steadying_usable_min,
            f"{steadying_arm} usable {steady_usable:.1%} "
            f"(need >= {t.steadying_usable_min:.0%})",
        ),
        GateResult(
            "cameras",
            len(camera_frame_counts) == t.n_cameras
            and all(c == n_frames for c in camera_frame_counts.values()),
            f"{len(camera_frame_counts)}/{t.n_cameras} streams, counts "
            f"{dict(sorted(camera_frame_counts.items()))} vs {n_frames} frames",
        ),
        GateResult(
            "duration", t.duration_min_s <= duration_s <= t.duration_max_s,
            f"{duration_s:.1f}s (need {t.duration_min_s}-{t.duration_max_s}s)",
        ),
    ]
    return EpisodeQC(passed=all(r.passed for r in results), results=tuple(results))


def format_qc(qc: EpisodeQC, episode_index: int) -> str:
    """Operator-facing one-screen verdict."""
    head = "PASS" if qc.passed else "FAIL"
    lines = [f"episode_{episode_index:06d}   {head}"]
    for r in qc.results:
        lines.append(f"  {'ok  ' if r.passed else 'FAIL'} {r.name:20s} {r.detail}")
    if not qc.passed:
        lines.append("")
        lines.append("  -> REDO THIS EPISODE NOW (the setup still exists)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_collection_qc.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/robodriver_robot_deepcybo_lite_umi_ros2/collection_qc.py \
        robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_collection_qc.py
git commit -m "Add pure per-episode collection QC gate logic"
```

---

### Task 2: Episode loader and CLI

**Files:**
- Create: `robodriver_robot_deepcybo_lite_umi_ros2/qc_episode.py`
- Modify: `pyproject.toml` (add console script)
- Test: `tests/test_qc_episode.py`

**Interfaces:**
- Consumes: `evaluate_gates`, `EpisodeQC`, `QCThresholds`, `GRIPPER_COL`, `format_qc` (Task 1); `_smooth_episode_frame(df, max_gap_s) -> (state_out, action_out, provenance, coverage)` from `.smooth_episodes`; `ARM_LAYOUT` from `.smoothing`; `make_tiny_dataset(root, with_provenance=False, state=None)` and `default_state(n)` from `tests/dataset_fixture.py`.
- Produces:
  - `latest_episode_index(root: Path) -> int`
  - `load_episode_inputs(root: Path, episode_index: int, max_gap_s: float = 0.25) -> dict` — the exact kwargs `evaluate_gates` needs (minus `picking_arm`/`thresholds`)
  - `check_episode(root, episode_index=None, picking_arm="right", max_gap_s=0.25, thresholds=QCThresholds()) -> tuple[int, EpisodeQC]`
  - `append_session_log(log_path: Path, entry: dict) -> None`
  - `main(argv: list[str] | None = None) -> int` — console script `umi-qc-episode`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qc_episode.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from dataset_fixture import default_state, make_tiny_dataset  # noqa: E402

from robodriver_robot_deepcybo_lite_umi_ros2.qc_episode import (  # noqa: E402
    append_session_log, check_episode, latest_episode_index,
    load_episode_inputs, main,
)
from robodriver_robot_deepcybo_lite_umi_ros2.collection_qc import (  # noqa: E402
    QCThresholds,
)

N = 300  # 10 s at 30 Hz


def _good_state(n=N):
    """All tracked, picking (right) gripper sweeps, left gripper constant."""
    s = default_state(n)
    s[:, 15] = np.linspace(0.0, 0.6, n).astype(np.float32)   # right gripper moves
    s[:, 7] = 0.2                                            # left holds steady
    return s


@pytest.fixture()
def good_ds(tmp_path):
    root = tmp_path / "ds"
    make_tiny_dataset(root, with_provenance=False, state=_good_state())
    return root


def test_latest_episode_index(good_ds):
    assert latest_episode_index(good_ds) == 0


def test_load_episode_inputs_shapes(good_ds):
    kw = load_episode_inputs(good_ds, 0)
    assert kw["n_frames"] == N
    assert kw["duration_s"] == pytest.approx((N - 1) / 30.0, abs=1e-3)
    assert set(kw["coverage"]) == {"left", "right"}
    assert kw["raw_tracked_frac"]["right"] == pytest.approx(1.0)
    assert kw["gripper_range"]["right"] == pytest.approx(0.6, abs=1e-3)
    assert kw["gripper_range"]["left"] == pytest.approx(0.0, abs=1e-6)
    assert set(kw["camera_frame_counts"]) == {
        "image_head", "image_wrist_left", "image_wrist_right"}
    assert all(c == N for c in kw["camera_frame_counts"].values())


def test_check_episode_passes_on_good_data(good_ds):
    idx, qc = check_episode(good_ds)
    assert idx == 0
    assert qc.passed, [f.detail for f in qc.failures]


def test_check_episode_catches_constant_gripper(tmp_path):
    s = _good_state()
    s[:, 15] = 0.0                       # the 2026-07-15 stub
    root = tmp_path / "stub"
    make_tiny_dataset(root, with_provenance=False, state=s)
    _, qc = check_episode(root)
    assert not qc.passed
    assert "gripper_moved" in [f.name for f in qc.failures]


def test_check_episode_catches_raw_tracking_rot(tmp_path):
    s = _good_state()
    s[::3, 19] = 0.0                     # right tracked only 2/3 of frames
    root = tmp_path / "rot"
    make_tiny_dataset(root, with_provenance=False, state=s)
    _, qc = check_episode(root)
    assert not qc.passed
    assert "picking_raw_tracked" in [f.name for f in qc.failures]


def test_input_dataset_is_not_modified(good_ds):
    import hashlib
    snap = {p.relative_to(good_ds).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(good_ds.rglob("*")) if p.is_file()}
    check_episode(good_ds)
    after = {p.relative_to(good_ds).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(good_ds.rglob("*")) if p.is_file()}
    assert snap == after


def test_append_session_log_is_jsonl(tmp_path):
    log = tmp_path / "session.jsonl"
    append_session_log(log, {"episode_index": 0, "passed": True})
    append_session_log(log, {"episode_index": 1, "passed": False})
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
    assert [e["episode_index"] for e in lines] == [0, 1]
    assert lines[1]["passed"] is False


def test_cli_pass(good_ds, tmp_path, capsys):
    log = tmp_path / "session.jsonl"
    rc = main(["--root", str(good_ds), "--no-prompt", "--session-log", str(log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "episode_000000" in out
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["passed"] is True and entry["episode_index"] == 0


def test_cli_fail_returns_nonzero_and_logs(tmp_path, capsys):
    s = _good_state()
    s[:, 15] = 0.0
    root = tmp_path / "bad"
    make_tiny_dataset(root, with_provenance=False, state=s)
    log = tmp_path / "session.jsonl"
    rc = main(["--root", str(root), "--no-prompt", "--session-log", str(log)])
    assert rc == 1
    assert "REDO" in capsys.readouterr().out
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["passed"] is False
    assert "gripper_moved" in entry["failures"]


def test_cli_records_placement_cell(good_ds, tmp_path):
    log = tmp_path / "session.jsonl"
    main(["--root", str(good_ds), "--no-prompt", "--session-log", str(log),
          "--cell", "B3"])
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["cell"] == "B3"


def test_cli_picking_arm_left(good_ds, tmp_path):
    # left gripper is constant in the fixture, so gating left as picker fails
    log = tmp_path / "session.jsonl"
    rc = main(["--root", str(good_ds), "--no-prompt", "--session-log", str(log),
               "--picking-arm", "left"])
    assert rc == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_qc_episode.py -v`
Expected: FAIL at import — `ModuleNotFoundError: ... qc_episode`

- [ ] **Step 3: Write the implementation**

```python
# robodriver_robot_deepcybo_lite_umi_ros2/qc_episode.py
"""At-the-rig per-episode QC (spec 2026-07-20, stage 2).

Reads the episode just written by the recorder, applies the collection gates,
and prints a PASS/FAIL verdict in about two seconds so a bad episode is redone
while the setup still exists.

Coverage comes from smooth_episodes._smooth_episode_frame — the same code the
offline smoother uses — so the QC verdict and the eventual smoothed output can
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
        # NOTE: the on-disk convention is observation.images.<key>, verified
        # against the real recording — NOT observation.<key>.
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
    dataset — the dataset schema is already settled."""
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
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
umi-qc-episode = "robodriver_robot_deepcybo_lite_umi_ros2.qc_episode:main"
```

- [ ] **Step 5: Run tests and the CLI help**

Run: `python -m pytest tests/test_qc_episode.py -v`
Expected: all PASS
Run: `python -m pytest tests/ -q`
Expected: all pass, 1 pre-existing skip (the Linux-gated canonical-reader spike)
Run: `python -m robodriver_robot_deepcybo_lite_umi_ros2.qc_episode --help`
Expected: usage text listing `--root`, `--episode`, `--picking-arm`, `--session-log`, `--cell`, `--no-prompt`

- [ ] **Step 6: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add umi-qc-episode at-the-rig per-episode QC checker"
```

---

### Task 3: Real-dataset validation and gripper-column guard

**Files:**
- Test: `tests/test_qc_real_episode.py`

**Interfaces:**
- Consumes: `check_episode`, `load_episode_inputs` (Task 2); `GRIPPER_COL` (Task 1). The real recording is at `D:\Desktop\Mystuff\robotics\umi_imp\umi_real_rec_2026-07-15`, i.e. `Path(__file__).resolve().parents[4].parent / "umi_real_rec_2026-07-15"`. Verify that arithmetic resolves and correct it if not; the test must RUN, not silently skip.

The real recording is a **known-bad** episode by design (constant-0.0 grippers, 82.1% right raw tracked, 0.70 s left dropout). It is the ideal regression fixture: the checker must **reject** it, for exactly the right reasons.

- [ ] **Step 1: Write the test**

```python
# tests/test_qc_real_episode.py
"""The 2026-07-15 recording is known-bad: constant-0.0 grippers, 82.1% right
raw tracked, a 0.70 s left dropout. The checker must reject it for exactly
those reasons — this is the regression guard that the gates work on real data.
"""
import json
from pathlib import Path

import pytest

from robodriver_robot_deepcybo_lite_umi_ros2.collection_qc import GRIPPER_COL
from robodriver_robot_deepcybo_lite_umi_ros2.qc_episode import (
    check_episode, load_episode_inputs,
)

REAL = Path(__file__).resolve().parents[4].parent / "umi_real_rec_2026-07-15"

pytestmark = pytest.mark.skipif(
    not REAL.is_dir(), reason=f"real recording not found at {REAL}"
)


def test_gripper_columns_match_recorded_feature_names():
    """Pin GRIPPER_COL against the real dataset's declared feature names, so a
    layout change upstream fails here instead of silently gating the wrong
    column."""
    info = json.loads((REAL / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"]["observation.state"]["names"]
    assert names[GRIPPER_COL["left"]] == "left_gripper.pos"
    assert names[GRIPPER_COL["right"]] == "right_gripper.pos"


def test_real_episode_is_rejected_for_the_right_reasons():
    idx, qc = check_episode(REAL)
    assert idx == 0
    assert not qc.passed
    names = {f.name for f in qc.failures}
    # constant-0.0 gripper stub
    assert "gripper_moved" in names
    # right arm raw tracked 82.1%, below the 90% floor
    assert "picking_raw_tracked" in names


def test_real_episode_inputs_match_known_values():
    kw = load_episode_inputs(REAL, 0)
    assert kw["n_frames"] == 240
    assert kw["raw_tracked_frac"]["right"] == pytest.approx(197 / 240, abs=1e-3)
    assert kw["raw_tracked_frac"]["left"] == pytest.approx(178 / 240, abs=1e-3)
    assert kw["gripper_range"]["left"] == pytest.approx(0.0, abs=1e-9)
    assert kw["gripper_range"]["right"] == pytest.approx(0.0, abs=1e-9)
    # left arm carries the 0.70 s unfillable dropout
    assert kw["coverage"]["left"].unfillable == 29
    assert kw["coverage"]["right"].unfillable == 0
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_qc_real_episode.py -v`
Expected: 3 passed, none skipped. If it skips, fix the `REAL` path arithmetic — a silent skip defeats the purpose.

- [ ] **Step 3: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2/tests/test_qc_real_episode.py
git commit -m "Guard QC gates against the real known-bad recording"
```

---

### Task 4: Placement-cell prompter

**Files:**
- Create: `robodriver_robot_deepcybo_lite_umi_ros2/placement_cells.py`
- Modify: `pyproject.toml`
- Test: `tests/test_placement_cells.py`

**Interfaces:**
- Produces:
  - `cell_names(rows: int, cols: int) -> list[str]` — `["A1", "A2", ..., "B1", ...]`
  - `balanced_sequence(rows: int, cols: int, n: int, seed: int | None = None) -> list[str]` — every cell used within one of every other (each full pass is a shuffled permutation)
  - `main(argv: list[str] | None = None) -> int` — console script `umi-placement-cells`

Purpose (spec): unassisted human randomization clumps, and the bias only becomes visible after training. Systematic coverage of object start position is the primary variation axis.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_placement_cells.py
from collections import Counter

from robodriver_robot_deepcybo_lite_umi_ros2.placement_cells import (
    balanced_sequence, cell_names, main,
)


def test_cell_names_grid():
    assert cell_names(2, 3) == ["A1", "A2", "A3", "B1", "B2", "B3"]


def test_balanced_sequence_length():
    assert len(balanced_sequence(3, 4, 100, seed=0)) == 100


def test_balanced_sequence_is_balanced():
    seq = balanced_sequence(3, 4, 120, seed=0)      # 12 cells, 120 draws
    counts = Counter(seq)
    assert set(counts) == set(cell_names(3, 4))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_partial_pass_still_near_balanced():
    seq = balanced_sequence(3, 4, 30, seed=1)       # 2.5 passes
    counts = Counter(seq)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_each_full_pass_is_a_permutation():
    seq = balanced_sequence(2, 2, 8, seed=3)
    assert sorted(seq[:4]) == cell_names(2, 2)
    assert sorted(seq[4:]) == cell_names(2, 2)


def test_seed_is_deterministic():
    assert balanced_sequence(3, 3, 20, seed=7) == balanced_sequence(3, 3, 20, seed=7)


def test_shuffling_actually_happens():
    # a different seed should give a different order for a long enough sequence
    assert balanced_sequence(3, 4, 60, seed=1) != balanced_sequence(3, 4, 60, seed=2)


def test_cli_prints_requested_count(capsys):
    assert main(["--rows", "2", "--cols", "2", "-n", "4", "--seed", "0"]) == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert len(lines) == 4
    assert all(any(c in ln for c in cell_names(2, 2)) for ln in lines)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_placement_cells.py -v`
Expected: FAIL at import — `ModuleNotFoundError: ... placement_cells`

- [ ] **Step 3: Write the implementation**

```python
# robodriver_robot_deepcybo_lite_umi_ros2/placement_cells.py
"""Systematic object-placement prompter (spec 2026-07-20, Variation).

Object start position is the PRIMARY variation axis — the difference between a
policy and a replayed trajectory. Unassisted human randomization clumps, and
the bias only becomes visible after training, so the cell to use is dictated
rather than chosen.

Each full pass over the grid is an independent shuffled permutation, so
coverage stays balanced even if a session stops partway.
"""
from __future__ import annotations

import argparse
import random
from string import ascii_uppercase


def cell_names(rows: int, cols: int) -> list[str]:
    if not (1 <= rows <= 26) or cols < 1:
        raise ValueError(f"bad grid {rows}x{cols}")
    return [f"{ascii_uppercase[r]}{c + 1}" for r in range(rows) for c in range(cols)]


def balanced_sequence(
    rows: int, cols: int, n: int, seed: int | None = None
) -> list[str]:
    """n placements, balanced across the grid (max-min occupancy <= 1)."""
    cells = cell_names(rows, cols)
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        pass_ = cells[:]
        rng.shuffle(pass_)
        out.extend(pass_)
    return out[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a balanced object-placement cell sequence for a "
        "collection session."
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("-n", type=int, default=30, help="number of episodes")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    for i, cell in enumerate(
        balanced_sequence(args.rows, args.cols, args.n, args.seed)
    ):
        print(f"episode {i:3d}   place object at {cell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
umi-placement-cells = "robodriver_robot_deepcybo_lite_umi_ros2.placement_cells:main"
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_placement_cells.py -v`
Expected: all PASS
Run: `python -m pytest tests/ -q`
Expected: all pass, 1 pre-existing skip

- [ ] **Step 6: Commit**

```bash
git add robodriver/robots/robodriver-robot-deepcybo-lite-umi-ros2
git commit -m "Add balanced placement-cell prompter for systematic position coverage"
```

---

## Self-Review Notes

- **Spec coverage:** stage-2 gates including the raw-tracked floor (T1, T2), manual review flag (T2 `main`, logged as `manual_ok`), session log with placement cell (T2), placement prompter (T4), reuse of `_smooth_episode_frame` as the single coverage source (T2), never modifying the input (T2 test), gripper-column layout pinned against real feature names (T3). Latency measurement is deliberately **not** in this plan — it needs a live ROS environment, is a one-off rather than per-session, and belongs with the rig work; the spec lists it as a pilot-time action.
- **Deliberate omission:** the session pre-flight checker. Per the spec it is "a checklist plus a topic-rate script"; wire routing and face redundancy are operator checks read off the existing debug overlay, and camera rates come from `ros2 topic hz`. No new code is warranted, and inventing some would be YAGNI.
- **Type consistency:** `EpisodeQC.failures` is a property (used as `qc.failures` throughout); `coverage` is always `dict[str, ArmCoverage]`; `check_episode` returns `(int, EpisodeQC)` in T2 and is unpacked that way in T3; `GRIPPER_COL` defined T1, consumed T2/T3.
- **Known simplification:** `test_shuffling_actually_happens` compares two seeded sequences and could in principle collide; with 12 cells over 5 passes the probability is negligible.
