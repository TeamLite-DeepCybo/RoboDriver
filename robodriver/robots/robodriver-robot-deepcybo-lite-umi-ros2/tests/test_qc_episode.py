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
