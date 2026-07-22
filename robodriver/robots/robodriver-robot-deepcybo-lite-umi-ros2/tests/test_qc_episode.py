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


@pytest.fixture()
def short_duration_ds(tmp_path):
    """~2 s at 30 Hz -- below the 5 s default duration_min_s, otherwise clean."""
    n = 60
    root = tmp_path / "short"
    make_tiny_dataset(root, with_provenance=False, state=_good_state(n))
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


# ---------------------------------------------------------------------------
# Fix 1: CLI threshold overrides
# ---------------------------------------------------------------------------

def test_cli_duration_override_changes_verdict(short_duration_ds, tmp_path):
    log = tmp_path / "session.jsonl"
    rc_default = main(["--root", str(short_duration_ds), "--no-prompt",
                       "--session-log", str(log)])
    assert rc_default == 1
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert "duration" in entry["failures"]

    log2 = tmp_path / "session2.jsonl"
    rc_override = main(["--root", str(short_duration_ds), "--no-prompt",
                        "--session-log", str(log2), "--duration-min-s", "1.0"])
    assert rc_override == 0


def test_cli_threshold_override_recorded_in_log(short_duration_ds, tmp_path):
    log = tmp_path / "session.jsonl"
    main(["--root", str(short_duration_ds), "--no-prompt", "--session-log", str(log),
          "--duration-min-s", "1.0"])
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["thresholds"]["duration_min_s"] == pytest.approx(1.0)
    # untouched thresholds keep their spec default
    assert entry["thresholds"]["picking_usable_min"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Fix 2: tool errors (exit 2) distinct from redo (exit 1)
# ---------------------------------------------------------------------------

def test_cli_nonexistent_root_exits_2_cleanly(tmp_path, capsys):
    rc = main(["--root", str(tmp_path / "does_not_exist"), "--no-prompt"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    # a clean one-line message, not a raw traceback
    assert "Traceback" not in captured.err


def test_cli_prompt_survives_noninteractive_stdin(good_ds, tmp_path, monkeypatch):
    # No --no-prompt: main() must reach the input() prompt. Simulate a
    # wrapper script piping in a non-terminal (closed) stdin, which raises
    # EOFError from input() -- this must not crash a passing episode.
    def _eof(*_a, **_k):
        raise EOFError
    monkeypatch.setattr("builtins.input", _eof)

    log = tmp_path / "session.jsonl"
    rc = main(["--root", str(good_ds), "--session-log", str(log)])
    assert rc == 0
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["manual_ok"] is None


# ---------------------------------------------------------------------------
# Fix 3: session log provenance + default sibling path
# ---------------------------------------------------------------------------

def test_default_session_log_path_is_sibling_of_dataset_root(good_ds):
    rc = main(["--root", str(good_ds), "--no-prompt"])
    assert rc == 0
    resolved = good_ds.resolve()
    expected_log = resolved.with_name(resolved.name + ".qc_log.jsonl")
    assert expected_log.is_file()
    assert expected_log.parent == resolved.parent


def test_default_session_log_leaves_dataset_untouched(good_ds):
    import hashlib
    before = {
        p.relative_to(good_ds).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(good_ds.rglob("*")) if p.is_file()
    }
    main(["--root", str(good_ds), "--no-prompt"])
    after = {
        p.relative_to(good_ds).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(good_ds.rglob("*")) if p.is_file()
    }
    assert before == after


def test_session_log_entry_carries_provenance(good_ds, tmp_path):
    log = tmp_path / "session.jsonl"
    main(["--root", str(good_ds), "--no-prompt", "--session-log", str(log),
          "--operator", "ray", "--note", "left cam relit"])
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["dataset_root"] == str(good_ds.resolve())
    assert entry["max_gap_s"] == pytest.approx(0.25)
    assert isinstance(entry["thresholds"], dict)
    assert entry["thresholds"]["steadying_raw_tracked_min"] == pytest.approx(0.80)
    assert entry["operator"] == "ray"
    assert entry["note"] == "left cam relit"
    assert "timestamp" in entry and entry["timestamp"]


def test_two_sessions_in_one_log_are_distinguishable(good_ds, tmp_path):
    # Two sessions against two different roots, same log file: episode_index
    # alone would show 0 for both. timestamp + dataset_root disambiguate.
    root2 = tmp_path / "ds2"
    make_tiny_dataset(root2, with_provenance=False, state=_good_state())
    log = tmp_path / "session.jsonl"
    main(["--root", str(good_ds), "--no-prompt", "--session-log", str(log),
          "--operator", "alice"])
    main(["--root", str(root2), "--no-prompt", "--session-log", str(log),
          "--operator", "bob"])
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
    assert len(lines) == 2
    assert lines[0]["dataset_root"] != lines[1]["dataset_root"]
    assert lines[0]["operator"] != lines[1]["operator"]


# ---------------------------------------------------------------------------
# Fix 7: camera frame counting ignores non-image files
# ---------------------------------------------------------------------------

def test_stray_non_image_file_does_not_affect_camera_count(good_ds):
    cam_dir = good_ds / "images" / "observation.images.image_head" / "episode_000000"
    (cam_dir / "Thumbs.db").write_bytes(b"not an image")
    kw = load_episode_inputs(good_ds, 0)
    assert kw["camera_frame_counts"]["image_head"] == N
