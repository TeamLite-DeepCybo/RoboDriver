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
    INTERPOLATED, MEASURED, ArmCoverage,
)
from robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes import (  # noqa: E402
    process_episode_parquet,
)
import robodriver_robot_deepcybo_lite_umi_ros2.smooth_episodes as smooth_episodes_mod  # noqa: E402


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


def test_measured_anchor_mismatch_raises(raw_ds, tmp_path, monkeypatch):
    """If arm_coverage's `measured` disagrees with the anchor count that
    process_episode_parquet computed itself, that's smooth_state and
    process_episode_parquet disagreeing on the anchor mask -- must raise."""
    src = raw_ds / "data/chunk-000/episode_000000.parquet"
    dst = tmp_path / "out.parquet"

    real_arm_coverage = smooth_episodes_mod.arm_coverage

    def fake_arm_coverage(times, anchors, provenance):
        cov = real_arm_coverage(times, anchors, provenance)
        return ArmCoverage(
            n=cov.n,
            measured=cov.measured + 1,  # disagrees with true anchor count
            interpolated=cov.interpolated,
            unfillable=cov.unfillable,
            gap_hist=cov.gap_hist,
            longest_gap_s=cov.longest_gap_s,
        )

    monkeypatch.setattr(smooth_episodes_mod, "arm_coverage", fake_arm_coverage)

    with pytest.raises(AssertionError, match="disagree on the anchor mask"):
        process_episode_parquet(src, dst, max_gap_s=0.25)


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
    # every untouched stats key must be byte-for-byte (exactly, not approx)
    # equal to the input -- these are never recomputed, just copied through.
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert stats_out["stats"][key] == stats_in["stats"][key]

    df = pd.read_parquet(out / "data/chunk-000/episode_000000.parquet")
    s = np.stack(df["observation.state"])
    a = np.stack(df["action"])
    for stat_key, arr in (("observation.state", s), ("action", a)):
        expected = {
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "count": [len(arr)],
        }
        for field, value in expected.items():
            assert stats_out["stats"][stat_key][field] == pytest.approx(value), (
                f"{stat_key}.{field} mismatch"
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


def test_dry_run_and_real_run_report_same_coverage(raw_ds, tmp_path):
    """Fix 2: dry_run and a real run must agree on coverage for every
    episode and every arm -- defense-in-depth on top of both paths sharing
    _smooth_episode_frame (Fix 1)."""
    out_dry = tmp_path / "dry"
    out_real = tmp_path / "real"
    dry_results = smooth_dataset(raw_ds, out_dry, max_gap_s=0.25, dry_run=True)
    real_results = smooth_dataset(raw_ds, out_real, max_gap_s=0.25, dry_run=False)

    assert len(dry_results) == len(real_results) == 1
    for dry_r, real_r in zip(dry_results, real_results):
        assert dry_r.episode_index == real_r.episode_index
        assert dry_r.coverage.keys() == real_r.coverage.keys()
        for arm in dry_r.coverage:
            assert dry_r.coverage[arm] == real_r.coverage[arm]


def test_smooth_dataset_multi_episode(tmp_path):
    """Fix 3: smooth_dataset must walk every episode, not just the first,
    and each output episode's recomputed stats must reflect ITS OWN data."""
    root = tmp_path / "raw_multi"
    make_tiny_dataset(root, with_provenance=False, n_episodes=2)
    out = tmp_path / "smoothed_multi"

    results = smooth_dataset(root, out, max_gap_s=0.25)
    assert [r.episode_index for r in results] == [0, 1]

    stats_out = [
        json.loads(line)
        for line in (out / "meta/episodes_stats.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(stats_out) == 2

    per_episode_state_min = {}
    for ep in (0, 1):
        dst = out / f"data/chunk-000/episode_{ep:06d}.parquet"
        assert dst.is_file()
        df = pd.read_parquet(dst)
        assert "observation.provenance" in df.columns

        rec = next(r for r in stats_out if r["episode_index"] == ep)
        s = np.stack(df["observation.state"])
        a = np.stack(df["action"])
        assert rec["stats"]["observation.state"]["min"] == pytest.approx(
            s.min(0).tolist()
        )
        assert rec["stats"]["observation.state"]["max"] == pytest.approx(
            s.max(0).tolist()
        )
        assert rec["stats"]["action"]["mean"] == pytest.approx(a.mean(0).tolist())
        per_episode_state_min[ep] = rec["stats"]["observation.state"]["min"]

    # the two episodes' recomputed stats must not have been cross-wired
    assert per_episode_state_min[0] != per_episode_state_min[1]


def test_smooth_dataset_refuses_output_aliasing_root(raw_ds):
    """Fix 5: out == root (or an ancestor of root) must be refused before
    overwrite=True's shutil.rmtree(out) can destroy the input dataset."""
    with pytest.raises(ValueError, match="ancestor"):
        smooth_dataset(raw_ds, raw_ds, max_gap_s=0.25, overwrite=True)

    # input survived untouched
    assert (raw_ds / "meta" / "info.json").is_file()
    raw_df = pd.read_parquet(raw_ds / "data/chunk-000/episode_000000.parquet")
    assert "observation.provenance" not in raw_df.columns


def test_smooth_dataset_refuses_output_ancestor_of_root(raw_ds):
    """out one level above root is also an ancestor -- must be refused too."""
    with pytest.raises(ValueError, match="ancestor"):
        smooth_dataset(raw_ds, raw_ds.parent, max_gap_s=0.25, overwrite=True)
    assert (raw_ds / "meta" / "info.json").is_file()


def _files_under(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def test_smooth_dataset_refuses_output_nested_inside_root(raw_ds):
    """Mirror case of the aliasing guard: out nested INSIDE root must be
    refused too -- the input dataset directory must never be written to,
    in any code path."""
    before = _files_under(raw_ds)

    with pytest.raises(ValueError, match="nested inside"):
        smooth_dataset(raw_ds, raw_ds / "smoothed", max_gap_s=0.25)

    after = _files_under(raw_ds)
    assert before == after


def test_smooth_dataset_refuses_deeply_nested_output(raw_ds):
    """A deeper nesting under root must also be refused."""
    before = _files_under(raw_ds)

    with pytest.raises(ValueError, match="nested inside"):
        smooth_dataset(raw_ds, raw_ds / "a" / "b", max_gap_s=0.25)

    after = _files_under(raw_ds)
    assert before == after


def test_smooth_dataset_accepts_prefix_sharing_sibling(raw_ds, tmp_path):
    """Regression guard: a sibling output dir whose name merely shares the
    input's directory name as a string PREFIX (raw -> raw_smoothed) must
    still be accepted. A naive string-prefix check on the resolved paths
    would wrongly reject this since str(out).startswith(str(root)) is True
    even though raw_smoothed is not inside raw -- the guard must use
    Path.resolve()/.parents semantics instead."""
    out = tmp_path / "raw_smoothed"
    # confirms this is the tricky prefix case a naive string check would trip on
    assert str(out).startswith(str(raw_ds))

    results = smooth_dataset(raw_ds, out, max_gap_s=0.25)
    assert len(results) == 1
    assert (out / "data/chunk-000/episode_000000.parquet").is_file()


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
