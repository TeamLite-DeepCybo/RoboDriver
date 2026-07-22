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


def test_passes_are_distinct_orderings():
    """Guards against regression: shuffling once and replaying per pass.

    A plausible bug is shuffling the cell list once before the loop and
    reusing that same shuffled ordering for every pass, which would make
    placements predictable and defeat randomization. This test detects that
    by generating several full passes and verifying they are not all identical.
    """
    # Use a 3x4 grid (12 cells) with enough passes that collision is negligible
    seq = balanced_sequence(3, 4, 48, seed=42)  # 4 full passes
    pass1 = tuple(seq[0:12])
    pass2 = tuple(seq[12:24])
    pass3 = tuple(seq[24:36])
    pass4 = tuple(seq[36:48])

    # Assert that the number of distinct pass-tuples is > 1
    # (i.e., they are not all the same ordering)
    distinct_passes = len({pass1, pass2, pass3, pass4})
    assert distinct_passes > 1, "All passes had the same ordering"


def test_balanced_sequence_negative_n():
    """Negative n should raise ValueError, not silently return empty list."""
    import pytest
    with pytest.raises(ValueError, match="n must be non-negative"):
        balanced_sequence(3, 4, -5)


def test_cli_invalid_grid_exits_cleanly(capsys):
    """Invalid grid rows should exit with SystemExit, not raise ValueError."""
    import pytest
    with pytest.raises(SystemExit):
        main(["--rows", "0", "--cols", "4", "-n", "10"])
    captured = capsys.readouterr()
    # argparse.error() prints usage and exits; should see usage text
    assert "usage" in captured.err.lower() or "bad grid" in captured.err
