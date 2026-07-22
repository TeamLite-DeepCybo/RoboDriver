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
