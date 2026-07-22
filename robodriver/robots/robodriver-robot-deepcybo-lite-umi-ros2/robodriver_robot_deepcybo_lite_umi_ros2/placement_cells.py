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
