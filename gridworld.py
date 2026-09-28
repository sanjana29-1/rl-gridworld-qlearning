"""A small GridWorld environment (no external RL libraries needed).

The agent starts at S and must reach the goal G while avoiding pits (X).
Walls (#) block movement. Each step costs -1, falling in a pit costs -50
and ends the episode, reaching the goal gives +50.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MAP = [
    "S...#....",
    ".##.#.##.",
    ".#..X..#.",
    ".#.###.#.",
    "...#X..#.",
    ".#.#.#.#.",
    ".#...#.#X",
    "...#....G",
]

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class GridWorld:
    def __init__(self, grid: list[str] = DEFAULT_MAP, slip_prob: float = 0.1, max_steps: int = 200, seed: int | None = None):
        self.grid = [list(row) for row in grid]
        self.h, self.w = len(grid), len(grid[0])
        self.start = self._find("S")
        self.goal = self._find("G")
        self.slip_prob = slip_prob  # chance the agent moves in a random direction (stochastic env)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.n_states, self.n_actions = self.h * self.w, len(ACTIONS)

    def _find(self, ch: str) -> tuple[int, int]:
        for r, row in enumerate(self.grid):
            if ch in row:
                return r, row.index(ch)
        raise ValueError(ch)

    def to_state(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.w + pos[1]

    def reset(self) -> int:
        self.pos, self.steps = self.start, 0
        return self.to_state(self.pos)

    def step(self, action: int) -> tuple[int, float, bool]:
        if self.rng.random() < self.slip_prob:
            action = int(self.rng.integers(self.n_actions))
        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < self.h and 0 <= c < self.w and self.grid[r][c] != "#":
            self.pos = (r, c)
        self.steps += 1
        cell = self.grid[self.pos[0]][self.pos[1]]
        if cell == "G":
            return self.to_state(self.pos), 50.0, True
        if cell == "X":
            return self.to_state(self.pos), -50.0, True
        return self.to_state(self.pos), -1.0, self.steps >= self.max_steps
