"""Small post-processing utilities extracted from the notebook workflow."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

Grid = Sequence[Sequence[float]]
Point = tuple[int, int]


def _neighbors(row: int, col: int, height: int, width: int) -> Iterable[Point]:
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue
            n_row = row + d_row
            n_col = col + d_col
            if 0 <= n_row < height and 0 <= n_col < width:
                yield n_row, n_col


def hysteresis_components(probabilities: Grid, low: float, high: float) -> list[list[int]]:
    """Keep weak connected components only when they touch a strong seed.

    This mirrors the idea used in the inference workflow: high-confidence voxels
    act as anchors, while lower-confidence nearby regions are retained only if
    they connect to those anchors. The helper is intentionally 2D and lightweight
    so it can be unit-tested without GPU or medical-imaging dependencies.
    """

    if not 0.0 <= low < high <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
    if not probabilities:
        return []

    height = len(probabilities)
    width = len(probabilities[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    visited = [[False for _ in range(width)] for _ in range(height)]

    for row in range(height):
        for col in range(width):
            if visited[row][col] or probabilities[row][col] < low:
                continue

            queue: deque[Point] = deque([(row, col)])
            component: list[Point] = []
            touches_strong = False
            visited[row][col] = True

            while queue:
                cur_row, cur_col = queue.popleft()
                component.append((cur_row, cur_col))
                touches_strong = touches_strong or probabilities[cur_row][cur_col] >= high

                for n_row, n_col in _neighbors(cur_row, cur_col, height, width):
                    if visited[n_row][n_col] or probabilities[n_row][n_col] < low:
                        continue
                    visited[n_row][n_col] = True
                    queue.append((n_row, n_col))

            if touches_strong:
                for comp_row, comp_col in component:
                    output[comp_row][comp_col] = 1

    return output
