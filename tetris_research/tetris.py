from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .game import ActionEvaluation

WIDTH, HEIGHT = 10, 20

# Unique rotation states, normalized to the top-left. Wall kicks, hold, previews,
# lock delay, and spins are intentionally outside this prototype.
SHAPES = {
    "I": (((0, 0), (1, 0), (2, 0), (3, 0)), ((0, 0), (0, 1), (0, 2), (0, 3))),
    "O": (((0, 0), (1, 0), (0, 1), (1, 1)),),
    "T": (((0, 0), (1, 0), (2, 0), (1, 1)), ((1, 0), (0, 1), (1, 1), (1, 2)), ((1, 0), (0, 1), (1, 1), (2, 1)), ((0, 0), (0, 1), (1, 1), (0, 2))),
    "S": (((1, 0), (2, 0), (0, 1), (1, 1)), ((0, 0), (0, 1), (1, 1), (1, 2))),
    "Z": (((0, 0), (1, 0), (1, 1), (2, 1)), ((1, 0), (0, 1), (1, 1), (0, 2))),
    "J": (((0, 0), (0, 1), (1, 1), (2, 1)), ((0, 0), (1, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (2, 0), (2, 1)), ((1, 0), (1, 1), (0, 2), (1, 2))),
    "L": (((2, 0), (0, 1), (1, 1), (2, 1)), ((0, 0), (0, 1), (0, 2), (1, 2)), ((0, 0), (1, 0), (2, 0), (0, 1)), ((0, 0), (1, 0), (1, 1), (1, 2))),
}


@dataclass(frozen=True)
class TetrisState:
    rows: tuple[int, ...] = (0,) * HEIGHT  # y=0 is the floor
    game_over: bool = False


class TetrisAdapter:
    feature_names = ("holes", "max_height", "bumpiness", "lines_cleared")

    def initial_state(self) -> TetrisState:
        return TetrisState()

    def sample_environment(self, rng: Random) -> str:
        return rng.choice(tuple(SHAPES))

    def terminal(self, state: TetrisState) -> bool:
        return state.game_over

    def legal_actions(self, state: TetrisState, piece: str) -> list[ActionEvaluation]:
        if state.game_over:
            return []
        results = []
        for rotation, cells in enumerate(SHAPES[piece]):
            shape_width = max(x for x, _ in cells) + 1
            for x in range(WIDTH - shape_width + 1):
                y = HEIGHT
                while y > 0 and not self._collides(state.rows, cells, x, y - 1):
                    y -= 1
                if self._collides(state.rows, cells, x, y):
                    continue
                occupied = [(x + dx, y + dy) for dx, dy in cells]
                if any(py >= HEIGHT for _, py in occupied):
                    continue
                rows = list(state.rows)
                for px, py in occupied:
                    rows[py] |= 1 << px
                full = (1 << WIDTH) - 1
                cleared = sum(row == full for row in rows)
                rows = [row for row in rows if row != full] + [0] * cleared
                next_state = TetrisState(tuple(rows))
                raw = self._features(next_state, cleared)
                # Comparable scales keep learning numerically tame and interpretable.
                features = (-raw["holes"] / 20, -raw["max_height"] / HEIGHT,
                            -raw["bumpiness"] / 40, raw["lines_cleared"] / 4)
                reward = cleared * cleared + 0.01 - 0.02 * raw["holes"]
                results.append(ActionEvaluation(
                    (rotation, x), features, raw, next_state, reward,
                    {"piece": piece, "rotation": rotation, "x": x, "landing_y": y,
                     "lines_cleared": cleared},
                ))
        return results

    @staticmethod
    def _collides(rows: tuple[int, ...], cells: tuple[tuple[int, int], ...], x: int, y: int) -> bool:
        return any(y + dy < 0 or (y + dy < HEIGHT and rows[y + dy] & (1 << (x + dx))) for dx, dy in cells)

    @staticmethod
    def _features(state: TetrisState, cleared: int) -> dict[str, float]:
        heights, holes = [], 0
        for x in range(WIDTH):
            occupied = [y for y, row in enumerate(state.rows) if row & (1 << x)]
            height = max(occupied) + 1 if occupied else 0
            heights.append(height)
            holes += sum(not (state.rows[y] & (1 << x)) for y in range(height))
        return {"holes": float(holes), "max_height": float(max(heights)),
                "bumpiness": float(sum(abs(a - b) for a, b in zip(heights, heights[1:]))),
                "lines_cleared": float(cleared)}

    def serialize_state(self, state: TetrisState) -> list[str]:
        return [format(row, f"0{WIDTH}b") for row in reversed(state.rows)]
