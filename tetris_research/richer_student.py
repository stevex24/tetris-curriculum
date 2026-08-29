"""Interpretable placement policy and episodic policy-gradient learner for Day 4."""
from __future__ import annotations

import ast
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .game import ActionEvaluation
from .student import ActionScore, AgentExperience, Placement, PlacementDecision, StudentAgent
from .tetris import HEIGHT, SHAPES, WIDTH, TetrisState


FEATURE_NAMES = (
    "holes", "hole_depth", "rows_with_holes", "aggregate_height",
    "maximum_height", "height_stddev", "bumpiness", "cliffs",
    "landing_height", "completed_lines", "eroded_piece_cells",
    "row_transitions", "column_transitions", "cumulative_wells",
    "deep_wells", "covered_columns", "high_stack_cells", "surface_blocks",
)


def richer_features(state: TetrisState, piece: str, choice: ActionEvaluation) -> tuple[float, ...]:
    """Return 18 bounded/scaled, independently defined placement features."""
    rows = choice.next_state.rows
    heights, holes_by_column, depths = [], [], []
    hole_rows: set[int] = set()
    covered_columns = 0
    for x in range(WIDTH):
        occupied = [y for y, row in enumerate(rows) if row & (1 << x)]
        height = max(occupied) + 1 if occupied else 0
        heights.append(height)
        column_holes = [y for y in range(height) if not rows[y] & (1 << x)]
        holes_by_column.append(len(column_holes))
        hole_rows.update(column_holes)
        depths.extend(height - y for y in column_holes)
        covered_columns += bool(column_holes)
    mean_height = sum(heights) / WIDTH
    height_sd = math.sqrt(sum((height - mean_height) ** 2 for height in heights) / WIDTH)
    differences = [abs(a - b) for a, b in zip(heights, heights[1:])]

    row_transitions = 0
    for y in range(HEIGHT):
        previous = 1
        for x in range(WIDTH):
            occupied = int(bool(rows[y] & (1 << x)))
            row_transitions += occupied != previous
            previous = occupied
        row_transitions += previous != 1
    column_transitions = 0
    for x in range(WIDTH):
        previous = 1
        for y in range(HEIGHT):
            occupied = int(bool(rows[y] & (1 << x)))
            column_transitions += occupied != previous
            previous = occupied

    wells = deep_wells = 0
    for x in range(WIDTH):
        depth = 0
        for y in range(HEIGHT):
            occupied = bool(rows[y] & (1 << x))
            left = x == 0 or bool(rows[y] & (1 << (x - 1)))
            right = x == WIDTH - 1 or bool(rows[y] & (1 << (x + 1)))
            if not occupied and left and right:
                depth += 1
                wells += depth
                deep_wells += depth >= 3
            else:
                depth = 0

    rotation, x = map(int, choice.action)
    cells = SHAPES[piece][rotation]
    landing_y = int(choice.info["landing_y"])
    cleared = int(choice.info["lines_cleared"])
    cleared_rows: set[int] = set()
    if cleared:
        before = list(state.rows)
        for dx, dy in cells:
            before[landing_y + dy] |= 1 << (x + dx)
        cleared_rows = {y for y, row in enumerate(before) if row == (1 << WIDTH) - 1}
    eroded = cleared * sum(landing_y + dy in cleared_rows for _, dy in cells)
    surface_blocks = sum(bool(row) for row in rows)
    values = (
        sum(holes_by_column) / 40.0,
        sum(depths) / 200.0,
        len(hole_rows) / HEIGHT,
        sum(heights) / (WIDTH * HEIGHT),
        max(heights) / HEIGHT,
        height_sd / HEIGHT,
        sum(differences) / (WIDTH * HEIGHT),
        sum(value >= 3 for value in differences) / (WIDTH - 1),
        (landing_y + sum(dy for _, dy in cells) / 4.0) / HEIGHT,
        cleared / 4.0,
        eroded / 16.0,
        row_transitions / (2 * WIDTH * HEIGHT),
        column_transitions / (2 * WIDTH * HEIGHT),
        wells / 200.0,
        deep_wells / (WIDTH * HEIGHT),
        covered_columns / WIDTH,
        sum(max(0, height - 14) for height in heights) / 60.0,
        surface_blocks / HEIGHT,
    )
    return tuple(float(value) for value in values)


class RicherRLStudent(StudentAgent):
    """Linear softmax policy trained only by episodic REINFORCE returns."""

    VERSION = "interpretable-18-feature-reinforce-v1"

    def __init__(self, agent_id: str, weights: Sequence[float] | None = None, *,
                 learning_rate: float = 0.02, discount: float = 0.99,
                 temperature: float = 0.35, seed: int = 0,
                 learning_enabled: bool = True):
        self._agent_id = agent_id
        self.weights = list(weights if weights is not None else (0.0,) * len(FEATURE_NAMES))
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError(f"richer student requires {len(FEATURE_NAMES)} weights")
        self.learning_rate, self.discount, self.temperature = map(float, (learning_rate, discount, temperature))
        self.seed, self._learning_enabled = int(seed), bool(learning_enabled)
        self._rng = random.Random(self.seed)
        self._episode: list[tuple[tuple[float, ...], float]] = []
        self.updates = self.episodes_learned = self.placements_seen = 0
        self.return_baseline = 0.0
        self.return_observations = 0

    @property
    def agent_id(self) -> str: return self._agent_id

    @property
    def agent_version(self) -> str: return self.VERSION

    @property
    def learning_enabled(self) -> bool: return self._learning_enabled

    @staticmethod
    def _placement(choice: ActionEvaluation) -> Placement:
        return Placement(int(choice.action[0]), int(choice.action[1]))

    def clone(self, agent_id: str | None = None) -> "RicherRLStudent":
        result = RicherRLStudent(agent_id or self.agent_id, self.weights,
                                 learning_rate=self.learning_rate, discount=self.discount,
                                 temperature=self.temperature, seed=self.seed,
                                 learning_enabled=self.learning_enabled)
        result._rng.setstate(self._rng.getstate())
        result._episode = list(self._episode)
        result.updates, result.episodes_learned = self.updates, self.episodes_learned
        result.placements_seen = self.placements_seen
        result.return_baseline, result.return_observations = self.return_baseline, self.return_observations
        return result

    def _vectors(self, state: TetrisState, piece: str,
                 legal: Sequence[ActionEvaluation]) -> list[tuple[float, ...]]:
        return [richer_features(state, piece, choice) for choice in legal]

    def action_scores(self, state: TetrisState, piece: str,
                      legal_placements: Sequence[ActionEvaluation]) -> tuple[ActionScore, ...]:
        vectors = self._vectors(state, piece, legal_placements)
        return tuple(ActionScore(self._placement(choice), sum(w * f for w, f in zip(self.weights, vector)))
                     for choice, vector in zip(legal_placements, vectors))

    def choose_placement(self, state: TetrisState, piece: str,
                         legal_placements: Sequence[ActionEvaluation], *, learn: bool = False,
                         deterministic: bool = False) -> PlacementDecision:
        if not legal_placements:
            raise ValueError("choose_placement requires a legal placement")
        if learn and not self.learning_enabled:
            raise RuntimeError("learning is disabled for this agent")
        vectors = self._vectors(state, piece, legal_placements)
        scores = [sum(w * f for w, f in zip(self.weights, vector)) for vector in vectors]
        if deterministic:
            selected = max(range(len(scores)), key=scores.__getitem__)
        else:
            logits = [score / self.temperature for score in scores]
            peak = max(logits)
            exps = [math.exp(value - peak) for value in logits]
            probabilities = [value / sum(exps) for value in exps]
            draw, selected, cumulative = self._rng.random(), len(scores) - 1, 0.0
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if draw <= cumulative:
                    selected = index
                    break
            if learn:
                expected = tuple(sum(probability * vector[j]
                                     for probability, vector in zip(probabilities, vectors))
                                 for j in range(len(FEATURE_NAMES)))
                gradient = tuple((vectors[selected][j] - expected[j]) / self.temperature
                                 for j in range(len(FEATURE_NAMES)))
                self._episode.append((gradient, math.nan))
        choice = legal_placements[selected]
        preferences = tuple(ActionScore(self._placement(item), score)
                            for item, score in zip(legal_placements, scores))
        return PlacementDecision(self._placement(choice), choice, preferences)

    def update(self, experience: AgentExperience) -> None:
        if not self.learning_enabled or not self._episode or not math.isnan(self._episode[-1][1]):
            raise RuntimeError("update must follow exactly one learning choice")
        gradient, _ = self._episode[-1]
        self._episode[-1] = (gradient, float(experience.reward))
        self.placements_seen += 1

    def learn_from_label(self, state: TetrisState, piece: str,
                         legal_placements: Sequence[ActionEvaluation], label: Placement,
                         *, learning_rate: float | None = None) -> float:
        """Apply one supervised softmax update in the student's 18-feature space."""
        if not self.learning_enabled:
            raise RuntimeError("learning is disabled for this agent")
        if self._episode:
            raise RuntimeError("imitation update cannot interrupt an RL trajectory")
        indices = [index for index, choice in enumerate(legal_placements)
                   if self._placement(choice) == label]
        if len(indices) != 1:
            raise ValueError("label must identify exactly one legal placement")
        vectors = self._vectors(state, piece, legal_placements)
        logits = [sum(weight * value for weight, value in zip(self.weights, vector)) /
                  self.temperature for vector in vectors]
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        total = sum(exps)
        probabilities = [value / total for value in exps]
        target = indices[0]
        expected = [sum(probability * vector[j]
                        for probability, vector in zip(probabilities, vectors))
                    for j in range(len(FEATURE_NAMES))]
        rate = self.learning_rate if learning_rate is None else float(learning_rate)
        self.weights = [weight + rate * (wanted - average) / self.temperature
                        for weight, wanted, average in zip(
                            self.weights, vectors[target], expected)]
        self.updates += 1
        self.placements_seen += 1
        return -math.log(max(probabilities[target], 1e-300))

    def finish_episode(self, *, learned: bool) -> None:
        if not learned:
            if self._episode:
                raise RuntimeError("cannot discard pending learning trajectory")
            return
        if not self._episode or any(math.isnan(reward) for _, reward in self._episode):
            raise RuntimeError("learned episode requires completed experiences")
        returns, value = [], 0.0
        for _, reward in reversed(self._episode):
            value = reward + self.discount * value
            returns.append(value)
        returns.reverse()
        # The baseline contains only earlier trajectories, so it cannot leak a
        # later reward backward except through the explicitly computed return.
        baseline = self.return_baseline
        for (gradient, _), value in zip(self._episode, returns):
            advantage = max(-10.0, min(10.0, value - baseline))
            self.weights = [weight + self.learning_rate * advantage * component
                            for weight, component in zip(self.weights, gradient)]
            self.updates += 1
        total = self.return_baseline * self.return_observations + sum(returns)
        self.return_observations += len(returns)
        self.return_baseline = total / self.return_observations
        self._episode.clear()
        self.episodes_learned += 1

    def serialize_state(self) -> Mapping[str, Any]:
        return {"format": self.VERSION, "agent_id": self.agent_id,
                "agent_version": self.agent_version, "learning_enabled": self.learning_enabled,
                "feature_names": list(FEATURE_NAMES), "weights": list(self.weights),
                "learning_rate": self.learning_rate, "discount": self.discount,
                "temperature": self.temperature, "seed": self.seed,
                "rng_state": repr(self._rng.getstate()), "updates": self.updates,
                "episodes_learned": self.episodes_learned,
                "placements_seen": self.placements_seen, "pending_steps": len(self._episode),
                "return_baseline": self.return_baseline,
                "return_observations": self.return_observations,
                "expert_parameters": None}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.serialize_state(), indent=2) + "\n")

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "RicherRLStudent":
        if state.get("format") != cls.VERSION or state.get("pending_steps"):
            raise ValueError("unsupported or mid-episode richer student state")
        result = cls(str(state["agent_id"]), state["weights"],
                     learning_rate=float(state["learning_rate"]), discount=float(state["discount"]),
                     temperature=float(state["temperature"]), seed=int(state["seed"]),
                     learning_enabled=bool(state["learning_enabled"]))
        result._rng.setstate(ast.literal_eval(str(state["rng_state"])))
        result.updates, result.episodes_learned = int(state["updates"]), int(state["episodes_learned"])
        result.placements_seen = int(state["placements_seen"])
        result.return_baseline = float(state["return_baseline"])
        result.return_observations = int(state["return_observations"])
        return result

    @classmethod
    def load(cls, path: Path) -> "RicherRLStudent":
        return cls.from_state(json.loads(path.read_text()))
