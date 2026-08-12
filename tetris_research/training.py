from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable

from .agent import LearningAgent
from .tetris import HEIGHT, WIDTH, TetrisAdapter, TetrisState

CONTROL = "control"
RATING_ONLY = "rating_only"
RATING_HISTORY = "rating_history"
CONDITIONS = (CONTROL, RATING_ONLY, RATING_HISTORY)


@dataclass(frozen=True)
class TrainingMaterial:
    condition: str
    state: TetrisState
    difficulty: str
    diagnosed_weakness: str | None
    diagnosis: dict[str, float]
    rationale: str
    selection_seed: int | None


def rating_difficulty(rating: float) -> str:
    """Prototype assumption, not a calibrated Tetris-pedagogy mapping."""
    if rating < 1400:
        return "introductory"
    if rating < 1600:
        return "intermediate"
    return "advanced"


def _state_from_bottom_rows(rows: Iterable[str]) -> TetrisState:
    source = list(rows)
    if len(source) > HEIGHT or any(len(row) != WIDTH or set(row) - {".", "#"} for row in source):
        raise ValueError("training rows must be at most 20 strings of ten '.'/'#' cells")
    masks = [sum(1 << x for x, cell in enumerate(row) if cell == "#") for row in source]
    if any(mask == (1 << WIDTH) - 1 for mask in masks):
        raise ValueError("a starting board may not contain an uncleared full row")
    return TetrisState(tuple(masks + [0] * (HEIGHT - len(masks))))


def board_rows(state: TetrisState) -> list[str]:
    """Human-oriented rows, top first, with x=0 at the left."""
    rows = ["".join("#" if state.rows[y] & (1 << x) else "." for x in range(WIDTH))
            for y in range(HEIGHT - 1, -1, -1)]
    first = next((i for i, row in enumerate(rows) if "#" in row), len(rows) - 1)
    return rows[first:]


def _rating_board(difficulty: str, seed: int) -> TetrisState:
    # Seed controls a harmless mirror, making material reproducible without
    # diagnosing an individual player. These tiers are prototype assumptions.
    templates = {
        "introductory": ["####...###", "#####.####"],
        "intermediate": ["####...###", "#####.####", "###...####", "######.###"],
        "advanced": ["####...###", "#####.####", "###...####", "######.###",
                     "####...###", "#####..###"],
    }
    rows = templates[difficulty]
    if seed % 2:
        rows = [row[::-1] for row in rows]
    return _state_from_bottom_rows(rows)


def select_rating_only_material(rating: float, seed: int) -> TrainingMaterial:
    difficulty = rating_difficulty(rating)
    return TrainingMaterial(
        RATING_ONLY, _rating_board(difficulty, seed), difficulty, None, {},
        "Board height/complexity comes only from the prototype Elo tier; no player history was inspected.",
        seed,
    )


def observable_history_features(records: Iterable[dict[str, Any]], recent_steps: int = 40) -> dict[str, float]:
    steps = [step for record in records for step in record.get("steps", [])][-recent_steps:]
    if not steps:
        return {"hole_avoidance": 0.0, "stack_height": 0.0, "bumpiness": 0.0,
                "line_efficiency": 1.0, "steps_observed": 0.0}
    mean = lambda key: sum(float(step["features"][key]) for step in steps) / len(steps)
    lines = sum(float(step["features"]["lines_cleared"]) for step in steps)
    # All four are dimensionless severity scores; larger means weaker. The
    # constants are transparent prototype reference scales, not calibrated norms.
    return {
        "hole_avoidance": mean("holes") / 20.0,
        "stack_height": mean("max_height") / HEIGHT,
        "bumpiness": mean("bumpiness") / 40.0,
        "line_efficiency": max(0.0, 1.0 - lines / (len(steps) / 10.0)),
        "steps_observed": float(len(steps)),
    }


def diagnose_weakness(features: dict[str, float]) -> str:
    dimensions = ("hole_avoidance", "stack_height", "bumpiness", "line_efficiency")
    return max(dimensions, key=lambda name: features[name])


def _history_board(weakness: str, difficulty: str, seed: int) -> TetrisState:
    templates = {
        "hole_avoidance": ["###..#####", "####.#####", "##.#.#####"],
        "stack_height": ["####.#####"] * 6,
        "bumpiness": ["#.#.#.#.#.", "#.#...#.#.", "#.....#..."],
        "line_efficiency": ["####.#####"] * 3,
    }
    rows = list(templates[weakness])
    # Elo controls only a broad exposure adjustment shared conceptually with the
    # rating-only tier: higher tiers get one/two additional non-full foundation rows.
    if difficulty in ("intermediate", "advanced"):
        rows.insert(0, "#####.####")
    if difficulty == "advanced":
        rows.insert(0, "####.#####")
    if seed % 2:
        rows = [row[::-1] for row in rows]
    return _state_from_bottom_rows(rows)


def select_history_material(rating: float, recent_records: Iterable[dict[str, Any]],
                            seed: int) -> TrainingMaterial:
    diagnosis = observable_history_features(recent_records)
    weakness = diagnose_weakness(diagnosis)
    difficulty = rating_difficulty(rating)
    rationales = {
        "hole_avoidance": "Pre-existing cavities and competing landing surfaces expose choices that add different numbers of holes.",
        "stack_height": "A tall, narrow-well stack makes height control and line reduction immediately relevant.",
        "bumpiness": "An alternating skyline exposes placements with sharply different bumpiness outcomes.",
        "line_efficiency": "Nearly complete rows sharing a well provide repeated observable line-clear opportunities.",
    }
    return TrainingMaterial(RATING_HISTORY, _history_board(weakness, difficulty, seed), difficulty,
                            weakness, diagnosis, rationales[weakness], seed)


def control_material() -> TrainingMaterial:
    return TrainingMaterial(CONTROL, TetrisAdapter().initial_state(), "ordinary", None, {},
                            "Ordinary practice starts from an empty board.", None)


def train(agent: LearningAgent, material: TrainingMaterial, rating: float, steps: int,
          seed: int) -> dict[str, Any]:
    """Apply exactly `steps` ordinary policy-learning placements."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    adapter, piece_rng = TetrisAdapter(), Random(seed)
    before, state, placements, episodes = agent.weights.copy(), material.state, [], 1
    while len(placements) < steps:
        piece = adapter.sample_environment(piece_rng)
        choices = adapter.legal_actions(state, piece)
        if not choices:
            state, episodes = material.state, episodes + 1
            continue
        old_weights = agent.weights.copy()
        choice = agent.choose(choices, learn=True)
        state = choice.next_state
        placements.append({"piece": piece, "action": list(choice.action), "reward": choice.reward,
                           "features": choice.raw_features, "weights_before": old_weights,
                           "weights_after": agent.weights.copy()})
    agent.finish_game(learned=True)
    return {
        "agent_id": agent.name, "condition": material.condition, "starting_elo": rating,
        "training_steps": len(placements), "recent_history_features": material.diagnosis,
        "diagnosed_weakness": material.diagnosed_weakness, "difficulty": material.difficulty,
        "selection_rationale": material.rationale, "tutorial_board": board_rows(material.state),
        "policy_weights_before": before, "policy_weights_after": agent.weights.copy(),
        "selection_seed": material.selection_seed, "piece_stream_seed": seed,
        "random_seed": seed, "episodes_used": episodes, "placements": placements,
        "learning_path": "LearningAgent.choose(choices, learn=True)",
    }


def write_training_log(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n")
