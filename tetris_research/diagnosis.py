"""Transparent Day 5 diagnosis of recurring expert-relative behavioral errors."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .expert import DellacherieSearchExpert
from .richer_student import FEATURE_NAMES, richer_features
from .student import Placement
from .tetris import HEIGHT, TetrisAdapter, TetrisState


FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "hole_management": ("holes", "hole_depth", "rows_with_holes", "covered_columns"),
    "stack_height_danger": ("aggregate_height", "maximum_height", "high_stack_cells", "landing_height"),
    "surface_smoothness": ("height_stddev", "bumpiness", "cliffs", "surface_blocks"),
    "well_management": ("cumulative_wells", "deep_wells"),
    "transition_structure": ("row_transitions", "column_transitions"),
    "line_clear_recovery": ("completed_lines", "eroded_piece_cells"),
}

FEATURE_TO_FAMILY = {feature: family for family, features in FEATURE_FAMILIES.items()
                     for feature in features}
BENEFICIAL_FEATURES = {"completed_lines", "eroded_piece_cells"}


def _rows(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != HEIGHT or not all(isinstance(x, int) for x in value):
        raise ValueError("history rows must contain exactly 20 integer bitmasks")
    return tuple(value)


def diagnose_history(records: Sequence[Mapping[str, Any]], *, regret_threshold: float = 1.0,
                     delta_threshold: float = 0.01,
                     expert: DellacherieSearchExpert | None = None) -> dict[str, Any]:
    """Replay observable history and return a family-level weakness profile.

    Unknown fields are ignored deliberately: policy parameters are neither needed nor read.
    """
    if set(FEATURE_TO_FAMILY) != set(FEATURE_NAMES):
        raise AssertionError("feature-family mapping must cover every richer feature exactly once")
    game = TetrisAdapter()
    oracle = expert or DellacherieSearchExpert(beam_width=1)
    contributions: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    games: dict[str, set[str]] = defaultdict(set)
    decisions = material = near_zero = 0
    event_order: list[dict[str, Any]] = []
    seen_games: set[str] = set()
    for record in records:
        game_id = str(record["game_id"])
        seen_games.add(game_id)
        previous_step = -1
        for step in record["steps"]:
            index = int(step["step_index"])
            if index != previous_step + 1:
                raise ValueError("step indices must be contiguous within each game")
            previous_step = index
            state, piece = TetrisState(_rows(step["board_before"])), str(step["piece"])
            legal = game.legal_actions(state, piece)
            chosen = Placement(*map(int, step["student_action"]))
            selected = next((item for item in legal if tuple(item.action) ==
                             (chosen.rotation, chosen.x)), None)
            if selected is None:
                raise ValueError("recorded student action is not legal on recorded board")
            if list(selected.next_state.rows) != step["board_after"]:
                raise ValueError("recorded afterstate cannot be independently reconstructed")
            observed = tuple(float(x) for x in step["feature_vector"])
            if len(observed) != len(FEATURE_NAMES) or any(not math.isfinite(x) for x in observed):
                raise ValueError("history requires a finite 18-feature chosen-action vector")
            replayed = richer_features(state, piece, selected)
            if any(abs(a - b) > 1e-12 for a, b in zip(observed, replayed)):
                raise ValueError("recorded feature vector does not match replay")
            decisions += 1
            # The recorded action is validated and fixed before this first oracle call.
            event_order.append({"game_id": game_id, "step_index": index,
                                "events": ["recorded_student_decision", "external_oracle_evaluation"]})
            ranking = oracle.rank_placements(state, piece, legal)
            values = {item.placement: item.value for item in ranking}
            regret = max(0.0, ranking[0].value - values[chosen])
            if regret < regret_threshold:
                near_zero += 1
                continue
            material += 1
            best_choice = next(item for item in legal if tuple(item.action) ==
                               (ranking[0].placement.rotation, ranking[0].placement.x))
            best = richer_features(state, piece, best_choice)
            evidence: dict[str, float] = defaultdict(float)
            family_feature_counts: dict[str, int] = defaultdict(int)
            for name, student_value, expert_value in zip(FEATURE_NAMES, observed, best):
                harmful = (expert_value - student_value if name in BENEFICIAL_FEATURES
                           else student_value - expert_value)
                if harmful > delta_threshold:
                    family = FEATURE_TO_FAMILY[name]
                    evidence[family] += harmful
                    family_feature_counts[family] += 1
            evidence = {family: value / family_feature_counts[family]
                        for family, value in evidence.items()}
            total = sum(evidence.values())
            if total <= 0:
                continue
            for family, value in evidence.items():
                share = value / total
                contributions[family] += regret * share
                counts[family] += 1
                games[family].add(game_id)
    total_games = len(seen_games)
    scores = {}
    for family in FEATURE_FAMILIES:
        coverage = len(games[family]) / total_games if total_games else 0.0
        scores[family] = contributions[family] / decisions * coverage if decisions else 0.0
    ranking = sorted(scores, key=lambda name: (-scores[name], name))
    return {
        "schema_version": 1,
        "history_only": True,
        "feature_families": {key: list(value) for key, value in FEATURE_FAMILIES.items()},
        "thresholds": {"material_regret": regret_threshold, "feature_delta": delta_threshold},
        "scoring_rule": "attributed regret / all decisions * fraction of games containing attribution",
        "decision_count": decisions, "game_count": total_games,
        "material_regret_decisions": material, "below_threshold_decisions": near_zero,
        "families": {family: {"score": scores[family],
                               "attributed_regret": contributions[family],
                               "attributed_decisions": counts[family],
                               "games_with_attribution": len(games[family])}
                     for family in FEATURE_FAMILIES},
        "ranking": ranking,
        "event_order": event_order,
    }
