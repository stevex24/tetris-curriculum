"""Transparent Day 6 tutorial-state generation and equal-budget RL plumbing."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .day4 import gameplay_reward
from .diagnosis import FEATURE_FAMILIES
from .richer_student import FEATURE_NAMES, RicherRLStudent, richer_features
from .student import AgentExperience
from .tetris import HEIGHT, SHAPES, WIDTH, TetrisAdapter, TetrisState

SUPPORTED_FAMILIES = ("surface_smoothness", "well_management", "hole_management")


@dataclass(frozen=True)
class TutorialSituation:
    """A serializable board and current piece; no answer or reward is stored."""
    situation_id: str
    family: str
    rows: tuple[int, ...]
    piece: str
    source_id: str
    perturbation: str

    @property
    def state(self) -> TetrisState:
        return TetrisState(self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {"situation_id": self.situation_id, "family": self.family,
                "rows": list(self.rows), "piece": self.piece,
                "source_id": self.source_id, "perturbation": self.perturbation}


def _validate_rows(rows: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(row) for row in rows)
    full = (1 << WIDTH) - 1
    if len(result) != HEIGHT or any(row < 0 or row > full or row == full for row in result):
        raise ValueError("tutorial board must contain 20 legal, uncleared 10-bit rows")
    return result


def targeting_metric(state: TetrisState, piece: str, family: str) -> float:
    """Mean target-family feature range over legal actions (larger = more choice sensitivity)."""
    if family not in FEATURE_FAMILIES:
        raise ValueError(f"unknown family: {family}")
    legal = TetrisAdapter().legal_actions(state, piece)
    if len(legal) < 2:
        return 0.0
    indices = [FEATURE_NAMES.index(name) for name in FEATURE_FAMILIES[family]]
    vectors = [richer_features(state, piece, choice) for choice in legal]
    return sum(max(v[i] for v in vectors) - min(v[i] for v in vectors)
               for i in indices) / len(indices)


def _plausible(state: TetrisState) -> bool:
    occupied = sum(bin(row).count("1") for row in state.rows)
    highest = max((y + 1 for y, row in enumerate(state.rows) if row), default=0)
    return not state.game_over and 4 <= occupied <= 120 and highest <= 16


def candidate_pool(history: Sequence[Mapping[str, Any]], seed: int) -> list[tuple[TetrisState, str, str, str]]:
    """Natural states plus legal one-placement neighbors, deterministically sampled."""
    game, rng, candidates = TetrisAdapter(), random.Random(seed), []
    pieces = tuple(SHAPES)
    for game_record in history:
        for step in game_record.get("steps", []):
            source = f'{step["game_id"]}:{step["step_index"]}'
            base = TetrisState(_validate_rows(step["board_before"]))
            if _plausible(base):
                candidates.append((base, str(step["piece"]), source, "historical"))
            legal = game.legal_actions(base, str(step["piece"]))
            if legal:
                # One legal action keeps the variant exactly one natural simulator step away.
                choice = legal[rng.randrange(len(legal))]
                neighbor = choice.next_state
                next_piece = pieces[rng.randrange(len(pieces))]
                if _plausible(neighbor) and len(game.legal_actions(neighbor, next_piece)) >= 2:
                    candidates.append((neighbor, next_piece, source, "one_legal_placement"))
    return candidates


def generate_tutorial_sets(history: Sequence[Mapping[str, Any]], *, seed: int,
                           per_family: int = 8) -> dict[str, tuple[TutorialSituation, ...]]:
    pool = candidate_pool(history, seed)
    result: dict[str, tuple[TutorialSituation, ...]] = {}
    for family in SUPPORTED_FAMILIES:
        ranked = sorted(pool, key=lambda row: (-targeting_metric(row[0], row[1], family),
                                               row[2], row[1], row[0].rows))
        chosen, boards, sources = [], set(), set()
        for state, piece, source, perturbation in ranked:
            fingerprint = (state.rows, piece)
            if fingerprint in boards:
                continue
            # Require multiple historical origins, while allowing more than one useful variant per origin.
            if len(chosen) >= 2 and source in sources and len(sources) < min(4, per_family):
                continue
            digest = hashlib.sha256((family + piece + repr(state.rows)).encode()).hexdigest()[:12]
            chosen.append(TutorialSituation(f"{family}-{digest}", family, state.rows, piece,
                                             source, perturbation))
            boards.add(fingerprint); sources.add(source)
            if len(chosen) == per_family:
                break
        if len(chosen) != per_family:
            raise ValueError(f"insufficient diverse candidates for {family}")
        result[family] = tuple(chosen)
    return result


def generic_tutorial_set(history: Sequence[Mapping[str, Any]], *, seed: int,
                         count: int = 8) -> tuple[TutorialSituation, ...]:
    """Profile-blind deterministic sample from plausible natural/neighbor states."""
    pool = candidate_pool(history, seed)
    rng = random.Random(seed + 1)
    rng.shuffle(pool)
    chosen, seen = [], set()
    for state, piece, source, perturbation in pool:
        key = (state.rows, piece)
        if key in seen:
            continue
        digest = hashlib.sha256(("generic" + piece + repr(state.rows)).encode()).hexdigest()[:12]
        chosen.append(TutorialSituation(f"generic-{digest}", "generic", state.rows, piece,
                                         source, perturbation))
        seen.add(key)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("insufficient generic candidates")
    return tuple(chosen)


def allocate_profile(profile: Mapping[str, Any], budget: int) -> dict[str, int]:
    """Allocate to supported top two by score using deterministic largest remainder."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    scores = profile["families"]
    ranked = [name for name in profile["ranking"] if name in SUPPORTED_FAMILIES][:2]
    if not ranked or budget == 0:
        return {name: 0 for name in ranked}
    values = [max(0.0, float(scores[name]["score"])) for name in ranked]
    total = sum(values)
    quotas = ([budget / len(ranked)] * len(ranked) if total == 0 else
              [budget * value / total for value in values])
    counts = [int(value) for value in quotas]
    for index in sorted(range(len(ranked)), key=lambda i: (-(quotas[i] - counts[i]), i))[:budget-sum(counts)]:
        counts[index] += 1
    return dict(zip(ranked, counts))


def select_personalized(profile: Mapping[str, Any], sets: Mapping[str, Sequence[TutorialSituation]],
                        budget: int) -> tuple[TutorialSituation, ...]:
    allocation = allocate_profile(profile, budget)
    selected = []
    for family in [name for name in profile["ranking"] if name in allocation]:
        options = sets[family]
        selected.extend(options[i % len(options)] for i in range(allocation[family]))
    return tuple(selected)


def train_on_situations(student: RicherRLStudent, situations: Sequence[TutorialSituation],
                        budget: int) -> dict[str, Any]:
    """One scenario reset per exposure; each uses the unchanged Day 4 RL calls/reward."""
    if budget < 0 or (budget and not situations):
        raise ValueError("a positive budget requires situations")
    game, before = TetrisAdapter(), student.serialize_state()
    rewards = []
    for index in range(budget):
        situation = situations[index % len(situations)]
        state = situation.state
        legal = game.legal_actions(state, situation.piece)
        if not legal:
            raise ValueError(f"unplayable tutorial: {situation.situation_id}")
        decision = student.choose_placement(state, situation.piece, legal, learn=True,
                                            deterministic=False)
        reward = gameplay_reward(decision.evaluation)
        student.update(AgentExperience(state, situation.piece, decision, reward,
                                       decision.evaluation.next_state, False))
        student.finish_episode(learned=True)
        rewards.append(reward)
    after = student.serialize_state()
    return {"requested_exposures": budget, "placements": budget,
            "updates_before": int(before["updates"]), "updates_after": int(after["updates"]),
            "updates_applied": int(after["updates"]) - int(before["updates"]),
            "episodes_applied": int(after["episodes_learned"]) - int(before["episodes_learned"]),
            "reward_source": "day4.gameplay_reward: 0.02 + simulator lines_cleared",
            "expert_calls": 0, "scenario_resets": budget,
            "situation_ids": [situations[i % len(situations)].situation_id for i in range(budget)],
            "rewards": rewards,
            "learning_path": "RicherRLStudent.choose_placement/update/finish_episode"}


def fingerprint_situations(situations: Sequence[TutorialSituation]) -> str:
    return hashlib.sha256(json.dumps([row.as_dict() for row in situations],
                                     sort_keys=True).encode()).hexdigest()
