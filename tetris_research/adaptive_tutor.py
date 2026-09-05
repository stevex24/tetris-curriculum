"""Responsive full-profile tutorial selection for the Phase-2 experiment."""
from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .diagnosis import BENEFICIAL_FEATURES
from .expert import DellacherieSearchExpert
from .richer_student import FEATURE_NAMES, RicherRLStudent, richer_features
from .student import Placement
from .tetris import TetrisAdapter, TetrisState
from .tutorials import TutorialSituation, train_on_situations


@dataclass(frozen=True)
class CandidateValue:
    situation: TutorialSituation
    predicted_rating_gain: float
    profile_alignment: float
    mastery_factor: float
    retired: bool

    @property
    def ordering(self) -> tuple[float, float, str]:
        return (self.predicted_rating_gain * self.mastery_factor,
                self.profile_alignment * self.mastery_factor,
                self.situation.situation_id)


def _normalise(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(values.get(name, 0.0))) for name in FEATURE_NAMES)
    if total <= 0:
        return {name: 1.0 / len(FEATURE_NAMES) for name in FEATURE_NAMES}
    return {name: max(0.0, float(values.get(name, 0.0))) / total for name in FEATURE_NAMES}


def diagnose_full_history(records: Sequence[Mapping[str, Any]], *,
                          expert: DellacherieSearchExpert | None = None,
                          regret_threshold: float = 1.0) -> dict[str, Any]:
    """Attribute observable expert-relative errors to all 18 features.

    The student's action is already fixed in the supplied history.  No policy
    parameters are accepted or inspected.  The oracle is used only afterward
    to attribute costly behavioral differences.
    """
    game = TetrisAdapter()
    oracle = expert or DellacherieSearchExpert(beam_width=1)
    totals = {name: 0.0 for name in FEATURE_NAMES}
    counts = {name: 0 for name in FEATURE_NAMES}
    decisions = material = 0
    for record in records:
        for step in record["steps"]:
            decisions += 1
            state = TetrisState(tuple(int(row) for row in step["board_before"]))
            piece = str(step["piece"])
            legal = game.legal_actions(state, piece)
            chosen = Placement(*map(int, step["student_action"]))
            selected = next(item for item in legal if tuple(item.action) ==
                            (chosen.rotation, chosen.x))
            observed = richer_features(state, piece, selected)
            ranking = oracle.rank_placements(state, piece, legal)
            values = {item.placement: item.value for item in ranking}
            regret = max(0.0, ranking[0].value - values[chosen])
            if regret < regret_threshold:
                continue
            material += 1
            best_choice = next(item for item in legal if tuple(item.action) ==
                               (ranking[0].placement.rotation, ranking[0].placement.x))
            best = richer_features(state, piece, best_choice)
            raw = {}
            for name, actual, preferred in zip(FEATURE_NAMES, observed, best):
                harmful = preferred - actual if name in BENEFICIAL_FEATURES else actual - preferred
                raw[name] = max(0.0, harmful)
            scale = sum(raw.values())
            if scale <= 0:
                continue
            for name in FEATURE_NAMES:
                contribution = regret * raw[name] / scale
                totals[name] += contribution
                counts[name] += contribution > 0
    severity = {name: totals[name] / decisions if decisions else 0.0 for name in FEATURE_NAMES}
    normalised = _normalise(severity)
    return {
        "feature_names": list(FEATURE_NAMES), "dimensions": len(FEATURE_NAMES),
        "decision_count": decisions, "material_regret_decisions": material,
        "severity": severity, "normalised_severity": normalised,
        "attributed_decisions": counts,
        "ranking": sorted(FEATURE_NAMES, key=lambda name: (-severity[name], name)),
        "source": "nonlearning recorded behavior; oracle queried after student decisions",
    }


def situation_signature(situation: TutorialSituation) -> dict[str, float]:
    """Return the action-sensitive range for every richer feature."""
    legal = TetrisAdapter().legal_actions(situation.state, situation.piece)
    vectors = [richer_features(situation.state, situation.piece, choice) for choice in legal]
    if len(vectors) < 2:
        return {name: 0.0 for name in FEATURE_NAMES}
    return {name: max(row[i] for row in vectors) - min(row[i] for row in vectors)
            for i, name in enumerate(FEATURE_NAMES)}


def profile_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return sum(abs(float(left["normalised_severity"][name]) -
                   float(right["normalised_severity"][name])) for name in FEATURE_NAMES)


def value_candidates(student: RicherRLStudent, profile: Mapping[str, Any],
                     initial_profile: Mapping[str, Any],
                     candidates: Sequence[TutorialSituation],
                     rating_probe: Callable[[RicherRLStudent], float],
                     nonpositive_counts: Mapping[str, int]) -> list[CandidateValue]:
    """Estimate one standard RL update's rating gain on cloned students.

    Rating gain is the primary ordering key.  Full-profile alignment is a
    deterministic tie-breaker.  No candidate can edit the real student.
    """
    before = rating_probe(student)
    current = profile["normalised_severity"]
    initial = initial_profile["normalised_severity"]
    valued = []
    for situation in candidates:
        signature = situation_signature(situation)
        alignment = sum(float(current[name]) * signature[name] for name in FEATURE_NAMES)
        exposed = sorted(FEATURE_NAMES, key=lambda name: (-signature[name], name))[:3]
        ratios = [float(current[name]) / max(float(initial[name]), 1e-12) for name in exposed]
        mastery = min(1.0, statistics.mean(ratios))
        clone = student.clone(f"counterfactual-{situation.situation_id}")
        train_on_situations(clone, (situation,), 1)
        gain = rating_probe(clone) - before
        retired = (all(ratio <= 0.70 for ratio in ratios) and gain <= 0.0) or \
                  int(nonpositive_counts.get(situation.situation_id, 0)) >= 2
        valued.append(CandidateValue(situation, gain, alignment, mastery, retired))
    return valued


def select_block(values: Sequence[CandidateValue], block_size: int,
                 shortlist: int = 4) -> tuple[TutorialSituation, ...]:
    available = [row for row in values if not row.retired]
    if not available:
        available = list(values)
    chosen = sorted(available, key=lambda row: (-row.ordering[0], -row.ordering[1],
                                                 row.ordering[2]))[:shortlist]
    return tuple(chosen[index % len(chosen)].situation for index in range(block_size))


def profile_fingerprint(profile: Mapping[str, Any]) -> str:
    payload = ",".join(f"{name}:{float(profile['severity'][name]):.12g}" for name in FEATURE_NAMES)
    return hashlib.sha256(payload.encode()).hexdigest()
