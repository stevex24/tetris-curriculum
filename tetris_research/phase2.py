"""Preregistered comparison of a responsive full-profile Tetris tutor."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adaptive_tutor import (diagnose_full_history, profile_distance,
                             profile_fingerprint, select_block, value_candidates)
from .day3 import collect_states
from .day4 import evaluate_games, expert_diagnostics
from .day5 import collect_history
from .day7 import paired_summary, starting_student, train_imitation, train_ordinary
from .diagnosis import diagnose_history
from .richer_student import FEATURE_NAMES, RicherRLStudent
from .tetris import SHAPES, TetrisAdapter, TetrisState
from .tutorials import (TutorialSituation, candidate_pool, generate_tutorial_sets,
                        select_personalized, train_on_situations)

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "experiments/phase2"
CONDITIONS = ("ordinary", "imitation", "static_personalized", "responsive_personalized")
REFERENCE_NAMES = ("rl@0", "rl@250", "rl@750", "rl@1500", "rl@3000")


@dataclass(frozen=True)
class Phase2Config:
    name: str
    starting_checkpoint: str
    replicate_seeds: tuple[int, ...]
    training_budget: int
    block_size: int
    history_seeds: tuple[int, ...]
    history_maximum: int
    tutorial_generation_seed: int
    responsive_diagnostic_seeds: tuple[int, ...]
    responsive_diagnostic_maximum: int
    candidate_count: int
    shortlist: int
    rating_calibration_seeds: tuple[int, ...]
    rating_probe_seeds: tuple[int, ...]
    evaluation_game_seeds: tuple[int, ...]
    evaluation_maximum: int
    diagnostic_state_seeds: tuple[int, ...]
    diagnostic_states: int
    imitation_learning_rate: float
    bootstrap_seed: int
    bootstrap_samples: int


SMOKE = Phase2Config(
    "development-smoke", "rl@750", (920101, 920102), 40, 20,
    (920001, 920002), 30, 920050, (920061, 920062, 920063), 8, 8, 2,
    (920201, 920202, 920203), (920301,), (920401, 920402, 920403), 100,
    (920501,), 4, .08, 920601, 200)

FINAL = Phase2Config(
    "preregistered-final", "rl@750", tuple(range(2026100101, 2026100109)), 500, 50,
    tuple(range(2026100001, 2026100007)), 80, 2026100050,
    tuple(range(2026100061, 2026100072)), 20, 16, 4,
    tuple(range(2026100201, 2026100213)), (2026100301, 2026100302),
    tuple(range(2026101101, 2026101113)), 500,
    (2026101201, 2026101202), 16, .08, 2026101301, 10000)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalised_state(student: RicherRLStudent) -> dict[str, Any]:
    result = copy.deepcopy(dict(student.serialize_state()))
    result.pop("agent_id", None)
    return result


def validate_config(config: Phase2Config) -> None:
    if config.training_budget <= 0 or config.block_size <= 0 or \
            config.training_budget % config.block_size:
        raise ValueError("training budget must be a positive multiple of block size")
    blocks = config.training_budget // config.block_size
    if len(config.responsive_diagnostic_seeds) != blocks + 1:
        raise ValueError("responsive diagnosis requires one seed per boundary")
    domains = {
        "training": set(config.replicate_seeds),
        "history": set(config.history_seeds) | {config.tutorial_generation_seed},
        "responsive_diagnosis": set(config.responsive_diagnostic_seeds),
        "rating_calibration": set(config.rating_calibration_seeds),
        "rating_probe": set(config.rating_probe_seeds),
        "evaluation": set(config.evaluation_game_seeds) | set(config.diagnostic_state_seeds),
        "statistics": {config.bootstrap_seed},
    }
    names = list(domains)
    if any(domains[names[i]] & domains[names[j]] for i in range(len(names))
           for j in range(i + 1, len(names))):
        raise ValueError("Phase-2 seed domains must be disjoint")


def reference_students() -> dict[str, RicherRLStudent]:
    artifact = json.loads((ROOT / "experiments/day4/final_results.json").read_text())
    return {name: RicherRLStudent.from_state(copy.deepcopy(
        artifact["evaluation_before_states"][name])) for name in REFERENCE_NAMES}


def _outcomes(students: Mapping[str, RicherRLStudent], seeds: Sequence[int],
              maximum: int) -> dict[str, list[dict[str, int]]]:
    _, rows = evaluate_games(dict(students), seeds, maximum)
    return {name: [dict(row[name]) for row in rows] for name in students}


def _result(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    a = (int(left["lines_cleared"]), int(left["placements"]))
    b = (int(right["lines_cleared"]), int(right["placements"]))
    return 1.0 if a > b else 0.0 if a < b else 0.5


def _fit_reference_ratings(outcomes: Mapping[str, Sequence[Mapping[str, int]]]) -> dict[str, float]:
    names = list(REFERENCE_NAMES)
    ratings = {name: 0.0 for name in names}
    pairs = [(a, b, statistics.mean(_result(x, y) for x, y in zip(outcomes[a], outcomes[b])))
             for i, a in enumerate(names) for b in names[i + 1:]]
    # Batch Bradley-Terry/Elo logistic fit, centered each iteration.  The scale
    # is Elo-compatible but deliberately not presented as FIDE rating.
    for iteration in range(600):
        gradient = {name: 0.0 for name in names}
        for a, b, observed in pairs:
            expected = 1.0 / (1.0 + 10.0 ** ((ratings[b] - ratings[a]) / 400.0))
            gradient[a] += observed - expected
            gradient[b] -= observed - expected
        step = 18.0 / math.sqrt(iteration + 1.0)
        ratings = {name: ratings[name] + step * gradient[name] for name in names}
        center = statistics.mean(ratings.values())
        ratings = {name: value - center for name, value in ratings.items()}
    return ratings


def calibrate_ladder(seeds: Sequence[int], maximum: int, *, bootstrap_seed: int,
                     bootstrap_samples: int = 500) -> dict[str, Any]:
    outcomes = _outcomes(reference_students(), seeds, maximum)
    ratings = _fit_reference_ratings(outcomes)
    rng = random.Random(bootstrap_seed)
    boot = {name: [] for name in REFERENCE_NAMES}
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(seeds)) for _ in seeds]
        sample = {name: [outcomes[name][i] for i in indices] for name in REFERENCE_NAMES}
        fitted = _fit_reference_ratings(sample)
        for name in REFERENCE_NAMES:
            boot[name].append(fitted[name])
    intervals = {name: [sorted(boot[name])[int(.05 * (bootstrap_samples - 1))],
                        sorted(boot[name])[int(.95 * (bootstrap_samples - 1))]]
                 for name in REFERENCE_NAMES}
    order = sorted(REFERENCE_NAMES, key=lambda name: ratings[name])
    stable = all(intervals[order[i]][1] < intervals[order[i + 1]][0]
                 for i in range(len(order) - 1))
    return {"method": "matched-stream Bradley-Terry fit on line wins with placements as tie-breaker",
            "anchor": "reference mean constrained to zero; values are simulator ladder points, not FIDE Elo",
            "seeds": list(seeds), "games": len(seeds), "cap": maximum,
            "ratings": ratings, "bootstrap_90_ci": intervals, "ordering": order,
            "stable_ordering": stable,
            "ordering_note": "adjacent 90% intervals must not overlap for full stability",
            "outcomes": outcomes}


def performance_rating(candidate: Sequence[Mapping[str, int]],
                       reference_outcomes: Mapping[str, Sequence[Mapping[str, int]]],
                       reference_ratings: Mapping[str, float]) -> float:
    estimates = []
    for name in REFERENCE_NAMES:
        results = [_result(left, right) for left, right in zip(candidate, reference_outcomes[name])]
        score = sum(results)
        # Jeffreys smoothing prevents infinite ratings for finite matched samples.
        probability = (score + .5) / (len(results) + 1.0)
        estimates.append(float(reference_ratings[name]) + 400.0 * math.log10(
            probability / (1.0 - probability)))
    return statistics.mean(estimates)


def student_rating(student: RicherRLStudent, seeds: Sequence[int], maximum: int,
                   reference_outcomes: Mapping[str, Sequence[Mapping[str, int]]],
                   ratings: Mapping[str, float]) -> float:
    outcomes = _outcomes({"candidate": student}, seeds, maximum)["candidate"]
    return performance_rating(outcomes, reference_outcomes, ratings)


def full_profile_candidates(history: Sequence[Mapping[str, Any]], *, seed: int,
                            count: int) -> tuple[TutorialSituation, ...]:
    chosen, seen = [], set()
    for state, piece, source, perturbation in candidate_pool(history, seed):
        key = (state.rows, piece)
        if key in seen:
            continue
        digest = hashlib.sha256((piece + repr(state.rows)).encode()).hexdigest()[:12]
        chosen.append(TutorialSituation(f"full-profile-{digest}", "full_profile",
                                         state.rows, piece, source, perturbation))
        seen.add(key)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("insufficient full-profile tutorial candidates")
    return tuple(chosen)


def train_responsive(student: RicherRLStudent, candidates: Sequence[TutorialSituation],
                     config: Phase2Config, probe_reference: Mapping[str, Sequence[Mapping[str, int]]],
                     ratings: Mapping[str, float]) -> dict[str, Any]:
    before = _normalised_state(student)
    blocks, initial_profile, previous_profile = [], None, None
    previous_ids: tuple[str, ...] | None = None
    nonpositive: dict[str, int] = {}
    curriculum_changes = diagnostic_placements = oracle_calls = 0

    def probe(agent: RicherRLStudent) -> float:
        return student_rating(agent, config.rating_probe_seeds, 80,
                              probe_reference, ratings)

    for block_index in range(config.training_budget // config.block_size):
        frozen = _normalised_state(student)
        history = collect_history(student, (config.responsive_diagnostic_seeds[block_index],),
                                  config.responsive_diagnostic_maximum)
        if _normalised_state(student) != frozen:
            raise AssertionError("responsive diagnosis mutated student")
        profile = diagnose_full_history(history)
        diagnostic_placements += profile["decision_count"]
        oracle_calls += profile["decision_count"]
        if initial_profile is None:
            initial_profile = copy.deepcopy(profile)
        values = value_candidates(student, profile, initial_profile, candidates, probe, nonpositive)
        selected = select_block(values, config.block_size, config.shortlist)
        ids = tuple(dict.fromkeys(row.situation_id for row in selected))
        behavior_change = None if previous_profile is None else profile_distance(previous_profile, profile)
        if previous_ids is not None and ids != previous_ids:
            curriculum_changes += 1
            if not behavior_change:
                # Counterfactual policy behavior is also evidence, but an exact
                # unchanged profile cannot justify a silent curriculum switch.
                raise AssertionError("curriculum changed without profile evidence")
        audit = train_on_situations(student, selected, config.block_size)
        for value in values:
            if value.predicted_rating_gain <= 0:
                nonpositive[value.situation.situation_id] = nonpositive.get(
                    value.situation.situation_id, 0) + 1
            else:
                nonpositive[value.situation.situation_id] = 0
        blocks.append({"block": block_index, "profile": profile,
                       "profile_fingerprint": profile_fingerprint(profile),
                       "profile_distance_from_previous": behavior_change,
                       "selected_ids": list(ids),
                       "predicted_values": [{"id": row.situation.situation_id,
                           "predicted_rating_gain": row.predicted_rating_gain,
                           "profile_alignment": row.profile_alignment,
                           "mastery_factor": row.mastery_factor,
                           "retired": row.retired} for row in values],
                       "training": audit})
        previous_ids, previous_profile = ids, profile

    final_frozen = _normalised_state(student)
    final_history = collect_history(student, (config.responsive_diagnostic_seeds[-1],),
                                    config.responsive_diagnostic_maximum)
    if _normalised_state(student) != final_frozen:
        raise AssertionError("final responsive diagnosis mutated student")
    final_profile = diagnose_full_history(final_history)
    diagnostic_placements += final_profile["decision_count"]
    oracle_calls += final_profile["decision_count"]
    return {"updates_applied": student.updates - int(before["updates"]),
            "decision_states_seen": config.training_budget,
            "training_placements": config.training_budget,
            "tutorial_states_seen": config.training_budget,
            "ordinary_gameplay_states_seen": 0, "resets": config.training_budget,
            "expert_labels_seen": 0, "expert_calls_during_learning": 0,
            "diagnostic_placements": diagnostic_placements,
            "oracle_calls_for_nonlearning_diagnosis": oracle_calls,
            "curriculum_changes": curriculum_changes, "blocks": blocks,
            "initial_profile": initial_profile, "final_profile": final_profile,
            "learning_path": "RicherRLStudent.choose_placement/update/finish_episode",
            "reward_source": "day4.gameplay_reward: 0.02 + simulator lines_cleared"}


def run(config: Phase2Config) -> dict[str, Any]:
    validate_config(config)
    ladder = calibrate_ladder(config.rating_calibration_seeds, 300,
                              bootstrap_seed=config.bootstrap_seed + 100,
                              bootstrap_samples=100 if config.name == "development-smoke" else 500)
    refs = reference_students()
    evaluation_refs = _outcomes(refs, config.evaluation_game_seeds, config.evaluation_maximum)
    probe_refs = _outcomes(refs, config.rating_probe_seeds, 80)

    shared = starting_student(config.starting_checkpoint, config.replicate_seeds[0])
    shared_before = _normalised_state(shared)
    history = collect_history(shared, config.history_seeds, config.history_maximum)
    family_sets = generate_tutorial_sets(history, seed=config.tutorial_generation_seed,
                                          per_family=8)
    family_profile = diagnose_history(history)
    static_situations = select_personalized(family_profile, family_sets, config.training_budget)
    responsive_candidates = full_profile_candidates(history, seed=config.tutorial_generation_seed,
                                                     count=config.candidate_count)
    if _normalised_state(shared) != shared_before:
        raise AssertionError("shared material generation mutated student")

    diagnostic_states = collect_states(config.diagnostic_state_seeds, config.diagnostic_states)
    replicates, gains = [], []
    for replicate_id, seed in enumerate(config.replicate_seeds):
        initial = starting_student(config.starting_checkpoint, seed)
        students = {name: initial.clone(f"phase2-r{replicate_id}-{name}") for name in CONDITIONS}
        starts = {name: _normalised_state(student) for name, student in students.items()}
        if len({_canonical(value) for value in starts.values()}) != 1:
            raise AssertionError("condition starts differ")
        baseline = initial.clone(f"phase2-r{replicate_id}-baseline")
        audits = {
            "ordinary": train_ordinary(students["ordinary"], seed=seed, budget=config.training_budget),
            "imitation": train_imitation(students["imitation"], seed=seed,
                                          budget=config.training_budget,
                                          learning_rate=config.imitation_learning_rate),
            "static_personalized": train_on_situations(
                students["static_personalized"], static_situations, config.training_budget),
        }
        audits["static_personalized"].update({"decision_states_seen": config.training_budget,
            "training_placements": config.training_budget, "tutorial_states_seen": config.training_budget,
            "ordinary_gameplay_states_seen": 0, "resets": config.training_budget,
            "diagnostic_placements": 0, "expert_labels_seen": 0,
            "expert_calls_during_learning": 0})
        audits["responsive_personalized"] = train_responsive(
            students["responsive_personalized"], responsive_candidates, config, probe_refs,
            ladder["ratings"])
        if {audit["updates_applied"] for audit in audits.values()} != {config.training_budget}:
            raise AssertionError("unequal learning-update budgets")

        evaluated = {"baseline": baseline, **students}
        frozen_before = {name: _normalised_state(student) for name, student in evaluated.items()}
        metrics, game_rows = evaluate_games(evaluated, config.evaluation_game_seeds,
                                            config.evaluation_maximum)
        diagnostics = expert_diagnostics(evaluated, diagnostic_states)
        frozen_after = {name: _normalised_state(student) for name, student in evaluated.items()}
        if frozen_before != frozen_after:
            raise AssertionError("evaluation mutated students")
        ratings = {name: performance_rating([row[name] for row in game_rows], evaluation_refs,
                                             ladder["ratings"]) for name in evaluated}
        improvement = {name: ratings[name] - ratings["baseline"] for name in CONDITIONS}
        gains.append(improvement)
        replicates.append({"replicate_id": replicate_id, "training_seed": seed,
                           "initial_states": starts, "training_audits": audits,
                           "trained_states": {name: dict(student.serialize_state())
                                              for name, student in students.items()},
                           "metrics": metrics, "ratings": ratings,
                           "rating_gains": improvement, "secondary_expert_diagnostics": diagnostics,
                           "game_rows": game_rows,
                           "evaluation_before_sha256": hashlib.sha256(
                               _canonical(frozen_before).encode()).hexdigest(),
                           "evaluation_after_sha256": hashlib.sha256(
                               _canonical(frozen_after).encode()).hexdigest()})

    summaries = {name: {"mean_rating_gain": statistics.mean(row[name] for row in gains),
                        "median_rating_gain": statistics.median(row[name] for row in gains)}
                 for name in CONDITIONS}
    pairs = (("responsive_personalized", "ordinary"),
             ("responsive_personalized", "static_personalized"),
             ("responsive_personalized", "imitation"),
             ("static_personalized", "ordinary"), ("imitation", "ordinary"))
    contrasts = {f"{a}_minus_{b}": paired_summary(gains, a, b,
                 seed=config.bootstrap_seed + i, samples=config.bootstrap_samples)
                 for i, (a, b) in enumerate(pairs)}
    ro = contrasts["responsive_personalized_minus_ordinary"]
    rs = contrasts["responsive_personalized_minus_static_personalized"]
    ri = contrasts["responsive_personalized_minus_imitation"]
    beats = lambda row: row["mean_difference"] > 0 and row["bootstrap_90_ci"][0] > 0
    if ro["mean_difference"] < 0:
        classification = "negative"
    elif not beats(ro):
        classification = "null"
    elif beats(rs) and beats(ri):
        classification = "strong success"
    elif beats(rs):
        classification = "adaptive success"
    else:
        classification = "minimum success"
    return {"schema_version": 1, "experiment": "Phase-2 responsive full-profile tutor",
            "configuration": asdict(config), "conditions": list(CONDITIONS),
            "starting_checkpoint": config.starting_checkpoint,
            "rating_ladder": ladder,
            "primary_metric": "matched-stream simulator performance-rating gain from baseline per equal learner-update budget",
            "rating_formula": "Bradley-Terry/Elo logistic reference fit; Jeffreys-smoothed performance rating against fixed references",
            "secondary_metrics": ["lines", "placements", "expert agreement", "expert regret",
                                  "18-feature profile evolution", "curriculum switches", "feature exposure"],
            "shared_history_sha256": hashlib.sha256(_canonical(history).encode()).hexdigest(),
            "static_allocation": {name: sum(row.family == name for row in static_situations)
                                  for name in family_sets},
            "responsive_candidate_count": len(responsive_candidates),
            "replicates": replicates, "rating_gain_summary": summaries,
            "paired_contrasts": contrasts,
            "success_hierarchy": {"minimum": "responsive reliably beats ordinary",
                "adaptive": "responsive reliably beats ordinary and static personalized",
                "strong": "responsive reliably beats ordinary, static personalized, and imitation",
                "null": "responsive does not reliably beat ordinary and its mean is nonnegative",
                "negative": "responsive mean gain is below ordinary"},
            "classification": classification}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "final"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = SMOKE if args.mode == "smoke" else FINAL
    if args.mode == "final":
        locked = json.loads((PHASE2 / "preregistration.json").read_text())
        if locked.get("status") != "locked before final run" or \
                _canonical(locked["configuration"]) != _canonical(asdict(config)):
            raise RuntimeError("code and locked Phase-2 preregistration differ")
    result = run(config)
    output = args.output or PHASE2 / ("smoke_results.json" if args.mode == "smoke" else "final_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"classification": result["classification"],
                      "rating_gain_summary": result["rating_gain_summary"],
                      "paired_contrasts": result["paired_contrasts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
