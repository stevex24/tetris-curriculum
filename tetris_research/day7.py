"""Day 7 equal-budget comparison of practice, imitation, and tutorials."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .day3 import collect_states
from .day4 import evaluate_games, expert_diagnostics, gameplay_reward
from .day5 import collect_history
from .diagnosis import diagnose_history
from .expert import DELLACHERIE_WEIGHTS, DellacherieSearchExpert
from .richer_student import FEATURE_NAMES, RicherRLStudent
from .student import AgentExperience
from .tetris import SHAPES, TetrisAdapter, TetrisState
from .tutorials import (fingerprint_situations, generate_tutorial_sets,
                        select_personalized, train_on_situations)

ROOT = Path(__file__).resolve().parents[1]
DAY7 = ROOT / "experiments/day7"


@dataclass(frozen=True)
class Day7Config:
    name: str
    starting_checkpoint: str
    replicate_seeds: tuple[int, ...]
    training_budget: int
    history_seeds: tuple[int, ...]
    history_maximum: int
    tutorial_generation_seed: int
    evaluation_game_seeds: tuple[int, ...]
    evaluation_maximum: int
    diagnostic_state_seeds: tuple[int, ...]
    diagnostic_states: int
    imitation_learning_rate: float
    bootstrap_seed: int
    bootstrap_samples: int


SMOKE = Day7Config("development-smoke", "rl@750", (70101, 70102), 20,
                   (70001, 70002), 40, 70050, (71101, 71102), 100,
                   (71201,), 4, 0.08, 71301, 200)
FINAL = Day7Config("preregistered-final", "rl@750",
                   tuple(range(2026090101, 2026090109)), 500,
                   tuple(range(2026090001, 2026090007)), 80, 2026090050,
                   tuple(range(2026091101, 2026091113)), 500,
                   (2026091201, 2026091202), 16, 0.08, 2026091301, 10000)

CONDITIONS = ("ordinary", "imitation", "personalized")
PRIMARY = "mean_lines_cleared"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalized_state(student: RicherRLStudent) -> dict[str, Any]:
    state = copy.deepcopy(dict(student.serialize_state()))
    state.pop("agent_id", None)
    return state


def starting_student(checkpoint: str, replicate_seed: int) -> RicherRLStudent:
    """Load committed Day 4 parameters/state and assign a fresh paired RNG."""
    day4 = json.loads((ROOT / "experiments/day4/final_results.json").read_text())
    source = copy.deepcopy(day4["evaluation_before_states"][checkpoint])
    source["agent_id"] = f"day7-start-{replicate_seed}"
    source["seed"] = replicate_seed
    source["rng_state"] = repr(random.Random(replicate_seed).getstate())
    return RicherRLStudent.from_state(source)


def train_ordinary(student: RicherRLStudent, *, seed: int, budget: int) -> dict[str, Any]:
    game, rng, state = TetrisAdapter(), random.Random(seed), TetrisState()
    before, pieces, resets, trajectory = student.updates, [], 0, 0
    lengths: list[int] = []
    for _ in range(budget):
        piece = rng.choice(tuple(SHAPES)); pieces.append(piece)
        legal = game.legal_actions(state, piece)
        if not legal:
            student.finish_episode(learned=True)
            lengths.append(trajectory); trajectory = 0; resets += 1
            state, legal = TetrisState(), game.legal_actions(TetrisState(), piece)
        decision = student.choose_placement(state, piece, legal, learn=True, deterministic=False)
        reward = gameplay_reward(decision.evaluation)
        student.update(AgentExperience(state, piece, decision, reward,
                                       decision.evaluation.next_state, False))
        state = decision.evaluation.next_state
        trajectory += 1
    student.finish_episode(learned=True)
    lengths.append(trajectory)
    return {"updates_applied": student.updates - before, "decision_states_seen": budget,
            "pieces_consumed": budget, "resets": resets,
            "ordinary_gameplay_states_seen": budget, "tutorial_states_seen": 0,
            "expert_labels_seen": 0, "expert_calls_during_learning": 0,
            "trajectory_lengths": lengths,
            "piece_stream_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
            "learning_path": "RicherRLStudent.choose_placement/update/finish_episode",
            "reward_source": "day4.gameplay_reward"}


def train_imitation(student: RicherRLStudent, *, seed: int, budget: int,
                    learning_rate: float) -> dict[str, Any]:
    states = collect_states((seed,), budget)
    game, expert, losses = TetrisAdapter(), DellacherieSearchExpert(beam_width=1), []
    before = student.updates
    state_hashes = []
    for state, piece in states:
        legal = game.legal_actions(state, piece)
        label = expert.preferred_placement(state, piece, legal).placement
        losses.append(student.learn_from_label(state, piece, legal, label,
                                               learning_rate=learning_rate))
        state_hashes.append(hashlib.sha256((piece + repr(state.rows)).encode()).hexdigest())
    return {"updates_applied": student.updates - before, "decision_states_seen": budget,
            "pieces_consumed": budget, "resets": 0,
            "ordinary_gameplay_states_seen": 0, "tutorial_states_seen": 0,
            "expert_labels_seen": budget, "expert_calls_during_learning": budget,
            "mean_cross_entropy": statistics.mean(losses), "state_hashes": state_hashes,
            "learning_path": "RicherRLStudent.learn_from_label",
            "reward_source": None, "expert_parameter_copy": False,
            "imitation_learning_rate": learning_rate}


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def paired_summary(rows: Sequence[Mapping[str, float]], left: str, right: str,
                   *, seed: int, samples: int) -> dict[str, Any]:
    differences = [float(row[left]) - float(row[right]) for row in rows]
    rng = random.Random(seed)
    boot = [statistics.mean(differences[rng.randrange(len(differences))]
                            for _ in differences) for _ in range(samples)]
    return {"contrast": f"{left} - {right}", "replicate_differences": differences,
            "mean_difference": statistics.mean(differences),
            "median_difference": statistics.median(differences),
            "bootstrap_90_ci": [_percentile(boot, .05), _percentile(boot, .95)],
            "bootstrap_unit": "paired replicate", "bootstrap_samples": samples}


def _seed_domains(config: Day7Config) -> dict[str, set[int]]:
    return {"training": set(config.replicate_seeds), "history": set(config.history_seeds) |
            {config.tutorial_generation_seed}, "evaluation": set(config.evaluation_game_seeds) |
            set(config.diagnostic_state_seeds), "statistics": {config.bootstrap_seed}}


def validate_config(config: Day7Config) -> None:
    domains = _seed_domains(config)
    names = list(domains)
    if any(domains[names[i]] & domains[names[j]] for i in range(len(names))
           for j in range(i + 1, len(names))):
        raise ValueError("Day 7 seed domains must be disjoint")
    if len(set(config.replicate_seeds)) != len(config.replicate_seeds):
        raise ValueError("replicate identities must be unique")
    if config.training_budget <= 0 or config.bootstrap_samples <= 0:
        raise ValueError("budgets must be positive")


def run(config: Day7Config) -> dict[str, Any]:
    validate_config(config)
    shared = starting_student(config.starting_checkpoint, config.replicate_seeds[0])
    shared_before = _normalized_state(shared)
    history = collect_history(shared, config.history_seeds, config.history_maximum)
    profile = diagnose_history(history)
    tutorial_sets = generate_tutorial_sets(history, seed=config.tutorial_generation_seed,
                                            per_family=8 if config.training_budget >= 24 else 4)
    situations = select_personalized(profile, tutorial_sets, config.training_budget)
    if _normalized_state(shared) != shared_before:
        raise AssertionError("shared history or diagnosis mutated the starting student")

    diagnostic_states = collect_states(config.diagnostic_state_seeds, config.diagnostic_states)
    replicates, improvement_rows = [], []
    for replicate_id, seed in enumerate(config.replicate_seeds):
        initial = starting_student(config.starting_checkpoint, seed)
        students = {name: initial.clone(f"r{replicate_id}-{name}") for name in CONDITIONS}
        starts = {name: _normalized_state(student) for name, student in students.items()}
        if len({_canonical(value) for value in starts.values()}) != 1:
            raise AssertionError("condition starts differ within replicate")
        baseline = initial.clone(f"r{replicate_id}-baseline")
        audits = {
            "ordinary": train_ordinary(students["ordinary"], seed=seed,
                                       budget=config.training_budget),
            "imitation": train_imitation(students["imitation"], seed=seed,
                                         budget=config.training_budget,
                                         learning_rate=config.imitation_learning_rate),
            "personalized": train_on_situations(students["personalized"], situations,
                                                 config.training_budget),
        }
        audits["personalized"].update({"decision_states_seen": config.training_budget,
            "pieces_consumed": config.training_budget, "resets": config.training_budget,
            "ordinary_gameplay_states_seen": 0, "tutorial_states_seen": config.training_budget,
            "expert_labels_seen": 0, "expert_calls_during_learning": 0})
        if {audits[name]["updates_applied"] for name in CONDITIONS} != {config.training_budget}:
            raise AssertionError("unequal update counts")
        evaluated = {"baseline": baseline, **students}
        evaluation_before = {name: _normalized_state(student) for name, student in evaluated.items()}
        metrics, game_rows = evaluate_games(evaluated, config.evaluation_game_seeds,
                                            config.evaluation_maximum)
        diagnostics = expert_diagnostics(evaluated, diagnostic_states)
        evaluation_after = {name: _normalized_state(student) for name, student in evaluated.items()}
        if evaluation_before != evaluation_after:
            raise AssertionError("evaluation mutated a student")
        improvements = {name: metrics[name][PRIMARY] - metrics["baseline"][PRIMARY]
                        for name in CONDITIONS}
        improvement_rows.append(improvements)
        replicates.append({"replicate_id": replicate_id, "training_seed": seed,
                           "initial_states": starts, "training_audits": audits,
                           "trained_states": {name: dict(student.serialize_state())
                                              for name, student in students.items()},
                           "metrics": metrics, "improvements": improvements,
                           "expert_diagnostics": diagnostics, "game_rows": game_rows,
                           "diagnostic_event_order": ["all_frozen_student_decisions",
                                                      "external_expert_scoring"],
                           "evaluation_before_sha256": hashlib.sha256(
                               _canonical(evaluation_before).encode()).hexdigest(),
                           "evaluation_after_sha256": hashlib.sha256(
                               _canonical(evaluation_after).encode()).hexdigest()})

    summaries = {name: {"mean_improvement": statistics.mean(row[name] for row in improvement_rows),
                        "median_improvement": statistics.median(row[name] for row in improvement_rows)}
                 for name in CONDITIONS}
    contrasts = {
        "personalized_minus_ordinary": paired_summary(
            improvement_rows, "personalized", "ordinary", seed=config.bootstrap_seed,
            samples=config.bootstrap_samples),
        "personalized_minus_imitation": paired_summary(
            improvement_rows, "personalized", "imitation", seed=config.bootstrap_seed + 1,
            samples=config.bootstrap_samples),
        "imitation_minus_ordinary": paired_summary(
            improvement_rows, "imitation", "ordinary", seed=config.bootstrap_seed + 2,
            samples=config.bootstrap_samples),
    }
    po, pi = contrasts["personalized_minus_ordinary"], contrasts["personalized_minus_imitation"]
    minimum = po["mean_difference"] >= 2.0 and po["bootstrap_90_ci"][0] > 0.0
    strong = minimum and pi["mean_difference"] >= 2.0 and pi["bootstrap_90_ci"][0] > 0.0
    classification = ("strong success" if strong else "minimum success" if minimum else
                      "negative" if po["mean_difference"] < 0 else "null")
    caps = {name: sum(row[name]["placements"] == config.evaluation_maximum
                      for replicate in replicates for row in replicate["game_rows"])
            for name in ("baseline", *CONDITIONS)}
    return {"schema_version": 1, "experiment": "Day 7 equal-budget comparison",
            "configuration": asdict(config), "conditions": list(CONDITIONS),
            "starting_checkpoint_provenance": "experiments/day4/final_results.json evaluation_before_states rl@750",
            "starting_checkpoint_reason": "partially competent, substantially above zero policy and below rl@3000",
            "history_design": "Option A: one shared deterministic nonlearning history before cloning; zero learner updates",
            "shared_history": {"game_count": len(history),
                               "decision_count": sum(len(row["steps"]) for row in history),
                               "sha256": hashlib.sha256(_canonical(history).encode()).hexdigest(),
                               "student_nonmutating": True},
            "weakness_profile": profile,
            "tutorial_allocation": {name: sum(row.family == name for row in situations)
                                    for name in tutorial_sets},
            "tutorial_fingerprint": fingerprint_situations(situations),
            "primary_metric": "change in matched held-out mean lines cleared from frozen baseline",
            "secondary_metrics": ["mean placements survived", "expert agreement",
                                  "mean expert-relative regret"],
            "replicates": replicates, "improvement_summary": summaries,
            "paired_contrasts": contrasts, "cap_hits": caps,
            "success_rule": {"minimum": "personalized-ordinary mean >= 2.0 lines and paired-bootstrap 90% CI lower bound > 0",
                             "strong": "minimum plus personalized-imitation meeting the same rule"},
            "classification": classification, "ratings": None,
            "expert_coefficients_sha256": hashlib.sha256(_canonical(DELLACHERIE_WEIGHTS).encode()).hexdigest(),
            "expert_coefficients_copied": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "final"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = SMOKE if args.mode == "smoke" else FINAL
    if args.mode == "final":
        locked = json.loads((DAY7 / "preregistration.json").read_text())
        if (_canonical(locked["configuration"]) != _canonical(asdict(config)) or
                locked["status"] != "locked before final run"):
            raise RuntimeError("code and locked Day 7 preregistration differ")
    result = run(config)
    output = args.output or DAY7 / ("smoke_results.json" if args.mode == "smoke" else "final_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"classification": result["classification"],
                      "improvement_summary": result["improvement_summary"],
                      "paired_contrasts": result["paired_contrasts"],
                      "cap_hits": result["cap_hits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
