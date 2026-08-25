"""Predeclared Day 3 direct-imitation transfer experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .expert import DELLACHERIE_WEIGHTS, DellacherieSearchExpert
from .imitation_student import LinearImitationStudent
from .student import Placement
from .tetris import SHAPES, TetrisAdapter, TetrisState

ROOT = Path(__file__).resolve().parents[1]
DAY3 = ROOT / "experiments/day3"


@dataclass(frozen=True)
class Day3Config:
    name: str
    training_budget: int
    held_out_states: int
    evaluation_games: int
    maximum_placements: int
    expert_beam_width: int
    training_seeds: tuple[int, ...]
    evaluation_state_seeds: tuple[int, ...]
    evaluation_game_seeds: tuple[int, ...]
    learning_rate: float = 0.35


SMOKE = Day3Config("development-smoke", 8, 5, 2, 40, 1, (31001,), (41001,), (51001, 51002))
FINAL = Day3Config("final-held-out", 80, 24, 10, 300, 3,
                   (2026082401, 2026082402, 2026082403, 2026082404),
                   (2026082501, 2026082502, 2026082503),
                   tuple(range(2026082601, 2026082611)))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _state_hash(state: TetrisState, piece: str) -> str:
    return hashlib.sha256(_canonical([piece, list(state.rows)]).encode()).hexdigest()


def collect_states(seeds: Sequence[int], count: int, *,
                   exclude_hashes: set[str] | None = None) -> list[tuple[TetrisState, str]]:
    """Collect policy-independent states using uniform pieces and legal actions."""
    game, result = TetrisAdapter(), []
    for seed in seeds:
        rng, state = random.Random(seed), TetrisState()
        while len(result) < count:
            piece = rng.choice(tuple(SHAPES))
            legal = game.legal_actions(state, piece)
            if not legal:
                state = TetrisState()
                continue
            if exclude_hashes is None or _state_hash(state, piece) not in exclude_hashes:
                result.append((state, piece))
            # State generation is independent of all experimental students and labels.
            state = legal[rng.randrange(len(legal))].next_state
        if len(result) >= count:
            break
    return result


def _placement(choice: Any) -> Placement:
    return Placement(int(choice.action[0]), int(choice.action[1]))


def train_matched(initial: LinearImitationStudent,
                  states: Sequence[tuple[TetrisState, str]], expert: Any,
                  control_seed: int = 2026082499) -> tuple[dict[str, LinearImitationStudent], dict[str, Any]]:
    students = {"taught": initial.clone("taught"), "random_label_control": initial.clone("random_label_control")}
    rng, game = random.Random(control_seed), TetrisAdapter()
    losses = {name: [] for name in students}
    state_hashes, labels = [], []
    for state, piece in states:
        legal = game.legal_actions(state, piece)
        expert_label = expert.preferred_placement(state, piece, legal).placement
        control_label = _placement(legal[rng.randrange(len(legal))])
        losses["taught"].append(students["taught"].learn_from_label(state, piece, legal, expert_label))
        losses["random_label_control"].append(students["random_label_control"].learn_from_label(
            state, piece, legal, control_label))
        state_hashes.append(_state_hash(state, piece))
        labels.append({"expert": asdict(expert_label), "control": asdict(control_label)})
    audit = {"opportunities": {name: len(states) for name in students},
             "updates": {name: student.updates for name, student in students.items()},
             "state_hashes": state_hashes, "labels": labels,
             "mean_training_loss": {name: statistics.mean(values) for name, values in losses.items()}}
    return students, audit


def evaluate_decisions(students: dict[str, LinearImitationStudent],
                       states: Sequence[tuple[TetrisState, str]], expert: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    game = TetrisAdapter()
    totals = {name: {"agreement": 0, "regret": 0.0} for name in students}
    trace = []
    for state, piece in states:
        legal = game.legal_actions(state, piece)
        # Student decisions are completed first. Only the external evaluator then invokes the expert.
        decisions = {name: student.choose_placement(state, piece, legal, learn=False, deterministic=True)
                     for name, student in students.items()}
        ranking = expert.rank_placements(state, piece, legal)
        values = {item.placement: item.value for item in ranking}
        preferred = ranking[0].placement
        row = {"state_hash": _state_hash(state, piece), "piece": piece,
               "event_order": [*[f"student_choose:{name}" for name in students], "external_expert_score"],
               "expert_preferred": asdict(preferred), "decisions": {}}
        for name, decision in decisions.items():
            regret = ranking[0].value - values[decision.placement]
            totals[name]["agreement"] += decision.placement == preferred
            totals[name]["regret"] += regret
            row["decisions"][name] = {"placement": asdict(decision.placement), "regret": regret}
        trace.append(row)
    metrics = {name: {"held_out_agreement": values["agreement"] / len(states),
                      "held_out_mean_regret": values["regret"] / len(states)}
               for name, values in totals.items()}
    return metrics, trace


def evaluate_games(students: dict[str, LinearImitationStudent], seeds: Sequence[int],
                   maximum: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    game, rows = TetrisAdapter(), []
    for seed in seeds:
        rng = random.Random(seed)
        stream = [rng.choice(tuple(SHAPES)) for _ in range(maximum)]
        result = {"seed": seed, "stream_sha256": hashlib.sha256("".join(stream).encode()).hexdigest()}
        for name, source in students.items():
            agent, state, lines, placements = source.clone(), TetrisState(), 0, 0
            before = dict(agent.serialize_state())
            for piece in stream:
                legal = game.legal_actions(state, piece)
                if not legal:
                    break
                decision = agent.choose_placement(state, piece, legal, learn=False, deterministic=True)
                state = decision.evaluation.next_state
                lines += int(decision.evaluation.info["lines_cleared"])
                placements += 1
            if before != agent.serialize_state():
                raise AssertionError("evaluation mutated a student")
            result[name] = {"placements": placements, "lines_cleared": lines}
        rows.append(result)
    metrics = {}
    for name in students:
        metrics[name] = {"mean_placements": statistics.mean(row[name]["placements"] for row in rows),
                         "mean_lines_cleared": statistics.mean(row[name]["lines_cleared"] for row in rows)}
    return metrics, rows


def success(metrics: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    initial, taught, control = (metrics[name] for name in ("initial", "taught", "random_label_control"))
    checks = {
        "agreement_gain_vs_initial_10pp": taught["held_out_agreement"] >= initial["held_out_agreement"] + .10,
        "agreement_gain_vs_control_10pp": taught["held_out_agreement"] >= control["held_out_agreement"] + .10,
        "regret_reduction_vs_initial_10pct": taught["held_out_mean_regret"] <= initial["held_out_mean_regret"] * .90,
        "regret_reduction_vs_control_10pct": taught["held_out_mean_regret"] <= control["held_out_mean_regret"] * .90,
        "gameplay_better_than_initial_and_control": (
            taught["mean_placements"] > max(initial["mean_placements"], control["mean_placements"])
            or taught["mean_lines_cleared"] > max(initial["mean_lines_cleared"], control["mean_lines_cleared"])),
    }
    return all(checks.values()), checks


def run(config: Day3Config) -> dict[str, Any]:
    if set(config.training_seeds) & (set(config.evaluation_state_seeds) | set(config.evaluation_game_seeds)):
        raise ValueError("training and evaluation seeds overlap")
    expert = DellacherieSearchExpert(beam_width=config.expert_beam_width)
    initial = LinearImitationStudent("initial", learning_rate=config.learning_rate)
    initial_state = dict(initial.serialize_state())
    initial_clones = {name: initial.clone(name) for name in ("taught", "random_label_control")}
    probe_state, probe_piece = TetrisState(), "T"
    probe_legal = TetrisAdapter().legal_actions(probe_state, probe_piece)
    initial_probe_decisions = {name: asdict(agent.choose_placement(
        probe_state, probe_piece, probe_legal, learn=False, deterministic=True).placement)
        for name, agent in initial_clones.items()}
    training_states = collect_states(config.training_seeds, config.training_budget)
    trained, training_audit = train_matched(initial, training_states, expert)
    evaluation_states = collect_states(config.evaluation_state_seeds, config.held_out_states,
                                       exclude_hashes=set(training_audit["state_hashes"]))
    all_students = {"initial": initial.clone("initial"), **trained}
    before_evaluation = {name: dict(agent.serialize_state()) for name, agent in all_students.items()}
    decision_metrics, decision_trace = evaluate_decisions(all_students, evaluation_states, expert)
    game_metrics, game_rows = evaluate_games(all_students, config.evaluation_game_seeds,
                                             config.maximum_placements)
    after_evaluation = {name: dict(agent.serialize_state()) for name, agent in all_students.items()}
    metrics = {name: {**decision_metrics[name], **game_metrics[name]} for name in all_students}
    passed, checks = success(metrics)
    return {
        "schema_version": 1, "experiment": "Day 3 direct expert imitation baseline",
        "configuration": asdict(config),
        "predeclaration": {
            "conditions": ["initial", "taught", "random_label_control"],
            "learning_rule": "one multiclass action-softmax cross-entropy SGD update per state",
            "control": "same states and updates; uniformly random legal labels without expert information",
            "success_criterion": "all five predeclared checks must pass",
            "expert_disconnection": "students contain no expert reference; choose calls complete before external scoring; learn=False",
        },
        "initial_state": initial_state,
        "initial_clone_states": {name: dict(agent.serialize_state()) for name, agent in initial_clones.items()},
        "initial_probe_decisions": initial_probe_decisions,
        "trained_states": {name: dict(agent.serialize_state()) for name, agent in trained.items()},
        "expert_parameter_names": sorted(DELLACHERIE_WEIGHTS),
        "expert_parameter_values_sha256": hashlib.sha256(_canonical(DELLACHERIE_WEIGHTS).encode()).hexdigest(),
        "training_audit": training_audit,
        "evaluation_state_hashes": [_state_hash(state, piece) for state, piece in evaluation_states],
        "evaluation_before_states": before_evaluation, "evaluation_after_states": after_evaluation,
        "decision_trace": decision_trace, "game_rows": game_rows,
        "metrics": metrics, "success_checks": checks, "knowledge_transfer_success": passed,
        "ratings": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "final"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = SMOKE if args.mode == "smoke" else FINAL
    output = args.output or (DAY3 / ("smoke.json" if args.mode == "smoke" else "final_results.json"))
    declaration = {"configuration": asdict(config), "success_criterion": "all five checks in success() must pass"}
    print("PREDECLARATION")
    print(json.dumps(declaration, indent=2))
    result = run(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT")
    print(json.dumps({"metrics": result["metrics"], "success_checks": result["success_checks"],
                      "knowledge_transfer_success": result["knowledge_transfer_success"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
