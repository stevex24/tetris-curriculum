"""Predeclared Day 4 sequential reinforcement-learning experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .day3 import collect_states
from .expert import DellacherieSearchExpert
from .richer_student import FEATURE_NAMES, RicherRLStudent
from .student import AgentExperience
from .tetris import SHAPES, TetrisAdapter, TetrisState

ROOT = Path(__file__).resolve().parents[1]
DAY4 = ROOT / "experiments/day4"


@dataclass(frozen=True)
class Day4Config:
    name: str
    checkpoints: tuple[int, ...]
    evaluation_games: int
    evaluation_maximum: int
    diagnostic_states: int
    training_seed: int
    evaluation_game_seeds: tuple[int, ...]
    evaluation_state_seeds: tuple[int, ...]
    learning_rate: float = 0.008
    discount: float = 0.99
    temperature: float = 0.35


SMOKE = Day4Config("development-smoke", (0, 25, 75), 3, 80, 4, 42001,
                   (43001, 43002, 43003), (44001,))
FINAL = Day4Config("final-held-out", (0, 250, 750, 1500, 3000), 12, 300, 16,
                   2026082701, tuple(range(2026082801, 2026082813)),
                   (2026082901, 2026082902))


def gameplay_reward(choice: Any) -> float:
    """Environment outcome only: survival placement plus rows actually cleared."""
    return 0.02 + float(choice.info["lines_cleared"])


def train_exposure(student: RicherRLStudent, control: RicherRLStudent,
                   config: Day4Config) -> tuple[dict[int, RicherRLStudent], dict[str, Any]]:
    """Give both conditions the same count and indexed uniform piece stream."""
    game, rng = TetrisAdapter(), random.Random(config.training_seed)
    states = {"rl": TetrisState(), "ordinary_play_control": TetrisState()}
    agents = {"rl": student, "ordinary_play_control": control}
    episodes = {name: 0 for name in agents}
    pieces = [rng.choice(tuple(SHAPES)) for _ in range(config.checkpoints[-1])]
    snapshots = {0: student.clone("rl@0")}
    episode_lengths: list[int] = []
    current_rl_length = 0
    for exposure, piece in enumerate(pieces, 1):
        for name, agent in agents.items():
            legal = game.legal_actions(states[name], piece)
            if not legal:
                if name == "rl":
                    if current_rl_length:
                        agent.finish_episode(learned=True)
                        episode_lengths.append(current_rl_length)
                    current_rl_length = 0
                episodes[name] += 1
                states[name] = TetrisState()
                legal = game.legal_actions(states[name], piece)
            learning = name == "rl"
            decision = agent.choose_placement(states[name], piece, legal,
                                              learn=learning, deterministic=False)
            if learning:
                reward = gameplay_reward(decision.evaluation)
                agent.update(AgentExperience(states[name], piece, decision, reward,
                                             decision.evaluation.next_state, False))
                current_rl_length += 1
            states[name] = decision.evaluation.next_state
        if exposure in config.checkpoints[1:]:
            # Checkpoints are declared trajectory boundaries, so all preceding
            # consequences contribute to a return before frozen evaluation.
            student.finish_episode(learned=True)
            episode_lengths.append(current_rl_length)
            current_rl_length = 0
            snapshots[exposure] = student.clone(f"rl@{exposure}")
    audit = {
        "opportunities": {name: len(pieces) for name in agents},
        "piece_stream_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
        "training_seed": config.training_seed, "episodes": episodes,
        "rl_trajectory_lengths": episode_lengths,
        "rl_updates": student.updates, "control_updates": control.updates,
        "expert_calls": 0,
        "reward_source": "0.02 per legal placement + 1.0 per simulator-cleared row",
    }
    return snapshots, audit


def evaluate_games(students: dict[str, RicherRLStudent], seeds: Sequence[int],
                   maximum: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    game, rows = TetrisAdapter(), []
    before = {name: dict(student.serialize_state()) for name, student in students.items()}
    for seed in seeds:
        rng = random.Random(seed)
        stream = [rng.choice(tuple(SHAPES)) for _ in range(maximum)]
        row: dict[str, Any] = {"seed": seed,
                              "stream_sha256": hashlib.sha256("".join(stream).encode()).hexdigest()}
        for name, source in students.items():
            state, lines, placements = TetrisState(), 0, 0
            for piece in stream:
                legal = game.legal_actions(state, piece)
                if not legal:
                    break
                decision = source.choose_placement(state, piece, legal,
                                                   learn=False, deterministic=True)
                state = decision.evaluation.next_state
                lines += int(decision.evaluation.info["lines_cleared"])
                placements += 1
            row[name] = {"placements": placements, "lines_cleared": lines}
        rows.append(row)
    after = {name: dict(student.serialize_state()) for name, student in students.items()}
    if before != after:
        raise AssertionError("held-out evaluation mutated a student")
    metrics = {name: {
        "mean_placements": statistics.mean(row[name]["placements"] for row in rows),
        "mean_lines_cleared": statistics.mean(row[name]["lines_cleared"] for row in rows),
    } for name in students}
    return metrics, rows


def expert_diagnostics(students: dict[str, RicherRLStudent], states: Sequence[tuple[TetrisState, str]]) -> dict[str, Any]:
    game, expert = TetrisAdapter(), DellacherieSearchExpert(beam_width=1)
    totals = {name: [0, 0.0] for name in students}
    for state, piece in states:
        legal = game.legal_actions(state, piece)
        # Every student decision is final before the external expert is created/called for scoring.
        decisions = {name: student.choose_placement(state, piece, legal,
                                                    learn=False, deterministic=True)
                     for name, student in students.items()}
        ranking = expert.rank_placements(state, piece, legal)
        values = {item.placement: item.value for item in ranking}
        for name, decision in decisions.items():
            totals[name][0] += decision.placement == ranking[0].placement
            totals[name][1] += ranking[0].value - values[decision.placement]
    return {name: {"expert_agreement": agreement / len(states),
                   "mean_expert_regret": regret / len(states)}
            for name, (agreement, regret) in totals.items()}


def run(config: Day4Config) -> dict[str, Any]:
    if config.checkpoints[0] != 0 or tuple(sorted(set(config.checkpoints))) != config.checkpoints:
        raise ValueError("checkpoints must be unique, increasing, and begin at zero")
    seed_domains = ({config.training_seed}, set(config.evaluation_game_seeds),
                    set(config.evaluation_state_seeds))
    if any(seed_domains[i] & seed_domains[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("training and evaluation seed domains overlap")
    initial = RicherRLStudent("initial", learning_rate=config.learning_rate,
                              discount=config.discount, temperature=config.temperature,
                              seed=config.training_seed)
    rl, control = initial.clone("rl"), initial.clone("ordinary_play_control")
    snapshots, training = train_exposure(rl, control, config)
    students = {f"rl@{checkpoint}": snapshots[checkpoint] for checkpoint in config.checkpoints}
    students["ordinary_play_control"] = control
    before = {name: dict(agent.serialize_state()) for name, agent in students.items()}
    metrics, game_rows = evaluate_games(students, config.evaluation_game_seeds,
                                        config.evaluation_maximum)
    after = {name: dict(agent.serialize_state()) for name, agent in students.items()}
    diagnostic_states = collect_states(config.evaluation_state_seeds, config.diagnostic_states)
    diagnostic_students = {"initial": snapshots[0], "final_rl": snapshots[config.checkpoints[-1]],
                           "ordinary_play_control": control}
    diagnostics = expert_diagnostics(diagnostic_students, diagnostic_states)
    first, final = metrics["rl@0"], metrics[f"rl@{config.checkpoints[-1]}"]
    matched = metrics["ordinary_play_control"]
    checks = {
        "placements_gain_at_least_25pct_and_10": final["mean_placements"] >=
            max(first["mean_placements"] * 1.25, first["mean_placements"] + 10),
        "lines_gain_at_least_25pct": final["mean_lines_cleared"] >=
            first["mean_lines_cleared"] * 1.25,
        "final_exceeds_matched_control": final["mean_placements"] > matched["mean_placements"]
            and final["mean_lines_cleared"] > matched["mean_lines_cleared"],
    }
    return {
        "schema_version": 1, "experiment": "Day 4 sequential reinforcement learning",
        "configuration": asdict(config), "feature_names": list(FEATURE_NAMES),
        "trainable_parameters": len(FEATURE_NAMES),
        "algorithm": "episodic linear-softmax REINFORCE with a causal running return baseline",
        "credit_assignment": "G_t = r_t + 0.99 G_(t+1); each action receives clipped (G_t - mean return from earlier trajectories) times grad log pi(a_t|s_t)",
        "reward": "0.02 per survived legal placement plus 1.0 per row actually cleared",
        "initial_state": dict(initial.serialize_state()), "training_audit": training,
        "evaluation_before_states": before, "evaluation_after_states": after,
        "evaluation_state_hashes": [hashlib.sha256((piece + repr(state.rows)).encode()).hexdigest()
                                    for state, piece in diagnostic_states],
        "learning_curve": {str(checkpoint): metrics[f"rl@{checkpoint}"]
                           for checkpoint in config.checkpoints},
        "matched_control": metrics["ordinary_play_control"], "game_rows": game_rows,
        "expert_diagnostics": diagnostics, "success_checks": checks,
        "rl_success": all(checks.values()), "ratings": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "final"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = SMOKE if args.mode == "smoke" else FINAL
    output = args.output or DAY4 / ("smoke.json" if args.mode == "smoke" else "final_results.json")
    print("PREDECLARATION")
    print(json.dumps({"configuration": asdict(config),
                      "success_criterion": "all three checks in run() must pass"}, indent=2))
    result = run(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT")
    print(json.dumps({"learning_curve": result["learning_curve"],
                      "matched_control": result["matched_control"],
                      "expert_diagnostics": result["expert_diagnostics"],
                      "success_checks": result["success_checks"],
                      "rl_success": result["rl_success"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
