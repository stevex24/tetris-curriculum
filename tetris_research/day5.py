"""Day 5 history collection, controlled profiles, and command-line experiment."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diagnosis import FEATURE_FAMILIES, diagnose_history
from .richer_student import FEATURE_NAMES, RicherRLStudent, richer_features
from .student import StudentAgent
from .tetris import SHAPES, TetrisAdapter, TetrisState

ROOT = Path(__file__).resolve().parents[1]
DAY5 = ROOT / "experiments/day5"
SEEDS = tuple(range(2026083001, 2026083007))


def collect_history(student: StudentAgent, seeds: Sequence[int], maximum: int = 80) -> list[dict[str, Any]]:
    """Record decisions first; no oracle exists in this collection path."""
    game, records = TetrisAdapter(), []
    for game_number, seed in enumerate(seeds):
        rng, state, steps, lines = random.Random(seed), TetrisState(), [], 0
        for index in range(maximum):
            piece = rng.choice(tuple(SHAPES))
            legal = game.legal_actions(state, piece)
            if not legal:
                break
            decision = student.choose_placement(state, piece, legal, learn=False, deterministic=True)
            vector = richer_features(state, piece, decision.evaluation)
            steps.append({"game_id": f"g{game_number}", "step_index": index,
                          "board_before": list(state.rows), "piece": piece,
                          "student_action": [decision.placement.rotation, decision.placement.x],
                          "learning_enabled": False, "feature_vector": list(vector),
                          "board_after": list(decision.evaluation.next_state.rows),
                          "lines_cleared": int(decision.evaluation.info["lines_cleared"])})
            lines += int(decision.evaluation.info["lines_cleared"])
            state = decision.evaluation.next_state
        records.append({"game_id": f"g{game_number}", "seed": seed,
                        "learning_enabled": False, "steps_played": len(steps),
                        "lines_cleared": lines, "terminal": len(steps) < maximum,
                        "final_board": list(state.rows), "steps": steps})
    return records


def _policy(name: str, omitted_family: str) -> RicherRLStudent:
    # Fixed hand-constructed policies create histories; these weights never enter diagnosis.
    weights = []
    beneficial = {"completed_lines", "eroded_piece_cells"}
    omitted = set(FEATURE_FAMILIES[omitted_family])
    for feature in FEATURE_NAMES:
        if feature in omitted:
            weights.append(0.0)
        elif feature in beneficial:
            weights.append(40.0)
        else:
            weights.append(-40.0)
    return RicherRLStudent(name, weights, seed=2026083000)


def run(*, seeds: Sequence[int] = SEEDS, maximum: int = 80) -> dict[str, Any]:
    day4 = json.loads((ROOT / "experiments/day4/final_results.json").read_text())
    natural_state = copy.deepcopy(day4["evaluation_before_states"]["rl@3000"])
    natural_state["agent_id"] = "natural-day4-rl"
    students = {
        "hole_weak": _policy("hole-weak", "hole_management"),
        "well_weak": _policy("well-weak", "well_management"),
        "natural_day4_rl": RicherRLStudent.from_state(natural_state),
    }
    profiles, histories, mutation = {}, {}, {}
    for name, student in students.items():
        before = copy.deepcopy(student.serialize_state())
        history = collect_history(student, seeds, maximum)
        after_collection = copy.deepcopy(student.serialize_state())
        profiles[name] = diagnose_history(history)
        after_diagnosis = copy.deepcopy(student.serialize_state())
        histories[name] = history
        mutation[name] = {"collection_nonmutating": before == after_collection,
                          "diagnosis_nonmutating": after_collection == after_diagnosis}
    hidden_copy = copy.deepcopy(histories["hole_weak"])
    for record in hidden_copy:
        record["weights"] = [999.0] * len(FEATURE_NAMES)
        record["serialized_parameters"] = {"weights": [-999.0] * len(FEATURE_NAMES)}
    invariant = diagnose_history(hidden_copy) == profiles["hole_weak"]
    intended = {"hole_weak": "hole_management", "well_weak": "well_management"}
    top_two = {name: family in profiles[name]["ranking"][:2] for name, family in intended.items()}
    l1 = sum(abs(profiles["hole_weak"]["families"][family]["score"] -
                 profiles["well_weak"]["families"][family]["score"])
             for family in FEATURE_FAMILIES)
    checks = {"intended_families_top_two": all(top_two.values()),
              "profiles_distinguishable": l1 >= .01,
              "hidden_weight_invariant": invariant,
              "student_nonmutating": all(all(row.values()) for row in mutation.values())}
    return {"schema_version": 1, "experiment": "Day 5 behavioral diagnosis",
            "configuration": {"seeds": list(seeds), "games_per_profile": len(seeds),
                              "maximum_placements_per_game": maximum,
                              "oracle": "DellacherieSearchExpert beam_width=1",
                              "student_decision_mode": "deterministic nonlearning"},
            "history_schema": ["game_id", "step_index", "board_before", "piece",
                               "student_action", "learning_enabled", "feature_vector",
                               "board_after", "lines_cleared", "game outcome"],
            "synthetic_construction": {
                "hole_weak": "common fixed policy with hole_management penalties omitted",
                "well_weak": "common fixed policy with well_management penalties omitted"},
            "profiles": profiles, "intended_top_two": top_two,
            "profile_l1_distance": l1, "mutation_audit": mutation,
            "hidden_weight_invariance": invariant, "success_checks": checks,
            "success": all(checks.values()), "histories": histories}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    args = parser.parse_args(argv)
    result = run(seeds=SEEDS[:2], maximum=30) if args.mode == "smoke" else run()
    if args.mode == "run":
        DAY5.mkdir(parents=True, exist_ok=True)
        (DAY5 / "final_results.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {"success": result["success"], "checks": result["success_checks"],
               "rankings": {name: profile["ranking"] for name, profile in result["profiles"].items()}}
    print(json.dumps(summary, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
