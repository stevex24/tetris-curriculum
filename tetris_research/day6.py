"""Day 6 targeted tutorial generation experiment (not the Day 7 comparison)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .richer_student import RicherRLStudent
from .tutorials import (SUPPORTED_FAMILIES, allocate_profile, fingerprint_situations,
                        generate_tutorial_sets, generic_tutorial_set, select_personalized,
                        targeting_metric, train_on_situations)

ROOT = Path(__file__).resolve().parents[1]
DAY6 = ROOT / "experiments/day6"
GENERATION_SEED = 2026083101
TRAINING_BUDGET = 24


def _mean_metric(rows: Any, family: str) -> float:
    return sum(targeting_metric(row.state, row.piece, family) for row in rows) / len(rows)


def run(*, smoke: bool = False) -> dict[str, Any]:
    day5 = json.loads((ROOT / "experiments/day5/final_results.json").read_text())
    history_before = copy.deepcopy(day5["histories"]["natural_day4_rl"])
    profile = copy.deepcopy(day5["profiles"]["natural_day4_rl"])
    per_family = 4 if smoke else 8
    sets = generate_tutorial_sets(history_before, seed=GENERATION_SEED, per_family=per_family)
    generic = generic_tutorial_set(history_before, seed=GENERATION_SEED, count=per_family)
    budget = 8 if smoke else TRAINING_BUDGET
    personalized = select_personalized(profile, sets, budget)
    synthetic_profile = copy.deepcopy(profile)
    synthetic_profile["ranking"] = ["hole_management", "well_management", "surface_smoothness",
                                     "stack_height_danger", "transition_structure", "line_clear_recovery"]
    synthetic_profile["families"]["hole_management"]["score"] = 90.0
    synthetic_profile["families"]["well_management"]["score"] = 10.0
    contrasting = select_personalized(synthetic_profile, sets, budget)
    target_metrics = {family: {"targeted_mean": _mean_metric(rows, family),
                               "generic_mean": _mean_metric(generic, family),
                               "ratio": _mean_metric(rows, family) / max(_mean_metric(generic, family), 1e-12)}
                      for family, rows in sets.items()}
    natural_state = copy.deepcopy(json.loads((ROOT / "experiments/day4/final_results.json").read_text())
                                  ["evaluation_before_states"]["rl@3000"])
    natural_state["agent_id"] = "day6-base"
    targeted_student = RicherRLStudent.from_state(natural_state)
    generic_student = RicherRLStudent.from_state({**natural_state, "agent_id": "day6-generic"})
    targeted_before = copy.deepcopy(targeted_student.serialize_state())
    generic_before = copy.deepcopy(generic_student.serialize_state())
    targeted_audit = train_on_situations(targeted_student, personalized, budget)
    generic_audit = train_on_situations(generic_student, generic, budget)
    checks = {
        "profile_determines_selection": allocate_profile(profile, budget) != allocate_profile(synthetic_profile, budget),
        "legal_replayable": all(len(__import__("tetris_research.tetris", fromlist=["TetrisAdapter"]).TetrisAdapter()
                                    .legal_actions(row.state, row.piece)) >= 2
                                for rows in sets.values() for row in rows),
        "target_metrics_ratio_at_least_1_25": all(row["ratio"] >= 1.25 for row in target_metrics.values()),
        "different_profiles_differ": fingerprint_situations(personalized) != fingerprint_situations(contrasting),
        "equal_exact_exposure": targeted_audit["updates_applied"] == generic_audit["updates_applied"] == budget,
        "ordinary_rl_path": targeted_audit["learning_path"] == generic_audit["learning_path"] ==
                            "RicherRLStudent.choose_placement/update/finish_episode",
        "no_expert_training_calls": targeted_audit["expert_calls"] == generic_audit["expert_calls"] == 0,
        "environment_reward_only": targeted_audit["reward_source"] == generic_audit["reward_source"],
        "diverse": all(len({row.rows for row in rows}) == per_family and
                       len({row.source_id for row in rows}) >= min(4, per_family) for rows in sets.values()),
        "history_not_mutated": history_before == day5["histories"]["natural_day4_rl"],
        "standard_agent_version": targeted_before["agent_version"] == generic_before["agent_version"] ==
                                  targeted_student.agent_version,
    }
    return {"schema_version": 1, "experiment": "Day 6 targeted tutorial generation",
            "configuration": {"generation_seed": GENERATION_SEED, "training_budget": budget,
                              "situations_per_family": per_family},
            "representation": "20 row masks plus current tetromino; no action label or stored reward",
            "sets": {family: [row.as_dict() for row in rows] for family, rows in sets.items()},
            "generic_set": [row.as_dict() for row in generic],
            "target_metrics": target_metrics,
            "natural_profile_ranking": profile["ranking"],
            "natural_allocation": allocate_profile(profile, budget),
            "natural_selected_ids": [row.situation_id for row in personalized],
            "contrasting_allocation": allocate_profile(synthetic_profile, budget),
            "contrasting_selected_ids": [row.situation_id for row in contrasting],
            "learning_smoke": {"targeted": targeted_audit, "generic": generic_audit,
                               "initial_weights_equal": targeted_before["weights"] == generic_before["weights"],
                               "weights_changed": targeted_before["weights"] != targeted_student.weights and
                                                  generic_before["weights"] != generic_student.weights},
            "success_checks": checks, "success": all(checks.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    args = parser.parse_args(argv)
    result = run(smoke=args.mode == "smoke")
    if args.mode == "run":
        DAY6.mkdir(parents=True, exist_ok=True)
        (DAY6 / "final_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"success": result["success"], "checks": result["success_checks"],
                      "natural_allocation": result["natural_allocation"],
                      "contrasting_allocation": result["contrasting_allocation"],
                      "target_metrics": result["target_metrics"]}, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
