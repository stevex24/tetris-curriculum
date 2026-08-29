"""Independent artifact validator for the Day 7 comparison."""
from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from . import day7, tutorials
from .richer_student import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _without_id(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value)); result.pop("agent_id", None)
    return result


def validate(result: Mapping[str, Any], preregistration: Mapping[str, Any]) -> dict[str, Any]:
    config, replicates = result["configuration"], result["replicates"]
    conditions = tuple(result["conditions"])
    starts_equal = all(len({json.dumps(_without_id(row["initial_states"][name]), sort_keys=True)
                            for name in conditions}) == 1 for row in replicates)
    updates_equal = all({row["training_audits"][name]["updates_applied"]
                         for name in conditions} == {config["training_budget"]}
                        for row in replicates)
    exposure_equal = all({row["training_audits"][name]["decision_states_seen"]
                          for name in conditions} == {config["training_budget"]}
                         for row in replicates)
    training = set(config["replicate_seeds"])
    history = set(config["history_seeds"]) | {config["tutorial_generation_seed"]}
    evaluation = set(config["evaluation_game_seeds"]) | set(config["diagnostic_state_seeds"])
    source = ast.parse(inspect.getsource(tutorials.select_personalized))
    names = {node.id for node in ast.walk(source) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(source) if isinstance(node, ast.Attribute)}
    checks = {
        "identical_starting_students": starts_equal,
        "equal_update_counts": updates_equal,
        "equal_decision_exposure": exposure_equal,
        "history_collection_has_no_free_learning": (
            result["shared_history"]["student_nonmutating"] and
            result["history_design"].startswith("Option A")),
        "expert_coefficients_not_copied": (not result["expert_coefficients_copied"] and
            all(row["trained_states"]["imitation"]["expert_parameters"] is None and
                len(row["trained_states"]["imitation"]["weights"]) == len(FEATURE_NAMES)
                for row in replicates)),
        "no_expert_in_personalized_learning": all(
            row["training_audits"]["personalized"]["expert_calls_during_learning"] == 0
            for row in replicates),
        "student_decisions_precede_diagnostic_expert": all(
            row["diagnostic_event_order"] == ["all_frozen_student_decisions", "external_expert_scoring"]
            for row in replicates),
        "no_personalized_reward_shaping": all(
            row["training_audits"]["personalized"]["reward_source"] ==
            "day4.gameplay_reward: 0.02 + simulator lines_cleared" for row in replicates),
        "same_rl_algorithm_and_reward": all(
            row["training_audits"]["ordinary"]["learning_path"] ==
            row["training_audits"]["personalized"]["learning_path"] and
            row["training_audits"]["ordinary"]["reward_source"] == "day4.gameplay_reward"
            for row in replicates),
        "evaluation_nonmutating": all(row["evaluation_before_sha256"] ==
                                      row["evaluation_after_sha256"] for row in replicates),
        "training_evaluation_seeds_disjoint": not ((training | history) & evaluation),
        "condition_evaluation_streams_matched": all(
            all(set(game_row) >= {"seed", "stream_sha256", "baseline", *conditions}
                for game_row in row["game_rows"]) for row in replicates),
        "selection_does_not_use_hidden_weights": not ({"weights", "parameters"} & (names | attributes)),
        "locked_configuration_matches_result": (preregistration["status"] == "locked before final run" and
                                                preregistration["configuration"] == config),
        "common_cap_for_all_conditions": all(
            all(all(name in game_row for name in ("baseline", *conditions))
                for game_row in row["game_rows"]) for row in replicates),
        "paired_replicate_identities_preserved": ([row["replicate_id"] for row in replicates] ==
                                                   list(range(len(replicates))) and
                                                   [row["training_seed"] for row in replicates] ==
                                                   config["replicate_seeds"]),
        "primary_is_independent_gameplay": result["primary_metric"].startswith(
            "change in matched held-out mean lines"),
        "no_uncalibrated_rating": result["ratings"] is None,
    }
    return {"schema_version": 1, "validator": "independent Day 7 artifact validator",
            "checks": checks, "success": all(checks.values())}


def main() -> int:
    result = json.loads((ROOT / "experiments/day7/final_results.json").read_text())
    prereg = json.loads((ROOT / "experiments/day7/preregistration.json").read_text())
    validation = validate(result, prereg)
    (ROOT / "experiments/day7/validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
