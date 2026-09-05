"""Independent structural and artifact checks for the Phase-2 experiment."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from . import adaptive_tutor
from .richer_student import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def validate(result: Mapping[str, Any], preregistration: Mapping[str, Any]) -> dict[str, Any]:
    config = result["configuration"]
    reps = result["replicates"]
    conditions = tuple(result["conditions"])
    source = ast.parse(inspect.getsource(adaptive_tutor.value_candidates) +
                       inspect.getsource(adaptive_tutor.select_block))
    attributes = {node.attr for node in ast.walk(source) if isinstance(node, ast.Attribute)}
    training = set(config["replicate_seeds"])
    diagnostics = set(config["responsive_diagnostic_seeds"])
    calibration = set(config["rating_calibration_seeds"]) | set(config["rating_probe_seeds"])
    evaluation = set(config["evaluation_game_seeds"]) | set(config["diagnostic_state_seeds"])
    all_domains = [training, diagnostics, calibration, evaluation,
                   set(config["history_seeds"]) | {config["tutorial_generation_seed"]}]
    checks = {
        "responsive_does_not_read_or_set_weights": "weights" not in attributes,
        "responsive_receives_no_expert_actions": all(
            row["training_audits"]["responsive_personalized"]["expert_labels_seen"] == 0
            for row in reps),
        "no_expert_reward_shaping": all(
            row["training_audits"]["responsive_personalized"]["reward_source"] ==
            "day4.gameplay_reward: 0.02 + simulator lines_cleared" for row in reps),
        "equal_learner_updates": all({row["training_audits"][name]["updates_applied"]
                                      for name in conditions} == {config["training_budget"]}
                                     for row in reps),
        "static_and_responsive_same_rl_algorithm": all(
            row["training_audits"]["static_personalized"]["learning_path"] ==
            row["training_audits"]["responsive_personalized"]["learning_path"]
            for row in reps),
        "diagnosis_nonlearning": all(
            row["training_audits"]["responsive_personalized"]["diagnostic_placements"] > 0 and
            row["training_audits"]["responsive_personalized"]["expert_calls_during_learning"] == 0
            for row in reps),
        "evaluation_nonmutating": all(row["evaluation_before_sha256"] ==
                                      row["evaluation_after_sha256"] for row in reps),
        "training_evaluation_seeds_disjoint": all(
            not (all_domains[i] & all_domains[j]) for i in range(len(all_domains))
            for j in range(i + 1, len(all_domains))),
        "curriculum_changes_have_behavioral_evidence": all(
            all(block["profile_distance_from_previous"] is None or
                block["profile_distance_from_previous"] > 0
                for block in row["training_audits"]["responsive_personalized"]["blocks"])
            for row in reps),
        "retirement_rule_active": all(
            all("mastery_factor" in value and "retired" in value
                for block in row["training_audits"]["responsive_personalized"]["blocks"]
                for value in block["predicted_values"]) for row in reps),
        "full_18_dimensions_preserved": all(
            all(block["profile"]["dimensions"] == 18 and
                set(block["profile"]["severity"]) == set(FEATURE_NAMES)
                for block in row["training_audits"]["responsive_personalized"]["blocks"])
            for row in reps),
        "rating_uses_no_training_seeds": not (training & (calibration | evaluation)),
        "common_rating_opponents": tuple(result["rating_ladder"]["ratings"]) ==
                                   ("rl@0", "rl@250", "rl@750", "rl@1500", "rl@3000"),
        "demo_seed_prespecified": preregistration.get("demo_seed_rule") ==
                                  "first final held-out evaluation seed",
        "day7_not_used_for_tuning": preregistration.get("no_tuning_rule", "").startswith(
                                    "Day 7 final outcomes are historical context only"),
        "locked_preregistration_matches": preregistration.get("status") ==
            "locked before final run" and preregistration["configuration"] == config,
    }
    return {"schema_version": 1, "validator": "independent Phase-2 artifact validator",
            "checks": checks, "success": all(checks.values())}


def main() -> int:
    result = json.loads((ROOT / "experiments/phase2/final_results.json").read_text())
    prereg = json.loads((ROOT / "experiments/phase2/preregistration.json").read_text())
    report = validate(result, prereg)
    (ROOT / "experiments/phase2/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
