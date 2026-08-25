"""Independent structural and behavioral audit for the Day 3 result."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Mapping

from .expert import DELLACHERIE_WEIGHTS


def validate_day3(result: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    initial = result["initial_state"]
    trained = result["trained_states"]
    checks["identical_weak_start"] = (initial["weights"] == [0.0] * 4 and initial["updates"] == 0)
    def policy_state(state: Mapping[str, Any]) -> dict[str, Any]:
        copy = dict(state); copy.pop("agent_id", None); return copy
    checks["identical_weak_start"] &= all(policy_state(state) == policy_state(initial)
                                           for state in result["initial_clone_states"].values())
    checks["identical_initial_decisions"] = len(
        {json.dumps(value, sort_keys=True) for value in result["initial_probe_decisions"].values()}) == 1
    expert_values = set(float(value) for value in DELLACHERIE_WEIGHTS.values())
    student_values = [float(value) for state in [initial, *trained.values()] for value in state["weights"]]
    checks["no_expert_coefficients_in_student"] = (
        all(state.get("expert_parameters") is None for state in [initial, *trained.values()])
        and not any(value in expert_values for value in student_values))
    opportunities = result["training_audit"]["opportunities"]
    updates = result["training_audit"]["updates"]
    checks["equal_training_budgets"] = len(set(opportunities.values())) == 1 and opportunities == updates
    checks["training_evaluation_state_separation"] = set(
        result["training_audit"]["state_hashes"]).isdisjoint(result["evaluation_state_hashes"])
    config = result["configuration"]
    checks["training_evaluation_seed_separation"] = set(config["training_seeds"]).isdisjoint(
        set(config["evaluation_state_seeds"]) | set(config["evaluation_game_seeds"]))
    checks["evaluation_learning_disabled"] = result["evaluation_before_states"] == result["evaluation_after_states"]
    required_order = [*[f"student_choose:{name}" for name in ("initial", "taught", "random_label_control")],
                      "external_expert_score"]
    checks["expert_absent_during_student_choice"] = all(
        row["event_order"] == required_order for row in result["decision_trace"])
    checks["student_has_no_expert_dependency"] = all(
        set(state) == {"format", "agent_id", "agent_version", "learning_enabled", "feature_names",
                       "weights", "learning_rate", "updates", "expert_parameters"}
        for state in [initial, *trained.values()])
    checks["behavior_not_parameter_only"] = bool(result["knowledge_transfer_success"]) and all(
        result["success_checks"].values())
    checks["no_rating_claim"] = result.get("ratings") is None
    return {"schema_version": 1, "checks": checks,
            "overall": "PASS" if all(checks.values()) else "FAIL",
            "independence": "Recomputes controls from serialized result; does not call the Day 3 trainer or success function."}


def validate_file(path: Path) -> dict[str, Any]:
    return validate_day3(json.loads(path.read_text()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_file(args.result)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
