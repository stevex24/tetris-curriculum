"""Independent artifact audit for the Day 4 RL experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .expert import DELLACHERIE_WEIGHTS


def validate_day4(result: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    config, audit = result["configuration"], result["training_audit"]
    initial = result["initial_state"]
    checks["richer_interpretable_representation"] = (
        12 <= result["trainable_parameters"] <= 30
        and len(result["feature_names"]) == len(set(result["feature_names"]))
        == result["trainable_parameters"])
    expert_values = set(map(float, DELLACHERIE_WEIGHTS.values()))
    checks["no_expert_coefficients"] = (initial.get("expert_parameters") is None
                                          and not expert_values.intersection(map(float, initial["weights"])))
    checks["expert_absent_from_training"] = audit.get("expert_calls") == 0
    reward = str(result["reward"]).lower()
    checks["environment_reward_only"] = ("expert" not in reward and "regret" not in reward
                                          and "actually cleared" in reward)
    checks["equal_exposure"] = (len(set(audit["opportunities"].values())) == 1
                                and next(iter(audit["opportunities"].values())) == config["checkpoints"][-1])
    checks["rl_updates_match_exposure"] = (audit["rl_updates"] == config["checkpoints"][-1]
                                            and audit["control_updates"] == 0)
    domains = ({config["training_seed"]}, set(config["evaluation_game_seeds"]),
               set(config["evaluation_state_seeds"]))
    checks["training_evaluation_seed_separation"] = all(
        not domains[i].intersection(domains[j]) for i in range(3) for j in range(i + 1, 3))
    checks["evaluation_nonmutating"] = result["evaluation_before_states"] == result["evaluation_after_states"]
    checks["delayed_credit_declared"] = ("G_t = r_t + 0.99 G_(t+1)" in result["credit_assignment"]
                                         and any(length > 1 for length in audit["rl_trajectory_lengths"]))
    checks["behavior_not_parameters_only"] = bool(result["rl_success"]) and all(result["success_checks"].values())
    checks["primary_gameplay_metrics_present"] = all(
        {"mean_placements", "mean_lines_cleared"} <= set(metrics)
        for metrics in result["learning_curve"].values())
    checks["expert_is_secondary_only"] = set(result["expert_diagnostics"]) == {
        "initial", "final_rl", "ordinary_play_control"}
    checks["no_uncalibrated_rating"] = result.get("ratings") is None
    return {"schema_version": 1, "checks": checks,
            "overall": "PASS" if all(checks.values()) else "FAIL",
            "independence": "Recomputes artifact invariants without importing the Day 4 trainer or success rule."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_day4(json.loads(args.result.read_text()))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
