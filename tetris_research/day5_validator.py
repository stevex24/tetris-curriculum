"""Independent structural audit for a serialized Day 5 result."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .diagnosis import FEATURE_FAMILIES, diagnose_history
from .richer_student import FEATURE_NAMES
from .tetris import HEIGHT, TetrisAdapter, TetrisState


def validate_day5(result: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    histories, profiles = result["histories"], result["profiles"]
    checks["documented_mapping_complete"] = (
        set().union(*map(set, FEATURE_FAMILIES.values())) == set(FEATURE_NAMES)
        and sum(map(len, FEATURE_FAMILIES.values())) == len(FEATURE_NAMES))
    required = {"game_id", "step_index", "board_before", "piece", "student_action",
                "learning_enabled", "feature_vector", "board_after", "lines_cleared"}
    complete = replayable = True
    adapter = TetrisAdapter()
    for records in histories.values():
        for record in records:
            for step in record["steps"]:
                complete &= required <= set(step)
                try:
                    legal = adapter.legal_actions(TetrisState(tuple(step["board_before"])), step["piece"])
                    selected = next(x for x in legal if list(x.action) == step["student_action"])
                    replayable &= (len(step["board_before"]) == HEIGHT and
                                   list(selected.next_state.rows) == step["board_after"])
                except (StopIteration, KeyError, TypeError, ValueError):
                    replayable = False
    checks["history_fields_complete"] = complete
    checks["independently_replayable"] = replayable
    checks["student_decision_precedes_oracle"] = all(
        event["events"] == ["recorded_student_decision", "external_oracle_evaluation"]
        for profile in profiles.values() for event in profile["event_order"])
    checks["near_zero_excluded"] = all(
        profile["material_regret_decisions"] + profile["below_threshold_decisions"] ==
        profile["decision_count"] for profile in profiles.values())
    checks["cross_game_consistency_in_score"] = all(
        "fraction of games" in profile["scoring_rule"] for profile in profiles.values())
    hidden = copy.deepcopy(histories["hole_weak"])
    for record in hidden:
        record["weights"] = [1e100] * len(FEATURE_NAMES)
        record["training_only"] = {"weights": [-1e100] * len(FEATURE_NAMES)}
    checks["hidden_weights_irrelevant"] = diagnose_history(hidden) == profiles["hole_weak"]
    checks["distinct_synthetic_profiles"] = (
        result["profile_l1_distance"] >= .01 and
        profiles["hole_weak"]["ranking"] != profiles["well_weak"]["ranking"])
    checks["intended_weaknesses_recovered"] = all(result["intended_top_two"].values())
    checks["diagnosis_nonmutating"] = all(
        audit["collection_nonmutating"] and audit["diagnosis_nonmutating"]
        for audit in result["mutation_audit"].values())
    checks["no_hidden_training_data"] = all(profile.get("history_only") is True
                                              for profile in profiles.values())
    return {"schema_version": 1, "checks": checks,
            "overall": "PASS" if all(checks.values()) else "FAIL",
            "independence": "Replays histories and recomputes diagnosis invariance without importing the Day 5 experiment runner."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_day5(json.loads(args.result.read_text()))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
