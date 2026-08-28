"""Independent behavioral checks for the locked Day 6 artifact."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from .richer_student import RicherRLStudent
from .tutorials import TutorialSituation, allocate_profile, select_personalized, train_on_situations

ROOT = Path(__file__).resolve().parents[1]


def validate(result: dict[str, Any]) -> dict[str, Any]:
    source = (ROOT / "tetris_research/tutorials.py").read_text()
    tree = ast.parse(source)
    generator_functions = {"candidate_pool", "generate_tutorial_sets", "generic_tutorial_set",
                           "allocate_profile", "select_personalized", "targeting_metric"}
    forbidden_names = {"weights", "DellacherieSearchExpert", "rank_placements"}
    forbidden = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in generator_functions:
            names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
            attrs = {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)}
            forbidden.extend(sorted((names | attrs) & forbidden_names))
    no_labels = all(set(row) == {"situation_id", "family", "rows", "piece", "source_id", "perturbation"}
                    for rows in result["sets"].values() for row in rows)
    day4_source = (ROOT / "tetris_research/day4.py").read_text()
    day4_hash = __import__("hashlib").sha256(day4_source.encode()).hexdigest()
    expected_hash = __import__("hashlib").sha256(
        __import__("subprocess").check_output(["git", "show", "bc1f859:tetris_research/day4.py"],
                                             cwd=ROOT).decode().encode()).hexdigest()
    checks = {"result_success": result["success"], "selector_ast_has_no_hidden_weight_or_oracle_names": not forbidden,
              "scenario_schema_has_no_action_labels_or_rewards": no_labels,
              "day4_source_unchanged_from_checkpoint": day4_hash == expected_hash,
              "equal_budget": result["learning_smoke"]["targeted"]["updates_applied"] ==
                              result["learning_smoke"]["generic"]["updates_applied"],
              "target_labels_behaviorally_valid": all(row["ratio"] >= 1.25
                                                       for row in result["target_metrics"].values()),
              "multiple_sources": all(len({row["source_id"] for row in rows}) >= 4
                                      for rows in result["sets"].values()),
              "nonterminal_plausible": all(sum(bin(int(mask)).count("1") for mask in row["rows"]) <= 120
                                           and max((i + 1 for i, mask in enumerate(row["rows"]) if mask), default=0) <= 16
                                           for rows in result["sets"].values() for row in rows),
              "overlap_not_claimed_exclusive": "exclusive" not in result["representation"].lower()}
    return {"schema_version": 1, "validator": "independent Day 6 behavioral validator",
            "checks": checks, "success": all(checks.values()), "forbidden_generator_references": forbidden}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("result", nargs="?",
        default=str(ROOT / "experiments/day6/final_results.json"))
    args = parser.parse_args(argv)
    validation = validate(json.loads(Path(args.result).read_text()))
    output = ROOT / "experiments/day6/validation.json"
    output.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
