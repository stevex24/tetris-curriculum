"""Permanent, training-independent validation of agents and saved experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Mapping

from .legacy_student import FourFeatureStudentAdapter
from .tetris import HEIGHT, TetrisAdapter, TetrisState

CONDITIONS = ("control", "rating_only", "rating_history")


@dataclass(frozen=True)
class Check:
    status: str
    detail: str
    independent: bool = True


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _policy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    copy = dict(state)
    copy.pop("agent_id", None)
    return copy


def _state(raw: list[int]) -> TetrisState:
    if len(raw) != HEIGHT:
        raise ValueError("a validation board must contain exactly 20 row masks")
    return TetrisState(tuple(int(x) for x in raw))


def _trajectory(agent: FourFeatureStudentAdapter, pieces: list[str], max_placements: int,
                *, deterministic: bool) -> dict[str, Any]:
    game, board, actions, lines = TetrisAdapter(), TetrisState(), [], 0
    for piece in pieces[:max_placements]:
        choices = game.legal_actions(board, piece)
        if not choices:
            break
        decision = agent.choose_placement(board, piece, choices, learn=False,
                                          deterministic=deterministic)
        actions.append([decision.placement.rotation, decision.placement.x])
        board = decision.evaluation.next_state
        lines += int(decision.evaluation.info["lines_cleared"])
    digest = hashlib.sha256(_canonical(list(board.rows)).encode()).hexdigest()
    return {"actions": actions, "placements": len(actions), "lines_cleared": lines,
            "final_board_sha256": digest}


def validate_agent_bundle(bundle: Mapping[str, Any], artifact_root: Path) -> dict[str, Any]:
    """Recompute behavior from raw serialized agents, boards, and piece streams.

    This function does not import curriculum, trainer, rating, or experiment modules.
    It shares only canonical game physics and the format-specific agent loader.
    """
    initial = bundle["agents"]["initial"]
    trained = bundle["agents"]["trained"]
    initial_states = [_policy_state(initial[c]) for c in CONDITIONS]
    checks: dict[str, Check] = {}
    checks["baseline_agents_identical"] = Check(
        "PASS" if len({_canonical(x) for x in initial_states}) == 1 else "FAIL",
        "initial serialized policy/RNG states match after ignoring labels")

    budgets = bundle["training_budgets"]
    checks["training_budgets_equal"] = Check(
        "PASS" if len({int(budgets[c]) for c in CONDITIONS}) == 1 else "FAIL",
        f"budgets={dict(budgets)}")
    training_seeds = set(int(x) for x in bundle["seeds"]["training"])
    evaluation_seeds = set(int(x) for x in bundle["seeds"]["evaluation"])
    checks["training_evaluation_seeds_disjoint"] = Check(
        "PASS" if training_seeds.isdisjoint(evaluation_seeds) else "FAIL",
        f"overlap={sorted(training_seeds & evaluation_seeds)}")
    checks["held_out_evaluation_unseen"] = checks["training_evaluation_seeds_disjoint"]

    trained_states = [_policy_state(trained[c]) for c in CONDITIONS]
    state_distinct = len({_canonical(x) for x in trained_states}) == len(CONDITIONS)
    checks["trained_agent_states_differ"] = Check(
        "PASS" if state_distinct else "FAIL", "pairwise serialized trained-state comparison")

    agents = {c: FourFeatureStudentAdapter.from_state(trained[c]) for c in CONDITIONS}
    preferences: dict[str, list[list[int]]] = {c: [] for c in CONDITIONS}
    sampled: dict[str, list[list[int]]] = {c: [] for c in CONDITIONS}
    game = TetrisAdapter()
    for item in bundle["corpus"]:
        board, piece = _state(item["rows"]), str(item["piece"])
        choices = game.legal_actions(board, piece)
        for condition in CONDITIONS:
            deterministic = agents[condition].clone()
            d = deterministic.choose_placement(board, piece, choices, deterministic=True)
            preferences[condition].append([d.placement.rotation, d.placement.x])
            stochastic = agents[condition].clone()
            s = stochastic.choose_placement(board, piece, choices, deterministic=False)
            sampled[condition].append([s.placement.rotation, s.placement.x])
    distinct_preferences = len({_canonical(preferences[c]) for c in CONDITIONS}) > 1
    checks["deterministic_preferences_differ"] = Check(
        "PASS" if distinct_preferences else "FAIL", "fixed-corpus argmax placements recomputed")
    checks["matched_rng_decisions_differ"] = Check(
        "PASS" if len({_canonical(sampled[c]) for c in CONDITIONS}) > 1 else "WARNING",
        "fixed-corpus sampled placements recomputed from saved RNG states")

    replay_ok, trajectory_divergence = True, False
    replayed: dict[str, Any] = {}
    for stream in bundle["streams"]:
        stream_rows = {}
        for condition in CONDITIONS:
            result = _trajectory(FourFeatureStudentAdapter.from_state(trained[condition]),
                                 list(stream["pieces"]), int(stream["max_placements"]),
                                 deterministic=bool(stream.get("deterministic", False)))
            stream_rows[condition] = result
            replay_ok &= result == stream["expected"][condition]
        trajectory_divergence |= len({_canonical(stream_rows[c]) for c in CONDITIONS}) > 1
        replayed[str(stream["id"])] = stream_rows
    checks["complete_trajectories_diverge"] = Check(
        "PASS" if trajectory_divergence else "FAIL", "predetermined full streams replayed")
    checks["saved_replay_reconstruction_matches"] = Check(
        "PASS" if replay_ok else "FAIL", "actions, counts, lines, and final-board hashes recomputed")

    hashes_ok = True
    for relative, expected in bundle.get("artifact_hashes", {}).items():
        path = artifact_root / relative
        hashes_ok &= path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    checks["artifact_hashes_valid"] = Check("PASS" if hashes_ok else "FAIL",
                                             "SHA-256 recomputed from artifact bytes")
    statuses = [check.status for check in checks.values()]
    overall = "FAIL" if "FAIL" in statuses else ("WARNING" if "WARNING" in statuses else "PASS")
    return {"schema_version": 1, "experiment": bundle.get("experiment", "agent bundle"),
            "repository_commit": bundle.get("repository_commit", "unspecified"),
            "checks": {name: check.__dict__ for name, check in checks.items()},
            "preferences": preferences, "sampled_decisions": sampled,
            "replayed_trajectories": replayed, "overall_status": overall,
            "independence": "No curriculum/training/experiment/statistics module is imported; canonical Tetris physics and the legacy state-format adapter are shared."}


def _working_tree_matches(root: Path, commit: str, directory: str) -> bool:
    listing = subprocess.run(["git", "ls-tree", "-r", commit, directory], cwd=root,
                             text=True, capture_output=True, check=True).stdout.splitlines()
    for line in listing:
        metadata, relative = line.split("\t", 1)
        expected = metadata.split()[2]
        path = root / relative
        if not path.is_file():
            return False
        actual = subprocess.run(["git", "hash-object", str(path)], cwd=root, text=True,
                                capture_output=True, check=True).stdout.strip()
        if actual != expected:
            return False
    return True


def validate_legacy_experiment(experiment: Path, root: Path) -> dict[str, Any]:
    """Audit committed Hour-9 raw tables without rerunning its experiment."""
    manifest = json.loads((root / "baseline/hours_1_9_manifest.json").read_text())
    provenance = json.loads((experiment / "results/configuration_seeds_provenance.json").read_text())
    behavioral = json.loads((experiment / "behavioral_validation.json").read_text())
    rows = list(csv.DictReader((experiment / "results/replicate_results.csv").open()))
    checks: dict[str, Check] = {}
    controls = provenance["controls"]
    checks["baseline_agents_identical"] = Check("PASS" if controls["identical_initial_states"] else "FAIL",
                                                 "legacy provenance control", False)
    budgets = {int(row["training_steps"]) for row in rows}
    checks["training_budgets_equal"] = Check("PASS" if budgets == {int(provenance["config"]["training_placements"])} else "FAIL",
                                              f"recomputed from {len(rows)} CSV rows: {sorted(budgets)}")
    overlap = []
    for seeds in provenance["replicate_seeds"]:
        train = {seeds["baseline_history"], seeds["tutorial_selection"], seeds["training_stream"]}
        overlap.extend(train & set(seeds["evaluation"]))
    checks["training_evaluation_seeds_disjoint"] = Check("PASS" if not overlap else "FAIL",
                                                          f"recomputed overlap count={len(overlap)}")
    checks["held_out_evaluation_unseen"] = checks["training_evaluation_seeds_disjoint"]
    action_counts = [n for row in behavioral["replicates"]
                     for n in row["pair_action_differences_on_shared_corpus"].values()]
    checks["trained_policies_behaviorally_distinct"] = Check("PASS" if any(n > 0 for n in action_counts) else "FAIL",
                                                               "recomputed from saved fixed-corpus action-difference counts")
    replay_claim = behavioral["checks"]["reconstructed_trajectories_match_saved_outputs"]
    checks["saved_replay_reconstruction_matches"] = Check("PASS" if replay_claim else "FAIL",
                                                            "preserved Hour-9 attestation; raw actions were not saved", False)
    artifact_ok = all(_working_tree_matches(root, manifest["git_head"], directory)
                      for directory, item in manifest["scientific_artifacts"].items()
                      if "git_tree_sha1" in item)
    checks["artifact_hashes_valid"] = Check("PASS" if artifact_ok else "FAIL",
                                             "every committed artifact blob rehashed from the working tree")
    performance = {}
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        performance[condition] = {"before": mean(float(r["pre_primary"]) for r in selected),
                                  "after": mean(float(r["post_primary"]) for r in selected)}
    statuses = [c.status for c in checks.values()]
    # A non-independent critical replay claim prevents a full Day-1 certificate PASS.
    overall = "FAIL" if "FAIL" in statuses else "WARNING"
    return {"schema_version": 1, "experiment": str(experiment.relative_to(root)),
            "repository_commit": manifest["git_head"],
            "checks": {name: check.__dict__ for name, check in checks.items()},
            "performance_primary_outcome": performance, "ratings": None,
            "overall_status": overall,
            "independence": "CSV budgets, seed disjointness, means, and artifact hashes are independently recomputed. Initial-clone and replay claims remain legacy attestations because Hour 9 did not save raw initial agents or action-level evaluation replays."}


def certificate(report: Mapping[str, Any]) -> str:
    labels = {
        "baseline_agents_identical": "Baseline agents identical",
        "training_budgets_equal": "Training budgets equal",
        "training_evaluation_seeds_disjoint": "Training/evaluation seeds disjoint",
        "held_out_evaluation_unseen": "Held-out evaluation unseen",
        "trained_policies_behaviorally_distinct": "Trained policies behaviorally distinct",
        "deterministic_preferences_differ": "Deterministic preferences differ",
        "matched_rng_decisions_differ": "Matched-RNG decisions differ",
        "complete_trajectories_diverge": "Complete trajectories diverge",
        "saved_replay_reconstruction_matches": "Saved replay reconstruction matches",
        "artifact_hashes_valid": "Artifact hashes valid",
    }
    lines = ["ADAPTIVE TETRIS VALIDATION CERTIFICATE", "",
             f"Repository commit: {report['repository_commit']}",
             f"Experiment: {report['experiment']}", ""]
    for key, item in report["checks"].items():
        if key in labels:
            suffix = " (legacy attestation)" if not item.get("independent", True) else ""
            lines.append(f"{labels[key]}: {item['status']}{suffix}")
    for condition, values in report.get("performance_primary_outcome", {}).items():
        lines.append(f"Performance {condition} before/after: {values['before']:.6g} / {values['after']:.6g}")
    if report.get("ratings") is None:
        lines.append("Ratings: NOT REPORTED (no externally calibrated rating)")
    lines += ["", f"Overall status: {report['overall_status']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an independently auditable validation certificate")
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--json", type=Path, help="also write the detailed machine-readable report")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    experiment = args.experiment.resolve()
    bundle_path = experiment / "validation_bundle.json"
    report = (validate_agent_bundle(json.loads(bundle_path.read_text()), experiment)
              if bundle_path.exists() else validate_legacy_experiment(experiment, root))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(certificate(report))
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
