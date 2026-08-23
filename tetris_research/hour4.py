from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any, Iterable

from .agent import LearningAgent
from .legacy_student import as_student_agent
from .student import StudentAgent
from .elo import EloRatings
from .hour3 import _t_critical_975, _t_two_sided_p
from .tetris import TetrisAdapter
from .training import (CONDITIONS, CONTROL, RATING_HISTORY, RATING_ONLY, control_material,
                       select_history_material, select_rating_only_material, train)

SKILL_METRICS = ("hole_avoidance", "stack_height_management", "surface_smoothness",
                 "line_clearing_efficiency")


@dataclass(frozen=True)
class Hour4Config:
    master_seed: int = 400_004
    replicates: int = 50
    baseline_history_placements: int = 40
    training_placements: int = 40
    evaluation_challenges: int = 6
    evaluation_max_placements: int = 120
    initial_elo: float = 1500.0
    elo_k: float = 32.0
    initial_weights: tuple[float, ...] = (0.20, 0.10, 0.15, 0.25)
    learning_rate: float = 0.04
    temperature: float = 0.25
    primary_measure: str = "mean successful placements per held-out challenge"


def replicate_seeds(master_seed: int, replicate: int) -> dict[str, Any]:
    """Reproducible Hour-4-specific domains; evaluation never shares a training seed."""
    def derive(label: str) -> int:
        digest = hashlib.sha256(f"hour4:{master_seed}:{replicate}:{label}".encode()).digest()
        return int.from_bytes(digest[:8], "big") & 0x7fffffff
    return {"baseline_agent": derive("baseline-agent"),
            "baseline_history": derive("baseline-history"),
            "tutorial_selection": derive("tutorial-selection"),
            "training_stream": derive("training-stream"),
            "evaluation": [derive(f"evaluation-{i}") for i in range(64)]}


def skill_scores_from_steps(steps: Iterable[dict[str, float]]) -> dict[str, float]:
    """Observable trajectory scores, all oriented so larger means better skill."""
    rows = list(steps)
    if not rows:
        return {name: 0.0 for name in SKILL_METRICS}
    return {
        "hole_avoidance": -mean(float(row["holes"]) for row in rows),
        "stack_height_management": -mean(float(row["max_height"]) for row in rows),
        "surface_smoothness": -mean(float(row["bumpiness"]) for row in rows),
        "line_clearing_efficiency": sum(float(row["lines_cleared"]) for row in rows) / len(rows),
    }


def primary_score(challenges: Iterable[dict[str, Any]]) -> float:
    """Predeclared general outcome: mean successful placements; larger is better."""
    rows = list(challenges)
    if not rows:
        raise ValueError("at least one evaluation challenge is required")
    return mean(float(row["successful_placements"]) for row in rows)


def _evaluate_challenge(agent: StudentAgent, seed: int, max_placements: int) -> dict[str, Any]:
    adapter, environment_rng = TetrisAdapter(), Random(seed)
    state, lines, placements, trajectory = adapter.initial_state(), 0, 0, []
    while placements < max_placements:
        piece = adapter.sample_environment(environment_rng)
        choices = adapter.legal_actions(state, piece)
        if not choices:
            break
        choice = agent.choose_placement(state, piece, choices, learn=False).evaluation
        state, placements = choice.next_state, placements + 1
        lines += int(choice.info["lines_cleared"])
        trajectory.append(dict(choice.raw_features))
    return {"seed": seed, "lines_cleared": float(lines),
            "successful_placements": float(placements),
            "reached_placement_cap": placements == max_placements,
            "skill_scores": skill_scores_from_steps(trajectory)}


def evaluate(agent: StudentAgent | LearningAgent, seeds: Iterable[int], max_placements: int) -> dict[str, Any]:
    """Use a disposable clone and observable play only; the supplied agent is untouched."""
    source = as_student_agent(agent)
    evaluation_agent = source.clone(f"{source.agent_id}_hour4_evaluation_copy")
    challenges = [_evaluate_challenge(evaluation_agent, seed, max_placements) for seed in seeds]
    skill_scores = {name: mean(c["skill_scores"][name] for c in challenges)
                    for name in SKILL_METRICS}
    metrics = {"successful_placements": mean(c["successful_placements"] for c in challenges),
               "lines_cleared": mean(c["lines_cleared"] for c in challenges)}
    return {"challenges": challenges, "metrics": metrics, "skill_scores": skill_scores,
            "primary_score": primary_score(challenges)}


def _elo(evaluations: dict[str, dict[str, Any]], config: Hour4Config) -> dict[str, float]:
    ratings = EloRatings(config.initial_elo, config.elo_k)
    for index in range(config.evaluation_challenges):
        for a, b in ((CONTROL, RATING_ONLY), (CONTROL, RATING_HISTORY),
                     (RATING_ONLY, RATING_HISTORY)):
            x = evaluations[a]["challenges"][index]["lines_cleared"]
            y = evaluations[b]["challenges"][index]["lines_cleared"]
            ratings.update(a, b, 1.0 if x > y else (0.0 if x < y else 0.5))
    return {condition: ratings.rating(condition) for condition in CONDITIONS}


def _target_metric(weakness: str) -> str:
    return {"hole_avoidance": "hole_avoidance", "stack_height": "stack_height_management",
            "bumpiness": "surface_smoothness",
            "line_efficiency": "line_clearing_efficiency"}[weakness]


def run_replicate(config: Hour4Config, replicate: int) -> dict[str, Any]:
    seeds = replicate_seeds(config.master_seed, replicate)
    eval_seeds = seeds["evaluation"][:config.evaluation_challenges]
    training_seeds = {seeds["baseline_history"], seeds["tutorial_selection"],
                      seeds["training_stream"]}
    if training_seeds.intersection(eval_seeds):
        raise AssertionError("evaluation and training/tutorial seeds overlap")
    baseline = LearningAgent("baseline", list(config.initial_weights), config.learning_rate,
                             config.temperature, seeds["baseline_agent"])
    baseline_record = train(baseline, control_material(), config.initial_elo,
                            config.baseline_history_placements, seeds["baseline_history"])
    prior_history = [{"agent": baseline.name, "seed": seeds["baseline_history"],
                      "steps": baseline_record["placements"]}]
    agents = {condition: baseline.clone(condition) for condition in CONDITIONS}
    fingerprints = [{"weights": a.weights, "elo": config.initial_elo,
                     "rng": repr(a._rng.getstate()), "games": a.games_learned,
                     "history": prior_history} for a in agents.values()]
    identical = len({json.dumps(x, sort_keys=True) for x in fingerprints}) == 1
    if not identical:
        raise AssertionError("condition clones are not experimentally identical")
    pre = {c: evaluate(a, eval_seeds, config.evaluation_max_placements) for c, a in agents.items()}
    pre_elo = _elo(pre, config)
    materials = {CONTROL: control_material(),
                 RATING_ONLY: select_rating_only_material(pre_elo[RATING_ONLY],
                                                          seeds["tutorial_selection"]),
                 RATING_HISTORY: select_history_material(pre_elo[RATING_HISTORY], prior_history,
                                                         seeds["tutorial_selection"])}
    training = {c: train(agents[c], materials[c], pre_elo[c], config.training_placements,
                         seeds["training_stream"]) for c in CONDITIONS}
    if {x["training_steps"] for x in training.values()} != {config.training_placements}:
        raise AssertionError("unequal training exposure")
    post = {c: evaluate(a, eval_seeds, config.evaluation_max_placements) for c, a in agents.items()}
    post_elo = _elo(post, config)
    weakness = training[RATING_HISTORY]["diagnosed_weakness"]
    target_metric = _target_metric(weakness)
    target = {"diagnosed_weakness": weakness,
              "diagnostic_evidence": training[RATING_HISTORY]["recent_history_features"],
              "tutorial_type": weakness, "target_metric": target_metric,
              "pre_score": pre[RATING_HISTORY]["skill_scores"][target_metric],
              "post_score": post[RATING_HISTORY]["skill_scores"][target_metric]}
    target["improvement"] = target["post_score"] - target["pre_score"]
    rows = []
    for condition in CONDITIONS:
        row = {"replicate": replicate, "condition": condition,
               "pre_primary": pre[condition]["primary_score"],
               "post_primary": post[condition]["primary_score"],
               "improvement": post[condition]["primary_score"] - pre[condition]["primary_score"],
               "pre_lines_cleared": pre[condition]["metrics"]["lines_cleared"],
               "post_lines_cleared": post[condition]["metrics"]["lines_cleared"],
               "lines_improvement": post[condition]["metrics"]["lines_cleared"] - pre[condition]["metrics"]["lines_cleared"],
               "pre_elo": pre_elo[condition], "post_elo": post_elo[condition],
               "training_steps": training[condition]["training_steps"],
               "diagnosed_weakness": weakness if condition == RATING_HISTORY else ""}
        for skill in SKILL_METRICS:
            row[f"pre_{skill}"] = pre[condition]["skill_scores"][skill]
            row[f"post_{skill}"] = post[condition]["skill_scores"][skill]
        rows.append(row)
    return {"replicate": replicate, "seeds": {**seeds, "evaluation": eval_seeds},
            "identical_initial_state": identical, "evaluation_seeds_equal": True,
            "evaluation_training_seeds_disjoint": True, "rows": rows,
            "pre_evaluation": pre, "post_evaluation": post, "training": training,
            "target_skill": target}


def _describe(values: list[float]) -> dict[str, Any]:
    avg, n = mean(values), len(values)
    sd = stdev(values) if n > 1 else float("nan")
    margin = _t_critical_975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": avg, "sd": sd, "ci95": [avg - margin, avg + margin]}


def _wilcoxon(differences: list[float]) -> dict[str, Any]:
    """Exact two-sided signed-rank test; average ranks and zero differences are handled."""
    nonzero = [(abs(x), x > 0) for x in differences if x != 0]
    if not nonzero:
        return {"n_nonzero": 0, "zero_differences": len(differences), "w_plus": 0.0,
                "p_two_sided_exact": 1.0, "method": "exact signed-rank; zeros omitted"}
    ordered = sorted(nonzero)
    ranks, i = [], 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]: j += 1
        average_rank = ((i + 1) + j) / 2
        ranks.extend((average_rank, sign) for _, sign in ordered[i:j]); i = j
    observed2 = int(round(2 * sum(rank for rank, sign in ranks if sign)))
    rank2 = [int(round(2 * rank)) for rank, _ in ranks]
    counts = {0: 1}
    for rank in rank2:
        updated = dict(counts)
        for total, count in counts.items(): updated[total + rank] = updated.get(total + rank, 0) + count
        counts = updated
    total_rank2, outcomes = sum(rank2), 2 ** len(rank2)
    tail = min(observed2, total_rank2 - observed2)
    p = min(1.0, 2 * sum(count for score, count in counts.items() if score <= tail) / outcomes)
    return {"n_nonzero": len(nonzero), "zero_differences": len(differences) - len(nonzero),
            "w_plus": observed2 / 2, "p_two_sided_exact": p,
            "method": "exact signed-rank conditional on absolute ranks; zeros omitted; average tied ranks"}


def summarize(rows: list[dict[str, Any]], targets: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {c: sorted((r for r in rows if r["condition"] == c), key=lambda x: x["replicate"])
              for c in CONDITIONS}
    conditions = {c: {"pre": _describe([r["pre_primary"] for r in g]),
                      "post": _describe([r["post_primary"] for r in g]),
                      "improvement": _describe([r["improvement"] for r in g]),
                      "lines_secondary": {"pre": _describe([r["pre_lines_cleared"] for r in g]),
                                          "post": _describe([r["post_lines_cleared"] for r in g]),
                                          "improvement": _describe([r["lines_improvement"] for r in g])}}
                  for c, g in groups.items()}
    comparisons = {}
    for other in (CONTROL, RATING_ONLY):
        diffs = [h["improvement"] - o["improvement"]
                 for h, o in zip(groups[RATING_HISTORY], groups[other])]
        desc, n = _describe(diffs), len(diffs)
        if desc["sd"] == 0:
            t = 0.0 if desc["mean"] == 0 else math.copysign(float("inf"), desc["mean"])
            p = 1.0 if desc["mean"] == 0 else 0.0
            dz = 0.0 if desc["mean"] == 0 else math.copysign(float("inf"), desc["mean"])
        else:
            t = desc["mean"] / (desc["sd"] / math.sqrt(n))
            p, dz = _t_two_sided_p(abs(t), n - 1), desc["mean"] / desc["sd"]
        comparisons[f"rating_history_minus_{other}"] = {
            "paired_mean_difference": desc["mean"], "ci95": desc["ci95"], "t": t,
            "df": n - 1, "p_two_sided": p, "cohens_dz": dz, "wilcoxon": _wilcoxon(diffs)}
    target_groups = {}
    for weakness in sorted({t["diagnosed_weakness"] for t in targets}):
        group = [t for t in targets if t["diagnosed_weakness"] == weakness]
        target_groups[weakness] = {"n": len(group), "target_metric": group[0]["target_metric"],
                                   "pre": _describe([t["pre_score"] for t in group]),
                                   "post": _describe([t["post_score"] for t in group]),
                                   "improvement": _describe([t["improvement"] for t in group])}
    return {"primary_measure": Hour4Config.primary_measure, "conditions": conditions,
            "paired_comparisons": comparisons,
            "diagnosed_weakness_counts": dict(Counter(t["diagnosed_weakness"] for t in targets)),
            "targeted_learning_by_weakness": target_groups,
            "interpretation_boundary": "Targeted learning is mechanistic/exploratory; general transfer uses the predeclared primary outcome."}


def resolution(results: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [c["successful_placements"] for r in results for phase in
                    (r["pre_evaluation"], r["post_evaluation"]) for ev in phase.values()
                    for c in ev["challenges"]]
    minimum = min(observations)
    return {"n_challenge_observations": len(observations), "minimum_score": minimum,
            "proportion_at_zero": sum(x == 0 for x in observations) / len(observations),
            "proportion_at_observed_minimum": sum(x == minimum for x in observations) / len(observations),
            "number_distinct_values": len(set(observations)),
            "proportion_at_cap": sum(x == 120 for x in observations) / len(observations),
            "observed_range": [minimum, max(observations)]}


def run_experiment(config: Hour4Config, output_dir: Path, phase: str) -> dict[str, Any]:
    started = time.perf_counter()
    results = [run_replicate(config, i) for i in range(config.replicates)]
    elapsed = time.perf_counter() - started
    rows, targets = [x for r in results for x in r["rows"]], [r["target_skill"] for r in results]
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "replicate_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output_dir / "target_skill_results.csv").open("w", newline="") as stream:
        fields = ["replicate", "diagnosed_weakness", "diagnostic_evidence", "tutorial_type",
                  "target_metric", "pre_score", "post_score", "improvement"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r, target in zip(results, targets): writer.writerow({"replicate": r["replicate"], **target,
            "diagnostic_evidence": json.dumps(target["diagnostic_evidence"], sort_keys=True)})
    stats = summarize(rows, targets)
    (output_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output_dir / "challenge_results.jsonl").write_text("".join(json.dumps({
        "replicate": r["replicate"], "seeds": r["seeds"], "pre": r["pre_evaluation"],
        "post": r["post_evaluation"]}, separators=(",", ":")) + "\n" for r in results))
    provenance = {"phase": phase, "config": asdict(config), "replicate_seeds": [r["seeds"] for r in results],
                  "elapsed_seconds": elapsed, "seconds_per_replicate": elapsed / config.replicates,
                  "python": sys.version, "platform": platform.platform(),
                  "git_commit_at_run": subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                      capture_output=True, check=True).stdout.strip(), "resolution": resolution(results),
                  "controls": {"all_initial_states_identical": all(r["identical_initial_state"] for r in results),
                      "all_evaluation_challenges_matched": all(r["evaluation_seeds_equal"] for r in results),
                      "all_evaluation_seeds_disjoint": all(r["evaluation_training_seeds_disjoint"] for r in results),
                      "all_training_budgets_equal": all({x["training_steps"] for x in r["training"].values()} == {config.training_placements} for r in results)}}
    (output_dir / "configuration_seeds_timing_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
