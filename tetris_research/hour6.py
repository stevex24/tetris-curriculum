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
from .elo import EloRatings
from .hour3 import _t_critical_975, _t_two_sided_p
from .hour4 import _wilcoxon, evaluate
from .hour5 import (CALIBRATION_SEED, Calibration, diagnose, measure_profile,
                    normalize_profile, tutorial_for)
from .training import (CONDITIONS, CONTROL, RATING_HISTORY, RATING_ONLY, TrainingMaterial,
                       _history_board, control_material, select_rating_only_material, train)

HOUR6_MASTER_SEED = 600_006
MIXED_MARGIN = 0.35
TARGET_METRICS = {"hole_management": "hole_avoidance",
                  "height_management": "stack_height_management",
                  "surface_management": "surface_smoothness"}
TUTORIAL_TYPES = {"hole_management": "hole_avoidance",
                  "height_management": "stack_height",
                  "surface_management": "bumpiness"}
PRIOR_EXPERIMENT_MASTER_SEEDS = {300_003, 400_004, 500_005, 500_105}


@dataclass(frozen=True)
class Hour6Config:
    master_seed: int = HOUR6_MASTER_SEED
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
    mixed_margin: float = MIXED_MARGIN
    primary_measure: str = "mean successful placements per held-out challenge"


def load_frozen_calibration(path: Path) -> Calibration:
    raw = json.loads(path.read_text())
    raw["dimensions"] = tuple(raw["dimensions"])
    raw["history_ids"] = tuple(raw["history_ids"])
    calibration = Calibration(**raw)
    if calibration.seed != CALIBRATION_SEED or calibration.placements_per_history != 40:
        raise ValueError("not the committed Hour 5 calibration")
    return calibration


def replicate_seeds(master_seed: int, replicate: int) -> dict[str, Any]:
    def derive(label: str) -> int:
        digest = hashlib.sha256(f"hour6:{master_seed}:{replicate}:{label}".encode()).digest()
        return int.from_bytes(digest[:8], "big") & 0x7fffffff
    return {"baseline_agent": derive("baseline-agent"),
            "baseline_history": derive("baseline-history"),
            "tutorial_selection": derive("tutorial-selection"),
            "training_stream": derive("training-stream"),
            "evaluation": [derive(f"evaluation-{i}") for i in range(6)]}


def _fingerprint(agent: LearningAgent, elo: float, history: Any) -> str:
    state = dict(as_student_agent(agent).serialize_state())
    state.pop("agent_id", None)
    return json.dumps({"agent_state": state, "elo": elo, "history": history}, sort_keys=True)


def _elo(evaluations: dict[str, dict[str, Any]], config: Hour6Config) -> dict[str, float]:
    ratings = EloRatings(config.initial_elo, config.elo_k)
    for index in range(config.evaluation_challenges):
        for a, b in ((CONTROL, RATING_ONLY), (CONTROL, RATING_HISTORY),
                     (RATING_ONLY, RATING_HISTORY)):
            x, y = (evaluations[a]["challenges"][index]["lines_cleared"],
                    evaluations[b]["challenges"][index]["lines_cleared"])
            ratings.update(a, b, 1.0 if x > y else (0.0 if x < y else 0.5))
    return {condition: ratings.rating(condition) for condition in CONDITIONS}


def history_material(diagnosis: Any, rating: float, seed: int) -> TrainingMaterial:
    tutorial = tutorial_for(diagnosis, rating, seed)
    expected = TUTORIAL_TYPES[diagnosis.primary]
    if tutorial["tutorial_type"] != expected:
        raise AssertionError("Hour 5 diagnosis/tutorial mapping failure")
    return TrainingMaterial(RATING_HISTORY, _history_board(expected, tutorial["difficulty"], seed),
                            tutorial["difficulty"], diagnosis.primary,
                            dict(diagnosis.normalized_weakness), tutorial["rationale"], seed)


def run_replicate(config: Hour6Config, calibration: Calibration, replicate: int) -> dict[str, Any]:
    seeds = replicate_seeds(config.master_seed, replicate)
    used_training = {seeds["baseline_history"], seeds["tutorial_selection"], seeds["training_stream"]}
    if used_training & set(seeds["evaluation"]):
        raise AssertionError("held-out evaluation seed overlaps diagnosis or training")
    baseline = LearningAgent("baseline", list(config.initial_weights), config.learning_rate,
                             config.temperature, seeds["baseline_agent"])
    baseline_record = train(baseline, control_material(), config.initial_elo,
                            config.baseline_history_placements, seeds["baseline_history"])
    history = [{"history_id": f"hour6-{replicate:03d}", "seed": seeds["baseline_history"],
                "steps": baseline_record["placements"]}]
    raw_profile = measure_profile(history, config.initial_elo, config.baseline_history_placements)
    profile = normalize_profile(raw_profile, calibration)
    diagnosis = diagnose(profile, config.mixed_margin)
    agents = {condition: baseline.clone(condition) for condition in CONDITIONS}
    identical = len({_fingerprint(agent, config.initial_elo, history) for agent in agents.values()}) == 1
    if not identical:
        raise AssertionError("matched clones differ before treatment")
    pre = {condition: evaluate(agent, seeds["evaluation"], config.evaluation_max_placements)
           for condition, agent in agents.items()}
    pre_elo = _elo(pre, config)
    materials = {CONTROL: control_material(),
                 RATING_ONLY: select_rating_only_material(pre_elo[RATING_ONLY], seeds["tutorial_selection"]),
                 RATING_HISTORY: history_material(diagnosis, pre_elo[RATING_HISTORY],
                                                  seeds["tutorial_selection"])}
    training = {condition: train(agents[condition], materials[condition], pre_elo[condition],
                                 config.training_placements, seeds["training_stream"])
                for condition in CONDITIONS}
    if {row["training_steps"] for row in training.values()} != {config.training_placements}:
        raise AssertionError("training budgets differ")
    post = {condition: evaluate(agent, seeds["evaluation"], config.evaluation_max_placements)
            for condition, agent in agents.items()}
    metric = TARGET_METRICS[diagnosis.primary]
    target_pre = pre[RATING_HISTORY]["skill_scores"][metric]
    target_post = post[RATING_HISTORY]["skill_scores"][metric]
    raw_improvement = target_post - target_pre
    standardized = raw_improvement / calibration.sample_sds[diagnosis.primary]
    profile_row = {"replicate": replicate, **{f"raw_{k}": v for k, v in asdict(raw_profile).items()
                                              if k != "normalized_weakness"},
                   **{f"z_{k}": v for k, v in profile.normalized_weakness.items()},
                   "primary_weakness": diagnosis.primary, "secondary_weakness": diagnosis.secondary,
                   "top_two_margin": diagnosis.margin, "confidence": diagnosis.confidence,
                   "mixed": diagnosis.mixed, "selected_tutorial": TUTORIAL_TYPES[diagnosis.primary]}
    target = {"replicate": replicate, "diagnosed_weakness": diagnosis.primary,
              "target_metric": metric, "pre_score": target_pre, "post_score": target_post,
              "improvement": raw_improvement, "standardized_improvement": standardized,
              "general_improvement": post[RATING_HISTORY]["primary_score"] - pre[RATING_HISTORY]["primary_score"],
              "confidence": diagnosis.confidence, "mixed": diagnosis.mixed}
    rows = []
    for condition in CONDITIONS:
        row = {"replicate": replicate, "condition": condition,
               "pre_primary": pre[condition]["primary_score"], "post_primary": post[condition]["primary_score"],
               "improvement": post[condition]["primary_score"] - pre[condition]["primary_score"],
               "pre_lines_cleared": pre[condition]["metrics"]["lines_cleared"],
               "post_lines_cleared": post[condition]["metrics"]["lines_cleared"],
               "lines_improvement": post[condition]["metrics"]["lines_cleared"] - pre[condition]["metrics"]["lines_cleared"],
               "pre_elo": pre_elo[condition], "training_steps": training[condition]["training_steps"],
               "tutorial": ("ordinary_practice" if condition == CONTROL else
                            (materials[condition].diagnosed_weakness or "rating_tier_board"))}
        for skill, value in pre[condition]["skill_scores"].items(): row[f"pre_{skill}"] = value
        for skill, value in post[condition]["skill_scores"].items(): row[f"post_{skill}"] = value
        rows.append(row)
    return {"replicate": replicate, "seeds": seeds, "identical_initial_state": identical,
            "profile": profile_row, "rows": rows, "target": target, "pre": pre, "post": post,
            "training": training, "tutorials_differ": TUTORIAL_TYPES[diagnosis.primary] != "rating_tier_board"}


def _describe(values: list[float]) -> dict[str, Any]:
    n, avg = len(values), mean(values)
    sd = stdev(values) if n > 1 else None
    margin = _t_critical_975(n - 1) * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": avg, "sd": sd, "ci95": [avg - margin, avg + margin] if margin is not None else None}


def _paired(diffs: list[float]) -> dict[str, Any]:
    desc, n = _describe(diffs), len(diffs)
    if desc["sd"] == 0:
        t = 0.0 if desc["mean"] == 0 else math.copysign(float("inf"), desc["mean"])
        p = 1.0 if desc["mean"] == 0 else 0.0
        dz = 0.0 if desc["mean"] == 0 else math.copysign(float("inf"), desc["mean"])
    else:
        t = desc["mean"] / (desc["sd"] / math.sqrt(n))
        p, dz = _t_two_sided_p(abs(t), n - 1), desc["mean"] / desc["sd"]
    return {"paired_mean_difference": desc["mean"], "ci95": desc["ci95"], "t": t,
            "df": n - 1, "p_two_sided": p, "cohens_dz": dz, "wilcoxon": _wilcoxon(diffs)}


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda row: row[1]); result = [0.0] * len(values); i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]: j += 1
        rank = ((i + 1) + j) / 2
        for original, _ in ordered[i:j]: result[original] = rank
        i = j
    return result


def _correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    mx, my = mean(x), mean(y)
    denominator = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / denominator if denominator else 0.0
    if len(x) > 2 and abs(r) < 1:
        t = r * math.sqrt((len(x) - 2) / (1 - r * r)); p = _t_two_sided_p(abs(t), len(x) - 2)
    else: t, p = (math.copysign(float("inf"), r), 0.0) if abs(r) == 1 else (0.0, 1.0)
    return {"n": len(x), "r": r, "t": t, "df": len(x) - 2, "p_two_sided": p}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for result in results for row in result["rows"]]
    groups = {c: sorted((r for r in rows if r["condition"] == c), key=lambda r: r["replicate"])
              for c in CONDITIONS}
    conditions = {c: {"pre": _describe([r["pre_primary"] for r in g]),
                      "post": _describe([r["post_primary"] for r in g]),
                      "improvement": _describe([r["improvement"] for r in g]),
                      "lines_secondary": {"pre": _describe([r["pre_lines_cleared"] for r in g]),
                                          "post": _describe([r["post_lines_cleared"] for r in g]),
                                          "improvement": _describe([r["lines_improvement"] for r in g]),
                                          "zero_proportion_pre": mean([r["pre_lines_cleared"] == 0 for r in g]),
                                          "zero_proportion_post": mean([r["post_lines_cleared"] == 0 for r in g])}}
                  for c, g in groups.items()}
    comparisons = {f"rating_history_minus_{other}": _paired([
        h["improvement"] - o["improvement"] for h, o in zip(groups[RATING_HISTORY], groups[other])])
        for other in (CONTROL, RATING_ONLY)}
    targets = [r["target"] for r in results]
    target_groups = {weakness: {"target_metric": TARGET_METRICS[weakness],
                                "pre": _describe([t["pre_score"] for t in targets if t["diagnosed_weakness"] == weakness]),
                                "post": _describe([t["post_score"] for t in targets if t["diagnosed_weakness"] == weakness]),
                                "improvement": _describe([t["improvement"] for t in targets if t["diagnosed_weakness"] == weakness])}
                     for weakness in sorted({t["diagnosed_weakness"] for t in targets})}
    profiles = [r["profile"] for r in results]
    x = [t["standardized_improvement"] for t in targets]; y = [t["general_improvement"] for t in targets]
    mapping = Counter((p["primary_weakness"], p["selected_tutorial"]) for p in profiles)
    return {"primary_measure": Hour6Config.primary_measure, "conditions": conditions,
            "paired_comparisons": comparisons,
            "targeted_learning": {"overall_standardized_improvement": _describe(x), "by_weakness": target_groups},
            "transfer_exploratory": {"pearson": _correlation(x, y),
                                     "spearman": _correlation(_ranks(x), _ranks(y))},
            "diagnosis": {"primary_counts": dict(Counter(p["primary_weakness"] for p in profiles)),
                          "secondary_counts": dict(Counter(p["secondary_weakness"] for p in profiles)),
                          "confidence_counts": dict(Counter(p["confidence"] for p in profiles)),
                          "mixed_counts": dict(Counter(str(p["mixed"]).lower() for p in profiles)),
                          "z_score_distributions": {d: _describe([p[f"z_{d}"] for p in profiles])
                                                    for d in TARGET_METRICS}},
            "tutorial_mapping": {f"{d} -> {t}": n for (d, t), n in sorted(mapping.items())},
            "history_vs_rating_tutorial_different": {"n": sum(r["tutorials_differ"] for r in results),
                                                      "total": len(results)}}


def resolution(results: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [c["successful_placements"] for r in results for phase in (r["pre"], r["post"])
                    for evaluation in phase.values() for c in evaluation["challenges"]]
    targets = [t for r in results for phase in (r["pre"], r["post"])
               for t in [phase[RATING_HISTORY]["skill_scores"][TARGET_METRICS[r["profile"]["primary_weakness"]]]]]
    return {"general": {"n": len(observations), "range": [min(observations), max(observations)],
                        "distinct_values": len(set(observations)),
                        "zero_proportion": mean([v == 0 for v in observations]),
                        "cap_proportion": mean([v == 120 for v in observations])},
            "target": {"n": len(targets), "range": [min(targets), max(targets)],
                       "distinct_values": len(set(targets))}}


def write_experiment(results: list[dict[str, Any]], config: Hour6Config, calibration_path: Path,
                     output: Path, phase: str, elapsed: float) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    rows, profiles, targets = ([x for r in results for x in r["rows"]],
                               [r["profile"] for r in results], [r["target"] for r in results])
    for name, data in (("replicate_results.csv", rows), ("diagnosis_profiles.csv", profiles),
                       ("target_skill_results.csv", targets)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    with (output / "challenge_results.jsonl").open("w") as stream:
        for r in results:
            stream.write(json.dumps({"replicate": r["replicate"], "seeds": r["seeds"],
                                     "pre": r["pre"], "post": r["post"]}, separators=(",", ":")) + "\n")
    summary = summarize(results)
    (output / "statistical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "transfer_analysis.json").write_text(json.dumps(summary["transfer_exploratory"], indent=2) + "\n")
    (output / "tutorial_mapping.json").write_text(json.dumps({"frequency_table": summary["tutorial_mapping"],
        "history_vs_rating_tutorial_different": summary["history_vs_rating_tutorial_different"]}, indent=2) + "\n")
    provenance = {"phase": phase, "config": asdict(config), "master_seed": config.master_seed,
                  "calibration": {"path": str(calibration_path), "seed": CALIBRATION_SEED,
                                  "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                                  "reuse": "exact committed Hour 5 parameters; not regenerated"},
                  "replicate_seeds": [r["seeds"] for r in results], "elapsed_seconds": elapsed,
                  "python": sys.version, "platform": platform.platform(),
                  "git_commit_at_run": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                        text=True, check=True).stdout.strip(),
                  "resolution": resolution(results),
                  "controls": {"identical_initial_states": all(r["identical_initial_state"] for r in results),
                               "equal_training_budgets": all({x["training_steps"] for x in r["training"].values()} == {config.training_placements} for r in results),
                               "matched_evaluation_seeds": True, "replicate_seed_records_unique": len({json.dumps(r["seeds"], sort_keys=True) for r in results}) == len(results)}}
    (output / "configuration_seeds_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"summary": summary, "provenance": provenance}


def run_experiment(config: Hour6Config, calibration_path: Path, output: Path, phase: str) -> dict[str, Any]:
    calibration = load_frozen_calibration(calibration_path); started = time.perf_counter()
    results = [run_replicate(config, calibration, i) for i in range(config.replicates)]
    return write_experiment(results, config, calibration_path, output, phase, time.perf_counter() - started)
