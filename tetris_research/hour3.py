from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any, Iterable

from .agent import LearningAgent
from .elo import EloRatings
from .tetris import TetrisAdapter
from .training import (CONDITIONS, CONTROL, RATING_HISTORY, RATING_ONLY, control_material,
                       select_history_material, select_rating_only_material, train)


@dataclass(frozen=True)
class PilotConfig:
    master_seed: int = 300_003
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
    primary_measure: str = "mean lines cleared per evaluation challenge"


def replicate_seeds(master_seed: int, replicate: int) -> dict[str, Any]:
    """Derive reproducible, domain-separated seeds without shared mutable RNGs."""
    def derive(label: str) -> int:
        digest = hashlib.sha256(f"hour3:{master_seed}:{replicate}:{label}".encode()).digest()
        return int.from_bytes(digest[:8], "big") & 0x7fffffff
    return {
        "baseline_agent": derive("baseline-agent"),
        "baseline_history": derive("baseline-history"),
        "tutorial_selection": derive("tutorial-selection"),
        "training_stream": derive("training-stream"),
        "evaluation": [derive(f"evaluation-{i}") for i in range(64)],
    }


def _evaluate_challenge(agent: LearningAgent, seed: int, max_placements: int) -> dict[str, float]:
    adapter, environment_rng = TetrisAdapter(), Random(seed)
    state, lines, placements, peak_height = adapter.initial_state(), 0, 0, 0.0
    final = {"holes": 0.0, "max_height": 0.0, "bumpiness": 0.0}
    while placements < max_placements:
        choices = adapter.legal_actions(state, adapter.sample_environment(environment_rng))
        if not choices:
            break
        choice = agent.choose(choices, learn=False)
        state, placements = choice.next_state, placements + 1
        lines += int(choice.info["lines_cleared"])
        final = choice.raw_features
        peak_height = max(peak_height, float(final["max_height"]))
    return {"seed": seed, "lines_cleared": float(lines),
            "successful_placements": float(placements), "survived": float(placements == max_placements),
            "holes": float(final["holes"]), "maximum_stack_height": peak_height,
            "bumpiness": float(final["bumpiness"])}


def evaluate(agent: LearningAgent, seeds: Iterable[int], max_placements: int) -> dict[str, Any]:
    """Evaluate a disposable clone, leaving every aspect of the supplied agent untouched."""
    evaluation_agent = agent.clone(f"{agent.name}_evaluation_copy")
    challenges = [_evaluate_challenge(evaluation_agent, seed, max_placements) for seed in seeds]
    metrics = {key: mean(row[key] for row in challenges)
               for key in ("lines_cleared", "successful_placements", "survived", "holes",
                           "maximum_stack_height", "bumpiness")}
    return {"challenges": challenges, "metrics": metrics,
            "primary_score": metrics["lines_cleared"]}


def _elo_from_evaluations(evaluations: dict[str, dict[str, Any]], config: PilotConfig) -> dict[str, float]:
    ratings = EloRatings(config.initial_elo, config.elo_k)
    # Fixed, declared round-robin order. Each game is one matched unseen challenge.
    pairs = ((CONTROL, RATING_ONLY), (CONTROL, RATING_HISTORY), (RATING_ONLY, RATING_HISTORY))
    for challenge_index in range(config.evaluation_challenges):
        for a, b in pairs:
            score_a = evaluations[a]["challenges"][challenge_index]["lines_cleared"]
            score_b = evaluations[b]["challenges"][challenge_index]["lines_cleared"]
            outcome = 1.0 if score_a > score_b else (0.0 if score_a < score_b else 0.5)
            ratings.update(a, b, outcome)
    return {condition: ratings.rating(condition) for condition in CONDITIONS}


def run_replicate(config: PilotConfig, replicate: int) -> dict[str, Any]:
    seeds = replicate_seeds(config.master_seed, replicate)
    eval_seeds = seeds["evaluation"][:config.evaluation_challenges]
    training_seeds = {seeds["baseline_history"], seeds["tutorial_selection"], seeds["training_stream"]}
    if training_seeds.intersection(eval_seeds):
        raise AssertionError("evaluation and training/tutorial seeds overlap")

    baseline = LearningAgent("baseline", list(config.initial_weights), config.learning_rate,
                             config.temperature, seeds["baseline_agent"])
    baseline_record = train(baseline, control_material(), config.initial_elo,
                            config.baseline_history_placements, seeds["baseline_history"])
    # Convert existing placement logs into the observable schema used by Hour 2 diagnosis.
    prior_history = [{"agent": baseline.name, "seed": seeds["baseline_history"],
                      "steps": baseline_record["placements"]}]
    agents = {condition: baseline.clone(condition) for condition in CONDITIONS}
    initial_states = {condition: {"weights": agent.weights.copy(), "elo": config.initial_elo,
                                  "rng_state": repr(agent._rng.getstate()),
                                  "games_learned": agent.games_learned,
                                  "prior_history": prior_history}
                      for condition, agent in agents.items()}
    identical_initial_state = len({json.dumps(value, sort_keys=True)
                                   for value in initial_states.values()}) == 1
    if not identical_initial_state:
        raise AssertionError("condition clones are not experimentally identical")

    pre = {condition: evaluate(agent, eval_seeds, config.evaluation_max_placements)
           for condition, agent in agents.items()}
    pre_elo = _elo_from_evaluations(pre, config)
    materials = {
        CONTROL: control_material(),
        RATING_ONLY: select_rating_only_material(pre_elo[RATING_ONLY], seeds["tutorial_selection"]),
        RATING_HISTORY: select_history_material(pre_elo[RATING_HISTORY], prior_history,
                                                seeds["tutorial_selection"]),
    }
    training = {condition: train(agents[condition], materials[condition], pre_elo[condition],
                                 config.training_placements, seeds["training_stream"])
                for condition in CONDITIONS}
    if {record["training_steps"] for record in training.values()} != {config.training_placements}:
        raise AssertionError("unequal training exposure")
    post = {condition: evaluate(agent, eval_seeds, config.evaluation_max_placements)
            for condition, agent in agents.items()}
    post_elo = _elo_from_evaluations(post, config)

    rows = []
    for condition in CONDITIONS:
        row = {"replicate": replicate, "condition": condition,
               "pre_primary": pre[condition]["primary_score"],
               "post_primary": post[condition]["primary_score"],
               "improvement": post[condition]["primary_score"] - pre[condition]["primary_score"],
               "pre_elo": pre_elo[condition], "post_elo": post_elo[condition],
               "delta_elo": post_elo[condition] - pre_elo[condition],
               "training_steps": training[condition]["training_steps"],
               "diagnosed_weakness": training[condition]["diagnosed_weakness"] or ""}
        for phase_name, phase in (("pre", pre[condition]), ("post", post[condition])):
            for metric, value in phase["metrics"].items():
                row[f"{phase_name}_{metric}"] = value
        rows.append(row)
    return {"replicate": replicate, "seeds": {**seeds, "evaluation": eval_seeds},
            "identical_initial_state": identical_initial_state,
            "evaluation_seeds_equal": True, "evaluation_training_seeds_disjoint": True,
            "rows": rows, "pre_evaluation": pre, "post_evaluation": post,
            "training": training, "initial_states": initial_states}


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam, c, d, h = a + b, a + 1, a - 1, 1.0, 1.0, 1.0
    d = 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1.0 + aa * d, 1.0 + aa / c
        d, c = (1e-30 if abs(d) < 1e-30 else d), (1e-30 if abs(c) < 1e-30 else c)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1.0 + aa * d, 1.0 + aa / c
        d, c = (1e-30 if abs(d) < 1e-30 else d), (1e-30 if abs(c) < 1e-30 else c)
        d = 1.0 / d
        delta, h = d * c, h * d * c
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    return (front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2)
            else 1 - front * _betacf(b, a, 1 - x) / b)


def _t_two_sided_p(t_value: float, df: int) -> float:
    return _regularized_beta(df / (df + t_value * t_value), df / 2, 0.5)


def _t_critical_975(df: int) -> float:
    low, high = 0.0, 20.0
    for _ in range(80):
        mid = (low + high) / 2
        if _t_two_sided_p(mid, df) > 0.05: low = mid
        else: high = mid
    return (low + high) / 2


def _describe(values: list[float]) -> dict[str, Any]:
    n, avg = len(values), mean(values)
    sd = stdev(values) if n > 1 else float("nan")
    margin = _t_critical_975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": avg, "sd": sd, "ci95": [avg - margin, avg + margin]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {condition: sorted((r for r in rows if r["condition"] == condition),
                                      key=lambda r: r["replicate"])
                    for condition in CONDITIONS}
    conditions = {}
    for condition, group in by_condition.items():
        conditions[condition] = {
            "pre": _describe([r["pre_primary"] for r in group]),
            "post": _describe([r["post_primary"] for r in group]),
            "improvement": _describe([r["improvement"] for r in group]),
            "elo": {"mean_pre": mean(r["pre_elo"] for r in group),
                    "mean_post": mean(r["post_elo"] for r in group),
                    "mean_delta": mean(r["delta_elo"] for r in group)},
        }
    comparisons = {}
    for other in (CONTROL, RATING_ONLY):
        differences = [h["improvement"] - o["improvement"]
                       for h, o in zip(by_condition[RATING_HISTORY], by_condition[other])]
        desc, n = _describe(differences), len(differences)
        if desc["sd"] == 0:
            t_value = 0.0 if desc["mean"] == 0 else math.copysign(float("inf"), desc["mean"])
            p_value = 1.0 if desc["mean"] == 0 else 0.0
            dz = 0.0 if desc["mean"] == 0 else math.copysign(float("inf"), desc["mean"])
        else:
            t_value = desc["mean"] / (desc["sd"] / math.sqrt(n))
            p_value, dz = _t_two_sided_p(abs(t_value), n - 1), desc["mean"] / desc["sd"]
        comparisons[f"rating_history_minus_{other}"] = {
            "paired_mean_difference": desc["mean"], "ci95": desc["ci95"],
            "t": t_value, "df": n - 1, "p_two_sided": p_value, "cohens_dz": dz}
    return {"primary_measure": "mean lines cleared per evaluation challenge",
            "conditions": conditions, "paired_comparisons": comparisons,
            "caution": "Pilot inference is exploratory; paired t assumptions should be checked, especially with small n."}


def run_experiment(config: PilotConfig, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    replicate_results = [run_replicate(config, i) for i in range(config.replicates)]
    elapsed = time.perf_counter() - started
    rows = [row for result in replicate_results for row in result["rows"]]
    output_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = list(rows[0])
    with (output_dir / "replicate_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "statistics.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")
    provenance = {"config": asdict(config), "replicate_seeds": [r["seeds"] for r in replicate_results],
                  "elapsed_seconds": elapsed, "seconds_per_replicate": elapsed / config.replicates,
                  "python": sys.version, "platform": platform.platform(),
                  "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                                                capture_output=True, check=True).stdout.strip(),
                  "controls": {"all_initial_states_identical": all(r["identical_initial_state"] for r in replicate_results),
                               "all_evaluation_challenges_matched": all(r["evaluation_seeds_equal"] for r in replicate_results),
                               "all_evaluation_seeds_disjoint": all(r["evaluation_training_seeds_disjoint"] for r in replicate_results)}}
    (output_dir / "configuration_and_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output_dir / "challenge_results.jsonl").write_text("".join(json.dumps({
        "replicate": r["replicate"], "seeds": r["seeds"], "pre": r["pre_evaluation"],
        "post": r["post_evaluation"]}, separators=(",", ":")) + "\n" for r in replicate_results))
    return provenance
