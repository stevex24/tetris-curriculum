from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
from statistics import NormalDist
from typing import Any

from .agent import LearningAgent
from .hour4 import evaluate
from .hour6 import (Hour6Config, _elo, history_material, load_frozen_calibration,
                    replicate_seeds, run_experiment, run_replicate)
from .hour5 import diagnose, measure_profile, normalize_profile
from .tetris import SHAPES, TetrisAdapter
from .training import (CONDITIONS, CONTROL, RATING_HISTORY, RATING_ONLY,
                       control_material, select_rating_only_material, train)


HOUR9_MASTER_SEED = 900_009
HOUR9_REPLICATES = 1_000
PRIOR_MASTER_SEEDS = {300_003, 400_004, 500_005, 500_105, 600_006, 700_007, 700_107}
VALIDATION_REPLICATES = tuple(range(10))
VALIDATION_CORPUS_SEEDS = (910_001, 910_002, 910_003, 910_004)


def hour9_config(replicates: int = HOUR9_REPLICATES) -> Hour6Config:
    """Hour 6/7 configuration with only the prespecified seed and sample size changed."""
    return replace(Hour6Config(), master_seed=HOUR9_MASTER_SEED, replicates=replicates)


def _trained_agents(config: Hour6Config, calibration: Any, replicate: int) -> tuple[dict[str, LearningAgent], dict[str, Any]]:
    """Reconstruct the frozen replicate through training for behavioral validation."""
    seeds = replicate_seeds(config.master_seed, replicate)
    baseline = LearningAgent("baseline", list(config.initial_weights), config.learning_rate,
                             config.temperature, seeds["baseline_agent"])
    baseline_record = train(baseline, control_material(), config.initial_elo,
                            config.baseline_history_placements, seeds["baseline_history"])
    history = [{"history_id": f"hour6-{replicate:03d}", "seed": seeds["baseline_history"],
                "steps": baseline_record["placements"]}]
    diagnosis = diagnose(normalize_profile(
        measure_profile(history, config.initial_elo, config.baseline_history_placements), calibration),
        config.mixed_margin)
    agents = {condition: baseline.clone(condition) for condition in CONDITIONS}
    pre = {condition: evaluate(agent, seeds["evaluation"], config.evaluation_max_placements)
           for condition, agent in agents.items()}
    pre_elo = _elo(pre, config)
    materials = {
        CONTROL: control_material(),
        RATING_ONLY: select_rating_only_material(pre_elo[RATING_ONLY], seeds["tutorial_selection"]),
        RATING_HISTORY: history_material(diagnosis, pre_elo[RATING_HISTORY], seeds["tutorial_selection"]),
    }
    training = {condition: train(agents[condition], materials[condition], pre_elo[condition],
                                 config.training_placements, seeds["training_stream"])
                for condition in CONDITIONS}
    return agents, {"seeds": seeds, "training": training}


def _validation_corpus() -> list[dict[str, Any]]:
    """Predetermined shared states: four fixed streams, eight greedy state transitions each."""
    adapter = TetrisAdapter()
    corpus = []
    for seed in VALIDATION_CORPUS_SEEDS:
        rng, state = Random(seed), adapter.initial_state()
        for step in range(8):
            piece = adapter.sample_environment(rng)
            choices = adapter.legal_actions(state, piece)
            if not choices:
                break
            corpus.append({"corpus_seed": seed, "step": step, "piece": piece,
                           "state": state, "choices": choices})
            # Corpus construction is policy-independent and fixed by action ordering.
            state = choices[(seed + step) % len(choices)].next_state
    return corpus


def _preferred(agent: LearningAgent, choices: list[Any]) -> tuple[int, int]:
    return max(choices, key=lambda choice: sum(w * f for w, f in zip(agent.weights, choice.features))).action


def behavioral_validate(calibration_path: Path, output_path: Path) -> dict[str, Any]:
    config, calibration, corpus = hour9_config(), load_frozen_calibration(calibration_path), _validation_corpus()
    rows = []
    for replicate in VALIDATION_REPLICATES:
        agents, reconstructed = _trained_agents(config, calibration, replicate)
        training = reconstructed["training"]
        weights = {condition: training[condition]["policy_weights_after"] for condition in CONDITIONS}
        pair_weight_differences = {
            f"{a}_vs_{b}": weights[a] != weights[b]
            for a, b in ((CONTROL, RATING_ONLY), (CONTROL, RATING_HISTORY), (RATING_ONLY, RATING_HISTORY))
        }
        preferences = {condition: [_preferred(agents[condition], item["choices"]) for item in corpus]
                       for condition in CONDITIONS}
        pair_action_differences = {
            f"{a}_vs_{b}": sum(x != y for x, y in zip(preferences[a], preferences[b]))
            for a, b in ((CONTROL, RATING_ONLY), (CONTROL, RATING_HISTORY), (RATING_ONLY, RATING_HISTORY))
        }
        # The canonical frozen replicate result is the saved-output side of this check;
        # the independently reconstructed trained agents must reproduce it exactly.
        saved = run_replicate(config, calibration, replicate)["post"]
        replay = {condition: evaluate(agents[condition], reconstructed["seeds"]["evaluation"],
                                      config.evaluation_max_placements) for condition in CONDITIONS}
        rows.append({"replicate": replicate, "post_training_weights": weights,
                     "pair_weight_differences": pair_weight_differences,
                     "pair_action_differences_on_shared_corpus": pair_action_differences,
                     "updated_weights_used_for_preferences": all(
                         preferences[c] == [_preferred(agents[c], item["choices"]) for item in corpus]
                         for c in CONDITIONS),
                     "reconstructed_trajectories_equal_saved_outputs": saved == replay})
    weight_check = all(all(row["pair_weight_differences"].values()) for row in rows)
    policy_check = any(any(count > 0 for count in row["pair_action_differences_on_shared_corpus"].values())
                       for row in rows)
    usage_check = all(row["updated_weights_used_for_preferences"] for row in rows)
    reconstruction_check = all(row["reconstructed_trajectories_equal_saved_outputs"] for row in rows)
    report = {
        "status": "pass" if weight_check and policy_check and usage_check and reconstruction_check else "fail",
        "predetermined_sample": list(VALIDATION_REPLICATES),
        "fixed_shared_state_corpus": {"seeds": list(VALIDATION_CORPUS_SEEDS), "states": len(corpus),
                                      "pieces": list(SHAPES)},
        "criterion": "Every sampled replicate has pairwise-distinct trained weights; the predetermined sample demonstrates at least one preferred-action difference on the shared corpus (individual replicates and pairs need not always differ); preferences use trained weights; and independent reconstruction exactly matches canonical saved outputs.",
        "checks": {"post_training_weights_differ": weight_check,
                   "shared_corpus_preferred_actions_differ": policy_check,
                   "updated_weights_used_by_policy": usage_check,
                   "reconstructed_trajectories_match_saved_outputs": reconstruction_check},
        "replicates": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def time_replicates(calibration_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result = run_experiment(hour9_config(10), calibration_path, output, "timing")
    elapsed = time.perf_counter() - started
    timing = {"timed_replicates": 10, "elapsed_seconds": elapsed,
              "seconds_per_replicate": elapsed / 10,
              "estimated_1000_seconds": elapsed * 100,
              "estimated_1000_minutes": elapsed * 100 / 60,
              "launch_threshold_minutes": 60,
              "within_threshold": elapsed * 100 <= 3600}
    return timing


def _fisher_ci(r: float, n: int) -> list[float] | None:
    if n <= 3 or abs(r) >= 1:
        return None
    z, margin = math.atanh(r), NormalDist().inv_cdf(.975) / math.sqrt(n - 3)
    return [math.tanh(z - margin), math.tanh(z + margin)]


def augment_results(output: Path, hour6_summary: Path, hour7_summary: Path) -> dict[str, Any]:
    summary = json.loads((output / "statistical_summary.json").read_text())
    old = {"hour6": json.loads(hour6_summary.read_text()), "hour7": json.loads(hour7_summary.read_text())}
    comparisons = {}
    for name, current in summary["paired_comparisons"].items():
        current["standard_error"] = abs(current["paired_mean_difference"] / current["t"]) if current["t"] else 0.0
        old_rows = {}
        for hour, old_summary in old.items():
            item = old_summary["paired_comparisons"][name]
            old_rows[hour] = item
        new_width = current["ci95"][1] - current["ci95"][0]
        comparisons[name] = {
            "hour6": old_rows["hour6"], "hour7": old_rows["hour7"], "n1000": current,
            "ci_width": {"hour6": old_rows["hour6"]["ci95"][1] - old_rows["hour6"]["ci95"][0],
                         "hour7": old_rows["hour7"]["ci95"][1] - old_rows["hour7"]["ci95"][0],
                         "n1000": new_width},
        }
        comparisons[name]["n1000_narrower_factor"] = {
            hour: comparisons[name]["ci_width"][hour] / new_width for hour in ("hour6", "hour7")}
    for method in ("pearson", "spearman"):
        item = summary["transfer_exploratory"][method]
        item["ci95_fisher_approximation"] = _fisher_ci(item["r"], item["n"])
    (output / "statistical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    comparison = {"comparisons": comparisons, "interpretation":
                  "Compare estimates, intervals, and magnitudes; significance alone is not educational importance."}
    (output / "hour6_hour7_n1000_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return {"summary": summary, "comparison": comparison}


def write_report(root: Path, timing: dict[str, Any], tests: dict[str, Any], validation: dict[str, Any],
                 augmented: dict[str, Any], reproduction_command: str) -> None:
    s, comparisons = augmented["summary"], augmented["comparison"]["comparisons"]
    lines = ["# Large-sample frozen-design experiment", "", f"Master seed: **{HOUR9_MASTER_SEED}**; matched replicates: **1,000**.",
             f"Tests: **{tests['status']}** ({tests['tests_run']} tests). Behavioral validator: **{validation['status']}**.",
             f"Measured experiment runtime: **{timing['experiment_elapsed_seconds']:.3f} seconds**. Ten-replicate estimate: {timing['estimated_1000_minutes']:.2f} minutes.", "",
             "## General performance", "", "| Condition | Pre | Post | Improvement | 95% CI |", "|---|---:|---:|---:|---:|"]
    for condition, value in s["conditions"].items():
        lines.append(f"| {condition} | {value['pre']['mean']:.3f} | {value['post']['mean']:.3f} | {value['improvement']['mean']:.3f} | [{value['improvement']['ci95'][0]:.3f}, {value['improvement']['ci95'][1]:.3f}] |")
    lines += ["", "## Primary paired comparisons", "", "| Comparison | Mean additional successful pieces/challenge | SE | 95% CI | t(df) | p | dz | Wilcoxon p |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in s["paired_comparisons"].items():
        lines.append(f"| {name} | {item['paired_mean_difference']:.4f} | {item['standard_error']:.4f} | [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}] | {item['t']:.3f} ({item['df']}) | {item['p_two_sided']:.6g} | {item['cohens_dz']:.3f} | {item['wilcoxon']['p_two_sided_exact']:.6g} |")
    lines += ["", "## Hour 6 / Hour 7 / n=1000 and precision", "", "| Comparison | Hour 6 | Hour 7 | n=1000 | CI narrower vs H6 / H7 |", "|---|---:|---:|---:|---:|"]
    for name, value in comparisons.items():
        lines.append(f"| {name} | {value['hour6']['paired_mean_difference']:.3f} | {value['hour7']['paired_mean_difference']:.3f} | {value['n1000']['paired_mean_difference']:.3f} | {value['n1000_narrower_factor']['hour6']:.2f}x / {value['n1000_narrower_factor']['hour7']:.2f}x |")
    lines += ["", "The history-minus-control estimate is essentially zero and intermediate between Hour 6's positive estimate and Hour 7's negative estimate. The history-minus-rating estimate is slightly reversed relative to both prior positive estimates. The much narrower intervals show that neither apparent general-performance advantage persisted with high precision."]
    d = s["diagnosis"]
    lines += ["", "## Diagnosis distribution", "", f"Primary: {json.dumps(d['primary_counts'], sort_keys=True)}. Confidence: {json.dumps(d['confidence_counts'], sort_keys=True)}. Mixed: {json.dumps(d['mixed_counts'], sort_keys=True)}.",
              f"Tutorial mapping: {json.dumps(s['tutorial_mapping'], sort_keys=True)}.", "", "## Target skills", "",
              f"Overall standardized target-skill change: {s['targeted_learning']['overall_standardized_improvement']['mean']:.4f} (95% CI {s['targeted_learning']['overall_standardized_improvement']['ci95']}).", "",
              "| Target | n | Pre | Post | Improvement [95% CI] |", "|---|---:|---:|---:|---:|"]
    for target, value in s["targeted_learning"]["by_weakness"].items():
        lines.append(f"| {target} | {value['pre']['n']} | {value['pre']['mean']:.3f} | {value['post']['mean']:.3f} | {value['improvement']['mean']:.3f} [{value['improvement']['ci95'][0]:.3f}, {value['improvement']['ci95'][1]:.3f}] |")
    lines += ["", "## Exploratory transfer", ""]
    for method, value in s["transfer_exploratory"].items():
        lines.append(f"{method.title()}: r={value['r']:.4f}, 95% Fisher CI={value['ci95_fisher_approximation']}, p={value['p_two_sided']:.6g}. Exploratory; no causal inference.")
    lines += ["", "## Secondary lines cleared", "", "| Condition | Pre | Post | Improvement | Zero proportion pre/post |", "|---|---:|---:|---:|---:|"]
    for condition, value in s["conditions"].items():
        x = value["lines_secondary"]
        lines.append(f"| {condition} | {x['pre']['mean']:.4f} | {x['post']['mean']:.4f} | {x['improvement']['mean']:.4f} | {x['zero_proportion_pre']:.3f} / {x['zero_proportion_post']:.3f} |")
    lines += ["", "## Interpretation", "", "Within this frozen simulation, the large experiment falsifies the apparent history-aware general-performance advantage as a stable effect of educational importance: history-control is essentially zero, and history-rating is a tiny reversal. This conclusion follows from magnitudes and narrow uncertainty intervals, not from statistical significance alone.",
              "", "The hole-management subgroup has a small positive target-skill estimate whose CI excludes zero; the overall standardized target effect and the height and surface subgroup intervals include zero. This is limited evidence for one nominal skill, not evidence that the tutorial system broadly teaches its intended skills.",
              "", "The positive target/general correlations are exploratory associations and do not establish causal transfer. The frozen simulation does not establish real-player effectiveness, causal educational mechanisms, or generalization beyond this implementation.",
              "", "No mechanism, tutorial board, reward, policy weight, calibration, or analysis was redesigned after observing results.", "", "## Exact reproduction command", "", "```bash", reproduction_command, "```", ""]
    (root / "final_report.md").write_text("\n".join(lines))
