from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

from tetris_research.hour5 import (CALIBRATION_SEED, DIAGNOSTIC_DIMENSIONS,
                                   DIAGNOSTIC_POPULATION_SEED, diagnose, fit_calibration,
                                   measure_profile, natural_profiles, normalize_profile,
                                   pearson_matrix, synthetic_history, tutorial_for, write_json)


def run(output: Path, calibration_count: int = 80, diagnostic_count: int = 50) -> None:
    output.mkdir(parents=True, exist_ok=False)
    calibration_rows = natural_profiles(CALIBRATION_SEED, "calibration", calibration_count)
    calibration = fit_calibration([(i, p) for i, p, _ in calibration_rows], CALIBRATION_SEED,
                                  "naturally generated baseline agents; calibration split",
                                  40)
    write_json(output / "calibration_parameters.json", asdict(calibration))
    with (output / "calibration_profiles.csv").open("w", newline="") as stream:
        fields = ["history_id", "split", "master_seed", "ability_elo", "mean_holes", "mean_max_height",
                  "mean_bumpiness", "line_clear_rate", "lines_cleared", "placements_observed"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for identifier, profile, _ in calibration_rows:
            row = asdict(profile); row.pop("normalized_weakness")
            writer.writerow({"history_id": identifier, "split": "calibration",
                             "master_seed": CALIBRATION_SEED, **row})

    diagnostic_rows = natural_profiles(DIAGNOSTIC_POPULATION_SEED, "diagnostic", diagnostic_count)
    natural = []
    for identifier, raw, _ in diagnostic_rows:
        profile = normalize_profile(raw, calibration); result = diagnose(profile)
        natural.append({"history_id": identifier, "profile": asdict(profile),
                        "diagnosis": asdict(result)})
    raw_profiles = [row[1] for row in diagnostic_rows]
    variation = {name: {"mean": mean([getattr(p, name) for p in raw_profiles]),
                        "sample_sd": stdev([getattr(p, name) for p in raw_profiles]),
                        "min": min(getattr(p, name) for p in raw_profiles),
                        "max": max(getattr(p, name) for p in raw_profiles)}
                 for name in ("mean_holes", "mean_max_height", "mean_bumpiness", "line_clear_rate")}
    signatures = Counter(tuple(1 if item["profile"]["normalized_weakness"][key] >= 0 else 0
                               for key in DIAGNOSTIC_DIMENSIONS) for item in natural)
    write_json(output / "diagnostic_population.json", {
        "seed": DIAGNOSTIC_POPULATION_SEED, "split": "diagnostic",
        "disjoint_from_calibration": set(calibration.history_ids).isdisjoint(x[0] for x in diagnostic_rows),
        "variation": variation, "correlations": pearson_matrix(raw_profiles),
        "diagnosis_counts": Counter(x["diagnosis"]["primary"] for x in natural),
        "above_below_calibration_mean_patterns": {str(k): v for k, v in signatures.items()},
        "profiles": natural,
        "interpretation": "Observable variation and several relative patterns exist, but agents share one policy/learning process; seed-driven variation is not evidence of stable learner types."})

    construct_kinds = ["hole_management", "height_management", "surface_management",
                       "mixed_hole_height", "balanced"]
    constructs = []
    for index, kind in enumerate(construct_kinds):
        history = synthetic_history(kind, 510_000 + index)
        profile = normalize_profile(measure_profile(history, 1500.0), calibration)
        result = diagnose(profile)
        constructs.append({"known_construct": kind, "profile": asdict(profile),
                           "diagnosis": asdict(result),
                           "tutorial": tutorial_for(result, profile.ability_elo, 520_000 + index)})
    expected = DIAGNOSTIC_DIMENSIONS
    confusion = {known: {diagnosed: sum(x["known_construct"] == known and
                                       x["diagnosis"]["primary"] == diagnosed for x in constructs)
                         for diagnosed in expected} for known in expected}
    write_json(output / "construct_validation_and_demo.json", {
        "construction_frozen_in": "tetris_research.hour5.synthetic_history",
        "classifier_received_labels": False, "confusion_matrix": confusion,
        "line_efficiency_status": "observable but under-resolved; excluded from primary diagnosis",
        "examples": constructs})
    print(json.dumps({"output": str(output), "calibration_n": calibration_count,
                      "diagnostic_n": diagnostic_count, "construct_confusion": confusion,
                      "natural_diagnoses": Counter(x["diagnosis"]["primary"] for x in natural)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hour 5 diagnostic audit/demo (no training-effectiveness evaluation)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hour5/demo"))
    parser.add_argument("--calibration-count", type=int, default=80)
    parser.add_argument("--diagnostic-count", type=int, default=50)
    args = parser.parse_args()
    run(args.output, args.calibration_count, args.diagnostic_count)
