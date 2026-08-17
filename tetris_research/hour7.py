from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

from .hour6 import (HOUR6_MASTER_SEED, PRIOR_EXPERIMENT_MASTER_SEEDS, Hour6Config,
                    _t_critical_975, run_experiment)

HOUR7_MASTER_SEED = 700_007
HOUR7_SMOKE_SEED = 700_107
ALL_PRIOR_MASTER_SEEDS = PRIOR_EXPERIMENT_MASTER_SEEDS | {HOUR6_MASTER_SEED}


def hour7_config(master_seed: int = HOUR7_MASTER_SEED, replicates: int = 50) -> Hour6Config:
    """The Hour 6 scientific configuration with only fresh experimental randomness."""
    return replace(Hour6Config(), master_seed=master_seed, replicates=replicates)


def run_hour7(calibration_path: Path, output: Path, phase: str, replicates: int) -> dict:
    seed = HOUR7_MASTER_SEED if phase == "confirmatory" else HOUR7_SMOKE_SEED
    return run_experiment(hour7_config(seed, replicates), calibration_path, output, phase)


def fixed_effect_summary(hour6_csv: Path, hour7_csv: Path) -> dict:
    """Inverse-variance fixed-effect summary of the two independent study estimates."""
    def study(path: Path) -> dict[str, dict[str, float]]:
        rows = list(csv.DictReader(path.open()))
        by_rep = {}
        for row in rows:
            by_rep.setdefault(int(row["replicate"]), {})[row["condition"]] = float(row["improvement"])
        result = {}
        for other in ("control", "rating_only"):
            diffs = [values["rating_history"] - values[other] for _, values in sorted(by_rep.items())]
            n = len(diffs); estimate = sum(diffs) / n
            variance = sum((x - estimate) ** 2 for x in diffs) / (n - 1)
            result[f"rating_history_minus_{other}"] = {
                "n": n, "estimate": estimate, "sampling_variance": variance / n
            }
        return result

    studies = {"hour6": study(hour6_csv), "hour7": study(hour7_csv)}
    combined = {}
    for comparison in studies["hour6"]:
        a, b = studies["hour6"][comparison], studies["hour7"][comparison]
        weights = [1 / a["sampling_variance"], 1 / b["sampling_variance"]]
        estimate = (weights[0] * a["estimate"] + weights[1] * b["estimate"]) / sum(weights)
        se = math.sqrt(1 / sum(weights))
        # Normal fixed-effect CI is conventional; retain study-level t CIs separately.
        combined[comparison] = {
            "hour6_estimate": a["estimate"], "hour7_estimate": b["estimate"],
            "fixed_effect_estimate": estimate, "standard_error": se,
            "ci95": [estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se],
            "method": "inverse-variance fixed effect across the two independent study estimates"
        }
    return {"status": "exploratory; secondary to the independent Hour 7 replication",
            "studies": studies, "combined": combined}
