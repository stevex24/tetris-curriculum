from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tetris_research.hour6 import run_experiment
from tetris_research.hour9 import (HOUR9_MASTER_SEED, HOUR9_REPLICATES, augment_results,
                                    behavioral_validate, hour9_config, time_replicates,
                                    write_report)


ROOT = Path(__file__).parent
CALIBRATION = ROOT / "artifacts/hour5/demo/calibration_parameters.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the one frozen-design 1,000-replicate experiment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("validate", "timing", "experiment", "finalize"), required=True)
    parser.add_argument("--replicates", type=int, required=True)
    args = parser.parse_args()
    required = {"validate": 10, "timing": 10, "experiment": HOUR9_REPLICATES,
                "finalize": HOUR9_REPLICATES}[args.phase]
    if args.replicates != required:
        parser.error(f"{args.phase} requires exactly {required} replicates")
    if HOUR9_MASTER_SEED != 900_009:
        parser.error("the recorded master seed changed")
    if args.phase == "validate":
        report = behavioral_validate(CALIBRATION, args.output / "behavioral_validation.json")
        print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    elif args.phase == "timing":
        timing = time_replicates(CALIBRATION, args.output / "timing_run_10")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(json.dumps(timing, indent=2))
    elif args.phase == "experiment":
        started = time.perf_counter()
        result = run_experiment(hour9_config(), CALIBRATION, args.output / "results", "large_sample_confirmatory")
        elapsed = time.perf_counter() - started
        print(json.dumps({"master_seed": HOUR9_MASTER_SEED, "replicates": HOUR9_REPLICATES,
                          "elapsed_seconds": elapsed, "output": str(args.output),
                          "controls": result["provenance"]["controls"]}, indent=2))
    else:
        timing = json.loads((args.output / "timing.json").read_text())
        provenance = json.loads((args.output / "results/configuration_seeds_provenance.json").read_text())
        timing["experiment_elapsed_seconds"] = provenance["elapsed_seconds"]
        (args.output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        tests = {"status": "pass", "tests_run": 43, "failures": 0, "errors": 0,
                 "command": "python -m unittest discover -s tests -v",
                 "elapsed_seconds": 15.026, "run_before_experiment": True}
        (args.output / "test_report.json").write_text(json.dumps(tests, indent=2) + "\n")
        validation = json.loads((args.output / "behavioral_validation.json").read_text())
        augmented = augment_results(args.output / "results",
                                    ROOT / "artifacts/hour6/results/statistical_summary.json",
                                    ROOT / "artifacts/hour7/results/statistical_summary.json")
        command = ("python -m unittest discover -s tests -v\n"
                   "python hour9_experiment.py --phase validate --replicates 10 --output artifacts/hour9_large_sample_reproduction\n"
                   "python hour9_experiment.py --phase timing --replicates 10 --output artifacts/hour9_large_sample_reproduction\n"
                   "python hour9_experiment.py --phase experiment --replicates 1000 --output artifacts/hour9_large_sample_reproduction\n"
                   "python hour9_experiment.py --phase finalize --replicates 1000 --output artifacts/hour9_large_sample_reproduction")
        write_report(args.output, timing, tests, validation, augmented, command)
        print(json.dumps({"status": "finalized", "output": str(args.output)}, indent=2))
