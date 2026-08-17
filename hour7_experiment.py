from __future__ import annotations

import argparse
import json
from pathlib import Path

from tetris_research.hour7 import fixed_effect_summary, run_hour7


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the locked independent Hour 7 replication")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hour7/results"))
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--phase", choices=("smoke", "confirmatory"), default="confirmatory")
    args = parser.parse_args()
    required = 50 if args.phase == "confirmatory" else 3
    if args.replicates != required:
        parser.error(f"the locked Hour 7 {args.phase} run requires exactly {required} replicates")
    result = run_hour7(Path("artifacts/hour5/demo/calibration_parameters.json"),
                       args.output, args.phase, args.replicates)
    if args.phase == "confirmatory":
        pooled = fixed_effect_summary(Path("artifacts/hour6/results/replicate_results.csv"),
                                      args.output / "replicate_results.csv")
        (args.output / "exploratory_combined_hour6_hour7.json").write_text(json.dumps(pooled, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "resolution": result["provenance"]["resolution"],
                      "controls": result["provenance"]["controls"]}, indent=2))
