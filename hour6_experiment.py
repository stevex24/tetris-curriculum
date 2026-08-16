from __future__ import annotations

import argparse
import json
from pathlib import Path

from tetris_research.hour6 import Hour6Config, run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the frozen Hour 6 effectiveness experiment")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hour6/results"))
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--phase", choices=("smoke", "confirmatory"), default="confirmatory")
    args = parser.parse_args()
    if args.phase == "confirmatory" and args.replicates != 50:
        parser.error("the confirmatory Hour 6 run is frozen at exactly 50 replicates")
    if args.phase == "smoke" and args.replicates != 3:
        parser.error("the Hour 6 smoke test is frozen at exactly 3 replicates")
    result = run_experiment(Hour6Config(replicates=args.replicates),
                            Path("artifacts/hour5/demo/calibration_parameters.json"),
                            args.output, args.phase)
    print(json.dumps({"output": str(args.output), "resolution": result["provenance"]["resolution"],
                      "controls": result["provenance"]["controls"]}, indent=2))
