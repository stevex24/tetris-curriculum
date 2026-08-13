import argparse
from dataclasses import replace
from pathlib import Path

from tetris_research.hour3 import PilotConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled Hour 3 pilot")
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, default=300_003)
    args = parser.parse_args()
    config = replace(PilotConfig(), replicates=args.replicates, master_seed=args.master_seed)
    provenance = run_experiment(config, args.output)
    print(f"completed {args.replicates} replicates in {provenance['elapsed_seconds']:.3f}s")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
