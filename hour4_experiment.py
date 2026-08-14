import argparse
from dataclasses import replace
from pathlib import Path

from tetris_research.hour4 import Hour4Config, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the frozen Hour 4 experiment")
    parser.add_argument("--replicates", type=int, choices=(3, 50), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, default=400_004)
    args = parser.parse_args()
    phase = "smoke" if args.replicates == 3 else "confirmatory"
    provenance = run_experiment(replace(Hour4Config(), replicates=args.replicates,
                                master_seed=args.master_seed), args.output, phase)
    print(f"completed {args.replicates} replicates in {provenance['elapsed_seconds']:.3f}s")
    print(json.dumps(provenance["resolution"], sort_keys=True))


if __name__ == "__main__":
    import json
    main()
