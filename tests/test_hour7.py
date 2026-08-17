import copy
import hashlib
import json
import subprocess
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from tetris_research.agent import LearningAgent
from tetris_research.hour4 import evaluate
from tetris_research.hour5 import CALIBRATION_SEED, LearnerProfile
from tetris_research.hour6 import (HOUR6_MASTER_SEED, Hour6Config, TARGET_METRICS,
                                   TUTORIAL_TYPES, load_frozen_calibration, replicate_seeds,
                                   run_replicate)
from tetris_research.hour7 import (ALL_PRIOR_MASTER_SEEDS, HOUR7_MASTER_SEED,
                                   HOUR7_SMOKE_SEED, hour7_config)

ROOT = Path(__file__).parents[1]
CALIBRATION = ROOT / "artifacts/hour5/demo/calibration_parameters.json"


class Hour7LockedReplicationTests(unittest.TestCase):
    def test_scientific_parameters_exactly_match_hour6(self):
        h6, h7 = asdict(Hour6Config()), asdict(hour7_config())
        self.assertEqual({k: v for k, v in h6.items() if k != "master_seed"},
                         {k: v for k, v in h7.items() if k != "master_seed"})
        self.assertEqual(50, h7["replicates"])
        self.assertEqual(40, h7["baseline_history_placements"])
        self.assertEqual(40, h7["training_placements"])
        self.assertEqual(6, h7["evaluation_challenges"])
        self.assertEqual("mean successful placements per held-out challenge", h7["primary_measure"])
        self.assertEqual({"hole_management", "height_management", "surface_management"}, set(TARGET_METRICS))
        self.assertEqual({"hole_management", "height_management", "surface_management"}, set(TUTORIAL_TYPES))

    def test_exact_frozen_calibration_and_new_disjoint_seeds(self):
        calibration = load_frozen_calibration(CALIBRATION)
        self.assertEqual(500005, CALIBRATION_SEED)
        self.assertEqual(500005, calibration.seed)
        self.assertEqual("c69823decfe715ffbb8ec54885783881cef547bbbc1b59bb6a2de49bf114e476",
                         hashlib.sha256(CALIBRATION.read_bytes()).hexdigest())
        self.assertNotIn(HOUR7_MASTER_SEED, ALL_PRIOR_MASTER_SEEDS)
        self.assertNotIn(HOUR7_SMOKE_SEED, ALL_PRIOR_MASTER_SEEDS | {HOUR7_MASTER_SEED})
        records = [replicate_seeds(HOUR7_MASTER_SEED, i) for i in range(50)]
        flat = [v for row in records for item in row.values() for v in (item if isinstance(item, list) else [item])]
        self.assertEqual(len(flat), len(set(flat)))

    def test_matched_nonlearning_evaluation_and_hidden_policy_unavailable(self):
        self.assertFalse(any("weight" in name or "policy" in name for name in LearnerProfile.__dataclass_fields__))
        agent = LearningAgent("x", [.2, .1, .15, .25], seed=9); before = copy.deepcopy(agent)
        evaluate(agent, [701], 10)
        self.assertEqual(before.weights, agent.weights)
        self.assertEqual(before._rng.getstate(), agent._rng.getstate())
        self.assertEqual(before.games_learned, agent.games_learned)
        config = replace(hour7_config(), baseline_history_placements=5, training_placements=5,
                         evaluation_challenges=2, evaluation_max_placements=12)
        result = run_replicate(config, load_frozen_calibration(CALIBRATION), 0)
        self.assertEqual({5}, {x["training_steps"] for x in result["training"].values()})
        for phase in ("pre", "post"):
            seeds = [[x["seed"] for x in result[phase][c]["challenges"]]
                     for c in ("control", "rating_only", "rating_history")]
            self.assertEqual(seeds[0], seeds[1]); self.assertEqual(seeds[1], seeds[2])

    def test_hours_3_through_6_committed_artifacts_unchanged(self):
        names = subprocess.run(["git", "ls-tree", "-r", "--name-only", "3e71ee2",
                                "artifacts/hour3", "artifacts/hour4", "artifacts/hour5", "artifacts/hour6"],
                               cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
        for name in names:
            expected = subprocess.run(["git", "show", f"3e71ee2:{name}"], cwd=ROOT,
                                      capture_output=True, check=True).stdout
            self.assertEqual(hashlib.sha256(expected).digest(), hashlib.sha256((ROOT / name).read_bytes()).digest(), name)


if __name__ == "__main__":
    unittest.main()
