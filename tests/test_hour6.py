import copy
import hashlib
import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path

from tetris_research.agent import LearningAgent
from tetris_research.hour4 import evaluate, skill_scores_from_steps
from tetris_research.hour5 import Calibration, Diagnosis, LearnerProfile
from tetris_research.hour6 import (HOUR6_MASTER_SEED, MIXED_MARGIN, PRIOR_EXPERIMENT_MASTER_SEEDS,
                                   Hour6Config, history_material, load_frozen_calibration,
                                   replicate_seeds, run_replicate)


ROOT = Path(__file__).parents[1]
CALIBRATION_PATH = ROOT / "artifacts/hour5/demo/calibration_parameters.json"


class Hour6ValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_frozen_calibration(CALIBRATION_PATH)
        cls.config = replace(Hour6Config(), baseline_history_placements=5,
                             training_placements=5, evaluation_challenges=2,
                             evaluation_max_placements=12)

    def test_exact_committed_hour5_calibration_is_loaded(self):
        raw = json.loads(CALIBRATION_PATH.read_text())
        self.assertEqual(raw["means"], self.calibration.means)
        self.assertEqual(raw["sample_sds"], self.calibration.sample_sds)
        self.assertEqual(500005, self.calibration.seed)

    def test_observables_only_and_hidden_weights_cannot_select(self):
        names = set(LearnerProfile.__dataclass_fields__)
        self.assertFalse(any("weight" in name or "policy" in name for name in names))
        diagnosis = Diagnosis("hole_management", "height_management", .8, "high", False,
                              {"hole_management": 1, "height_management": .2, "surface_management": 0})
        a = history_material(diagnosis, 1500, 7)
        b = history_material(copy.deepcopy(diagnosis), 1500, 7)
        self.assertEqual(a, b)

    def test_target_score_direction_and_distinct_mapping(self):
        good = skill_scores_from_steps([{"holes": 0, "max_height": 2, "bumpiness": 1, "lines_cleared": 0}])
        bad = skill_scores_from_steps([{"holes": 3, "max_height": 8, "bumpiness": 9, "lines_cleared": 0}])
        self.assertGreater(good["hole_avoidance"], bad["hole_avoidance"])
        self.assertGreater(good["stack_height_management"], bad["stack_height_management"])
        self.assertGreater(good["surface_smoothness"], bad["surface_smoothness"])
        tutorials = set()
        for primary in ("hole_management", "height_management", "surface_management"):
            other = next(x for x in ("hole_management", "height_management", "surface_management") if x != primary)
            d = Diagnosis(primary, other, 1, "high", False, {primary: 1, other: 0})
            tutorials.add(history_material(d, 1500, 3).diagnosed_weakness)
        self.assertEqual(tutorials, {"hole_management", "height_management", "surface_management"})

    def test_mixed_rule_uses_primary_and_retains_flag(self):
        diagnosis = Diagnosis("surface_management", "hole_management", .1, "low/mixed", True,
                              {"surface_management": 1, "hole_management": .9, "height_management": 0})
        material = history_material(diagnosis, 1500, 9)
        self.assertEqual("surface_management", material.diagnosed_weakness)
        self.assertEqual(.35, MIXED_MARGIN)

    def test_matched_held_out_nonmutating_evaluation_equal_budgets_and_no_leakage(self):
        result = run_replicate(self.config, self.calibration, 0)
        self.assertEqual({5}, {x["training_steps"] for x in result["training"].values()})
        for phase in ("pre", "post"):
            seed_lists = [[x["seed"] for x in result[phase][c]["challenges"]]
                          for c in ("control", "rating_only", "rating_history")]
            self.assertEqual(seed_lists[0], seed_lists[1]); self.assertEqual(seed_lists[1], seed_lists[2])
        used = result["seeds"]
        self.assertFalse(set(used["evaluation"]) & {used["baseline_history"], used["training_stream"], used["tutorial_selection"]})
        agent = LearningAgent("x", [.2, .1, .15, .25], seed=4); before = copy.deepcopy(agent)
        evaluate(agent, [101], 10)
        self.assertEqual(before.weights, agent.weights); self.assertEqual(before._rng.getstate(), agent._rng.getstate())
        self.assertEqual(before.games_learned, agent.games_learned)
        first = run_replicate(self.config, self.calibration, 1)
        run_replicate(self.config, self.calibration, 2)
        self.assertEqual(first["rows"], run_replicate(self.config, self.calibration, 1)["rows"])

    def test_hour6_seeds_disjoint_and_unique(self):
        self.assertNotIn(HOUR6_MASTER_SEED, PRIOR_EXPERIMENT_MASTER_SEEDS)
        records = [replicate_seeds(HOUR6_MASTER_SEED, i) for i in range(50)]
        flat = [value for row in records for key, item in row.items()
                for value in (item if isinstance(item, list) else [item])]
        self.assertEqual(len(flat), len(set(flat)))

    def test_hours_3_through_5_artifacts_unchanged_from_commit(self):
        names = subprocess.run(["git", "ls-tree", "-r", "--name-only", "74dcc05", "artifacts/hour3", "artifacts/hour4", "artifacts/hour5"],
                               cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
        for name in names:
            expected = subprocess.run(["git", "show", f"74dcc05:{name}"], cwd=ROOT,
                                      capture_output=True, check=True).stdout
            self.assertEqual(hashlib.sha256(expected).hexdigest(), hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), name)


if __name__ == "__main__":
    unittest.main()
