import copy
import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

from tetris_research.agent import LearningAgent
from tetris_research.hour4 import (Hour4Config, evaluate, primary_score, replicate_seeds,
                                   run_replicate, skill_scores_from_steps)
from tetris_research.training import CONDITIONS


class Hour4ValidityTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(Hour4Config(), baseline_history_placements=5,
                              training_placements=5, evaluation_challenges=2,
                              evaluation_max_placements=12)

    def test_skill_metrics_expected_direction(self):
        poor = [{"holes": 4, "max_height": 10, "bumpiness": 12, "lines_cleared": 0},
                {"holes": 2, "max_height": 8, "bumpiness": 8, "lines_cleared": 0}]
        good = [{"holes": 0, "max_height": 3, "bumpiness": 2, "lines_cleared": 1},
                {"holes": 1, "max_height": 4, "bumpiness": 4, "lines_cleared": 0}]
        a, b = skill_scores_from_steps(poor), skill_scores_from_steps(good)
        self.assertTrue(all(b[key] > a[key] for key in a))

    def test_primary_is_computable_and_rewards_survival(self):
        agent = LearningAgent("test", [0.2, 0.1, 0.15, 0.25], seed=19)
        result = evaluate(agent, [101, 102], 15)
        self.assertEqual(2, len(result["challenges"]))
        self.assertTrue(all(isinstance(c["successful_placements"], float) for c in result["challenges"]))
        self.assertGreater(primary_score([{"successful_placements": 12}]),
                           primary_score([{"successful_placements": 5}]))

    def test_target_evaluation_is_observable_and_non_mutating(self):
        agent = LearningAgent("test", [0.2, 0.1, 0.15, 0.25], seed=19)
        before = copy.deepcopy(agent)
        result = evaluate(agent, [101], 15)
        self.assertEqual(set(result["skill_scores"]), {"hole_avoidance", "stack_height_management",
                         "surface_smoothness", "line_clearing_efficiency"})
        self.assertNotIn("weights", json.dumps(result))
        self.assertEqual(before.weights, agent.weights)
        self.assertEqual(before._rng.getstate(), agent._rng.getstate())
        self.assertEqual(before.games_learned, agent.games_learned)

    def test_controls_and_reproducible_separated_seeds(self):
        result = run_replicate(self.config, 0)
        self.assertEqual({5}, {r["training_steps"] for r in result["training"].values()})
        for phase in ("pre_evaluation", "post_evaluation"):
            seeds = [[x["seed"] for x in result[phase][c]["challenges"]] for c in CONDITIONS]
            self.assertEqual(seeds[0], seeds[1]); self.assertEqual(seeds[1], seeds[2])
        used = result["seeds"]
        self.assertFalse(set(used["evaluation"]) & {used["baseline_history"],
                         used["tutorial_selection"], used["training_stream"]})
        self.assertEqual(replicate_seeds(7, 2), replicate_seeds(7, 2))
        self.assertNotEqual(replicate_seeds(7, 2), replicate_seeds(7, 3))

    def test_no_cross_replicate_leakage(self):
        first = run_replicate(self.config, 1)
        run_replicate(self.config, 2)
        self.assertEqual(first["rows"], run_replicate(self.config, 1)["rows"])

    def test_hour3_artifacts_match_frozen_manifest(self):
        root = Path(__file__).parents[1]
        manifest = json.loads((root / "artifacts/hour4/hour3_artifact_sha256.json").read_text())
        actual = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((root / "artifacts/hour3").rglob("*")) if path.is_file()}
        self.assertEqual(manifest, actual)


if __name__ == "__main__":
    unittest.main()
