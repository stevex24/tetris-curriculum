import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tetris_research.agent import LearningAgent
from tetris_research.hour3 import (_t_critical_975, _t_two_sided_p, PilotConfig, evaluate,
                                   replicate_seeds, run_replicate)
from tetris_research.training import CONDITIONS, select_history_material


class Hour3ValidityTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(PilotConfig(), baseline_history_placements=5,
                              training_placements=5, evaluation_challenges=2,
                              evaluation_max_placements=12)

    def test_evaluation_is_fully_non_mutating(self):
        agent = LearningAgent("test", [0.2, 0.1, 0.15, 0.25], seed=19)
        before = copy.deepcopy(agent)
        evaluate(agent, [101, 102], 15)
        self.assertEqual(before.weights, agent.weights)
        self.assertEqual(before._rng.getstate(), agent._rng.getstate())
        self.assertEqual(before.games_learned, agent.games_learned)

    def test_budgets_challenges_and_seed_domains_are_controlled(self):
        result = run_replicate(self.config, 0)
        self.assertEqual({5}, {r["training_steps"] for r in result["training"].values()})
        challenge_seeds = [result["pre_evaluation"][c]["challenges"][0]["seed"]
                           for c in CONDITIONS]
        self.assertEqual(1, len(set(challenge_seeds)))
        self.assertEqual(result["pre_evaluation"]["control"]["challenges"],
                         result["pre_evaluation"]["rating_only"]["challenges"])
        used = result["seeds"]
        self.assertFalse(set(used["evaluation"]) & {used["baseline_history"],
                         used["tutorial_selection"], used["training_stream"]})

    def test_tutorial_selection_does_not_use_hidden_weights(self):
        observable = [{"steps": [{"features": {"holes": 3, "max_height": 4,
                        "bumpiness": 5, "lines_cleared": 0}}], "final_weights": [1, 2, 3, 4]}]
        first = select_history_material(1500, observable, 7)
        observable[0]["final_weights"] = [-100, 500, 900, -3]
        self.assertEqual(first, select_history_material(1500, observable, 7))

    def test_replicate_seeds_are_reproducible_and_distinct(self):
        self.assertEqual(replicate_seeds(12, 3), replicate_seeds(12, 3))
        self.assertNotEqual(replicate_seeds(12, 3), replicate_seeds(12, 4))

    def test_no_state_leaks_between_replicates(self):
        first = run_replicate(self.config, 1)
        run_replicate(self.config, 2)
        repeated = run_replicate(self.config, 1)
        self.assertEqual(first["rows"], repeated["rows"])
        self.assertEqual(first["training"], repeated["training"])
        self.assertTrue(first["identical_initial_state"])

    def test_student_t_calculations_match_known_values(self):
        self.assertAlmostEqual(1.0, _t_two_sided_p(0.0, 10), places=12)
        self.assertAlmostEqual(0.5, _t_two_sided_p(1.0, 1), places=12)
        self.assertAlmostEqual(2.009575, _t_critical_975(49), places=5)


if __name__ == "__main__":
    unittest.main()
