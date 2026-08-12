import json
import tempfile
import unittest
from pathlib import Path

from tetris_research import EloRatings, LearningAgent, TetrisAdapter
from tetris_research.experiment import play_game
from tetris_research.training import (CONTROL, RATING_HISTORY, RATING_ONLY, control_material,
                                      diagnose_weakness, observable_history_features,
                                      rating_difficulty, select_history_material,
                                      select_rating_only_material, train)


class PrototypeTests(unittest.TestCase):
    def test_placements_are_legal_and_features_present(self):
        adapter = TetrisAdapter()
        choices = adapter.legal_actions(adapter.initial_state(), "I")
        self.assertEqual(17, len(choices))
        self.assertEqual(set(adapter.feature_names), set(choices[0].raw_features))

    def test_clone_and_seed_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            original = LearningAgent("one", [0.1] * 4, seed=9)
            clone = original.clone("two")
            a = play_game(TetrisAdapter(), original, 44, path, 25, learn=False)
            b = play_game(TetrisAdapter(), clone, 44, path, 25, learn=False)
            self.assertEqual([x["action"] for x in a["steps"]], [x["action"] for x in b["steps"]])

    def test_learning_persists_and_history_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = LearningAgent("learner", [0.0] * 4, seed=3)
            before = agent.weights.copy()
            result = play_game(TetrisAdapter(), agent, 4, root / "games.jsonl", 30)
            self.assertNotEqual(before, agent.weights)
            agent.save(root / "agent.json")
            self.assertEqual(agent.weights, LearningAgent.load(root / "agent.json").weights)
            logged = json.loads((root / "games.jsonl").read_text().splitlines()[0])
            self.assertEqual(result["steps"], logged["steps"])

    def test_standard_elo(self):
        elo = EloRatings()
        update = elo.update("a", "b", 1.0)
        self.assertEqual(1516.0, update["a_after"])
        self.assertEqual(1484.0, update["b_after"])

    def test_rating_mapping_and_material_are_deterministic(self):
        self.assertEqual("introductory", rating_difficulty(1399))
        self.assertEqual("intermediate", rating_difficulty(1400))
        self.assertEqual("advanced", rating_difficulty(1600))
        a = select_rating_only_material(1500, 8)
        b = select_rating_only_material(1500, 8)
        self.assertEqual(RATING_ONLY, a.condition)
        self.assertEqual(a, b)
        self.assertEqual({}, a.diagnosis)
        self.assertIsNone(a.diagnosed_weakness)

    def test_history_diagnosis_uses_only_observable_step_features(self):
        records = [{"steps": [
            {"features": {"holes": 30, "max_height": 5, "bumpiness": 4, "lines_cleared": 1}},
            {"features": {"holes": 34, "max_height": 6, "bumpiness": 5, "lines_cleared": 1}},
        ], "final_weights": [999, 999, 999, 999]}]
        features = observable_history_features(records)
        self.assertEqual("hole_avoidance", diagnose_weakness(features))
        material = select_history_material(1500, records, 4)
        self.assertEqual(RATING_HISTORY, material.condition)
        self.assertEqual("hole_avoidance", material.diagnosed_weakness)
        # Hidden weights are deliberately irrelevant to diagnosis.
        records[0]["final_weights"] = [-999, -999, -999, -999]
        self.assertEqual(material, select_history_material(1500, records, 4))

    def test_equal_step_training_uses_existing_choose_update(self):
        class CountingAgent(LearningAgent):
            calls = 0

            def choose(self, choices, learn=True):
                self.calls += 1
                self.assert_learning = learn
                return super().choose(choices, learn=learn)

        original = CountingAgent("base", [0.1, 0.2, 0.3, 0.4], seed=12)
        agents = [original.clone(name) for name in ("c", "r", "h")]
        fake_history = [{"steps": [{"features": {"holes": 0, "max_height": 1,
                                                   "bumpiness": 2, "lines_cleared": 0}}]}]
        materials = [control_material(), select_rating_only_material(1500, 9),
                     select_history_material(1500, fake_history, 9)]
        records = [train(agent, material, 1500, 7, 10)
                   for agent, material in zip(agents, materials)]
        self.assertEqual([7, 7, 7], [record["training_steps"] for record in records])
        self.assertTrue(all(agent.calls == 7 and agent.assert_learning for agent in agents))
        self.assertEqual([CONTROL, RATING_ONLY, RATING_HISTORY],
                         [record["condition"] for record in records])
        self.assertTrue(all(record["policy_weights_before"] == [0.1, 0.2, 0.3, 0.4]
                            for record in records))


if __name__ == "__main__":
    unittest.main()
