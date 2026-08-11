import json
import tempfile
import unittest
from pathlib import Path

from tetris_research import EloRatings, LearningAgent, TetrisAdapter
from tetris_research.experiment import play_game


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


if __name__ == "__main__":
    unittest.main()
