import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class DemoIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data = json.loads((ROOT / "demo/demo_data.json").read_text())

    def test_predeclared_learner_mapping(self):
        self.assertEqual((self.data["selection"]["study"], self.data["selection"]["replicate"]), ("Hour 7", 0))
        self.assertEqual((self.data["profile"]["primary"], self.data["tutorial"]["type"]), ("height_management", "stack_height"))
        self.assertEqual(self.data["profile"]["confidence"], "high")

    def test_same_deterministic_challenge(self):
        self.assertEqual(self.data["challenge_seed"], 1370957669)
        self.assertEqual(len(self.data["before"]) - 1, 28)
        self.assertEqual(set(self.data["after"]), {"control", "rating_only", "rating_history"})
        self.assertTrue(self.data["integrity"]["same_after_challenge_seed"])
        self.assertEqual({condition: len(frames) - 1 for condition, frames in self.data["after"].items()},
                         {"control": 26, "rating_only": 26, "rating_history": 26})

    def test_matched_clones_and_saved_counts(self):
        self.assertTrue(self.data["integrity"]["same_baseline_clone"])
        self.assertTrue(self.data["integrity"]["matches_saved_hour7"])
        self.assertEqual(self.data["final_counts"],
                         {"before": 28, "control": 26, "rating_only": 26, "rating_history": 26})
        saved = json.loads((ROOT / "artifacts/hour7/results/challenge_results.jsonl").read_text().splitlines()[0])
        for condition in ("control", "rating_only", "rating_history"):
            challenge = saved["post"][condition]["challenges"][0]
            self.assertEqual(challenge["seed"], self.data["challenge_seed"])
            self.assertEqual(challenge["successful_placements"], self.data["final_counts"][condition])
            self.assertEqual(challenge["lines_cleared"], self.data["after"][condition][-1]["lines"])

    def test_condition_labels_are_scientifically_explicit(self):
        conditions = self.data["conditions"]
        self.assertEqual(conditions["control"]["training"], "Ordinary practice")
        self.assertIn("without access to history", conditions["rating_only"]["training"])
        self.assertIn("calibrated learner profile", conditions["rating_history"]["training"])
        self.assertEqual(self.data["selection"]["label"],
                         "One matched replicate — illustrative, not the aggregate statistical result.")

    def test_saved_result_numbers(self):
        expected = {"6": ((.4866666666666664, .09574267364838257, .8775906596849503),
                          (.40666666666666645, .0775069232289804, .7358264101043526)),
                    "7": ((-.1433333333333332, -.5411272171911756, .25446055052450917),
                          (.21999999999999978, -.17657539236824868, .6165753923682482))}
        for hour, pairs in expected.items():
            for key, values in zip(("rating_history_minus_control", "rating_history_minus_rating_only"), pairs):
                row = self.data["results"][hour][key]
                self.assertEqual((row["paired_mean_difference"], *row["ci95"]), values)

if __name__ == "__main__": unittest.main()
