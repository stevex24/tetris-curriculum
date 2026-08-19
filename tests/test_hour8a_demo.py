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
        self.assertEqual(len(self.data["after"]) - 1, 26)

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
