import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "demo/build_day4_rl_comparison.py"
SPEC = importlib.util.spec_from_file_location("day4_demo_builder", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
EXPOSURES, SEED = builder.EXPOSURES, builder.SEED
build, verify = builder.build, builder.verify

ORACLE_SCRIPT = Path(__file__).resolve().parents[1] / "demo/build_day4_rl_oracle_comparison.py"
ORACLE_SPEC = importlib.util.spec_from_file_location("day4_oracle_demo_builder", ORACLE_SCRIPT)
assert ORACLE_SPEC and ORACLE_SPEC.loader
oracle_builder = importlib.util.module_from_spec(ORACLE_SPEC)
ORACLE_SPEC.loader.exec_module(oracle_builder)


class Day4DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build()

    def test_integrity(self):
        verify(self.data)
        self.assertEqual([0, 750, 3000], self.data["exposures"])
        self.assertEqual(2026082801, SEED)
        self.assertTrue(self.data["integrity"]["seed_is_preregistered"])
        self.assertFalse(self.data["integrity"]["expert_invoked"])

    def test_frozen_results_match_day4(self):
        expected = {0: (25, 0), 750: (70, 12), 3000: (300, 108)}
        for exposure in EXPOSURES:
            policy = self.data["policies"][str(exposure)]
            self.assertFalse(policy["learning_enabled"])
            self.assertEqual(expected[exposure], (policy["placements"], policy["lines"]))


class Day4OracleDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output = Path(__file__).resolve().parents[1] / "demo/day4_rl_oracle_comparison.json"
        cls.data = __import__("json").loads(output.read_text())

    def test_four_board_integrity(self):
        oracle_builder.verify(self.data)
        self.assertEqual(["0", "750", "3000", "oracle"], self.data["policy_order"])
        self.assertTrue(self.data["integrity"]["all_start_empty"])
        self.assertFalse(self.data["integrity"]["oracle_altered_learner_state"])

    def test_exact_reference_oracle(self):
        oracle = self.data["policies"]["oracle"]
        self.assertEqual("dellacherie-depth2-expectimax-v1", oracle["expert_id"])
        self.assertEqual(3, oracle["beam_width"])
        self.assertEqual(0.35, oracle["continuation_discount"])
        self.assertLessEqual(oracle["placements"], 300)


if __name__ == "__main__":
    unittest.main()
