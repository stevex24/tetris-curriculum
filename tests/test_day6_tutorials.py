import copy
import json
import unittest
from pathlib import Path

from tetris_research.richer_student import RicherRLStudent
from tetris_research.tutorials import (allocate_profile, generate_tutorial_sets,
    generic_tutorial_set, select_personalized, targeting_metric, train_on_situations)

ROOT = Path(__file__).resolve().parents[1]


class Day6TutorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.day5 = json.loads((ROOT / "experiments/day5/final_results.json").read_text())
        cls.history = cls.day5["histories"]["natural_day4_rl"]
        cls.profile = cls.day5["profiles"]["natural_day4_rl"]
        cls.sets = generate_tutorial_sets(cls.history, seed=2026083101, per_family=4)

    def test_generation_is_deterministic_diverse_legal_and_nonmutating(self):
        before = copy.deepcopy(self.history)
        again = generate_tutorial_sets(self.history, seed=2026083101, per_family=4)
        self.assertEqual(self.sets, again); self.assertEqual(before, self.history)
        for rows in self.sets.values():
            self.assertEqual(4, len({(row.rows, row.piece) for row in rows}))
            self.assertEqual(4, len({row.source_id for row in rows}))
            self.assertTrue(any(row.perturbation == "one_legal_placement" for row in rows))

    def test_target_metric_exceeds_profile_blind_control(self):
        generic = generic_tutorial_set(self.history, seed=2026083101, count=4)
        for family, rows in self.sets.items():
            target = sum(targeting_metric(x.state, x.piece, family) for x in rows) / len(rows)
            control = sum(targeting_metric(x.state, x.piece, family) for x in generic) / len(generic)
            self.assertGreaterEqual(target, 1.25 * control)

    def test_profiles_change_allocation_but_hidden_fields_do_not(self):
        hidden = copy.deepcopy(self.profile); hidden["weights"] = [999] * 18
        self.assertEqual(allocate_profile(self.profile, 24), allocate_profile(hidden, 24))
        contrast = copy.deepcopy(self.profile)
        contrast["ranking"] = ["hole_management", "well_management", "surface_smoothness"]
        contrast["families"]["hole_management"]["score"] = 90
        contrast["families"]["well_management"]["score"] = 10
        self.assertNotEqual(allocate_profile(self.profile, 24), allocate_profile(contrast, 24))

    def test_equal_budget_uses_standard_rl_and_environment_reward(self):
        state = json.loads((ROOT / "experiments/day4/final_results.json").read_text())[
            "evaluation_before_states"]["rl@3000"]
        a = RicherRLStudent.from_state({**state, "agent_id": "a"})
        b = RicherRLStudent.from_state({**state, "agent_id": "b"})
        selected = select_personalized(self.profile, self.sets, 7)
        generic = generic_tutorial_set(self.history, seed=2026083101, count=4)
        x, y = train_on_situations(a, selected, 7), train_on_situations(b, generic, 7)
        self.assertEqual((7, 7), (x["updates_applied"], y["updates_applied"]))
        self.assertEqual(x["learning_path"], y["learning_path"])
        self.assertEqual(0, x["expert_calls"] + y["expert_calls"])
        self.assertEqual(x["reward_source"], y["reward_source"])

    def test_illegal_and_empty_training_inputs_fail(self):
        state = json.loads((ROOT / "experiments/day4/final_results.json").read_text())[
            "evaluation_before_states"]["rl@3000"]
        student = RicherRLStudent.from_state({**state, "agent_id": "bad"})
        with self.assertRaises(ValueError): train_on_situations(student, (), 1)
        with self.assertRaises(ValueError): allocate_profile(self.profile, -1)


if __name__ == "__main__": unittest.main()
