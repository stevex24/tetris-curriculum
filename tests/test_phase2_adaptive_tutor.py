import copy
import unittest

from tetris_research.adaptive_tutor import (diagnose_full_history, profile_distance,
                                             select_block, situation_signature)
from tetris_research.day5 import collect_history
from tetris_research.day7 import starting_student
from tetris_research.phase2 import (SMOKE, calibrate_ladder, full_profile_candidates,
                                    performance_rating, validate_config)
from tetris_research.phase2_validator import validate
from tetris_research.richer_student import FEATURE_NAMES


class Phase2AdaptiveTutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.student = starting_student("rl@750", 99101)
        cls.history = collect_history(cls.student, (99102,), 8)

    def test_full_profile_has_all_dimensions_and_is_nonmutating(self):
        before = copy.deepcopy(self.student.serialize_state())
        profile = diagnose_full_history(self.history)
        self.assertEqual(profile["dimensions"], 18)
        self.assertEqual(set(profile["severity"]), set(FEATURE_NAMES))
        self.assertEqual(before, self.student.serialize_state())

    def test_candidates_expose_all_features_not_only_families(self):
        candidates = full_profile_candidates(self.history, seed=99103, count=4)
        signatures = [situation_signature(row) for row in candidates]
        self.assertTrue(all(set(row) == set(FEATURE_NAMES) for row in signatures))
        self.assertGreater(sum(value > 0 for row in signatures for value in row.values()), 18)

    def test_profile_distance_responds_to_any_feature(self):
        a = {"normalised_severity": {name: 0.0 for name in FEATURE_NAMES}}
        b = copy.deepcopy(a)
        b["normalised_severity"][FEATURE_NAMES[-1]] = 1.0
        self.assertEqual(profile_distance(a, b), 1.0)

    def test_select_block_obeys_budget(self):
        from tetris_research.adaptive_tutor import CandidateValue
        situations = full_profile_candidates(self.history, seed=99104, count=3)
        values = [CandidateValue(row, float(i), 0.1, 1.0, False)
                  for i, row in enumerate(situations)]
        self.assertEqual(len(select_block(values, 20, 2)), 20)

    def test_rating_formula_is_monotonic(self):
        refs = {name: [{"lines_cleared": i, "placements": 10}]
                for i, name in enumerate(("rl@0", "rl@250", "rl@750", "rl@1500", "rl@3000"))}
        ratings = {name: float(i * 100) for i, name in enumerate(refs)}
        weak = performance_rating([{"lines_cleared": 0, "placements": 9}], refs, ratings)
        strong = performance_rating([{"lines_cleared": 9, "placements": 10}], refs, ratings)
        self.assertGreater(strong, weak)

    def test_smoke_config_domains_and_blocks(self):
        validate_config(SMOKE)

    def _validator_fixture(self):
        config = {key: list(value) if isinstance(value, tuple) else value
                  for key, value in SMOKE.__dict__.items()}
        profile = {"dimensions": 18, "severity": {name: 0.0 for name in FEATURE_NAMES}}
        training = {
            "ordinary": {"updates_applied": 40},
            "imitation": {"updates_applied": 40},
            "static_personalized": {"updates_applied": 40, "learning_path": "rl"},
            "responsive_personalized": {
                "updates_applied": 40, "learning_path": "rl", "expert_labels_seen": 0,
                "expert_calls_during_learning": 0, "diagnostic_placements": 10,
                "reward_source": "day4.gameplay_reward: 0.02 + simulator lines_cleared",
                "blocks": [{"profile": profile, "profile_distance_from_previous": None,
                            "predicted_values": [{"mastery_factor": 1.0, "retired": False}]}],
            },
        }
        result = {"configuration": config, "conditions": ["ordinary", "imitation",
                  "static_personalized", "responsive_personalized"],
                  "replicates": [{"training_audits": training,
                    "evaluation_before_sha256": "same", "evaluation_after_sha256": "same"}],
                  "rating_ladder": {"ratings": {name: 0.0 for name in
                    ("rl@0", "rl@250", "rl@750", "rl@1500", "rl@3000")}}}
        prereg = {"status": "locked before final run", "configuration": copy.deepcopy(config),
                  "demo_seed_rule": "first final held-out evaluation seed",
                  "no_tuning_rule": "Day 7 final outcomes are historical context only and fixed."}
        return result, prereg

    def test_independent_validator_accepts_controlled_fixture(self):
        result, prereg = self._validator_fixture()
        report = validate(result, prereg)
        self.assertTrue(report["success"], report)

    def test_validator_catches_expert_leak_unequal_budget_and_mutation(self):
        result, prereg = self._validator_fixture()
        audit = result["replicates"][0]["training_audits"]["responsive_personalized"]
        audit["expert_labels_seen"] = 1
        audit["updates_applied"] = 41
        result["replicates"][0]["evaluation_after_sha256"] = "changed"
        checks = validate(result, prereg)["checks"]
        self.assertFalse(checks["responsive_receives_no_expert_actions"])
        self.assertFalse(checks["equal_learner_updates"])
        self.assertFalse(checks["evaluation_nonmutating"])

    def test_validator_catches_profile_collapse_and_prereg_tampering(self):
        result, prereg = self._validator_fixture()
        block = result["replicates"][0]["training_audits"]["responsive_personalized"]["blocks"][0]
        block["profile"]["dimensions"] = 2
        prereg["configuration"]["training_budget"] = 999
        checks = validate(result, prereg)["checks"]
        self.assertFalse(checks["full_18_dimensions_preserved"])
        self.assertFalse(checks["locked_preregistration_matches"])


if __name__ == "__main__":
    unittest.main()
