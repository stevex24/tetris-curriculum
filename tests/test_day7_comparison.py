import copy
import unittest

from tetris_research.day7 import (CONDITIONS, FINAL, paired_summary, starting_student,
                                  train_imitation, train_ordinary, validate_config)
from tetris_research.day7_validator import validate
from tetris_research.expert import DELLACHERIE_WEIGHTS
from tetris_research.richer_student import FEATURE_NAMES


def fixture():
    config = {**FINAL.__dict__, "replicate_seeds": list(FINAL.replicate_seeds),
              "history_seeds": list(FINAL.history_seeds),
              "evaluation_game_seeds": list(FINAL.evaluation_game_seeds),
              "diagnostic_state_seeds": list(FINAL.diagnostic_state_seeds)}
    state = dict(starting_student("rl@750", FINAL.replicate_seeds[0]).serialize_state())
    states = {name: {**copy.deepcopy(state), "agent_id": name} for name in CONDITIONS}
    audit = {
        "ordinary": {"updates_applied": 500, "decision_states_seen": 500,
                     "learning_path": "RicherRLStudent.choose_placement/update/finish_episode",
                     "reward_source": "day4.gameplay_reward"},
        "imitation": {"updates_applied": 500, "decision_states_seen": 500},
        "personalized": {"updates_applied": 500, "decision_states_seen": 500,
                         "expert_calls_during_learning": 0,
                         "learning_path": "RicherRLStudent.choose_placement/update/finish_episode",
                         "reward_source": "day4.gameplay_reward: 0.02 + simulator lines_cleared"},
    }
    trained = {name: {**states[name], "expert_parameters": None} for name in CONDITIONS}
    game = {"seed": config["evaluation_game_seeds"][0], "stream_sha256": "x",
            "baseline": {}, **{name: {} for name in CONDITIONS}}
    replicate = {"replicate_id": 0, "training_seed": config["replicate_seeds"][0],
                 "initial_states": states, "training_audits": audit, "trained_states": trained,
                 "diagnostic_event_order": ["all_frozen_student_decisions", "external_expert_scoring"],
                 "evaluation_before_sha256": "same", "evaluation_after_sha256": "same",
                 "game_rows": [game]}
    result = {"configuration": config, "conditions": list(CONDITIONS), "replicates": [replicate],
              "shared_history": {"student_nonmutating": True},
              "history_design": "Option A: shared", "expert_coefficients_copied": False,
              "primary_metric": "change in matched held-out mean lines cleared from frozen baseline",
              "ratings": None}
    # Keep a one-replicate fixture internally consistent.
    result["configuration"]["replicate_seeds"] = [config["replicate_seeds"][0]]
    prereg = {"status": "locked before final run", "configuration": copy.deepcopy(result["configuration"])}
    return result, prereg


class Day7LearningTests(unittest.TestCase):
    def test_richer_imitation_is_student_owned_and_one_update_per_label(self):
        student = starting_student("rl@750", 81001)
        before = list(student.weights)
        audit = train_imitation(student, seed=81001, budget=3, learning_rate=.08)
        self.assertEqual(audit["updates_applied"], 3)
        self.assertEqual(audit["expert_labels_seen"], 3)
        self.assertNotEqual(before, student.weights)
        self.assertEqual(len(student.weights), len(FEATURE_NAMES))
        self.assertNotEqual(student.weights[:len(DELLACHERIE_WEIGHTS)], list(DELLACHERIE_WEIGHTS.values()))

    def test_ordinary_is_real_sequential_rl(self):
        student = starting_student("rl@750", 82001)
        before = list(student.weights)
        audit = train_ordinary(student, seed=82001, budget=5)
        self.assertEqual(audit["updates_applied"], 5)
        self.assertEqual(audit["ordinary_gameplay_states_seen"], 5)
        self.assertNotEqual(before, student.weights)

    def test_paired_bootstrap_preserves_replicate_differences(self):
        rows = [{"personalized": 4, "ordinary": 1},
                {"personalized": 7, "ordinary": 5}]
        result = paired_summary(rows, "personalized", "ordinary", seed=1, samples=100)
        self.assertEqual(result["replicate_differences"], [3.0, 2.0])
        self.assertEqual(result["mean_difference"], 2.5)

    def test_final_seed_domains_are_disjoint(self):
        validate_config(FINAL)


class Day7ValidatorTests(unittest.TestCase):
    def test_valid_fixture_passes(self):
        result, prereg = fixture()
        report = validate(result, prereg)
        self.assertTrue(report["success"], report)

    def test_unequal_updates_fail(self):
        result, prereg = fixture()
        result["replicates"][0]["training_audits"]["personalized"]["updates_applied"] = 501
        self.assertFalse(validate(result, prereg)["checks"]["equal_update_counts"])

    def test_start_mismatch_and_evaluation_mutation_fail(self):
        result, prereg = fixture()
        result["replicates"][0]["initial_states"]["imitation"]["weights"][0] += 1
        result["replicates"][0]["evaluation_after_sha256"] = "changed"
        checks = validate(result, prereg)["checks"]
        self.assertFalse(checks["identical_starting_students"])
        self.assertFalse(checks["evaluation_nonmutating"])

    def test_free_history_expert_leak_and_prereg_change_fail(self):
        result, prereg = fixture()
        result["shared_history"]["student_nonmutating"] = False
        result["replicates"][0]["training_audits"]["personalized"]["expert_calls_during_learning"] = 1
        prereg["configuration"]["training_budget"] = 999
        checks = validate(result, prereg)["checks"]
        self.assertFalse(checks["history_collection_has_no_free_learning"])
        self.assertFalse(checks["no_expert_in_personalized_learning"])
        self.assertFalse(checks["locked_configuration_matches_result"])


if __name__ == "__main__":
    unittest.main()
