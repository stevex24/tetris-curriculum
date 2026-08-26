import copy
import unittest

from tetris_research.day4 import SMOKE, gameplay_reward, run
from tetris_research.day4_validator import validate_day4
from tetris_research.richer_student import FEATURE_NAMES, RicherRLStudent, richer_features
from tetris_research.student import AgentExperience
from tetris_research.tetris import TetrisAdapter, TetrisState


class RicherStudentTests(unittest.TestCase):
    def test_features_are_distinct_and_finite(self):
        state = TetrisState((28, 60, 62, 30) + (0,) * 16)
        legal = TetrisAdapter().legal_actions(state, "T")
        columns = list(zip(*(richer_features(state, "T", choice) for choice in legal)))
        self.assertEqual(18, len(FEATURE_NAMES))
        self.assertEqual(18, len(set(FEATURE_NAMES)))
        self.assertGreaterEqual(len(set(columns)), 15)

    def test_feature_groups_induce_distinct_placement_preferences(self):
        state = TetrisState((28, 60, 62, 30) + (0,) * 16)
        legal = TetrisAdapter().legal_actions(state, "T")
        placements = []
        for feature in ("holes", "maximum_height", "cumulative_wells"):
            weights = [0.0] * len(FEATURE_NAMES)
            weights[FEATURE_NAMES.index(feature)] = -10.0
            student = RicherRLStudent(feature, weights)
            placements.append(student.choose_placement(state, "T", legal, deterministic=True).placement)
        self.assertEqual(3, len(set(placements)))

    def test_delayed_return_changes_an_earlier_action(self):
        game, state = TetrisAdapter(), TetrisState()
        base = RicherRLStudent("base", learning_rate=.01, seed=3)
        low, high = base.clone("low"), base.clone("high")
        for agent, later_reward in ((low, 0.0), (high, 5.0)):
            for index, (piece, reward) in enumerate((("T", 0.0), ("I", later_reward))):
                legal = game.legal_actions(state, piece)
                decision = agent.choose_placement(state, piece, legal, learn=True)
                agent.update(AgentExperience(state, piece, decision, reward,
                                             decision.evaluation.next_state, False))
            agent.finish_episode(learned=True)
        self.assertNotEqual(low.weights, high.weights)
        self.assertNotEqual(low.weights[0], high.weights[0])

    def test_evaluation_choice_is_nonlearning(self):
        student = RicherRLStudent("x")
        before = copy.deepcopy(student.serialize_state())
        legal = TetrisAdapter().legal_actions(TetrisState(), "L")
        student.choose_placement(TetrisState(), "L", legal, learn=False, deterministic=True)
        self.assertEqual(before, student.serialize_state())

    def test_reward_contains_only_survival_and_lines(self):
        legal = TetrisAdapter().legal_actions(TetrisState(), "I")
        self.assertEqual(.02, gameplay_reward(legal[0]))


class Day4ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(SMOKE)

    def check(self, mutation=None):
        value = copy.deepcopy(self.result)
        if mutation:
            mutation(value)
        return validate_day4(value)

    def test_valid_smoke_passes(self):
        self.assertEqual("PASS", self.check()["overall"])

    def test_expert_training_call_is_rejected(self):
        self.assertFalse(self.check(lambda x: x["training_audit"].update(expert_calls=1))["checks"]["expert_absent_from_training"])

    def test_expert_reward_is_rejected(self):
        self.assertFalse(self.check(lambda x: x.update(reward="expert regret"))["checks"]["environment_reward_only"])

    def test_unequal_exposure_is_rejected(self):
        self.assertFalse(self.check(lambda x: x["training_audit"]["opportunities"].update(rl=1))["checks"]["equal_exposure"])

    def test_seed_leakage_is_rejected(self):
        def mutate(x):
            x["configuration"]["evaluation_game_seeds"] = tuple(x["configuration"]["evaluation_game_seeds"]) + (x["configuration"]["training_seed"],)
        self.assertFalse(self.check(mutate)["checks"]["training_evaluation_seed_separation"])

    def test_evaluation_mutation_is_rejected(self):
        def mutate(x): x["evaluation_after_states"]["rl@0"]["updates"] += 1
        self.assertFalse(self.check(mutate)["checks"]["evaluation_nonmutating"])

    def test_parameter_only_claim_is_rejected(self):
        def mutate(x): x["success_checks"]["final_exceeds_matched_control"] = False
        self.assertFalse(self.check(mutate)["checks"]["behavior_not_parameters_only"])


if __name__ == "__main__":
    unittest.main()
