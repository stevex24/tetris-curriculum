import copy
import unittest

from tetris_research.day3 import SMOKE, collect_states, evaluate_decisions, run, train_matched
from tetris_research.day3_validator import validate_day3
from tetris_research.expert import DELLACHERIE_WEIGHTS, DellacherieSearchExpert
from tetris_research.imitation_student import LinearImitationStudent
from tetris_research.tetris import TetrisAdapter, TetrisState


class Day3StudentTests(unittest.TestCase):
    def test_exact_clones_have_identical_state_and_decisions(self):
        initial = LinearImitationStudent("weak")
        a, b = initial.clone("a"), initial.clone("b")
        self.assertEqual(a.weights, b.weights)
        legal = TetrisAdapter().legal_actions(TetrisState(), "T")
        self.assertEqual(a.choose_placement(TetrisState(), "T", legal, deterministic=True).placement,
                         b.choose_placement(TetrisState(), "T", legal, deterministic=True).placement)

    def test_cross_entropy_update_is_student_owned_and_not_coefficient_copy(self):
        student = LinearImitationStudent("student")
        state, piece = TetrisState(), "T"
        legal = TetrisAdapter().legal_actions(state, piece)
        label = DellacherieSearchExpert(beam_width=1).preferred_placement(state, piece, legal).placement
        student.learn_from_label(state, piece, legal, label)
        self.assertEqual(student.updates, 1)
        self.assertFalse(set(student.weights) & set(DELLACHERIE_WEIGHTS.values()))
        self.assertIsNone(student.serialize_state()["expert_parameters"])

    def test_matched_training_uses_same_opportunity_and_update_budget(self):
        states = collect_states((31,), 3)
        trained, audit = train_matched(LinearImitationStudent("weak"), states,
                                       DellacherieSearchExpert(beam_width=1), control_seed=9)
        self.assertEqual(audit["opportunities"], {"taught": 3, "random_label_control": 3})
        self.assertEqual(audit["updates"], audit["opportunities"])
        self.assertEqual({x.updates for x in trained.values()}, {3})

    def test_evaluation_is_nonlearning_and_student_chooses_before_expert(self):
        class OrderingExpert:
            def __init__(self): self.calls = 0
            def rank_placements(self, state, piece, legal):
                self.calls += 1
                return DellacherieSearchExpert(beam_width=1).rank_placements(state, piece, legal)
        expert = OrderingExpert()
        class GuardedStudent(LinearImitationStudent):
            def choose_placement(self, *args, **kwargs):
                if expert.calls:
                    raise AssertionError("expert was queried before student choice")
                return super().choose_placement(*args, **kwargs)
        students = {"x": GuardedStudent("x")}
        before = copy.deepcopy(students["x"].serialize_state())
        metrics, trace = evaluate_decisions(students, [(TetrisState(), "I")], expert)
        self.assertEqual(before, students["x"].serialize_state())
        self.assertEqual(expert.calls, 1)
        self.assertEqual(trace[0]["event_order"], ["student_choose:x", "external_expert_score"])
        self.assertIn("held_out_mean_regret", metrics["x"])


class Day3AdversarialValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(SMOKE)
        # The tiny smoke run is for structure; force only the claimed outcome fixture
        # so each adversarial mutation can target one independent guard.
        cls.result["knowledge_transfer_success"] = True
        cls.result["success_checks"] = {name: True for name in cls.result["success_checks"]}

    def check(self, mutation=None):
        value = copy.deepcopy(self.result)
        if mutation: mutation(value)
        return validate_day3(value)

    def test_valid_structural_fixture_passes(self):
        self.assertEqual("PASS", self.check()["overall"])

    def test_expert_initialization_or_copy_is_rejected(self):
        def mutate(x): x["trained_states"]["taught"]["weights"][0] = next(iter(DELLACHERIE_WEIGHTS.values()))
        self.assertFalse(self.check(mutate)["checks"]["no_expert_coefficients_in_student"])

    def test_unequal_budget_is_rejected(self):
        def mutate(x): x["training_audit"]["opportunities"]["taught"] += 1
        self.assertFalse(self.check(mutate)["checks"]["equal_training_budgets"])

    def test_learning_during_evaluation_is_rejected(self):
        def mutate(x): x["evaluation_after_states"]["taught"]["updates"] += 1
        self.assertFalse(self.check(mutate)["checks"]["evaluation_learning_disabled"])

    def test_state_leakage_is_rejected(self):
        def mutate(x): x["evaluation_state_hashes"][0] = x["training_audit"]["state_hashes"][0]
        self.assertFalse(self.check(mutate)["checks"]["training_evaluation_state_separation"])

    def test_expert_query_before_choice_is_rejected(self):
        def mutate(x): x["decision_trace"][0]["event_order"].reverse()
        self.assertFalse(self.check(mutate)["checks"]["expert_absent_during_student_choice"])

    def test_parameter_change_without_improvement_is_rejected(self):
        def mutate(x): x["success_checks"]["agreement_gain_vs_control_10pp"] = False
        self.assertFalse(self.check(mutate)["checks"]["behavior_not_parameter_only"])


if __name__ == "__main__":
    unittest.main()
