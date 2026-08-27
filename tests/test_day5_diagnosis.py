import copy
import unittest

from tetris_research.day5 import _policy, collect_history, run
from tetris_research.day5_validator import validate_day5
from tetris_research.diagnosis import FEATURE_FAMILIES, diagnose_history
from tetris_research.richer_student import FEATURE_NAMES


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = collect_history(_policy("hole", "hole_management"), (91, 92), 25)

    def test_mapping_covers_every_feature_once(self):
        flat = [feature for values in FEATURE_FAMILIES.values() for feature in values]
        self.assertCountEqual(FEATURE_NAMES, flat)
        self.assertEqual(len(flat), len(set(flat)))

    def test_history_is_replayable_and_has_required_fields(self):
        profile = diagnose_history(self.history)
        self.assertEqual(sum(len(x["steps"]) for x in self.history), profile["decision_count"])
        broken = copy.deepcopy(self.history)
        broken[0]["steps"][0]["board_after"][0] ^= 1
        with self.assertRaises(ValueError):
            diagnose_history(broken)

    def test_hidden_weights_and_irrelevant_fields_do_not_change_profile(self):
        altered = copy.deepcopy(self.history)
        for game in altered:
            game["weights"] = [999] * 18
            game["training_only"] = {"weights": [-999] * 18}
        self.assertEqual(diagnose_history(self.history), diagnose_history(altered))

    def test_zero_regret_decisions_do_not_create_weaknesses(self):
        # Reuse the oracle's choice as a purely behavioral history fixture.
        from tetris_research.expert import DellacherieSearchExpert
        from tetris_research.richer_student import richer_features
        from tetris_research.tetris import TetrisAdapter, TetrisState
        game, state, piece = TetrisAdapter(), TetrisState(), "T"
        legal = game.legal_actions(state, piece)
        best = DellacherieSearchExpert(beam_width=1).preferred_placement(state, piece, legal).placement
        selected = next(x for x in legal if tuple(x.action) == (best.rotation, best.x))
        step = {"step_index": 0, "board_before": list(state.rows), "piece": piece,
                "student_action": [best.rotation, best.x], "learning_enabled": False,
                "feature_vector": list(richer_features(state, piece, selected)),
                "board_after": list(selected.next_state.rows), "lines_cleared": 0}
        profile = diagnose_history([{"game_id": "one", "steps": [step]}])
        self.assertTrue(all(row["score"] == 0 for row in profile["families"].values()))

    def test_cross_game_factor_suppresses_isolated_blunder(self):
        one = diagnose_history(self.history[:1])
        padded = copy.deepcopy(self.history)
        # Make the second game oracle-optimal so evidence remains confined to one game.
        from tetris_research.expert import DellacherieSearchExpert
        from tetris_research.richer_student import richer_features
        from tetris_research.tetris import TetrisAdapter, TetrisState
        expert, adapter = DellacherieSearchExpert(beam_width=1), TetrisAdapter()
        for step in padded[1]["steps"]:
            state = TetrisState(tuple(step["board_before"])); legal = adapter.legal_actions(state, step["piece"])
            best = expert.preferred_placement(state, step["piece"], legal).placement
            selected = next(x for x in legal if tuple(x.action) == (best.rotation, best.x))
            step["student_action"] = [best.rotation, best.x]
            step["feature_vector"] = list(richer_features(state, step["piece"], selected))
            step["board_after"] = list(selected.next_state.rows)
            # A changed action invalidates later sequential boards; retain only first step.
            padded[1]["steps"] = padded[1]["steps"][:1]
            break
        two = diagnose_history(padded)
        self.assertLessEqual(max(x["score"] for x in two["families"].values()),
                             max(x["score"] for x in one["families"].values()))


class Day5ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(seeds=(101, 102), maximum=30)

    def test_controlled_profiles_recover_distinct_weaknesses(self):
        self.assertTrue(all(self.result["intended_top_two"].values()))
        self.assertGreaterEqual(self.result["profile_l1_distance"], .01)

    def test_student_state_is_not_mutated(self):
        self.assertTrue(all(all(audit.values()) for audit in self.result["mutation_audit"].values()))

    def test_independent_validator_passes(self):
        self.assertEqual("PASS", validate_day5(self.result)["overall"])

    def test_validator_rejects_oracle_before_decision(self):
        bad = copy.deepcopy(self.result)
        bad["profiles"]["hole_weak"]["event_order"][0]["events"].reverse()
        self.assertFalse(validate_day5(bad)["checks"]["student_decision_precedes_oracle"])


if __name__ == "__main__":
    unittest.main()
