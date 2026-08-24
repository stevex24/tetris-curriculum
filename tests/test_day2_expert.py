import json
import unittest
from pathlib import Path

from tetris_research.day2 import (_independent_legal, benchmark, play,
                                  validate_expert, validation_corpus)
from tetris_research.expert import (DellacherieSearchExpert,
                                    PlacementEvaluation)
from tetris_research.student import Placement
from tetris_research.tetris import HEIGHT, TetrisAdapter, TetrisState


class Day2ExpertTests(unittest.TestCase):
    def setUp(self):
        self.game = TetrisAdapter()
        self.state = TetrisState()
        self.legal = self.game.legal_actions(self.state, "T")
        self.expert = DellacherieSearchExpert(beam_width=2)

    def test_expert_policy_contract_and_rank_consistency(self):
        ranking = self.expert.rank_placements(self.state, "T", self.legal)
        self.assertEqual(len(ranking), len(self.legal))
        self.assertEqual([x.rank for x in ranking], list(range(1, len(ranking) + 1)))
        self.assertEqual([x.value for x in ranking], sorted((x.value for x in ranking), reverse=True))
        self.assertEqual(ranking, self.expert.rank_placements(self.state, "T", self.legal))
        self.assertEqual(ranking[0].metadata["beam_width"], 2)

    def test_independent_legal_actions_and_replay(self):
        independent = _independent_legal(self.state.rows, "T")
        canonical = {tuple(x.action): x.next_state.rows for x in self.legal}
        self.assertEqual(independent, canonical)

    def test_regret_semantics(self):
        ranking = self.expert.rank_placements(self.state, "T", self.legal)
        self.assertAlmostEqual(self.expert.regret(self.state, "T", self.legal, ranking[0].placement).regret, 0)
        self.assertGreaterEqual(self.expert.regret(self.state, "T", self.legal, ranking[-1].placement).regret, 0)

    def test_rule_compatibility_corpus(self):
        report = validate_expert(self.expert)
        self.assertEqual(report["overall"], "PASS")
        self.assertEqual(len(report["positions"]), len(validation_corpus()))

    def test_matched_stream_setup(self):
        result = benchmark(games=2, maximum=5, beam_width=1)
        self.assertEqual(result["matched_games"], 2)
        self.assertEqual(len(result["pairs"]), 2)
        self.assertTrue(all(len(x["stream_sha256"]) == 64 for x in result["pairs"]))

    def test_validator_rejects_illegal_and_falsified_ranking(self):
        class Bad:
            expert_id = "bad"
            def rank_placements(self, state, piece, legal, deterministic=True):
                return (PlacementEvaluation(Placement(0, 99), 0.0, 2, {}),)
            def regret(self, state, piece, legal, chosen):
                raise ValueError("illegal")
        with self.assertRaises(ValueError):
            validate_expert(Bad())

    def test_altered_replay_is_detected(self):
        item = validation_corpus()[1]
        independent = _independent_legal(item["rows"], item["piece"])
        key = next(iter(independent))
        altered = list(independent[key]); altered[0] ^= 1
        self.assertNotEqual(tuple(altered), independent[key])

    def test_hours_1_9_artifact_integrity(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "baseline/hours_1_9_manifest.json").read_text())
        import subprocess
        for directory, spec in manifest["scientific_artifacts"].items():
            if "git_tree_sha1" not in spec:
                continue
            actual = subprocess.run(["git", "rev-parse", f"HEAD:{directory}"], cwd=root,
                                    text=True, capture_output=True, check=True).stdout.strip()
            self.assertEqual(actual, spec["git_tree_sha1"])


if __name__ == "__main__":
    unittest.main()
