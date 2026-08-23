import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tetris_research.agent import LearningAgent
from tetris_research.hour4 import evaluate
from tetris_research.legacy_student import FourFeatureStudentAdapter
from tetris_research.tetris import HEIGHT, TetrisAdapter, TetrisState
from tetris_research.validator import _trajectory, validate_agent_bundle

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "baseline/hours_1_9_manifest.json"


class Day1ArchitectureTests(unittest.TestCase):
    def test_adapter_matches_legacy_deterministic_placements(self):
        game = TetrisAdapter()
        boards = [game.initial_state(), TetrisState((0b1110010111, 0b1100011111) + (0,) * 18)]
        for index, board in enumerate(boards):
            for piece in ("I", "T", "L"):
                legacy = LearningAgent("legacy", [.2, .1, .15, .25], seed=100 + index)
                choices = game.legal_actions(board, piece)
                expected = max(choices, key=lambda c: sum(w * f for w, f in zip(legacy.weights, c.features)))
                actual = FourFeatureStudentAdapter(legacy).choose_placement(
                    board, piece, choices, deterministic=True).evaluation
                self.assertEqual(expected.action, actual.action)
                self.assertEqual(expected.next_state, actual.next_state)

    def test_adapter_stochastic_path_is_exact_and_clones_are_independent(self):
        game, board = TetrisAdapter(), TetrisState()
        choices = game.legal_actions(board, "T")
        legacy = LearningAgent("legacy", [.2, .1, .15, .25], seed=41)
        control = copy.deepcopy(legacy)
        adapted = FourFeatureStudentAdapter(legacy)
        self.assertEqual(control.choose(choices, learn=True).action,
                         adapted.choose_placement(board, "T", choices, learn=True).evaluation.action)
        self.assertEqual(control.weights, legacy.weights)
        clone = adapted.clone("clone")
        clone.learner.weights[0] += 99
        clone.learner._rng.random()
        self.assertNotEqual(clone.learner.weights, adapted.learner.weights)
        self.assertNotEqual(clone.learner._rng.getstate(), adapted.learner._rng.getstate())

    def test_evaluation_through_adapter_is_nonlearning(self):
        student = FourFeatureStudentAdapter(LearningAgent("x", [.2, .1, .15, .25], seed=8))
        before = copy.deepcopy(student.serialize_state())
        evaluate(student, [7001, 7002], 12)
        self.assertEqual(before, student.serialize_state())

    def test_hours_3_through_9_artifacts_match_frozen_commit(self):
        manifest = json.loads(MANIFEST.read_text())
        for directory, item in manifest["scientific_artifacts"].items():
            if "git_tree_sha1" not in item:
                continue
            self.assertEqual(item["git_tree_sha1"], subprocess.run(
                ["git", "rev-parse", f"{manifest['git_head']}:{directory}"], cwd=ROOT,
                text=True, capture_output=True, check=True).stdout.strip())
            listing = subprocess.run(["git", "ls-tree", "-r", manifest["git_head"], directory],
                                     cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
            for line in listing:
                metadata, relative = line.split("\t", 1)
                expected_blob = metadata.split()[2]
                actual_blob = subprocess.run(["git", "hash-object", str(ROOT / relative)], cwd=ROOT,
                                             text=True, capture_output=True, check=True).stdout.strip()
                self.assertEqual(expected_blob, actual_blob, relative)


class ValidatorFailureModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        payload = self.root / "payload.txt"
        payload.write_text("unaltered replay attachment\n")
        initial_base = FourFeatureStudentAdapter(
            LearningAgent("baseline", [.2, .1, .15, .25], temperature=.05, seed=17))
        initial = {c: dict(initial_base.clone(c).serialize_state())
                   for c in ("control", "rating_only", "rating_history")}
        trained = {
            "control": dict(FourFeatureStudentAdapter(LearningAgent(
                "control", [4, .1, .1, .1], temperature=.05, seed=31)).serialize_state()),
            "rating_only": dict(FourFeatureStudentAdapter(LearningAgent(
                "rating_only", [.1, 4, .1, .1], temperature=.05, seed=31)).serialize_state()),
            "rating_history": dict(FourFeatureStudentAdapter(LearningAgent(
                "rating_history", [.1, .1, 4, .1], temperature=.05, seed=31)).serialize_state()),
        }
        corpus = [
            {"rows": [0] * HEIGHT, "piece": "T"},
            {"rows": [0b1110010111, 0b1100011111] + [0] * 18, "piece": "I"},
            {"rows": [0b1010101010, 0b0010101000] + [0] * 18, "piece": "L"},
        ]
        pieces = list("TILSZJO" * 6)
        expected = {c: _trajectory(FourFeatureStudentAdapter.from_state(trained[c]), pieces, 36,
                                   deterministic=True)
                    for c in trained}
        self.bundle = {
            "experiment": "validator-test-fixture", "repository_commit": "fixture",
            "agents": {"initial": initial, "trained": trained},
            "training_budgets": {c: 40 for c in trained},
            "seeds": {"training": [1, 2, 3], "evaluation": [101, 102]},
            "corpus": corpus,
            "streams": [{"id": "held-out-101", "pieces": pieces, "max_placements": 36,
                         "deterministic": True, "expected": expected}],
            "artifact_hashes": {"payload.txt": hashlib.sha256(payload.read_bytes()).hexdigest()},
        }

    def tearDown(self):
        self.temp.cleanup()

    def check(self, bundle=None):
        return validate_agent_bundle(self.bundle if bundle is None else bundle, self.root)

    def test_valid_bundle_passes(self):
        report = self.check()
        self.assertEqual("PASS", report["overall_status"], report)

    def test_reusing_same_trained_agent_fails(self):
        broken = copy.deepcopy(self.bundle)
        broken["agents"]["trained"]["rating_history"] = copy.deepcopy(
            broken["agents"]["trained"]["control"])
        broken["agents"]["trained"]["rating_history"]["agent_id"] = "rating_history"
        self.assertEqual("FAIL", self.check(broken)["checks"]["trained_agent_states_differ"]["status"])

    def test_shared_clone_mutation_fails_initial_match(self):
        broken = copy.deepcopy(self.bundle)
        broken["agents"]["initial"]["rating_only"]["weights"][0] += .01
        self.assertEqual("FAIL", self.check(broken)["checks"]["baseline_agents_identical"]["status"])

    def test_training_seed_reused_for_evaluation_fails(self):
        broken = copy.deepcopy(self.bundle)
        broken["seeds"]["evaluation"].append(2)
        self.assertEqual("FAIL", self.check(broken)["checks"]["training_evaluation_seeds_disjoint"]["status"])

    def test_altered_saved_replay_fails(self):
        broken = copy.deepcopy(self.bundle)
        broken["streams"][0]["expected"]["control"]["actions"][0][0] += 1
        self.assertEqual("FAIL", self.check(broken)["checks"]["saved_replay_reconstruction_matches"]["status"])

    def test_altered_artifact_fails_hash(self):
        (self.root / "payload.txt").write_text("tampered\n")
        self.assertEqual("FAIL", self.check()["checks"]["artifact_hashes_valid"]["status"])

    def test_identical_rankings_fail_behavior_claim_even_if_states_differ(self):
        broken = copy.deepcopy(self.bundle)
        common = broken["agents"]["trained"]["control"]["weights"]
        for index, condition in enumerate(("control", "rating_only", "rating_history")):
            broken["agents"]["trained"][condition]["weights"] = list(common)
            broken["agents"]["trained"][condition]["games_learned"] = index
        report = self.check(broken)
        self.assertEqual("PASS", report["checks"]["trained_agent_states_differ"]["status"])
        self.assertEqual("FAIL", report["checks"]["deterministic_preferences_differ"]["status"])


if __name__ == "__main__":
    unittest.main()
