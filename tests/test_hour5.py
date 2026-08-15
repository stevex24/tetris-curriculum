import dataclasses
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tetris_research.hour5 import (CALIBRATION_SEED, DIAGNOSTIC_DIMENSIONS,
                                   DIAGNOSTIC_POPULATION_SEED, LearnerProfile, diagnose,
                                   fit_calibration, measure_profile, normalize_profile,
                                   synthetic_history, tutorial_for)


def profile(identifier, holes, height, bumpiness):
    return identifier, LearnerProfile(1500.0, holes, height, bumpiness, 0.0, 0, 40)


class Hour5ProfileTests(unittest.TestCase):
    def setUp(self):
        rows = [profile("a", 20, 8, 10), profile("b", 30, 10, 15),
                profile("c", 40, 12, 20), profile("d", 50, 14, 25)]
        self.calibration = fit_calibration(rows, CALIBRATION_SEED, "test calibration split", 40)

    def test_profile_contains_observables_only(self):
        names = {field.name for field in dataclasses.fields(LearnerProfile)}
        self.assertEqual(names, {"ability_elo", "mean_holes", "mean_max_height",
                                 "mean_bumpiness", "line_clear_rate", "lines_cleared",
                                 "placements_observed", "normalized_weakness"})
        self.assertFalse(any("weight" in name or "policy" in name for name in names))

    def test_calibration_is_deterministic_and_documents_z_score(self):
        rows = [profile("a", 20, 8, 10), profile("b", 30, 10, 15),
                profile("c", 40, 12, 20), profile("d", 50, 14, 25)]
        self.assertEqual(self.calibration,
                         fit_calibration(rows, CALIBRATION_SEED, "test calibration split", 40))
        at_mean = LearnerProfile(1500, self.calibration.means["hole_management"],
                                 self.calibration.means["height_management"],
                                 self.calibration.means["surface_management"], 0, 0, 40)
        self.assertTrue(all(abs(x) < 1e-12 for x in
                            normalize_profile(at_mean, self.calibration).normalized_weakness.values()))
        one_sd = dataclasses.replace(at_mean, mean_holes=at_mean.mean_holes +
                                    self.calibration.sample_sds["hole_management"])
        self.assertAlmostEqual(1.0, normalize_profile(one_sd, self.calibration).
                               normalized_weakness["hole_management"])

    def test_calibration_and_diagnostic_seed_domains_are_separate(self):
        self.assertNotEqual(CALIBRATION_SEED, DIAGNOSTIC_POPULATION_SEED)
        with self.assertRaises(ValueError):
            fit_calibration([profile("same", 1, 2, 3), profile("same", 2, 3, 4)],
                            1, "bad", 40)

    def test_known_constructs_recover_weakness(self):
        realistic = fit_calibration([profile("a", 27, 11, 15), profile("b", 31, 12, 18),
                                     profile("c", 35, 13, 21)], 2, "fixed test reference", 40)
        for index, kind in enumerate(DIAGNOSTIC_DIMENSIONS):
            measured = measure_profile(synthetic_history(kind, 800 + index), 1500)
            self.assertEqual(kind, diagnose(normalize_profile(measured, realistic)).primary)

    def test_mixed_profile_has_reduced_confidence(self):
        mixed = LearnerProfile(1500, 0, 0, 0, 0, 0, 40,
                               {"hole_management": 1.1, "height_management": 1.0,
                                "surface_management": -0.2})
        result = diagnose(mixed)
        self.assertTrue(result.mixed)
        self.assertEqual("low/mixed", result.confidence)
        self.assertAlmostEqual(0.1, result.margin)

    def test_hidden_weights_cannot_affect_measurement_or_diagnosis(self):
        history = synthetic_history("hole_management", 91)
        history[0]["final_weights"] = [999] * 4
        first = diagnose(normalize_profile(measure_profile(history, 1500), self.calibration))
        history[0]["final_weights"] = [-999] * 4
        second = diagnose(normalize_profile(measure_profile(history, 1500), self.calibration))
        self.assertEqual(first, second)

    def test_diagnoses_map_to_distinct_existing_tutorials(self):
        tutorials = set()
        for key in DIAGNOSTIC_DIMENSIONS:
            scores = {name: 0.0 for name in DIAGNOSTIC_DIMENSIONS}; scores[key] = 2.0
            measured = LearnerProfile(1500, 1, 1, 1, 0, 0, 40, scores)
            tutorials.add(tutorial_for(diagnose(measured), 1500, 7)["tutorial_type"])
        self.assertEqual(tutorials, {"hole_avoidance", "stack_height", "bumpiness"})

    def test_hour3_and_hour4_artifacts_equal_hour4_commit(self):
        root = Path(__file__).parents[1]
        for directory in ("artifacts/hour3", "artifacts/hour4"):
            names = subprocess.run(["git", "ls-tree", "-r", "--name-only", "bce5132", directory],
                                   cwd=root, text=True, capture_output=True, check=True).stdout.splitlines()
            for name in names:
                expected = subprocess.run(["git", "show", f"bce5132:{name}"], cwd=root,
                                          capture_output=True, check=True).stdout
                self.assertEqual(hashlib.sha256(expected).hexdigest(),
                                 hashlib.sha256((root / name).read_bytes()).hexdigest(), name)


if __name__ == "__main__":
    unittest.main()
