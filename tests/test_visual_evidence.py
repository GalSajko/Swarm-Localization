"""Regression checks for the visual-to-personal-evidence boundary."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import decision_making as dm
from utils import softmax_over_vector


class VisualEvidenceTests(unittest.TestCase):
    def test_equal_scores_are_uniform_and_uninformative(self):
        for score in (-1.0, -0.1, 0.0, 0.5, 1.0):
            with self.subTest(score=score):
                evidence = softmax_over_vector(np.full(400, score))
                np.testing.assert_allclose(evidence, np.full(400, 1 / 400))
                self.assertAlmostEqual(evidence.sum(), 1.0)
                self.assertAlmostEqual(dm._get_entropy_based_confidence(evidence), 0.0)
                theta = frozenset(range(400))
                mass = dm._build_spatial_topk_personal_mass(evidence, theta, (20, 20))
                self.assertAlmostEqual(mass[theta], 1.0)

    def test_common_offset_does_not_change_evidence(self):
        scores = np.linspace(-0.4, 0.4, 400)
        np.testing.assert_allclose(
            softmax_over_vector(scores), softmax_over_vector(scores + 0.5)
        )

    def test_relative_weights_keep_existing_temperature(self):
        scores = np.array([0.10, 0.12, 0.15])
        evidence = softmax_over_vector(scores)
        self.assertTrue(np.all(np.diff(evidence) > 0))
        self.assertAlmostEqual(evidence[2] / evidence[0], np.exp(1.0))

    def test_dominant_match_produces_informative_evidence(self):
        scores = np.zeros(400)
        scores[123] = 1.0
        evidence = softmax_over_vector(scores)
        self.assertEqual(int(np.argmax(evidence)), 123)
        self.assertGreater(evidence[123], 0.999)
        self.assertGreater(dm._get_entropy_based_confidence(evidence), 0.99)

    def test_small_temperature_remains_finite_and_normalized(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            evidence = softmax_over_vector([-1.0, 0.0, 1.0], tau=0.001)
        self.assertTrue(np.all(np.isfinite(evidence)))
        self.assertTrue(np.all(evidence >= 0))
        self.assertAlmostEqual(evidence.sum(), 1.0)
        self.assertEqual(int(np.argmax(evidence)), 2)

    def test_invalid_inputs_are_rejected(self):
        for scores in ([], [[0.1, 0.2]], [np.nan], [np.inf], [-np.inf]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                softmax_over_vector(scores)
        for tau in (0.0, -0.1, np.nan, np.inf):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                softmax_over_vector([0.1, 0.2], tau=tau)


if __name__ == "__main__":
    unittest.main()
