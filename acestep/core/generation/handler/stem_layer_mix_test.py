"""Unit tests for stem_layer_mix's summing and safety limiter."""

import unittest

import numpy as np

from acestep.core.generation.handler.stem_layer_mix import (
    DEFAULT_STEM_GAINS,
    apply_safety_limiter,
    mix_stems_into_instrumental,
)


class MixStemsIntoInstrumentalTests(unittest.TestCase):
    def setUp(self):
        self.n = 1000
        self.instrumental = np.full((self.n, 2), 0.1)
        self.stem = np.full((self.n, 2), 0.2)

    def test_applies_default_gains(self):
        mixed = mix_stems_into_instrumental(self.instrumental, {"drums": self.stem})
        expected = 0.1 + 0.2 * DEFAULT_STEM_GAINS["drums"]
        np.testing.assert_allclose(mixed, expected)

    def test_gain_override_takes_precedence(self):
        mixed = mix_stems_into_instrumental(self.instrumental, {"drums": self.stem}, gains={"drums": 0.5})
        np.testing.assert_allclose(mixed, 0.1 + 0.2 * 0.5)

    def test_unknown_gain_key_is_ignored_without_error(self):
        mixed = mix_stems_into_instrumental(self.instrumental, {"drums": self.stem}, gains={"made_up": 2.0})
        expected = 0.1 + 0.2 * DEFAULT_STEM_GAINS["drums"]
        np.testing.assert_allclose(mixed, expected)

    def test_zero_gain_contributes_nothing(self):
        mixed = mix_stems_into_instrumental(self.instrumental, {"drums": self.stem}, gains={"drums": 0.0})
        np.testing.assert_allclose(mixed, self.instrumental)

    def test_multiple_stems_sum(self):
        stems = {"drums": self.stem, "bass": self.stem, "other": self.stem}
        mixed = mix_stems_into_instrumental(self.instrumental, stems)
        expected = 0.1 + 0.2 * (
            DEFAULT_STEM_GAINS["drums"] + DEFAULT_STEM_GAINS["bass"] + DEFAULT_STEM_GAINS["other"]
        )
        np.testing.assert_allclose(mixed, expected)

    def test_none_stem_value_is_skipped(self):
        mixed = mix_stems_into_instrumental(self.instrumental, {"drums": None})
        np.testing.assert_allclose(mixed, self.instrumental)


class ApplySafetyLimiterTests(unittest.TestCase):
    def test_below_limit_is_unchanged(self):
        quiet = np.full((100, 2), 0.3)
        limited = apply_safety_limiter(quiet, limit=0.92)
        np.testing.assert_allclose(limited, quiet)

    def test_above_limit_is_brought_under_ceiling(self):
        loud = np.full((100, 2), 1.5)
        limited = apply_safety_limiter(loud, limit=0.92)
        self.assertLessEqual(np.max(np.abs(limited)), 0.92 + 1e-6)

    def test_preserves_relative_dynamics_below_knee(self):
        """Samples well under the knee should keep their relative proportions."""
        mix = np.array([[0.1, 0.1], [0.2, 0.2], [2.0, 2.0]])
        limited = apply_safety_limiter(mix, limit=0.92, knee=0.1)
        self.assertAlmostEqual(limited[0, 0] / limited[1, 0], 0.1 / 0.2, places=4)

    def test_silence_stays_silent(self):
        silence = np.zeros((100, 2))
        limited = apply_safety_limiter(silence, limit=0.92)
        np.testing.assert_allclose(limited, silence)


if __name__ == "__main__":
    unittest.main()
