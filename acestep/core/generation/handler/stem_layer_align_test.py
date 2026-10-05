"""Unit tests for stem_layer_align's offset measurement/correction."""

import unittest

import numpy as np

from acestep.core.generation.handler.stem_layer_align import align_to_length, measure_offset_samples


def _test_signal(sr: int, seconds: float = 5.0, seed: int = 0) -> np.ndarray:
    n = int(sr * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * 220 * t) + 0.3 * rng.standard_normal(n)


class MeasureOffsetSamplesTests(unittest.TestCase):
    """Sign-convention and accuracy checks for measure_offset_samples."""

    def setUp(self):
        self.sr = 22050
        self.signal = _test_signal(self.sr)

    def test_leading_probe_returns_positive_offset(self):
        """A probe that is an EARLY preview of the reference leads -> positive offset."""
        d = 1000
        probe = np.concatenate([self.signal[d:], np.zeros(d)])
        self.assertEqual(measure_offset_samples(self.signal, probe, self.sr), d)

    def test_lagging_probe_returns_negative_offset(self):
        """A probe that is a DELAYED copy of the reference lags -> negative offset."""
        d = 500
        probe = np.concatenate([np.zeros(d), self.signal[: len(self.signal) - d]])
        self.assertEqual(measure_offset_samples(self.signal, probe, self.sr), -d)

    def test_identical_signal_returns_zero(self):
        self.assertEqual(measure_offset_samples(self.signal, self.signal.copy(), self.sr), 0)

    def test_offset_bounded_by_max_shift(self):
        """A lag outside max_shift_sec must not be reported beyond that window."""
        d = int(0.3 * self.sr)
        probe = np.concatenate([self.signal[d:], np.zeros(d)])
        measured = measure_offset_samples(self.signal, probe, self.sr, max_shift_sec=0.1)
        self.assertLessEqual(abs(measured), int(0.1 * self.sr))

    def test_empty_input_returns_zero(self):
        self.assertEqual(measure_offset_samples(np.array([]), np.array([]), self.sr), 0)


class AlignToLengthTests(unittest.TestCase):
    """Correction behaviour: the offset reported above must be exactly undone."""

    def setUp(self):
        self.sr = 22050
        self.signal = _test_signal(self.sr)
        self.n = len(self.signal)

    def _stereo(self, mono: np.ndarray) -> np.ndarray:
        return mono[:, None].repeat(2, axis=1)

    def test_corrects_leading_probe(self):
        d = 1000
        probe = self._stereo(np.concatenate([self.signal[d:], np.zeros(d)]))
        aligned = align_to_length(probe, offset_samples=d, target_len=self.n)
        self.assertEqual(aligned.shape, (self.n, 2))
        valid = self.n - d - 5
        corr = np.corrcoef(aligned[:valid, 0], self.signal[:valid])[0, 1]
        self.assertGreater(corr, 0.98)

    def test_corrects_lagging_probe(self):
        d = 500
        probe = self._stereo(np.concatenate([np.zeros(d), self.signal[: self.n - d]]))
        aligned = align_to_length(probe, offset_samples=-d, target_len=self.n)
        self.assertEqual(aligned.shape, (self.n, 2))
        valid = self.n - d - 5
        corr = np.corrcoef(aligned[:valid, 0], self.signal[:valid])[0, 1]
        self.assertGreater(corr, 0.999)

    def test_zero_offset_only_pads_or_trims_length(self):
        probe = self._stereo(self.signal[:-200])  # shorter than target
        aligned = align_to_length(probe, offset_samples=0, target_len=self.n)
        self.assertEqual(aligned.shape, (self.n, 2))
        np.testing.assert_allclose(aligned[-200:, 0], 0.0)

    def test_trims_when_longer_than_target(self):
        probe = self._stereo(np.concatenate([self.signal, self.signal[:300]]))
        aligned = align_to_length(probe, offset_samples=0, target_len=self.n)
        self.assertEqual(aligned.shape, (self.n, 2))


if __name__ == "__main__":
    unittest.main()
