"""Unit tests for stem_layer_gate's noise gate + boundary fade."""

import unittest

import numpy as np

from acestep.core.generation.handler.stem_layer_gate import (
    apply_boundary_fade,
    compute_stem_gate_envelope,
)


class ComputeStemGateEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.sr = 48000

    def test_silence_converges_towards_floor(self):
        instrumental = np.zeros(self.sr)  # 1s of true silence throughout
        gate = compute_stem_gate_envelope(instrumental, self.sr, floor=0.15)
        self.assertLess(gate[-1], 0.2)
        self.assertGreaterEqual(gate[-1], 0.15 - 1e-6)

    def test_loud_constant_signal_stays_near_full_gain(self):
        t = np.arange(self.sr) / self.sr
        instrumental = 0.8 * np.sin(2 * np.pi * 220 * t)
        gate = compute_stem_gate_envelope(instrumental, self.sr)
        self.assertGreater(gate[-1], 0.95)

    def test_attack_opens_faster_than_release_closes(self):
        # Silence, then a sudden loud half - how fast the gate reaches near
        # full gain after the jump vs. how far it has fallen back an equal
        # time after a loud-to-silence jump.
        half = self.sr // 2
        t = np.arange(half) / self.sr
        loud = 0.8 * np.sin(2 * np.pi * 220 * t)

        opening = np.concatenate([np.zeros(half), loud])
        gate_opening = compute_stem_gate_envelope(opening, self.sr)
        window = int(self.sr * 0.05)  # 50ms after the jump
        gain_after_opening = gate_opening[half + window]

        closing = np.concatenate([loud, np.zeros(half)])
        gate_closing = compute_stem_gate_envelope(closing, self.sr)
        gain_after_closing = gate_closing[half + window]

        # Attack (20ms time constant) should have the gate almost fully open
        # again within this 50ms window; release (300ms time constant) should
        # have barely moved in the same window - i.e. attack is much faster.
        self.assertGreater(gain_after_opening, 0.9)
        self.assertGreater(gain_after_closing, 0.7)

    def test_never_drops_below_floor(self):
        instrumental = np.zeros(self.sr)
        gate = compute_stem_gate_envelope(instrumental, self.sr, floor=0.2)
        self.assertTrue(np.all(gate >= 0.2 - 1e-9))

    def test_empty_input_returns_empty(self):
        gate = compute_stem_gate_envelope(np.zeros(0), self.sr)
        self.assertEqual(len(gate), 0)


class ApplyBoundaryFadeTests(unittest.TestCase):
    def test_first_and_last_sample_are_zeroed(self):
        mix = np.ones((48000, 2))
        faded = apply_boundary_fade(mix, 48000, fade_ms=25.0)
        self.assertTrue(np.allclose(faded[0], 0.0))
        self.assertTrue(np.allclose(faded[-1], 0.0))

    def test_interior_is_unchanged(self):
        mix = np.full((48000, 2), 0.5)
        faded = apply_boundary_fade(mix, 48000, fade_ms=25.0)
        self.assertTrue(np.allclose(faded[24000], 0.5))

    def test_mono_shape_supported(self):
        mix = np.ones(48000)
        faded = apply_boundary_fade(mix, 48000, fade_ms=25.0)
        self.assertAlmostEqual(faded[0], 0.0)
        self.assertAlmostEqual(faded[-1], 0.0)

    def test_short_signal_does_not_crash(self):
        mix = np.ones((10, 2))
        faded = apply_boundary_fade(mix, 48000, fade_ms=25.0)
        self.assertEqual(faded.shape, mix.shape)


if __name__ == "__main__":
    unittest.main()
