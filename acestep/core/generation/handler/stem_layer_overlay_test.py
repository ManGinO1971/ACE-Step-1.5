"""Unit tests for stem_layer_overlay's gating and fail-safe orchestration.

External/heavy dependencies (demucs model load, GPU) are mocked out per
AGENTS.md - these tests only exercise the gating logic and the
fail-safe-returns-original-audio contract, not real separation quality
(covered instead by the prototype validation done before this was wired in).
"""

import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from acestep.core.generation.handler.stem_layer_overlay import apply_stem_layering, should_apply_stem_layering


def _params(**overrides):
    defaults = dict(task_type="text2music", flow_edit_morph=True, instrumental=True, src_audio="/tmp/x.mp3")
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class ShouldApplyStemLayeringTests(unittest.TestCase):
    def test_true_for_instrumental_flow_edit_morph_job(self):
        self.assertTrue(should_apply_stem_layering(_params()))

    def test_false_when_not_flow_edit_morph(self):
        self.assertFalse(should_apply_stem_layering(_params(flow_edit_morph=False)))

    def test_false_when_not_instrumental(self):
        self.assertFalse(should_apply_stem_layering(_params(instrumental=False)))

    def test_false_when_not_text2music(self):
        self.assertFalse(should_apply_stem_layering(_params(task_type="cover")))

    def test_false_when_no_src_audio(self):
        self.assertFalse(should_apply_stem_layering(_params(src_audio=None)))


class ApplyStemLayeringFailSafeTests(unittest.TestCase):
    def setUp(self):
        self.sr = 48000
        self.audio = torch.zeros(2, self.sr)  # 1s stereo silence

    @patch("acestep.core.generation.handler.stem_layer_overlay.separate_background_stems")
    def test_returns_original_audio_on_separation_failure(self, mock_separate):
        mock_separate.side_effect = RuntimeError("no GPU / checkpoint unavailable")
        result = apply_stem_layering(self.audio, self.sr, "/tmp/original.mp3")
        self.assertTrue(torch.equal(result, self.audio))

    @patch("acestep.core.generation.handler.stem_layer_overlay.separate_background_stems")
    def test_returns_original_audio_when_no_stems_found(self, mock_separate):
        mock_separate.return_value = {}
        result = apply_stem_layering(self.audio, self.sr, "/tmp/original.mp3")
        self.assertTrue(torch.equal(result, self.audio))

    @patch("acestep.core.generation.handler.stem_layer_overlay.get_separator_sample_rate")
    @patch("acestep.core.generation.handler.stem_layer_overlay.separate_background_stems")
    def test_overlay_runs_end_to_end_with_matching_sample_rate(self, mock_separate, mock_sr):
        mock_sr.return_value = self.sr
        quiet_stem = np.full((self.sr, 2), 0.01, dtype=np.float32)
        mock_separate.return_value = {"drums": quiet_stem, "bass": quiet_stem, "other": quiet_stem}
        result = apply_stem_layering(self.audio, self.sr, "/tmp/original.mp3")
        self.assertEqual(result.shape, self.audio.shape)
        # Silence + quiet stems should no longer be exactly silent.
        self.assertFalse(torch.equal(result, self.audio))
        self.assertLessEqual(torch.max(torch.abs(result)).item(), 0.92 + 1e-4)


if __name__ == "__main__":
    unittest.main()
