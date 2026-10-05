"""Unit tests for stem_layer_separate - demucs itself is mocked out entirely
(no GPU/checkpoint download in CI), per AGENTS.md ("Mock GPU, filesystem,
network, and external services where possible")."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import acestep.core.generation.handler.stem_layer_separate as stem_layer_separate


class _FakeSeparator:
    samplerate = 44100

    def separate_audio_file(self, path):
        channels, samples = 2, 44100
        separated = {
            "drums": torch.full((channels, samples), 0.1),
            "bass": torch.full((channels, samples), 0.2),
            "other": torch.full((channels, samples), 0.3),
            "vocals": torch.full((channels, samples), 0.9),
        }
        return torch.zeros(channels, samples), separated


class SeparateBackgroundStemsTests(unittest.TestCase):
    def setUp(self):
        stem_layer_separate._separator_cache.clear()
        self.fake_demucs_api = types.ModuleType("demucs.api")
        self.fake_demucs_api.Separator = MagicMock(return_value=_FakeSeparator())
        self.fake_demucs_pkg = types.ModuleType("demucs")
        patcher = patch.dict(
            sys.modules, {"demucs": self.fake_demucs_pkg, "demucs.api": self.fake_demucs_api}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_vocals_are_excluded(self):
        stems = stem_layer_separate.separate_background_stems("/tmp/song.mp3", device="cpu")
        self.assertNotIn("vocals", stems)
        self.assertEqual(set(stems.keys()), {"drums", "bass", "other"})

    def test_output_shape_is_samples_first(self):
        stems = stem_layer_separate.separate_background_stems("/tmp/song.mp3", device="cpu")
        self.assertEqual(stems["drums"].shape, (44100, 2))

    def test_separator_is_cached_across_calls(self):
        stem_layer_separate.separate_background_stems("/tmp/a.mp3", device="cpu")
        stem_layer_separate.separate_background_stems("/tmp/b.mp3", device="cpu")
        self.assertEqual(self.fake_demucs_api.Separator.call_count, 1)

    def test_get_separator_sample_rate(self):
        self.assertEqual(stem_layer_separate.get_separator_sample_rate(device="cpu"), 44100)


if __name__ == "__main__":
    unittest.main()
