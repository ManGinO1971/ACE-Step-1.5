"""Automatic background-stem overlay for flow_edit_morph instrumentals.

User-requested feature (Oct 2026): when "Instrumental erzeugen" uses
flow_edit_morph to strip the vocal from an already-generated song, some
background instruments come through quieter than in the original mix (the
model partially "ducked" them to make room for the vocal it is now
removing). Confirmed fix, prototyped and approved by the user before this
was wired into the pipeline: separate the ORIGINAL (with-vocals) song into
stems via htdemucs_ft, discard the vocal stem, and layer drums/bass/other
back onto the freshly generated instrumental - after correcting the small,
constant timing offset separation can introduce (see stem_layer_align).

This always runs automatically for every flow_edit_morph instrumental (the
user explicitly asked for "always automatic", not an opt-in toggle) and
fails safe: any error returns the untouched instrumental rather than
breaking generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from loguru import logger

from acestep.core.generation.handler.stem_layer_align import align_to_length, measure_offset_samples
from acestep.core.generation.handler.stem_layer_mix import apply_safety_limiter, mix_stems_into_instrumental
from acestep.core.generation.handler.stem_layer_separate import (
    get_separator_sample_rate,
    separate_background_stems,
)


def should_apply_stem_layering(params: Any) -> bool:
    """Whether a given generation job is an instrumental flow_edit_morph job.

    Args:
        params: The ``GenerationParams`` for this job.

    Returns:
        bool: True only for ``text2music`` + ``flow_edit_morph=True`` +
        ``instrumental=True`` jobs that actually carry a source audio path -
        exactly the "Instrumental erzeugen" path, never any other task type.
    """
    return bool(
        params.task_type == "text2music"
        and params.flow_edit_morph
        and params.instrumental
        and params.src_audio
    )


def _to_numpy_samples_first(audio_tensor: torch.Tensor) -> np.ndarray:
    """``[channels, samples]`` torch tensor -> ``[samples, channels]`` numpy."""
    return audio_tensor.detach().cpu().numpy().T.astype(np.float64)


def _resample_if_needed(stem: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample a ``[samples, channels]`` stem to ``target_sr`` if needed."""
    if src_sr == target_sr:
        return stem
    import torchaudio

    as_tensor = torch.from_numpy(stem.T.copy())  # [channels, samples]
    resampled = torchaudio.transforms.Resample(src_sr, target_sr)(as_tensor)
    return resampled.numpy().T


def _align_stem(stem: np.ndarray, instrumental_mono: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    """Measure + correct one stem's offset against the instrumental, then pad/trim it."""
    stem_mono = stem.mean(axis=1) if stem.ndim == 2 and stem.shape[1] > 1 else stem.reshape(-1)
    offset = measure_offset_samples(instrumental_mono, stem_mono, sr)
    return align_to_length(stem, offset, target_len)


def apply_stem_layering(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    src_audio_path: str,
    gains: dict[str, float] | None = None,
) -> torch.Tensor:
    """Layer the original song's drums/bass/other back onto a generated instrumental.

    Args:
        audio_tensor: Generated instrumental, ``[channels, samples]`` float32.
        sample_rate: Sample rate of ``audio_tensor``.
        src_audio_path: Path to the original (with-vocals) source song.
        gains: Optional per-stem linear gain overrides, see
            ``stem_layer_mix.DEFAULT_STEM_GAINS``.

    Returns:
        torch.Tensor: The overlaid result at the same shape/sample rate as
        ``audio_tensor`` on success; the original, unmodified
        ``audio_tensor`` unchanged if anything in this step fails (separation
        error, unreadable source file, etc.) - this feature must never be the
        reason a song generation fails outright.
    """
    try:
        instrumental_np = _to_numpy_samples_first(audio_tensor)
        instrumental_mono = instrumental_np.mean(axis=1)
        target_len = instrumental_np.shape[0]

        stems = separate_background_stems(src_audio_path)
        if not stems:
            logger.warning("[stem_layer_overlay] No stems returned, skipping overlay")
            return audio_tensor

        # All stems come back from the SAME demucs call at htdemucs_ft's one
        # fixed native sample rate - this is a cheap cached lookup (the
        # separator instance from the call above), not a second separation.
        stem_sr = get_separator_sample_rate()

        aligned_stems: dict[str, np.ndarray] = {}
        for name, stem in stems.items():
            resampled = _resample_if_needed(stem, stem_sr, sample_rate)
            aligned_stems[name] = _align_stem(resampled, instrumental_mono, sample_rate, target_len)

        mixed = mix_stems_into_instrumental(instrumental_np, aligned_stems, gains)
        limited = apply_safety_limiter(mixed)

        result = torch.from_numpy(limited.T.copy()).to(dtype=audio_tensor.dtype)
        logger.info(
            f"[stem_layer_overlay] Overlaid {sorted(aligned_stems.keys())} onto instrumental "
            f"(src='{src_audio_path}')"
        )
        return result
    except Exception:
        logger.exception("[stem_layer_overlay] Stem overlay failed, returning instrumental unmodified")
        return audio_tensor
