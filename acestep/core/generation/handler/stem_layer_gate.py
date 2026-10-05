"""Noise gate for overlaid stems, plus a boundary fade for the finished mix.

User-reported regression (Okt 2026, right after the stem overlay feature
shipped): two new, small-but-audible artifacts that were never present with
the plain AI instrumental.

1. A soft hiss at the very start (before the song "comes in") and the very
   end (after it ends). htdemucs_ft never separates 100% cleanly - its
   drums/bass/other stems typically carry a little residual broadband noise,
   most noticeable exactly where nothing else masks it: the instrumental's
   own quiet intro/outro. ``compute_stem_gate_envelope`` ducks the overlaid
   stems there, keyed off how loud the instrumental itself is at each point.

2. A click at the exact first/last sample. ``stem_layer_align.align_to_length``
   pads or trims a stem's front with a hard cut, which almost never lands on
   a zero-crossing - audible as a click once real recorded-stem material (not
   the AI instrumental's own naturally soft edges) sits right at the
   boundary. ``apply_boundary_fade`` is a cheap, always-safe net against that,
   independent of the root cause.
"""

from __future__ import annotations

import numpy as np

_FRAME_MS = 10.0


def compute_stem_gate_envelope(
    instrumental_mono: np.ndarray,
    sr: int,
    threshold_db: float = -36.0,
    floor: float = 0.5,
    attack_ms: float = 20.0,
    release_ms: float = 900.0,
) -> np.ndarray:
    """Per-sample gain in ``[floor, 1.0]``, keyed by the instrumental's own loudness.

    Uses the same frame-RMS + exponential attack/release + ``np.interp()``
    upsampling idiom as heltech_music_v3.py's ``apply_limiter`` /
    ``apply_bus_compressor`` (``np.interp``, not ``np.repeat`` - that codebase
    already hit and fixed an audible click caused by a stair-step gain curve;
    reusing the fix rather than the mistake).

    Args:
        instrumental_mono: 1-D mono mixdown of the generated instrumental.
        sr: Sample rate of ``instrumental_mono``.
        threshold_db: Level, relative to the instrumental's own peak, below
            which the gate starts closing.
        floor: Minimum gain applied to the stems even fully closed - never a
            hard mute, so true silence doesn't itself become an audible edit.
        attack_ms: Time constant while the instrumental is getting louder
            (gate opening) - fast, so the stems' entrance isn't clipped off.
        release_ms: Time constant while the instrumental is getting quieter
            (gate closing) - slower than attack so it doesn't pump/chatter.

    Returns:
        np.ndarray: 1-D gain envelope, same length as ``instrumental_mono``.
    """
    n = len(instrumental_mono)
    if n == 0:
        return np.zeros(0)

    # True (or near-) silence has no meaningful peak to measure a threshold
    # against - short-circuit to a flat floor rather than let two mismatched
    # epsilon terms (peak's vs. rms's) decide the outcome by accident.
    raw_peak = np.max(np.abs(instrumental_mono))
    if raw_peak < 1e-9:
        return np.full(n, floor)

    peak = raw_peak + 1e-12
    thresh_lin = peak * (10 ** (threshold_db / 20.0))

    frame_len = max(1, int(sr * _FRAME_MS / 1000.0))
    n_frames = max(1, n // frame_len)
    if n_frames < 2:
        rms = np.sqrt(np.mean(instrumental_mono ** 2) + 1e-24)
        target = floor + (1.0 - floor) * np.clip(rms / thresh_lin, 0.0, 1.0)
        return np.full(n, target)

    attack_coef = np.exp(-1.0 / (sr / frame_len * attack_ms / 1000.0))
    release_coef = np.exp(-1.0 / (sr / frame_len * release_ms / 1000.0))

    gain = np.ones(n_frames)
    # Start closed (floor), not open: with no evidence yet about the very
    # first frame, assuming quiet is the safer default - a real loud start
    # only costs one fast attack_ms ramp-up, while assuming loud would leave
    # a real quiet intro's noise unprotected for a whole release_ms instead.
    current = floor
    for i in range(n_frames):
        seg = instrumental_mono[i * frame_len:(i + 1) * frame_len]
        rms = np.sqrt(np.mean(seg ** 2) + 1e-24)
        target = floor + (1.0 - floor) * np.clip(rms / thresh_lin, 0.0, 1.0)
        if target < current:
            current = release_coef * current + (1 - release_coef) * target
        else:
            current = attack_coef * current + (1 - attack_coef) * target
        gain[i] = current

    frame_centers = (np.arange(n_frames) + 0.5) * frame_len
    sample_idx = np.arange(n, dtype=np.float64)
    return np.interp(sample_idx, frame_centers, gain, left=gain[0], right=gain[-1])


def apply_boundary_fade(mix: np.ndarray, sr: int, fade_ms: float = 25.0) -> np.ndarray:
    """Short linear fade-in/out at the very start/end of a finished mix.

    A cheap click-safety net: harmless even if the mix already starts/ends at
    (near) zero, but removes the hard discontinuity a stem's padded/trimmed
    edge can otherwise leave at exactly the first/last sample.

    Args:
        mix: ``[samples, channels]`` (or 1-D) finished mix.
        sr: Sample rate of ``mix``.
        fade_ms: Fade length in milliseconds, applied at both ends.

    Returns:
        np.ndarray: Same shape as ``mix``, both ends faded.
    """
    n = mix.shape[0]
    fade_len = min(int(sr * fade_ms / 1000.0), n // 2)
    if fade_len <= 0:
        return mix

    faded = mix.astype(np.float64).copy()
    ramp = np.linspace(0.0, 1.0, fade_len)
    if faded.ndim > 1:
        ramp = ramp[:, None]
    faded[:fade_len] *= ramp
    faded[-fade_len:] *= ramp[::-1]
    return faded
