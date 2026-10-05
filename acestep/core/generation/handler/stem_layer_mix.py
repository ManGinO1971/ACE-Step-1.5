"""Gain-staged summing + a simple lookahead-free peak limiter for stem overlay.

Both operate on numpy arrays shaped ``[samples, channels]`` (stereo or mono
with ``channels == 1``), matching the layout stems arrive in after
``stem_layer_separate.separate_background_stems``.
"""

from __future__ import annotations

import numpy as np

# Starting-point gains (Oct 2026 prototype, confirmed by the user as a good
# balance on a reggae test song: drums/bass carry the groove and blend in
# almost 1:1, "other" - guitars/keys/horns - sits slightly further back so it
# doesn't fight the AI-generated instrumental's own harmonic content). Not
# yet tuned across genres - a reasonable default, not a proven-optimal one.
DEFAULT_STEM_GAINS: dict[str, float] = {"drums": 0.85, "bass": 0.85, "other": 0.75}


def mix_stems_into_instrumental(
    instrumental: np.ndarray,
    stems: dict[str, np.ndarray],
    gains: dict[str, float] | None = None,
) -> np.ndarray:
    """Sum the instrumental with gain-weighted stems.

    Args:
        instrumental: ``[samples, channels]`` base track, already the target
            length/sample rate every stem has been aligned to.
        stems: Mapping of stem name -> ``[samples, channels]`` array, each
            already aligned (see ``stem_layer_align.align_to_length``) and
            matching ``instrumental``'s shape exactly.
        gains: Optional per-stem linear gain overrides; stems without an
            entry here fall back to ``DEFAULT_STEM_GAINS``, and an unknown
            key in ``gains`` is ignored since it has nothing to multiply.

    Returns:
        np.ndarray: The summed mix, same shape as ``instrumental``. Not yet
        peak-limited - call ``apply_safety_limiter`` on the result.
    """
    effective_gains = {**DEFAULT_STEM_GAINS, **(gains or {})}
    mix = instrumental.astype(np.float64).copy()
    for name, stem in stems.items():
        gain = effective_gains.get(name)
        if gain is None or stem is None:
            continue
        mix += stem.astype(np.float64) * gain
    return mix


def apply_safety_limiter(mix: np.ndarray, limit: float = 0.92, knee: float = 0.1) -> np.ndarray:
    """Soft-knee peak limiter, so a hot mix doesn't clip on export.

    No lookahead/attack-release smoothing (no sidechain pumping risk here -
    this runs once on a finished mix, not in real time): any sample whose
    absolute value sits above ``limit - knee`` is compressed towards
    ``limit`` with a smooth (tanh-shaped) knee rather than hard-clipped, which
    avoids the audible "staircase" distortion a naive ``np.clip`` would add
    on a bass-heavy mix (see mastering.py's own Runde-131 lesson on exactly
    that artifact).

    Args:
        mix: ``[samples, channels]`` mix, arbitrary peak level.
        limit: Target absolute peak ceiling (linear, 0..1).
        knee: Width of the soft-knee region below ``limit`` where gain
            reduction ramps in smoothly instead of kicking in abruptly.

    Returns:
        np.ndarray: Limited mix with peak at or below ``limit``.
    """
    peak = np.max(np.abs(mix)) + 1e-12
    if peak <= limit:
        return mix

    knee_start = max(limit - knee, 1e-6)
    abs_mix = np.abs(mix)
    over = np.clip((abs_mix - knee_start) / max(limit - knee_start, 1e-6), 0.0, None)
    # tanh saturates smoothly towards `limit` instead of a hard ceiling.
    target_abs = np.where(
        abs_mix <= knee_start,
        abs_mix,
        knee_start + (limit - knee_start) * np.tanh(over),
    )
    gain = np.where(abs_mix > 1e-12, target_abs / np.maximum(abs_mix, 1e-12), 1.0)
    limited = mix * gain

    # Final hard safety net in case the tanh knee still leaves an outlier
    # (e.g. a single-sample click) above `limit`.
    final_peak = np.max(np.abs(limited)) + 1e-12
    if final_peak > limit:
        limited = limited * (limit / final_peak)
    return limited
