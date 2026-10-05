"""Sample-accurate offset measurement/correction for stem overlay.

Separation tools (including ``htdemucs_ft`` itself, depending on how the
source audio was resampled/padded before separation) can introduce a small,
*constant* timing offset between a separated stem and the audio it was
separated from. This offset is not a tempo drift - it is a fixed shift,
verified (manual user testing, Oct 2026) to be identical across stems and
stable over the whole track - so a single cross-correlation measurement per
stem is enough to correct it before mixing.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

_DEFAULT_MAX_SHIFT_SEC = 0.5


def measure_offset_samples(
    reference: np.ndarray,
    probe: np.ndarray,
    sr: int,
    max_shift_sec: float = _DEFAULT_MAX_SHIFT_SEC,
) -> int:
    """Measure the lag of ``probe`` relative to ``reference`` via FFT cross-correlation.

    Both signals are expected mono (callers should reduce stereo to mono for
    this measurement only - the correction itself is applied per channel).
    Uses ``scipy.signal.fftconvolve`` rather than ``numpy.correlate``, which
    is O(n*m) and impractically slow for multi-million-sample audio.

    Args:
        reference: Mono reference signal (e.g. the generated instrumental).
        probe: Mono signal to measure against the reference (e.g. a stem).
        sr: Sample rate shared by both signals.
        max_shift_sec: Only lags within +/- this many seconds are considered -
            a true alignment offset from a separation tool is on the order of
            milliseconds, so a generous but bounded window avoids locking
            onto an unrelated, far-away correlation peak.

    Returns:
        int: Best-matching lag in samples. A positive value means ``probe``
        leads ``reference`` (pad that many samples of silence at the front
        of ``probe`` to align it); negative means ``probe`` lags behind
        ``reference`` (trim that many samples from the front of ``probe``).
        ``align_to_length`` applies exactly this correction.
    """
    n = min(len(reference), len(probe))
    if n == 0:
        return 0
    ref = reference[:n].astype(np.float64)
    prb = probe[:n].astype(np.float64)
    ref = (ref - ref.mean()) / (ref.std() + 1e-8)
    prb = (prb - prb.mean()) / (prb.std() + 1e-8)

    corr = fftconvolve(ref, prb[::-1], mode="full")
    lags = np.arange(-len(prb) + 1, len(ref))

    max_shift = max(1, int(round(max_shift_sec * sr)))
    mask = (lags >= -max_shift) & (lags <= max_shift)
    corr_sub = corr[mask]
    lags_sub = lags[mask]
    if len(corr_sub) == 0:
        return 0

    # Sign convention: fftconvolve(ref, prb[::-1]) peaks at lag L such that
    # probe[i - L] ~= reference[i] - i.e. probe leads reference by L samples.
    return int(lags_sub[np.argmax(corr_sub)])


def align_to_length(probe: np.ndarray, offset_samples: int, target_len: int) -> np.ndarray:
    """Shift ``probe`` by the measured offset and pad/trim it to ``target_len``.

    Works on mono or multi-channel ([samples] or [samples, channels]) arrays;
    the shift is applied along axis 0 in both cases.

    Args:
        probe: The stem signal to correct (same array that was measured, or
            its stereo counterpart - offset is assumed identical per channel,
            per the module-level docstring).
        offset_samples: Lag returned by ``measure_offset_samples`` (positive
            = probe leads and must be delayed/padded at the front; negative
            = probe lags and must be trimmed from the front).
        target_len: Desired length in samples (normally the instrumental's
            length) that the output is padded/trimmed to match exactly.

    Returns:
        np.ndarray: ``probe`` shifted and padded/trimmed to exactly
        ``target_len`` samples along axis 0.
    """
    if offset_samples > 0:
        # Probe leads - delay it by padding silence at the front.
        pad_shape = (offset_samples,) + probe.shape[1:]
        shifted = np.concatenate([np.zeros(pad_shape, dtype=probe.dtype), probe], axis=0)
    elif offset_samples < 0:
        # Probe lags - trim the leading part it was delayed by.
        shifted = probe[-offset_samples:]
    else:
        shifted = probe

    current_len = shifted.shape[0]
    if current_len == target_len:
        return shifted
    if current_len > target_len:
        return shifted[:target_len]
    pad_shape = (target_len - current_len,) + shifted.shape[1:]
    return np.concatenate([shifted, np.zeros(pad_shape, dtype=shifted.dtype)], axis=0)
