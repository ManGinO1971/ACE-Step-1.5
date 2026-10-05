"""htdemucs_ft stem separation for the background-stem overlay feature.

Isolated into its own module because it is the one piece of this feature
that needs the ``demucs`` package and (on first use per worker) a one-time
checkpoint download - keeping it separate means ``stem_layer_align``/
``stem_layer_mix`` stay trivially unit-testable without a GPU or network
access, and a demucs import/load failure can be caught at a single, obvious
call site.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

_MODEL_NAME = "htdemucs_ft"
# Keep every stem except vocals - the whole point of this feature is to
# layer the ORIGINAL song's drums/bass/other back onto the AI-generated
# instrumental; the vocal stem is exactly what the instrumental is meant to
# replace, so it is discarded right here rather than carried further.
_KEEP_STEMS = ("drums", "bass", "other")

_separator_cache: dict[str, object] = {}


def _get_separator(device: str):
    """Load (and cache) the htdemucs_ft ``Separator`` for the given device.

    Cached at module level: loading the checkpoint is a multi-second GPU
    operation, and a single worker process typically serves many requests.
    """
    cached = _separator_cache.get(device)
    if cached is not None:
        return cached
    from demucs.api import Separator

    logger.info(f"[stem_layer_separate] Loading {_MODEL_NAME} on {device} (first call on this worker)")
    separator = Separator(model=_MODEL_NAME, device=device)
    _separator_cache[device] = separator
    return separator


def separate_background_stems(audio_path: str, device: str | None = None) -> dict[str, np.ndarray]:
    """Run htdemucs_ft on ``audio_path`` and return the non-vocal stems.

    Args:
        audio_path: Path to the original (with-vocals) source audio - the
            same file already uploaded for the flow_edit_morph instrumental
            generation (``params.src_audio``), so no extra upload is needed.
        device: Torch device string (``"cuda"``/``"cpu"``); defaults to CUDA
            when available, matching every other GPU-bound step in this
            pipeline.

    Returns:
        dict[str, np.ndarray]: ``{"drums": ..., "bass": ..., "other": ...}``,
        each ``[samples, channels]`` float32 at the separator's native
        sample rate (``separator.samplerate``, 44100 for htdemucs_ft) -
        resampling to the instrumental's own rate is the caller's job (see
        ``stem_layer_overlay``), since the caller already knows that rate.

    Raises:
        Exception: Propagates any demucs/torch failure (missing checkpoint,
            OOM, unreadable file) - callers are expected to catch this and
            fall back to the un-overlaid instrumental, exactly like every
            other optional post-processing step in this pipeline.
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    separator = _get_separator(resolved_device)

    _origin, separated = separator.separate_audio_file(audio_path)

    stems: dict[str, np.ndarray] = {}
    for name in _KEEP_STEMS:
        tensor = separated.get(name)
        if tensor is None:
            logger.warning(f"[stem_layer_separate] demucs output missing stem '{name}'")
            continue
        # demucs tensors are [channels, samples] - overlay code works in
        # [samples, channels], matching stem_layer_mix/stem_layer_align.
        stems[name] = tensor.detach().cpu().numpy().T.astype(np.float32)

    return stems


def get_separator_sample_rate(device: str | None = None) -> int:
    """Return the sample rate ``separate_background_stems`` output is in."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return int(_get_separator(resolved_device).samplerate)
