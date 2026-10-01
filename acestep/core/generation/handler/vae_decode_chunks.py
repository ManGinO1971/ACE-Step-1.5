"""Chunk-level VAE decode helpers used by tiled decode orchestration."""

import math
import os

import torch
from loguru import logger
from tqdm import tqdm


class VaeDecodeChunksMixin:
    """Implement chunked decode strategies for GPU and CPU-offload modes."""

    # Equal-power crossfade length applied at each INTERNAL tiled-decode
    # splice point (never at the very start/end of the song). Each chunk is
    # decoded independently through the VAE from its own overlapping window;
    # even with overlap-discard trimming, conv/normalization edge effects
    # mean the two independently-decoded versions of the same instant are
    # almost never bit-identical, so a hard cut+concat at the trim boundary
    # can produce an audible millisecond-scale "jump". Blending a short
    # window instead of hard-cutting hides that mismatch.
    #
    # Rollback switch: set ACESTEP_DISABLE_SPLICE_CROSSFADE=1 to restore the
    # previous hard-cut behavior instantly, without reverting this file.
    _SPLICE_CROSSFADE_SEC = 0.015  # 15ms; short enough to stay transparent
    _LATENT_FRAME_HZ = 25.0  # ACE-Step latent frame rate (see inference.py)

    def _tiled_decode_inner(self, latents, chunk_size, overlap, offload_wav_to_cpu):
        """Run tiled decode with adaptive overlap and OOM fallbacks."""
        bsz, _channels, latent_frames = latents.shape

        # Batch-sequential decode keeps peak VRAM stable across batch sizes.
        if bsz > 1:
            logger.info(f"[tiled_decode] Batch size {bsz} > 1; decoding samples sequentially to save VRAM")
            per_sample_results = []
            for b_idx in range(bsz):
                single = latents[b_idx : b_idx + 1]
                decoded = self._tiled_decode_inner(single, chunk_size, overlap, offload_wav_to_cpu)
                per_sample_results.append(decoded.cpu() if decoded.device.type != "cpu" else decoded)
                self._empty_cache()
            result = torch.cat(per_sample_results, dim=0)
            if latents.device.type != "cpu" and not offload_wav_to_cpu:
                result = result.to(latents.device)
            return result

        min_overlap = 4  # Minimum floor to prevent audio artifacts at chunk boundaries
        effective_overlap = overlap
        while chunk_size - 2 * effective_overlap <= 0 and effective_overlap > min_overlap:
            effective_overlap = effective_overlap // 2
        # Enforce minimum overlap floor to avoid near-zero values that cause corruption
        if effective_overlap < min_overlap and overlap >= min_overlap:
            effective_overlap = min_overlap
        if effective_overlap != overlap:
            logger.warning(
                f"[tiled_decode] Reduced overlap from {overlap} to {effective_overlap} for chunk_size={chunk_size}"
            )
        overlap = effective_overlap

        if latent_frames <= chunk_size:
            try:
                decoder_output = self.vae.decode(latents)
                result = decoder_output.sample
                del decoder_output
                return result
            except torch.cuda.OutOfMemoryError:
                logger.warning("[tiled_decode] OOM on direct decode, falling back to CPU VAE decode")
                self._empty_cache()
                return self._decode_on_cpu(latents)

        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")

        num_steps = math.ceil(latent_frames / stride)

        if offload_wav_to_cpu:
            try:
                return self._tiled_decode_offload_cpu(latents, bsz, latent_frames, stride, overlap, num_steps)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"[tiled_decode] OOM during offload_cpu decode with chunk_size={chunk_size}, "
                    "falling back to CPU VAE decode"
                )
                self._empty_cache()
                return self._decode_on_cpu(latents)

        try:
            return self._tiled_decode_gpu(latents, stride, overlap, num_steps)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                f"[tiled_decode] OOM during GPU decode with chunk_size={chunk_size}, "
                "falling back to CPU offload path"
            )
            self._empty_cache()
            try:
                return self._tiled_decode_offload_cpu(latents, bsz, latent_frames, stride, overlap, num_steps)
            except torch.cuda.OutOfMemoryError:
                logger.warning("[tiled_decode] OOM even with offload path, falling back to full CPU VAE decode")
                self._empty_cache()
                return self._decode_on_cpu(latents)

    def _splice_crossfade_enabled(self) -> bool:
        """Runtime kill-switch for the splice crossfade (instant rollback)."""
        return os.environ.get("ACESTEP_DISABLE_SPLICE_CROSSFADE", "0").lower() not in ("1", "true", "yes")

    def _splice_fade_samples(self, upsample_factor: float, overlap_latent_frames: int) -> int:
        """Samples to crossfade at each internal tiled-decode splice point.

        Clamped to the overlap margin itself (audio that is already being
        decoded redundantly on both sides of a splice), so this can only
        ever replace part of a hard cut with a blend -- it never reads
        latent content that wasn't already being decoded anyway, and it
        never changes chunk sizing / VRAM behavior.
        """
        if not self._splice_crossfade_enabled():
            return 0
        est_sample_rate = upsample_factor * self._LATENT_FRAME_HZ
        fade = int(round(self._SPLICE_CROSSFADE_SEC * est_sample_rate))
        max_fade = int(overlap_latent_frames * upsample_factor)
        return max(0, min(fade, max_fade))

    @staticmethod
    def _crossfade_merge(result: torch.Tensor, seg: torch.Tensor, fade_samples: int) -> torch.Tensor:
        """Append ``seg`` to ``result`` with an equal-power crossfade splice.

        ``result``'s trailing ``fade_samples`` and ``seg``'s leading
        ``fade_samples`` are expected to cover the SAME underlying audio
        instant (decoded redundantly from two overlapping chunk windows).
        Blending them instead of hard-cutting hides any tiny mismatch
        between the two independent VAE decodes at that instant, and keeps
        the total length identical to the previous hard-cut behavior.
        """
        fade = min(fade_samples, result.shape[-1], seg.shape[-1])
        if fade <= 0:
            return torch.cat([result, seg], dim=-1)
        t = torch.linspace(0.0, 1.0, fade, device=seg.device, dtype=seg.dtype)
        fade_out = torch.cos(t * (math.pi / 2.0)) ** 2
        fade_in = torch.sin(t * (math.pi / 2.0)) ** 2
        blended = result[:, :, -fade:] * fade_out + seg[:, :, :fade] * fade_in
        return torch.cat([result[:, :, :-fade], blended, seg[:, :, fade:]], dim=-1)

    def _tiled_decode_gpu(self, latents, stride, overlap, num_steps):
        """Decode chunks and keep decoded audio tensors on GPU."""
        decoded_audio_list = []
        upsample_factor = None
        fade_samples = 0

        for i in tqdm(range(num_steps), desc="Decoding audio chunks", disable=self.disable_tqdm):
            core_start = i * stride
            core_end = min(core_start + stride, latents.shape[-1])
            win_start = max(0, core_start - overlap)
            win_end = min(latents.shape[-1], core_end + overlap)

            latent_chunk = latents[:, :, win_start:win_end]
            decoder_output = self.vae.decode(latent_chunk)
            audio_chunk = decoder_output.sample
            del decoder_output

            if upsample_factor is None:
                upsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]
                fade_samples = self._splice_fade_samples(upsample_factor, overlap)

            added_start = core_start - win_start
            trim_start = int(round(added_start * upsample_factor))
            added_end = win_end - core_end
            trim_end = int(round(added_end * upsample_factor))
            # Keep `fade_samples` of overlap material at the end of every
            # non-final chunk instead of discarding it, so it can be
            # crossfaded against the next chunk's start below.
            if i < num_steps - 1:
                trim_end = max(0, trim_end - fade_samples)

            audio_len = audio_chunk.shape[-1]
            end_idx = audio_len - trim_end if trim_end > 0 else audio_len
            audio_core = audio_chunk[:, :, trim_start:end_idx]
            decoded_audio_list.append(audio_core)

        if not decoded_audio_list:
            return torch.cat(decoded_audio_list, dim=-1)
        result = decoded_audio_list[0]
        for seg in decoded_audio_list[1:]:
            result = self._crossfade_merge(result, seg, fade_samples)
        return result

    def _tiled_decode_offload_cpu(self, latents, bsz, latent_frames, stride, overlap, num_steps):
        """Decode chunks on GPU and copy trimmed audio cores to a CPU buffer."""
        first_core_end = min(stride, latent_frames)
        first_win_end = min(latent_frames, first_core_end + overlap)
        first_latent_chunk = latents[:, :, 0:first_win_end]
        first_decoder_output = self.vae.decode(first_latent_chunk)
        first_audio_chunk = first_decoder_output.sample
        del first_decoder_output

        upsample_factor = first_audio_chunk.shape[-1] / first_latent_chunk.shape[-1]
        audio_channels = first_audio_chunk.shape[1]
        fade_samples = self._splice_fade_samples(upsample_factor, overlap)

        total_audio_length = int(round(latent_frames * upsample_factor))
        final_audio = torch.zeros(bsz, audio_channels, total_audio_length, dtype=first_audio_chunk.dtype, device="cpu")

        first_added_end = first_win_end - first_core_end
        first_trim_end = int(round(first_added_end * upsample_factor))
        # Keep `fade_samples` of overlap material at the end of the first
        # chunk (unless it's also the last) so it can be crossfaded against
        # the next chunk's start instead of hard-cut.
        if num_steps > 1:
            first_trim_end = max(0, first_trim_end - fade_samples)
        first_audio_len = first_audio_chunk.shape[-1]
        first_end_idx = first_audio_len - first_trim_end if first_trim_end > 0 else first_audio_len

        first_audio_core = first_audio_chunk[:, :, :first_end_idx]
        audio_write_pos = first_audio_core.shape[-1]
        final_audio[:, :, :audio_write_pos] = first_audio_core.cpu()

        del first_audio_chunk, first_audio_core, first_latent_chunk

        for i in tqdm(range(1, num_steps), desc="Decoding audio chunks", disable=self.disable_tqdm):
            core_start = i * stride
            core_end = min(core_start + stride, latent_frames)
            win_start = max(0, core_start - overlap)
            win_end = min(latent_frames, core_end + overlap)

            latent_chunk = latents[:, :, win_start:win_end]
            decoder_output = self.vae.decode(latent_chunk)
            audio_chunk = decoder_output.sample
            del decoder_output

            added_start = core_start - win_start
            trim_start = int(round(added_start * upsample_factor))
            added_end = win_end - core_end
            trim_end = int(round(added_end * upsample_factor))
            if i < num_steps - 1:
                trim_end = max(0, trim_end - fade_samples)

            audio_len = audio_chunk.shape[-1]
            end_idx = audio_len - trim_end if trim_end > 0 else audio_len
            audio_core = audio_chunk[:, :, trim_start:end_idx]

            # The leading `fade_samples` of this chunk's core cover the same
            # audio instant as the trailing `fade_samples` already written
            # into the buffer by the previous chunk (its kept-back overlap
            # tail) -- blend them in place instead of hard-overwriting.
            fade = min(fade_samples, audio_write_pos, audio_core.shape[-1])
            if fade > 0:
                t = torch.linspace(0.0, 1.0, fade, device=audio_core.device, dtype=audio_core.dtype)
                fade_out = torch.cos(t * (math.pi / 2.0)) ** 2
                fade_in = torch.sin(t * (math.pi / 2.0)) ** 2
                prev_tail = final_audio[:, :, audio_write_pos - fade : audio_write_pos].to(audio_core.device)
                blended = prev_tail * fade_out + audio_core[:, :, :fade] * fade_in
                final_audio[:, :, audio_write_pos - fade : audio_write_pos] = blended.cpu()
                remainder = audio_core[:, :, fade:]
            else:
                remainder = audio_core

            core_len = remainder.shape[-1]
            final_audio[:, :, audio_write_pos : audio_write_pos + core_len] = remainder.cpu()
            audio_write_pos += core_len

            del audio_chunk, audio_core, latent_chunk

        return final_audio[:, :, :audio_write_pos]
