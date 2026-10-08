#!/usr/bin/env python3
"""
Zusatz-Schritt NACH RVC (Schritt 4 der ACE-Step -> RVC Pipeline): Rauschen/
Zischen entfernen, das der RVC-Vocoder selbst einbringen kann.
Laeuft in der acestep_stems Umgebung (dieselbe, die auch schon in
stem_clean_step.py fuer DeepFilterNet genutzt wird).

Hintergrund (warum ein eigener, schlanker Schritt statt stem_clean_step.py
wiederzuverwenden): stem_clean_step.py macht Schritt 1+2 der Pipeline
(Demucs-Trennung + DeepFilter-Bereinigung + Rueckfuehrung der Restspur in
die Instrumentalspur) - das ist hier nicht mehr noetig, die Stimme ist nach
RVC schon fertig getrennt/umgewandelt. Dieser Schritt wendet NUR dieselbe
DeepFilterNet-Entrauschung an, ohne Demucs-Trennung und ohne eine
Restspur zu erzeugen - genau das, was nach RVC noch gebraucht wird.

Standard-Reihenfolge (so wie vom Nutzer bestaetigt): nach RVC zuerst
DENOISE (dieses Skript), danach GATE (gate_vocal_stem.py) - Denoise
glaettet erst das durchgehende Rauschen/Zischen im ganzen Signal, Gate
drueckt danach die stillen Zwischenraeume zwischen den Phrasen zusaetzlich
auf nahezu Stille.

Aufruf:
    python denoise_step.py --input vocals_rvc.wav --output vocals_rvc_denoised.wav
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import os

import numpy as np


def denoise_deepfilter(audio: np.ndarray, sr: int):
    """Entrauscht Mono- oder Stereo-Audio Kanal-fuer-Kanal mit DeepFilterNet,
    identische Technik wie stem_clean_step.py (clean_vocals_deepfilter_stereo),
    aber ohne Restspur-Berechnung - hier wird nur das bereinigte Ergebnis
    gebraucht."""
    from df.enhance import enhance, init_df
    from df.io import load_audio, save_audio
    import soundfile as sf
    import librosa

    model, df_state, _ = init_df()
    df_sr = df_state.sr()

    if audio.ndim == 1:
        channels = [audio]
    else:
        channels = [audio[:, i] for i in range(audio.shape[1])]

    cleaned_channels = []
    for ch in channels:
        tmp_in = tempfile.mktemp(suffix=".wav")
        tmp_out = tempfile.mktemp(suffix=".wav")
        sf.write(tmp_in, ch, sr)

        in_audio, _meta = load_audio(tmp_in, sr=df_sr)
        enhanced = enhance(model, df_state, in_audio)
        save_audio(tmp_out, enhanced, df_sr)

        cleaned_at_dfsr, _ = sf.read(tmp_out)
        os.unlink(tmp_in)
        os.unlink(tmp_out)

        if df_sr != sr:
            cleaned = librosa.resample(cleaned_at_dfsr, orig_sr=df_sr, target_sr=sr)
        else:
            cleaned = cleaned_at_dfsr
        cleaned_channels.append(cleaned)

    min_len = min(len(c) for c in cleaned_channels)
    cleaned_channels = [c[:min_len] for c in cleaned_channels]

    if len(cleaned_channels) == 1:
        return cleaned_channels[0]
    return np.stack(cleaned_channels, axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Pfad zur RVC-umgewandelten Gesangsspur.")
    p.add_argument("--output", required=True, help="Zielpfad fuer die entrauschte Datei.")
    args = p.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        print("FEHLER: 'soundfile' fehlt in dieser Umgebung (acestep_stems aktivieren).", flush=True)
        return 1
    try:
        import df  # noqa: F401
    except ImportError:
        print("FEHLER: 'DeepFilterNet' (df) fehlt in dieser Umgebung "
              "(acestep_stems aktivieren, selbe Umgebung wie stem_clean_step.py).", flush=True)
        return 1

    print(f"Lade: {args.input}", flush=True)
    audio, sr = sf.read(args.input)
    channels_desc = "mono" if audio.ndim == 1 else f"{audio.shape[1]}-kanalig"
    print(f"  {len(audio)} Samples, {sr} Hz, {len(audio) / sr:.1f}s ({channels_desc})", flush=True)

    print("Entrausche (DeepFilterNet) - entfernt Rauschen/Zischen vom RVC-Vocoder...", flush=True)
    cleaned = denoise_deepfilter(audio, sr)

    sf.write(args.output, cleaned, sr)
    print(f"Fertig -> {args.output}", flush=True)
    print("Bitte reinhoeren: ist das Rauschen/Zischen jetzt weg, und bleibt die", flush=True)
    print("Stimme selbst dabei unveraendert/unverzerrt?", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
