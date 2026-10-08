#!/usr/bin/env python3
"""Rauschtor (Noise-Gate) fuer eine per Demucs getrennte Gesangsspur.

Zweck (Teil des Vocals-Erzeugen-Tests, Schritt 1b): Demucs trennt den Gesang
schon brauchbar vom Rest, aber die Pausen zwischen den gesungenen Phrasen
sind oft nicht wirklich still - es bleibt ein leises Rauschen/Bleed von den
Instrumenten uebrig. Dieses Skript druenkt genau diese leisen Zwischenraeume
auf nahezu Stille, waehrend die tatsaechlich gesungenen Stellen unangetastet
bleiben.

Technik identisch zur schon vorhandenen, vom Nutzer bestaetigten Loesung fuer
die Instrumental-Richtung (acestep/core/generation/handler/stem_layer_gate.py,
compute_stem_gate_envelope: Frame-RMS + exponentieller Attack/Release +
np.interp()-Hochskalierung, um eine Treppenstufen-Huelllkurve und damit
Klick-Artefakte zu vermeiden). Hier aber mit eigenen Standardwerten: deutlich
tieferer floor (0.03 statt 0.5) und kuerzeres release_ms (200 statt 900),
weil hier ausdruecklich RICHTIG SAUBERE Stille in den Luecken gewuenscht ist
(nicht nur ein Daempfen wie beim Instrumental-Overlay).

Nie ein harter Mute (floor > 0): ein echter Hard-Cut auf exakt 0 kann selbst
wieder eine hoerbare Kante erzeugen, siehe apply_boundary_fade() unten fuer
den gleichen Grundgedanken an den Datei-Raendern.

Nutzung:
    python gate_vocal_stem.py --input vocals.wav --output vocals_gated.wav
    python gate_vocal_stem.py --input vocals.wav --output vocals_gated.wav \
        --threshold_db -30 --floor 0.02 --release_ms 150
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

_FRAME_MS = 10.0


def compute_gate_envelope(
    reference_mono: np.ndarray,
    sr: int,
    threshold_db: float,
    floor: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """Pro-Sample-Gain in [floor, 1.0], gesteuert von der Lautstaerke des Signals selbst."""
    n = len(reference_mono)
    if n == 0:
        return np.zeros(0)

    raw_peak = np.max(np.abs(reference_mono))
    if raw_peak < 1e-9:
        return np.full(n, floor)

    peak = raw_peak + 1e-12
    thresh_lin = peak * (10 ** (threshold_db / 20.0))

    frame_len = max(1, int(sr * _FRAME_MS / 1000.0))
    n_frames = max(1, n // frame_len)
    if n_frames < 2:
        rms = np.sqrt(np.mean(reference_mono ** 2) + 1e-24)
        target = floor + (1.0 - floor) * np.clip(rms / thresh_lin, 0.0, 1.0)
        return np.full(n, target)

    attack_coef = np.exp(-1.0 / (sr / frame_len * attack_ms / 1000.0))
    release_coef = np.exp(-1.0 / (sr / frame_len * release_ms / 1000.0))

    gain = np.ones(n_frames)
    current = floor  # Start geschlossen, nicht offen - siehe stem_layer_gate.py-Vorbild.
    for i in range(n_frames):
        seg = reference_mono[i * frame_len:(i + 1) * frame_len]
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


def apply_boundary_fade(audio: np.ndarray, sr: int, fade_ms: float = 25.0) -> np.ndarray:
    """Kurzer linearer Fade-In/Out an Anfang/Ende, verhindert einen Klick an der harten Kante."""
    n = audio.shape[0]
    fade_len = min(int(sr * fade_ms / 1000.0), n // 2)
    if fade_len <= 0:
        return audio
    faded = audio.astype(np.float64).copy()
    ramp = np.linspace(0.0, 1.0, fade_len)
    if faded.ndim > 1:
        ramp = ramp[:, None]
    faded[:fade_len] *= ramp
    faded[-fade_len:] *= ramp[::-1]
    return faded


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Pfad zur per Demucs getrennten Gesangsspur (z.B. vocals.wav).")
    p.add_argument("--output", required=True, help="Zielpfad fuer die gegatete Datei.")
    p.add_argument("--threshold_db", type=float, default=-32.0,
                   help="Pegel relativ zum eigenen Spitzenwert, ab dem das Tor zu schliessen beginnt (Default -32).")
    p.add_argument("--floor", type=float, default=0.03,
                   help="Minimaler Gain bei komplett geschlossenem Tor - nie 0, um keine harte Kante zu erzeugen (Default 0.03).")
    p.add_argument("--attack_ms", type=float, default=15.0,
                   help="Oeffnungszeit, wenn die Stimme einsetzt - bewusst schnell (Default 15).")
    p.add_argument("--release_ms", type=float, default=200.0,
                   help="Schliesszeit, nachdem die Stimme verklingt - bewusst kurz fuer richtig saubere Stille in den Luecken (Default 200).")
    p.add_argument("--boundary_fade_ms", type=float, default=25.0,
                   help="Fade-Laenge an Anfang/Ende der Datei (Default 25).")
    args = p.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        print("FEHLER: 'soundfile' ist in dieser Umgebung nicht installiert "
              "(pip install soundfile, oder in der acestep_stems-Umgebung ausfuehren).", flush=True)
        return 1

    print(f"Lade: {args.input}", flush=True)
    audio, sr = sf.read(args.input, always_2d=True)  # [samples, channels]
    print(f"  {audio.shape[0]} Samples, {audio.shape[1]} Kanal/Kanaele, {sr} Hz, "
          f"{audio.shape[0] / sr:.1f}s", flush=True)

    reference_mono = np.mean(audio, axis=1)
    envelope = compute_gate_envelope(
        reference_mono, sr,
        threshold_db=args.threshold_db,
        floor=args.floor,
        attack_ms=args.attack_ms,
        release_ms=args.release_ms,
    )

    gated = audio * envelope[:, None]
    gated = apply_boundary_fade(gated, sr, fade_ms=args.boundary_fade_ms)

    sf.write(args.output, gated, sr)

    closed_fraction = float(np.mean(envelope < (args.floor + 0.05)))
    print(f"Fertig -> {args.output}", flush=True)
    print(f"  Einstellungen: threshold_db={args.threshold_db}, floor={args.floor}, "
          f"attack_ms={args.attack_ms}, release_ms={args.release_ms}", flush=True)
    print(f"  Anteil der Datei, der nahezu stillgelegt wurde: {closed_fraction * 100:.1f}%", flush=True)
    print("Bitte reinhoeren: sind die Luecken zwischen den Phrasen jetzt wirklich still, "
          "und klingen die gesungenen Stellen selbst unveraendert (kein Pumpen/Abschneiden "
          "von Wortanfaengen/-enden)?", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
