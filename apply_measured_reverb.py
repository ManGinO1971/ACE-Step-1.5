#!/usr/bin/env python3
"""
Zusatz-Schritt: Misst grob die Nachhall-Stärke im Original-Gesang
(über das Verhältnis von Restspur- zu bereinigter-Stimme-Energie)
und wendet einen entsprechenden Nachhall auf die RVC-Stimme an, damit
sie klanglich naeher am LoRA-Original liegt.

Aufruf:
  python apply_measured_reverb.py <vocals_rvc.wav> <residue.wav> <vocals_clean.wav> <output.wav>
"""
import sys
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Reverb


def rms(audio):
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def to_mono(audio):
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def main():
    if len(sys.argv) < 5:
        print("Nutzung: python apply_measured_reverb.py <vocals_rvc.wav> <residue.wav> <vocals_clean.wav> <output.wav>")
        sys.exit(1)

    rvc_path, residue_path, clean_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    rvc_audio, sr = sf.read(rvc_path)
    residue_audio, sr_r = sf.read(residue_path)
    clean_audio, sr_c = sf.read(clean_path)

    residue_mono = to_mono(residue_audio)
    clean_mono = to_mono(clean_audio)

    residue_rms = rms(residue_mono)
    clean_rms = rms(clean_mono)

    # Verhältnis Restspur/bereinigte Stimme als grobe Nachhall-Stärke
    wet_ratio = residue_rms / clean_rms if clean_rms > 0 else 0.0
    wet_ratio = min(max(wet_ratio, 0.0), 0.6)  # sinnvoll begrenzen

    print(f"Gemessenes Verhältnis (Restspur/bereinigte Stimme): {wet_ratio:.3f}")

    # Nachhall-Stärke aus dem gemessenen Verhältnis ableiten
    wet_level = min(wet_ratio * 0.8, 0.35)
    room_size = min(0.3 + wet_ratio * 0.5, 0.7)

    print(f"Angewendet: room_size={room_size:.2f}, wet_level={wet_level:.2f}")

    board = Pedalboard([
        Reverb(room_size=room_size, damping=0.5, wet_level=wet_level, dry_level=1.0, width=1.0)
    ])

    if rvc_audio.ndim == 1:
        processed = board(rvc_audio.reshape(1, -1).astype(np.float32), sr)
        processed = processed.reshape(-1)
    else:
        processed = board(rvc_audio.T.astype(np.float32), sr).T

    sf.write(output_path, processed, sr)
    print(f"OK: {output_path}")


if __name__ == "__main__":
    main()
