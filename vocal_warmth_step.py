#!/usr/bin/env python3
"""
Schritt 4 der "Vocals erzeugen"-Kette (korrigierter Plan, 7. Okt 2026):
ffmpeg Waerme-/Glue-Filterkette auf das Ergebnis von Schritt 3
(reseparate_vocal_step.py). Reine ffmpeg-Filterkette, kein zusaetzliches
Python-Paket noetig - braucht nur ein vorhandenes ffmpeg im PATH der
jeweiligen Umgebung (wie remix_step.py/apply_measured_reverb.py).

Bestaetigte Werte (ganzer Testtag vor dieser Session, per Hoertest
gegengeprueft):
  acompressor  threshold=-18dB, ratio=3, attack=20, release=250, makeup=2dB
  highpass     f=80Hz (Rumpeln raus)
  equalizer    f=3000Hz, g=3dB (Praesenz)
  volume       +4dB
  asoftclip    tanh, threshold=0.9 (Waerme/Saettigung)

Bewusst NICHT enthalten: De-Essing - das war eine fruehere, bewusste
Entscheidung des Nutzers, das lieber gezielt in LUNA/Logic zu machen statt
grob per ffmpeg uebers ganze Signal.

Aufruf:
  python vocal_warmth_step.py <input.wav> <output.wav>

Erzeugt:
  <output.wav> - Eingang fuer Schritt 5 (WORLD-Resynthese,
                  world_resynth_step.py)
"""
import sys
import subprocess

FILTER_CHAIN = (
    "acompressor=threshold=-18dB:ratio=3:attack=20:release=250:makeup=2dB,"
    "highpass=f=80,"
    "equalizer=f=3000:g=3,"
    "volume=4dB,"
    "asoftclip=type=tanh:threshold=0.9"
)


def apply_warmth_chain(input_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", FILTER_CHAIN,
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg-Wärmekette fehlgeschlagen: {result.stderr[-500:]}")


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python vocal_warmth_step.py <input.wav> <output.wav>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("Wende ffmpeg-Wärmekette an (Kompressor, Highpass, Präsenz-EQ, Sättigung)...")
    apply_warmth_chain(input_path, output_path)
    print(f"OK: {output_path}")


if __name__ == "__main__":
    main()
