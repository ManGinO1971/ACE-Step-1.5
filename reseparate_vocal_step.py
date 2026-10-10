#!/usr/bin/env python3
"""
Schritt 3 der "Vocals erzeugen"-Kette (korrigierter Plan, 7. Okt 2026):
Demucs-NACHtrennung auf das Ergebnis von Schritt 2 (flow_edit_morph,
siehe generate_vocal_only_variant() in heltech_music_v3.py). Das Modell
erzeugt dort technisch immer noch eine vollstaendige Song-Mischung (auch
wenn das Ziel "a cappella" war) - es bleiben idR Rest-Instrumentenanteile
uebrig, die hier per Demucs entfernt werden, bevor es in die bestehende
WORLD-Resynthese/RVC/Reverb-Kette (Schritt 5-7) weitergeht.

Laeuft in der acestep_stems Umgebung (gleiche Demucs-Installation wie
stem_clean_step.py). Bewusst ANDERS konfiguriert als stem_clean_step.py:
htdemucs_ft (staerkeres, langsameres Modell) statt htdemucs, zusaetzlich
--shifts 5 --overlap 0.5 fuer eine gruendlichere Trennung - vom Nutzer an
einem ganzen Testtag vor dieser Session gegen mehrere Alternativen
(u.a. --shifts 10 --overlap 0.75) gegengehoert und als Sweet Spot
bestaetigt. Device "mps" ist NUR fuer Geschwindigkeit (Apple-GPU auf dem
Mac) - kein erwarteter Klangunterschied zu CPU, daher bei Fehlschlag
automatischer Rueckfall auf CPU statt harten Abbruchs.

Aufruf:
  python reseparate_vocal_step.py <input.wav> <output_dir>

Erzeugt in <output_dir>:
  vocals_reseparated.wav  - erneut isolierter Gesang, MONO (Eingang fuer
                            Schritt 4, die ffmpeg-Waermekette)
"""
import sys
import os
import subprocess
import numpy as np
import soundfile as sf

DEMUCS_MODEL = "htdemucs_ft"
# NEU (10. Okt 2026): per RunPod-Umgebungsvariable HIELTECH_RESEP_SHIFTS
# testweise aenderbar (z.B. 2 = schneller, minimal anderer Klang). Ohne
# Variable bleibt es beim bestaetigten Sweet Spot 5.
DEMUCS_SHIFTS = str(os.environ.get("HIELTECH_RESEP_SHIFTS") or "5").strip()
if not DEMUCS_SHIFTS.isdigit() or not (1 <= int(DEMUCS_SHIFTS) <= 10):
    DEMUCS_SHIFTS = "5"
DEMUCS_OVERLAP = "0.5"


def _pick_torch_device():
    """Erkennt automatisch die schnellste verfuegbare Hardware: CUDA (RunPod/
    Linux-GPU), sonst MPS (Apple-GPU auf dem Mac), sonst None (Demucs waehlt
    dann selbst/faellt auf CPU zurueck). NEU (9. Okt 2026, Nutzerwunsch
    "soll alles moeglicher auf gpu laufen"): vorher stand hier fest "mps"
    (nur fuer den Mac gedacht) - auf RunPod (Linux, kein Apple-Metal) ist
    "mps" dort NIE verfuegbar, dieser besonders teure Schritt (shifts=5,
    also 5x Inferenz) ist bisher auf JEDEM RunPod-Lauf auf CPU gelandet,
    obwohl eine GPU bereitstand und bezahlt wurde."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return None


def _run_demucs(input_path, work_dir, device):
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", DEMUCS_MODEL,
        "--shifts", DEMUCS_SHIFTS,
        "--overlap", DEMUCS_OVERLAP,
        "--two-stems", "vocals",
        "--out", work_dir,
    ]
    if device:
        cmd += ["-d", device]
    cmd.append(input_path)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)


def separate_vocals_only(input_path, work_dir):
    """Nutzt Demucs (htdemucs_ft, shifts=5, overlap=0.5), liefert nur den
    Gesang zurueck (--two-stems vocals spart Zeit, da nur vocals.wav +
    no_vocals.wav statt aller vier Stems berechnet werden muessen)."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(work_dir, DEMUCS_MODEL, base_name, "vocals.wav")

    # NEU (10. Okt 2026, Geschwindigkeit): "--two-stems vocals" spart in
    # Demucs selbst KEINE Rechenzeit - bei htdemucs_ft laufen trotzdem alle 4
    # Spezialmodelle x 5 shifts = 20 Durchlaeufe, obwohl nur das Vocals-Modell
    # in die Gesangsspur eingeht (Gewicht 0 fuer die anderen 3). Der
    # Schnellweg rechnet nur dieses eine Modell (5 statt 20 Durchlaeufe) mit
    # denselben bestaetigten Werten shifts=5/overlap=0.5 - ergebnisgleich,
    # siehe demucs_vocals_fast.py. Bei Problemen: Kommandozeile wie bisher.
    try:
        from demucs_vocals_fast import separate_vocals
        if separate_vocals(input_path, vocals_path, model_name=DEMUCS_MODEL,
                           shifts=int(DEMUCS_SHIFTS), overlap=float(DEMUCS_OVERLAP),
                           device=_pick_torch_device()):
            vocals, sr = sf.read(vocals_path)
            return vocals.astype(np.float64), sr
    except Exception as e:  # noqa: BLE001 - Rueckfall auf Kommandozeile unten
        print(f"Demucs-Schnellweg nicht möglich ({e}), nutze Kommandozeile...", file=sys.stderr)

    result = _run_demucs(input_path, work_dir, device=_pick_torch_device())
    if result.returncode != 0 and "Separated tracks" not in result.stderr:
        # Rueckfall auf CPU, falls weder CUDA noch MPS in dieser Umgebung
        # verfuegbar ist/fehlschlaegt - rein fuer Robustheit, kein erwarteter
        # Klangunterschied (siehe Modul-Docstring).
        print("GPU-Trennung fehlgeschlagen, versuche CPU-Rückfall...", file=sys.stderr)
        result = _run_demucs(input_path, work_dir, device="cpu")
        if result.returncode != 0 and "Separated tracks" not in result.stderr:
            raise RuntimeError(f"Demucs-Nachtrennung fehlgeschlagen: {result.stderr[-500:]}")

    vocals, sr = sf.read(vocals_path)
    return vocals.astype(np.float64), sr


def to_mono(audio):
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python reseparate_vocal_step.py <input.wav> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Demucs-Nachtrennung ({DEMUCS_MODEL}, shifts={DEMUCS_SHIFTS}, overlap={DEMUCS_OVERLAP})...")
    import tempfile
    with tempfile.TemporaryDirectory() as work_dir:
        vocals, sr = separate_vocals_only(input_path, work_dir)

    vocals_mono = to_mono(vocals)
    vocals_out = os.path.join(output_dir, "vocals_reseparated.wav")
    sf.write(vocals_out, vocals_mono, sr)
    print(f"OK: {vocals_out}")


if __name__ == "__main__":
    main()
