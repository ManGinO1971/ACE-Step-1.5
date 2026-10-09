#!/usr/bin/env python3
"""
Schritt 1 der "Vocals erzeugen"-Kette (Korrektur, 8. Okt 2026, nach Pruefung
des Original-Testtags durch den Nutzer): isoliert den Gesang aus dem
kompletten Song und entfernt Hall/Echo, BEVOR das Ergebnis als src_audio an
flow_edit_morph (Schritt 2, generate_vocal_only_variant()) geht.

Nutzer-Begruendung (woertlich): "wir haben gedacht ... das die song die wir
fuer ACE-Step flow_edit_morph-Route schwierig war weil alle instrumente sind
da ... wenn wir die vocals allein trennen und saeubern sehr vernuenftig,
somit hat ACE-Step flow_edit_morph-Route ein einfacher aufgabe" sowie
explizit zum Hall: "was gut ist soll kein reverb im vocals sein, damit den
reconstruktion besser klappt, und nicht was schief laeuft wegen den hall."

Ersetzt NICHT stem_clean_step.py (das bleibt unveraendert fuer seinen
eigenen Zweck: instrumental_full.wav/residue.wav fuer Remix/Reverb-Referenz
an anderer Stelle der Pipeline). Dieses Skript liefert ausschliesslich die
gesaeuberte Gesangsspur, die als flow_edit_morph-Quelle dient.

Kette, exakt wie am Testtag per Hand (Terminal) bestaetigt:
  1. Demucs htdemucs_ft, --two-stems=vocals (OHNE --shifts/--overlap - das
     ist bewusst die einfache/schnelle Variante, siehe reseparate_vocal_step.py
     fuer die gruendlichere Variante NACH flow_edit_morph).
  2. UVR-DeEcho-DeReverb.pth (Ziel-Stem "no reverb") - entfernt Hall.
  3. UVR-De-Echo-Aggressive.pth (Ziel-Stem "no echo") - entfernt Echo, auf
     dem Ergebnis von Schritt 2 oben (Reihenfolge entspricht der
     chronologischen Testreihenfolge des Nutzers: output_dereverb vor
     output_deecho_aggressive).
Beide UVR-Modelle laufen ueber das Paket "audio-separator" (VR Architecture),
da die Modell-IDs exakt mit den vom Nutzer per UVR5-GUI erzeugten
Testdateien uebereinstimmen (vocals_(No Reverb)_UVR-DeEcho-DeReverb.flac,
vocals_(No Echo)_UVR-De-Echo-Aggressive.flac).

Laeuft in der acestep_stems Umgebung (gleiche Demucs-Installation wie
stem_clean_step.py/reseparate_vocal_step.py), zusaetzlich benoetigt:
pip install "audio-separator[cpu]" (oder "[gpu]" auf RunPod). Die beiden
UVR-Modelldateien werden beim ersten Lauf automatisch in <output_dir der
Modelle>/uvr_models neben diesem Skript heruntergeladen und danach
wiederverwendet (kein erneuter Download bei spaeteren Laeufen).

Aufruf:
  python vocal_separate_uvr_clean_step.py <input_song.wav> <output_dir>

Erzeugt in <output_dir>:
  vocals_uvr_clean.wav  - gesaeuberte Gesangsspur, STEREO, dereverbt+deechot,
                          Eingang fuer gate_vocal_stem.py und danach
                          flow_edit_morph (Schritt 2).
"""
import logging
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

DEMUCS_MODEL = "htdemucs_ft"
UVR_DEREVERB_MODEL = "UVR-DeEcho-DeReverb.pth"
UVR_DEECHO_MODEL = "UVR-De-Echo-Aggressive.pth"


def _pick_torch_device():
    """Erkennt automatisch die schnellste verfuegbare Hardware: CUDA (RunPod/
    Linux-GPU), sonst MPS (Apple-GPU auf dem Mac), sonst None (Demucs waehlt
    dann selbst/faellt auf CPU zurueck). NEU (9. Okt 2026, Nutzerwunsch
    "soll alles moeglicher auf gpu laufen"): vorher stand hier fest "mps"
    (nur fuer den Mac gedacht) - auf RunPod (Linux, kein Apple-Metal) ist
    "mps" dort NIE verfuegbar, Demucs ist also bisher auf JEDEM RunPod-Lauf
    erst gescheitert und dann auf CPU zurueckgefallen, obwohl eine GPU
    bereitstand und bezahlt wurde."""
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
    import subprocess

    cmd = [sys.executable, "-m", "demucs", "--name", DEMUCS_MODEL, "--two-stems", "vocals", "--out", work_dir]
    if device:
        cmd += ["-d", device]
    cmd.append(input_path)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)


def separate_vocals_htdemucs_ft(input_path, work_dir):
    """Demucs htdemucs_ft, --two-stems=vocals, OHNE --shifts/--overlap (siehe
    Modul-Docstring) - liefert den isolierten Gesang STEREO zurueck."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(work_dir, DEMUCS_MODEL, base_name, "vocals.wav")

    # NEU (10. Okt 2026, Geschwindigkeit): nur das Vocals-Spezialmodell von
    # htdemucs_ft rechnen statt aller 4 - ergebnisgleich (die anderen 3
    # gehen fuer "vocals" mit Gewicht 0 ein), ca. 4x schneller. Siehe
    # demucs_vocals_fast.py. shifts=1/overlap=0.25 = Kommandozeilen-
    # Standard, also exakt die bisherigen Werte dieses Schritts.
    try:
        from demucs_vocals_fast import separate_vocals
        if separate_vocals(input_path, vocals_path, model_name=DEMUCS_MODEL,
                           shifts=1, overlap=0.25, device=_pick_torch_device()):
            vocals, sr = sf.read(vocals_path)
            return vocals.astype(np.float64), sr
    except Exception as e:  # noqa: BLE001 - Rueckfall auf Kommandozeile unten
        print(f"Demucs-Schnellweg nicht möglich ({e}), nutze Kommandozeile...", file=sys.stderr)

    result = _run_demucs(input_path, work_dir, device=_pick_torch_device())
    if result.returncode != 0 and "Separated tracks" not in result.stderr:
        print("GPU-Trennung fehlgeschlagen, versuche CPU-Rückfall...", file=sys.stderr)
        result = _run_demucs(input_path, work_dir, device="cpu")
        if result.returncode != 0 and "Separated tracks" not in result.stderr:
            raise RuntimeError(f"Demucs-Trennung fehlgeschlagen: {result.stderr[-500:]}")

    vocals, sr = sf.read(vocals_path)
    return vocals.astype(np.float64), sr


def run_uvr_stage(input_path, output_dir, model_filename, target_stem, model_file_dir):
    """Laedt ein einzelnes UVR-VR-Modell (via audio-separator) und wendet es
    auf input_path an, nur der gewuenschte Ziel-Stem wird berechnet/geschrieben."""
    from audio_separator.separator import Separator

    os.makedirs(output_dir, exist_ok=True)
    separator = Separator(
        output_dir=output_dir,
        output_format="WAV",
        output_single_stem=target_stem,
        model_file_dir=model_file_dir,
        log_level=logging.WARNING,
    )
    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(input_path)
    if not output_files:
        raise RuntimeError(f"UVR-Modell {model_filename} hat keine Ausgabedatei erzeugt.")

    out_path = output_files[0]
    if not os.path.isabs(out_path):
        out_path = os.path.join(output_dir, out_path)
    return out_path


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python vocal_separate_uvr_clean_step.py <input.wav> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    model_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uvr_models")

    print(f"Trenne Gesang ({DEMUCS_MODEL}, two-stems=vocals)...")
    with tempfile.TemporaryDirectory() as work_dir:
        vocals, sr = separate_vocals_htdemucs_ft(input_path, work_dir)
        vocals_isolated_path = os.path.join(work_dir, "vocals_isolated.wav")
        sf.write(vocals_isolated_path, vocals, sr)

        print("Entferne Hall (UVR-DeEcho-DeReverb, Ziel-Stem 'no reverb')...")
        dereverb_path = run_uvr_stage(
            vocals_isolated_path, os.path.join(work_dir, "dereverb"),
            UVR_DEREVERB_MODEL, "no reverb", model_file_dir,
        )

        print("Entferne Echo (UVR-De-Echo-Aggressive, Ziel-Stem 'no echo')...")
        deecho_path = run_uvr_stage(
            dereverb_path, os.path.join(work_dir, "deecho"),
            UVR_DEECHO_MODEL, "no echo", model_file_dir,
        )

        final_audio, final_sr = sf.read(deecho_path)

    out_path = os.path.join(output_dir, "vocals_uvr_clean.wav")
    sf.write(out_path, final_audio, final_sr)
    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
