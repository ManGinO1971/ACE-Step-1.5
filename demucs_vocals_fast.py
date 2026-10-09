#!/usr/bin/env python3
"""
Schnelle, ERGEBNISGLEICHE Gesangstrennung mit Demucs htdemucs_ft (NEU,
10. Okt 2026, Nutzerwunsch "13-14 Minuten fuer einen 3:30-Song - geht das
noch schneller?").

Hintergrund: htdemucs_ft ist ein "Bag" aus 4 Spezialmodellen (Drums, Bass,
Other, Vocals) mit der Gewichtstabelle (demucs/remote/htdemucs_ft.yaml)
    [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
Die Gesangsspur kommt also zu 100% aus dem 4. Modell - die Ausgaben der
anderen drei werden fuer "vocals" mit 0 multipliziert (demucs/apply.py,
BagOfModels-Zweig). Trotzdem rechnet die Kommandozeile ("python -m demucs
--two-stems vocals") alle 4 Modelle komplett durch, inklusive aller
--shifts. Bei shifts=5 sind das 20 Durchlaeufe, von denen nur 5 ins
Ergebnis eingehen.

Dieses Modul rechnet nur das Vocals-Spezialmodell - mit exakt derselben
Vor-/Nachbearbeitung wie die Kommandozeile (demucs.api.Separator: gleiche
Normalisierung, gleiches split/overlap/shifts/segment; demucs.api.save_audio
mit den Kommandozeilen-Standards clip="rescale", 16 Bit). Ergebnis: gleiche
Gesangsspur, ca. 1/4 der Rechenzeit. (Einzige Einschraenkung: --shifts
nutzt in Demucs ungeseedete Zufallsverschiebungen - das Ergebnis schwankt
also schon immer minimal von Lauf zu Lauf, mit wie ohne diese Optimierung.)

Funktioniert die Abkuerzung aus irgendeinem Grund nicht (anderes Modell,
unerwartete Gewichte, Import-Fehler), gibt separate_vocals() None zurueck -
die aufrufenden Skripte fallen dann auf die bisherige Kommandozeile zurueck.
"""
import os
import sys


def pick_vocals_specialist(model):
    """Gibt das Untermodell zurueck, das ALLEIN die Gesangsspur liefert, oder
    None, falls die Gewichte das nicht eindeutig hergeben (dann ist die
    Abkuerzung nicht ergebnisgleich und wird nicht benutzt)."""
    from demucs.apply import BagOfModels

    if not isinstance(model, BagOfModels):
        return model
    vi = model.sources.index("vocals")
    contributing = [i for i, w in enumerate(model.weights) if float(w[vi]) != 0.0]
    if len(contributing) != 1:
        return None
    sub = model.models[contributing[0]]
    if list(sub.sources) != list(model.sources):
        return None
    return sub


def separate_vocals(input_path, output_wav, model_name="htdemucs_ft", shifts=1, overlap=0.25, device=None):
    """Schreibt die Gesangsspur nach output_wav (16 Bit, wie die Kommandozeile).
    Gibt output_wav zurueck, oder None, wenn die Abkuerzung nicht anwendbar
    war bzw. fehlschlug (dann bitte Kommandozeilen-Rueckfall nutzen)."""
    try:
        import torch
        from demucs.api import Separator, save_audio

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        sep = Separator(
            model=model_name, device=device, shifts=int(shifts), overlap=float(overlap),
            split=True, progress=False, jobs=0, segment=None,
        )
        sub = pick_vocals_specialist(sep._model)
        if sub is None:
            print("[Demucs-Schnellweg] Gewichte nicht eindeutig – nutze normale Kommandozeile.",
                  file=sys.stderr, flush=True)
            return None
        sep._model = sub
        _origin, res = sep.separate_audio_file(input_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
        save_audio(
            res["vocals"], output_wav, samplerate=sep.samplerate,
            bitrate=320, preset=2, clip="rescale", as_float=False, bits_per_sample=16,
        )
        print(f"[Demucs-Schnellweg] nur Vocals-Spezialmodell gerechnet ({model_name}, "
              f"shifts={shifts}, overlap={overlap}, device={device})", file=sys.stderr, flush=True)
        return output_wav
    except Exception as e:  # noqa: BLE001 - Aufrufer faellt auf Kommandozeile zurueck
        print(f"[Demucs-Schnellweg] nicht möglich ({e}) – nutze normale Kommandozeile.",
              file=sys.stderr, flush=True)
        return None
