"""HIELTECH (21. Nov 2026): eine Trennung mit audio-separator in der eigenen Umgebung /opt/hieltech_sep_venv
(audio-separator 0.44.5 = dieselbe Fassung wie auf dem Mac – braucht numpy 2, die Hauptumgebung hat numpy 1).
Aufruf: python hieltech_sep_lauf.py <eingabe> <modell> <ausgabe-ordner> <modell-ordner>
Letzte Zeile der Ausgabe: HIELTECH_SEP_JSON [liste der erzeugten Dateien]"""
import sys
import os
import json
import logging


def main():
    eingabe, modell, roh, modelle = sys.argv[1:5]
    try:
        import hieltech_librosa_kompat  # noqa: F401 – librosa.get_duration(filename=) für 0.44.5
    except Exception:  # noqa: BLE001
        pass
    from audio_separator.separator import Separator
    os.makedirs(roh, exist_ok=True)
    os.makedirs(modelle, exist_ok=True)
    s = Separator(output_dir=roh, model_file_dir=modelle, output_format="WAV", log_level=logging.WARNING)
    s.load_model(model_filename=modell)
    dateien = [f if os.path.isabs(f) else os.path.join(roh, f) for f in s.separate(eingabe)]
    print("HIELTECH_SEP_JSON " + json.dumps(dateien), flush=True)


if __name__ == "__main__":
    main()
