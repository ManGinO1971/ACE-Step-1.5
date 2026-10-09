#!/usr/bin/env python3
"""
Laedt alle Modell-Gewichte der "Vocals erzeugen"-Kette schon beim Docker-
Build ins Image (NEU, 9. Okt 2026, Nutzerwunsch "die Bearbeitung dauert
ewig").

Warum: ohne diesen Schritt laedt JEDER frisch gestartete RunPod-Worker beim
ersten Vocals-Auftrag erst einmal aus dem Internet nach:
  - Demucs htdemucs_ft (4 Teilmodelle) + htdemucs  -> ~/.cache/torch/hub
  - UVR-DeEcho-DeReverb.pth + UVR-De-Echo-Aggressive.pth -> /app/uvr_models
    (genau der Ordner, den vocal_separate_uvr_clean_step.py nutzt)
  - DeepFilterNet3 -> ~/.cache/DeepFilterNet
Da RunPod Worker regelmaessig austauscht/neu startet, passierte das nicht
nur einmal, sondern immer wieder - und jeder Download kann zusaetzlich
haengen oder scheitern. Im Image liegen die Dateien sofort bereit.

Wirft NIE einen Fehler (Exit-Code immer 0): schlaegt ein Download beim Build
fehl, laedt das jeweilige Skript es zur Laufzeit wie bisher selbst nach -
der Build bricht deswegen nicht ab.
"""
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _try(name, fn):
    try:
        fn()
        print(f"[Vorab-Download] OK: {name}", flush=True)
    except Exception as e:  # noqa: BLE001 - darf den Build nie abbrechen
        print(f"[Vorab-Download] übersprungen: {name} ({e})", flush=True)


def _demucs():
    from demucs.pretrained import get_model
    get_model("htdemucs_ft")
    get_model("htdemucs")


def _uvr():
    from audio_separator.separator import Separator
    model_dir = os.path.join(HERE, "uvr_models")
    os.makedirs(model_dir, exist_ok=True)
    # vocals_mel_band_roformer.ckpt: NEU (10. Okt 2026) fuer den RoFormer-
    # A/B-Test (HIELTECH_VOCALS_SEPARATOR=roformer, siehe
    # vocal_separate_uvr_clean_step.py) - sonst muesste jeder frische Worker
    # das grosse Modell beim ersten Test-Lauf erst herunterladen.
    for model in ("UVR-DeEcho-DeReverb.pth", "UVR-De-Echo-Aggressive.pth", "vocals_mel_band_roformer.ckpt"):
        sep = Separator(model_file_dir=model_dir, output_dir=model_dir, log_level=logging.WARNING)
        sep.load_model(model_filename=model)


def _deepfilternet():
    from df.enhance import init_df
    init_df()


if __name__ == "__main__":
    _try("Demucs htdemucs_ft + htdemucs", _demucs)
    _try("UVR DeEcho/DeReverb-Modelle + MelBand-RoFormer", _uvr)
    _try("DeepFilterNet3", _deepfilternet)
    sys.exit(0)
