#!/usr/bin/env python3
"""
Schritt 1 von "Background Vocals erzeugen" (NEU, 13. Okt 2026): trennt die
vorhandenen Background-Vocals vom Lead-Gesang und reinigt sie - nach
demselben Prinzip wie Schritt 1 von "Vocals erzeugen"
(vocal_separate_uvr_clean_step.py).

Demucs kann Lead und Background NICHT trennen (liefert nur eine gemeinsame
Gesangsspur). Deshalb:
  1. Demucs htdemucs_ft -> komplette Gesangsspur (Lead + Background), wie bei
     "Vocals erzeugen".
  2. Karaoke-Trennmodell (audio-separator) auf dieser Gesangsspur: trennt
     Lead (Hauptstimme) von allem anderen = Background-Vocals. Mehrere
     Modelle werden der Reihe nach versucht (HIELTECH_KARAOKE_MODELS).
  3. Background-Spur reinigen: UVR-DeEcho-DeReverb -> UVR-De-Echo-
     Aggressive (gleiche Stufen wie bei "Vocals erzeugen"). Das Noise-Gate
     setzt danach runpod_entrypoint.py (gate_vocal_stem.py).
  4. Pruefung, ob ueberhaupt Background-Gesang vorhanden ist (Lautheit der
     Background-Spur im Verhaeltnis zur gesamten Gesangsspur).

Aufruf: python backing_separate_step.py <song> <output_dir>
Erzeugt in <output_dir>: backing_clean.wav (nur wenn vorhanden) und
context_no_lead.wav (Song OHNE Lead-Gesang = Instrumente + Original-
Background; NEU 15. Okt 2026, Vorlage fuer den Lego-Modus, damit das Modell
den Lead nicht mitsingt)
Letzte stdout-Zeile: JSON {"ok": true, "has_backing": bool, ...}
"""
import json
import math
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vocal_separate_uvr_clean_step import (  # noqa: E402
    UVR_DEECHO_MODEL,
    UVR_DEREVERB_MODEL,
    separate_vocals_htdemucs_ft,
    run_uvr_stage,
)
from instrumental_balance_step import lufs  # noqa: E402

DEFAULT_KARAOKE_MODELS = (
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt,"
    "6_HP-Karaoke-UVR.pth,"
    "5_HP-Karaoke-UVR.pth"
)
# Background gilt als "nicht vorhanden", wenn die Spur absolut sehr leise ist
# oder viel leiser als die gesamte Gesangsspur (dann sind es nur Trennreste).
ABSENT_ABS_LUFS = float(os.environ.get("HIELTECH_BACKING_ABSENT_LUFS", "-45"))
ABSENT_REL_DB = float(os.environ.get("HIELTECH_BACKING_ABSENT_REL_DB", "20"))


def karaoke_models():
    raw = os.environ.get("HIELTECH_KARAOKE_MODELS") or DEFAULT_KARAOKE_MODELS
    return [m.strip() for m in raw.split(",") if m.strip()]


def run_karaoke(vocals_path, out_dir, model_file_dir):
    """Gibt (backing_path, modellname) zurueck. Nimmt jeweils die Ausgabe, die
    NICHT die Hauptstimme ist (Dateiname ohne "(Vocals)")."""
    import logging

    from audio_separator.separator import Separator

    last_err = None
    for model in karaoke_models():
        try:
            d = os.path.join(out_dir, "karaoke_" + "".join(c if c.isalnum() else "_" for c in model)[:40])
            os.makedirs(d, exist_ok=True)
            sep = Separator(output_dir=d, output_format="WAV", model_file_dir=model_file_dir, log_level=logging.WARNING)
            sep.load_model(model_filename=model)
            outs = sep.separate(vocals_path) or []
            outs = [o if os.path.isabs(o) else os.path.join(d, o) for o in outs]
            if len(outs) < 2:
                raise RuntimeError(f"{model}: erwartet 2 Ausgaben, bekommen {len(outs)}")
            lead = [o for o in outs if "(vocals)" in os.path.basename(o).lower()]
            others = [o for o in outs if o not in lead]
            if not lead or not others:
                raise RuntimeError(f"{model}: Ausgaben nicht eindeutig zuordenbar: {[os.path.basename(o) for o in outs]}")
            print(f"Lead/Background-Trennung mit {model}", file=sys.stderr, flush=True)
            return others[0], model, lead[0]
        except Exception as e:  # noqa: BLE001 - naechstes Modell versuchen
            last_err = e
            print(f"Karaoke-Modell {model} nicht nutzbar ({e}) – versuche nächstes.", file=sys.stderr, flush=True)
    raise RuntimeError(f"Kein Karaoke-Trennmodell nutzbar: {last_err}")


def _load_stereo(path, sr):
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=False)
    y = np.atleast_2d(y)
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    return y[:2]


def write_context_without_lead(song, lead_path, sr, out_path):
    """Song minus Lead-Stimme (beide auf dieselbe Abtastrate gebracht)."""
    mix = _load_stereo(song, sr)
    lead = _load_stereo(lead_path, sr)
    n = min(mix.shape[1], lead.shape[1])
    ctx = mix[:, :n] - lead[:, :n]
    if mix.shape[1] > n:
        ctx = np.concatenate([ctx, mix[:, n:]], axis=1)
    peak = float(np.max(np.abs(ctx))) if ctx.size else 0.0
    if peak > 0.99:
        ctx = ctx * (0.99 / peak)
    sf.write(out_path, ctx.T, sr)
    print("Song ohne Lead-Gesang erzeugt (Vorlage für Lego)", file=sys.stderr, flush=True)


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python backing_separate_step.py <song> <output_dir>")
        sys.exit(1)
    song, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    model_file_dir = os.path.join(HERE, "uvr_models")

    with tempfile.TemporaryDirectory() as work:
        print("Trenne Gesang (htdemucs_ft)...", file=sys.stderr, flush=True)
        vocals, sr = separate_vocals_htdemucs_ft(song, work)
        vocals_path = os.path.join(work, "vocals_all.wav")
        sf.write(vocals_path, vocals, sr)

        backing_raw, model_used, lead_path = run_karaoke(vocals_path, work, model_file_dir)
        try:
            write_context_without_lead(song, lead_path, sr, os.path.join(out_dir, "context_no_lead.wav"))
        except Exception as e:  # noqa: BLE001 - nur fuer Lego noetig, nie fatal
            print(f"Song ohne Lead nicht erzeugbar ({e})", file=sys.stderr, flush=True)
        braw, bsr = sf.read(backing_raw)
        l_vocals = lufs(np.atleast_2d(vocals.T) if vocals.ndim > 1 else vocals[None, :], sr)
        l_back = lufs(np.atleast_2d(braw.T) if braw.ndim > 1 else braw[None, :], bsr)
        has_backing = (
            math.isfinite(l_back)
            and l_back > ABSENT_ABS_LUFS
            and (not math.isfinite(l_vocals) or l_back > l_vocals - ABSENT_REL_DB)
        )
        report = {
            "ok": True,
            "has_backing": bool(has_backing),
            "karaoke_model": model_used,
            "vocals_lufs": round(l_vocals, 1) if math.isfinite(l_vocals) else None,
            "backing_lufs": round(l_back, 1) if math.isfinite(l_back) else None,
        }
        if has_backing:
            print("Entferne Hall/Echo aus der Background-Spur...", file=sys.stderr, flush=True)
            dereverb = run_uvr_stage(backing_raw, os.path.join(work, "dereverb"), UVR_DEREVERB_MODEL, "no reverb", model_file_dir)
            deecho = run_uvr_stage(dereverb, os.path.join(work, "deecho"), UVR_DEECHO_MODEL, "no echo", model_file_dir)
            clean, csr = sf.read(deecho)
            sf.write(os.path.join(out_dir, "backing_clean.wav"), clean, csr)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
