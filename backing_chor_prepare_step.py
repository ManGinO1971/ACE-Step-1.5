#!/usr/bin/env python3
"""
Background-Chor (NEU, 23. Okt 2026) – Schritt 1: saubere Gesangs-Vorlage.

Vom Inhaber per Hörtest gewählt (22. Okt 2026, "Mischung 8 – die Sauberste"):
  1. Gesang vom Song trennen mit DREI Modellen und pro Frequenz/Zeitpunkt das
     Leiseste nehmen (Min-Spec-Ensemble) = am wenigsten Instrumenten-Reste:
       - vocals_mel_band_roformer.ckpt (Kim Mel-Band-RoFormer)
       - model_bs_roformer_ep_317_sdr_12.9755.ckpt (BS-RoFormer Viperx 1297)
       - mel_band_roformer_kim_ft2_bleedless_unwa.ckpt (Kim FT2 Bleedless)
     Lead UND Background bleiben zusammen (der ganze Songtext wird nachgesungen).
  2. Hall raus: dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt
  3. Rauschen raus: denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt
  4. Tonart des Songs schätzen (nur Rückfall, falls die App keine schickt).

Aufruf: python backing_chor_prepare_step.py <song> <ausgabeordner>
Erzeugt: chor_vorlage.wav. Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import logging
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_MODELS = [
    "vocals_mel_band_roformer.ckpt",
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
]
DEREVERB_MODEL = os.environ.get("HIELTECH_CHOR_DEREVERB", "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt")
DENOISE_MODEL = os.environ.get("HIELTECH_CHOR_DENOISE", "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt")
NAMEN = ["c", "cis", "d", "es", "e", "f", "fis", "g", "as", "a", "b", "h"]


def run_model(path, model, out_dir, want):
    """Ein Trennmodell; gibt die Datei zurück, deren Stamm-Name 'want' enthält (ohne 'no ...')."""
    from audio_separator.separator import Separator

    os.makedirs(out_dir, exist_ok=True)
    sep = Separator(output_dir=out_dir, output_format="WAV", model_file_dir=os.path.join(HERE, "uvr_models"),
                    log_level=logging.WARNING)
    sep.load_model(model_filename=model)
    files = [f if os.path.isabs(f) else os.path.join(out_dir, f) for f in sep.separate(path)]
    for f in files:
        low = os.path.basename(f).lower()
        if f"({want})" in low:
            return f
    raise RuntimeError(f"{model}: keine '{want}'-Spur in {[os.path.basename(f) for f in files]}")


def min_spec(paths):
    import librosa

    xs = [sf.read(p, always_2d=True) for p in paths]
    sr = xs[0][1]
    arrs = []
    for a, s in xs:
        a = a.T.astype(np.float32)
        if s != sr:
            a = np.vstack([librosa.resample(c, orig_sr=s, target_sr=sr) for c in a])
        if a.shape[0] == 1:
            a = np.vstack([a, a])
        arrs.append(a[:2])
    n = min(a.shape[1] for a in arrs)
    out = []
    for ch in range(2):
        S = np.stack([librosa.stft(a[ch, :n], n_fft=2048, hop_length=512) for a in arrs])
        idx = np.argmin(np.abs(S), axis=0)
        M = np.take_along_axis(S, idx[None], 0)[0]
        out.append(librosa.istft(M, hop_length=512, length=n))
    return np.stack(out, 1), sr


def detect_key(path):
    """Krumhansl-Schmuckler auf der Chroma des Songs -> 'a-moll' / 'c-dur'."""
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True, duration=240)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = (-2.0, "a-moll")
    for t in range(12):
        for prof, art in ((major, "dur"), (minor, "moll")):
            r = float(np.corrcoef(chroma, np.roll(prof, t))[0, 1])
            if r > best[0]:
                best = (r, f"{NAMEN[t]}-{art}")
    return best[1], round(best[0], 3)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    song, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    report = {"ok": True, "ensemble": [], "dereverb": False, "denoise": False}
    with tempfile.TemporaryDirectory() as work:
        stems = []
        for i, m in enumerate(ENSEMBLE_MODELS):
            try:
                stems.append(run_model(song, m, os.path.join(work, f"m{i}"), "vocals"))
                report["ensemble"].append(m)
                print(f"Trennung {i + 1}/3 fertig: {m}", file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001 - ein Modell darf fehlen
                print(f"Trennmodell {m} übersprungen: {e}", file=sys.stderr, flush=True)
        if not stems:
            print(json.dumps({"ok": False, "error": "Kein Trennmodell lief"}), flush=True)
            return 2
        if len(stems) == 1:
            vocals, sr = sf.read(stems[0], always_2d=True)
        else:
            vocals, sr = min_spec(stems)
        cur = os.path.join(work, "mischung8.wav")
        sf.write(cur, vocals, sr, subtype="PCM_24")
        for model, want, key in ((DEREVERB_MODEL, "noreverb", "dereverb"), (DENOISE_MODEL, "dry", "denoise")):
            try:
                cur = run_model(cur, model, os.path.join(work, key), want)
                report[key] = True
                print(f"{key} fertig: {model}", file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001 - Vorlage bleibt dann ohne diesen Schritt
                print(f"{key} übersprungen: {e}", file=sys.stderr, flush=True)
        y, ysr = sf.read(cur, always_2d=True)
        sf.write(os.path.join(out_dir, "chor_vorlage.wav"), y, ysr, subtype="PCM_24")
        report["samplerate"] = int(ysr)
        report["seconds"] = round(len(y) / ysr, 1)
    try:
        report["key_detected"], report["key_conf"] = detect_key(song)
    except Exception as e:  # noqa: BLE001
        report["key_detected"] = ""
        print(f"Tonart nicht geschätzt: {e}", file=sys.stderr, flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
