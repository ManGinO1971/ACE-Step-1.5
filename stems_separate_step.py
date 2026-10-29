#!/usr/bin/env python3
"""
HIELTECH Stems – Test Stufe 1 (29. Okt 2026): Song auf der GPU in Spuren trennen.

  1. Gesang: Mischung 8 (Min-Spec aus Kim Mel-RoFormer + BS-RoFormer Viperx + Kim FT2 Bleedless)
  2. Lead / Backing: Karaoke-RoFormer auf dem Gesang (alle Backing-Stimmen in EINER Spur)
  3. Instrumente: Demucs htdemucs_6s auf dem Instrumental -> drums, bass, guitar, piano, other

Aufruf: python stems_separate_step.py <song> <ausgabeordner>
Erzeugt: lead.wav backing.wav drums.wav bass.wav guitar.wav piano.wav other.wav instrumental.wav
Letzte stdout-Zeile: JSON-Bericht (Zeiten je Schritt).
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from backing_chor_prepare_step import ENSEMBLE_MODELS, min_spec, run_model  # noqa: E402
from backing_separate_step import run_karaoke  # noqa: E402


def to_wav(src, dst):
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", dst],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + (r.stderr or "")[-300:])


def fit(a, n):
    a = a[:n]
    return np.pad(a, ((0, n - a.shape[0]), (0, 0))) if a.shape[0] < n else a


def main():
    song, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="stems_sep_")
    zeiten = {}
    t = time.time()
    wav = os.path.join(work, "song.wav")
    to_wav(song, wav)
    mix, sr = sf.read(wav, always_2d=True)
    n = mix.shape[0]

    # 1. Gesang (Mischung 8)
    parts = [run_model(wav, m, os.path.join(work, f"v{i}"), "vocals") for i, m in enumerate(ENSEMBLE_MODELS)]
    vocals, vsr = min_spec(parts)
    if vsr != sr:
        raise RuntimeError(f"Abtastrate passt nicht ({vsr} statt {sr})")
    vocals = fit(vocals, n)
    inst = mix - vocals
    sf.write(os.path.join(work, "vocals.wav"), vocals, sr, subtype="FLOAT")
    sf.write(os.path.join(out, "instrumental.wav"), inst, sr, subtype="PCM_24")
    zeiten["gesang"] = round(time.time() - t, 1)

    # 2. Lead / Backing
    t = time.time()
    backing, karaoke_model, lead = run_karaoke(os.path.join(work, "vocals.wav"), os.path.join(work, "kar"), os.path.join(HERE, "uvr_models"))
    for name, p in (("lead", lead), ("backing", backing)):
        a, s = sf.read(p, always_2d=True)
        sf.write(os.path.join(out, f"{name}.wav"), fit(a if a.shape[1] == 2 else np.repeat(a, 2, 1), n), s, subtype="PCM_24")
    zeiten["lead_backing"] = round(time.time() - t, 1)

    # 3. Instrumente (Demucs 6 Spuren auf dem Instrumental)
    t = time.time()
    import torch
    from demucs.api import Separator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sep = Separator(model="htdemucs_6s", device=dev, shifts=1, overlap=0.25, split=True, progress=False, jobs=0)
    _, res = sep.separate_audio_file(os.path.join(out, "instrumental.wav"))
    for k, v in res.items():
        if k == "vocals":
            continue  # Gesangsreste im Instrumental – nicht gebraucht
        a = v.detach().cpu().numpy().T
        sf.write(os.path.join(out, f"{k}.wav"), fit(a, n), sep.samplerate, subtype="PCM_24")
    zeiten["instrumente"] = round(time.time() - t, 1)

    print(json.dumps({"ok": True, "geraet": dev, "karaoke_model": karaoke_model, "sekunden_song": round(n / sr, 1),
                      "zeiten": zeiten, "spuren": sorted(f[:-4] for f in os.listdir(out) if f.endswith(".wav"))}))


if __name__ == "__main__":
    main()
