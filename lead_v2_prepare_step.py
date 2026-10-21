#!/usr/bin/env python3
"""
Lead Vocals v2 (NEU, 20. Okt 2026) – Schritt 1: Vorbereitung.

Hörtest-Kette des Inhabers (Root Nah Bow, T6/T7): das Modell soll den Lead
frisch neu singen (Extract mit dem Base-Modell), darf dabei den Background
aber gar nicht erst HÖREN – sonst singt es ihn teilweise mit.

  1. Mel-Band-RoFormer (vocals_mel_band_roformer.ckpt): Gesang vom Song
     trennen. Rückfall: Demucs htdemucs_ft.
  2. Karaoke-Modell (wie backing_separate_step.py): Lead von Background.
  3. Song OHNE Background = Original-Song minus Background-Spur
     (-> Quelle für Extract).
  4. Lead- und Background-Spur bleiben als Referenz für die Zeitmaske
     (lead_mask_step.py) liegen.

Aufruf: python lead_v2_prepare_step.py <song> <ausgabeordner>
Erzeugt: song_ohne_background.wav, lead_ref.wav, backing_ref.wav
Letzte stdout-Zeile: JSON {"ok": true, ...}
"""
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backing_separate_step import _load_stereo, run_karaoke, write_context_without_lead  # noqa: E402
from vocal_separate_uvr_clean_step import (  # noqa: E402
    ROFORMER_VOCALS_MODEL,
    run_uvr_stage,
    separate_vocals_htdemucs_ft,
)


def separate_vocals(song, work, model_file_dir):
    """Gesang (Lead + Background) als Datei. RoFormer, sonst Demucs."""
    try:
        out = run_uvr_stage(song, os.path.join(work, "roformer"), ROFORMER_VOCALS_MODEL, "vocals", model_file_dir)
        # WICHTIG: neutral umbenennen - run_karaoke() erkennt den Lead am
        # "(Vocals)" im Dateinamen, und die RoFormer-Ausgabe traegt das schon.
        path = os.path.join(work, "vocals_all.wav")
        v, vsr = sf.read(out, always_2d=True)
        sf.write(path, v, vsr)
        print("Gesang getrennt mit Mel-Band-RoFormer", file=sys.stderr, flush=True)
        return path, "roformer"
    except Exception as e:  # noqa: BLE001 - Rueckfall
        print(f"RoFormer nicht nutzbar ({e}) – Rückfall auf htdemucs_ft", file=sys.stderr, flush=True)
    vocals, sr = separate_vocals_htdemucs_ft(song, work)
    path = os.path.join(work, "vocals_all.wav")
    sf.write(path, vocals, sr)
    return path, "htdemucs_ft"


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python lead_v2_prepare_step.py <song> <ausgabeordner>")
        return 1
    song, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    model_file_dir = os.path.join(HERE, "uvr_models")

    with tempfile.TemporaryDirectory() as work:
        vocals_path, separator = separate_vocals(song, work, model_file_dir)
        backing_raw, karaoke_model, lead_path = run_karaoke(vocals_path, work, model_file_dir)

        import librosa

        sr = int(librosa.get_samplerate(song))
        mix = _load_stereo(song, sr)
        backing = _load_stereo(backing_raw, sr)
        n = min(mix.shape[1], backing.shape[1])
        out = mix.copy()
        out[:, :n] = mix[:, :n] - backing[:, :n]
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out *= 0.99 / peak
        sf.write(os.path.join(out_dir, "song_ohne_background.wav"), out.T, sr)

        lead, lsr = sf.read(lead_path, always_2d=True)
        sf.write(os.path.join(out_dir, "lead_ref.wav"), lead, lsr)
        braw, bsr = sf.read(backing_raw, always_2d=True)
        sf.write(os.path.join(out_dir, "backing_ref.wav"), braw, bsr)
        # NEU (21. Okt 2026, Background V2): Song OHNE Lead (Quelle für das
        # Neu-Singen des Backgrounds) + wie viele Sekunden Background es gibt.
        backing_seconds = None
        try:
            write_context_without_lead(song, lead_path, sr, os.path.join(out_dir, "song_ohne_lead.wav"))
            bm = braw.mean(axis=1)
            hop = max(1, int(bsr * 0.01))
            nfr = len(bm) // hop
            if nfr:
                e = np.sqrt(np.mean(bm[: nfr * hop].reshape(nfr, hop) ** 2, axis=1) + 1e-12)
                db = 20 * np.log10(e)
                lm = lead.mean(axis=1)
                lhop = max(1, int(lsr * 0.01))
                lfr = len(lm) // lhop
                ref = float(np.percentile(20 * np.log10(np.sqrt(np.mean(lm[: lfr * lhop].reshape(lfr, lhop) ** 2, axis=1) + 1e-12)), 95)) if lfr else float(db.max())
                backing_seconds = round(float(np.sum(db > max(ref, float(np.percentile(db, 95))) - 30)) / 100.0, 1)
        except Exception as e:  # noqa: BLE001 - Lead-Kette braucht das nicht
            print(f"Song ohne Lead nicht erzeugt: {e}", file=sys.stderr, flush=True)

    report = {"ok": True, "separator": separator, "karaoke_model": karaoke_model, "samplerate": sr, "backing_seconds": backing_seconds}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
