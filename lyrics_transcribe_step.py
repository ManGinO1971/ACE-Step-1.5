#!/usr/bin/env python3
"""
"Text korrigieren" (NEU, 18. Okt 2026, Nutzerwunsch: der gesungene Song weicht
manchmal vom eingegebenen Songtext ab - Woerter fehlen, Stellen werden
wiederholt. Ein Knopf soll den Songtext automatisch an das tatsaechlich
Gesungene anpassen, inkl. Background in (...) und {...}).

Dieses Skript macht NUR das Hoeren (GPU):
  1. Demucs htdemucs_ft -> komplette Gesangsspur (wie ueberall in der Kette)
  2. Karaoke-Trennmodell -> Lead und Background getrennt
  3. UVR-DeEcho-DeReverb auf beiden Spuren (Hall stoert die Erkennung)
  4. Whisper (transformers, Standard openai/whisper-large-v3) schreibt Lead
     und Background getrennt mit, mit Zeitstempel pro Wort
  5. Pro Wort wird die Lautstaerke der Spur gemessen (dB relativ zur lauten
     Stelle der Spur) - damit kann der VPS "erfundene" Woerter in stillen
     Stellen (typischer Whisper-Fehler) aussortieren.
Das Abgleichen mit dem Original-Songtext macht der VPS (runpod-handler.js),
damit es ohne neuen RunPod-Build verbessert werden kann.

Aufruf: python lyrics_transcribe_step.py <song> <ausgabe.json>
Umgebung: HIELTECH_ASR_MODEL (Standard openai/whisper-large-v3),
          HIELTECH_ASR_LANGUAGE (Standard leer = automatisch)
"""
import gc
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ASR_MODEL = os.environ.get("HIELTECH_ASR_MODEL") or "openai/whisper-large-v3"
ASR_LANGUAGE = (os.environ.get("HIELTECH_ASR_LANGUAGE") or "").strip() or None
# 18. Okt 2026: erster Live-Test lief in "GPU-Speicher voll" - Whisper lief im
# selben Prozess wie die Trennmodelle (deren Speicher war noch belegt) und
# mit batch_size 8. Jetzt: Whisper in einem EIGENEN Prozess (Trennung ist
# dann komplett freigegeben) und kleinere Stapel; bei Speichermangel
# automatisch nochmal mit Stapel 1.
ASR_BATCH = int(os.environ.get("HIELTECH_ASR_BATCH") or "2")


def _cache_dir():
    for c in (os.environ.get("HF_HOME"), "/runpod-volume/hf_cache", "/app/checkpoints/hf_cache"):
        if c and os.path.isdir(c):
            return c
    return None


def _mono16k(path):
    import librosa

    y, _ = librosa.load(path, sr=16000, mono=True)
    return y.astype(np.float32)


def _level_db_track(y, hop=160):
    """RMS in dB pro 10 ms, relativ zum 95. Perzentil der lauten Stellen."""
    n = len(y) // hop
    if n <= 0:
        return np.zeros(0), 0.0
    frames = y[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    voiced = rms[rms > 1e-4]
    ref = float(np.percentile(voiced, 95)) if voiced.size else 1.0
    return 20 * np.log10(rms / max(ref, 1e-9)), ref


def _word_db(level, s, e):
    if level.size == 0:
        return -99.0
    a = max(0, int(s * 100))
    b = min(level.size, max(a + 1, int(math.ceil(e * 100))))
    if a >= level.size:
        return -99.0
    return float(np.max(level[a:b]))


_ASR = None


def _asr():
    global _ASR
    if _ASR is None:
        import torch
        from transformers import pipeline

        cuda = torch.cuda.is_available()
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL,
            torch_dtype=torch.float16 if cuda else torch.float32,
            device=0 if cuda else -1,
            model_kwargs={"cache_dir": _cache_dir()} if _cache_dir() else {},
            chunk_length_s=30,
            batch_size=max(1, ASR_BATCH),
        )
    return _ASR


def transcribe(path):
    y = _mono16k(path)
    level, _ref = _level_db_track(y)
    gk = {"task": "transcribe"}
    if ASR_LANGUAGE:
        gk["language"] = ASR_LANGUAGE
    out = _asr()({"raw": y, "sampling_rate": 16000}, return_timestamps="word", generate_kwargs=gk)
    words = []
    dur = len(y) / 16000.0
    for ch in out.get("chunks") or []:
        text = (ch.get("text") or "").strip()
        ts = ch.get("timestamp") or (None, None)
        s = ts[0] if ts[0] is not None else (words[-1]["e"] if words else 0.0)
        e = ts[1] if ts[1] is not None else min(dur, s + 0.5)
        if not text:
            continue
        words.append({"w": text, "s": round(float(s), 3), "e": round(float(e), 3), "db": round(_word_db(level, s, e), 1)})
    return words


def _free_gpu():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def transcribe_in_subprocess(path, label):
    """Whisper in einem frischen Prozess - bei Speichermangel nochmal mit Stapel 1."""
    out = path + ".asr.json"
    for batch in (str(max(1, ASR_BATCH)), "1"):
        env = dict(os.environ, HIELTECH_ASR_BATCH=batch)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--asr", path, out],
                           capture_output=True, text=True, env=env, cwd=HERE)
        if r.returncode == 0 and os.path.isfile(out):
            with open(out, "r", encoding="utf-8") as f:
                return json.load(f)
        tail = ((r.stdout or "") + (r.stderr or ""))[-1200:]
        print(f"Whisper ({label}, Stapel {batch}) fehlgeschlagen: {tail}", file=sys.stderr, flush=True)
        if "out of memory" not in tail.lower() or batch == "1":
            break
    raise RuntimeError(f"Whisper ({label}) fehlgeschlagen")


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--asr":
        words = transcribe(sys.argv[2])
        with open(sys.argv[3], "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False)
        return
    if len(sys.argv) < 3:
        print("Nutzung: python lyrics_transcribe_step.py <song> <ausgabe.json>")
        sys.exit(1)
    song, out_json = sys.argv[1], sys.argv[2]
    from backing_separate_step import (
        UVR_DEREVERB_MODEL,
        run_karaoke,
        run_uvr_stage,
        separate_vocals_htdemucs_ft,
    )
    from instrumental_balance_step import lufs

    model_file_dir = os.path.join(HERE, "uvr_models")
    report = {"ok": True, "asr_model": ASR_MODEL}
    with tempfile.TemporaryDirectory() as work:
        print("Text korrigieren: trenne Gesang (htdemucs_ft)...", file=sys.stderr, flush=True)
        vocals, sr = separate_vocals_htdemucs_ft(song, work)
        vocals_path = os.path.join(work, "vocals_all.wav")
        sf.write(vocals_path, vocals, sr)
        backing_raw, model_used, lead_raw = run_karaoke(vocals_path, work, model_file_dir)
        report["karaoke_model"] = model_used

        def dereverb(p, name):
            try:
                return run_uvr_stage(p, os.path.join(work, name), UVR_DEREVERB_MODEL, "no reverb", model_file_dir)
            except Exception as e:  # noqa: BLE001 - dann eben mit Hall erkennen
                print(f"Hall-Entfernung ({name}) übersprungen: {e}", file=sys.stderr, flush=True)
                return p

        lead = dereverb(lead_raw, "lead_dry")
        braw, bsr = sf.read(backing_raw)
        l_back = lufs(np.atleast_2d(braw.T) if braw.ndim > 1 else braw[None, :], bsr)
        report["backing_lufs"] = round(l_back, 1) if math.isfinite(l_back) else None
        has_backing = math.isfinite(l_back) and l_back > float(os.environ.get("HIELTECH_BACKING_ABSENT_LUFS", "-45"))
        report["has_backing"] = bool(has_backing)

        backing_dry = dereverb(backing_raw, "backing_dry") if has_backing else None
        _free_gpu()
        print(f"Text korrigieren: Whisper ({ASR_MODEL}) hört den Lead...", file=sys.stderr, flush=True)
        report["lead"] = transcribe_in_subprocess(lead, "Lead")
        report["backing"] = []
        if backing_dry:
            print("Text korrigieren: Whisper hört den Background...", file=sys.stderr, flush=True)
            try:
                report["backing"] = transcribe_in_subprocess(backing_dry, "Background")
            except Exception as e:  # noqa: BLE001 - Lead reicht notfalls
                print(f"Background nicht mitgehört: {e}", file=sys.stderr, flush=True)
                report["has_backing"] = False
        report["duration"] = round(len(vocals) / float(sr), 2) if vocals.ndim == 1 else round(vocals.shape[0] / float(sr), 2)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    print(json.dumps({"ok": True, "lead_words": len(report["lead"]), "backing_words": len(report["backing"])}), flush=True)


if __name__ == "__main__":
    main()
