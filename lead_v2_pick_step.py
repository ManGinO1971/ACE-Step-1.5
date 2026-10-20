#!/usr/bin/env python3
"""
Lead Vocals v2 (NEU, 20. Okt 2026) – "automatisch die beste von 2".

Das Modell singt den Lead bei jedem Durchlauf etwas anders (Hörtest 20.10.:
an 1:43/1:44 fehlten in einem Durchlauf kurze Stücke). Darum werden mehrere
Varianten erzeugt und hier die beste gewählt:

  1. Text-Score: Text-Abgleich (lyrics_align_step.py --align, wav2vec2, wie
     "Text korrigieren") gegen den Lead-Text. Summe über alle gefundenen
     Zeilen: Sicherheit x Anzahl gesungener Wörter -> "sicher gesungene Wörter".
  2. Lücken: Sekunden, in denen im ORIGINAL der Lead singt (Lead-Referenz),
     die Variante aber stumm ist.
  Gewählt wird die Variante mit dem höchsten Text-Score; liegen zwei Varianten
  weniger als 2 % auseinander, entscheidet die geringere Lückenzeit.

Aufruf: python lead_v2_pick_step.py <lead_text.txt> <lead_ref.wav|-> <variante1> <variante2> [...]
Letzte stdout-Zeile: JSON {"ok": true, "chosen": <index>, "scores": [...]}
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def lead_lines(text):
    lines = []
    for raw in (text or "").splitlines():
        t = raw.strip()
        if not t or (t.startswith("[") and t.endswith("]")):
            continue
        words = t.split()
        if words:
            lines.append(words)
    return lines


def text_score(variant, lines_path, work):
    out = os.path.join(work, os.path.basename(variant) + ".align.json")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    r = subprocess.run([sys.executable, os.path.join(HERE, "lyrics_align_step.py"), "--align", variant, lines_path, out],
                       capture_output=True, text=True, cwd=HERE, env=env, timeout=600)
    if r.returncode != 0 or not os.path.isfile(out):
        print(f"Text-Abgleich für {os.path.basename(variant)} fehlgeschlagen: {(r.stderr or '')[-300:]}", file=sys.stderr, flush=True)
        return None, 0
    with open(out, "r", encoding="utf-8") as f:
        inst = json.load(f).get("instances", [])
    score = sum(float(i.get("conf", 0)) * sum(i.get("words", [])) for i in inst)
    return round(score, 2), len(inst)


def env_db(path, hop_s=0.01):
    import librosa
    import soundfile as sf

    y, sr = sf.read(path, always_2d=True)
    hop = int(sr * hop_s)
    e = 20 * np.log10(librosa.feature.rms(y=y.mean(axis=1), frame_length=hop * 4, hop_length=hop)[0] + 1e-9)
    return e


def gap_seconds(variant, lead_ref):
    if not lead_ref or lead_ref == "-" or not os.path.isfile(lead_ref):
        return None
    r = env_db(lead_ref)
    v = env_db(variant)
    n = min(len(r), len(v))
    r, v = r[:n], v[:n]
    ref_level = np.percentile(r[r > r.max() - 60], 95)
    v_peak = np.percentile(v, 99)
    gap = (r > ref_level - 30) & (v < v_peak - 45)
    # nur Lücken ab 150 ms zählen
    total, i = 0, 0
    while i < n:
        if gap[i]:
            j = i
            while j < n and gap[j]:
                j += 1
            if j - i >= 15:
                total += j - i
            i = j
        else:
            i += 1
    return round(total / 100.0, 2)


def main():
    if len(sys.argv) < 4:
        print("Nutzung: python lead_v2_pick_step.py <lead_text.txt> <lead_ref.wav|-> <variante1> [variante2 ...]")
        return 1
    text_path, lead_ref, variants = sys.argv[1], sys.argv[2], [os.path.abspath(v) for v in sys.argv[3:]]
    if lead_ref != "-":
        lead_ref = os.path.abspath(lead_ref)
    with open(text_path, "r", encoding="utf-8") as f:
        lines = lead_lines(f.read())
    scores = []
    with tempfile.TemporaryDirectory() as work:
        lines_path = os.path.join(work, "lines.json")
        with open(lines_path, "w", encoding="utf-8") as f:
            json.dump(lines, f, ensure_ascii=False)
        for idx, v in enumerate(variants):
            ts, n_inst = text_score(v, lines_path, work) if lines else (None, 0)
            gs = gap_seconds(v, lead_ref)
            scores.append({"variant": idx, "text_score": ts, "lines_found": n_inst, "gap_seconds": gs})
            print(f"Variante {idx + 1}: sicher gesungene Wörter {ts}, Zeilen {n_inst}, Lücken {gs} s", file=sys.stderr, flush=True)

    def key(s):
        return (s["text_score"] if s["text_score"] is not None else -1.0)

    best = max(scores, key=key)
    top = key(best)
    # Text-Abgleich nicht möglich (alle None/0) -> nur nach Lücken entscheiden
    near = [s for s in scores if key(s) >= top * 0.98] if top > 0 else list(scores)
    chosen = min(near, key=lambda s: (s["gap_seconds"] if s["gap_seconds"] is not None else 1e9, -key(s)))
    print(json.dumps({"ok": True, "chosen": chosen["variant"], "scores": scores}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
