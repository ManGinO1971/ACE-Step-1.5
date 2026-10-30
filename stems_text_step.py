#!/usr/bin/env python3
"""
HIELTECH Stems – Songtext automatisch mitschreiben (30. Okt 2026).

Die Gesangskette (Lead V2 / Background V2 aus dem Musik-Generator) braucht den Songtext. In der Stems-App gibt es
keinen – deshalb hört ein Spracherkennungs-Modell (Whisper large-v3, MIT-Lizenz) den GETRENNTEN Gesang ab:
  - Lead-Spur  -> Lead-Zeilen (eine Zeile je erkanntem Satzstück, Zeitstempel)
  - Background-Spur -> Text in ( … ) an der passenden Stelle eingefügt (wie im Musik-Generator)
Der Text wird in der App ANGEZEIGT und kann korrigiert werden, bevor neu gesungen wird (Wunsch 30.10.).
Stellen, an denen die Spur praktisch still ist, werden verworfen (Whisper "erfindet" dort sonst Wörter).

Aufruf: python stems_text_step.py <lead_ref.wav> <backing_ref.wav> <ausgabe.json>
Ausgabe-JSON: {"ok", "lyrics": "<Text mit (…)>", "lead": [{start,end,text}], "backing": [...], "sprache": ...}
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

MODELL = os.environ.get("HIELTECH_WHISPER_MODELL", "openai/whisper-large-v3")
CACHE = os.environ.get("HIELTECH_WHISPER_CACHE") or ("/runpod-volume/hf_whisper" if os.path.isdir("/runpod-volume") else None)
SR16 = 16000


def lade16(p):
    import librosa
    a, sr = sf.read(p, always_2d=True, dtype="float32")
    return librosa.resample(a.mean(1), orig_sr=sr, target_sr=SR16)


def erkenner():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    m = AutoModelForSpeechSeq2Seq.from_pretrained(MODELL, torch_dtype=dt, low_cpu_mem_usage=True, cache_dir=CACHE).to(dev)
    p = AutoProcessor.from_pretrained(MODELL, cache_dir=CACHE)
    return pipeline("automatic-speech-recognition", model=m, tokenizer=p.tokenizer, feature_extractor=p.feature_extractor,
                    torch_dtype=dt, device=dev, chunk_length_s=30, batch_size=8)


def mitschreiben(asr, y):
    if len(y) < SR16 or float(np.abs(y).max()) < 1e-4:
        return []
    y = (y / max(float(np.abs(y).max()), 1e-6) * 0.9).astype(np.float32)
    res = asr({"raw": y, "sampling_rate": SR16}, return_timestamps=True, generate_kwargs={"task": "transcribe"})
    pegel_max = 20 * np.log10(np.sqrt(np.convolve(y ** 2, np.ones(SR16 // 2) / (SR16 // 2), mode="same")).max() + 1e-9)
    out, letzte = [], ""
    for c in res.get("chunks", []):
        t0, t1 = c.get("timestamp", (None, None))
        txt = (c.get("text") or "").strip()
        if not txt or t0 is None:
            continue
        t1 = t1 if t1 is not None else t0 + 2.0
        seg = y[int(t0 * SR16):int(t1 * SR16)]
        if len(seg) < SR16 // 10:
            continue
        pegel = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)
        if pegel < pegel_max - 35:          # praktisch still -> erfundener Text
            continue
        if txt == letzte:                   # Whisper-Wiederholungsschleife
            continue
        letzte = txt
        out.append({"start": round(float(t0), 2), "end": round(float(t1), 2), "text": txt})
    return out


def zusammensetzen(lead, backing):
    """Ein Text wie im Musik-Generator: Lead-Zeilen, Background-Stellen in ( … ) dazwischen, nach Zeit sortiert.
    Background, das zeitlich in einer Lead-Zeile liegt, kommt ans Zeilenende ( … )."""
    zeilen = [dict(z, art="lead") for z in lead]
    for b in backing:
        drin = [z for z in zeilen if z["art"] == "lead" and z["start"] - 0.3 <= b["start"] <= z["end"] + 0.3]
        if drin:
            drin[0].setdefault("bg", []).append(b["text"])
        else:
            zeilen.append(dict(b, art="backing"))
    zeilen.sort(key=lambda z: z["start"])
    text = []
    for z in zeilen:
        if z["art"] == "backing":
            text.append(f"({z['text']})")
        else:
            text.append(z["text"] + "".join(f" ({t})" for t in z.get("bg", [])))
    return "\n".join(text)


def main():
    lead_p, back_p, out_p = sys.argv[1:4]
    asr = erkenner()
    lead = mitschreiben(asr, lade16(lead_p))
    backing = mitschreiben(asr, lade16(back_p)) if os.path.isfile(back_p) else []
    bericht = {"ok": True, "modell": MODELL, "lyrics": zusammensetzen(lead, backing), "lead": lead, "backing": backing}
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(bericht, f, ensure_ascii=False, indent=1)
    print(json.dumps({"ok": True, "zeilen_lead": len(lead), "stellen_backing": len(backing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
