#!/usr/bin/env python3
"""
HIELTECH Stems – Songtext automatisch mitschreiben (30. Okt 2026).

Die Gesangskette (Lead V2 / Background V2 aus dem Musik-Generator) braucht den Songtext. In der Stems-App gibt es
keinen – deshalb hört ein Spracherkennungs-Modell (Whisper large-v3-turbo, MIT-Lizenz) den GETRENNTEN Gesang ab:
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

# Xet-Zwischenspeicher liegt sonst auf dem (vollen) Volume – vor dem ersten huggingface-Import abschalten
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODELL = os.environ.get("HIELTECH_WHISPER_MODELL", "openai/whisper-large-v3-turbo")   # 1,6 GB statt 3 GB, fast gleich genau
CACHE = os.environ.get("HIELTECH_WHISPER_CACHE") or ("/runpod-volume/hf_whisper" if os.path.isdir("/runpod-volume") else None)
SR16 = 16000


def lade16(p):
    import librosa
    a, sr = sf.read(p, always_2d=True, dtype="float32")
    return librosa.resample(a.mean(1), orig_sr=sr, target_sr=SR16)


def _laden(cache, dt, nur_lokal=False):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    m = AutoModelForSpeechSeq2Seq.from_pretrained(MODELL, torch_dtype=dt, low_cpu_mem_usage=True, cache_dir=cache,
                                                  local_files_only=nur_lokal)
    p = AutoProcessor.from_pretrained(MODELL, cache_dir=cache, local_files_only=nur_lokal)
    return m, p


def erkenner():
    import tempfile
    import torch
    from transformers import pipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    # 31.10.: Volume voll ("Disk quota exceeded"). Whisper liegt aber schon fertig auf dem Volume
    # (hf_cache/models--openai--whisper-large-v3-turbo) -> zuerst NUR LESEN, ohne etwas zu schreiben.
    m = p = None
    for cache in ("/runpod-volume/hf_cache", None, CACHE):
        try:
            m, p = _laden(cache, dt, nur_lokal=True)
            print(f"[Stems-Text] Whisper aus vorhandenem Speicher: {cache}", file=sys.stderr, flush=True)
            break
        except Exception:  # noqa: BLE001
            continue
    if m is None:
        # 31.10.: Whisper auf dem Volume war unvollständig (Abbruch beim vollen Volume). Volume jetzt 80 GB ->
        # zuerst AUFS VOLUME laden (bleibt dort, nächstes Mal sofort da). Der Worker-Speicher (/tmp) ist zu
        # klein ("Disk quota exceeded" auch dort) – nur noch letzter Versuch.
        fehler = []
        for ziel in [c for c in (CACHE, "/runpod-volume/hf_cache") if c] + ["tmp"]:
            try:
                if ziel == "tmp":
                    ziel = tempfile.mkdtemp(prefix="whisper_", dir=os.environ.get("TMPDIR", "/tmp"))
                os.makedirs(ziel, exist_ok=True)
                m, p = _laden(ziel, dt)
                print(f"[Stems-Text] Whisper geladen nach: {ziel}", file=sys.stderr, flush=True)
                break
            except Exception as e:  # noqa: BLE001
                frei = ""
                try:
                    st = os.statvfs(ziel)
                    frei = f" (frei {st.f_bavail * st.f_frsize / 1e9:.1f} GB)"
                except Exception:  # noqa: BLE001
                    pass
                fehler.append(f"{ziel}{frei}: {type(e).__name__}: {str(e)[-160:]}")
        if m is None:
            raise RuntimeError("Whisper nicht ladbar – " + " | ".join(fehler))
    m = m.to(dev)
    return pipeline("automatic-speech-recognition", model=m, tokenizer=p.tokenizer, feature_extractor=p.feature_extractor,
                    torch_dtype=dt, device=dev, chunk_length_s=30, batch_size=8)


# Typische von Whisper "erfundene" Sätze (aus YouTube-Untertiteln gelernt) – nie Songtext
ERFUNDEN = ("we'll be right back", "see you next time", "thanks for watching", "thank you for watching",
            "subscribe", "like and subscribe", "please subscribe", "thank you.", "bye.", "you.")


def phrasen(y, rahmen=0.05, abstand_db=30.0, luecke=0.6, min_s=0.4, max_s=12.0):
    """31.10.: Gesangs-Phrasen nach Lautstärke finden (Pausen = neue Zeile). Lange Phrasen an der leisesten
    Stelle teilen. Ergebnis: Liste (start_s, ende_s)."""
    h = int(rahmen * SR16)
    n = len(y) // h
    if n < 2:
        return []
    e = 20 * np.log10(np.sqrt((y[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    an = e > e.max() - abstand_db
    stuecke, i = [], 0
    while i < n:
        if an[i]:
            j = i
            while j < n and an[j]:
                j += 1
            stuecke.append([i, j])
            i = j
        else:
            i += 1
    zus = []
    for s in stuecke:                                   # kurze Pausen überbrücken
        if zus and (s[0] - zus[-1][1]) * rahmen < luecke:
            zus[-1][1] = s[1]
        else:
            zus.append(s)
    fertig = []

    def teilen(a, b):
        if (b - a) * rahmen <= max_s:
            fertig.append((a, b))
            return
        lo, hi = a + int(3 / rahmen), b - int(3 / rahmen)
        m = lo + int(np.argmin(e[lo:hi])) if hi > lo else (a + b) // 2
        teilen(a, m)
        teilen(m, b)
    for a, b in zus:
        if (b - a) * rahmen >= min_s:
            teilen(a, b)
    return [(max(0.0, a * rahmen - 0.15), min(len(y) / SR16, b * rahmen + 0.15)) for a, b in fertig]


def mitschreiben(asr, y):
    if len(y) < SR16 or float(np.abs(y).max()) < 1e-4:
        return []
    y = (y / max(float(np.abs(y).max()), 1e-6) * 0.9).astype(np.float32)
    # 31.10.: Erster Lauf gab ab Zeile 8 alles in EINEM Block (Zeitstempel verloren) + erfundene Sätze.
    # Jetzt: jede Gesangs-Phrase EINZELN erkennen -> eine Zeile je Phrase, Stille wird gar nicht erst gehört.
    ph = phrasen(y)
    if not ph:
        return []
    eingaben = [{"raw": y[int(a * SR16):int(b * SR16)], "sampling_rate": SR16} for a, b in ph]
    res = asr(eingaben, generate_kwargs={"task": "transcribe"})
    out, letzte = [], ""
    for (a, b), r in zip(ph, res):
        txt = (r.get("text") or "").strip()
        klein = txt.lower().strip(" .!?,")
        if not txt or any(klein == f.strip(" .") or klein.startswith(f.strip(" .")) for f in ERFUNDEN):
            continue
        if len(txt.split()) > 6 * (b - a) + 3:          # unmöglich viele Wörter -> Wiederholungsschleife
            continue
        if txt == letzte:
            continue
        letzte = txt
        out.append({"start": round(a, 2), "end": round(b, 2), "text": txt})
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
