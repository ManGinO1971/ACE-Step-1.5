#!/usr/bin/env python3
"""HELTECH Stems – Lead-Maske für JEDEN Song (11. Nov 2026).

Hintergrund (Hörtest "Miss Yuh So Much, Mama", Patois): beide neu gesungenen Lead-Varianten sangen
vollständig (0 s Lücken), trotzdem kam nur ~45 % des Leads an. Ursache: die alte Kette (lead_mask_step.py)
schaltete hart stumm, wo
  (a) die Text-Erkennung (wav2vec2) keine Zeile gefunden hat – bei Patois/Darija/Akzenten findet sie wenig,
  (b) die Karaoke-Trennung den Gesang dem "Background" zugeschlagen hat (z. B. bei {…} Lead + Chor zusammen),
und nahm EINE Variante für den ganzen Song.

Neue Regeln (keine Werte für einen bestimmten Song, alles relativ pro Song gemessen):
  1. SOLL = wo im Original der Lead singen muss. Grundlage ist die Original-Lead-Spur. Enthält der Text
     (fast) keine reinen Chor-Stellen "(…)", singt der Lead überall, wo im Original überhaupt gesungen
     wird -> dann zählt die ganze Original-Gesangsspur (Lead + Background), egal wie die Trennung sie aufteilt.
  2. Text-Maske und Chor-Vorrang-Maske sind nur noch HILFEN: sie werden nur angewendet, wenn sie
     zuverlässig sind (Text: genug Zeilen gefunden) und höchstens einen kleinen Teil des SOLL wegnehmen.
     Sonst werden sie für diesen Song übersprungen (steht im Bericht).
  3. Abschnittsweise beste Variante: für jede gesungene Phrase die Variante, die dort wirklich singt.
  4. Abdeckung messen: Anteil des SOLL, in dem der neue Lead hörbar ist -> Bericht/Warnung.

Aufruf: python stems_lead_maske.py <aus.wav> <lead_ref.wav> <backing_ref.wav|-> <lyrics.txt|-> <segmente.json|-> <variante1> [variante2 ...]
  segmente.json: {"0": [[t0,t1,conf],...], "1": [...]} (Text-Abgleich je Variante), darf fehlen.
Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import os
import re
import sys

import librosa
import numpy as np
import soundfile as sf

E = lambda k, d: float(os.environ.get(k, d))
HOP_S = 0.01


def lies(p, sr=None):
    y, s = sf.read(p, always_2d=True)
    if sr and s != sr:
        y = np.stack([librosa.resample(y[:, c], orig_sr=s, target_sr=sr) for c in range(y.shape[1])], 1)
        s = sr
    return y, s


def huelle_db(mono, sr):
    hop = int(sr * HOP_S)
    e = librosa.feature.rms(y=mono, frame_length=hop * 4, hop_length=hop)[0]
    return 20 * np.log10(e + 1e-9)


def aktiv(db, unter_spitze):
    """aktiv relativ zum eigenen Pegel dieses Songs (95.-Perzentil der lauten Stellen)."""
    if not np.isfinite(db).any() or db.max() < -100:
        return np.zeros(len(db), bool)
    pegel = np.percentile(db[db > db.max() - 60], 95)
    return db > pegel - unter_spitze


def mini_weg(m, frames):
    m = m.copy()
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i < frames:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m


def halten(m, vor, nach):
    g = m.copy()
    for t in np.flatnonzero(m):
        g[max(0, t - vor):t + nach + 1] = True
    return g


def bereiche(m):
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def versatz(ref_db, neu_db):
    n = min(len(ref_db), len(neu_db))
    a, b = ref_db[:n] - ref_db[:n].mean(), neu_db[:n] - neu_db[:n].mean()
    return max(range(-50, 51), key=lambda L: float(np.dot(a[max(0, L):n + min(0, L)], b[max(0, -L):n - max(0, L)])))


def chor_anteil(text):
    """Anteil reiner Chor-Stellen "(…)" am gesungenen Text (Zeichen). {…} zählt als Lead (Lead + Chor)."""
    if not text:
        return None
    zeilen = [z for z in text.splitlines() if z.strip() and not re.match(r"^\s*\[.*\]\s*$", z)]
    gesamt = "".join(zeilen)
    nur_chor = "".join(re.findall(r"\(([^)]*)\)", gesamt))
    ohne = re.sub(r"[\s\W_]+", "", gesamt, flags=re.UNICODE)
    return round(len(re.sub(r"[\s\W_]+", "", nur_chor, flags=re.UNICODE)) / max(1, len(ohne)), 3)


def lead_zeilen(text):
    n = 0
    for z in (text or "").splitlines():
        z = re.sub(r"\([^)]*\)", "", z).strip()
        if z and not re.match(r"^\[.*\]$", z):
            n += 1
    return n


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        return 1
    aus, ref_p, back_p, text_p, seg_p = sys.argv[1:6]
    var_p = sys.argv[6:]
    bericht = {"ok": False, "regeln": [], "warnungen": []}

    varianten, sr = [], None
    for p in var_p:
        y, s = lies(p, sr)
        sr = sr or s
        varianten.append(y)
    n_smp = min(len(v) for v in varianten)
    varianten = [v[:n_smp] for v in varianten]
    kanaele = max(v.shape[1] for v in varianten)
    varianten = [np.repeat(v, kanaele, 1) if v.shape[1] == 1 else v for v in varianten]

    ref, _ = lies(ref_p, sr)
    ref_db = huelle_db(ref.mean(1), sr)
    voll = ref.mean(1)
    back_db = None
    if back_p != "-" and os.path.isfile(back_p):
        bk, _ = lies(back_p, sr)
        m = min(len(bk), len(voll))
        voll = voll[:m] + bk[:m].mean(1)
        back_db = huelle_db(bk.mean(1), sr)
    voll_db = huelle_db(voll, sr)

    var_db = [huelle_db(v.mean(1), sr) for v in varianten]
    nf = min([len(ref_db), len(voll_db)] + [len(d) for d in var_db] + ([len(back_db)] if back_db is not None else []))
    lag = versatz(ref_db[:nf], var_db[0][:nf])
    ref_db, voll_db = np.roll(ref_db[:nf], lag), np.roll(voll_db[:nf], lag)
    if back_db is not None:
        back_db = np.roll(back_db[:nf], lag)
    var_db = [d[:nf] for d in var_db]
    bericht["versatz_ms"] = lag * 10

    text = open(text_p, encoding="utf-8").read() if text_p != "-" and os.path.isfile(text_p) else ""
    anteil = chor_anteil(text)
    bericht["chor_anteil_text"] = anteil
    schwelle_db = E("LEAD_SOLL_DB", 30)

    # 1) SOLL
    soll_lead = aktiv(ref_db, schwelle_db)
    soll_voll = aktiv(voll_db, schwelle_db)
    if anteil is not None and anteil <= E("LEAD_CHOR_ANTEIL_MAX", 0.15):
        soll = soll_lead | soll_voll
        bericht["regeln"].append("soll=ganzer_original_gesang (kaum reine Chor-Stellen im Text)")
    else:
        soll = soll_lead.copy()
        bericht["regeln"].append("soll=original_lead")
        # Chor-Vorrang: wo der Background das Original klar übertönt -> zu (nur wenn es wenig wegnimmt)
        if back_db is not None:
            weg = soll & ~(ref_db > back_db + E("MASK_VS_BACKING_DB", -10))
            if weg.sum() <= E("LEAD_MASKE_MAX_WEG", 0.2) * max(1, soll.sum()):
                soll &= ~weg
                bericht["regeln"].append(f"chor_vorrang ({weg.sum() / 100:.1f} s)")
            else:
                bericht["warnungen"].append(f"Chor-Vorrang übersprungen (hätte {weg.sum() / max(1, soll.sum()) * 100:.0f} % des Leads stummgeschaltet)")
    soll = mini_weg(soll, int(E("MASK_MIN_MS", 90) / 10))
    soll_roh = soll.copy()

    # 2) Text-Maske nur, wenn zuverlässig
    segmente = {}
    if seg_p != "-" and os.path.isfile(seg_p):
        try:
            segmente = {int(k): v for k, v in json.load(open(seg_p)).items()}
        except Exception:  # noqa: BLE001
            segmente = {}
    n_zeilen = max(1, lead_zeilen(text))
    gefunden = max([len([s for s in v if s[2] >= E("MASK_KEEP_MIN_CONF", 0.5)]) for v in segmente.values()] or [0])
    bericht["text_zeilen"] = {"lead": n_zeilen, "gefunden_max": gefunden}
    if segmente and gefunden >= E("LEAD_TEXT_MIN_ANTEIL", 0.6) * n_zeilen:
        keep = np.zeros(nf, bool)
        pad = E("MASK_KEEP_PAD_S", 1.0)
        for v in segmente.values():
            for t0, t1, c in v:
                if c >= E("MASK_KEEP_MIN_CONF", 0.5):
                    keep[max(0, int((t0 - pad) * 100)):int((t1 + pad) * 100) + 1] = True
        weg = soll & ~keep
        if weg.sum() <= E("LEAD_MASKE_MAX_WEG", 0.2) * max(1, soll.sum()):
            soll &= keep
            bericht["regeln"].append(f"text_maske ({weg.sum() / 100:.1f} s)")
        else:
            bericht["warnungen"].append(f"Text-Maske übersprungen (hätte {weg.sum() / max(1, soll.sum()) * 100:.0f} % des Leads stummgeschaltet)")
    elif segmente:
        bericht["warnungen"].append(f"Text-Maske übersprungen (Text-Erkennung fand nur {gefunden} von {n_zeilen} Zeilen – z. B. Patois/Darija)")

    tor = halten(soll, int(E("MASK_LOOK_MS", 40) / 10), int(E("MASK_HOLD_MS", 300) / 10))

    # 3) je Phrase die Variante, die dort singt
    var_akt = [aktiv(d, E("LEAD_VAR_DB", 35)) for d in var_db]
    wahl = np.zeros(nf, int)
    zaehl = [0] * len(varianten)
    for a, b in bereiche(tor):
        treffer = [int((va[a:b] & soll_roh[a:b]).sum()) for va in var_akt]
        k = int(np.argmax(treffer)) if max(treffer) > 0 else 0
        wahl[a:b] = k
        zaehl[k] += 1
    bericht["phrasen_je_variante"] = zaehl

    # weiche Rampen für Tor und Wechsel
    def glatt(g, att_ms=25, rel_ms=200):
        g = g.astype(float)
        sm = g.copy()
        att, rel = max(1, int(att_ms / 10)), max(1, int(rel_ms / 10))
        for t in range(1, len(sm)):
            tgt = g[t]
            st = 1.0 / (att if tgt > sm[t - 1] else rel)
            sm[t] = min(tgt, sm[t - 1] + st) if tgt > sm[t - 1] else max(tgt, sm[t - 1] - st)
        return 0.5 - 0.5 * np.cos(np.pi * sm)

    hop = int(sr * HOP_S)
    x = np.arange(n_smp)
    t_frames = np.arange(nf) * hop
    tor_s = np.interp(x, t_frames, glatt(tor))
    mix = np.zeros((n_smp, kanaele))
    for k, v in enumerate(varianten):
        w = np.interp(x, t_frames, glatt(wahl == k, 30, 30))
        mix += v * w[:, None]
    mix *= tor_s[:, None]
    sf.write(aus, mix, sr, subtype="PCM_24")

    # 4) Abdeckung
    neu_db = huelle_db(mix.mean(1), sr)[:nf]
    neu_akt = aktiv(neu_db, E("LEAD_VAR_DB", 35))
    soll_n = max(1, int(soll_roh.sum()))
    abdeckung = float((neu_akt & soll_roh).sum()) / soll_n
    bericht.update({"ok": True, "abdeckung": round(abdeckung, 3), "soll_sekunden": round(soll_n / 100, 1),
                    "hoerbar_sekunden": round(float(neu_akt.sum()) / 100, 1)})
    if abdeckung < E("LEAD_ABDECKUNG_WARNUNG", 0.85):
        bericht["warnungen"].append(f"Lead deckt nur {abdeckung * 100:.0f} % der Original-Gesangsstellen ab")
    print(json.dumps(bericht, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
