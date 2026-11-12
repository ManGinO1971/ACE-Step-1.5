#!/usr/bin/env python3
"""
HELTECH Stems – PILOT „Schlagzeug aus der Karte neu aufbauen“ (3. Nov 2026, ohne GPU).

Idee: Die getrennten Teile (Kick, Snare, Hi-Hat) liefern nur noch die KARTE (wann + wie stark geschlagen wird).
Der Klang kommt komplett aus sauberen Einzelschlägen (Reggae-Shots) -> keine Trenn-Artefakte, kein Übersprechen.

Schritte je Teil:
  1. Schläge finden (Onset) + Stärke (Spitzenpegel der ersten 50 ms) + Klangabdruck (Mel-Spektrum der ersten 120 ms)
  2. Übersprechen aussortieren (Schlag deutlich leiser als die echten Schläge dieses Teils)
  3. Schlag-Arten trennen: Snare -> Rimshot / volle Snare, Hi-Hat -> zu / offen (nach Klangabdruck bzw. Ausklang)
  4. Je Schlag-Art die 3 ähnlichsten Shots aus der Bibliothek wählen (immer dieselben 3 -> ein Set, abwechselnd -> nicht maschinell)
  5. Shot genau auf die Spitze des Original-Schlags legen, Lautstärke = Original-Lautstärke
  6. Hi-Hat: neuer Schlag dämpft den vorigen (wie ein echtes Hi-Hat)
  7. Gegenprobe: Schläge im Ergebnis wieder suchen und mit der Karte vergleichen (Treffer, Zeitabweichung, Lautstärke)
Aufruf: python drum_ersatz.py <ordner_mit_kick_snare_hihat_becken_toms.flac> <shots_ordner> <ausgabe_ordner>
"""
import glob
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf

SR = 44100
HOP = 256


def lade(p, sr=SR):
    x, s = sf.read(p, always_2d=True, dtype="float32")
    if s != sr:
        x = np.stack([librosa.resample(x[:, c], orig_sr=s, target_sr=sr) for c in range(x.shape[1])], 1)
    return x


def abdruck(m, sr=SR, dauer=0.12):
    seg = m[: int(dauer * sr)]
    if len(seg) < int(dauer * sr):
        seg = np.pad(seg, (0, int(dauer * sr) - len(seg)))
    S = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=2048, hop_length=512, n_mels=40, fmax=16000)
    v = np.log(S.mean(1) + 1e-9)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def ausklang(m, sr=SR):
    """Zeit vom Spitzenwert bis -30 dB (Sekunden)."""
    h = int(0.005 * sr)
    n = len(m) // h
    if n < 2:
        return 0.0
    e = 20 * np.log10(np.sqrt((m[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    p = int(e.argmax())
    unter = np.where(e[p:] < e[p] - 30)[0]
    return (unter[0] if len(unter) else n - p) * 0.005


def nachschlag(m, sr=SR, bis_s=1.5, sprung_db=8.0, boden_db=None):
    """True, wenn nach dem Hauptschlag noch ein zweiter Anschlag kommt (Hüllkurve steigt wieder deutlich an).
    Für lang ausklingende Teile (Toms, Becken), bei denen die normale Onset-Suche im Ausklang Fehlalarm gibt."""
    h = int(0.01 * sr)
    n = min(len(m), int(bis_s * sr)) // h
    if n < 3:
        return False
    e = 20 * np.log10(np.sqrt((m[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    p = int(e.argmax())
    if boden_db is not None:                      # Rauschen im leisen Ausklang nicht als Nachschlag werten
        e = np.maximum(e, e[p] - boden_db)
    tief = np.minimum.accumulate(e[p:])
    return bool(np.any(e[p:] - tief > sprung_db))


def bibliothek(muster, pruefung="onsets", max_spitze_s=0.06, boden_db=None):
    out = []
    for f in sorted(glob.glob(muster)):
        try:
            x = lade(f)
        except Exception:
            continue
        m = x.mean(1)
        a = np.abs(m)
        if a.max() < 1e-4:
            continue
        start = int(np.argmax(a > a.max() * 0.03))           # Anfang (-30 dB)
        start = max(0, start - int(0.002 * SR))
        x = x[start:]
        m = x.mean(1)
        e = np.abs(m)
        hinten = np.where(e > e.max() * 10 ** (-70 / 20))[0]
        x = x[: hinten[-1] + 1] if len(hinten) else x
        fl = min(len(x), int(0.01 * SR))
        x[-fl:] *= np.linspace(1, 0, fl)[:, None]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, 1)
        roh_pegel = float(20 * np.log10(np.abs(x).max() + 1e-12))   # Original-Lautstärke (für Stärke-Stufen)
        x = x[:, :2] / np.abs(x).max()
        m = x.mean(1)
        # nur saubere Einzelschläge: genau ein Anschlag, und der liegt am Anfang
        if int(np.abs(m).argmax()) > int(max_spitze_s * SR):
            continue
        if pruefung == "onsets":
            ons = librosa.onset.onset_detect(y=m, sr=SR, hop_length=HOP, units="time", backtrack=True)
            if len(ons) != 1 or ons[0] > 0.04:
                continue
        elif nachschlag(m, boden_db=boden_db):
            continue
        out.append({"datei": os.path.basename(f), "x": x.astype(np.float32), "abdruck": abdruck(m),
                    "spitze": int(np.abs(m).argmax()), "ausklang": ausklang(m),
                    "roh_pegel": roh_pegel, "pfad": f})
    return out


def schlaege(x, teil):
    m = x.mean(1)
    env = librosa.onset.onset_strength(y=m, sr=SR, hop_length=HOP)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=True, units="samples",
                                    delta=0.05, wait=int(0.06 * SR / HOP))
    hits = []
    for i, o in enumerate(on):
        ende = on[i + 1] if i + 1 < len(on) else len(m)
        fenster = m[o: min(o + int(0.05 * SR), ende)]
        if len(fenster) < 16:
            continue
        p = o + int(np.abs(fenster).argmax())
        hits.append({"start": int(o), "spitze": int(p), "pegel": float(20 * np.log10(np.abs(m[p]) + 1e-9)),
                     "abdruck": abdruck(m[o:]), "ausklang": ausklang(m[o:min(ende, o + int(1.0 * SR))])})
    if not hits:
        return []
    ref = np.percentile([h["pegel"] for h in hits], 90)
    grenze = {"kick": 18, "snare": 20, "hihat": 24}[teil]
    return [h for h in hits if h["pegel"] > ref - grenze]   # leiseres = Übersprechen


def zwei_gruppen(werte):
    """einfaches 2-Gruppen-Verfahren (k-means 1D/ND), gibt Labels 0/1 zurück"""
    w = np.asarray(werte, float)
    if w.ndim == 1:
        w = w[:, None]
    a, b = w[w[:, 0].argmin()], w[w[:, 0].argmax()]
    for _ in range(20):
        lab = (np.linalg.norm(w - b, axis=1) < np.linalg.norm(w - a, axis=1)).astype(int)
        if lab.min() == lab.max():
            break
        a, b = w[lab == 0].mean(0), w[lab == 1].mean(0)
    return lab


def waehle_set(hits, lib, n=3):
    A = np.stack([h["abdruck"] for h in hits])
    B = np.stack([s["abdruck"] for s in lib])
    aehnl = (A @ B.T).mean(0)
    best = np.argsort(-aehnl)[:n]
    return [lib[i] for i in best], [float(aehnl[i]) for i in best]


def rendern(laenge, hits, sets, choke=False):
    y = np.zeros((laenge, 2), np.float32)
    zaehler = {}
    for h in sorted(hits, key=lambda h: h["spitze"]):
        s_list = sets[h["art"]]
        k = zaehler.get(h["art"], 0)
        zaehler[h["art"]] = k + 1
        s = s_list[k % len(s_list)]
        g = 10 ** (h["pegel"] / 20) / max(np.abs(s["x"].mean(1)[s["spitze"]]), 1e-6)
        a = h["spitze"] - s["spitze"]
        if choke:
            # vorigen Hi-Hat-Klang ab hier in 8 ms abdämpfen
            f = int(0.008 * SR)
            if 0 < a < laenge:
                ende = min(laenge, a + f)
                y[a:ende] *= np.linspace(1, 0, ende - a)[:, None]
                y[ende:] = 0.0
        q0 = max(0, -a)
        a0 = max(0, a)
        n = min(len(s["x"]) - q0, laenge - a0)
        if n > 0:
            y[a0:a0 + n] += g * s["x"][q0:q0 + n]
    return y


def gegenprobe(neu, hits, teil):
    gef = schlaege(neu, teil)
    t_neu = np.array([h["spitze"] for h in gef])
    if len(t_neu) == 0:
        return {"treffer_%": 0}
    ab, pd = [], []
    for h in hits:
        j = int(np.abs(t_neu - h["spitze"]).argmin())
        d = abs(int(t_neu[j]) - h["spitze"]) / SR * 1000
        if d <= 20:
            ab.append(d)
            pd.append(gef[j]["pegel"] - h["pegel"])
    return {"schlaege_karte": len(hits), "schlaege_neu_gefunden": len(gef),
            "treffer_%": round(100 * len(ab) / max(1, len(hits)), 1),
            "zeit_abweichung_ms_mittel": round(float(np.mean(ab)), 2) if ab else None,
            "pegel_abweichung_db_mittel": round(float(np.mean(np.abs(pd))), 2) if pd else None}


def main():
    quelle, shots, ziel = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(ziel, exist_ok=True)
    libs = {
        "kick": bibliothek(os.path.join(shots, "Drums Shot/Kick/*")),
        "rimshot": bibliothek(os.path.join(shots, "Drums Shot/Rimshot/*")),
        "snare": bibliothek(os.path.join(shots, "Drums Shot/Snare/*")),
        "hh_zu": [s for s in bibliothek(os.path.join(shots, "Drumkit/Hi Hat/*Closed*"))],
        "hh_offen": [s for s in bibliothek(os.path.join(shots, "Drumkit/Hi Hat/*Open*"))],
    }
    bericht = {"bibliothek": {k: len(v) for k, v in libs.items()}}
    teile = {}
    for teil in ("kick", "snare", "hihat"):
        x = lade(os.path.join(quelle, f"{teil}.flac"))
        laenge = len(x)
        hits = schlaege(x, teil)
        sets, wahl = {}, {}
        if teil == "kick":
            for h in hits:
                h["art"] = "kick"
            sets["kick"], sim = waehle_set(hits, libs["kick"])
            wahl["kick"] = list(zip([s["datei"] for s in sets["kick"]], sim))
        elif teil == "snare":
            # Rimshot oder volle Snare? -> je Schlag: welcher Bibliothek ist er ähnlicher
            R = np.stack([s["abdruck"] for s in libs["rimshot"]])
            S = np.stack([s["abdruck"] for s in libs["snare"]])
            for h in hits:
                h["art"] = "rimshot" if (R @ h["abdruck"]).max() >= (S @ h["abdruck"]).max() else "snare"
            for art in ("rimshot", "snare"):
                hh = [h for h in hits if h["art"] == art]
                if hh:
                    sets[art], sim = waehle_set(hh, libs[art])
                    wahl[art] = list(zip([s["datei"] for s in sets[art]], sim))
        else:
            lab = zwei_gruppen([h["ausklang"] for h in hits])
            lang = np.mean([h["ausklang"] for h, l in zip(hits, lab) if l == 1]) if lab.max() else 0
            kurz = np.mean([h["ausklang"] for h, l in zip(hits, lab) if l == 0])
            offen_ok = lab.max() == 1 and lang > 0.18 and lang > 2 * kurz
            for h, l in zip(hits, lab):
                h["art"] = "hh_offen" if (offen_ok and l == 1) else "hh_zu"
            for art in ("hh_zu", "hh_offen"):
                hh = [h for h in hits if h["art"] == art]
                if hh:
                    sets[art], sim = waehle_set(hh, libs[art])
                    wahl[art] = list(zip([s["datei"] for s in sets[art]], sim))
        y = rendern(laenge, hits, sets, choke=(teil == "hihat"))
        sf.write(os.path.join(ziel, f"{teil}_neu.wav"), y, SR, subtype="PCM_24")
        teile[teil] = y
        arten = {}
        for h in hits:
            arten[h["art"]] = arten.get(h["art"], 0) + 1
        bericht[teil] = {"schlaege": len(hits), "arten": arten,
                         "gewaehlte_shots(aehnlichkeit)": {k: [(d, round(s, 3)) for d, s in v] for k, v in wahl.items()},
                         "gegenprobe": gegenprobe(y, hits, teil)}
        print(f"✔ {teil}: {len(hits)} Schläge {arten}", flush=True)
    # komplett = neue Kick/Snare/Hi-Hat + Original-Becken + Original-Toms
    rest = [lade(os.path.join(quelle, f"{t}.flac")) for t in ("becken", "toms")]
    n = min([len(v) for v in teile.values()] + [len(r) for r in rest])
    mix = sum(v[:n] for v in teile.values()) + sum(r[:n, :2] for r in rest)
    mix *= min(1.0, 0.98 / max(float(np.abs(mix).max()), 1e-6))
    sf.write(os.path.join(ziel, "drums_neu_komplett.wav"), mix, SR, subtype="PCM_24")
    with open(os.path.join(ziel, "bericht.json"), "w") as f:
        json.dump(bericht, f, indent=1, ensure_ascii=False)
    print(json.dumps(bericht, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
