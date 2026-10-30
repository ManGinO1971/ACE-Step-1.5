#!/usr/bin/env python3
"""HIELTECH Stems – Djembe/Percussion und Schlagzeug Schlag für Schlag sortieren (30. Okt 2026).
Hörtest: im Intro ist alles Djembe außer 3 Tom-Schlägen (s 4 zwei schnelle, s 7 einer). Die 53er-Schlagzeug-
Modelle (snare/toms) haben im Intro aber auch die DJEMBE mitgenommen (snare.flac dort laut) – deshalb reicht
"Schlagzeug-Vorlage stärker -> raus" nicht. Neue Regel je SCHLAG, für alle Songs:
  - Djembe/Percussion-Vorlage (congas, percussion) und Schlagzeug-Vorlage (kick, snare, toms) im Anschlag (60 ms) vergleichen.
  - Die 53er-Modelle verwechseln Djembe-Töne (150–360 Hz) mit Snare/Toms (Intro: s 4,2–5,3 Djembe nur in snare/toms).
  - Schlagzeug-Schlag NUR wenn: Schlagzeug-Vorlage >= 30 dB lauter als Djembe-Vorlage UND laut (> -30 dB)
    UND tiefer Grundton (< 130 Hz, unter dem Djembe-Ton) -> raus aus Percussion, rein in Toms (>= 75 Hz) bzw. Kick.
  - Alles andere bleibt Djembe.
Ergebnis: Percussion nur Djembe; die herausgenommenen Schläge als eigene Spuren (für Schlagzeug/teile).
Aufruf: python perkussion_sortieren.py <percussion_unterlage> <getrennt_ordner> <ref53_ordner> <ausgabe>"""
import os, sys, json
import numpy as np, soundfile as sf, librosa

SR = 44100
ABSTAND_DB = float(os.environ.get("HIELTECH_SORT_ABSTAND_DB", "30"))
TIEF_HZ = float(os.environ.get("HIELTECH_SORT_TIEF_HZ", "130"))   # Djembe-Ton/Slap liegt darüber, Toms/Kick darunter

def lade(p, n):
    a, _ = sf.read(p, always_2d=True, dtype="float32")
    return a[:n] if len(a) >= n else np.pad(a, ((0, n - len(a)), (0, 0)))

def pegel(m, a, b):
    seg = m[a:b]
    return 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9) if len(seg) else -200.0

def grundton(m, a):
    seg = m[a:a + int(0.15 * SR)]
    if len(seg) < 512:
        return 0.0
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 16)); f = np.fft.rfftfreq(1 << 16, 1 / SR)
    sel = (f > 35) & (f < 400)
    return float(f[sel][np.argmax(S[sel])])

def sortieren(x, refs):
    """x: (n,2) Percussion-Unterlage. refs: dict name->(n,2). -> (percussion, {ziel: spur}, liste)"""
    n = len(x)
    mono = {k: v.mean(1) for k, v in refs.items()}
    summe = sum(mono.values())
    on = librosa.onset.onset_detect(y=summe, sr=SR, units="samples", backtrack=True, delta=0.05, hop_length=256)
    on = list(on) + [n]
    gp = np.ones(n); ziele = {"toms": np.zeros(n), "kick": np.zeros(n)}
    liste = []
    f = int(0.004 * SR)
    for i in range(len(on) - 1):
        a, b = int(on[i]), int(min(on[i + 1], on[i] + int(0.5 * SR)))
        if b - a < int(0.03 * SR):
            continue
        w = a + int(0.06 * SR)
        dj = max(pegel(mono["congas"], a, w), pegel(mono["percussion"], a, w))
        kit = {k: pegel(mono[k], a, w) for k in ("kick", "toms", "snare")}
        kmax = max(kit.values())
        g0 = grundton(mono["kick"] + mono["toms"], a)
        if kmax - dj >= ABSTAND_DB and kmax > -30 and 0 < g0 < TIEF_HZ:
            ziel = "toms" if g0 >= 75 else "kick"
            # weich: 4 ms Ein-/Ausblende
            r = np.ones(b - a); r[:f] = np.linspace(0, 1, f); r[-f:] = np.linspace(1, 0, f)
            gp[a:b] -= r; ziele[ziel][a:b] += r
            liste.append({"zeit_s": round(a / SR, 2), "ziel": ziel, "grundton_hz": round(g0), "abstand_db": round(float(kmax - dj), 1)})
    gp = np.clip(gp, 0, 1)
    perc = x * gp[:, None]
    spuren = {k: x * np.clip(v, 0, 1)[:, None] for k, v in ziele.items()}
    return perc, spuren, liste

if __name__ == "__main__":
    unter, G, R, outdir = sys.argv[1:5]
    os.makedirs(outdir, exist_ok=True)
    x, _ = sf.read(unter, always_2d=True, dtype="float32"); n = len(x)
    refs = {"congas": lade(f"{G}/congas.flac", n), "percussion": lade(f"{G}/percussion.flac", n),
            "kick": lade(f"{R}/kick.flac", n), "toms": lade(f"{R}/toms.flac", n), "snare": lade(f"{R}/snare.flac", n)}
    perc, spuren, liste = sortieren(x, refs)
    sf.write(os.path.join(outdir, "7_percussion_nur_djembe.wav"), perc, SR, subtype="PCM_24")
    for k, v in spuren.items():
        sf.write(os.path.join(outdir, f"zu_schlagzeug_{k}.wav"), v, SR, subtype="PCM_24")
    json.dump(liste, open(os.path.join(outdir, "sortiert_liste.json"), "w"), ensure_ascii=False, indent=1)
    print("Schlagzeug-Schläge aus der Percussion genommen:", len(liste))
    for e in liste[:40]:
        print("  ", e)
