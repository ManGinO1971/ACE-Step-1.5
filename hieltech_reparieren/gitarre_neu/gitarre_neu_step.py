#!/usr/bin/env python3
"""
HELTECH Stems – Rhythmus-Gitarre (Skank, „tchak“) neu einspielen. Stand 5. Nov 2026,
getestet mit „Rise Up Jah Children“ (Gitarre v3).

Eingabe: der GANZE Song (Piano, Bass und Schlagzeug werden gebraucht). Ablauf:
  1. Trennung in 6 Spuren (BS-Roformer-SW; Rückfall: htdemucs_6s)
  2. Taktraster aus dem Schlagzeug, Tonart aus Harmonie und Bass
  3. Akkord je Schlag aus allen Instrumenten (nur Akkorde der Tonart, Grundton aus dem Bass, geglättet),
     Stellen ohne Akkorde, Takt-Eins aus den Akkordwechseln
  4. Skank-Stellen: Offbeats in den Abschnitten, in denen die Gitarre spielt (auch wo die Trennung schwach
     ist), einzelne starke Anschläge außerhalb. Rhythmus-Schlag wählbar (--muster): automatisch wie im
     Original (Doppelschläge/Vorschläge/Kratzer nur dort, wo sie im Original sind), einfach „tchak“,
     doppelt „tchak-chack“, Vorschlag „chack-tchak“, mit Scratch (gedämpfte Kratzer), Shuffle (geswingt)
  5. Timing: echter Anschlag der Gitarre; mit --timing-korrigieren genau auf den Schlagzeug-Schlag
  6. Neu spielen mit den eigenen Akkord-Aufnahmen (samples/), feste Griffposition je Akkord, gleichmäßige
     Lautstärke, „C fett“-Ton (EQ wie klassische Reggae-Skanks, Kompression, kurzer Raum, Stereo-Doppelung)
  7. Ausgabe: gitarre_neu.wav, song_mit_neuer_gitarre.wav, gitarre_neu.mid, Akkord-Plan, bericht.json

Aufruf:
  python gitarre_neu_step.py <song.wav> <ausgabe_ordner> [--timing-korrigieren] [--midi-bpm 120]
        [--lautstaerke-db 0] [--modelle ORDNER] [--muster auto|einfach|doppelt|vorschlag|scratch|shuffle]
Ausgabe auf stdout (letzte Zeile): JSON {"ok": true, "datei": ..., ...}
"""
import argparse
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys

import numpy as np
import soundfile as sf

HIER = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HIER, "samples")
SR = 44100
NAMEN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUAL = {"major": (0, 4, 7), "minor": (0, 3, 7), "major7": (0, 4, 7, 11), "minor7": (0, 3, 7, 10)}
KURZ = {"major": "", "minor": "m", "major7": "maj7", "minor7": "m7"}
STIMMEN = ("drums", "bass", "guitar", "piano", "other", "vocals")
# Rhythmus-Schläge: Lage im Schlag (0 = Schlag, 0,5 = Offbeat „und“) und Art des Anschlags
#   haupt = voller Tchak (Abschlag), nach = leichterer Aufschlag, ghost = gedämpfter Kratzer („chk“)
MUSTER = {
    "einfach": [(0.5, "haupt")],                                   # tchak (Roots / One Drop)
    "doppelt": [(0.5, "haupt"), (0.75, "nach")],                   # tchak-chack (Rockers / Double Skank)
    "vorschlag": [(0.25, "nach"), (0.5, "haupt")],                 # chack-tchak
    "scratch": [(0.25, "ghost"), (0.5, "haupt"), (0.75, "ghost")],  # Steppers / Dancehall mit Kratzern
    "shuffle": [(2 / 3, "haupt")],                                  # geswingt (Rocksteady / Early Reggae)
}
MUSTER_NAMEN = {"auto": "automatisch wie im Original", "einfach": "einfach – tchak", "doppelt": "doppelt – tchak-chack",
                "vorschlag": "Vorschlag – chack-tchak", "scratch": "mit Scratch – chk-tchak-chk", "shuffle": "Shuffle (geswingt)"}


def melde(t):
    print(t, flush=True)

def _gpu():
    """GPU-Schalter der App (gpu_rechner.py): Modul zurück, wenn „GPU“ gewählt ist, sonst None."""
    try:
        sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
        import gpu_rechner as G
        return G if G.gpu_an() else None
    except Exception:  # noqa: BLE001
        return None


def _speicher():
    """Speicher der App für schon gerechnete Trennungen (gpu_rechner.py) – unabhängig vom Schalter GPU/CPU."""
    try:
        sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
        import gpu_rechner as G
        return G
    except Exception:  # noqa: BLE001
        return None


try:
    sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
    from merker import merke
except Exception:  # noqa: BLE001 – ohne Merker läuft alles wie bisher
    def merke(*_a, **_k):
        return lambda fn: fn



def librosa():
    import librosa as _l
    return _l


# ------------------------------------------------------------ 1. Trennung
def trennen(eingabe, ziel, modelle):
    roh = os.path.join(ziel, "spuren_roh")
    os.makedirs(roh, exist_ok=True)
    G = _gpu()
    if G:
        try:
            return G.sechs_spuren(eingabe)
        except Exception as e:  # noqa: BLE001
            melde(f"… GPU nicht verfügbar ({str(e)[:150]}) – rechne auf dem Mac")
    Sp = _speicher()
    if Sp:
        for auftrag, name in ((Sp.A_SECHS, "BS-Roformer-SW"), (Sp.a_demucs("htdemucs_6s"), "htdemucs_6s")):
            da = Sp.gespeichert(eingabe, auftrag)
            if da and all(s in da for s in ("drums", "bass", "guitar")):
                melde("… Trennung dieses Songs schon vorhanden (Zwischenspeicher) – wird wiederverwendet")
                return {s: da[s] for s in STIMMEN if s in da}, name
    try:
        import torch
        torch.backends.mps.is_available = lambda: False      # wie im App-Server
        from audio_separator.separator import Separator
        sep = Separator(output_dir=roh, model_file_dir=modelle, output_format="WAV")
        sep.load_model("BS-Roformer-SW.ckpt")
        dateien = [f if os.path.isabs(f) else os.path.join(roh, f) for f in sep.separate(eingabe)]
        out = {}
        for s in STIMMEN:
            f = [p for p in dateien if f"({s})" in os.path.basename(p)]
            if f:
                out[s] = f[0]
        if all(s in out for s in ("drums", "bass", "guitar")):
            return (Sp.merken(eingabe, Sp.A_SECHS, out) if Sp else out), "BS-Roformer-SW"
        melde("… BS-Roformer-SW lieferte nicht alle Spuren, nehme htdemucs_6s")
    except Exception as e:
        melde(f"… BS-Roformer-SW nicht verfügbar ({str(e)[:120]}), nehme htdemucs_6s")
    cmd = [sys.executable, "-m", "demucs", "--name", "htdemucs_6s", "--out", roh, eingabe]
    p = subprocess.run(cmd, capture_output=True, text=True)
    d = os.path.join(roh, "htdemucs_6s", os.path.splitext(os.path.basename(eingabe))[0])
    if not os.path.isdir(d):
        raise RuntimeError("Trennung fehlgeschlagen: " + (p.stderr or "")[-300:])
    out = {s: os.path.join(d, s + ".wav") for s in STIMMEN if os.path.exists(os.path.join(d, s + ".wav"))}
    return (Sp.merken(eingabe, Sp.a_demucs("htdemucs_6s"), out) if Sp else out), "htdemucs_6s"


def lade_mono(p):
    lb = librosa()
    m, _ = lb.load(p, sr=SR, mono=True)
    return m.astype(np.float64)


# ------------------------------------------------------------ 2.–3. Raster, Tonart, Akkorde
@merke("gitarre_taktraster")
def taktraster(drums):
    lb = librosa()
    hop = 441
    o = lb.onset.onset_strength(y=drums, sr=SR, hop_length=hop)
    _, beats = lb.beat.beat_track(onset_envelope=o, sr=SR, hop_length=hop, start_bpm=86, tightness=200, units="time")
    if len(beats) < 8:
        raise RuntimeError("kein Taktraster gefunden (zu wenig Schlagzeug?)")
    P = float(np.median(np.diff(beats)))
    while beats[0] - P > 0:
        beats = np.r_[beats[0] - P, beats]
    while beats[-1] + P < len(drums) / SR:
        beats = np.r_[beats, beats[-1] + P]
    return beats


KRUMHANSL_DUR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MOLL = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@merke("gitarre_tonart")
def tonart(m):
    lb = librosa()
    C = lb.feature.chroma_cqt(y=m, sr=SR, hop_length=4096).sum(1)
    best = None
    for r in range(12):
        for art, P in (("dur", KRUMHANSL_DUR), ("moll", KRUMHANSL_MOLL)):
            c = np.corrcoef(C, np.roll(P, r))[0, 1]
            if best is None or c > best[0]:
                best = (c, r, art)
    return best[1], best[2]


def erlaubte(r, art):
    stufen = [(0, "m"), (3, "M"), (5, "m"), (7, "m"), (7, "M"), (8, "M"), (10, "M")] if art == "moll" else \
             [(0, "M"), (2, "m"), (4, "m"), (5, "M"), (7, "M"), (9, "m")]
    out = set()
    for i, t in stufen:
        for q in (("major", "major7") if t == "M" else ("minor", "minor7")):
            out.add(((r + i) % 12, q))
    return sorted(out)


@merke("gitarre_akkord_plan")
def akkord_plan(beats, harm, bass):
    lb = librosa()
    r_key, art = tonart(harm + 0.5 * bass)
    erl = erlaubte(r_key, art)

    def vorlage(root, q):
        v = np.zeros(12)
        for i, iv in enumerate(QUAL[q]):
            v[(root + iv) % 12] = 1.0 if i < 3 else 0.6
        return v / np.linalg.norm(v)
    T = np.array([vorlage(r, q) for r, q in erl])
    hopc = 2048
    C = lb.feature.chroma_cqt(y=harm, sr=SR, hop_length=hopc, bins_per_octave=36)
    tc = np.arange(C.shape[1]) * hopc / SR
    f0, vf, _ = lb.pyin(bass, fmin=35, fmax=300, sr=SR, frame_length=4096, hop_length=1024)
    tb = np.arange(len(f0)) * 1024 / SR
    e_h = lb.feature.rms(y=harm, hop_length=hopc)[0]
    e_b = lb.feature.rms(y=bass, hop_length=hopc)[0]
    n = len(beats) - 1
    S = np.zeros((n, len(erl)))
    energie = np.full(n, -120.0)
    for i in range(n):
        a, b = beats[i], beats[i + 1]
        sel = (tc >= a) & (tc < b)
        c = C[:, sel].mean(1) if sel.any() else np.zeros(12)
        c = c / (np.linalg.norm(c) + 1e-9)
        selb = (tb >= a) & (tb < b) & vf
        br = int(np.round(np.median(lb.hz_to_midi(f0[selb])))) % 12 if selb.sum() >= 3 else None
        S[i] = T @ c + np.array([0.12 if br == r else 0 for r, q in erl])
        if sel.any():
            energie[i] = 20 * np.log10(np.mean(e_h[sel]) + np.mean(e_b[sel]) + 1e-9)
    k = len(erl)
    D = S[0].copy()
    B = np.zeros((n, k), int)
    for i in range(1, n):
        cand = D[:, None] - 0.12 * (1 - np.eye(k))
        B[i] = np.argmax(cand, 0)
        D = cand[B[i], np.arange(k)] + S[i]
    p = [int(np.argmax(D))]
    for i in range(n - 1, 0, -1):
        p.append(B[i][p[-1]])
    p = p[::-1]
    akk = [erl[j] for j in p]
    ruhig = energie < np.median(energie) - 15
    akk = [None if ruhig[i] else a for i, a in enumerate(akk)]
    wechsel = [i for i in range(1, n) if akk[i] != akk[i - 1]]
    phase = max(range(4), key=lambda ph: sum(1 for w in wechsel if w % 4 == ph))
    return akk, phase, (r_key, art)


def akkord_name(a):
    return "N.C." if a is None else NAMEN[a[0]] + KURZ[a[1]]


# ------------------------------------------------------------ 4. Skank-Stellen
def hochpass(m, f=800):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(4, f, "high", fs=SR, output="sos"), m)


def einsatz(m, a, fenster=0.05):
    w = np.abs(m[a:a + int(fenster * SR)])
    return a + int(np.argmax(w > w.max() * 0.1)) if len(w) and w.max() > 0 else a


def skank_stellen(beats, akk, phase, git, muster="einfach"):
    lb = librosa()
    h = hochpass(git)
    hop = 128
    o = lb.onset.onset_strength(y=h, sr=SR, hop_length=hop)
    to = np.arange(len(o)) * hop / SR

    def ev(t, w=0.045):
        s = (to >= t - w) & (to <= t + w)
        return float(o[s].max()) if s.any() else 0.0
    n = len(beats) - 1
    LAGEN = (0.0, 0.5, 0.25, 0.75, 1 / 3, 2 / 3)
    E_alle = np.array([[ev(beats[i] + p * (beats[i + 1] - beats[i]), 0.03 if p not in (0.0, 0.5) else 0.045)
                        for p in LAGEN] for i in range(n)])
    E = E_alle[:, :2]
    ref = np.percentile(E, 90) + 1e-9
    E = E / ref
    Z = {p: E_alle[:, k] / ref for k, p in enumerate(LAGEN)}
    takte = list(range(phase, n, 4))
    aktiv = np.array([(E[t:t + 4, 1] > 0.25).sum() >= 2 for t in takte])
    idx = np.where(aktiv)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= 3:
            aktiv[a:b] = True                  # Lücken bis 2 Takte füllen (Trennung schwach)
    # Rhythmus-Schlag: fest gewählt oder automatisch wie im Original
    gewaehlt = muster
    akt_beats = [i for k, t in enumerate(takte) if aktiv[k] for i in range(t, min(t + 4, n)) if akk[i] is not None]
    if muster == "auto":
        med = lambda p: float(np.median(Z[p][akt_beats])) if akt_beats else 0.0   # noqa: E731
        swing = med(2 / 3) > 1.3 * med(0.5) and med(2 / 3) > 0.3
        gewaehlt = "shuffle" if swing else "auto"
    # Grundrauschen der Spur an Stellen, an denen kein Skank liegt (Übersprechen anderer Instrumente)
    boden = float(np.median(np.r_[Z[1 / 3][akt_beats], Z[0.0][akt_beats]])) if akt_beats else 0.05
    grund = MUSTER["shuffle"] if gewaehlt == "shuffle" else MUSTER.get(gewaehlt, MUSTER["einfach"])
    if gewaehlt == "auto":
        grund = MUSTER["einfach"]
    haupt_p = [p for p, r in grund if r == "haupt"][0]
    slots = {}
    for k, t in enumerate(takte):
        for i in range(t, min(t + 4, n)):
            if akk[i] is None:
                continue
            if aktiv[k] or E[i, 1] > 0.5:
                for p, r in grund:
                    slots[(i, p)] = r
                if gewaehlt == "auto":                       # Zusatz-Schläge nur dort, wo sie im Original klingen
                    for p, r in ((0.75, "nach"), (0.25, "nach")):
                        # im Takt mindestens zweimal (ein Muster, kein einzelner Zufallstreffer)
                        treffer = sum(1 for j in range(t, min(t + 4, n))
                                      if Z[p][j] > 0.15 * max(Z[0.5][j], 1e-9) and Z[p][j] > 4 * boden)
                        if treffer >= 2 and Z[p][i] > 0.15 * max(Z[0.5][i], 1e-9) and Z[p][i] > 4 * boden:
                            slots[(i, p)] = r
            if not aktiv[k] and E[i, 0] > 0.5:
                slots[(i, 0.0)] = "haupt"
    for i in range(0, phase):
        if akk[i] is not None and E[i, 1] > 0.5:
            slots[(i, haupt_p)] = "haupt"
    slots = sorted((i, p, r) for (i, p), r in slots.items())
    # Zeitpunkt: echter Gitarren-Anschlag in ±40 ms, sonst Raster + typischer Versatz
    on = lb.onset.onset_detect(y=h, sr=SR, hop_length=128, units="samples", backtrack=True, delta=0.05)
    on_t = np.array([einsatz(git, int(x)) / SR for x in on])
    zeiten, versatz = [], []
    for i, p, r in slots:
        t = beats[i] + p * (beats[i + 1] - beats[i])
        if len(on_t):
            j = int(np.argmin(np.abs(on_t - t)))
            if abs(on_t[j] - t) < (0.04 if p in (0.0, 0.5) else 0.025):
                zeiten.append(float(on_t[j]))
                versatz.append(on_t[j] - t)
                continue
        zeiten.append(None)
    dv = float(np.median(versatz)) if versatz else 0.0
    gefunden = sum(z is not None for z in zeiten)
    zeiten = [z if z is not None else beats[i] + p * (beats[i + 1] - beats[i]) + dv for z, (i, p, r) in zip(zeiten, slots)]
    from collections import Counter
    return slots, np.array(zeiten), {"stellen": len(slots), "im_stem_gefunden": gefunden, "ergaenzt": len(slots) - gefunden,
                                     "muster": muster, "muster_erkannt": gewaehlt if muster == "auto" else None,
                                     "schlaege": dict(Counter(r for _, _, r in slots))}


def schlagzeug_anschlaege(drums):
    lb = librosa()
    on = lb.onset.onset_detect(y=drums, sr=SR, hop_length=128, units="samples", backtrack=True, delta=0.05)
    return np.array([einsatz(drums, int(o), 0.03) / SR for o in on])


def am_schlagzeug(zeiten, D, fenster=0.04):
    if len(D) == 0:
        return zeiten, 0
    i = np.clip(np.searchsorted(D, zeiten), 1, len(D) - 1)
    n = np.where(np.abs(D[i] - zeiten) < np.abs(D[i - 1] - zeiten), D[i], D[i - 1])
    ok = np.abs(n - zeiten) <= fenster
    return np.where(ok, n, zeiten), int(ok.sum())


# ------------------------------------------------------------ 6. Spielen + Ton
def chroma(m, a):
    lb = librosa()
    seg = m[a + int(.01 * SR):a + int(.15 * SR)]
    if len(seg) < 2048:
        seg = np.pad(seg, (0, 2048 - len(seg)))
    C = lb.feature.chroma_cqt(y=seg, sr=SR, hop_length=512, bins_per_octave=36, n_octaves=5,
                              fmin=lb.note_to_hz("E2")).mean(1)
    return C / (np.linalg.norm(C) + 1e-9)


def bibliothek():
    lb = librosa()
    lib = []
    for f in sorted(glob.glob(os.path.join(SAMPLES, "*.flac"))):
        mm = re.match(r"([A-H]#?)_(major7|minor7|major|minor)_(Oben|Unten)_(\d+)\.flac$", os.path.basename(f))
        if not mm:
            continue
        x, s = sf.read(f, always_2d=True)
        m = x.mean(1)
        if s != SR:
            m = lb.resample(m, orig_sr=s, target_sr=SR)
        a = einsatz(m, 0, len(m) / SR)
        lib.append(dict(name=os.path.basename(f), root=NAMEN.index(mm.group(1).replace("H", "B")), q=mm.group(2),
                        lage=mm.group(3), x=m, a=a, chroma=chroma(m, a)))
    if not lib:
        raise RuntimeError(f"keine Akkord-Aufnahmen in {SAMPLES}")
    return lib


def griffe_waehlen(lib, git, slots, zeiten, akk):
    wahl = {}
    for a in sorted({akk[i] for i, *_ in slots}, key=str):
        ks = [s for s in lib if s["root"] == a[0] and s["q"] == a[1]]
        if not ks:                                       # Septime fehlt -> Dreiklang
            q = "major" if a[1].startswith("major") else "minor"
            ks = [s for s in lib if s["root"] == a[0] and s["q"] == q]
        if not ks:
            continue
        ch = [chroma(git, int(z * SR)) for z, (i, p, r) in zip(zeiten, slots) if akk[i] == a and r == "haupt"]
        ch = ch or [chroma(git, int(z * SR)) for z, (i, p, r) in zip(zeiten, slots) if akk[i] == a]
        mc = np.mean(ch, 0)
        wahl[a] = max(ks, key=lambda s: float(mc @ s["chroma"]))
    return wahl


def anschlag(seg, rolle):
    """Aus dem vollen Tchak den leichteren Aufschlag („chack“) bzw. den gedämpften Kratzer („chk“) formen."""
    from scipy.signal import butter, sosfiltfilt
    t = np.arange(len(seg)) / SR
    if rolle == "nach":                                   # Aufschlag: heller, kürzer, leiser
        seg = sosfiltfilt(butter(2, 280, "high", fs=SR, output="sos"), seg)
        return seg * np.exp(-np.maximum(t - 0.01, 0) / 0.09) * 10 ** (-5 / 20)
    if rolle == "ghost":                                  # gedämpft: nur das Kratzen der Saiten
        seg = sosfiltfilt(butter(2, 1500, "high", fs=SR, output="sos"), seg)
        return seg * np.exp(-np.maximum(t - 0.004, 0) / 0.014) * 10 ** (-7 / 20)
    return seg


def spielen(laenge, slots, zeiten, akk, wahl):
    rng = np.random.default_rng(7)
    y = np.zeros(laenge + SR)
    for z, (i, p, rolle) in zip(zeiten, slots):
        s = wahl.get(akk[i])
        if s is None:
            continue
        r = 2 ** (rng.uniform(-4, 4) / 1200)                  # ±4 Cent, damit es nicht nach Maschine klingt
        seg = anschlag(s["x"][max(0, s["a"] - int(.005 * SR)):s["a"] + int(.45 * SR)], rolle)
        tt = np.arange(0, len(seg) - 1, r)
        seg = np.interp(tt, np.arange(len(seg)), seg)
        a0 = int(z * SR) - int(.005 * SR)
        L = min(len(seg), len(y) - a0)
        if a0 < 0 or L <= 0:
            continue
        f = int(.02 * SR)
        w = np.ones(L)
        w[-f:] = np.linspace(1, 0, f)
        y[a0:a0 + L] += seg[:L] * w * 10 ** (rng.normal(0, 0.6) / 20)
    return y[:laenge]


# „C fett“: EQ-Kurve (dB je Band) aus dem Vergleich eigene Aufnahmen ↔ klassische Reggae-Skanks
BAND = [80, 120, 180, 250, 350, 500, 700, 1000, 1400, 2000, 2800, 4000, 5600, 8000, 11000]
FETT_EQ = [0.0, -13.3, -9.4, -3.6, -11.5, -15.0, -15.2, -18.1, -21.8, -12.0, -13.1, -17.1, -22.9, -19.6, -18.9]


def fett(mono):
    from scipy.signal import fftconvolve, butter, sosfiltfilt
    n = 8192
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.interp(np.log2(np.maximum(f, 20)), np.log2(BAND), FETT_EQ, left=FETT_EQ[0], right=FETT_EQ[-1])
    H = np.roll(np.fft.irfft(10 ** (g / 20)), n // 2) * np.hanning(n)
    y = fftconvolve(mono, H)[n // 2:n // 2 + len(mono)]
    x = np.stack([y, y])
    # Kompressor
    mm = np.abs(x).max(0)
    a_, r_ = np.exp(-1 / (0.003 * SR)), np.exp(-1 / (0.08 * SR))
    env = np.empty_like(mm)
    e = 0.0
    for i, v in enumerate(mm):
        e = a_ * e + (1 - a_) * v if v > e else r_ * e + (1 - r_) * v
        env[i] = e
    lv = 20 * np.log10(env + 1e-9)
    x = x * 10 ** ((np.minimum(0, (-18 - lv) * (1 - 1 / 3.0)) + 4.0) / 20)
    # kurzer Raum + Stereo-Doppelung
    rng = np.random.default_rng(1)
    k = int(.35 * SR)
    t = np.arange(k) / SR
    lp = butter(2, 5000, "low", fs=SR, output="sos")
    ir = [sosfiltfilt(lp, rng.standard_normal(k) * np.exp(-t / 0.07)) for _ in range(2)]
    ir = [i / np.sqrt(np.sum(i ** 2)) for i in ir]
    nass = np.stack([fftconvolve(x.mean(0), ir[0])[:x.shape[1]], fftconvolve(x.mean(0), ir[1])[:x.shape[1]]])
    dop = np.zeros_like(x)
    d1, d2 = int(.014 * SR), int(.009 * SR)
    dop[0, d1:] = x[1, :-d1] * 0.45
    dop[1, d2:] = x[0, :-d2] * 0.45
    x = x + dop + 0.18 * nass
    x = x / (np.abs(x).max() + 1e-12) * 0.7
    return (np.tanh(1.6 * x) / np.tanh(1.6)).T


# ------------------------------------------------------------ 7. MIDI + Akkord-Plan
def midi_schreiben(zeiten, slots, akk, ziel, bpm=120.0):
    """Je Skank ein Akkord (Grundton in Oktave 3, Akkordtöne darüber), 1/16 lang, Kanal 1, ohne Zusatz-Bibliothek."""
    ppq = 960
    tempo = float(bpm) if bpm and bpm > 0 else 120.0
    tick = lambda s: int(round(s * tempo / 60.0 * ppq))  # noqa: E731
    ev = []
    for z, (i, p, rolle) in zip(zeiten, slots):
        a = akk[i]
        if a is None:
            continue
        basis = 48 + a[0]                       # C3..B3
        vel, lang = {"haupt": (100, ppq // 4), "nach": (72, ppq // 8), "ghost": (40, ppq // 16)}.get(rolle, (100, ppq // 4))
        for iv in QUAL[a[1]]:
            ev.append((tick(z), 1, basis + iv, vel))
            ev.append((tick(z) + lang, 0, basis + iv, 0))
    ev.sort(key=lambda e: (e[0], e[1]))

    def vlq(n):
        b = [n & 0x7F]
        n >>= 7
        while n:
            b.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(b)

    def spur(d):
        return b"MTrk" + struct.pack(">I", len(d)) + d
    usek = int(round(60_000_000 / tempo))
    t0 = b"\x00\xff\x51\x03" + usek.to_bytes(3, "big") + b"\x00\xff\x58\x04\x04\x02\x18\x08" + b"\x00\xff\x2f\x00"
    nb = "Gitarre Skank repariert".encode("utf-8")
    d = b"\x00\xff\x03" + vlq(len(nb)) + nb
    letzt = 0
    for tk, an, ton, v in ev:
        d += vlq(tk - letzt) + bytes([0x90 if an else 0x80, ton, v])
        letzt = tk
    d += b"\x00\xff\x2f\x00"
    datei = os.path.join(ziel, "gitarre_neu.mid")
    with open(datei, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + spur(t0) + spur(d))
    return {"datei": datei, "tempo_bpm": tempo, "anschlaege": len(zeiten)}


def plan_schreiben(beats, akk, phase, ton, ziel, bpm_song, bpm_midi, wahl):
    z = ["Gitarre (Skank) – Akkord-Plan, erkannt aus allen Instrumenten",
         f"Tonart: {NAMEN[ton[0]]}-{ton[1]}, Song-Tempo ca. {bpm_song:.1f} BPM, 4 Schläge je Takt (A# = B♭, D# = E♭)",
         f"MIDI-Datei mit {bpm_midi:g} BPM geschrieben: Logic-Projekt auf {bpm_midi:g} BPM stellen.",
         "Griffe: " + ", ".join(f"{akkord_name(a)} = {s['name'][:-5]}" for a, s in wahl.items()), ""]
    if phase:
        z.append(f"{int(beats[0] // 60)}:{beats[0] % 60:05.2f}  Vortakt  " + " | ".join(akkord_name(a) for a in akk[:phase]))
    for k, t in enumerate(range(phase, len(akk), 4)):
        s = beats[t]
        z.append(f"{int(s // 60)}:{s % 60:05.2f}  Takt {k + 1:3d}  " + " | ".join(f"{akkord_name(a):6s}" for a in akk[t:t + 4]))
    with open(os.path.join(ziel, "gitarre_neu_Akkord_Plan.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(z) + "\n")


# ------------------------------------------------------------ Ablauf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe")
    ap.add_argument("ziel")
    ap.add_argument("--timing-korrigieren", action="store_true",
                    help="jeden Tchak genau auf den Schlagzeug-Schlag setzen (±40 ms)")
    ap.add_argument("--midi-bpm", type=float, default=120.0)
    ap.add_argument("--lautstaerke-db", type=float, default=0.0, help="Feinanpassung der Gitarren-Lautstärke")
    ap.add_argument("--spuren-ordner", default=None, help="schon getrennte Spuren (drums/bass/guitar/piano/other.wav)")
    ap.add_argument("--modelle", default=os.path.expanduser("~/HELTECH_Stems/models/"))
    ap.add_argument("--muster", default="auto", help="Rhythmus-Schlag: " + ", ".join(MUSTER_NAMEN))
    a = ap.parse_args()
    a.muster = a.muster if a.muster in MUSTER_NAMEN else "auto"
    os.makedirs(a.ziel, exist_ok=True)
    try:
        if a.spuren_ordner:
            sp = {s: glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))[0] for s in STIMMEN
                  if glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))}
            modell = "vorhanden"
        else:
            melde("… trenne den Song in 6 Spuren (dauert einige Minuten)")
            sp, modell = trennen(a.eingabe, a.ziel, a.modelle)
        melde(f"✔ Spuren getrennt ({modell})")
        st = {s: lade_mono(p) for s, p in sp.items() if s != "vocals"}
        n = min(len(v) for v in st.values())
        st = {s: v[:n] for s, v in st.items()}
        git = st["guitar"]
        harm = st.get("piano", 0) + st.get("other", 0) + 3 * git
        melde("… Taktraster und Akkorde")
        beats = taktraster(st["drums"])
        akk, phase, ton = akkord_plan(beats, harm, st["bass"])
        melde(f"✔ Tonart {NAMEN[ton[0]]}-{ton[1]}, {len(beats)} Schläge")
        melde("… Skank-Stellen")
        slots, zeiten, info = skank_stellen(beats, akk, phase, git, a.muster)
        if a.muster == "auto":
            melde(f"✔ Rhythmus-Schlag automatisch: {MUSTER_NAMEN.get(info['muster_erkannt'], 'wie im Original')} "
                  f"({', '.join(f'{v}× {k}' for k, v in info['schlaege'].items())})")
        timing = None
        if a.timing_korrigieren:
            zeiten, k = am_schlagzeug(zeiten, schlagzeug_anschlaege(st["drums"]))
            timing = {"am_schlagzeug": k, "von": len(zeiten)}
            melde(f"✔ Timing: {k} von {len(zeiten)} Tchaks genau auf dem Schlagzeug")
        melde("… spiele die Gitarre neu (eigene Akkord-Aufnahmen, „C fett“)")
        lib = bibliothek()
        wahl = griffe_waehlen(lib, git, slots, zeiten, akk)
        y = fett(spielen(n, slots, zeiten, akk, wahl))
        # gleichmäßige Lautstärke: wie die kräftigen Stellen der Original-Gitarre, +6 dB (wie „v2 lauter“)
        k = int(2.8 * SR)
        a0, b0 = int(zeiten[0] * SR) if len(zeiten) else 0, int(zeiten[-1] * SR) if len(zeiten) else n

        def takt_rms(x):
            return np.array([np.sqrt(np.mean(x[i:i + k] ** 2)) for i in range(a0, max(a0 + 1, b0 - k), k)])
        ziel_rms = np.percentile(takt_rms(git), 75)
        ist = np.median(takt_rms(y.mean(1)))
        y *= 2.0 * ziel_rms / max(ist, 1e-12) * 10 ** (a.lautstaerke_db / 20)
        rest = sum(v for s, v in st.items() if s != "guitar")
        if "vocals" in sp:
            rest = rest + lade_mono(sp["vocals"])[:n]
        s_max = max(np.abs(y).max(), 1e-12)
        if s_max > 0.98:
            y *= 0.98 / s_max
        mix = np.stack([rest, rest], 1) + y
        mix *= min(1.0, 0.98 / max(np.abs(mix).max(), 1e-12))
        out = os.path.join(a.ziel, "gitarre_neu.wav")
        sf.write(out, y, SR, subtype="PCM_24")
        sf.write(os.path.join(a.ziel, "song_mit_neuer_gitarre.wav"), mix, SR, subtype="PCM_24")
        bpm_song = 60.0 / float(np.median(np.diff(beats)))
        plan_schreiben(beats, akk, phase, ton, a.ziel, bpm_song, a.midi_bpm, wahl)
        bericht = {"trennung": modell, "tonart": f"{NAMEN[ton[0]]}-{ton[1]}", "tempo_bpm": round(bpm_song, 2),
                   "skank": info, "timing_am_schlagzeug": timing,
                   "griffe": {akkord_name(k_): s["name"] for k_, s in wahl.items()},
                   "akkorde": {akkord_name(x): sum(1 for i, _, _ in slots if akk[i] == x) for x in {akk[i] for i, _, _ in slots}}}
        try:
            bericht["midi"] = midi_schreiben(zeiten, slots, akk, a.ziel, a.midi_bpm)
        except Exception as e:
            bericht["midi"] = {"fehler": str(e)}
        json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
        shutil.rmtree(os.path.join(a.ziel, "spuren_roh"), ignore_errors=True)
        print(json.dumps({"ok": True, "datei": out, "ordner": a.ziel, "anschlaege": len(zeiten),
                          "tonart": bericht["tonart"]}, ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "fehler": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
