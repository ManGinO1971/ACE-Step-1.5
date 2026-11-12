#!/usr/bin/env python3
"""
HELTECH Stems – Bass neu einspielen (Stand 5. Nov 2026, getestet mit „Rise Up Jah Children“, Bass v1).

Eingabe: eine Bass-Spur (am besten sauber getrennt). Ablauf:
  1. Tonhöhenverlauf (pyin, 5 ms) und Lautstärke-Kurve (1 ms, 25 ms geglättet)
  2. Noten erkennen: Anschlag = Einbruch und Wiederanstieg, starker Anstieg nach leiser Stelle oder Tonwechsel
  3. Neu spielen mit den eigenen Bass-Aufnahmen (samples/, E1–D#4):
     - Tonhöhe folgt dem Original (Slides, Hammer-on/Pull-off: weich umblenden, kein neuer Anschlag)
     - Lautstärke-Kurve jeder Note aus dem Original (Länge, Halten, Abdämpfen)
     - optional: Klang (Langzeit-Spektrum) an das Original angleichen
  4. Ausgabe: bass_neu.wav, bass_neu.mid (+ Notenliste), bericht.json

Aufruf:
  python bass_neu_step.py <bass.wav> <ausgabe_ordner> [--klang-wie-original] [--midi-bpm 120]
        [--timing-korrigieren]      Timing-Wackeln entfernen (Raster aus der Spur selbst)
        [--schlagzeug DATEI]        stattdessen am Schlagzeug ausrichten (Audio oder drums_neu.mid)
Ausgabe auf stdout (letzte Zeile): JSON {"ok": true, "datei": ..., ...}
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np
import soundfile as sf

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIER)
SAMPLES = os.path.join(HIER, "samples")
SR = 44100


def melde(text):
    print(text, flush=True)


try:
    sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
    from merker import merke
except Exception:  # noqa: BLE001 – ohne Merker läuft alles wie bisher
    def merke(*_a, **_k):
        return lambda fn: fn


def lade(p):
    import librosa
    m, _ = librosa.load(p, sr=SR, mono=True)
    return m.astype(np.float64)


def glatt(x, n):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, max(1, int(n)))


# ------------------------------------------------------------ 1. Analyse
@merke("bass_tonkurve")
def tonkurve(m):
    """MIDI-Tonhöhe alle 5 ms (NaN = kein Ton)."""
    import librosa
    hop = int(SR * 0.005)
    f0, _, _ = librosa.pyin(m, fmin=35, fmax=420, sr=SR, frame_length=4096, hop_length=hop)
    return librosa.hz_to_midi(f0), hop


def huelle_db(m, glatt_ms=25):
    k = int(SR * 0.001)
    n = len(m) // k
    p = (m[:n * k].reshape(n, k) ** 2).mean(1)
    return 10 * np.log10(np.maximum(glatt(p, glatt_ms), 0) + 1e-18), k


# ------------------------------------------------------------ 2. Noten
def noten_finden(m, kurve, unter=32.0, dip=10.0):
    e, k = huelle_db(m)
    top = np.percentile(e, 99.5)
    an = e > top - unter
    midi = np.repeat(kurve, 5)[:len(e)]
    if len(midi) < len(e):
        midi = np.pad(midi, (0, len(e) - len(midi)), constant_values=np.nan)
    abschn, i, n = [], 0, len(e)
    while i < n:
        if not an[i]:
            i += 1
            continue
        j = i
        while j < n and an[j]:
            j += 1
        if j - i >= 25:
            abschn.append((i, j))
        i = j
    res = []
    for a, b in abschn:
        gr = [a]
        for t in range(a + 30, b - 30):        # Einbruch: Minimum mind. dip dB unter beiden Seiten (±60 ms)
            if e[t] == e[t - 10:t + 11].min() and e[t] < e[max(a, t - 60):t].max() - dip \
                    and e[t] < e[t + 1:t + 61].max() - dip and t - gr[-1] >= 40:
                gr.append(t)
        for t in range(a + 30, b - 40):        # neuer Anschlag nach leiser Stelle: +12 dB in 40 ms
            if e[t] == e[t - 10:t + 11].min() and e[t + 40] - e[t] >= 12 and all(abs(t - g) >= 40 for g in gr):
                gr.append(t)
        gr = sorted(gr)
        q = np.round(midi[a:b])                # Tonwechsel (je 30 ms stabil davor/danach)
        for t in range(a + 30, b - 30):
            v, w = q[t - a - 30:t - a], q[t - a:t - a + 30]
            if not np.isnan(v).any() and not np.isnan(w).any() and np.all(v == v[0]) and np.all(w == w[0]) \
                    and v[0] != w[0] and all(abs(t - g) >= 40 for g in gr):
                gr.append(t)
        gr = sorted(gr) + [b]
        for s, z in zip(gr[:-1], gr[1:]):
            if z - s < 25:
                continue
            mm = midi[s + 10:z - 5]
            mm = mm[~np.isnan(mm)]
            if len(mm) < 10:
                continue
            ton = float(np.median(mm))
            seg = m[s * k:z * k]
            pk = np.abs(seg).max()
            ein = s * k + int(np.argmax(np.abs(seg) > pk * 0.1))
            res.append(dict(start=s * k, einsatz=ein, ende=z * k, ton=ton, pegel=float(np.percentile(e[s:z], 90))))
    out = []                                    # kurzer leiser Rest (Slide/Ausklang) gehört zur Note davor
    for n_ in res:
        if out and n_["start"] - out[-1]["ende"] < 0.012 * SR and (n_["ende"] - n_["start"]) < 0.12 * SR \
                and n_["pegel"] < out[-1]["pegel"] - 8:
            out[-1]["ende"] = n_["ende"]
            continue
        out.append(dict(n_))
    return out


# ------------------------------------------------------------ 2b. Timing am Schlagzeug
def schlagzeug_einsaetze(pfad):
    """Anschläge der Schlagzeug-Spur (s), ausgerichtet wie beim Schlagzeug-Reparieren: erster Punkt über
    10 % der Spitze nach jedem Onset. Pfad darf eine Audio-Spur oder eine MIDI-Datei (drums_neu.mid) sein."""
    if pfad.lower().endswith((".mid", ".midi")):
        return np.array(_midi_einsaetze(pfad))
    import librosa
    d = lade(pfad)
    on = librosa.onset.onset_detect(y=d, sr=SR, hop_length=128, units="samples", backtrack=True)
    out = []
    for o in on:
        seg = np.abs(d[o:o + int(.03 * SR)])
        if len(seg) and seg.max() > 0:
            out.append((o + int(np.argmax(seg > seg.max() * 0.1))) / SR)
    return np.array(out)


def _midi_einsaetze(pfad):
    """Note-on-Zeiten (s) aus einer SMF-Datei, mit Tempo-Wechseln – ohne Zusatz-Bibliothek."""
    daten = open(pfad, "rb").read()
    ppq = struct.unpack(">H", daten[12:14])[0]
    i = 14
    ereignisse = []                            # (tick, art, wert)
    while i < len(daten) - 8:
        if daten[i:i + 4] != b"MTrk":
            break
        laenge = struct.unpack(">I", daten[i + 4:i + 8])[0]
        j, ende, tick, status = i + 8, i + 8 + laenge, 0, 0
        while j < ende:
            dt = 0
            while True:
                b_ = daten[j]; j += 1
                dt = (dt << 7) | (b_ & 0x7F)
                if not b_ & 0x80:
                    break
            tick += dt
            if daten[j] == 0xFF:
                typ, j = daten[j + 1], j + 2
                ln = 0
                while True:
                    b_ = daten[j]; j += 1
                    ln = (ln << 7) | (b_ & 0x7F)
                    if not b_ & 0x80:
                        break
                if typ == 0x51:
                    ereignisse.append((tick, 0, int.from_bytes(daten[j:j + 3], "big")))
                j += ln
            elif daten[j] in (0xF0, 0xF7):
                j += 1
                ln = 0
                while True:
                    b_ = daten[j]; j += 1
                    ln = (ln << 7) | (b_ & 0x7F)
                    if not b_ & 0x80:
                        break
                j += ln
            else:
                if daten[j] & 0x80:
                    status = daten[j]; j += 1
                art = status & 0xF0
                n_daten = 1 if art in (0xC0, 0xD0) else 2
                if art == 0x90 and daten[j + 1] > 0:
                    ereignisse.append((tick, 1, 0))
                j += n_daten
        i = ende
    ereignisse.sort()
    t, letzt, usek, out = 0.0, 0, 500000, []
    for tk, art, w in ereignisse:
        t += (tk - letzt) * usek / 1e6 / ppq
        letzt = tk
        if art == 0:
            usek = w
        else:
            out.append(t)
    return out


def timing_angleichen(N, schlaege, fenster=0.05, staerke=1.0):
    """Jede Bass-Note mit einem Schlagzeug-Schlag in ±fenster genau dorthin; Noten ohne Schlag um den
    mittleren Versatz ihrer Nachbarn (±2 s). Setzt n['schub'] (Samples). Rückgabe: Bericht."""
    if len(schlaege) == 0 or not N:
        return {"hinweis": "keine Schlagzeug-Schläge"}
    s = np.sort(np.asarray(schlaege))
    t = np.array([n["einsatz"] / SR for n in N])
    i = np.clip(np.searchsorted(s, t), 1, len(s) - 1)
    naechst = np.where(np.abs(s[i] - t) < np.abs(s[i - 1] - t), s[i], s[i - 1])
    d = naechst - t
    fest = np.abs(d) <= fenster
    schub = np.zeros(len(N))
    schub[fest] = d[fest]
    for k in np.where(~fest)[0]:
        nb = fest & (np.abs(t - t[k]) <= 2.0)
        schub[k] = float(np.median(d[nb])) if nb.any() else 0.0
    schub *= staerke
    for n, x in zip(N, schub):
        n["schub"] = int(round(x * SR))
    vorher = np.abs(d[fest]) * 1000
    return {"am_schlag": int(fest.sum()), "mitgezogen": int((~fest).sum()),
            "versatz_vorher_ms_median": round(float(np.median(d[fest]) * 1000), 1) if fest.any() else None,
            "versatz_vorher_ms_90": round(float(np.percentile(vorher, 90)), 1) if fest.any() else None}


# ------------------------------------------------------------ 3. Spielen
def bibliothek():
    import librosa
    lib = []
    for f in sorted(glob.glob(os.path.join(SAMPLES, "*.flac"))):
        x, s = sf.read(f, always_2d=True)
        mm = x.mean(1)
        if s != SR:
            mm = librosa.resample(mm, orig_sr=s, target_sr=SR)
        on = int(np.argmax(np.abs(mm) > np.abs(mm).max() * 0.1))
        f0 = np.nanmedian(librosa.pyin(mm[on + int(.1 * SR):on + int(.8 * SR)], fmin=35, fmax=420, sr=SR,
                                       frame_length=4096)[0])
        lib.append(dict(name=os.path.basename(f), x=mm, on=on, ton=float(librosa.hz_to_midi(f0))))
    if not lib:
        raise RuntimeError(f"keine Bass-Aufnahmen in {SAMPLES}")
    return lib


def env(y, ms=15):
    return np.sqrt(np.maximum(glatt(y ** 2, SR * ms / 1000), 0) + 1e-14)


def _glatt_kurve(p, ton):
    """Lücken interpolieren, Oktav-Ausreißer (> 6 Halbtöne vom Umfeld) entfernen."""
    from scipy.ndimage import median_filter
    p = p.copy()
    if np.isnan(p).all():
        return np.full(len(p), ton)
    i = np.arange(len(p))
    ok = ~np.isnan(p)
    p = np.interp(i, i[ok], p[ok])
    med = median_filter(p, size=int(.06 * SR), mode="nearest")
    bad = np.abs(p - med) > 6
    if bad.any() and (~bad).any():
        p[bad] = np.interp(i[bad], i[~bad], p[~bad])
    return glatt(p, .012 * SR)


def _strom(s, p, vorlauf):
    """Aufnahme s ab ihrem Anschlag abspielen, Tonhöhe folgt p (Halbtöne je Sample)."""
    r = 2 ** ((p - s["ton"]) / 12)
    pos = s["on"] - vorlauf * r[0] + np.concatenate([[0], np.cumsum(r[:-1])])
    ok = (pos >= 0) & (pos < len(s["x"]) - 1)
    z = np.zeros(len(p))
    z[ok] = np.interp(pos[ok], np.arange(len(s["x"])), s["x"])
    return z


def spielen(m, N, lib, kurve, hop):
    y = np.zeros(len(m) + SR)
    E = env(m)
    tonliste = np.array([q["ton"] for q in lib])
    benutzt = {}
    for i, n in enumerate(N):
        sh = n.get("schub", 0)                 # Timing-Verschiebung (Samples); Quelle = Original bei t - sh
        a = max(0, n["einsatz"] + sh - int(.008 * SR))
        nxt = N[i + 1]["einsatz"] + N[i + 1].get("schub", 0) - int(.008 * SR) if i + 1 < len(N) else len(m)
        # nie den nächsten Anschlag des Originals in diese Note ziehen
        quelle_ende = N[i + 1]["einsatz"] - int(.008 * SR) + sh if i + 1 < len(N) else len(m) + sh
        b = min(nxt, n["ende"] + sh + int(.15 * SR), quelle_ende, len(m))
        if b - a < int(.02 * SR) or a - sh < 0:
            continue
        t = np.arange(a, b) - sh
        p = _glatt_kurve(kurve[np.clip(t // hop, 0, len(kurve) - 1)], n["ton"])
        # je Zeitpunkt die Aufnahme mit dem nächsten Ton (Hysterese); Hammer-on/Pull-off weich umblenden,
        # OHNE neuen Anschlag (alle Ströme laufen ab dem Notenanschlag)
        wahl = np.empty(len(p), int)
        akt = None
        st = int(.005 * SR)
        for j in range(0, len(p), st):
            bester = int(np.argmin(np.abs(tonliste - p[j])))
            if akt is None or abs(tonliste[akt] - p[j]) > abs(tonliste[bester] - p[j]) + 0.6:
                akt = bester
            wahl[j:j + st] = akt
        z = np.zeros(len(p))
        for k in np.unique(wahl):
            w = glatt((wahl == k).astype(float), .012 * SR)
            z += w * _strom(lib[k], p, n["einsatz"] + sh - a)
            benutzt[lib[k]["name"]] = benutzt.get(lib[k]["name"], 0) + 1
        g = np.minimum(E[a - sh:b - sh] / env(z), 10 ** (36 / 20))   # Lautstärke-Kurve des Originals
        g = glatt(g, .008 * SR)
        f = int(.003 * SR)
        w = np.ones(b - a)
        w[:f] = np.linspace(0, 1, f)
        w[-f:] = np.linspace(1, 0, f)
        y[a:b] += z * g * w
        n["kurve"] = p[::st].tolist()
    return y[:len(m)], benutzt


def klang_angleichen(y, m, nfft=16384):
    """Langzeit-Spektrum an das Original angleichen (1/6 Oktave geglättet, ±24 dB, linearphasig)."""
    import librosa
    from scipy.signal import fftconvolve

    def ltas(z):
        return (np.abs(librosa.stft(z, n_fft=nfft, hop_length=nfft // 4)) ** 2).mean(1)
    f = librosa.fft_frequencies(sr=SR, n_fft=nfft)
    A, Y = ltas(m), ltas(y)
    lf = np.log2(np.maximum(f, 1))
    d = np.zeros_like(f)
    for i in range(len(f)):
        sel = np.abs(lf - lf[i]) < 1 / 12
        d[i] = 10 * np.log10(A[sel].sum() + 1e-20) - 10 * np.log10(Y[sel].sum() + 1e-20)
    d = np.clip(d, -24, 24)
    d[f < 25] = np.minimum(d[f < 25], 0)
    h = np.roll(np.fft.irfft(10 ** (d / 20)), nfft // 2) * np.hanning(nfft)
    z = fftconvolve(y, h)[nfft // 2:nfft // 2 + len(y)]
    return z, {str(q): round(float(np.interp(q, f, d)), 1) for q in (40, 60, 100, 150, 200, 300, 500, 1000, 2000)}


# ------------------------------------------------------------ 4. MIDI
NOTENNAMEN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def notenname(n):
    return f"{NOTENNAMEN[n % 12]}{n // 12 - 1}"


def midi_noten(N):
    """Je gespielter Note Teil-Noten je stabiler Tonstufe (Hammer-on/Pull-off = eigene Note)."""
    out = []
    dt = 0.005
    for n in N:
        if "kurve" not in n:
            continue
        q = np.round(np.array(n["kurve"])).astype(int)
        sh = n.get("schub", 0)
        t0, t_ende = (n["einsatz"] + sh) / SR, (n["ende"] + sh) / SR
        stufen, s = [], 0
        for j in range(1, len(q) + 1):
            if j == len(q) or q[j] != q[s]:
                stufen.append((s, j, q[s]))
                s = j
        lang = [st for st in stufen if (st[1] - st[0]) * dt >= 0.05] or [max(stufen, key=lambda st: st[1] - st[0])]
        for k, (a, b, ton) in enumerate(lang):
            beg = t0 if k == 0 else t0 + a * dt
            end = t0 + lang[k + 1][0] * dt if k + 1 < len(lang) else t_ende
            if end > beg + 0.01:
                out.append((beg, end, int(ton), n["pegel"]))
    return out


def midi_schreiben(noten, ziel, bpm=120.0):
    """SMF Typ 1, Kanal 1, PPQ 960 – ohne Zusatz-Bibliothek. Logic spielt MIDI im Projekttempo,
    darum Standard 120 BPM (= neues Logic-Projekt): dann liegt das MIDI genau unter dem Audio."""
    if not noten:
        return {"hinweis": "keine Noten"}
    ppq = 960
    tempo = float(bpm) if bpm and bpm > 0 else 120.0
    tick = lambda s: int(round(s * tempo / 60.0 * ppq))  # noqa: E731
    p95 = float(np.percentile([x[3] for x in noten], 95))
    ev = []
    for beg, end, ton, peg in noten:
        v = int(np.clip(round(110 + 2.5 * (peg - p95)), 20, 127))
        ev.append((tick(beg), 1, ton, v))
        ev.append((max(tick(end), tick(beg) + 1), 0, ton, 0))
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
    nb = "Bass repariert".encode("utf-8")
    d = b"\x00\xff\x03" + vlq(len(nb)) + nb
    letzt = 0
    for tk, an, ton, v in ev:
        d += vlq(tk - letzt) + bytes([0x90 if an else 0x80, ton, v])
        letzt = tk
    d += b"\x00\xff\x2f\x00"
    datei = os.path.join(ziel, "bass_neu.mid")
    with open(datei, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + spur(t0) + spur(d))
    from collections import Counter
    c = Counter(x[2] for x in noten)
    zeilen = [f"Bass repariert – MIDI-Notenliste (Kanal 1), MIDI-Tempo {tempo:g} BPM", ""]
    zeilen += [f"Note {k:3d}  {notenname(k):4s} {c[k]:5d}x" for k in sorted(c)]
    zeilen += ["", f"WICHTIG Tempo: Die Datei ist mit {tempo:g} BPM geschrieben. Logic-Projekt auf {tempo:g} BPM stellen,",
               "dann liegt das MIDI genau unter dem Audio (Logic übernimmt das Tempo der Datei beim Hineinziehen nicht).",
               "Hammer-on/Pull-off und Slides: jeder gehaltene Ton ist eine eigene Note (ohne Pitch-Bend)."]
    with open(os.path.join(ziel, "bass_neu_Notenliste.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(zeilen) + "\n")
    return {"datei": datei, "tempo_bpm": tempo, "noten": len(noten)}


# ------------------------------------------------------------ Ablauf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe")
    ap.add_argument("ziel")
    ap.add_argument("--klang-wie-original", action="store_true",
                    help="Klang der eigenen Aufnahmen an den Bass im Original angleichen (EQ)")
    ap.add_argument("--midi-bpm", type=float, default=120.0)
    ap.add_argument("--schlagzeug", default=None,
                    help="Schlagzeug-Spur (Audio oder drums_neu.mid) als Taktgeber: Bass-Timing daran ausrichten")
    ap.add_argument("--timing-fenster-ms", type=float, default=50.0)
    ap.add_argument("--timing-korrigieren", action="store_true",
                    help="Timing ohne Bezugsspur korrigieren: Raster aus der Spur selbst, Wackeln entfernen")
    a = ap.parse_args()
    os.makedirs(a.ziel, exist_ok=True)
    try:
        melde("… lade Bass-Spur")
        m = lade(a.eingabe)
        melde("… messe Tonhöhenverlauf (dauert etwas)")
        kurve, hop = tonkurve(m)
        melde("… erkenne Noten")
        N = noten_finden(m, kurve)
        melde(f"✔ {len(N)} Noten erkannt")
        timing = None
        if a.timing_korrigieren and not a.schlagzeug:
            import timing_korrektur as tk
            melde("… korrigiere das Timing (Raster aus der Spur selbst)")
            t = np.array([n_["einsatz"] / SR for n_ in N])
            w = np.array([10 ** (n_["pegel"] / 20) for n_ in N])
            schub, timing = tk.raster_schub(t, tk.tempo_schaetzen(m, SR), w)
            for n_, x in zip(N, schub):
                n_["schub"] = int(round(x * SR))
            melde(f"✔ Timing: {timing.get('verschoben', 0)} Noten aufs Raster ({timing.get('raster', '')})")
        if a.schlagzeug:
            melde("… richte das Timing am Schlagzeug aus")
            timing = timing_angleichen(N, schlagzeug_einsaetze(a.schlagzeug), a.timing_fenster_ms / 1000)
            melde(f"✔ {timing.get('am_schlag', 0)} Noten am Schlag, {timing.get('mitgezogen', 0)} mitgezogen")
        melde("… lade eigene Bass-Aufnahmen")
        lib = bibliothek()
        melde("… spiele Bass neu")
        y, benutzt = spielen(m, N, lib, kurve, hop)
        bericht = {"noten": len(N), "aufnahmen_benutzt": benutzt, "klang_wie_original": a.klang_wie_original,
                   "timing_am_schlagzeug": timing}
        if a.klang_wie_original:
            melde("… gleiche Klang an das Original an")
            y, bericht["eq_korrektur_db"] = klang_angleichen(y, m)
        spitze = float(np.abs(y).max())
        if spitze > 0.98:
            y *= 0.98 / spitze
        out = os.path.join(a.ziel, "bass_neu.wav")
        sf.write(out, np.stack([y, y], 1), SR, subtype="PCM_24")
        try:
            bericht["midi"] = midi_schreiben(midi_noten(N), a.ziel, a.midi_bpm)
        except Exception as e:                  # MIDI ist eine Zugabe – Audio nie daran scheitern lassen
            bericht["midi"] = {"fehler": str(e)}
        json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
        print(json.dumps({"ok": True, "datei": out, "ordner": a.ziel, "noten": len(N)}, ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "fehler": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
