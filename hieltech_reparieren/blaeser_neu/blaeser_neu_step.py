#!/usr/bin/env python3
"""
HELTECH Stems – Bläser / Melodica neu spielen (Reggae-Bläsersatz, Melodica). Stand 10. Nov 2026.

Ablauf:
  1. Bläser-Spur (53-Instrumente-Modell „brass“, sonst „harmonica“/„accordion“) – vom Server getrennt
  2. Leise Stellen angehoben (langsame Pegel-Regelung), dann alle gespielten Stimmen als Noten (Basic Pitch)
     und die Tonhöhen-Kurve der Oberstimme (pyin: Bends, Vibrato, Slides)
  3. Vollen Song mithören (wenn gewünscht): Song in 6 Spuren trennen; Takte, in denen die Bläser-Spur
     stumm ist, im vollen Song aber lange hohe Töne wie in den Bläser-Takten klingen -> diese Töne ergänzen
     (das Trennmodell hat sie verpasst). Jeder Ton wird am vollen Klang geprüft (Ton, ±1 Halbton, ±1 Oktave)
  4. Neu gespielt: Bläsersatz (Trompete + Tenorsax + Posaune eine Oktave tiefer, Mittelstimmen Altsax/Posaune)
     oder ein einzelnes Instrument oder Melodica; Lautstärke je Ton wie im Original, Oberstimme folgt der Kurve
  5. Klangfarbe ans Original angeglichen, etwas Sättigung und Raum, Pegel wie das Original
  6. Ausgabe: blaeser_neu.wav, song_mit_blaesern_neu.wav, blaeser_neu.mid (Spur je Instrument, Pitch-Bend),
     Notenliste

Aufruf:
  python blaeser_neu_step.py <blaeser.wav> <ziel> [--song song.wav] [--klang satz|trompete|altsax|tenorsax|posaune|melodica]
                             [--nicht-mithoeren] [--midi-bpm 120] [--modelle DIR] [--spuren-ordner DIR]
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np
import soundfile as sf

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIER = os.path.dirname(os.path.abspath(__file__))
for _d in ("piano_neu", "picking_neu", "lead_neu"):
    sys.path.insert(0, os.path.join(APP, _d))
import skank_kern as K  # noqa: E402

SR = 44100
HOP = 256
KLAENGE = {"satz": "Bläsersatz (Trompete, Tenorsax, Posaune)", "trompete": "Trompete", "altsax": "Altsaxophon",
           "tenorsax": "Tenorsaxophon", "posaune": "Posaune", "melodica": "Melodica"}
SPURNAME = {"tp": "Trompete", "as": "Altsaxophon", "ts": "Tenorsaxophon", "tb": "Posaune", "me": "Melodica"}
GM = {"tp": (56, "Trompete"), "as": (65, "Altsaxophon"), "ts": (66, "Tenorsaxophon"), "tb": (57, "Posaune"),
      "me": (22, "Melodica (GM Mundharmonika)")}


def melde(t):
    print(t, flush=True)


def lb():
    return K.librosa()


def bandpass(y, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y, axis=0)


def stereo(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return np.stack([x, x], 1)
    if x.shape[0] == 2 and x.shape[1] != 2:
        x = x.T
    return x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)


def kurve_fuellen(f):
    f = np.array(f, float)
    ok = np.isfinite(f)
    if not ok.any():
        return None
    f[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), f[ok])
    return f


def spiele_kurve(toene, kurve_hz, n):
    """Probe so lesen, dass sie genau der Tonhöhen-Kurve folgt (Bends, Vibrato, Slides)."""
    ziel = float(np.nanmedian(kurve_hz))
    f_p, x = min(toene, key=lambda p: abs(np.log2(p[0] / ziel)))
    rate = np.interp(np.arange(n), np.linspace(0, n - 1, len(kurve_hz)), kurve_hz) / f_p
    pos = np.minimum(np.cumsum(rate) - rate[0], len(x) - 2)
    return np.interp(pos, np.arange(len(x)), x)


# ------------------------------------------------------------ Klang
def sf_toene(bank, prog, lo=34, hi=92):
    import sf2_spieler as sp
    import piano_neu_step as P
    ins = sp.SoundFont(P.soundfont_sichern()).instrument(bank, prog)
    r = []
    for k in range(lo, hi, 3):
        z = sp.spielen(ins, [(0.0, 6.0, k, 100)], 6.4, nachklang_max=0.3)
        z = z.mean(1) if z.ndim > 1 else z
        r.append((float(lb().midi_to_hz(k)), z / (np.abs(z).max() + 1e-9)))
    return r


def melodica_toene(lo=43, hi=92):
    """Melodica: zwei leicht verstimmte Zungen, etwas Luft, Bandpass."""
    r = []
    for k in range(lo, hi, 3):
        f = lb().midi_to_hz(k)
        t = np.arange(int(6 * SR)) / SR
        ph = 2 * np.pi * f * t
        z = sum((1.0 / h ** 1.1) * np.sin(h * ph + 0.3 * h) for h in range(1, 18) if h * f < 15000)
        z2 = sum((1.0 / h ** 1.1) * np.sin(h * ph * 1.003 + 0.7 * h) for h in range(1, 18) if h * f < 15000)
        z = bandpass(0.6 * z + 0.4 * z2 + np.random.default_rng(k).standard_normal(len(t)) * 0.02, 150, 9000)
        z = z * np.minimum(1, t / 0.04)
        r.append((float(f), z / np.abs(z).max()))
    return r


def toene_fuer(klang, lo=34, hi=92):
    if klang == "melodica":
        return {"me": melodica_toene(max(lo, 43), hi)}
    brauch = {"satz": ("tp", "as", "ts", "tb"), "trompete": ("tp",), "altsax": ("as",), "tenorsax": ("ts",),
              "posaune": ("tb",)}[klang]
    return {k: sf_toene(0, GM[k][0], lo, hi) for k in brauch}


def besetzung(klang, p, oben):
    """(Instrument, Halbton-Versatz, Pan, Pegel, Verzögerung ms) je Ton."""
    if klang == "satz":
        if oben:
            return [("tp", 0, -0.35, 1.0, 0), ("ts", 0, 0.35, 0.8, 9), ("tb", -12, 0.0, 0.55, 5)]
        if p < 64:
            return [("as", 0, 0.5, 0.75, 6), ("tb", 0, -0.5, 0.7, 11)]
        return [("tp", 0, 0.45, 0.7, 7), ("as", 0, -0.45, 0.7, 12)]
    if klang == "melodica":
        return [("me", 0, 0.1 if oben else -0.3, 1.0 if oben else 0.6, 0)]
    ins = {"trompete": "tp", "altsax": "as", "tenorsax": "ts", "posaune": "tb"}[klang]
    tr = 0
    if ins == "tb" and p > 74:
        tr = -12
    if ins == "tp" and p < 54:
        tr = 12
    if oben:                                   # Doppelung wie zwei Spieler: leicht versetzt, breit
        return [(ins, tr, -0.3, 1.0, 0), (ins, tr + 0.08, 0.3, 0.7, 13)]
    return [(ins, tr, 0.45 if p < 64 else -0.45, 0.7, 7)]


def rendern(V, klang, T, L, env_fuer, kurve_fuer, spuren=None):
    """V: Töne (a, b, p, v). env_fuer(ton, n_frames, aus) -> Hüllkurve (HOP-Raster), kurve_fuer(ton) -> Hz-Kurve|None.
    Rückgabe: Stereo-Mix, Anzahl Töne mit Kurve, MIDI-Ereignisse je Instrument-Rolle.
    Mit spuren={} wird jedes Instrument zusätzlich direkt in seine eigene Spur gespielt (spuren[ins] = Stereo)."""
    L_ = lb()
    mix = np.zeros((L + SR, 2))
    n_kurve = 0
    midi = {}
    for ton in V:
        a, b, p, v = ton
        gleich = [q for q in V if q[0] < b and q[1] > a]
        oben = p >= max(q[2] for q in gleich)
        s, e = int(a * SR / HOP), max(int(a * SR / HOP) + 2, int(b * SR / HOP))
        k = kurve_fuer(ton) if oben else None
        if k is None:
            i = np.arange(e - s)
            vib = 1 + 0.003 * np.sin(2 * np.pi * 5.2 * i * HOP / SR) * np.minimum(1, i / 30)
            k = L_.midi_to_hz(p) * vib
        else:
            n_kurve += 1
        aus = 8
        k = np.r_[k, np.full(aus, k[-1])]
        n = len(k) * HOP
        env = env_fuer(ton, s, e, aus)
        g = np.interp(np.arange(n), np.linspace(0, n - 1, len(env)), env)
        for ins, tr, pan, pg, verz in besetzung(klang, p, oben):
            z = spiele_kurve(T[ins], k * 2 ** (tr / 12), n) * g * pg
            w = (pan + 1) * np.pi / 4
            a0 = s * HOP + int(verz * SR / 1000)
            if a0 + n > len(mix):
                continue
            mix[a0:a0 + n, 0] += z * np.cos(w) * np.sqrt(2)
            mix[a0:a0 + n, 1] += z * np.sin(w) * np.sqrt(2)
            if spuren is not None:
                sp_ = spuren.setdefault(ins, np.zeros((L + SR, 2)))
                sp_[a0:a0 + n, 0] += z * np.cos(w) * np.sqrt(2)
                sp_[a0:a0 + n, 1] += z * np.sin(w) * np.sqrt(2)
            if abs(tr - round(tr)) < 1e-6:     # Doppelung (leicht verstimmt) nicht ins MIDI
                rolle = (ins, "oben" if oben else "satz", int(tr))
                midi.setdefault(rolle, []).append((a, b, int(p + tr), float(np.percentile(env, 90)),
                                                   k if oben else None))
    if spuren is not None:
        for k in spuren:
            spuren[k] = spuren[k][:L]
    return mix[:L], n_kurve, midi


def klang_filter(neu, ref, staerke=0.5):
    """EQ-Filter, das die Klangfarbe sanft ans Original bringt (Terzbänder bis 4 kHz); None wenn zu kurz."""
    from scipy.signal import firwin2

    def spek(z):
        S = 0
        for i in range(0, len(z) - 8192, 4096 * 8):
            S = S + np.abs(np.fft.rfft(z[i:i + 8192].mean(1) * np.hanning(8192))) ** 2
        return S
    if len(neu) < 8192 * 4 or len(ref) < 8192 * 4:
        return None
    f = np.fft.rfftfreq(8192, 1 / SR)
    Sr, Sn = spek(ref), spek(neu)
    mitten = 2 ** np.arange(np.log2(50), np.log2(18000), 1 / 3)
    db = np.array([10 * np.log10((Sr[(f >= m / 2 ** (1 / 6)) & (f < m * 2 ** (1 / 6))].sum() + 1e-12) /
                                 (Sn[(f >= m / 2 ** (1 / 6)) & (f < m * 2 ** (1 / 6))].sum() + 1e-12)) for m in mitten])
    db = np.clip(np.convolve(db, np.ones(3) / 3, "same") * staerke, -9, 6)
    db[mitten < 200] = np.maximum(db[mitten < 200], -4)
    db[mitten > 4000] = 0.0
    db -= np.median(db[(mitten > 300) & (mitten < 3000)])
    return firwin2(2047, np.r_[0, mitten, SR / 2] / (SR / 2), 10 ** (np.r_[db[0], db, 0] / 20))


def veredeln(mix, ref=None, spuren=None):
    """Klangfarbe ans Original, etwas Sättigung, kleiner Raum.
    Mit spuren: jede Instrument-Spur wird einzeln genauso behandelt (gleiches EQ, gleiche Sättigungs-Schwelle,
    eigener Raum); der Mix ist dann genau die Summe der Einzelspuren."""
    from scipy.signal import fftconvolve
    L = len(mix)
    h = klang_filter(mix, ref, staerke=0.6) if ref is not None else None
    m = np.abs(mix).max() + 1e-9
    rng = np.random.default_rng(2)
    kk = int(0.6 * SR)
    tt = np.arange(kk) / SR
    irs = []
    for c in range(2):
        ir = rng.standard_normal(kk) * np.exp(-tt / 0.15)
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))

    def behandeln(x):
        if h is not None:
            x = np.stack([fftconvolve(x[:, c], h, "same") for c in range(2)], 1)
        x = np.tanh(1.6 * x / m) * m
        for c in range(2):
            x[:, c] += 0.18 * fftconvolve(x[:, c], irs[c])[:L]
        return x
    if not spuren:
        return behandeln(mix)
    for k in spuren:
        spuren[k] = behandeln(spuren[k])
    return sum(spuren.values())


# ------------------------------------------------------------ Analyse
def pegel_regeln(y):
    """Langsame Pegel-Regelung, damit leise Bläser-Stellen auch erkannt werden."""
    L_ = lb()
    r = L_.feature.rms(y=y, frame_length=4096, hop_length=1024)[0]
    r = np.convolve(r, np.ones(43) / 43, "same")
    ziel = np.percentile(r, 95) + 1e-9
    g = np.clip(ziel / (r + 0.02 * ziel), 1.0, 20.0)
    g = np.interp(np.arange(len(y)), np.arange(len(g)) * 1024, g)
    z = y * g
    return z * (0.9 / (np.abs(z).max() + 1e-9))


def noten_erkennen(pfad, ziel):
    import piano_neu_step as P
    noten, _, _ = P.noten_basic_pitch(pfad, ziel)
    return [(float(a), float(b), int(p), float(v)) for a, b, p, v in noten]


def gm_name(p):
    return f"{K.NAMEN[p % 12]}{p // 12 - 1}"


def midi_schreiben(rollen, ziel, bpm=120.0):
    """Eine Spur je Instrument-Rolle (eigener Kanal), Oberstimmen mit Pitch-Bend (±12 Halbtöne)."""
    L_ = lb()
    ppq = 960
    tempo = float(bpm) if bpm and bpm > 0 else 120.0
    tick = lambda s: int(round(s * tempo / 60.0 * ppq))  # noqa: E731

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
    spuren = [b"\x00\xff\x51\x03" + usek.to_bytes(3, "big") + b"\x00\xff\x58\x04\x04\x02\x18\x08\x00\xff\x2f\x00"]
    alle = [x for v in rollen.values() for x in v]
    pmax = max([x[3] for x in alle] + [1e-9])
    reihen = sorted(rollen.items(), key=lambda kv: (kv[0][1] != "oben", kv[0][0]))
    for kanal, ((ins, art, tr), noten) in enumerate(reihen[:15]):
        ch = kanal if kanal < 9 else kanal + 1             # Kanal 10 (Schlagzeug) auslassen
        prog, name = GM[ins]
        name = f"{name} - {'Oberstimme' if art == 'oben' else 'Satzstimmen'}" + (f" ({tr:+d} HT)" if tr else "")
        nb = name.encode("utf-8")
        ev = []
        for a, b, p, v, k in noten:
            vel = int(np.clip(45 + 80 * (v / pmax) ** 0.7, 30, 125))
            if k is not None:
                ev.append((max(0, tick(a) - 1), 0, bytes([0xE0 | ch, 0x00, 0x40])))
            ev.append((tick(a), 1, bytes([0x90 | ch, p, vel])))
            if k is not None:
                m = L_.hz_to_midi(np.asarray(k)) + tr
                letzt = 8192
                for q in range(0, len(m), 4):
                    w = int(np.clip(8192 + (m[q] - p) / 12.0 * 8191, 0, 16383))
                    if abs(w - letzt) >= 40:
                        ev.append((tick(a + q * HOP / SR), 2, bytes([0xE0 | ch, w & 0x7F, (w >> 7) & 0x7F])))
                        letzt = w
            ev.append((max(tick(b), tick(a) + 1), 3, bytes([0x80 | ch, p, 0])))
        ev.sort(key=lambda x: (x[0], x[1]))
        d = b"\x00\xff\x03" + vlq(len(nb)) + nb + b"\x00" + bytes([0xC0 | ch, prog])
        d += bytes([0x00, 0xB0 | ch, 0x65, 0x00, 0x00, 0xB0 | ch, 0x64, 0x00, 0x00, 0xB0 | ch, 0x06, 0x0C,
                    0x00, 0xB0 | ch, 0x26, 0x00, 0x00, 0xB0 | ch, 0x65, 0x7F, 0x00, 0xB0 | ch, 0x64, 0x7F])
        letzt = 0
        for tk, _, bb in ev:
            d += vlq(max(0, tk - letzt)) + bb
            letzt = max(letzt, tk)
        d += b"\x00\xff\x2f\x00"
        spuren.append(d)
    p = os.path.join(ziel, "blaeser_neu.mid")
    with open(p, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(spuren), ppq) + b"".join(spur(s) for s in spuren))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe", help="Bläser-Spur (getrennt)")
    ap.add_argument("ziel")
    ap.add_argument("--song", default="", help="ganzer Song (für den Song-Mix und zum Mithören)")
    ap.add_argument("--klang", default="satz")
    ap.add_argument("--nicht-mithoeren", action="store_true", help="vollen Song nicht zusätzlich anhören")
    ap.add_argument("--midi-bpm", type=float, default=120.0)
    ap.add_argument("--modelle", default=os.path.expanduser("~/HELTECH_Stems/models/"))
    ap.add_argument("--spuren-ordner", default=None)
    a = ap.parse_args()
    os.makedirs(a.ziel, exist_ok=True)
    try:
        L_ = lb()
        klang = a.klang if a.klang in KLAENGE else "satz"
        raw = stereo(L_.load(a.eingabe, sr=SR, mono=False)[0])
        yr = raw.mean(1)
        L = len(raw)
        rms_r = L_.feature.rms(y=yr, frame_length=2048, hop_length=HOP)[0]
        akt = rms_r > 0.03 * np.percentile(rms_r, 99)
        # 1) leise Stellen anheben, alle Stimmen als Noten, Tonhöhen-Kurve der Oberstimme
        melde("… höre die Bläser-Spur an (alle Stimmen und die Tonhöhen-Kurve)")
        ya = pegel_regeln(yr)
        agc = os.path.join(a.ziel, "_blaeser_geregelt.wav")
        sf.write(agc, ya, SR, subtype="PCM_16")
        N = noten_erkennen(agc, a.ziel)
        f0, _, _ = L_.pyin(ya, fmin=110, fmax=1400, sr=SR, frame_length=2048, hop_length=HOP)

        def aktiv(x0, x1):
            i, j = int(x0 * SR / HOP), max(int(x0 * SR / HOP) + 1, int(x1 * SR / HOP))
            return akt[i:j].mean() if j <= len(akt) else 0
        N = [x for x in N if 45 <= x[2] <= 88 and x[1] - x[0] >= 0.07 and aktiv(x[0], x[1]) >= 0.5]
        N.sort()
        V = []
        for x0, x1, p, v in N:                       # gleiche Töne mit kleiner Lücke verbinden
            k = next((k for k in range(len(V) - 1, max(-1, len(V) - 8), -1)
                      if V[k][2] == p and 0 <= x0 - V[k][1] < 0.05), None)
            if k is not None:
                V[k] = (V[k][0], max(x1, V[k][1]), p, max(v, V[k][3]))
                continue
            V.append((x0, x1, p, v))
        akt_orig = akt.copy()
        # 2) vollen Song mithören: was hat das Trennmodell verpasst? Tonhöhen am vollen Klang prüfen
        ergaenzt, korr, zeiten, Cq, modell = 0, 0, [], None, "keine"
        if a.song and not a.nicht_mithoeren:
            melde("… höre den vollen Song mit (trenne ihn in 6 Spuren, dauert einige Minuten)")
            if a.spuren_ordner:
                spuren = {s: glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))[0]
                          for s in K.STIMMEN if glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))}
                modell = "vorhanden"
            else:
                spuren, modell = K.trennen(a.song, a.ziel, a.modelle)
            if "other" in spuren:
                oth = K.lade_mono(spuren["other"])[:L]
                oth = np.r_[oth, np.zeros(L - len(oth))]
                Cq = np.abs(L_.cqt(oth + 0.5 * yr, sr=SR, hop_length=512, fmin=L_.midi_to_hz(40), n_bins=56))
                Cq = np.log1p(Cq / (np.percentile(Cq, 99) + 1e-12) * 50)
                beats = K.taktraster(K.lade_mono(spuren["drums"])) if "drums" in spuren else []
                if len(beats) >= 12:
                    ON = noten_erkennen(spuren["other"], a.ziel)
                    takte = [(beats[i], beats[min(i + 4, len(beats) - 1)]) for i in range(0, len(beats) - 4, 4)]
                    F = np.array([Cq[8:, int(x0 * SR / 512):max(int(x0 * SR / 512) + 1, int(x1 * SR / 512))].mean(1)
                                  for x0, x1 in takte])
                    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
                    E = np.array([np.sqrt(np.mean(yr[int(x0 * SR):int(x1 * SR)] ** 2) + 1e-20) for x0, x1 in takte])
                    E /= np.percentile(E, 95) + 1e-12
                    lang = lambda x0, x1: [x for x in ON if x0 <= x[0] < x1 and x[1] - x[0] >= 0.15 and 58 <= x[2] <= 88]  # noqa: E731
                    horn = E > 0.3
                    if horn.any():
                        dichte = np.median([len(lang(*takte[i])) for i in np.flatnonzero(horn)])
                        S = np.corrcoef(F)
                        for i, (x0, x1) in enumerate(takte):
                            if horn[i] or E[i] > 0.15:
                                continue
                            j = int(np.argmax(np.where(horn, S[i], -9)))
                            kand = lang(x0, x1)
                            if S[i, j] >= 0.5 and len(kand) >= max(1, 0.5 * dichte):
                                for x in kand:
                                    V.append((x[0], x[1], int(x[2]), -1.0))
                                    ergaenzt += 1
                                zeiten.append(x0)
                                akt[int(x0 * SR / HOP):int(x1 * SR / HOP)] = True
                    V.sort()

                def salienz(p, x0, x1):
                    i = int(x0 * SR / 512)
                    j = max(i + 1, int(x1 * SR / 512))
                    s_ = 0
                    for h, w in ((0, 1.0), (12, 0.6), (19, 0.4), (24, 0.3)):
                        k = p - 40 + h
                        if 0 <= k < Cq.shape[0]:
                            s_ += w * Cq[k, i:j].mean()
                    return s_
                V2 = []
                for x0, x1, p, v in V:
                    best = max((p, p - 1, p + 1, p - 12, p + 12),
                               key=lambda q: salienz(q, x0, x1) * (1.0 if q == p else 0.85))
                    korr += best != p
                    V2.append((x0, x1, best, v))
                V = V2
            if zeiten:
                melde(f"✔ vom vollen Song ergänzt: {ergaenzt} Töne (um "
                      + ", ".join(f"{int(t // 60)}:{int(t % 60):02d}" for t in zeiten[:12]) + ")")
            melde(f"✔ Tonhöhen am vollen Song geprüft: {korr} Töne korrigiert")
        blaeser_s = float(akt.sum() * HOP / SR)
        if len(V) < 4 or blaeser_s < 2.0:
            bericht = {"art": "keine_blaeser", "sekunden": round(blaeser_s, 1), "trennung": modell}
            json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
            melde("✔ In dieser Spur spielen keine Bläser / keine Melodica – nichts geändert")
            print(json.dumps({"ok": True, "datei": "", "ordner": a.ziel, "art": "keine_blaeser"}, ensure_ascii=False))
            return
        melde(f"✔ {len(V)} Töne gefunden – spiele neu: {KLAENGE[klang]}")
        # 3) neu spielen
        T = toene_fuer(klang)
        pegel_h = float(np.median(rms_r[akt_orig])) if akt_orig.any() else 0.05

        def env_fuer(ton, s, e, aus):
            if ton[3] >= 0 or Cq is None:
                env = np.r_[rms_r[s:e], np.zeros(aus)]
            else:                                     # ergänzter Ton: Verlauf aus dem vollen Song an seiner Tonhöhe
                k = Cq[min(max(ton[2] - 40, 0), Cq.shape[0] - 1)]
                i = np.clip((np.arange(s, e) * HOP / 512).astype(int), 0, len(k) - 1)
                c = k[i] / (k[i].max() + 1e-9)
                env = np.r_[pegel_h * np.clip(c, 0.3, 1.0), np.zeros(aus)]
            env = np.convolve(env, np.ones(5) / 5, "same")
            env[-aus:] = np.linspace(env[-aus - 1] if len(env) > aus else 0, 0, aus)
            return env

        def kurve_fuer(ton):
            x0, x1, p, _ = ton
            s, e = int(x0 * SR / HOP), max(int(x0 * SR / HOP) + 2, int(x1 * SR / HOP))
            k = f0[s:e]
            m = L_.hz_to_midi(k)
            ok = np.isfinite(m) & (np.abs(m - p) < 1.0)
            if len(ok) and ok.mean() > 0.5:
                return kurve_fuellen(np.where(ok, k, np.nan))
            return None
        einzel = {}
        mix, n_kurve, rollen = rendern(V, klang, T, L, env_fuer, kurve_fuer, einzel)
        mix = veredeln(mix, raw, einzel)
        m_ = np.repeat(akt, HOP)[:L]
        m_ = np.r_[m_, np.zeros(L - len(m_), bool)]
        ref = np.sqrt(np.mean(yr[m_ & akt_orig.repeat(HOP)[:L].astype(bool)] ** 2)) if akt_orig.any() else 0.05
        fak = ref / (np.sqrt(np.mean(mix.mean(1)[m_] ** 2)) + 1e-12)
        if np.abs(mix).max() * fak > 0.98:
            fak = 0.98 / np.abs(mix).max()
        mix *= fak
        out = os.path.join(a.ziel, "blaeser_neu.wav")
        sf.write(out, mix, SR, subtype="PCM_24")
        # jedes Instrument in seiner eigenen Spur (direkt so gespielt, nicht nachträglich getrennt)
        if len(einzel) > 1:
            eo = os.path.join(a.ziel, "Einzelspuren")
            os.makedirs(eo, exist_ok=True)
            for nr, ins in enumerate([k for k in ("tp", "ts", "as", "tb", "me") if k in einzel], 1):
                sf.write(os.path.join(eo, f"{nr}_{SPURNAME[ins]}.wav"), einzel[ins] * fak, SR, subtype="PCM_24")
        if a.song:
            song = stereo(L_.load(a.song, sr=SR, mono=False)[0])
            n = min(len(song), L)
            neu = song[:n] - raw[:n] + mix[:n]
            neu *= min(1.0, 0.98 / max(np.abs(neu).max(), 1e-12))
            sf.write(os.path.join(a.ziel, "song_mit_blaesern_neu.wav"), neu, SR, subtype="PCM_24")
        midi = midi_schreiben(rollen, a.ziel, a.midi_bpm)
        txt = [f"Bläser / Melodica – neu gespielt: {KLAENGE[klang]}",
               f"MIDI-Datei mit {a.midi_bpm:g} BPM: Logic-Projekt auf {a.midi_bpm:g} BPM stellen.",
               "Im MIDI: eine Spur je Instrument (Oberstimme mit Pitch-Bend ±12 Halbtöne für Bends/Vibrato, "
               "Satzstimmen ohne), Programmwechsel nach General MIDI.",
               f"Töne: {len(V)}, davon {ergaenzt} vom vollen Song ergänzt, {korr} Tonhöhen am vollen Song korrigiert.", "",
               "Zeit       Dauer   Ton    Herkunft"]
        for x0, x1, p, v in V:
            txt.append(f"{int(x0 // 60)}:{x0 % 60:06.3f}  {x1 - x0:5.2f}s  {gm_name(p):<5}  "
                       + ("voller Song" if v < 0 else "Bläser-Spur"))
        with open(os.path.join(a.ziel, "blaeser_neu_Notenliste.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(txt) + "\n")
        bericht = {"art": "repariert", "trennung": modell, "klang": KLAENGE[klang], "toene": len(V),
                   "ergaenzt": ergaenzt, "korrigiert": int(korr), "mit_kurve": n_kurve,
                   "sekunden": round(blaeser_s, 1), "midi": midi}
        json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
        import shutil
        shutil.rmtree(os.path.join(a.ziel, "spuren_roh"), ignore_errors=True)
        for f in glob.glob(os.path.join(a.ziel, "_*")):
            os.remove(f)
        print(json.dumps({"ok": True, "datei": out, "ordner": a.ziel, "art": "repariert"}, ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "fehler": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
