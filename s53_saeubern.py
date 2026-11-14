"""53er-Trennung, Fassung 2 (14. Nov 2026): sauberer und vollständiger.

1. Zwei Stufen: Das 53er-Modell trennt jedes Instrument einmal aus dem ganzen Song (A) und einmal nur aus der
   passenden 6-Spuren-Spur (B, z. B. Orgel nur aus „Piano + Rest“). In B stören viel weniger andere Instrumente.
2. Mitteln mit Schiedsrichter: A und B werden im Spektrum gemittelt (zufällige Artefakte heben sich auf); wo sie sich
   stark widersprechen, gilt der kleinere Wert (Übersprechen fliegt raus). Das Hör-Modell hört A und B ab und gibt
   je halbe Sekunde der Fassung mehr Gewicht, die klarer nach dem Instrument klingt.
3. Gegenrechnung: Das Ergebnis wird mit einer weichen Maske aus der Eltern-Spur geholt (Wiener-Filter) – so bleibt
   der echte Klang der Aufnahme (Anschlag, Ausklang) erhalten und es entstehen keine neuen Artefakte.
4. Säubern: In langen Pausen, in denen das Hör-Modell das Instrument sicher nicht hört und die Spur leise ist,
   wird sie ganz still (nur Reste der Trennung).
Sicherung: Liegt das Instrument zu großen Teilen NICHT in der Eltern-Spur (die 6-Spuren-Trennung hat es woanders
einsortiert), bleibt Fassung A – es geht nie Klang verloren."""
import os
import sys

import numpy as np

HIER = os.path.dirname(os.path.abspath(__file__))

# Eltern-Spuren (htdemucs_6s: drums, bass, guitar, piano, vocals, other) je 53er-Instrument
_D, _B, _G, _P, _V, _O = "drums", "bass", "guitar", "piano", "vocals", "other"
ELTERN = {}
for n in ("kick", "snare", "hh", "toms", "ride", "crash", "cymbals", "drums", "drum", "tom"):
    ELTERN[n] = (_D,)
for n in ("percussion", "congas", "bongos", "djembe", "shaker", "tambourine", "timbales", "cowbell", "claps"):
    ELTERN[n] = (_D, _O)
for n in ("bass", "double-bass", "contrabass"):
    ELTERN[n] = (_B,)
for n in ("synth-bass",):
    ELTERN[n] = (_B, _O)
for n in ("guitar", "electric-guitar", "acoustic-guitar", "lead-guitar", "rhythm-guitar", "banjo", "mandolin", "ukulele"):
    ELTERN[n] = (_G, _O)
for n in ("piano", "digital-piano", "keys", "organ", "synth", "harpsichord", "electric-piano", "accordion", "rhodes"):
    ELTERN[n] = (_P, _O)
for n in ("lead-vocal", "back-vocal", "vocals", "choir"):
    ELTERN[n] = (_V,)
for n in ("brass", "trumpet", "trombone", "saxophone", "sax", "horn", "french-horn", "tuba", "flute", "clarinet",
          "oboe", "bassoon", "harmonica", "woodwinds", "violin", "viola", "cello", "bowed_strings", "strings", "harp",
          "melodica", "pan-flute", "recorder"):
    ELTERN[n] = (_O, _P, _G)

for n in ("dobro", "sitar"):
    ELTERN[n] = (_G, _O)
for n in ("bells", "glockenspiel", "marimba", "wind-chimes", "triangle"):
    ELTERN[n] = (_O, _P, _D)
ELTERN["timpani"] = (_D, _O)
ELTERN["vocal"] = (_V,)
for n in ("wind", "woodwind"):
    ELTERN[n] = (_O, _P, _G)

HOER_GRUPPE = {}
for n in ("kick",):
    HOER_GRUPPE[n] = "kick"
for n in ("snare",):
    HOER_GRUPPE[n] = "snare"
for n in ("hh",):
    HOER_GRUPPE[n] = "hihat"
for n in ("ride", "crash", "cymbals"):
    HOER_GRUPPE[n] = "becken"
for n in ("toms", "tom", "drums", "drum"):
    HOER_GRUPPE[n] = "schlagzeug"
for n in ("percussion", "congas", "bongos", "djembe", "shaker", "tambourine", "timbales", "cowbell", "claps"):
    HOER_GRUPPE[n] = "percussion"
for n in ("bass", "double-bass", "contrabass", "synth-bass"):
    HOER_GRUPPE[n] = "bass"
for n in ("guitar", "electric-guitar", "acoustic-guitar", "lead-guitar", "rhythm-guitar", "banjo", "mandolin", "ukulele"):
    HOER_GRUPPE[n] = "gitarre"
for n in ("piano", "digital-piano", "keys", "harpsichord", "electric-piano", "rhodes"):
    HOER_GRUPPE[n] = "piano"
for n in ("dobro", "sitar"):
    HOER_GRUPPE[n] = "gitarre"
for n in ("bells", "glockenspiel", "marimba", "wind-chimes", "triangle", "timpani"):
    HOER_GRUPPE[n] = "percussion"
HOER_GRUPPE["vocal"] = "gesang"
for n in ("wind", "woodwind"):
    HOER_GRUPPE[n] = "blaeser"
HOER_GRUPPE["organ"] = "orgel"
HOER_GRUPPE["synth"] = "synth"
for n in ("lead-vocal", "back-vocal", "vocals", "choir"):
    HOER_GRUPPE[n] = "gesang"
for n in ("brass", "trumpet", "trombone", "saxophone", "sax", "horn", "french-horn", "tuba", "flute", "clarinet",
          "oboe", "bassoon", "harmonica", "woodwinds", "accordion", "melodica", "pan-flute", "recorder"):
    HOER_GRUPPE[n] = "blaeser"
for n in ("violin", "viola", "cello", "bowed_strings", "strings"):
    HOER_GRUPPE[n] = "streicher"

N_FFT, HOP = 4096, 1024


def _stft(x):
    import librosa
    return np.stack([librosa.stft(np.ascontiguousarray(x[c]), n_fft=N_FFT, hop_length=HOP) for c in range(x.shape[0])])


def _istft(X, n):
    import librosa
    return np.stack([librosa.istft(X[c], hop_length=HOP, length=n) for c in range(X.shape[0])]).astype(np.float32)


def _ht():
    for d in (HIER, os.path.join(HIER, "hieltech_reparieren"), os.path.expanduser("~/HELTECH_Stems")):
        if os.path.isfile(os.path.join(d, "hoer_tor.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            import hoer_tor
            return hoer_tor
    return None


def _schiedsrichter(A, B, sr, gruppe, n_frames):
    """Gewicht für B je STFT-Frame aus dem Hör-Modell (0,2 … 0,8; ohne Modell 0,5)."""
    ht = _ht()
    w = np.full(n_frames, 0.5, np.float32)
    if ht is None or gruppe is None:
        return w, None
    try:
        za, zb = ht.zeitleiste(A.T, sr), ht.zeitleiste(B.T, sr)
        if za is None or zb is None or gruppe not in za:
            return w, None
        pa, pb = za[gruppe], zb[gruppe]
        p90 = float(np.percentile(np.maximum(pa, pb), 90))
        if p90 < 0.08:
            return w, {"stand": "Hör-Modell unsicher – gleich gewichtet"}
        wb = np.clip(0.5 + (pb - pa) / (2 * p90), 0.2, 0.8)
        tf = np.arange(n_frames) * HOP / sr
        w = np.interp(tf, za["t"], wb).astype(np.float32)
        return w, {"anteil_B": round(float(w.mean()), 2)}
    except Exception as e:  # noqa: BLE001
        return w, {"stand": f"übersprungen ({str(e)[:80]})"}


def verbessern(name, A, B, P, sr, SP=None):
    """A: 53er aus dem Song, B: 53er aus der Eltern-Spur P (alle (2, n)). Rückgabe (y, bericht)."""
    n = A.shape[1]
    ber = {"stufen": "A+B"}
    ea = float(np.mean(A ** 2)) + 1e-12
    eb = float(np.mean(B ** 2)) + 1e-12
    ber["energie_B_zu_A_db"] = round(float(10 * np.log10(eb / ea)), 1)
    if eb < 0.35 * ea:                 # Instrument liegt großenteils außerhalb der Eltern-Spur -> nichts verlieren
        ber["stufen"] = "nur A (Instrument liegt nicht in der passenden 6er-Spur)"
        return A, ber
    SA, SB = _stft(A), _stft(B)
    SP = _stft(P) if SP is None else SP
    mA, mB = np.abs(SA), np.abs(SB)
    w, ri = _schiedsrichter(A, B, sr, HOER_GRUPPE.get(name), SA.shape[-1])
    if ri:
        ber["schiedsrichter"] = ri
    w = w[None, None, :]
    mittel = w * mB + (1 - w) * mA
    einig = np.abs(20 * np.log10((mA + 1e-9) / (mB + 1e-9))) < 6.0
    M = np.where(einig, mittel, np.minimum(mA, mB))                # Widerspruch: der kleinere Wert (kein Übersprechen)
    # Gegenrechnung: weiche Maske aus der Eltern-Spur (Wiener) – echter Klang der Aufnahme, keine neuen Artefakte
    rest = np.maximum(np.abs(SP) - M, 0.0)
    maske = M ** 2 / (M ** 2 + rest ** 2 + 1e-10)
    Y = SP * maske
    ber["wiener"] = True
    ber["einig_anteil"] = round(float(einig.mean()), 2)
    return _istft(Y, n), ber


def saeubern(name, y, sr):
    """Hör-Modell-Tor: lange, sichere Pausen ohne das Instrument still. y (2, n)."""
    ht = _ht()
    g = HOER_GRUPPE.get(name)
    if ht is None or g is None:
        return y, {"stand": "kein Hör-Modell für dieses Instrument"}
    b = {}
    out = ht.tor(y.T, y.T, sr, g, b)
    return np.ascontiguousarray(np.asarray(out, dtype=np.float32).T), b.get("hoer_modell", {}).get(g, {})
