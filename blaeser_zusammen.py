#!/usr/bin/env python3
"""HIELTECH Stems – Bläser zusammen: sauber, Ausklang bis zum Ende, Lücken gefüllt (29. Okt 2026).
Hörtest: brass.flac am besten, aber Hall abgeschnitten (Zeitmaske schnitt vor Hall-Ende), Rauschen,
Lücken zwischen den Wellen. Reihenfolge wie gewünscht: ERST entrauschen, DANN Ausklang/Hall.
  1. Quelle = neu erzeugte Bläser (A_roh, enthält den ganzen Ausklang); Vorlage = 53er-Trennung 'brass'.
  2. Entrauschen: Rauschabdruck aus Stellen, wo die Bläser im Original länger schweigen -> je Frequenz abziehen.
  3. Zeitmaske mit natürlichem Ausklingen statt hartem Schnitt (Hall klingt bis zum Ende aus).
  4. Ausklang halten wie beim Becken: Einbrüche zwischen den Wellen auffüllen, Resonanz bleibt stehen.
  5. Variante mit zusätzlichem Raum-Hall (nach dem Entrauschen).
Aufruf: python blaeser_zusammen.py <brass_A_roh> <vorlage_brass> <ausgabeordner>"""
import os, sys
import numpy as np, soundfile as sf, librosa

NFFT, HOP = 4096, 512

def frame_db(m, hop=HOP):
    n = int(np.ceil(len(m) / hop)); y = np.pad(m, (0, n * hop - len(m)))
    return 20 * np.log10(np.sqrt((y.reshape(n, hop) ** 2).mean(1) + 1e-12))

def aktivitaet(ref, sr):
    db = frame_db(ref.mean(1))
    return db > db.max() - 50

def entrauschen(X, ruhig):
    mag = np.abs(X)
    if ruhig.sum() < 50:
        prof = np.percentile(mag, 10, axis=1, keepdims=True)
    else:
        prof = np.median(mag[:, ruhig], axis=1, keepdims=True)
    G = np.clip(1 - 1.5 * prof / (mag + 1e-9), 0.03, 1.0)
    G = librosa.decompose.nn_filter(G, aggregate=np.median, metric="cosine", width=3) if False else \
        np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, G)
    return X * G

def ausklang_maske(akt, sr, t60):
    d = 10 ** (-3.0 * HOP / sr / t60)
    g = np.zeros(len(akt)); cur = 0.0
    for i, a in enumerate(akt):
        cur = 1.0 if a else cur * d
        g[i] = cur
    return g

def t60_blaeser(freq, faktor):
    f = np.maximum(freq, 1.0)
    return np.interp(np.log10(f), np.log10([150, 1000, 4000, 12000]), [1.6, 1.3, 0.9, 0.6]) * faktor

def halten(X, sr, faktor):
    mag = np.abs(X)
    freq = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    g = 10 ** (-3.0 * HOP / sr / t60_blaeser(freq, faktor))
    aktiv = np.clip((freq - 60) / 60, 0, 1)
    adv = 2 * np.pi * np.arange(len(freq)) * HOP / NFFT
    rng = np.random.default_rng(7)
    held = np.zeros(len(freq)); ph = np.angle(X[:, 0]); out = X.copy()
    for t in range(X.shape[1]):
        held = np.maximum(mag[:, t], held * g)
        ph = np.where(mag[:, t] >= 0.7 * held, np.angle(X[:, t]), ph + adv + rng.uniform(-0.6, 0.6, len(freq)))
        out[:, t] = X[:, t] + (held - mag[:, t]) * aktiv * np.exp(1j * ph)
    return out

def raumhall(y, sr, sek=1.8, anteil=0.18):
    n = int(sek * sr); t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / sek)[:, None]
    from scipy.signal import butter, sosfilt, fftconvolve
    ir = sosfilt(butter(2, [200, 7000], "band", fs=sr, output="sos"), ir, axis=0)
    ir[: int(0.02 * sr)] = 0
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
    nass = np.stack([fftconvolve(y[:, c], ir[:, c])[: len(y)] for c in range(2)], 1)
    nass *= np.sqrt((y ** 2).mean()) / (np.sqrt((nass ** 2).mean()) + 1e-12)
    return y + anteil * nass

def schutz(x):
    p = float(np.abs(x).max()) or 1.0
    return x / max(p / 0.98, 1.0)

def verarbeite(a, ref, sr, faktor):
    akt = aktivitaet(ref, sr)
    # ruhig = Original schweigt mindestens 3 s lang (dort ist in der neuen Spur nur Rauschen/Reste)
    k = int(3 * sr / HOP); naehe = np.convolve(akt.astype(float), np.ones(2 * k + 1), mode="same") > 0
    out = np.zeros_like(a)
    for c in range(a.shape[1]):
        X = librosa.stft(a[:, c], n_fft=NFFT, hop_length=HOP)
        T = X.shape[1]
        ak = np.resize(akt, T) if len(akt) >= T else np.pad(akt, (0, T - len(akt)))
        ru = ~(np.resize(naehe, T) if len(naehe) >= T else np.pad(naehe, (0, T - len(naehe))))
        X = entrauschen(X, ru)
        X = X * ausklang_maske(ak, sr, 3.0)[None, :]
        X = halten(X, sr, faktor)
        out[:, c] = librosa.istft(X, hop_length=HOP, length=len(a))
    return out

if __name__ == "__main__":
    src, refp, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    a, sr = sf.read(src, always_2d=True, dtype="float64")
    ref, _ = sf.read(refp, always_2d=True, dtype="float64")
    n = min(len(a), len(ref)); a, ref = a[:n], ref[:n]
    neu = verarbeite(a, ref, sr, 1.0)
    sf.write(os.path.join(outdir, "2_brass_NEU.wav"), schutz(neu), sr, subtype="PCM_24")
    lang = verarbeite(a, ref, sr, 1.6)
    sf.write(os.path.join(outdir, "3_brass_NEU_lang.wav"), schutz(lang), sr, subtype="PCM_24")
    sf.write(os.path.join(outdir, "4_brass_NEU_hall.wav"), schutz(raumhall(neu, sr)), sr, subtype="PCM_24")
