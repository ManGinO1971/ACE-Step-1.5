#!/usr/bin/env python3
"""HIELTECH Stems – Hi-Hat aus der Djembe-Percussion entfernen (30. Okt 2026).
Hörtest: 7_percussion_nur_djembe "genau so", nur Hi-Hat noch drin.
Hi-Hat lebt in den Höhen; der Djembe-Körper darunter bleibt UNANGETASTET.
  - unter 3 kHz: nichts ändern
  - darüber (weich ab 3 kHz, voll ab 5 kHz): je Zeit/Frequenz Hi-Hat-Vorlage (53er hh) gegen Djembe/Percussion-Vorlage;
    wo die Hi-Hat stärker ist -> raus (weiche Maske, geglättet). Djembe-Slap-Glanz bleibt, wo die Djembe dominiert.
Aufruf: python hihat_raus.py <percussion> <getrennt_ordner> <ref53_ordner> <ausgabe.wav> [staerke=2.0]"""
import sys
import numpy as np, soundfile as sf, librosa
from scipy.ndimage import uniform_filter

NFFT, HOP = 2048, 256

def lade(p, n):
    a, sr = sf.read(p, always_2d=True, dtype="float32")
    return (a[:n] if len(a) >= n else np.pad(a, ((0, n - len(a)), (0, 0)))), sr

def mag(x):
    return np.abs(librosa.stft(x.mean(1), n_fft=NFFT, hop_length=HOP))

def hihat_raus(x, sr, D, H, staerke=2.0):
    n = len(x)
    Md, Mh = uniform_filter(mag(D), size=(5, 5)), uniform_filter(mag(H), size=(5, 5))
    m = Md ** 2 / (Md ** 2 + staerke * Mh ** 2 + 1e-12)
    m = uniform_filter(m, size=(3, 3))
    freq = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    w = np.clip((freq - 3000) / 2000, 0, 1)[:, None]          # 0 unter 3 kHz, 1 ab 5 kHz
    m = 1 - w * (1 - m)
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        X = librosa.stft(x[:, c], n_fft=NFFT, hop_length=HOP); T = min(X.shape[1], m.shape[1])
        out[:, c] = librosa.istft(X[:, :T] * m[:, :T], hop_length=HOP, length=n)
    return out

if __name__ == "__main__":
    src, G, R, dst = sys.argv[1:5]
    st = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    x, sr = sf.read(src, always_2d=True, dtype="float32"); n = len(x)
    D = lade(f"{G}/congas.flac", n)[0] + lade(f"{G}/percussion.flac", n)[0]
    H = lade(f"{R}/hh.flac", n)[0]
    sf.write(dst, hihat_raus(x, sr, D, H, st), sr, subtype="PCM_24")


def hihat_raus_klang(x, sr, ab_hz=5000.0, glanz_s=0.04):
    """Rein nach Klang (für Hi-Hats, die die 53er-hh-Vorlage nicht kennt): Djembe hat über ~5 kHz nur den
    kurzen Glanz des Slap-Anschlags. Höhen bleiben deshalb nur ~40 ms nach einem Djembe-Anschlag, dessen
    Mitten (1–4 kHz) stärker sind als seine Höhen; alles andere über 5 kHz (Hi-Hat, Shaker-Zischen) raus."""
    n = len(x)
    X0 = librosa.stft(x.mean(1), n_fft=NFFT, hop_length=HOP); P = np.abs(X0) ** 2
    freq = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    mitte = P[(freq >= 1000) & (freq < 4000)].sum(0); hoch = P[freq >= 6000].sum(0)
    on = librosa.onset.onset_detect(y=x.mean(1), sr=sr, hop_length=HOP, delta=0.05, backtrack=True)
    T = P.shape[1]; erlaubt = np.zeros(T)
    L = max(1, int(glanz_s * sr / HOP))
    for o in on:
        a, b = o, min(T, o + L)
        if hoch[a:b].sum() < 0.5 * mitte[a:b].sum():      # Djembe-Slap: Mitten führen
            erlaubt[a:b] = np.maximum(erlaubt[a:b], np.linspace(1, 0, b - a))
    w = np.clip((freq - (ab_hz - 1500)) / 1500, 0, 1)[:, None]   # weich ab 3,5 kHz, voll ab 5 kHz
    m = 1 - w * (1 - erlaubt[None, :])
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        X = librosa.stft(x[:, c], n_fft=NFFT, hop_length=HOP); Tm = min(X.shape[1], T)
        out[:, c] = librosa.istft(X[:, :Tm] * m[:, :Tm], hop_length=HOP, length=n)
    return out
