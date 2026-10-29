#!/usr/bin/env python3
"""HIELTECH Stems – Becken-Ausklang stabilisieren (29. Okt 2026).
KI-/Trenn-Becken haben kurze Einbrüche im Ausklang. Wie ein echtes Becken darf der Ausklang nur natürlich
abklingen: je Frequenz wird der Pegel nach einem Schlag gehalten und klingt mit natürlicher Nachhallzeit ab;
nur wo das Original einbricht, wird der fehlende Anteil ergänzt (mit weiterlaufender Phase) – was da ist, bleibt.
Aufruf: python becken_halt.py <ein.flac> <aus.wav> [faktor_ausklang=1.0]"""
import sys
import numpy as np
import soundfile as sf
import librosa

def t60(freq, faktor):
    # Ausklangzeit je Frequenz (s): tiefe Becken-Anteile länger, ganz oben kürzer
    f = np.maximum(freq, 1.0)
    t = np.interp(np.log10(f), np.log10([500, 3000, 8000, 16000]), [3.5, 2.8, 2.0, 1.2])
    return t * faktor

def halten(y, sr, faktor):
    nfft, hop = 4096, 512
    X = librosa.stft(y, n_fft=nfft, hop_length=hop)
    mag = np.abs(X)
    freq = librosa.fft_frequencies(sr=sr, n_fft=nfft)
    g = 10 ** (-3.0 * hop / sr / t60(freq, faktor))          # -60 dB in T60
    aktiv = np.clip((freq - 1000) / 500, 0, 1)                # erst ab ~1,5 kHz (keine Snare-/Tom-Reste verlängern)
    adv = 2 * np.pi * np.arange(len(freq)) * hop / nfft
    rng = np.random.default_rng(7)
    held = np.zeros(len(freq)); ph = np.angle(X[:, 0])
    out = X.copy()
    for t in range(X.shape[1]):
        held = np.maximum(mag[:, t], held * g)
        ph = np.where(mag[:, t] >= 0.7 * held, np.angle(X[:, t]), ph + adv + rng.uniform(-0.6, 0.6, len(freq)))
        fehlt = (held - mag[:, t]) * aktiv
        out[:, t] = X[:, t] + fehlt * np.exp(1j * ph)
    return librosa.istft(out, hop_length=hop, length=len(y))

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    faktor = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    a, sr = sf.read(src, always_2d=True, dtype="float32")
    o = np.stack([halten(a[:, c], sr, faktor) for c in range(a.shape[1])], 1)
    p = float(np.abs(o).max())
    if p > 0.98:
        o *= 0.98 / p
    sf.write(dst, o, sr, subtype="PCM_24")
