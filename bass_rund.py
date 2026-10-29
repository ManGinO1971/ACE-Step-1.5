#!/usr/bin/env python3
"""HIELTECH Stems – Bass rund, fett, sauber (29. Okt 2026), Ausgang: bass_A_neu_roh.
Hörtest: in A ein metallischer Ton (Trenn-/KI-Rest) -> Bass klingt heller, nicht mehr dick/tief/rund.
Ein echter Bass besteht nur aus Grundton + Obertönen (ganzzahlige Vielfache). Metallische Reste liegen
DAZWISCHEN. Daher:
  V1 harmonisch gereinigt: nur Grundton + Obertöne behalten (Tonhöhe je 12 ms), alles dazwischen raus.
  V2 V1 + fett: sauberer Sub-Grundton darunter, warme Röhren-Sättigung (gerade Obertöne), Tiefen leicht angehoben.
  V3 neu aufgebaut: Bass komplett aus Grundton + 12 Obertönen neu gebaut (Lautstärke je Oberton aus A) – wie DI.
Aufruf: python bass_rund.py <bass_A.flac> <ausgabeordner>"""
import os, sys
import numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfiltfilt

NFFT, HOP = 8192, 512

def tonhoehe(mono, sr):
    sr2, hop2 = 2205, 32
    y = librosa.resample(mono, orig_sr=sr, target_sr=sr2)
    f0, v, _ = librosa.pyin(y, fmin=30, fmax=330, sr=sr2, frame_length=512, hop_length=hop2)
    f0 = np.where(v & np.isfinite(f0), f0, 0.0)
    return f0, np.arange(len(f0)) * hop2 / sr2

def f0_auf(t_src, f0, t_dst):
    vm = f0 > 0
    if vm.sum() < 10:
        return np.zeros_like(t_dst), np.zeros_like(t_dst)
    lf = np.interp(t_dst, t_src[vm], np.log(f0[vm]))
    stimm = np.interp(t_dst, t_src, vm.astype(float))
    return np.exp(lf), stimm

def harmonisch(x, sr, f0, tf):
    """-> gereinigtes Signal (n, ch)"""
    freq = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        X = librosa.stft(x[:, c], n_fft=NFFT, hop_length=HOP)
        tfr = np.arange(X.shape[1]) * HOP / sr
        ff, st = f0_auf(tf, f0, tfr)
        M = np.zeros(X.shape, dtype=np.float32)
        for k in range(1, 41):
            fk = k * ff                                   # (frames,)
            sig = np.maximum(6.0, 0.025 * fk)
            g = np.exp(-0.5 * ((freq[:, None] - fk[None, :]) / sig[None, :]) ** 2)
            g[:, fk > 4000] = 0
            M = np.maximum(M, g)
        M = M * st[None, :] + (1 - st[None, :]) * (freq[:, None] < 120)   # ohne Ton: nur Tiefen
        M = np.maximum(M, 0.04)                                           # Rest -28 dB (natürlich, nicht steril)
        out[:, c] = librosa.istft(X * M, hop_length=HOP, length=len(x))
    return out

def fett(x, sr, f0, tf):
    mono = x.mean(1)
    t = np.arange(len(mono)) / sr
    ff, st = f0_auf(tf, f0, t)
    ph = 2 * np.pi * np.cumsum(ff) / sr
    tief = sosfiltfilt(butter(4, 150, "low", fs=sr, output="sos"), mono)
    k = int(0.03 * sr)
    hull = np.sqrt(np.convolve(tief ** 2, np.ones(k) / k, mode="same")) * np.sqrt(2)
    sub = np.sin(ph) * hull * np.convolve(st, np.ones(k) / k, mode="same") * 0.5
    y = x + sub[:, None]
    # warme Sättigung: leichte gerade Obertöne (wie Röhre), Gleichanteil weg
    p = float(np.abs(y).max()) or 1.0
    z = y / p
    z = z + 0.12 * z ** 2
    z = sosfiltfilt(butter(2, 25, "high", fs=sr, output="sos"), z, axis=0)
    # Tiefen leicht anheben (unter 100 Hz ca. +3 dB)
    lo = sosfiltfilt(butter(2, 100, "low", fs=sr, output="sos"), z, axis=0)
    return (z + 0.4 * lo) * p

def neu_bauen(x, sr, f0, tf, K=12):
    mono = x.mean(1)
    X = np.abs(librosa.stft(mono, n_fft=NFFT, hop_length=HOP))
    freq = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    tfr = np.arange(X.shape[1]) * HOP / sr
    ff, st = f0_auf(tf, f0, tfr)
    wsum = np.hanning(NFFT).sum()
    amps = np.zeros((K, X.shape[1]))
    for k in range(1, K + 1):
        fk = k * ff
        for j in range(X.shape[1]):
            if st[j] < 0.5 or fk[j] > 5000:
                continue
            lo, hi = np.searchsorted(freq, [fk[j] * 0.97, fk[j] * 1.03])
            if hi > lo:
                amps[k - 1, j] = 2 * X[lo:hi + 1, j].max() / wsum
    from scipy.ndimage import median_filter, uniform_filter1d
    amps = uniform_filter1d(median_filter(amps, size=(1, 5)), 3, axis=1)
    t = np.arange(len(mono)) / sr
    fs_, ss_ = f0_auf(tf, f0, t)
    ph = 2 * np.pi * np.cumsum(fs_) / sr
    y = np.zeros(len(mono))
    for k in range(1, K + 1):
        ak = np.interp(t, tfr, amps[k - 1])
        y += ak * np.sin(k * ph) * (k * fs_ < 5000)
    k = int(0.005 * sr)
    y *= np.convolve(np.clip(ss_, 0, 1), np.ones(k) / k, mode="same")
    return np.stack([y, y], 1)

def schutz(x):
    p = float(np.abs(x).max()) or 1.0
    return x / max(p / 0.95, 1.0)

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    x, sr = sf.read(src, always_2d=True, dtype="float64")
    f0, tf = tonhoehe(x.mean(1), sr)
    print("Ton erkannt:", round(float((f0 > 0).mean()), 3), file=sys.stderr)
    v1 = harmonisch(x, sr, f0, tf)
    sf.write(os.path.join(out, "bass_V1_harmonisch.wav"), schutz(v1), sr, subtype="PCM_24")
    sf.write(os.path.join(out, "bass_V2_harmonisch_fett.wav"), schutz(fett(v1, sr, f0, tf)), sr, subtype="PCM_24")
    sf.write(os.path.join(out, "bass_V3_neu_aufgebaut.wav"), schutz(neu_bauen(x, sr, f0, tf)), sr, subtype="PCM_24")


def rms_aktiv(x, sr=44100):
    m = x.mean(1)
    hop = int(0.1 * sr)
    n = len(m) // hop
    r = np.sqrt((m[:n * hop].reshape(n, hop) ** 2).mean(1))
    return float(np.sqrt((r[r > r.max() * 0.05] ** 2).mean())) if n else 0.0


def bass_fertig(a, sr):
    """Hörtest 29.10.: V3 'neu aufgebaut' gewählt – mit der Lautstärke des Originals (A)."""
    f0, tf = tonhoehe(a.mean(1), sr)
    v = neu_bauen(a, sr, f0, tf)
    rv = rms_aktiv(v, sr)
    return v * (rms_aktiv(a, sr) / rv if rv > 0 else 1.0)
