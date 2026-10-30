#!/usr/bin/env python3
"""HIELTECH Stems – EIN einzelnes Streichinstrument (Geige/Bratsche/Cello; Hörtest mit Geige) (einstimmiges Streichinstrument) sauber + Hall bis zum Ende (30. Okt 2026).
Hörtest: violin_B_karte am besten ("hört sich original an"), Neu-Bauen klappt bei Streichern NICHT.
Probleme: wie beim Sax geschnitten (Hall-Ende fehlt) + etwas Keys/Synth mit drin.
  1. Klang = B (Karte/Vocoder) während die Geige spielt.
  2. Ausklang: wo B durch die Zeitmaske abbricht, übernimmt die unmaskierte Fassung (A) mit natürlichem
     Ausklingen (3 s) -> Hall bis zum Ende, kein Schnitt.
  3. Keys/Synth raus: die Geige spielt immer nur EINEN Ton. Tonhöhe aus der 53er-Geigen-Vorlage (sauber);
     nur Grundton + Obertöne dieses Tons bleiben (im Ausklang wird der letzte Ton gehalten -> eigener Hall bleibt),
     alles dazwischen (andere Akkord-Töne von Keys/Synth) stark abgesenkt; Bogen-Rauschen bleibt leise erhalten.
Aufruf: python geige_sauber.py <B_karte> <A_roh> <vorlage_violin> <ausgabe> [fmin fmax]"""
import sys
import numpy as np, soundfile as sf, librosa
from scipy.ndimage import uniform_filter1d

NFFT, HOP = 4096, 512

def lade(p, n=None):
    a, sr = sf.read(p, always_2d=True, dtype="float64")
    if n is not None:
        a = a[:n] if len(a) >= n else np.pad(a, ((0, n - len(a)), (0, 0)))
    return a, sr

def frame_db(m):
    n = int(np.ceil(len(m) / HOP)); y = np.pad(m, (0, n * HOP - len(m)))
    return 20 * np.log10(np.sqrt((y.reshape(n, HOP) ** 2).mean(1) + 1e-12))

def tonhoehe(ref, sr, fmin, fmax, T):
    y = librosa.resample(ref.mean(1), orig_sr=sr, target_sr=11025)
    f0, v, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=11025, frame_length=1024, hop_length=128)
    f0 = np.where(v & np.isfinite(f0), f0, 0.0)
    t_src = np.arange(len(f0)) * 128 / 11025; t = np.arange(T) * HOP / sr
    idx = np.clip(np.searchsorted(t_src, t), 0, len(f0) - 1)
    f = f0[idx]
    # letzten Ton im Ausklang halten (bis 3 s), damit der eigene Hall der Geige bleibt
    halte = int(3.0 * sr / HOP); last, cnt = 0.0, 0
    for i in range(T):
        if f[i] > 0: last, cnt = f[i], 0
        elif last > 0 and cnt < halte: f[i] = last; cnt += 1
    return f

def einzeln_sauber(B, A, R, sr, fmin=180.0, fmax=2200.0):
    """B = Karte/Vocoder-Fassung, A = unmaskierte neu erzeugte Fassung, R = 53er-Vorlage, alle (n, 2)."""
    n = len(B)
    dbB = frame_db(B.mean(1)); aktiv = dbB > dbB.max() - 50
    d = 10 ** (-3.0 * HOP / sr / 3.0); g = np.zeros(len(aktiv)); cur = 0.0
    for i, a in enumerate(aktiv):
        cur = 1.0 if a else cur * d; g[i] = cur
    w = uniform_filter1d(aktiv.astype(float), 5)
    t_f = np.arange(len(aktiv)) * HOP + HOP / 2; t = np.arange(n)
    wB = np.interp(t, t_f, w); gA = np.interp(t, t_f, g * (1 - w))
    X = B * wB[:, None] + A * gA[:, None]
    out = np.zeros_like(X)
    freq = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    M = None
    for c in range(X.shape[1]):
        S = librosa.stft(X[:, c], n_fft=NFFT, hop_length=HOP); T = S.shape[1]
        if M is None:
            f0 = tonhoehe(R, sr, fmin, fmax, T)
            M = np.zeros((len(freq), T), np.float32)
            for k in range(1, 41):
                fk = k * f0; sig = np.maximum(12.0, 0.025 * fk)
                gk = np.exp(-0.5 * ((freq[:, None] - fk[None, :]) / sig[None, :]) ** 2) * (f0 > 0)[None, :]
                gk[:, fk > 16000] = 0
                M = np.maximum(M, gk.astype(np.float32))
            M[:, f0 <= 0] = 1.0
            M = np.maximum(M, 0.08)
        out[:, c] = librosa.istft(S * M[:, :T], hop_length=HOP, length=n)
    return out / max(np.abs(out).max() / 0.98, 1.0)


def main():
    bp, ap, rp, outp = sys.argv[1:5]
    fmin, fmax = (float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 else (180.0, 2200.0)
    B, sr = lade(bp); n = len(B)
    A, _ = lade(ap, n); R, _ = lade(rp, n)
    sf.write(outp, einzeln_sauber(B, A, R, sr, fmin, fmax), sr, subtype="PCM_24")


if __name__ == "__main__":
    main()
