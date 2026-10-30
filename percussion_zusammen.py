#!/usr/bin/env python3
"""HIELTECH Stems – Percussion/Djembe zusammen (29. Okt 2026, TEST).
Befund: die neu erzeugte Percussion hat fast doppelt so viele Anschläge wie die Vorlage und viel mehr Höhen
(> 1 kHz) – fremde Elemente (Shaker-/Hi-Hat-artige Reste) sind mit drin -> "verschiedene Instrumente vermischt".
  1. Quelle = neu erzeugte Percussion (unmaskiert), die zur Vorlage am besten passt; Vorlage = 53er congas + percussion.
  2. Zuerst entrauschen (Rauschabdruck aus Stellen, wo das Original schweigt).
  3. Fremdes raus: nur was die Vorlage an dieser Stelle/Frequenz auch hat (weiche Maske, in Bändern und ~50 ms
     geglättet) – die neue Spur liefert den Klang, die Vorlage nur den Wegweiser.
  4. Natürliches Ausklingen statt hartem Schnitt (0,6 s), Variante mit Fell-Resonanz (kurzes Halten 0,35 s).
  5. Teile: Trommeln (Djembe/Congas, Wegweiser 53er-congas) + Klein-Percussion (Shaker usw.) – Summe = Satz.
Aufruf: python percussion_zusammen.py <A_roh_1> [<A_roh_2> ...] --ref <congas> <percussion> --out <ordner>"""
import os, sys
import numpy as np, soundfile as sf, librosa
from scipy.ndimage import uniform_filter

NFFT, HOP = 2048, 256

def lade(p, n=None):
    a, sr = sf.read(p, always_2d=True, dtype="float64")
    if n is not None:
        a = a[:n] if len(a) >= n else np.pad(a, ((0, n - len(a)), (0, 0)))
    return a, sr

def onset_kor(a, b, sr):
    oa = librosa.onset.onset_strength(y=a.mean(1).astype(np.float32), sr=sr, hop_length=512)
    ob = librosa.onset.onset_strength(y=b.mean(1).astype(np.float32), sr=sr, hop_length=512)
    m = min(len(oa), len(ob)); return float(np.corrcoef(oa[:m], ob[:m])[0, 1])

def frame_db(m):
    n = int(np.ceil(len(m) / HOP)); y = np.pad(m, (0, n * HOP - len(m)))
    return 20 * np.log10(np.sqrt((y.reshape(n, HOP) ** 2).mean(1) + 1e-12))

def release(akt, sr, t60):
    d = 10 ** (-3.0 * HOP / sr / t60); g = np.zeros(len(akt)); cur = 0.0
    for i, a in enumerate(akt):
        cur = 1.0 if a else cur * d; g[i] = cur
    return g

def halten(X, sr, t60):
    mag = np.abs(X); g = 10 ** (-3.0 * HOP / sr / t60)
    adv = 2 * np.pi * np.arange(X.shape[0]) * HOP / NFFT
    rng = np.random.default_rng(5); held = np.zeros(X.shape[0]); ph = np.angle(X[:, 0]); out = X.copy()
    freq = librosa.fft_frequencies(sr=sr, n_fft=NFFT); nur = (freq > 50) & (freq < 1500)   # Fell-Resonanz, nicht Shaker
    for t in range(X.shape[1]):
        held = np.maximum(mag[:, t], held * g)
        ph = np.where(mag[:, t] >= 0.7 * held, np.angle(X[:, t]), ph + adv + rng.uniform(-0.6, 0.6, len(adv)))
        out[:, t] = X[:, t] + (held - mag[:, t]) * nur * np.exp(1j * ph)
    return out

def verarbeite(a, R_all, R_cong, sr, resonanz):
    """-> satz, trommeln (beide (n,2)); klein = satz - trommeln"""
    n = len(a)
    ref_m = R_all.mean(1)
    db = frame_db(ref_m); akt = db > db.max() - 45
    k = int(1.0 * sr / HOP); naehe = np.convolve(akt.astype(float), np.ones(2 * k + 1), mode="same") > 0
    Ra = np.abs(librosa.stft(ref_m, n_fft=NFFT, hop_length=HOP))
    Rc = np.abs(librosa.stft(R_cong.mean(1), n_fft=NFFT, hop_length=HOP))
    satz = np.zeros_like(a); trom = np.zeros_like(a)
    for c in range(a.shape[1]):
        X = librosa.stft(a[:, c], n_fft=NFFT, hop_length=HOP); T = X.shape[1]
        ak = np.pad(akt, (0, max(0, T - len(akt))))[:T]; ru = ~np.pad(naehe, (0, max(0, T - len(naehe))))[:T]
        mag = np.abs(X)
        prof = np.median(mag[:, ru], axis=1, keepdims=True) if ru.sum() > 50 else np.percentile(mag, 10, axis=1, keepdims=True)
        G = np.clip(1 - 1.5 * prof / (mag + 1e-9), 0.03, 1.0)
        X = X * uniform_filter(G, size=(1, 3))
        # Fremdes raus: Pegel-Abgleich Vorlage<->neu je Band, weiche Maske
        A = uniform_filter(np.abs(X), size=(9, 9)); Rr = uniform_filter(Ra[:, :T], size=(9, 9))
        skal = np.median(A[:, ak]) / (np.median(Rr[:, ak]) + 1e-12) if ak.any() else 1.0
        M = np.clip((Rr * skal) / (A + 1e-9), 0, 1) ** 0.5
        M = np.maximum(uniform_filter(M, size=(5, 9)), 0.05)
        X = X * M * release(ak, sr, 0.6)[None, :]
        if resonanz:
            X = halten(X, sr, 0.35)
        # Teile: Trommeln nach 53er-congas, Rest = Klein-Percussion
        Cc = uniform_filter(Rc[:, :T], size=(5, 9)); Aa = uniform_filter(Ra[:, :T], size=(5, 9))
        Mt = np.clip(Cc / (np.maximum(Aa, Cc) + 1e-9), 0, 1)
        satz[:, c] = librosa.istft(X, hop_length=HOP, length=n)
        trom[:, c] = librosa.istft(X * Mt, hop_length=HOP, length=n)
    return satz, trom

def schutz_faktor(x):
    return max(float(np.abs(x).max()) / 0.98, 1.0)

if __name__ == "__main__":
    args = sys.argv[1:]
    i_ref, i_out = args.index("--ref"), args.index("--out")
    quellen, (pc, pp), outdir = args[:i_ref], args[i_ref + 1:i_ref + 3], args[i_out + 1]
    os.makedirs(os.path.join(outdir, "teile"), exist_ok=True)
    Rc, sr = lade(pc); Rp, _ = lade(pp, len(Rc)); R = Rc + Rp
    best, bs = None, -9
    for q in quellen:
        a, _ = lade(q, len(R)); s = onset_kor(R, a, sr)
        print("Quelle", os.path.basename(q), "Anschlag-Übereinstimmung", round(s, 3))
        if s > bs: best, bs, bn = a, s, q
    print("gewählt:", os.path.basename(bn))
    for name, res in (("2_percussion_NEU", False), ("3_percussion_NEU_resonanz", True)):
        satz, trom = verarbeite(best, R, Rc, sr, res)
        p = schutz_faktor(satz)
        sf.write(os.path.join(outdir, name + ".wav"), satz / p, sr, subtype="PCM_24")
        if res:
            sf.write(os.path.join(outdir, "teile", "trommeln.wav"), trom / p, sr, subtype="PCM_24")
            sf.write(os.path.join(outdir, "teile", "klein_percussion.wav"), (satz - trom) / p, sr, subtype="PCM_24")
