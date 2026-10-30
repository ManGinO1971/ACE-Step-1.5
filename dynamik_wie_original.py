#!/usr/bin/env python3
"""HIELTECH Stems – Lautstärke-Verlauf wie im Original (30. Okt 2026).
Hörtest: einige Stellen in der sauberen Fassung etwas zu leise im Vergleich zum Original.
Das Original "hören": Lautheit der Original-Vorlage (53er-Trennung) und der neuen Spur in 1-s-Fenstern vergleichen.
NUR den relativen Verlauf angleichen (Gesamtpegel bleibt), langsam (1,5 s geglättet -> kein Pumpen),
höchstens ±6 dB, nur wo das Instrument spielt; Pausen und Ausklang bleiben unberührt.
Aufruf: python dynamik_wie_original.py <neu> <vorlage> <ausgabe>"""
import sys
import numpy as np, soundfile as sf
from scipy.ndimage import uniform_filter1d

def angleichen(x, ref, sr, fenster=1.0, glatt=1.5, grenze=6.0):
    n = len(x); hop = int(0.1 * sr); L = int(fenster * sr)
    def lautheit(a):
        m = a.mean(1) ** 2
        k = np.convolve(m, np.ones(L) / L, mode="same")[::hop]
        return 10 * np.log10(k + 1e-12)
    ln, lr = lautheit(x), lautheit(ref)
    m = min(len(ln), len(lr)); ln, lr = ln[:m], lr[:m]
    aktiv = (lr > lr.max() - 35) & (ln > ln.max() - 40)
    diff = lr - ln
    if aktiv.sum() < 5:
        return x, 0.0
    diff = diff - np.median(diff[aktiv])                 # Gesamtpegel bleibt, nur Verlauf
    g = np.where(aktiv, np.clip(diff, -grenze, grenze), 0.0)
    g = uniform_filter1d(g, max(1, int(glatt / 0.1)))
    gain = 10 ** (np.interp(np.arange(n), np.arange(m) * hop, g) / 20)
    y = x * gain[:, None]
    return y / max(np.abs(y).max() / 0.98, 1.0), float(np.abs(g).max())

if __name__ == "__main__":
    x, sr = sf.read(sys.argv[1], always_2d=True, dtype="float64")
    r, _ = sf.read(sys.argv[2], always_2d=True, dtype="float64")
    r = r[:len(x)] if len(r) >= len(x) else np.pad(r, ((0, len(x) - len(r)), (0, 0)))
    y, gmax = angleichen(x, r, sr)
    sf.write(sys.argv[3], y, sr, subtype="PCM_24")
    print("größte Anpassung", round(gmax, 1), "dB")
