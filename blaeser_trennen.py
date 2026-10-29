#!/usr/bin/env python3
"""HIELTECH Stems – Bläser-Satz in Einzelspuren aufteilen (29. Okt 2026, TEST).
Ausgang: der gewählte saubere Bläser-Satz (NEU, trocken) – aufgeteilt, Summe = gewählter Satz (4_brass_NEU_hall).
  A) nach 53er-Modell: Trompete (53er-Trompete als Wegweiser) + übrige Bläser
  B) nach Stimmlage: hoch (Trompete) / mitte (Sax) / tief (Posaune) – mehrstimmige Tonhöhen je Moment,
     Obertöne jeder Stimme gehen in ihre Spur
Hall danach je Spur mit GENAU dem Hall von 4_brass_NEU_hall (gleiche Raum-Antwort, gleiche Stärke) -> Summe passt.
Aufruf: python blaeser_trennen.py <brass_NEU_trocken.wav> <getrennt_ordner> <ausgabe>"""
import os, sys
import numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.ndimage import uniform_filter

NFFT, HOP = 8192, 512

def hall_ir(sr, sek=1.8):
    n = int(sek * sr); t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / sek)[:, None]
    ir = sosfilt(butter(2, [200, 7000], "band", fs=sr, output="sos"), ir, axis=0)
    ir[: int(0.02 * sr)] = 0
    return ir / np.sqrt((ir ** 2).sum(0, keepdims=True))

def hall_lin(y, ir, g):
    return y + g * np.stack([fftconvolve(y[:, c], ir[:, c])[: len(y)] for c in range(2)], 1)

def stft2(y):
    return [librosa.stft(y[:, c], n_fft=NFFT, hop_length=HOP) for c in range(y.shape[1])]

def istft2(Xs, n):
    return np.stack([librosa.istft(X, hop_length=HOP, length=n) for X in Xs], 1)

def lade(p, n):
    a, _ = sf.read(p, always_2d=True, dtype="float64")
    return a[:n] if len(a) >= n else np.pad(a, ((0, n - len(a)), (0, 0)))

def nach_53(bus, trp, brs):
    n = len(bus)
    Rt = np.abs(librosa.stft(trp.mean(1), n_fft=NFFT, hop_length=HOP))
    Rb = np.abs(librosa.stft(brs.mean(1), n_fft=NFFT, hop_length=HOP))
    m = np.clip(Rt / (np.maximum(Rb, Rt) + 1e-9), 0, 1)
    m = uniform_filter(m, size=(3, 5))
    Xs = stft2(bus)
    T = min(m.shape[1], Xs[0].shape[1]); m = m[:, :T]
    tr = istft2([X[:, :T] * m for X in Xs], n)
    return {"trompete": tr, "uebrige_blaeser": bus - tr}

def nach_lage(bus, sr):
    mono = bus.mean(1)
    X = librosa.stft(mono, n_fft=NFFT, hop_length=HOP); mag = np.abs(X).astype(np.float32); del X
    freq = librosa.fft_frequencies(sr=sr, n_fft=NFFT); df = freq[1]
    cand = 70 * 2 ** (np.arange(0, int(12 * 8 * np.log2(1100 / 70))) / 96)
    K = 8; w = 0.84 ** np.arange(K)
    idx = np.clip(np.round(cand[:, None] * np.arange(1, K + 1)[None, :] / df).astype(int), 0, len(freq) - 1)
    T = mag.shape[1]
    energie = mag.sum(0); aktiv = energie > energie.max() * 10 ** (-50 / 20)
    f0s = np.zeros((3, T))
    for t in np.flatnonzero(aktiv):
        sal = (mag[idx, t] * w).sum(1)
        s0 = sal.max()
        for v in range(3):
            j = int(np.argmax(sal))
            if sal[j] < 0.3 * s0:
                break
            f0s[v, t] = cand[j]
            # Umgebung (±1 Halbton) und Obertöne dieser Stimme abschwächen, damit die nächste Stimme gefunden wird
            for mlt in (1, 2, 3):
                sel = np.abs(12 * np.log2(cand / (cand[j] * mlt))) < 1.0
                sal[sel] = 0
    # sortieren: tief < mitte < hoch
    f0s = np.sort(np.where(f0s > 0, f0s, np.inf), axis=0)
    f0s[np.isinf(f0s)] = 0
    # Masken je Stimme blockweise (Speicher): Gauß um jeden Oberton, Anteile nach Stärke teilen
    Xs = stft2(bus)
    Tm = min(T, Xs[0].shape[1])
    teile = {v: [np.zeros_like(Xc[:, :Tm]) for Xc in Xs] for v in range(3)}
    fr = freq.astype(np.float32)[:, None]
    gesamt = np.ones(3, np.float32) / 3
    B = 400
    for a0 in range(0, Tm, B):
        b0 = min(Tm, a0 + B)
        M = np.zeros((3, len(freq), b0 - a0), np.float32)
        for v in range(3):
            f0b = f0s[v, a0:b0].astype(np.float32)
            for k in range(1, 25):
                fk = k * f0b
                sig = np.maximum(8.0, 0.02 * fk)
                g = np.exp(-0.5 * ((fr - fk[None, :]) / sig[None, :]) ** 2) * (f0b > 0)[None, :] * np.float32(0.92 ** (k - 1))
                np.maximum(M[v], g, out=M[v])
        summe = M.sum(0)
        anteil = M / np.maximum(summe, 1e-9)
        rest = summe < 1e-3
        anteil[:, rest] = (gesamt[:, None] * np.ones((1, rest.sum()), np.float32))
        for v in range(3):
            for c, Xc in enumerate(Xs):
                teile[v][c][:, a0:b0] = Xc[:, a0:b0] * anteil[v]
    out = {}
    for v, name in enumerate(("tief_posaune", "mitte_sax", "hoch_trompete")):
        out[name] = istft2(teile[v], len(bus))
    return out, f0s

def stueckweise(fn, arrays, n, sr, seg_s=20, ov_s=2):
    """fn(*segmente) -> dict name->(m,2). Stücke mit Überblendung zusammensetzen (Speicher sparen)."""
    seg, ov = int(seg_s * sr), int(ov_s * sr)
    out = {}
    s0 = 0
    while s0 < n:
        a0, b0 = max(0, s0 - ov), min(n, s0 + seg + ov)
        res = fn(*[x[a0:b0] for x in arrays])
        L = b0 - a0
        w = np.ones(L)
        if a0 > 0:
            w[:2 * ov] = np.linspace(0, 1, 2 * ov)
        if b0 < n:
            w[-2 * ov:] = np.linspace(1, 0, 2 * ov)
        for k, v in res.items():
            out.setdefault(k, np.zeros((n, 2)))[a0:b0] += v[:L] * w[:, None]
        s0 += seg
    return out


if __name__ == "__main__":
    busp, D, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    bus, sr = sf.read(busp, always_2d=True, dtype="float64")
    n = len(bus)
    trp, brs = lade(f"{D}/trumpet.flac", n), lade(f"{D}/brass.flac", n)
    ir = hall_ir(sr)
    nass = np.stack([fftconvolve(bus[:, c], ir[:, c])[: len(bus)] for c in range(2)], 1)
    g = 0.18 * np.sqrt((bus ** 2).mean()) / (np.sqrt((nass ** 2).mean()) + 1e-12)   # = Hall von 4_brass_NEU_hall
    ref_bus = hall_lin(bus, ir, g)
    p = max(float(np.abs(ref_bus).max()) / 0.98, 1.0)
    a = stueckweise(nach_53, [bus, trp, brs], n, sr)
    for k, v in a.items():
        sf.write(f"{outdir}/A_53er_{k}.wav", hall_lin(v, ir, g) / p, sr, subtype="PCM_24")
    stimmen_alle = []
    def lage(seg):
        o, f0s = nach_lage(seg, sr)
        stimmen_alle.append((f0s > 0).sum(0))
        return o
    b = stueckweise(lage, [bus], n, sr)
    for k, v in b.items():
        sf.write(f"{outdir}/B_lage_{k}.wav", hall_lin(v, ir, g) / p, sr, subtype="PCM_24")
    stimmen = np.concatenate(stimmen_alle); akt = stimmen > 0
    print("Stimmen gleichzeitig (in Bläser-Momenten): 1:", round(float(np.mean(stimmen[akt] == 1)), 2),
          "2:", round(float(np.mean(stimmen[akt] == 2)), 2), "3:", round(float(np.mean(stimmen[akt] == 3)), 2))
    r = ref_bus / p
    for name, d in (("A", a), ("B", b)):
        s_ = sum(hall_lin(v, ir, g) / p for v in d.values())
        print(f"Summe {name} = Satz? max Abweichung", round(float(np.abs(s_ - r).max()), 5))


def satz_und_teile(bus, sr):
    """Hörtest 29.10.: Variante B (nach Stimmlage) gewählt. -> (Satz mit Hall, {teil: Spur mit Hall}),
    gleiche Raum-Antwort/Stärke wie 4_brass_NEU_hall, gleicher Übersteuerungsschutz -> Teile ergeben zusammen den Satz."""
    ir = hall_ir(sr)
    nass = np.stack([fftconvolve(bus[:, c], ir[:, c])[: len(bus)] for c in range(2)], 1)
    g = 0.18 * np.sqrt((bus ** 2).mean()) / (np.sqrt((nass ** 2).mean()) + 1e-12)
    satz = hall_lin(bus, ir, g)
    p = max(float(np.abs(satz).max()) / 0.98, 1.0)
    teile = stueckweise(lambda seg: nach_lage(seg, sr)[0], [bus], len(bus), sr)
    return satz / p, {k: hall_lin(v, ir, g) / p for k, v in teile.items()}
