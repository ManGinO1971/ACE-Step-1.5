"""HIELTECH Stems – Haupt-Snare-Schutz (18. Nov 2026, Mehr-Song-Test).

Das 53er-Modell steckt bei vielen Songs die echte Snare in „Percussion“. Das Trommeln-Abziehen würde dann echte
Snare-Schläge entfernen (Fire Burn: ganze Snare weg). Darum: die Haupt-Snare des Songs finden (lauter, regelmäßig
auf dem Backbeat wiederkehrender Klang) und jeden Schlag mit ihrem Klang aus der Original-Snare zurückholen.
"""
import numpy as np
import soundfile as sf
import librosa

SR = 44100
db = lambda a: 20 * np.log10(np.abs(a).max() + 1e-9) if len(a) else -120.0
f = np.fft.rfftfreq(8192, 1 / SR)


def feats(x, i):
    w = x[i:i + int(0.08 * SR)]
    W = np.abs(np.fft.rfft(np.pad(w, (0, 8192 - len(w))) * np.hanning(8192))) ** 2
    b = lambda a, c: W[(f > a) & (f < c)].sum() + 1e-15
    rausch = 10 * np.log10(b(4000, 12000) / b(80, 1000))
    fl = float(librosa.feature.spectral_flatness(y=w, n_fft=2048).mean())
    k = (f > 80) & (f < 1200)
    f0 = f[k][np.argmax(W[k])]
    e = np.sqrt(np.convolve(x[i:i + int(0.5 * SR)] ** 2, np.ones(441) / 441, "same"))
    p = e.argmax()
    dec = np.argmax(e[p:] < e[p] * 0.1) / SR * 1000
    return rausch, fl, f0, dec


def haupt_snare(x, drums):
    tempo, beats = librosa.beat.beat_track(y=drums, sr=SR, hop_length=256, units="time")
    P = float(np.median(np.diff(beats)))
    on = librosa.onset.onset_detect(y=x, sr=SR, hop_length=128, units="samples", backtrack=True, delta=0.04)
    H = [(i / SR, db(x[i:i + 3500])) + feats(x, i) for i in on]
    H = np.array([h for h in H if h[1] > -35])
    laut = H[H[:, 1] >= H[:, 1].max() - 10]
    st = 12 * np.log2(laut[:, 4] / 100.)
    best = None
    for c in np.unique(np.round(st)):
        k = np.abs(st - c) <= 1.5
        T = laut[k, 0]
        reg = sum(1 for t in T if any(abs(abs(t - u) - m * P) < 0.035 for u in T for m in (2, 4)))
        if best is None or reg > best[0]:
            best = (reg, c, np.median(laut[k, 4]), np.median(laut[k, 2]), k.sum())
    return H, best, P


def mel(x, i):
    w = x[i:i + int(0.1 * SR)]
    w = np.pad(w, (0, max(0, int(0.1 * SR) - len(w))))
    S = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=4096, hop_length=1024, n_mels=40, fmin=100, fmax=12000).mean(1)
    return 10 * np.log10(S / S.max() + 1e-9)


def schutz(x, drums):
    H, b, P = haupt_snare(x, drums)
    laut = H[H[:, 1] >= H[:, 1].max() - 10]
    st = 12 * np.log2(laut[:, 4] / 100.)
    k = np.abs(st - b[1]) <= 1.5
    kern = laut[k]
    tpl = np.median([mel(x, int(t * SR)) for t in kern[:, 0]], 0)
    dist = lambda t: float(np.sqrt(np.mean((mel(x, int(t * SR)) - tpl) ** 2)))
    dk = [dist(t) for t in kern[:, 0]]
    thr = np.percentile(dk, 90) * 1.7
    return [(t, lv, dist(t), dist(t) <= thr) for t, lv, *_ in H], thr


def wiederherstellen(orig_p, sauber_p, drums_p):
    """Haupt-Snare-Schläge, die das Trommeln-Abziehen (53er) entfernt hat, aus der Original-Snare zurückholen."""
    o, _ = sf.read(orig_p, always_2d=True)
    s, _ = sf.read(sauber_p, always_2d=True)
    n = min(len(o), len(s))
    o = o[:n]
    s = s[:n].copy()
    dr = sf.read(drums_p, always_2d=True)[0].mean(1)
    R, thr = schutz(o.mean(1), dr)
    T = [r[0] for r in R]
    zur = 0
    for k, (t, lv, dd, ok) in enumerate(R):
        if not ok:
            continue
        a = max(0, int((t - 0.01) * SR))
        e = min(n, int(min(T[k + 1] if k + 1 < len(T) else t + 0.45, t + 0.45) * SR))
        if e - a < 4:
            continue
        if db(s[a:e, 0]) < db(o[a:e, 0]) - 3:
            zur += 1
        fe = np.ones(e - a)
        r = min(220, len(fe) // 2)
        fe[:r] = np.linspace(0, 1, r)
        fe[-r:] = np.linspace(1, 0, r)
        s[a:e] = s[a:e] * (1 - fe[:, None]) + o[a:e] * fe[:, None]
    sf.write(sauber_p, s, SR, subtype="PCM_24")
    return {"haupt_snare_geschuetzt": int(sum(r[3] for r in R)), "zurueckgeholt": zur, "schwelle": round(float(thr), 1)}
