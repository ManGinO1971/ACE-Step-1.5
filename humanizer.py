"""Humanizer für Backing-Stimmen: jede gesungene Phrase (zwischen Pausen >= PAUSE_MS) wird um einen kleinen
zufälligen Zeitversatz verschoben (+-MAX_MS, mal vor, mal hinter), jede Stimme unabhängig, dazu leichte
Lautstärke-Schwankung je Phrase (+-DB). Schnitte liegen in Pausen -> keine Klicks, kein Tonhöhen-Wackeln.
Aufruf: humanizer.py <ein.wav> <aus.wav> <seed> [MAX_MS=25] [DB=1.0] [PAUSE_MS=120]"""
import sys, numpy as np, soundfile as sf
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
max_ms = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
db = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
pause_ms = float(sys.argv[6]) if len(sys.argv) > 6 else 120.0
y, sr = sf.read(src, always_2d=True); n = len(y)
rng = np.random.default_rng(seed)
h = int(sr * 0.01); fr = n // h
e = 20 * np.log10(np.sqrt((y[:fr * h].mean(1).reshape(fr, h) ** 2).mean(1)) + 1e-9)
act = e > e.max() - 45
# Phrasen: aktive Bereiche, Lücken < pause_ms zählen zur Phrase
pf = int(pause_ms / 10); segs = []; i = 0
while i < fr:
    if act[i]:
        j = i
        while True:
            while j < fr and act[j]: j += 1
            k = j
            while k < fr and not act[k] and k - j < pf: k += 1
            if k < fr and act[k] and k - j < pf: j = k; continue
            break
        segs.append((i, j)); i = j
    else: i += 1
out = np.zeros_like(y); fade = int(0.01 * sr); prev = 0.0; log = []
for a, b in segs:
    # spontan, aber nicht sprunghaft: neuer Versatz in der Nähe des vorigen, Vorzeichen wechselt oft
    off = float(np.clip(0.5 * prev + rng.uniform(-max_ms, max_ms), -max_ms, max_ms)); prev = off
    g = 10 ** (rng.uniform(-db, db) / 20)
    s0 = max(0, a * h - fade); s1 = min(n, b * h + fade)
    x = y[s0:s1].copy() * g
    w = np.ones(len(x)); w[:fade] = np.linspace(0, 1, fade); w[-fade:] = np.linspace(1, 0, fade)
    x *= w[:, None]
    d = int(round(off / 1000 * sr)); t0 = s0 + d; t1 = t0 + len(x)
    c0, c1 = max(0, t0), min(n, t1)
    out[c0:c1] += x[c0 - t0:c1 - t0]
    log.append(off)
sf.write(dst, out, sr, subtype='PCM_24')
print(f'{len(segs)} Phrasen, Versatz {np.min(log):+.0f} bis {np.max(log):+.0f} ms (Mittel |{np.mean(np.abs(log)):.0f}| ms), Pegel +-{db} dB')
