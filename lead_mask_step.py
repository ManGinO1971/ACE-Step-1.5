"""Lead Vocals v2 (NEU, 20. Okt 2026) – Zeitmaske: neu gesungene Lead-Spur nur dort hörbar lassen, wo im ORIGINAL-Song
der Lead wirklich singt (gemessen an der getrennten Original-Lead-Spur).
Wo im Original nur Background war -> echte Stille (weich ein-/ausgeblendet).

Aufruf: python lead_mask.py <neu.wav/flac> <original_lead.wav> <out.wav> [original_background.wav]
Werte per Umgebung (Hörtest 20.10., v2): MASK_DB (-30), MASK_HOLD_MS (300), MASK_VS_BACKING_DB (-10), MASK_LOOK_MS (40),
MASK_ATTACK_MS (25), MASK_RELEASE_MS (200), MASK_MIN_MS (90)
"""
import os, sys
import numpy as np, soundfile as sf, librosa

src, ref, dst = sys.argv[1:4]
E = lambda k, d: float(os.environ.get(k, d))
DB, HOLD, LOOK, ATT, REL, MINO = E("MASK_DB", -30), E("MASK_HOLD_MS", 300), E("MASK_LOOK_MS", 40), E("MASK_ATTACK_MS", 25), E("MASK_RELEASE_MS", 200), E("MASK_MIN_MS", 90)

y, sr = sf.read(src, always_2d=True)
r, rsr = sf.read(ref, always_2d=True)
r = librosa.resample(r.mean(1), orig_sr=rsr, target_sr=sr)
hop = int(sr * 0.01)
env_r = librosa.feature.rms(y=r, frame_length=hop * 4, hop_length=hop)[0]
env_y = librosa.feature.rms(y=y.mean(1), frame_length=hop * 4, hop_length=hop)[0]
# Zeitversatz neu <-> original (sollte 0 sein) per Kreuzkorrelation der Hüllkurven
n = min(len(env_r), len(env_y))
a, b = np.log(env_r[:n] + 1e-6), np.log(env_y[:n] + 1e-6)
a, b = a - a.mean(), b - b.mean()
lags = range(-50, 51)
lag = max(lags, key=lambda L: float(np.dot(a[max(0, L):n + min(0, L)], b[max(0, -L):n - max(0, L)])))
env_r = np.roll(env_r, lag)
db = 20 * np.log10(env_r + 1e-9)
ref_level = np.percentile(db[db > db.max() - 60], 95)
open_ = db > ref_level + DB
if len(sys.argv) > 4:  # wo der Background das Original klar dominiert -> zu
    bk, bsr = sf.read(sys.argv[4], always_2d=True)
    bk = librosa.resample(bk.mean(1), orig_sr=bsr, target_sr=sr)
    env_b = np.roll(librosa.feature.rms(y=bk, frame_length=hop * 4, hop_length=hop)[0], lag)
    m = min(len(env_b), len(open_))
    open_[:m] &= db[:m] > 20 * np.log10(env_b[:m] + 1e-9) + E("MASK_VS_BACKING_DB", -10)
# Mini-Schnipsel streichen, dann Halten + Vorlauf
k = int(MINO / 10)
i = 0
while i < len(open_):
    if open_[i]:
        j = i
        while j < len(open_) and open_[j]: j += 1
        if j - i < k: open_[i:j] = False
        i = j
    else:
        i += 1
hold, look = int(HOLD / 10), int(LOOK / 10)
g = open_.copy()
idx = np.flatnonzero(open_)
for t in idx:
    g[max(0, t - look):t + hold + 1] = True
# Weiche Rampen (Kosinus)
gain = g.astype(float)
att, rel = max(1, int(ATT / 10)), max(1, int(REL / 10))
sm = gain.copy()
for t in range(1, len(sm)):
    target = gain[t]
    step = 1.0 / (att if target > sm[t - 1] else rel)
    sm[t] = min(target, sm[t - 1] + step) if target > sm[t - 1] else max(target, sm[t - 1] - step)
sm = 0.5 - 0.5 * np.cos(np.pi * sm)
g_s = np.interp(np.arange(len(y)), np.arange(len(sm)) * hop, sm)
sf.write(dst, y * g_s[:, None], sr, subtype="PCM_24")
print(f"Zeitmaske: Versatz {lag * 10} ms, {np.mean(g) * 100:.0f} % hörbar, {100 - np.mean(g) * 100:.0f} % echte Stille (Schwelle {DB:.0f} dB)")
