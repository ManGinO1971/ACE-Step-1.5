"""Schlagzeug-Stem säubern – für jeden Song (28. Nov 2026, Money Tumble Down: „gut, bauen“).

    saeubern(drums, teile_ordner, s53, ziel)   s53 = {"congas": pfad, "tambourine": …, "percussion": …, "hh": …, …}
    python3 stem_saeubern.py DRUMS.wav TEILE_ORDNER S53_ORDNER ZIEL.wav [--andere OTHER.wav] [--bericht B.json]

1. Hand-Trommel-Prüfung: spielen Congas/Djembe/Bongos laut und oft (≥ 30 laute Einsätze je Minute)?
   Nein → der Stem bleibt wie er ist (z. B. Zion Callin, Stand The Ghetto).
2. Ja → je Frequenzband die beste Fassung (Inhaber, Money Tumble Down):
   Tief  (< 150 Hz): nur an bestätigten Kicks offen (DrumSep-Kick trägt den Schlag, Congas dort nicht lauter)
                     → kein Djembe-Bass als zweite Kick
   Mitte (150 Hz–5 kHz): 53er-Congas/Tamburin/Percussion/andere Instrumente heraus; an Kicks bleibt die Percussion
                     (das 53er legt den Rim oft dorthin); wo Snare/Toms klar lauter sind (Fills) die volle Fassung
   Hoch  (> 5 kHz): beide Trennungen, nur das Tamburin heraus – aber nicht dort, wo die 53er-Hi-Hat fehlt
                     (dann ist ihr „Tamburin“ die Hi-Hat, Zion Callin)"""
import argparse, json, os
import numpy as np, soundfile as sf
from scipy.signal import stft, istft, find_peaks
from scipy.ndimage import uniform_filter, uniform_filter1d
SR = 44100; N = 2048; H = 512


def lese(p, n=None):
    x, _ = sf.read(p, dtype='float32', always_2d=True)
    x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, 1)
    if n:
        x = np.pad(x, ((0, max(0, n - len(x))), (0, 0)))[:n]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('drums'); ap.add_argument('teile'); ap.add_argument('s53'); ap.add_argument('ziel')
    ap.add_argument('--andere', default=None); ap.add_argument('--bericht', default=None)
    a = ap.parse_args()
    s53 = {k: os.path.join(a.s53, k + '.flac') for k in ('toms', 'snare', 'congas', 'percussion', 'hh', 'tambourine')}
    ber = saeubern(a.drums, a.teile, s53, a.ziel, a.andere)
    if a.bericht:
        json.dump(ber, open(a.bericht, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in ber.items() if k != 'tiefe_reste_zeiten'}, ensure_ascii=False))


def saeubern(drums, teile, s53, ziel, andere=None):
    """Schreibt den (gesäuberten) Stem nach ziel und gibt den Bericht zurück. ber['gesaeubert'] = ob etwas geändert wurde."""
    D = lese(drums); n = len(D)
    f = np.fft.rfftfreq(N, 1 / SR)
    S = lambda x: stft(x, SR, nperseg=N, noverlap=N - H)[2]
    def P(p, w=1.0):
        return w * np.abs(S(lese(p, n).mean(1))) ** 2 if p and os.path.isfile(p) else 0.0
    t53 = lambda k: (s53 or {}).get(k) or ''; tt = lambda k: os.path.join(teile, k + '.flac')
    ber = {}
    # ---- 1. Hand-Trommel-Prüfung
    import librosa
    if not os.path.isfile(t53('congas')):
        return dict(hand_trommeln=dict(fehlt='53er-Congas'), gesaeubert=False)
    cg = lese(t53('congas'), n).mean(1)
    ref = 20 * np.log10(np.percentile(np.abs(D.mean(1)), 99.9) + 1e-9)
    oc = librosa.onset.onset_detect(y=cg, sr=SR, units='samples', hop_length=256, delta=0.25)
    laut = [o for o in oc if 20 * np.log10(np.abs(cg[o:o + int(.06 * SR)]).max() + 1e-9) - ref > -30]
    je_min = len(laut) / (n / SR) * 60
    ber['hand_trommeln'] = dict(laute_einsaetze_je_min=round(je_min, 1), da=bool(je_min >= 30))
    if not ber['hand_trommeln']['da']:
        ber['gesaeubert'] = False
        return ber
    # ---- 2. Masken
    Pd = np.abs(S(D.mean(1))) ** 2
    ds = sum(lese(tt(k), n) for k in ('kick', 'snare', 'toms', 'hihat', 'ride') if os.path.isfile(tt(k)))
    Pds = np.abs(S(ds.mean(1))) ** 2
    m1 = np.clip(uniform_filter(np.sqrt(np.minimum(Pd, Pds) / (Pd + 1e-12)), (3, 3)), 0.03, 1.0)   # Schnittmenge
    Pk, Pc, Pt, Pp = P(tt('kick')), P(t53('congas')), P(t53('tambourine')), P(t53('percussion'))
    Ph53 = P(t53('hh')); Pa = P(andere, 0.3)
    kit = P(t53('snare')) + P(t53('toms')) + Ph53 + Pk + P(tt('ride'))
    non = Pc + Pt + 0.5 * Pp + Pa
    e = 1e-3 * float(np.mean(kit + non)) + 1e-12
    m2 = np.clip(uniform_filter((kit + e) / (kit + non + e), (3, 3)), 0.03, 1.0)
    kit2 = kit + Pp; non2 = Pc + Pt + Pa
    e2 = 1e-3 * float(np.mean(kit2 + non2)) + 1e-12
    m2b = np.clip(uniform_filter((kit2 + e2) / (kit2 + non2 + e2), (3, 3)), 0.03, 1.0)
    # Kick-Hörmodell (tief)
    tief = (f >= 35) & (f < 150); db = lambda x: 10 * np.log10(x + 1e-12)
    ed, ek, ec = db(Pd[tief].sum(0)), db(Pk[tief].sum(0)), db(Pc[tief].sum(0)) if np.ndim(Pc) else np.full(Pd.shape[1], -120.0)
    fl = np.maximum(0, ed - np.roll(ed, 2)); on, _ = find_peaks(fl, height=6, distance=int(0.12 * SR / H))
    rf = np.percentile(ed, 99); kicks, weg = [], []
    for o in on:
        w = slice(o, min(o + 4, len(ed)))
        if ed[w].max() < rf - 30:
            continue
        (kicks if (ek[w].max() > ed[w].max() - 8 and ec[w].max() < ek[w].max() - 3) else weg).append(int(o))
    # Nur säubern, wenn die tiefe Hand-Trommel wirklich als zweite Kick durchkommt – sonst bleibt alles wie getrennt
    if len(weg) < max(8, 0.1 * len(kicks)):
        ber.update(gesaeubert=False, kicks=len(kicks), tiefe_reste_weg=len(weg),
                   grund='Hand-Trommeln da, aber keine falschen Kicks')
        return ber
    tor = np.full(Pd.shape[1], 0.03); rim = np.zeros(Pd.shape[1])
    for i, o in enumerate(kicks):
        e_ = min(o + int(0.35 * SR / H), kicks[i + 1] - 1 if i + 1 < len(kicks) else Pd.shape[1])
        tor[max(0, o - 2):e_] = 1.0; rim[max(0, o - 3):o + int(0.08 * SR / H)] = 1
    tor = uniform_filter1d(tor, 3); rim = uniform_filter1d(rim, 3)
    # Fill-Schutz (Mitte)
    mf = (f >= 150) & (f < 5000)
    ks = (P(t53('snare')) + P(t53('toms')))[mf].sum(0); ns = (Pc + Pp)[mf].sum(0)
    voll = np.clip(uniform_filter1d(((db(ks) - db(ns)) > 3).astype(float), 5), 0, 1)
    # Hoch: Tamburin heraus, außer wo die 53er-Hi-Hat fehlt (dort ist das „Tamburin“ die Hi-Hat)
    Phd = P(tt('hihat'))
    hs = (f >= 5000)
    rS = db(uniform_filter1d(Ph53[hs].sum(0), int(0.5 * SR / H))); rD = db(uniform_filter1d(Phd[hs].sum(0), int(0.5 * SR / H)))
    luecke = uniform_filter1d(((rS - np.percentile(rS, 90)) < (rD - np.percentile(rD, 90)) - 12).astype(float), int(.2 * SR / H))
    kit_h = Ph53 + Phd + P(tt('ride')); eh = 1e-3 * float(np.mean(kit_h)) + 1e-12
    mt = np.clip(uniform_filter((kit_h + eh) / (kit_h + Pt + eh), (3, 3)), 0.1, 1.0)
    mt = mt * (1 - luecke[None, :]) + luecke[None, :]
    g = lambda c, b: 1 / (1 + (f / c) ** (2 * b))
    lp = g(150, 4)[:, None]; hp = (1 - g(5000, 4))[:, None]; mid = 1 - lp - hp
    y = np.zeros_like(D)
    for k in range(2):
        X = S(D[:, k])
        Z = X * m1 * m2 * lp * tor[None, :] \
            + ((X * m2 * (1 - rim[None, :]) + X * m2b * rim[None, :]) * (1 - voll[None, :]) + X * m1 * voll[None, :]) * mid \
            + X * m1 * hp * mt
        y[:, k] = istft(Z, SR, nperseg=N, noverlap=N - H)[1][:n]
    sf.write(ziel, y, SR, subtype='PCM_24')
    t = lambda v: [round(i * H / SR, 2) for i in v]
    ber.update(gesaeubert=True, kicks=len(kicks), tiefe_reste_weg=len(weg), tiefe_reste_zeiten=t(weg)[:60],
               fill_schutz_prozent=round(100 * float(np.mean(voll > 0.5)), 1),
               hihat_luecke_prozent=round(100 * float(np.mean(luecke > 0.5)), 1))
    return ber


if __name__ == '__main__':
    main()
