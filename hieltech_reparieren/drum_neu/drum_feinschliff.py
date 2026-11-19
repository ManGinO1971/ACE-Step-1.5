"""HIELTECH Stems – Schlagzeug-Feinschliff (Hörtests 18. Nov 2026 mit Rise Up + 4 weiteren Songs).

Stand 19. Nov 2026: fest in drums_neu_step.py eingebaut (Ganzes Schlagzeug), läuft auf dem Mac und auf der GPU gleich.
Setzt auf die vorhandene Schlagzeug-Reparatur (drums_neu_step.py) auf und wendet die mit "Rise Up" im Hörtest
festgelegten Regeln an – jede Regel misst die Werte aus dem jeweiligen Song selbst (kein fester Standard):
  Hi-Hat : Schlag-Raster (Stellen im Beat) aus dem Song gelernt, Muster (Pegel + Lage je Stelle) von der saubersten
           Stelle des Songs, überall wo die Hi-Hat spielt gleich laut (Trennung frisst sonst Teile auf)
  Snare  : eine normale Snare mit festem Klang; hohe (Timbal-artige) Snare nur, wo der Schlag selbst klar hoch klingt;
           Artefakte (53er hört keine Snare) und Tom-Übersprechen raus; hohe Snare +9 dB, Dynamik ×1,6,
           Fills ab 3 Schlägen mit Steigerung, Einzel-Schläge hoch ≥ normale Snare +2 dB
  Kick   : ein fester Kick-Klang (am nächsten am Original), Klang zur Hälfte ans Original angeglichen
  Toms   : Lautheit = normale Snare +6 dB, Unterschiede zu 40 % erhalten, voller Ausklang
  Becken : alle Crash-Schläge gleich laut wie die lautesten; Becken mit langem Nachklang nie als Übersprechen löschen
Aufruf im Modul: feinschliff(drums, teile, rep, s53_snare, s53_hh)
"""
import os, sys, json, glob
import numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfilt, sosfiltfilt, firwin2, fftconvolve

SR = 44100
HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIER)
import kit_render as _kr  # noqa: E402
AK = os.path.join(_kr.BIB_DIR, "Akustik") + "/"
PFAD = {}                             # Pfade dieses Laufs (setzt feinschliff())


def db(a):
    return 20 * np.log10(np.abs(a).max() + 1e-9) if len(a) else -120.0


def _biquad(typ, G, Q, fc, fs):
    A = 10 ** (G / 40.0); w0 = 2 * np.pi * fc / fs; al = np.sin(w0) / (2 * Q); c = np.cos(w0)
    if typ == "shelf":
        b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
             A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)]
        a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al]
    else:
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + al, -2 * c, 1 - al]
    return np.array(b) / a[0], np.array(a) / a[0]


_K = (_biquad("shelf", 3.999843853973347, 0.7071752369554196, 1681.974450955533, SR),
      _biquad("hp", 0.0, 0.5003270373238773, 38.13547087602444, SR))


def _lautheit(x, block=0.1, ueberlapp=0.75):
    """Lautheit nach ITU-R BS.1770 (K-Gewichtung + Gating), wie pyloudnorm – ohne Zusatzpaket."""
    from scipy.signal import lfilter
    x = x if x.ndim > 1 else x[:, None]
    z = x.astype(np.float64)
    for b, a in _K:
        z = lfilter(b, a, z, axis=0)
    T = len(z) / SR
    if T < block:
        return -70.0
    nb = int(np.round((T - block) / (block * (1 - ueberlapp)))) + 1      # Blockeinteilung genau wie pyloudnorm
    ms = np.array([np.sum(z[int(block * j * (1 - ueberlapp) * SR):int(block * (j * (1 - ueberlapp) + 1) * SR)] ** 2,
                          axis=0) / (block * SR) for j in range(nb)])
    lj = -0.691 + 10 * np.log10(ms.sum(1) + 1e-20)
    g = lj >= -70
    if not g.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(ms[g].sum(1).mean() + 1e-20) - 10
    g2 = g & (lj > rel)
    if not g2.any():
        return -70.0
    return float(-0.691 + 10 * np.log10(ms[g2].sum(1).mean() + 1e-20))


def lufs(x, t, d):
    s = x[int(t * SR):int((t + d) * SR)]
    if len(s) < int(0.1 * SR):
        return -70.0
    v = _lautheit(s)
    return v if np.isfinite(v) else -70.0


def lese(p, n=None, mono=False):
    x, s = sf.read(p, always_2d=True)
    if s != SR:
        x = librosa.resample(x.T.astype(np.float32), orig_sr=s, target_sr=SR).T
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    x = x[:, :2].astype(np.float32)
    if n is not None:
        x = x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))
    return x.mean(1) if mono else x


def sample(rel, lang=None):
    y = lese(AK + rel)
    i = int(np.argmax(np.abs(y).max(1) > np.abs(y).max() * 0.1))
    y = y[max(0, i - 22):]
    if lang:
        L = int(lang * SR); y = y[:L].copy(); fo = int(0.15 * SR); y[-fo:] *= np.linspace(1, 0, fo)[:, None]
    return y


def setze(y, s, t, pegel_db):
    g = 10 ** (pegel_db / 20) / (np.abs(s).max() + 1e-9)
    i = int(t * SR) - int(np.argmax(np.abs(s).max(1) if s.ndim > 1 else np.abs(s)))   # Spitze des Samples auf t
    if i < 0:
        s = s[-i:]; i = 0
    e = min(len(y), i + len(s))
    y[i:e] += s[:e - i] * g


def treffer(x, delta=0.04, unter=-35):
    """Schläge: (Anschlag-Zeit der Spitze, Pegel bis zum nächsten Schlag)."""
    on = librosa.onset.onset_detect(y=x, sr=SR, hop_length=128, units="samples", backtrack=True, delta=delta)
    out = []
    for k, i in enumerate(on):
        e = min(on[k + 1] if k + 1 < len(on) else len(x), i + int(0.15 * SR))
        lv = db(x[i:e])
        if lv >= unter:
            out.append((i + int(np.argmax(np.abs(x[i:e]))), lv, i))
    return out


F16 = np.fft.rfftfreq(16384, 1 / SR)


def hoch_anteil(x, pk):
    w = x[pk:pk + int(0.1 * SR)]
    W = np.abs(np.fft.rfft(w * np.hanning(len(w)), 16384)) ** 2
    return 10 * np.log10(W[(F16 > 650) & (F16 < 820)].sum() / (W[(F16 > 220) & (F16 < 480)].sum() + 1e-12) + 1e-12)


# ----------------------------------------------------------------------------------------------------- Snare
def snare(d, n, ber, rep_ber):
    m = lese(f"{PFAD['teile']}/snare.flac", n, True)
    to = lese(f"{PFAD['teile']}/toms.flac", n, True)
    s53 = lese(PFAD["s53_snare"], n, True) if PFAD.get("s53_snare") else None
    T = []
    tr = treffer(m)
    # Artefakt-Grenze relativ zum Song: wie stark hört das 53er die kräftigen Snare-Schläge dieses Songs?
    abst = [db(s53[i:min(n, i + int(0.15 * SR))]) - lv if s53 is not None else 0.0 for pk, lv, i in tr]
    kraeftig = [a_ for a_, (pk, lv, i) in zip(abst, tr) if lv > max(l for _, l, _ in tr) - 8] if tr else []
    grenze_a = (float(np.median(kraeftig)) - 20.0) if (kraeftig and s53 is not None) else -999.0   # ohne 53er: keine Artefakt-Prüfung
    ber["snare_artefakt_grenze_db"] = round(grenze_a, 1)
    for (pk, lv, i), a_ in zip(tr, abst):
        e = min(n, i + int(0.15 * SR))
        if a_ < grenze_a:
            continue                                        # Artefakt: 53er hört dort (für diesen Song) keine Snare
        if lv < -12 and db(to[i:e]) >= lv - 3:
            continue                                        # Tom-Übersprechen
        T.append([pk / SR, lv, hoch_anteil(m, pk)])
    if not T:
        return np.zeros((n, 2), np.float32), {}
    T = np.array(T)
    grenze = max(-2.0, float(np.median(T[:, 2])) + 6.0)       # hoch = klar mehr 650–820 Hz als normale Snare
    hoch = T[:, 2] > grenze
    ref_off = rep_ber.get("pegel_angeglichen_db", {}).get("snare", -3.4)
    sk = rep_ber.get("snare_klaenge", {}).get("snare_1", {}).get("klang", "Akustik/Snare/snare 2.flac")
    normal = sample(sk.replace("Akustik/", ""))
    y = np.zeros((n + SR, 2), np.float32)
    hz = {}
    if hoch.any():
        # hohen Klang wählen: Sample mit dem ähnlichsten Hoch-Anteil
        def mel(x, pk):                                       # Klangfarbe 150–6000 Hz direkt nach dem Anschlag
            w = x[pk:pk + int(0.12 * SR)]; w = np.pad(w, (0, max(0, int(0.12 * SR) - len(w))))
            S = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=4096, hop_length=1024, n_mels=48, fmin=150, fmax=6000).mean(1)
            return 10 * np.log10(S / S.max() + 1e-9)
        ref_m = np.median([mel(m, int(T[k, 0] * SR)) for k in np.where(hoch)[0]], 0); beste = None
        for p in sorted(glob.glob(AK + "Snare_hoch/*")) + sorted(glob.glob(AK + "Timbales_RotoToms/*")):
            s = sample(os.path.relpath(p, AK), 0.6); sm = s.mean(1)
            a_ = float(np.sqrt(np.mean((mel(sm, int(np.argmax(np.abs(sm)))) - ref_m) ** 2)))
            if beste is None or a_ < beste[0]:
                beste = (a_, s, os.path.basename(p))
        hs = beste[1]; ber["snare_hoch_klang"] = beste[2]
        lp = butter(2, 3500, "low", fs=SR, output="sos")
        idx = np.where(hoch)[0]; ref = T[idx, 1].max()
        gruppen = []
        for k in idx:
            if gruppen and T[k, 0] - T[gruppen[-1][-1], 0] < 0.4:
                gruppen[-1].append(k)
            else:
                gruppen.append([k])
        rng = np.random.default_rng(3)
        for G in gruppen:
            if len(G) >= 3:                                   # Fill: Steigerung zum Akzent am Ende
                ramp = np.linspace(-7, 0, len(G))
                for j, k in enumerate(G):
                    hz[k] = max(ref - 16, ref + 2.2 * (T[k, 1] - ref) + ramp[j]) - 1.0
                hz[G[-1]] = ref - 1.0
            else:
                for k in G:
                    hz[k] = ref + 1.6 * (T[k, 1] - ref) + rng.uniform(-1, 1)
    def baue(extra):
        y = np.zeros((n + SR, 2), np.float32)
        for k, (t, lv, _) in enumerate(T):
            if hoch[k]:
                neu = hz[k] + extra.get(k, 0); weich = np.clip((T[hoch, 1].max() - neu) / 14, 0, 1)
                s = hs * (1 - weich) + sosfilt(lp, hs, axis=0).astype(np.float32) * weich
                setze(y, s, t, neu + 9.0 + ref_off)
            else:
                setze(y, normal, t, lv + ref_off)
        return y[:n]
    y = baue({})
    norm_l = np.median([lufs(y, t, 0.15) for k, (t, lv, _) in enumerate(T) if not hoch[k] and lv > T[:, 1].max() - 6] or [-18])
    if hoch.any():
        extra = {}
        for G in gruppen:
            if len(G) < 3:
                for k in G:
                    extra[k] = max(0.0, norm_l + 2 - lufs(y, T[k, 0], 0.15))
        y = baue(extra)
    ber.update(snare_normal=int((~hoch).sum()), snare_hoch=int(hoch.sum()),
               snare_hoch_zeiten=[round(float(t), 2) for t in T[hoch, 0]], snare_lautheit=round(float(norm_l), 1))
    return y, {"normal_lufs": norm_l}


# ----------------------------------------------------------------------------------------------------- Kick
def kick(d, n, ber, rep_ber):
    o = lese(f"{PFAD['teile']}/kick.flac", n, True)
    H = treffer(o, 0.04, -36)
    if not H:
        return np.zeros((n, 2), np.float32)
    N = 16384; f = np.fft.rfftfreq(N, 1 / SR)
    P = lambda w: np.abs(np.fft.rfft(np.pad(w, (0, max(0, N - len(w))))[:N] * np.hanning(N))) ** 2
    laut = [i for pk, lv, i in H if lv > max(l for _, l, _ in H) - 8]
    Po = np.mean([P(o[i:i + int(0.3 * SR)]) for i in laut], 0)
    fb = np.geomspace(20, 20000, 60)
    G = lambda p: np.array([p[(f > q / 1.12) & (f < q * 1.12)].mean() + 1e-20 for q in fb])
    beste = None
    for p in sorted(glob.glob(AK + "Kick/*")):
        s = sample(os.path.relpath(p, AK)); dd = 10 * np.log10(G(Po) / G(P(s.mean(1)[:int(0.3 * SR)])))
        dd -= np.interp(60, fb, dd); a = float(np.sqrt(np.mean(dd ** 2)))
        if beste is None or a < beste[0]:
            beste = (a, s, dd, os.path.basename(p))
    _, x, dd, name = beste
    dd = np.clip(0.5 * dd, -6, 3)                               # sanft angleichen
    fr = np.clip(np.concatenate([[0], fb / (SR / 2), [1]]), 0, 1); g = 10 ** (np.concatenate([[dd[0]], dd, [dd[-1]]]) / 20)
    h = firwin2(2049, fr, g)
    xe = np.stack([fftconvolve(x[:, c], h)[1024:1024 + len(x)] for c in range(2)], 1).astype(np.float32)
    off = rep_ber.get("pegel_angeglichen_db", {}).get("kick", 3.5)
    y = np.zeros((n + SR, 2), np.float32)
    sp = int(np.argmax(np.abs(xe).max(1)))
    for pk, lv, i in H:
        setze(y, xe, pk / SR, lv + off)
    ber.update(kick_klang=name, kick_schlaege=len(H))
    return y[:n]


# ----------------------------------------------------------------------------------------------------- Toms
def toms(d, n, ber, norm_l):
    r = lese(f"{PFAD['rep']}/toms_neu.flac", n)
    o = lese(f"{PFAD['teile']}/toms.flac", n, True)
    on = librosa.onset.onset_detect(y=r.mean(1), sr=SR, hop_length=128, units="samples", backtrack=True, delta=0.05)
    H = [(i / SR, db(o[i:i + 6000])) for i in on if db(r.mean(1)[i:i + 6000]) > -40]
    if not H:
        return r
    omax = max(l for _, l in H); g = np.ones(n); ramp = int(0.005 * SR)
    for t, lv in H:
        i = max(0, int(t * SR) - int(0.01 * SR)); soll = norm_l + 6.0 + 0.4 * (lv - omax)
        gg = 10 ** ((soll - lufs(r, t, 0.12)) / 20)
        alt = g[i - 1] if i > 0 else 1; g[i:i + ramp] = np.linspace(alt, gg, len(g[i:i + ramp])); g[i + ramp:] = gg
    ber["toms_schlaege"] = len(H)
    return r * g[:, None]


# ----------------------------------------------------------------------------------------------------- Becken
def becken(d, n, ber):
    r = lese(f"{PFAD['rep']}/becken_neu.flac", n)
    hp = butter(4, 6000, "high", fs=SR, output="sos")
    B = sosfiltfilt(hp, lese(f"{PFAD['teile']}/becken.flac", n, True))
    rms = lambda a, b: 20 * np.log10(np.sqrt(np.mean(B[int(a * SR):int(b * SR)] ** 2)) + 1e-9)
    on = librosa.onset.onset_detect(y=r.mean(1), sr=SR, hop_length=256, units="samples", backtrack=True, delta=0.05)
    gespielt = [i / SR for i in on if db(r.mean(1)[i:i + 6000]) > -45]
    # fehlende Becken: im Original klar angeschlagen UND lange nachklingend (Snare klingt nie so lange)
    ob = librosa.onset.onset_detect(y=lese(f"{PFAD['teile']}/becken.flac", n, True), sr=SR, hop_length=256,
                                    units="samples", backtrack=True, delta=0.05)
    # Maßstab: Anschlag der schon gespielten Becken (sonst die kräftigsten Stellen der Becken-Spur)
    anschl = [rms(t, t + 0.05) for t in gespielt] or [np.percentile([rms(t, t + 0.05) for t in np.arange(0, n / SR - 1, 0.05)], 99)]
    schwelle = float(np.median(anschl)) - 10
    neu = []
    for i in ob:
        t = i / SR
        if t > n / SR - 1 or any(abs(t - g_) < 1.0 for g_ in gespielt + neu):
            continue
        a, nach = rms(t, t + 0.05), rms(t + 0.3, t + 0.4)
        if a > schwelle and nach > a - 8:
            neu.append(t)
    y = r.copy()
    if neu:
        x = sample("Crash/Cymb01.flac")
        for t in neu:
            st = int(t * SR); seg = x[:n - st]; y[st:st + len(seg)] += seg * 0.3
    alle = sorted(gespielt + neu)
    if alle:
        L = [lufs(y, t, 0.5) for t in alle]
        ziel = float(np.median(sorted(L)[len(L) // 2:]))          # so laut wie die kräftigen Becken
        g = np.ones(n)
        for t, l in zip(alle, L):
            i = int(t * SR) - 220
            gg = 10 ** ((ziel - l) / 20)
            g[max(0, i):max(0, i) + 220] = np.linspace(g[max(0, i) - 1] if i > 0 else 1, gg, 220); g[max(0, i) + 220:] = gg
        y = y * g[:, None]
    ber.update(becken_gespielt=[round(t, 2) for t in gespielt], becken_ergaenzt=[round(t, 2) for t in neu])
    return y


# ----------------------------------------------------------------------------------------------------- Hi-Hat
def hihat(d, n, ber, rep_ber):
    """Hi-Hat-Muster über einen ganzen Takt (4 Beats) gelernt – so passen auch Muster, die nicht jeden Beat gleich sind
    (z. B. offene Hi-Hat nur auf 4, Achtel-/Sechzehntel-Wechsel)."""
    if not PFAD.get("s53_hh"):
        ber["hihat"] = "keine 53er-Hi-Hat (Eingabe war eine Drum-Spur) – Reparatur-Hi-Hat bleibt"
        return lese(f"{PFAD['rep']}/hihat_neu.flac", n)
    dr = lese(PFAD["drums"], n, True)
    h53 = lese(PFAD["s53_hh"], n, True)
    ki = lese(f"{PFAD['teile']}/kick.flac", n, True); sn = lese(f"{PFAD['teile']}/snare.flac", n, True)
    tempo, beats = librosa.beat.beat_track(y=dr, sr=SR, hop_length=256, units="time")
    beats = np.asarray(beats)
    on = librosa.onset.onset_detect(y=h53, sr=SR, hop_length=128, units="samples", backtrack=True, delta=0.03)
    H = []
    for k, i in enumerate(on):
        e = min(on[k + 1] if k + 1 < len(on) else n, i + int(0.1 * SR))
        spitze = np.abs(h53[i:e]).max() + 1e-9
        pk = i + int(np.argmax(np.abs(h53[i:min(e, i + int(0.03 * SR))])))     # Zeit = Spitze (wie Kick/Snare)
        H.append((pk / SR, db(h53[i:e]), db(ki[i:e]) > -20 or db(sn[i:e]) > -20,
                  float(np.sum(np.abs(h53[i:i + int(0.3 * SR)]) > spitze * 0.1)) / SR))
    TB = 4
    if len(H) < 8 or len(beats) < 2 * TB + 1:
        ber["hihat"] = "zu wenig erkannt – Reparatur-Hi-Hat bleibt"
        return lese(f"{PFAD['rep']}/hihat_neu.flac", n)
    H = np.array(H)
    H = H[H[:, 1] > np.percentile(H[:, 1], 95) - 30]
    nt = (len(beats) - 1) // TB                                 # ganze Takte
    def lage(t):                                                # Zeit -> (Takt, Lage 0..1 im Takt)
        b = np.searchsorted(beats, t) - 1
        if b < 0 or b >= nt * TB: return -1, -1.0
        return b // TB, ((b % TB) + (t - beats[b]) / (beats[b + 1] - beats[b])) / TB
    def zeit(takt, p):
        x = p * TB; b = takt * TB + int(x); return beats[b] + (x - int(x)) * (beats[b + 1] - beats[b])
    L = np.array([lage(t) for t in H[:, 0]])
    ok = L[:, 0] >= 0
    hist, kanten = np.histogram(L[ok, 1], bins=TB * 24, range=(0, 1))
    hist = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), [1, 2, 3, 2, 1], "same")[2:-2]
    nb = len(hist)
    gipfel = [j for j in range(nb) if hist[j] == max(hist[max(0, j - 3):j + 4]) and hist[j] > hist.max() * 0.15]
    PH = sorted({round((kanten[j] + kanten[j + 1]) / 2, 4) for j in gipfel})
    zeilen = []
    for tk in range(nt):
        for s, p in enumerate(PH):
            t = zeit(tk, p)
            k = np.where(np.abs(H[:, 0] - t) < 0.045)[0]
            if len(k):
                j = k[np.argmax(H[k, 1])]; zeilen.append((tk, s, H[j, 0], 1, H[j, 1], H[j, 2], H[j, 3]))
            else:
                zeilen.append((tk, s, t, 0, -99, 0, 0))
    R = np.array(zeilen)
    aktiv = sorted(set(R[R[:, 3] == 1, 0].astype(int)))
    gueltig = [s for s in range(len(PH)) if ((R[:, 1] == s) & (R[:, 3] == 1)).sum() >= 0.3 * len(aktiv)]
    # sauberste Stelle: der Takt, in dem die meisten gültigen Stellen gehört werden und am kräftigsten sind
    bestes = None
    for tk in aktiv:
        k = (R[:, 0] == tk) & np.isin(R[:, 1], gueltig)
        sc = (R[k, 3].sum(), np.mean(R[k & (R[:, 3] == 1), 4]) if (R[k, 3] == 1).any() else -99)
        if bestes is None or sc > bestes[0]:
            bestes = (sc, tk)
    t0 = bestes[1]
    muster = {}
    for s in gueltig:
        alle_s = (R[:, 1] == s) & (R[:, 3] == 1)
        hier = (R[:, 0] == t0) & (R[:, 1] == s) & (R[:, 3] == 1)
        if R[alle_s, 5].mean() > 0.5:
            lev = None                                          # meist von Kick/Snare verdeckt
        elif hier.any() and not R[hier, 5].any():
            lev = float(R[hier, 4][0])
        else:
            frei = alle_s & (R[:, 5] == 0)
            lev = float(np.percentile(R[frei, 4], 75)) if frei.any() else None
        lg = float(np.median([lage(t)[1] for t in R[alle_s, 2]])) if alle_s.any() else PH[s]   # Lage: Mittel aller gehörten
        muster[s] = [lev, lg, float(np.median(R[alle_s, 6])) > 0.15]
    frei = [v[0] for v in muster.values() if v[0] is not None]
    for s in muster:
        if muster[s][0] is None:
            muster[s][0] = (max(frei) - 3.0) if frei else -20.0
    # Regelmäßig machen: Schläge, die an derselben Stelle im Beat in mind. 2 der 4 Beats vorkommen, gelten für jeden Beat
    # (gleiches Muster, gleiche Lautstärke); was nur einmal im Takt kommt (z. B. offene Hi-Hat auf 4), bleibt einmal im Takt.
    q = {s: ((muster[s][1] * TB) % 1.0) for s in muster}
    q = {s: (v - 1.0 if v > 0.9 else v) for s, v in q.items()}            # 0,98 und 0,02 sind dieselbe Stelle
    gruppen = []
    for s in sorted(muster, key=lambda s: q[s]):
        if gruppen and q[s] - q[gruppen[-1][-1]] < 0.06: gruppen[-1].append(s)
        else: gruppen.append([s])
    neu_m = {}
    for G in gruppen:
        pos = {int(np.floor(muster[s][1] * TB + 0.5 - q[s])) % TB for s in G}
        if len(pos) >= 2:
            qm = float(np.median([q[s] for s in G]))
            frei_g = (np.isin(R[:, 1], G)) & (R[:, 3] == 1) & (R[:, 5] == 0)
            verdeckt = R[np.isin(R[:, 1], G) & (R[:, 3] == 1), 5].mean() > 0.5
            # Pegel der Stelle: oberes Viertel aller frei gehörten Schläge (die Trennung frisst eher, als dass sie dazudichtet)
            lv = None if (verdeckt or not frei_g.any()) else float(np.percentile(R[frei_g, 4], 75))
            of = sum(muster[s][2] for s in G) > len(G) / 2
            for j in range(TB):
                p = (j + qm) / TB
                neu_m[len(neu_m)] = [lv, p + (1.0 if p < 0 else 0.0), of]
        else:
            for s in G: neu_m[len(neu_m)] = list(muster[s])
    frei2 = [v[0] for v in neu_m.values() if v[0] is not None]
    for v in neu_m.values():
        if v[0] is None: v[0] = (max(frei2) - 3.0) if frei2 else -20.0      # verdeckt: Akzent = kräftigster freier −3 dB
    muster = neu_m
    A = set(aktiv)
    for tk in range(1, nt - 1):
        if tk - 1 in A and tk + 1 in A: A.add(tk)
    rep = lese(f"{PFAD['rep']}/hihat_neu.flac", n, True)
    ron = librosa.onset.onset_detect(y=rep, sr=SR, hop_length=128, units="samples", backtrack=True, delta=0.03)
    rp = [db(rep[i:i + 3000]) for i in ron]
    ziel_max = float(np.percentile(rp, 90)) if rp else -15.0
    mx = max(v[0] for v in muster.values())
    rang = sorted(muster, key=lambda s: -muster[s][0]); zu = [1, 3, 4, 2]; K = {}
    for r_, s in enumerate(rang):
        K[s] = sample(f"HiHat_offen/piatto aperto {1 + r_ % 3}.flac") if muster[s][2] else sample(f"HiHat_zu/piatto chiuso {zu[r_ % 4]}.flac")
    # Zweite Quelle fürs Timing: Anschläge der DrumSep-Hi-Hat (Spitzen)
    dsh = lese(f"{PFAD['teile']}/hihat.flac", n, True)
    don = librosa.onset.onset_detect(y=dsh, sr=SR, hop_length=128, units="samples", backtrack=True, delta=0.03)
    DS = np.array([(i + int(np.argmax(np.abs(dsh[i:i + int(0.03 * SR)])))) / SR for i in don]) if len(don) else np.zeros(0)
    y = np.zeros((n + SR, 2), np.float32); zahl = 0
    for tk in sorted(A):
        for s in muster:
            lev, lg, _ = muster[s]
            t = zeit(tk, lg)
            k = np.where(np.abs(H[:, 0] - t) < 0.045)[0]
            if len(k):
                t = float(H[k[np.argmin(np.abs(H[k, 0] - t))], 0])
            elif len(DS) and np.min(np.abs(DS - t)) < 0.08:
                t = float(DS[np.argmin(np.abs(DS - t))])
            setze(y, K[s], t, ziel_max + (lev - mx)); zahl += 1
    ber["hihat"] = {"tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 1), "stellen_im_takt": sorted(round(v[1] * TB, 2) for v in muster.values()),
                    "gespielte_stellen": len(muster), "mustertakt_s": round(float(beats[t0 * TB]), 2),
                    "muster_db": {round(muster[s][1] * TB, 2): round(muster[s][0] - mx, 1) for s in sorted(muster, key=lambda s: muster[s][1])},
                    "offen": [round(muster[s][1] * TB, 2) for s in muster if muster[s][2]], "schlaege": zahl}
    return y[:n]


def feinschliff(drums, teile, rep, s53_snare=None, s53_hh=None, melde=print):
    """Feinschliff auf eine fertige Reparatur (rep = Ordner mit *_neu.flac + bericht.json). Gibt die fünf neuen
    Teile (Stereo, gleiche Länge) und einen Bericht zurück. Schreibt selbst nichts."""
    PFAD.clear()
    PFAD.update(drums=drums, teile=teile, rep=rep, s53_snare=s53_snare, s53_hh=s53_hh)
    n = len(lese(PFAD["drums"], mono=True))
    rep_ber = json.load(open(f"{PFAD['rep']}/bericht.json"))
    ber = {}
    d = None
    melde("… Feinschliff Snare (Haupt-Snare, hohe Snare, Fills)")
    sn, info = snare(d, n, ber, rep_ber)
    melde("… Feinschliff Kick, Toms, Becken")
    teile_ = {"kick": kick(d, n, ber, rep_ber), "snare": sn, "toms": toms(d, n, ber, info.get("normal_lufs", -18)),
              "becken": becken(d, n, ber)}
    melde("… Feinschliff Hi-Hat (Muster des Songs)")
    teile_["hihat"] = hihat(d, n, ber, rep_ber)
    return {k: v[:n] for k, v in teile_.items()}, ber


if __name__ == "__main__":       # Test: python drum_feinschliff.py <ordner mit drums.wav, teile_sauber/, s53/, rep/>
    o = sys.argv[1]
    T, B = feinschliff(f"{o}/drums.wav", f"{o}/teile_sauber", f"{o}/rep", f"{o}/s53/snare.wav", f"{o}/s53/hh.wav")
    os.makedirs(f"{o}/fein", exist_ok=True)
    for k, v in T.items():
        sf.write(f"{o}/fein/{k}.flac", v, SR, subtype="PCM_24")
    json.dump(B, open(f"{o}/fein/bericht_fein.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(B, ensure_ascii=False))
