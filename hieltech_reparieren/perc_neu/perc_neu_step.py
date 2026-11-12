"""HIELTECH Stems – Percussion reparieren (Nov 2026, v2: alle Percussion-Gruppen, Klang-Gruppen zum Zuordnen).

Jeder Schlag der getrennten Percussion-Spur wird gefunden, eingeordnet und durch einen echten Einzelschlag ersetzt:
gleiche Zeit, gleiche Stärke, gleiche Links/Rechts-Lage.
  Trommeln (Hand-Trommeln, nach Tonhöhe in Stimmen getrennt, mit Slaps) -> Djembe / Bongo / Conga
  Schellen (Tambourin)                                                    -> vier Tambourin-Sätze
  Shaker / Rasseln                                                        -> Egg-Shaker / Maracas / Shaker / Cabasa
  Glocken (Agogo, Cowbell, Bell-Stick, Triangel, Mark Tree – auf die Tonhöhe des Originals gestimmt)
  Holz (Sticks, Wood Block, Claves, Kastagnetten) · Klatschen (Hand-Clap)
Jede Gruppe kann auch „Original behalten“ oder „aus“ sein. Trommeln lassen sich hoch/runter stimmen.
Mit --nur <gruppe> wird nur diese Gruppe bearbeitet, alles andere bleibt original (Schritt für Schritt).

Zwei Wege:
  automatisch: Trommeln / Tambourin / Shaker werden selbst erkannt (sicher erprobt).
  mit Analyse: „analyse“ sortiert alle Schläge in Klang-Gruppen und schreibt je Gruppe eine Hörprobe;
               der Nutzer sagt je Klang-Gruppe, was es ist (--zuordnung), dann wird repariert.

Aufruf: python perc_neu_step.py <percussion.wav> <ausgabe-ordner> [--congas c.wav] [--song song.wav]
        [--trommel djembe|bongo|conga|original|aus] [--stimmung -6..6] [--schelle ...] [--shaker ...] [--midi-bpm 120]
"""
import os, sys, glob, json, struct, argparse
import numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfiltfilt, firwin2, fftconvolve

SR = 44100
APP = os.path.dirname(os.path.abspath(__file__))
PROBEN = os.path.join(APP, "samples")
PROBEN_ZIP = os.path.join(APP, "samples.zip")          # alle Einzelschläge in einer Datei (FLAC im Zip)
HP200 = butter(2, 200, "highpass", fs=SR, output="sos")

TROMMELN = ("djembe", "bongo", "conga", "darbuka", "cajon", "goblet", "original", "aus")
SCHELLEN = ("freepats", "headless", "alt12", "klein", "original", "aus")
SHAKER = ("egg", "maracas", "shaker", "cabasa", "original", "aus")
GLOCKEN = ("agogo", "cowbell", "bellstick", "triangel", "markbaum", "original", "aus")
HOLZ = ("sticks", "woodblock", "claves", "kastagnetten", "original", "aus")
KLATSCHEN = ("handclap", "original", "aus")
GRUPPEN = ("trommel", "schelle", "shaker", "glocke", "holz", "klatschen")
NAMEN = {"djembe": "Djembe", "bongo": "Bongo", "conga": "Conga", "darbuka": "Darbuka", "cajon": "Cajón", "goblet": "Goblet-Trommel",
         "agogo": "Agogo", "cowbell": "Cowbell", "bellstick": "Bell-Stick", "triangel": "Triangel", "markbaum": "Mark Tree",
         "sticks": "Sticks", "woodblock": "Wood Block", "claves": "Claves", "kastagnetten": "Kastagnetten", "handclap": "Hand-Clap", "freepats": "Tambourin", "headless": "Tambourin ohne Fell",
         "alt12": "Altes 12-Zoll-Tambourin", "klein": "Kleines Tambourin", "egg": "Egg-Shaker", "maracas": "Maracas",
         "shaker": "Shaker", "cabasa": "Cabasa", "original": "Original", "aus": "aus"}


def melde(t):
    print(t, flush=True)


def lade_stereo(p):
    x, sr = sf.read(p, always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    x = x[:, :2]
    if sr != SR:
        x = np.stack([librosa.resample(x[:, c], orig_sr=sr, target_sr=SR) for c in range(2)], 1)
    return x


def lade_mono(p):
    return lade_stereo(p).mean(1)


# ------------------------------------------------------------ 1) Schläge finden und einordnen
def tonhoehe(y, a, lo=100, hi=2500):
    seg = y[a + int(0.005 * SR):a + int(0.005 * SR) + 8192]
    seg = np.pad(seg, (0, 8192 - len(seg)))
    S = np.abs(np.fft.rfft(seg * np.hanning(8192))) ** 2
    f = np.fft.rfftfreq(8192, 1 / SR)
    m = (f > lo) & (f < hi)
    i = np.where(m)[0][np.argmax(S[m])]
    if 0 < i < len(S) - 1:
        a_, b_, c_ = np.log(S[i - 1] + 1e-20), np.log(S[i] + 1e-20), np.log(S[i + 1] + 1e-20)
        i = i + 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_ + 1e-12)
    return float(max(i, 1) * SR / 8192)


def schlaege(st):
    y = st.mean(1)
    yh = sosfiltfilt(HP200, y)
    on = librosa.onset.onset_detect(y=y, sr=SR, units="samples", backtrack=True, delta=0.05, hop_length=128)
    if not len(on):
        return []
    pk = np.array([np.abs(y[o:o + int(0.03 * SR)]).max() for o in on])
    out = []
    for o in on[pk > 0.03 * pk.max()]:
        a = o + int(np.argmax(np.abs(y[o:o + int(0.025 * SR)])))
        if out and (a - out[-1]["a"]) < int(0.035 * SR):
            continue
        seg = y[o:o + int(0.08 * SR)]; seg = np.pad(seg, (0, int(0.08 * SR) - len(seg)))
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2; f = np.fft.rfftfreq(len(seg), 1 / SR); tot = S.sum() + 1e-12
        lang = y[a:a + int(0.4 * SR)]; e = np.cumsum(lang ** 2)
        L = np.sqrt(np.mean(st[a:a + int(0.04 * SR), 0] ** 2)); R = np.sqrt(np.mean(st[a:a + int(0.04 * SR), 1] ** 2))
        out.append(dict(a=int(a), t=a / SR, spitze=float(np.abs(y[a])),
                        spitze_hp=float(np.abs(yh[max(0, a - 64):a + int(0.01 * SR)]).max()),
                        hf=float(S[f > 5000].sum() / tot), mitt=float(S[(f > 1000) & (f < 5000)].sum() / tot),
                        abk=float(np.argmax(e > 0.85 * e[-1]) / SR) if e[-1] > 0 else 0.0,
                        pan=float((R - L) / (R + L + 1e-12)), f0=tonhoehe(y, o)))
    for i, s in enumerate(out):
        ende = out[i + 1]["a"] if i + 1 < len(out) else len(y)
        s["ende"] = int(ende)
        s["ton_f"], s["ton"] = nachklang_ton(y, s["a"], ende)
    return out


MEL = None


def fingerabdruck(y, a, ende):
    """Klang-Fingerabdruck eines Schlags: Mel-Spektrum des Anschlags (46 ms) + Ausklang-Länge."""
    global MEL
    if MEL is None:
        MEL = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=48, fmin=60, fmax=16000)
    seg = y[a:a + 2048]; seg = np.pad(seg, (0, 2048 - len(seg)))
    m = np.log(MEL @ (np.abs(np.fft.rfft(seg * np.hanning(2048))) ** 2) + 1e-10)
    z = y[a:max(a + 256, min(ende, a + int(0.25 * SR)))]
    env = np.sqrt(np.convolve(z ** 2, np.ones(256) / 256, "same"))
    pk = env[:1323].max() + 1e-12
    u = np.where(env[441:] < pk * 0.1)[0]
    ab = (441 + u[0]) / SR if len(u) else len(z) / SR
    return np.r_[m - m.max(), 8 * np.log(ab + 0.01)]


def nachklang_ton(y, a, ende):
    """Stärkste Tonhöhe im Nachklang (40–200 ms) und wie tonhaft er ist (Anteil der 5 stärksten Linien)."""
    n2 = y[a + int(0.04 * SR):min(ende, a + int(0.2 * SR))]
    if len(n2) < 2048:
        return 0.0, 0.0
    n2 = n2[:4096] if len(n2) >= 4096 else n2[:2048]
    S2 = np.abs(np.fft.rfft(n2 * np.hanning(len(n2)))) ** 2; f2 = np.fft.rfftfreq(len(n2), 1 / SR)
    m2 = (f2 > 300) & (f2 < 12000); v = S2[m2]
    return float(f2[m2][np.argmax(v)]), float(np.sort(v)[-5:].sum() / (v.sum() + 1e-12))


def auto_art(s):
    """Automatische Einordnung (sicher erprobt): Trommel / Tambourin / Shaker."""
    if s["hf"] >= 0.4:
        return "schelle" if s["abk"] >= 0.11 else "shaker"
    return "trommel"


def trommel_stimmen(tr):
    """Trommeln nach Tonhöhe in Stimmen (Häufungen im Halbton-Raster); setzt stimme/slap, gibt Tonhöhen zurück."""
    if not tr:
        return []
    mid = np.array([69 + 12 * np.log2(s["f0"] / 440) for s in tr])
    h = np.bincount(np.clip(np.round(mid).astype(int), 0, 127), minlength=128).astype(float)
    hg = np.convolve(h, [0.5, 1, 0.5], "same")
    mind = max(4, int(0.02 * len(tr)))
    spitzen = [k for k in range(1, 127) if hg[k] >= max(hg[k - 1], hg[k + 1]) and h[k] >= mind]
    stimmen = []
    for k in sorted(spitzen, key=lambda k: -h[k]):
        if all(abs(k - q) > 1 for q in stimmen) and len(stimmen) < 6:
            stimmen.append(k)
    stimmen = sorted(stimmen) or [int(round(float(np.median(mid))))]
    for s, m in zip(tr, mid):
        s["stimme"] = int(np.argmin([abs(m - q) for q in stimmen]))
        s["slap"] = bool(s["mitt"] >= 0.30)
    return [float(np.median([m for s, m in zip(tr, mid) if s["stimme"] == j] or [stimmen[j]])) for j in range(len(stimmen))]


# ------------------------------------------------------------ 2) Proben
def proben_dateien(ordner):
    """(name, lade-funktion) je Probe – aus dem Ordner samples/ oder aus samples.zip."""
    fs = sorted(glob.glob(os.path.join(PROBEN, ordner, "*.flac")))
    if fs:
        return [(f, lambda f=f: lade_mono(f)) for f in fs]
    import zipfile, io
    if not os.path.exists(PROBEN_ZIP):
        return []
    z = zipfile.ZipFile(PROBEN_ZIP)
    namen = sorted(n for n in z.namelist() if n.startswith(ordner.rstrip("/") + "/") and n.endswith(".flac")
                   and "/" not in n[len(ordner.rstrip("/")) + 1:])

    def lies(n):
        x, sr = sf.read(io.BytesIO(z.read(n)), always_2d=True)
        x = x.mean(1)
        return librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x
    return [(n, lambda n=n: lies(n)) for n in namen]


def pool(ordner, stimmung=0.0):
    P = []
    r = 2 ** (stimmung / 12)
    for _, laden in proben_dateien(ordner):
        x = laden()
        if abs(stimmung) > 0.01:                         # Umsampeln: Anschlag bleibt scharf
            x = np.interp(np.arange(0, len(x) - 1, r), np.arange(len(x)), x)
        x = x / (np.abs(x).max() + 1e-9)
        P.append(dict(x=x, a=int(np.argmax(np.abs(x[:int(0.05 * SR)]))),
                      spitze_hp=float(np.abs(sosfiltfilt(HP200, x)[:int(0.02 * SR)]).max()),
                      rms=float(np.sqrt(np.mean(x[:int(0.05 * SR)] ** 2)))))
    if not P:
        raise RuntimeError(f"Proben fehlen: {ordner}")
    return P


def trommel_saetze(klang, n_stimmen, stimmung):
    """Je Stimme (tief -> hoch) ein Satz offener Schläge und ein Satz Slaps."""
    if klang == "darbuka":
        t, h, s = pool("darbuka/tief", stimmung), pool("darbuka/hoch", stimmung), pool("darbuka/slap", stimmung)
        offen = [t, h, h]; slap = [h, s, s]
    elif klang == "cajon":
        t, h = pool("cajon/tief", stimmung), pool("cajon/hoch", stimmung)
        offen = [t, t, h]; slap = [h, h, h]
    elif klang == "goblet":
        t, m, h = pool("goblet/tief", stimmung), pool("goblet/mitte", stimmung), pool("goblet/hoch", stimmung)
        offen = [t, m, h]; slap = [m, h, h]
    elif klang == "conga":
        offen = [pool("conga/tief", stimmung), pool("conga/mitte", stimmung), pool("conga/hoch", stimmung)]
        slap = [pool("conga/tief_slap", stimmung), pool("conga/slap", stimmung), pool("conga/slap", stimmung)]
    elif klang == "bongo":
        t, h = pool("bongo/tief", stimmung), pool("bongo/hoch", stimmung)
        offen = [t, t, h]; slap = [t, h, h]
    else:
        b, m, s = pool("djembe/tief", stimmung), pool("djembe/mitte", stimmung), pool("djembe/slap", stimmung)
        offen = [b, m, m]; slap = [m, s, s]
    zu = list(np.round(np.linspace(0, 2, n_stimmen)).astype(int)) if n_stimmen > 1 else [1]
    return [(offen[k], slap[k]) for k in zu], zu


def probe_ton(x):
    """Stärkste Tonhöhe einer Probe im Nachklang (für Glocken)."""
    f, _ = nachklang_ton(np.r_[x, np.zeros(SR)], int(np.argmax(np.abs(x[:int(0.05 * SR)]))), len(x) + SR)
    return f


def gestimmt(P, ziel_f, max_st=6.0):
    """Proben (Glocken) auf die Tonhöhe des Originals bringen: nächste Lage, höchstens ±6 Halbtöne."""
    out = []
    for p in P:
        f = p.get("f") or probe_ton(p["x"]); p["f"] = f
        if not f or not ziel_f:
            out.append(p); continue
        st = 12 * np.log2(ziel_f / f); st -= 12 * np.round(st / 12); st = float(np.clip(st, -max_st, max_st))
        r = 2 ** (st / 12); x = np.interp(np.arange(0, len(p["x"]) - 1, r), np.arange(len(p["x"])), p["x"])
        out.append(dict(p, x=x, a=int(np.argmax(np.abs(x[:int(0.05 * SR)])))))
    return out


# ------------------------------------------------------------ 3) spielen
def spielen(hits, waehle, L, laenge=0.9):
    """hits: Schläge einer Gruppe; waehle(s) -> Probenliste. Gleiche Zeit, Stärke, Lage; Proben abwechselnd."""
    out = np.zeros((L + 2 * SR, 2))
    zaehler = {}
    gruppen = {}
    for s in hits:
        gruppen.setdefault(id(waehle(s)), []).append(s)
    for _, ss in gruppen.items():
        P = waehle(ss[0])
        ord_ = np.argsort([p["rms"] for p in P])
        sp = np.array([s["spitze"] for s in ss]); rang = np.argsort(np.argsort(sp)) / max(1, len(sp) - 1)
        for s, rg in zip(ss, rang):
            lo = int(rg * (len(P) - 1) * 0.6); hi = min(len(P), lo + max(2, int(0.4 * len(P))))
            kand = list(ord_[lo:hi]) or list(range(len(P)))
            k = id(P); i = kand[zaehler.get(k, 0) % len(kand)]; zaehler[k] = zaehler.get(k, 0) + 1
            p = P[i]; x = p["x"][:int(laenge * SR)].copy()
            fd = min(int(0.05 * SR), len(x) // 3); x[-fd:] *= np.cos(np.linspace(0, np.pi / 2, fd)) ** 2
            g = min(s["spitze_hp"] / (p["spitze_hp"] + 1e-9), 2.0 * s["spitze"] / (np.abs(x).max() + 1e-9))
            w = (np.clip(s["pan"] * 1.3, -1, 1) + 1) * np.pi / 4
            st = s["a"] - p["a"]
            if st < 0:
                x = x[-st:]; st = 0
            n = min(len(x), len(out) - st)
            out[st:st + n, 0] += x[:n] * g * np.cos(w) * np.sqrt(2)
            out[st:st + n, 1] += x[:n] * g * np.sin(w) * np.sqrt(2)
    return out[:L]


def fenster(hits, alle_t, L, nach=0.25):
    """Zeitfenster rund um die Schläge einer Gruppe (bis zum nächsten Schlag, höchstens 0,25 s), weich."""
    m = np.zeros(L)
    alle_t = np.sort(alle_t)
    for s in hits:
        j = np.searchsorted(alle_t, s["t"] + 1e-4)
        ende = min(s["t"] + nach, alle_t[j] - 0.005 if j < len(alle_t) else s["t"] + nach)
        a, b = max(0, int((s["t"] - 0.012) * SR)), min(L, int(ende * SR))
        m[a:b] = 1
    w = int(0.006 * SR)
    return np.convolve(m, np.ones(w) / w, "same")


def band(x, lo=None, hi=None):
    if lo and hi:
        sos = butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
    elif lo:
        sos = butter(4, lo, "highpass", fs=SR, output="sos")
    else:
        sos = butter(4, hi, "lowpass", fs=SR, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def pegel_angleichen(neu, ref, maske, lo, hi):
    """Lautstärke einer Gruppe an das Original (nur in den Fenstern der Gruppe, im hörbaren Band)."""
    if maske.sum() < SR * 0.05 or not np.abs(neu).max():
        return neu
    a = np.sqrt(np.sum(band(ref.mean(1), lo, hi) ** 2 * maske))
    b = np.sqrt(np.sum(band(neu.mean(1), lo, hi) ** 2 * maske))
    return neu * (a / (b + 1e-12))


def klang_angleichen(neu, ref, staerke=0.5):
    """Klangfarbe der Trommeln sanft ans Original (Terzbänder bis 4 kHz, -9/+6 dB, halbe Stärke)."""
    def spek(z):
        S = 0
        for i in range(0, len(z) - 8192, 4096 * 8):
            S = S + np.abs(np.fft.rfft(z[i:i + 8192].mean(1) * np.hanning(8192))) ** 2
        return S
    if len(neu) < 8192 * 4:
        return neu
    f = np.fft.rfftfreq(8192, 1 / SR); Sr, Sn = spek(ref), spek(neu)
    mitten = 2 ** np.arange(np.log2(50), np.log2(18000), 1 / 3)
    db = np.array([10 * np.log10((Sr[(f >= m / 2 ** (1 / 6)) & (f < m * 2 ** (1 / 6))].sum() + 1e-12) /
                                 (Sn[(f >= m / 2 ** (1 / 6)) & (f < m * 2 ** (1 / 6))].sum() + 1e-12)) for m in mitten])
    db = np.clip(np.convolve(db, np.ones(3) / 3, "same") * staerke, -9, 6)
    db[mitten < 200] = np.maximum(db[mitten < 200], -4)
    db[mitten > 4000] = 0.0
    db -= np.median(db[(mitten > 300) & (mitten < 3000)])       # nur Klangfarbe, Pegel macht pegel_angleichen
    h = firwin2(2047, np.r_[0, mitten, SR / 2] / (SR / 2), 10 ** (np.r_[db[0], db, 0] / 20))
    return np.stack([fftconvolve(neu[:, c], h, "same") for c in range(2)], 1)


def begrenzen(z, decke=0.95):
    return z * begrenz_faktor(z, decke)[:, None]


def begrenz_faktor(z, decke=0.95):
    """Verstärkungs-Verlauf des Begrenzers (gleicher Verlauf lässt sich auf Einzelspuren anwenden)."""
    a = np.abs(z).max(1)
    g = np.minimum(1.0, decke / (a + 1e-12))
    w = int(0.005 * SR)
    g = np.minimum.reduce([np.roll(g, k) for k in range(-w, w + 1, max(1, w // 8))])
    g = np.convolve(g, np.ones(w) / w, "same")
    g = np.minimum(g, 1.0)
    return g * min(1.0, decke / (np.abs(z * g[:, None]).max() + 1e-12))




# ------------------------------------------------------------ 4) Analyse: Klang-Gruppen zum Zuordnen
def kmeans(X, k, runden=40, seed=3):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):                                      # k-means++ Start
        d = np.min([np.sum((X - c) ** 2, 1) for c in C], 0)
        C.append(X[rng.choice(len(X), p=d / (d.sum() + 1e-12))])
    C = np.array(C)
    for _ in range(runden):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), 1)
        C2 = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(C2, C):
            break
        C = C2
    return lab


def zeit(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def beschreiben(ss):
    """Kurze Beschreibung einer Klang-Gruppe für den Nutzer + vorsichtiger Hinweis."""
    md = lambda k: float(np.median([s[k] for s in ss]))     # noqa: E731
    hf, mitt, abk, ton, ton_f, f0 = md("hf"), md("mitt"), md("abk"), md("ton"), md("ton_f"), md("f0")
    hell = "sehr hell" if hf >= 0.6 else ("hell" if hf + mitt >= 0.5 else ("mittel" if hf + mitt >= 0.15 else "dunkel"))
    lang = "lang nachklingend" if abk >= 0.2 else ("mittellang" if abk >= 0.1 else "kurz")
    hoehe = librosa.hz_to_note(ton_f if ton >= 0.6 and ton_f else f0).replace("♯", "#")
    hinweis = ""
    if ton >= 0.75 and abk >= 0.12 and (hf >= 0.6 or ton_f >= 700):
        hinweis = "klingt tonhaft und lang – vielleicht Glocke"
    elif abk < 0.05 and hf + mitt >= 0.5 and ton < 0.3:
        hinweis = "kurz und hell – vielleicht Holz oder Klatschen"
    return f"{hell}, {lang}, Tonhöhe um {hoehe}", hinweis


def analyse(perc, ziel, S):
    y = perc.mean(1)
    F = np.array([fingerabdruck(y, s["a"], s["ende"]) for s in S])
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    k = int(np.clip(round(len(S) / 35), 2, 12)) if len(S) >= 8 else 1
    lab = kmeans(F, k) if k > 1 else np.zeros(len(S), int)
    # Gruppen nach Größe ordnen, leere weg
    reihen = sorted({int(c) for c in lab}, key=lambda c: -int(np.sum(lab == c)))
    neu = {c: i for i, c in enumerate(reihen)}
    gruppen = []
    for c in reihen:
        ss = [s for s, l in zip(S, lab) if l == c]
        for s in ss:
            s["klang"] = neu[c]
        arten = [auto_art(s) for s in ss]
        vorschlag = max(set(arten), key=arten.count)
        text, hinweis = beschreiben(ss)
        # Hörprobe: bis zu 10 kräftige Schläge über den Song verteilt
        stark = sorted(ss, key=lambda s: -s["spitze"])[:max(10, len(ss) // 2)]
        wahl = sorted(stark, key=lambda s: s["t"])
        if len(wahl) > 10:
            wahl = [wahl[int(i)] for i in np.linspace(0, len(wahl) - 1, 10)]
        teile = []
        for s in wahl:
            a = max(0, s["a"] - int(0.005 * SR)); b = min(s["ende"] - int(0.003 * SR), a + int(0.45 * SR))
            seg = perc[a:max(b, a + int(0.06 * SR))].copy()
            f = min(len(seg) // 3, int(0.02 * SR)); seg[-f:] *= np.linspace(1, 0, f)[:, None]
            teile += [seg, np.zeros((int(0.15 * SR), 2))]
        probe = np.concatenate(teile) if teile else np.zeros((SR, 2))
        probe *= 0.9 / (np.abs(probe).max() + 1e-9)
        sf.write(os.path.join(ziel, f"klang_{neu[c]:02d}.wav"), probe, SR, subtype="PCM_16")
        gruppen.append(dict(id=neu[c], anzahl=len(ss), vorschlag=vorschlag, beschreibung=text, hinweis=hinweis,
                            von=zeit(min(s["t"] for s in ss)), bis=zeit(max(s["t"] for s in ss)),
                            ton_f=float(np.median([s["ton_f"] for s in ss])), ton=float(np.median([s["ton"] for s in ss]))))
    with open(os.path.join(ziel, "analyse.json"), "w", encoding="utf-8") as f:
        json.dump(dict(gruppen=gruppen, schlaege=[[int(s["a"]), int(s["klang"])] for s in S]), f, ensure_ascii=False)
    return gruppen


# ------------------------------------------------------------ MIDI + Notenliste (GM-Schlagzeug, Kanal 10)
GM_SCHELLE = 54
GM_SHAKER = {"egg": 70, "maracas": 70, "shaker": 70, "cabasa": 69}
GM_GLOCKE = {"agogo": 67, "cowbell": 56, "bellstick": 56, "triangel": 81, "markbaum": 84}
GM_HOLZ = {"sticks": 75, "woodblock": 76, "claves": 75, "kastagnetten": 85}
GM_OFFEN = {"conga": [64, 63, 63], "bongo": [61, 61, 60], "djembe": [64, 63, 63], "darbuka": [64, 63, 63], "cajon": [64, 63, 63],
            "goblet": [64, 63, 63]}
GM_SLAP = {"conga": [64, 62, 62], "bongo": [61, 60, 60], "djembe": [63, 62, 62], "darbuka": [63, 62, 62], "cajon": [63, 62, 62],
           "goblet": [63, 62, 62]}
GM_NAME = {39: "Hand-Clap", 54: "Tambourin", 56: "Cowbell", 60: "Bongo hoch", 61: "Bongo tief", 62: "Conga hoch gedämpft (Slap)",
           63: "Conga hoch offen", 64: "Conga tief", 67: "Agogo hoch", 69: "Cabasa", 70: "Maracas / Shaker", 75: "Claves / Sticks",
           76: "Wood Block", 81: "Triangel", 84: "Glöckchenbaum (GS)", 85: "Kastagnetten (GS)"}


def midi_und_liste(S, zu, opt, ziel, bpm):
    ppq = 960
    tick = lambda s: int(round(s * bpm / 60.0 * ppq))   # noqa: E731
    tk = opt.trommel if opt.trommel in GM_OFFEN else "conga"
    ev, liste = [], []
    for art in GRUPPEN:
        ss = [s for s in S if s["art"] == art]
        if not ss:
            continue
        sp = np.array([s["spitze"] for s in ss]); z = 20 * np.log10(sp / np.median(sp))
        for s, zz in zip(ss, z):
            if art == "trommel":
                n = (GM_SLAP if s.get("slap") else GM_OFFEN)[tk][zu[min(s.get("stimme", 0), len(zu) - 1)]]
            elif art == "schelle":
                n = GM_SCHELLE
            elif art == "shaker":
                n = GM_SHAKER.get(opt.shaker, 70)
            elif art == "glocke":
                n = GM_GLOCKE.get(opt.glocke, 56)
            elif art == "holz":
                n = GM_HOLZ.get(opt.holz, 75)
            else:
                n = 39
            v = int(np.clip(90 + 3 * zz, 30, 127))
            ev += [(tick(s["t"]), 1, 0x99, n, v), (tick(s["t"] + 0.1), 0, 0x89, n, 0)]
            liste.append((s["t"], n, v))
    ev.sort()

    def vlq(n):
        b = [n & 0x7F]; n >>= 7
        while n:
            b.insert(0, (n & 0x7F) | 0x80); n >>= 7
        return bytes(b)
    spur = lambda d: b"MTrk" + struct.pack(">I", len(d)) + d   # noqa: E731
    us = int(round(60_000_000 / bpm))
    t0 = b"\x00\xff\x51\x03" + us.to_bytes(3, "big") + b"\x00\xff\x58\x04\x04\x02\x18\x08\x00\xff\x2f\x00"
    nm = "Percussion".encode("utf-8"); d = b"\x00\xff\x03" + vlq(len(nm)) + nm
    letzt = 0
    for t_, _, a, b, c in ev:
        d += vlq(t_ - letzt) + bytes([a, b, c]); letzt = t_
    d += b"\x00\xff\x2f\x00"
    with open(os.path.join(ziel, "percussion_neu.mid"), "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + spur(t0) + spur(d))
    with open(os.path.join(ziel, "percussion_neu_Notenliste.txt"), "w", encoding="utf-8") as f:
        f.write(f"HIELTECH Stems – Percussion neu (Trommeln: {NAMEN.get(opt.trommel)}, Stimmung {opt.stimmung:+g} Halbtöne)\n")
        f.write(f"Tempo im MIDI: {bpm:g} BPM · MIDI-Kanal 10 (GM-Schlagzeug)\n\nZeit (s)   Note  Stärke  Instrument\n")
        for t_, n, v in sorted(liste):
            f.write(f"{t_:8.3f}   {n:4d}  {v:5d}   {GM_NAME.get(n, '')}\n")


# ------------------------------------------------------------ 5) neu zusammensetzen
TEIL = {  # Band und Fenster, in dem das Original einer Gruppe steckt (zum Herausnehmen)
    "trommel": (None, 6000, 0.6), "schelle": (4000, None, 0.25), "shaker": (4000, None, 0.25),
    "glocke": (None, None, 1.2), "holz": (None, None, 0.2), "klatschen": (None, None, 0.25)}


def teil_original(perc, hits, alle_t, L, g):
    lo, hi, nach = TEIL[g]
    x = band(perc, lo, hi) if (lo or hi) else perc
    return x * fenster(hits, alle_t, L, nach=nach)[:, None]


def neu_spielen(g, wahl, hits, perc, alle_t, L, ton, opt):
    """Neue Schläge einer Gruppe; gibt (Audio, zu) zurück."""
    zu = [1]
    if g == "trommel":
        saetze, zu = trommel_saetze(wahl, len(ton), opt.stimmung)
        y = spielen(hits, lambda s: saetze[min(s["stimme"], len(saetze) - 1)][1 if s["slap"] else 0], L)
        y = pegel_angleichen(y, perc, fenster(hits, alle_t, L), 200, 5000)
        return klang_angleichen(y, band(perc, hi=6000)), zu
    if g in ("schelle", "shaker"):
        P = pool(f"{g}/{wahl}")
        y = spielen(hits, lambda s, P=P: P, L)
        return pegel_angleichen(y, perc, fenster(hits, alle_t, L), 4000, 16000), zu
    if g == "glocke":
        P = pool(f"glocke/{wahl}")
        stimm = {}
        for s in hits:                                        # je Tonhöhe des Originals ein gestimmter Satz
            zf = s.get("ziel_f") or (s["ton_f"] if s["ton"] >= 0.5 else 0)
            key = int(round(12 * np.log2(zf / 440))) if zf else None
            s["_k"] = key
            if key not in stimm:
                stimm[key] = gestimmt(P, 440 * 2 ** (key / 12)) if key is not None else P
        laenge = {"triangel": 2.0, "markbaum": 3.0}.get(wahl, 0.8)
        y = spielen(hits, lambda s: stimm[s["_k"]], L, laenge=laenge)
        return pegel_angleichen(y, perc, fenster(hits, alle_t, L, nach=0.3), 300, 12000), zu
    P = pool(f"{g}/{wahl}")
    y = spielen(hits, lambda s, P=P: P, L, laenge=0.5)
    return pegel_angleichen(y, perc, fenster(hits, alle_t, L, nach=0.15), 300, 12000), zu


# ------------------------------------------------------------ Ablauf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe"); ap.add_argument("ziel")
    ap.add_argument("--modus", default="reparieren", choices=("reparieren", "analyse"))
    ap.add_argument("--analyse", default="", help="Ordner einer vorherigen Analyse (analyse.json)")
    ap.add_argument("--zuordnung", default="", help='JSON: {"0": "auto", "3": "glocke", "5": "original", ...}')
    ap.add_argument("--nur", default="alle", choices=("alle",) + GRUPPEN)
    ap.add_argument("--congas", default=""); ap.add_argument("--song", default="")
    ap.add_argument("--trommel", default="djembe", choices=TROMMELN)
    ap.add_argument("--stimmung", type=float, default=0.0)
    ap.add_argument("--schelle", default="freepats", choices=SCHELLEN)
    ap.add_argument("--shaker", default="egg", choices=SHAKER)
    ap.add_argument("--glocke", default="agogo", choices=GLOCKEN)
    ap.add_argument("--holz", default="sticks", choices=HOLZ)
    ap.add_argument("--klatschen", default="handclap", choices=KLATSCHEN)
    ap.add_argument("--midi-bpm", type=float, default=120.0)
    opt = ap.parse_args()
    opt.stimmung = float(np.clip(opt.stimmung, -6, 6))
    os.makedirs(opt.ziel, exist_ok=True)

    melde("Percussion-Spur lesen …")
    perc = lade_stereo(opt.eingabe); L = len(perc)
    S = schlaege(perc)
    if opt.congas and os.path.exists(opt.congas):
        # Congas-Spur nur zur Bestätigung: sehr leiser Trommelschlag, der dort fehlt -> Übersprechen, raus
        cg = lade_mono(opt.congas)
        on2 = librosa.onset.onset_detect(y=cg, sr=SR, units="time", backtrack=True, delta=0.05, hop_length=128)
        med = np.median([s["spitze"] for s in S]) if S else 0
        S = [s for s in S if s["hf"] >= 0.4 or (len(on2) and np.min(np.abs(on2 - s["t"])) < 0.03) or s["spitze"] > 0.5 * med]
    if not S:
        raise RuntimeError("Keine Percussion-Schläge gefunden")

    if opt.modus == "analyse":
        gruppen = analyse(perc, opt.ziel, S)
        melde(f"Analyse: {len(S)} Schläge in {len(gruppen)} Klang-Gruppen")
        return

    # --- Einordnen: automatisch oder nach Zuordnung des Nutzers
    zuordnung = json.loads(opt.zuordnung) if opt.zuordnung else {}
    klang_von = {}
    if opt.analyse and os.path.exists(os.path.join(opt.analyse, "analyse.json")):
        A = json.load(open(os.path.join(opt.analyse, "analyse.json"), encoding="utf-8"))
        klang_von = {a: c for a, c in A["schlaege"]}
        ziel_f = {str(g["id"]): (g["ton_f"] if g["ton"] >= 0.5 else 0) for g in A["gruppen"]}
        S = [s for s in S if s["a"] in klang_von]              # dieselben Schläge wie bei der Analyse
    for s in S:
        c = str(klang_von.get(s["a"], -1))
        z_ = zuordnung.get(c, "auto")
        s["art"] = auto_art(s) if z_ == "auto" else z_         # „Automatisch“: jeder Schlag einzeln wie ohne Analyse
        if s["art"] not in GRUPPEN:
            s["art"] = "original"                             # „Original lassen“ / unbekannt
        if klang_von:
            s["ziel_f"] = ziel_f.get(c, 0)
    if opt.nur != "alle":                                      # nur diese Gruppe bearbeiten, Rest bleibt
        for g in GRUPPEN:
            if g != opt.nur:
                setattr(opt, g, "original")
    tr = [s for s in S if s["art"] == "trommel"]
    ton = trommel_stimmen(tr)
    zaehl = {g: sum(1 for s in S if s["art"] == g) for g in GRUPPEN + ("original",)}
    melde(f"Schläge: {len(S)} {zaehl} · Trommel-Stimmen: {[librosa.midi_to_note(int(round(t))) for t in ton]}")

    alle_t = np.array([s["t"] for s in S])
    wahl = {g: getattr(opt, g) for g in GRUPPEN}
    behalten = zaehl["original"] > 0 or any(wahl[g] == "original" and zaehl[g] for g in GRUPPEN)
    neu = perc.copy() if behalten else np.zeros((L, 2))
    rest = neu.copy()                                          # was vom Original bleibt (eigene Spur)
    einzel = []                                                # jede Gruppe direkt in ihrer eigenen Spur
    zu = [1] * max(1, len(ton))
    for g in GRUPPEN:
        hits = [s for s in S if s["art"] == g]
        if not hits or wahl[g] == "original":
            continue
        if behalten:                                           # Original dieser Gruppe herausnehmen
            t_ = teil_original(perc, hits, alle_t, L, g)
            neu -= t_
            rest -= t_
        if wahl[g] == "aus":
            continue
        y, z_ = neu_spielen(g, wahl[g], hits, perc, alle_t, L, ton, opt)
        if g == "trommel":
            zu = z_
        neu += y
        einzel.append((g, wahl[g], y))
        melde(f"{g}: {len(hits)} Schläge neu ({NAMEN.get(wahl[g], wahl[g])})")
    fak = begrenz_faktor(neu)
    neu = neu * fak[:, None]
    sf.write(os.path.join(opt.ziel, "percussion_neu.wav"), neu, SR, subtype="PCM_24")
    spuren_n = len(einzel) + (1 if behalten else 0)
    if spuren_n > 1:                                           # Einzelspuren: Summe = percussion_neu.wav
        eo = os.path.join(opt.ziel, "Einzelspuren")
        os.makedirs(eo, exist_ok=True)
        titel = {"trommel": "Trommeln", "schelle": "Tambourin", "shaker": "Shaker", "glocke": "Glocke",
                 "holz": "Holz", "klatschen": "Klatschen"}
        for nr, (g, w_, y) in enumerate(einzel, 1):
            name = f"{nr}_{titel[g]}_{NAMEN.get(w_, w_)}".replace(" ", "_").replace("/", "-")
            sf.write(os.path.join(eo, name + ".wav"), y * fak[:, None], SR, subtype="PCM_24")
        if behalten:
            sf.write(os.path.join(eo, f"{len(einzel) + 1}_Original_behalten.wav"), rest * fak[:, None], SR, subtype="PCM_24")
        melde(f"{spuren_n} Einzelspuren geschrieben")
    melde("Percussion neu geschrieben")

    if opt.song and os.path.exists(opt.song):
        song = lade_stereo(opt.song)
        n = min(len(song), L)
        mix = song[:n] - perc[:n] + neu[:n]
        mix *= min(1.0, 0.98 / (np.abs(mix).max() + 1e-12))
        sf.write(os.path.join(opt.ziel, "song_mit_percussion_neu.wav"), mix, SR, subtype="PCM_24")
        melde("Song mit neuer Percussion geschrieben")
    midi_und_liste([s for s in S if s["art"] in GRUPPEN], zu, opt, opt.ziel, opt.midi_bpm if opt.midi_bpm > 0 else 120.0)
    bericht = dict(schlaege=len(S), arten=zaehl, stimmen=[librosa.midi_to_note(int(round(t))) for t in ton],
                   wahl=wahl, stimmung=opt.stimmung, nur=opt.nur, mit_analyse=bool(klang_von))
    with open(os.path.join(opt.ziel, "bericht.json"), "w", encoding="utf-8") as f:
        json.dump(bericht, f, ensure_ascii=False, indent=1)
    melde("Fertig")


if __name__ == "__main__":
    main()
