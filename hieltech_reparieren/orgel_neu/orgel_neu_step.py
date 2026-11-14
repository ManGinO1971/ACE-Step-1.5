#!/usr/bin/env python3
"""
HELTECH Stems – Reggae-Orgel („Bubble“) neu einspielen. Stand 5. Nov 2026, freigegeben mit „Rise Up Jah Children“
(Orgel Bubble v5, Swing). Funktioniert ohne Musikprogramm: das Ergebnis klingt fertig, MIDI zusätzlich.

Eingabe: der GANZE Song. Ablauf:
  1. Trennung in 6 Spuren (BS-Roformer-SW, Rückfall htdemucs_6s) – die Orgel steckt in „Other“ oder in „Piano“
     (13. Nov 2026: beide werden abgehört; liegt die Orgel schon aus der 53er-Trennung vor (--orgel-spur), ist sie
     die Quelle und wird im Song sauber ersetzt)
  2. Taktraster aus dem Schlagzeug, Akkorde aus allen Instrumenten (gleiche Berechnung wie Piano-Skank)
  3. Wo setzt die Orgel ein? Hör-Modell (EfficientAT, wie die Instrumenten-Erkennung) hört „Other“ ab
     (2-s-Fenster grob, 1-s-Fenster alle 0,25 s fein für den genauen Einsatz). Ab dem Einsatz spielt die Orgel
     durchgehend bis zum Song-Ende (wie Gitarre/Piano: auch dort, wo die Trennung die Orgel verliert)
  4. Bubble: rechte Hand auf dem „&“ GENAU auf dem Skank von Gitarre/Piano, linke Hand auf „e“ und „a“
     (Mitte zwischen Schlag und Skank) + Swing wie im Original gemessen (abschaltbar)
  5. Griffe je Hand aus den Tönen der Original-Orgel; zusätzliche Orgel-Töne nur wo das Hör-Modell klar Orgel hört
  6. Klang aus GeneralUser GS (Hammond, Perkussiv, Rock, Pfeife, Harmonium …), optional Leslie-Effekt,
     gleichmäßige Lautstärke wie die Original-Orgel
  7. Ausgabe: orgel_neu.wav, song_mit_neuer_orgel.wav („Other“ nur ab dem Orgel-Einsatz ersetzt), orgel_neu.mid,
     Akkord-Plan, bericht.json

Aufruf:
  python orgel_neu_step.py <song.wav> <ziel> [--klang 0:16] [--ohne-swing] [--leslie aus|langsam|schnell]
        [--ohne-zusatztoene] [--midi-bpm 120] [--lautstaerke-db 0] [--modelle ORDNER] [--spuren-ordner ORDNER]
"""
import argparse
import glob
import json
import os
import struct
import subprocess
import sys

import numpy as np
import soundfile as sf

HIER = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HIER)
sys.path.insert(0, HIER)
sys.path.insert(0, os.path.join(APP, "piano_neu"))           # SoundFont, SF2-Abspieler, Skank-Kern, Basic Pitch
import skank_kern as K                                        # noqa: E402
import sf2_spieler as sp                                      # noqa: E402
import piano_neu_step as P                                    # noqa: E402

SR = 44100
KLAENGE = {"0:16": "Hammond (Tonewheel)", "8:16": "Hammond verstimmt", "0:17": "Perkussive Orgel",
           "8:17": "Perkussive Orgel verstimmt", "0:18": "Rock-Orgel", "0:19": "Pfeifenorgel", "8:19": "Pfeifenorgel 2",
           "0:20": "Harmonium"}
DETECT_PYTHON = os.path.expanduser("~/miniforge3/envs/heltech_stems_detect/bin/python")
POS = {"e": 0.25, "&": 0.5, "a": 0.75}


def melde(t):
    print(t, flush=True)


# ------------------------------------------------------------ Hör-Modell: wo klingt Orgel?
HOER_HELFER = r'''
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, sys.argv[1])
import detect_instruments_eat as d
import librosa, torch
with contextlib.redirect_stdout(io.StringIO()):
    m, mel = d.modell()
dev = next(m.parameters()).device          # das Hör-Modell rechnet je nach Rechner auf cuda / mps / cpu
lab = d.ek.labels(d.EAT_DIR)
y, _ = librosa.load(sys.argv[2], sr=32000, mono=True)
S = librosa.stft(y, n_fft=2048, hop_length=512)
ton = librosa.istft(librosa.decompose.hpss(S, margin=(1.0, 3.0))[0], hop_length=512, length=len(y))
def orgel(sig, W, h):
    st = np.arange(0, max(1, len(sig) - W + 1), h)
    P = []
    for i in range(0, len(st), 32):
        x = torch.from_numpy(np.stack([np.pad(sig[s:s + W], (0, max(0, W - len(sig[s:s + W])))) for s in st[i:i + 32]])).float().to(dev)
        with torch.no_grad():
            p, _ = m(mel(x).unsqueeze(1))
        P.append(torch.sigmoid(p.float()).cpu().numpy())
    sc = d.ek.slug_scores(np.concatenate(P), lab)
    return st / 32000 + W / 64000, np.maximum(sc["organ"], sc["accordion"])   # Reggae-Orgel oft „Akkordeon“
out = {}
out["grob_t"], g1 = orgel(y, 64000, 32000)
out["grob"] = np.maximum(g1, orgel(ton, 64000, 32000)[1])     # ganz + Ton-Teil (wie die Erkennung)
out["fein_t"], out["fein"] = orgel(y, 32000, 8000)
np.savez(sys.argv[3], **out)
'''


try:
    sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
    from merker import merke
except Exception:  # noqa: BLE001 – ohne Merker läuft alles wie bisher
    def merke(*_a, **_k):
        return lambda fn: fn


@merke("orgel_zeitleiste", nur=(0,))
def orgel_zeitleiste(other, ziel):
    helfer = os.path.join(ziel, "_hoer_helfer.py")
    out = os.path.join(ziel, "_orgel_zeit.npz")
    with open(helfer, "w") as f:
        f.write(HOER_HELFER)
    fehler = ""
    for py in (DETECT_PYTHON, sys.executable):
        if not os.path.exists(py):
            continue
        r = subprocess.run([py, helfer, APP, other, out], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            z = np.load(out)
            return {k: z[k] for k in z.files}
        fehler = (r.stderr or r.stdout)[-300:]
    raise RuntimeError("Hör-Modell (Instrumenten-Erkennung) nicht verfügbar: " + fehler)


def energie_zeitleiste(pfad):
    """Zeitleiste wie vom Hör-Modell (grob/fein, Werte 0…1), aber aus der Lautstärke einer reinen Orgel-Spur
    (53er-Trennung): 0,2 ≈ 22 dB unter den lautesten Stellen. Kein Modell nötig."""
    y = K.lade_mono(pfad)

    def leiste(fen, hop):
        t, w = [], []
        for a in range(0, max(1, len(y) - fen), hop):
            t.append((a + fen / 2) / SR)
            w.append(10 * np.log10(np.mean(y[a:a + fen] ** 2) + 1e-12))
        return np.array(t), np.array(w)
    gt, gw = leiste(SR * 2, SR)
    ft, fw = leiste(SR, SR // 2)
    ref = float(np.percentile(gw, 95))
    norm = lambda w: np.clip((w - (ref - 26.0)) / 20.0, 0.0, 1.0)  # noqa: E731
    return {"grob_t": gt, "grob": norm(gw), "fein_t": ft, "fein": norm(fw)}


def laeufe(x):
    r, i = [], 0
    while i < len(x):
        j = i
        while j < len(x) and x[j] == x[i]:
            j += 1
        r.append((i, j, bool(x[i])))
        i = j
    return r


def orgel_einsatz(beats, phase, Z):
    """Erster Schlag, ab dem die Orgel spielt (oder None, wenn das Modell nirgends Orgel hört)."""
    from scipy.ndimage import median_filter
    nb = len(beats) - 1
    om = median_filter(Z["grob"], 3)
    tm = Z["grob_t"]
    ba = np.array([om[int(np.argmin(np.abs(tm - 0.5 * (beats[i] + beats[i + 1]))))] >= 0.2 for i in range(nb)])
    takte = [(0, phase)] if phase else []
    takte += [(t0, min(t0 + 4, nb)) for t0 in range(phase, nb, 4)]
    tk = np.array([ba[a:b].mean() >= 0.5 for a, b in takte])
    for i, j, v in laeufe(tk):
        if not v and j - i <= 1 and i > 0 and j < len(tk):
            tk[i:j] = True
    for i, j, v in laeufe(tk):
        if v and j - i < 2:
            tk[i:j] = False
    if not tk.any():
        return None, None
    aktiv = np.zeros(nb, bool)                   # Takte, in denen die Orgel spielt (je Schlag)
    for (a_, b_), v in zip(takte, tk):
        aktiv[a_:b_] = v
    erst = takte[int(np.argmax(tk))][0]
    # fein nachschärfen: erster klarer Orgel-Moment (2 von 3 Fenstern >= 0,2), Fenstermitte + 0,25 s = Einsatz
    tf, of = Z["fein_t"], Z["fein"]
    klar = (of >= 0.2) & (np.convolve(of >= 0.2, np.ones(3), "same") >= 2)
    lo, hi = beats[max(0, erst - 4)], beats[min(nb, erst + 8)]
    k = np.where((tf >= lo) & (tf <= hi) & klar)[0]
    if len(k):
        neu_erst = int(np.clip(np.searchsorted(beats, tf[k[0]] + 0.25, side="right") - 1, 0, nb - 1))
        if neu_erst < erst:
            aktiv[neu_erst:erst] = True
        else:
            aktiv[erst:neu_erst] = False
        erst = neu_erst
    return erst, aktiv


# ------------------------------------------------------------ Bubble planen
def bubble_plan(st, beats, akk, phase, erst, swing=True, aktiv=None):
    lb = K.librosa()
    nb = len(beats) - 1
    org = st.get("_orgel", st["other"])
    D = K.schlagzeug_anschlaege(st["drums"])
    A = beats.copy()
    j = np.clip(np.searchsorted(D, A), 1, len(D) - 1)
    nah = np.where(np.abs(D[j] - A) < np.abs(D[j - 1] - A), D[j], D[j - 1])
    A = np.where(np.abs(nah - A) < 0.03, nah, A)
    idx = np.arange(len(A))
    As = A.copy()
    for q in range(len(A)):
        sl = slice(max(0, q - 4), min(len(A), q + 5))
        As[q] = np.polyval(np.polyfit(idx[sl], A[sl], 1), q)
    # Skank-Stellen wie beim Piano (Gitarre + Piano, auf dem Schlagzeug)
    dsk = {}
    if st.get("guitar") is not None and st["guitar"].any():
        sl_g, z_g, _ = K.skank_stellen(beats, akk, phase, st["guitar"])
        dsk = dict(zip(sl_g, z_g))
    if st.get("piano") is not None and st["piano"].any():
        sl_p, z_p, _ = K.skank_stellen(beats, akk, phase, st["piano"])
        dsk.update(dict(zip(sl_p, z_p)))
    sk_i = sorted(i for i, p in dsk if p == 0.5)
    SK = {}
    if sk_i:
        sk_t, _ = K.am_schlagzeug(np.array([dsk[(i, 0.5)] for i in sk_i]), D)
        SK = dict(zip(sk_i, sk_t))
    dm = float(np.median([SK[i] - 0.5 * (As[i] + As[i + 1]) for i in SK])) if SK else 0.0

    def und(i):
        return SK.get(i, 0.5 * (As[i] + As[i + 1]) + dm)

    def basis(i, nm):
        u = und(i)
        return u if nm == "&" else (0.5 * (As[i] + u) if nm == "e" else 0.5 * (u + As[i + 1]))
    stellen = [(i, nm, basis(i, nm)) for i in range(erst, nb) if akk[i] is not None
               and (aktiv is None or aktiv[i]) for nm in POS]
    gitter = np.array([x[2] for x in stellen])
    on = lb.onset.onset_detect(y=K.hochpass(org, 150), sr=SR, hop_length=128, units="time", backtrack=True, delta=0.05)
    versatz = {"&": 0.0, "e": 0.0, "a": 0.0}
    gemessen = {}
    for nm in ("e", "a"):
        sel = np.array([x[1] == nm for x in stellen])
        if not sel.any() or len(on) < 2:
            continue
        j = np.clip(np.searchsorted(on, gitter[sel]), 1, len(on) - 1)
        nah = np.where(np.abs(on[j] - gitter[sel]) < np.abs(on[j - 1] - gitter[sel]), on[j], on[j - 1])
        dd = (nah - gitter[sel])[np.abs(nah - gitter[sel]) < 0.06]
        gemessen[nm] = float(np.clip(np.median(dd), -0.045, 0.045)) if len(dd) else 0.0
    if swing:
        versatz.update(gemessen)
    T = gitter + np.array([versatz[x[1]] for x in stellen])
    # erster und dritter Schlag folgen der Original-Orgel (so wie dort gespielt), der mittlere bleibt auf dem Skank
    gefolgt = 0
    if len(on) >= 2 and len(T):
        for q, (i, nm, _) in enumerate(stellen):
            if nm == "&":
                continue
            j = int(np.clip(np.searchsorted(on, T[q]), 1, len(on) - 1))
            o = on[j] if abs(on[j] - T[q]) < abs(on[j - 1] - T[q]) else on[j - 1]
            if abs(o - T[q]) <= 0.03:
                T[q] = o
                gefolgt += 1
    info = {"wie_original_gespielt": gefolgt, "skank_treffer": sum(1 for i, nm, _ in stellen if nm == "&" and i in SK),
            "bubble_schlaege": len(stellen) // 3, "swing_ms": {k: round(v * 1000, 1) for k, v in gemessen.items()},
            "swing_an": bool(swing)}
    return stellen, T, D, info


def _schlag_db(x, T, w0=-0.02, w1=0.06):
    """Lautstärke (dB) der Original-Spur kurz um jeden Zeitpunkt (Anschlag, Höhen über 150 Hz)."""
    h = K.hochpass(x, 150)
    out = []
    for t in T:
        a, b = max(0, int((t + w0) * SR)), min(len(h), int((t + w1) * SR))
        out.append(10 * np.log10(np.mean(h[a:b] ** 2) + 1e-12) if b > a else -120.0)
    return np.array(out)


def bubble_dynamik(org, stellen, T):
    """Wie laut spielt das Original den ersten (e), mittleren (&) und dritten (a) Schlag? dB relativ zur Mitte.
    Im Reggae-Bubble ist die Mitte (mit dem Skank) am lautesten, der erste etwas leiser, der dritte am leisesten –
    stimmt die Messung damit nicht zusammen oder ist sie unsicher, gilt diese Regel (−3 / −6 dB)."""
    regel = {"e": -3.0, "&": 0.0, "a": -6.0}
    if not len(T):
        return regel
    L = _schlag_db(org, T)
    pos = np.array([x[1] for x in stellen])
    lauf = L > np.percentile(L, 95) - 30          # nur Stellen, an denen die Orgel wirklich spielt
    m = {nm: float(np.median(L[(pos == nm) & lauf])) for nm in POS if ((pos == nm) & lauf).sum() >= 8}
    if len(m) < 3:
        return regel
    e, a_ = m["e"] - m["&"], m["a"] - m["&"]
    e = float(np.clip(e, -8.0, -1.5))
    a_ = float(np.clip(a_, -12.0, e - 2.0))
    return {"e": round(e, 1), "&": 0.0, "a": round(a_, 1)}


def mit_original_vergleichen(orig, neu, abschnitte, stellen, T):
    """Schläge der neuen Orgel gegen die Original-Orgel: Trefferquote (±35 ms), mittlere Abweichung, Dynamik."""
    lb = K.librosa()
    def ons(x):
        o = lb.onset.onset_detect(y=K.hochpass(np.asarray(x, np.float32), 150), sr=SR, hop_length=128,
                                  units="time", backtrack=True, delta=0.05)
        return np.array([t for t in o if any(a <= t <= b for a, b in abschnitte)])
    oa, on_ = ons(orig), ons(neu)
    ab = []
    for t in oa:
        if len(on_):
            ab.append(float(np.min(np.abs(on_ - t))))
    ab = np.array(ab)
    tr = float((ab <= 0.035).mean() * 100) if len(ab) else 0.0
    dyn_neu = {}
    if len(T):
        Ln = _schlag_db(np.asarray(neu, np.float32), T)
        pos = np.array([x[1] for x in stellen])
        mm = {nm: float(np.median(Ln[pos == nm])) for nm in POS if (pos == nm).any()}
        if "&" in mm:
            dyn_neu = {nm: round(v - mm["&"], 1) for nm, v in mm.items()}
    return {"original_schlaege": int(len(oa)), "neue_schlaege": int(len(on_)), "treffer_prozent": round(tr, 1),
            "abweichung_ms": round(float(np.median(ab[ab <= 0.035]) * 1000) if (ab <= 0.035).any() else 0.0, 1),
            "dynamik_neu_db": dyn_neu}


def griffe(stellen, T, akk, noten):
    non = np.array([x[0] for x in noten]) if noten else np.zeros(0)

    def griff(a, hand):
        pcs = {(a[0] + iv) % 12 for iv in K.QUAL[a[1]][:3]}
        zahl, cnt = {}, 0
        for (i, nm, _), t in zip(stellen, T):
            if akk[i] != a or (nm == "&") != (hand == "R"):
                continue
            sel = {noten[q][2] for q in np.where(np.abs(non - t) < 0.05)[0]} if len(non) else set()
            cnt += bool(sel)
            for p in sel:
                zahl[p] = zahl.get(p, 0) + 1
        lo, hi = (48, 64) if hand == "L" else (58, 76)
        g = sorted(p for p, c in zahl.items() if cnt and c >= 0.3 * cnt and p % 12 in pcs and lo <= p <= hi)
        if len({p % 12 for p in g}) >= 3 and len(g) <= 5:
            return g
        b0 = 52 if hand == "L" else 62
        return sorted(b0 + ((p - b0) % 12) for p in pcs)
    return {(a, h): griff(a, h) for a in {akk[i] for i, _, _ in stellen} for h in "LR"}


# ------------------------------------------------------------ Leslie
def leslie(x, art):
    """Drehlautsprecher: Horn (hoch) und Trommel (tief) drehen unterschiedlich schnell – Lautstärke-
    und Tonhöhen-Schwankung (Doppler), links/rechts versetzt."""
    if art not in ("langsam", "schnell"):
        return x
    from scipy.signal import butter, sosfiltfilt
    mono = x.mean(1)
    tief = sosfiltfilt(butter(4, 800, "low", fs=SR, output="sos"), mono)
    hoch = mono - tief
    f_h, f_t = (0.83, 0.67) if art == "langsam" else (6.7, 5.9)
    t = np.arange(len(mono)) / SR
    out = np.zeros((len(mono), 2))
    for sig, f, tiefe, dop in ((hoch, f_h, 0.35, 0.00030), (tief, f_t, 0.15, 0.00010)):
        for k, ph in enumerate((0.0, np.pi / 2)):
            w = 2 * np.pi * f * t + ph
            pos = np.arange(len(sig)) - (dop * SR) * (1 + np.sin(w))
            v = np.interp(pos, np.arange(len(sig)), sig, left=0, right=0)
            out[:, k] += v * (1 - tiefe * 0.5 * (1 + np.cos(w)))
    return out * (np.sqrt(np.mean(x ** 2)) / max(np.sqrt(np.mean(out ** 2)), 1e-12))


# ------------------------------------------------------------ MIDI
def midi_schreiben(noten, bank, programm, ziel, bpm=120.0):
    ppq = 960
    tempo = float(bpm) if bpm and bpm > 0 else 120.0
    tick = lambda s: int(round(s * tempo / 60.0 * ppq))  # noqa: E731
    ev = []
    for a, b, p, v in noten:
        ev.append((tick(a), 1, 0x90, p, int(np.clip(v, 1, 127))))
        ev.append((max(tick(b), tick(a) + 1), 0, 0x80, p, 0))
    ev.sort(key=lambda e: (e[0], e[1]))

    def vlq(n):
        b = [n & 0x7F]
        n >>= 7
        while n:
            b.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(b)

    def spur(d):
        return b"MTrk" + struct.pack(">I", len(d)) + d
    usek = int(round(60_000_000 / tempo))
    t0 = b"\x00\xff\x51\x03" + usek.to_bytes(3, "big") + b"\x00\xff\x58\x04\x04\x02\x18\x08" + b"\x00\xff\x2f\x00"
    nb = "Orgel repariert".encode("utf-8")
    d = b"\x00\xff\x03" + vlq(len(nb)) + nb
    d += b"\x00" + bytes([0xB0, 0, bank & 0x7F]) + b"\x00" + bytes([0xC0, programm & 0x7F])   # Bank + Klang
    letzt = 0
    for tk_, _, st_, a1, a2 in ev:
        d += vlq(tk_ - letzt) + bytes([st_, a1, a2])
        letzt = tk_
    d += b"\x00\xff\x2f\x00"
    datei = os.path.join(ziel, "orgel_neu.mid")
    with open(datei, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + spur(t0) + spur(d))
    return datei


# ------------------------------------------------------------ Ablauf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe")
    ap.add_argument("ziel")
    ap.add_argument("--klang", default="0:16", help="Bank:Programm aus GeneralUser GS (0:16 Hammond …)")
    ap.add_argument("--ohne-swing", action="store_true")
    ap.add_argument("--leslie", default="aus", choices=("aus", "langsam", "schnell"))
    ap.add_argument("--ohne-zusatztoene", action="store_true")
    ap.add_argument("--midi-bpm", type=float, default=120.0)
    ap.add_argument("--lautstaerke-db", type=float, default=0.0)
    ap.add_argument("--modelle", default=os.path.expanduser("~/HELTECH_Stems/models/"))
    ap.add_argument("--spuren-ordner", default=None)
    ap.add_argument("--noten-json", default=None, help="(Test) schon erkannte Noten der Other-Spur")
    ap.add_argument("--zeitleiste", default=None, help="(Test) schon berechnete Orgel-Zeitleiste (npz)")
    ap.add_argument("--orgel-spur", default=None, help="Orgel-Spur aus der 53er-Trennung (wenn schon vorhanden)")
    a = ap.parse_args()
    os.makedirs(a.ziel, exist_ok=True)
    try:
        lb = K.librosa()
        if a.klang not in KLAENGE:
            a.klang = "0:16"
        bank, prog = (int(x) for x in a.klang.split(":"))
        if a.spuren_ordner:
            spuren = {s: glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))[0]
                      for s in K.STIMMEN if glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))}
            modell = "vorhanden"
        else:
            melde("… trenne den Song in 6 Spuren (dauert einige Minuten)")
            spuren, modell = K.trennen(a.eingabe, a.ziel, a.modelle)
        melde(f"✔ Spuren getrennt ({modell})")
        if "other" not in spuren or "drums" not in spuren or "bass" not in spuren:
            raise RuntimeError("Trennung lieferte nicht alle nötigen Spuren (Other, Schlagzeug, Bass)")
        st = {s: K.lade_mono(p) for s, p in spuren.items() if s != "vocals"}
        n = min(len(v) for v in st.values())
        st = {s: v[:n] for s, v in st.items()}
        melde("… Taktraster und Akkorde aus allen Instrumenten")
        beats = K.taktraster(st["drums"])
        nb = len(beats) - 1
        akk, phase, ton = K.akkord_plan(beats, 3 * st.get("piano", 0) + st["other"] + st.get("guitar", 0), st["bass"])
        melde("… suche, wo die Orgel spielt")
        # Wo liegt die Orgel? 53er-Orgel (wenn vorhanden) > „Other“ > Piano-Spur
        kandidaten = []
        if a.orgel_spur and os.path.isfile(a.orgel_spur):
            kandidaten.append(("orgel53", a.orgel_spur))
        kandidaten += [(s_, spuren[s_]) for s_ in ("other", "piano") if s_ in spuren]
        quelle, Z, erst = None, None, None
        for name, pfad in kandidaten:
            if a.zeitleiste:
                z = np.load(a.zeitleiste)
                Zk = {k: z[k] for k in z.files}
            elif name == "orgel53":
                melde("… höre, wo die Orgel spielt (Lautstärke der 53er-Orgel)")
                Zk = energie_zeitleiste(pfad)
            else:
                Zk = orgel_zeitleiste(pfad, a.ziel)
            ek, ak = orgel_einsatz(beats, phase, Zk)
            if ek is not None:
                quelle, Z, erst, aktiv = name, Zk, ek, ak
                break
            if a.zeitleiste:
                break
        if erst is None:
            raise RuntimeError("In diesem Song ist keine Orgel zu hören (weder in „Other“ noch in der Piano-Spur) – "
                               "nichts zu reparieren")
        org_pfad = dict(kandidaten)[quelle]
        if quelle == "orgel53":
            o = K.lade_mono(org_pfad)
            st["_orgel"] = np.pad(o, (0, max(0, n - len(o))))[:n]
            melde("✔ Orgel aus der 53er-Trennung – wird im Song sauber ersetzt")
        else:
            st["_orgel"] = st[quelle]
            if quelle == "piano":
                melde("✔ Orgel liegt in der Piano-Spur – Piano bleibt im Song, die alte Orgel ist leise darunter "
                      "zu hören (sauberer: Orgel vorher bei den 53 Instrumenten trennen)")
        abschnitte = [(beats[i], beats[min(j, len(beats) - 1)]) for i, j, v in laeufe(aktiv) if v]
        melde(f"✔ Orgel setzt bei {beats[erst]:.1f} s ein, spielt in {len(abschnitte)} Abschnitt(en): "
              + ", ".join(f"{a_:.0f}–{b_:.0f} s" for a_, b_ in abschnitte[:6]))
        melde("… Bubble planen (Mitte genau auf dem Skank, erster und dritter Schlag wie im Original)")
        stellen, T, D, info = bubble_plan(st, beats, akk, phase, erst, swing=not a.ohne_swing, aktiv=aktiv)
        info["abschnitte_s"] = [[round(float(a_), 2), round(float(b_), 2)] for a_, b_ in abschnitte]
        if a.noten_json:
            noten = [tuple(x) for x in json.load(open(a.noten_json))]
        else:
            melde("… erkenne die Töne der Original-Orgel")
            try:
                noten, _, _ = P.noten_basic_pitch(org_pfad, a.ziel)
                noten = [(x[0], x[1], x[2], (x[3] - 30) / 100.0) for x in noten]   # zurück auf 0..1
            except Exception as e:
                melde(f"… Töne nicht erkennbar ({str(e)[:80]}), nehme Standard-Griffe")
                noten = []
        G = griffe(stellen, T, akk, noten)
        P_ = float(np.median(np.diff(beats)))
        dyn = bubble_dynamik(st["_orgel"], stellen, T)
        info["dynamik_original_db"] = dyn
        VEL = {nm: int(round(104 * 10 ** (dyn[nm] / 40.0))) for nm in POS}   # Anschlag ~ 40·log10(v) dB
        melde(f"✔ Dynamik der drei Schläge: erster {dyn['e']:+.1f} dB, Mitte 0 dB, dritter {dyn['a']:+.1f} dB")
        neu = []
        for (i, nm, _), t in zip(stellen, T):
            d, v, g = (0.30 * P_, VEL["&"], G[(akk[i], "R")]) if nm == "&" else (0.17 * P_, VEL[nm], G[(akk[i], "L")])
            for p in g:
                neu.append((float(t), float(t) + d, p, v))
        extra = []
        if not a.ohne_zusatztoene and noten:
            tf, of = Z["fein_t"], Z["fein"]
            skala = {(ton[0] + i) % 12 for i in ((0, 2, 3, 5, 7, 8, 10) if ton[1] == "moll" else (0, 2, 4, 5, 7, 9, 11))}
            in_takt = lambda t: bool(aktiv[min(len(aktiv) - 1, max(0, int(np.searchsorted(beats, t, side="right") - 1)))])  # noqa: E731
            extra = [x for x in noten if x[0] >= beats[erst] and in_takt(x[0]) and len(T) and np.min(np.abs(T - x[0])) > 0.07 and 55 <= x[2] <= 84
                     and x[2] % 12 in skala and x[1] - x[0] >= 0.1 and of[int(np.argmin(np.abs(tf - x[0])))] >= 0.3]
            if extra:
                ze, _ = K.am_schlagzeug(np.array([x[0] for x in extra]), D)
                extra = [(float(z), float(z + (b_ - a_)), p, int(np.clip(30 + 100 * v, 70, 100)))
                         for z, (a_, b_, p, v) in zip(ze, extra)]
        info["zusaetzliche_toene"] = len(extra)
        melde(f"✔ {info['bubble_schlaege']} Bubble-Schläge, {info['skank_treffer']} genau auf dem Skank, "
              f"{len(extra)} zusätzliche Orgel-Töne")
        klang = KLAENGE[a.klang]
        melde(f"… spiele neu: {klang}" + (f", Leslie {a.leslie}" if a.leslie != "aus" else ""))
        instr = sp.SoundFont(P.soundfont_sichern()).instrument(bank, prog)
        y = sp.spielen(instr, neu, n / SR, nachklang_max=0.08)
        if extra:
            y2 = sp.spielen(instr, extra, n / SR, nachklang_max=0.15)
            L = min(len(y), len(y2))
            y = y[:L] + y2[:L]
        y = leslie(y, a.leslie)
        # gleichmäßige Lautstärke wie die Original-Orgel (nur über die Orgel-Takte)
        kk = int(2.8 * SR)
        fen = [int(x * SR) for a_, b_ in abschnitte for x in np.arange(a_, b_ - 2.8, 2.8)] \
            or [int(x * SR) for x in np.arange(beats[erst], beats[-1] - 2.8, 2.8)]   # nur in den Orgel-Abschnitten
        tr = lambda x: np.array([np.sqrt(np.mean(x[q:q + kk] ** 2)) for q in fen])  # noqa: E731
        ist = np.median(tr(y.mean(1)))
        y *= np.percentile(tr(st["_orgel"]), 75) / max(ist, 1e-12) * 10 ** (a.lautstaerke_db / 20)
        if np.abs(y).max() > 0.98:
            y *= 0.98 / np.abs(y).max()
        out = os.path.join(a.ziel, "orgel_neu.wav")
        sf.write(out, y, SR, subtype="PCM_24")
        # Song: die alte Orgel nur ab dem Orgel-Einsatz ersetzen (davor bleibt alles, wie es war)
        from scipy.ndimage import uniform_filter1d

        def ab_einsatz(x):
            maske = np.zeros(len(x))
            maske[int(beats[erst] * SR):] = 1.0
            return x * uniform_filter1d(maske, int(0.08 * SR))[:, None]

        def stereo(p_):
            x, _ = lb.load(p_, sr=SR, mono=False)
            x = np.atleast_2d(x).T
            return np.repeat(x, 2, 1) if x.shape[1] == 1 else x
        rest = None
        for s_, p_ in spuren.items():
            x = stereo(p_)
            if s_ == "other" and quelle == "other":
                x = x - ab_einsatz(x)                    # „Other“ ab dem Einsatz raus (wie bisher)
            rest = x if rest is None else rest[:min(len(rest), len(x))] + x[:min(len(rest), len(x))]
        if quelle == "orgel53":                          # genau die alte Orgel abziehen, alles andere bleibt
            o = stereo(org_pfad)
            L0 = min(len(rest), len(o))
            rest = rest[:L0] - ab_einsatz(o[:L0])
        L = min(len(rest), len(y))
        mix = rest[:L] + y[:L]
        mix *= min(1.0, 0.98 / max(np.abs(mix).max(), 1e-12))
        sf.write(os.path.join(a.ziel, "song_mit_neuer_orgel.wav"), mix, SR, subtype="PCM_24")
        midi = midi_schreiben(sorted(neu + extra), bank, prog, a.ziel, a.midi_bpm)
        bpm_song = 60.0 / P_
        plan = ["Orgel (Bubble) – Akkord-Plan, erkannt aus allen Instrumenten",
                f"Klang: {klang} (GeneralUser GS, Bank {bank}, GM-Programm {prog + 1})"
                + (f", Leslie {a.leslie} (nur im Audio)" if a.leslie != "aus" else ""),
                f"Tonart: {K.NAMEN[ton[0]]}-{ton[1]}, Song-Tempo ca. {bpm_song:.1f} BPM (A# = B♭, D# = E♭)",
                f"Orgel-Einsatz: {int(beats[erst] // 60)}:{beats[erst] % 60:05.2f}; spielt in: "
                + ", ".join(f"{int(a_ // 60)}:{a_ % 60:04.1f}–{int(b_ // 60)}:{b_ % 60:04.1f}" for a_, b_ in abschnitte),
                f"Bubble: rechte Hand auf dem „&“ (genau mit Gitarre/Piano-Skank), linke Hand auf „e“ und „a“"
                + (f", Swing wie im Original ({info['swing_ms'].get('e', 0):.0f}/{info['swing_ms'].get('a', 0):.0f} ms)"
                   if not a.ohne_swing else ", gerade"),
                f"MIDI-Datei mit {a.midi_bpm:g} BPM geschrieben: Logic-Projekt auf {a.midi_bpm:g} BPM stellen.",
                "Im MIDI: Noten mit Anschlagstärke, Bank- und Programmwechsel (Klang), Tempo, 4/4-Takt.",
                "Griffe: " + ", ".join(f"{K.akkord_name(k[0])} {'links' if k[1] == 'L' else 'rechts'} = "
                                      + " ".join(K.NAMEN[p % 12] + str(p // 12 - 1) for p in g)
                                      for k, g in sorted(G.items(), key=str)), ""]
        for k, t0 in enumerate(range(phase, len(akk), 4)):
            s0 = beats[t0]
            plan.append(f"{int(s0 // 60)}:{s0 % 60:05.2f}  Takt {k + 1:3d}  "
                        + " | ".join(f"{K.akkord_name(x):6s}" for x in akk[t0:t0 + 4]))
        with open(os.path.join(a.ziel, "orgel_neu_Akkord_Plan.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(plan) + "\n")
        vergleich = mit_original_vergleichen(st["_orgel"], y.mean(1), abschnitte, stellen, T)
        melde(f"✔ Vergleich mit dem Original: {vergleich['treffer_prozent']:.0f} % der Original-Schläge getroffen, "
              f"mittlere Abweichung {vergleich['abweichung_ms']:.0f} ms")
        info["vergleich_mit_original"] = vergleich
        bericht = {"trennung": modell, "orgel_quelle": quelle, "alte_orgel_bleibt": quelle == "piano", "klang": klang, "bank": bank, "programm": prog, "leslie": a.leslie,
                   "tonart": f"{K.NAMEN[ton[0]]}-{ton[1]}", "tempo_bpm": round(bpm_song, 2),
                   "orgel_einsatz_s": round(float(beats[erst]), 2), **info, "midi": midi}
        json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
        import shutil
        shutil.rmtree(os.path.join(a.ziel, "spuren_roh"), ignore_errors=True)
        for f in glob.glob(os.path.join(a.ziel, "_*")):
            os.remove(f)
        print(json.dumps({"ok": True, "datei": out, "ordner": a.ziel, "klang": klang,
                          "einsatz_s": round(float(beats[erst]), 2)}, ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "fehler": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
