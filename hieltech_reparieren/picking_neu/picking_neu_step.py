#!/usr/bin/env python3
"""
HELTECH Stems – Reggae-Begleit-Gitarre („Picking“) – mit Erkennung/Reparatur einer vorhandenen Picking-Gitarre (Modus auto): eine gedämpfte Gitarre spielt die Basslinie eine Oktave
höher, genau mit jedem Bass-Anschlag. Stand 6. Nov 2026, freigegeben mit „Rise Up Jah Children“ (Picking v3,
Version B „genau mit dem Bass“). Das Ergebnis klingt fertig, MIDI zusätzlich.

Ablauf:
  1. Ganzer Song -> Trennung in 6 Spuren (BS-Roformer-SW, Rückfall htdemucs_6s)  (oder fertige Spuren)
  2. Basslinie aus der Bass-Spur (gleiche Notenerkennung wie die Bass-Reparatur)
     - Mini-Töne < 80 ms (Nachklang/Rutscher) raus
     - Einsatz auf den echten Bass-Anschlag (Anschlag-Spitze -50…+40 ms)
     - unsichere Tonhöhe außerhalb der Tonart -> nächster Ton der Tonart (klare Durchgangstöne bleiben)
  3. Lage: Basston + 1/2 Oktaven zwischen C3 und E4, Lage wechselt nur wenn nötig
  4. Timing: „mit dem Bass“ (Standard, jeder Ton exakt mit dem Bass) oder „auf dem Schlagzeug“
     (Schlag bzw. Sechzehntel-Raster)
  5. Klang: eigene gedämpfte Töne (samples/) oder Gitarren aus GeneralUser GS; Klangfarbe/Effekt darauf
  6. Lautstärke ca. 8 dB unter der Skank-Gitarre; Ausgabe picking_neu.wav, song_mit_picking.wav, MIDI, Notenliste

Aufruf:
  python picking_neu_step.py <song.wav> <ziel> [--klang eigene|0:28|8:28|…] [--effekt original|warm|fett|…]
        [--timing bass|schlagzeug] [--midi-bpm 120] [--lautstaerke-db 0] [--modelle ORDNER] [--spuren-ordner ORDNER]
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np
import soundfile as sf

HIER = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HIER)
sys.path.insert(0, HIER)
sys.path.insert(0, os.path.join(APP, "piano_neu"))          # Skank-Kern, SF2-Abspieler, SoundFont
sys.path.insert(0, os.path.join(APP, "bass_neu"))           # Bass-Notenerkennung
import skank_kern as K                                       # noqa: E402

SR = 44100
SAMPLES = os.path.join(HIER, "samples")
KLAENGE = {
    "eigene": "Eigene gedämpfte Gitarre",
    "0:28": "Gedämpfte E-Gitarre", "8:28": "Funk-Gitarre", "0:26": "Jazz-Gitarre", "0:27": "Clean-Gitarre",
    "8:27": "Clean-Gitarre mit Chorus", "12:27": "Clean-Gitarre 2", "0:24": "Nylon-Gitarre", "0:25": "Steel-Gitarre",
    "8:26": "Hawaii-Gitarre", "8:25": "12-saitige Gitarre", "0:29": "Overdrive-Gitarre", "11:29": "Wah-Gitarre",
    "8:24": "Ukulele", "16:25": "Mandoline"}
EFFEKTE = {"original": "Original", "warm": "Warm (Vintage)", "fett": "Fett", "hell": "Hell (Twang)",
           "chorus": "Chorus", "dub_echo": "Dub-Echo", "federhall": "Federhall", "phaser": "Phaser",
           "tremolo": "Tremolo", "autowah": "Auto-Wah"}


def melde(t):
    print(t, flush=True)


# ------------------------------------------------------------ Basslinie
def basslinie(bass, beats, D, ton):
    import bass_neu_step as B
    lb = K.librosa()
    from scipy.signal import butter, sosfiltfilt
    kurve, _ = B.tonkurve(bass)
    N = [n for n in B.noten_finden(bass, kurve) if (n["ende"] - n["einsatz"]) / SR >= 0.08]
    hp = sosfiltfilt(butter(4, [150, 3000], "bandpass", fs=SR, output="sos"), bass)
    o = lb.onset.onset_strength(y=hp, sr=SR, hop_length=64)
    to = np.arange(len(o)) * 64 / SR
    verf = 0
    for n in N:
        t = n["einsatz"] / SR
        s_ = (to >= t - 0.05) & (to <= t + 0.04)
        um = (to >= t - 0.5) & (to <= t + 0.5)
        if s_.any() and o[s_].max() >= 0.3 * o[um].max():
            pk = float(to[s_][np.argmax(o[s_])]) - 0.003
            if abs(pk - t) > 0.004:
                n["einsatz"] = int(pk * SR)
                verf += 1
    skala = {(ton[0] + i) % 12 for i in ((0, 2, 3, 5, 7, 8, 10) if ton[1] == "moll" else (0, 2, 4, 5, 7, 9, 11))}
    korr = 0
    for n in N:
        r = int(round(n["ton"]))
        if r % 12 not in skala and (abs(n["ton"] - r) > 0.3 or (n["ende"] - n["einsatz"]) / SR < 0.15):
            n["ton"] = float(min((r - 1, r + 1), key=lambda q: (q % 12 not in skala, abs(q - n["ton"]))))
            korr += 1
    return N, {"bass_noten": len(N), "einsaetze_verfeinert": verf, "tonhoehe_korrigiert": korr}


def raster_zeiten(t_roh, beats, D):
    A = beats.copy()
    j = np.clip(np.searchsorted(D, A), 1, len(D) - 1)
    nah = np.where(np.abs(D[j] - A) < np.abs(D[j - 1] - A), D[j], D[j - 1])
    A = np.where(np.abs(nah - A) < 0.03, nah, A)
    idx = np.arange(len(A))
    As = A.copy()
    for q in range(len(A)):
        sl = slice(max(0, q - 4), min(len(A), q + 5))
        As[q] = np.polyval(np.polyfit(idx[sl], A[sl], 1), q)
    G16 = np.concatenate([As[i] + np.arange(4) / 4 * (As[i + 1] - As[i]) for i in range(len(As) - 1)])
    out = t_roh.copy()
    for k, t in enumerate(t_roh):
        d = D[np.argmin(np.abs(D - t))]
        g = G16[np.argmin(np.abs(G16 - t))]
        if abs(d - t) <= 0.035 and abs(d - g) <= 0.025:
            out[k] = d
        elif abs(g - t) <= 0.07:
            out[k] = g
    return out


# ------------------------------------------------------------ vorhandene Picking-Gitarre erkennen / reparieren
BP_HELFER = r'''
import json, os, sys
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH as M
arg = dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=40)
try:
    _, _, ev = predict(sys.argv[1], M, **arg)
except Exception:
    from basic_pitch.inference import Model
    _, _, ev = predict(sys.argv[1], Model(os.path.join(os.path.dirname(M), "nmp.onnx")), **arg)
json.dump([[float(a), float(b), int(p), float(v)] for a, b, p, v, *_ in ev], open(sys.argv[2], "w"))
'''
# Python mit Basic Pitch: auf dem Mac die eigene Umgebung, auf dem Server (RunPod) per HIELTECH_MIDI_PYTHON
MIDI_ENV_PYTHON = os.environ.get("HIELTECH_MIDI_PYTHON") or os.path.expanduser("~/miniforge3/envs/heltech_stems_midi/bin/python")


try:
    sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
    from merker import merke
except Exception:  # noqa: BLE001 – ohne Merker läuft alles wie bisher
    def merke(*_a, **_k):
        return lambda fn: fn


@merke("picking_gitarren_noten", nur=(0,))
def gitarren_noten(pfad, ziel):
    import subprocess
    out = os.path.join(ziel, "_git_noten.json")
    helfer = os.path.join(ziel, "_git_helfer.py")
    with open(helfer, "w") as f:
        f.write(BP_HELFER)
    for py in (sys.executable, MIDI_ENV_PYTHON):
        if not os.path.exists(py):
            continue
        r = subprocess.run([py, helfer, pfad, out], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            return [tuple(x) for x in json.load(open(out))]
    raise RuntimeError("Notenerkennung (Basic Pitch) nicht verfügbar")



def gruppen(ev, fenster=0.04):
    gr, cur = [], []
    for e in sorted(ev):
        if cur and e[0] - cur[0][0] > fenster:
            gr.append(cur)
            cur = []
        cur.append(e)
    if cur:
        gr.append(cur)
    return gr


def erkennen(git_noten, bass_t, bass_ton, skank_t, beats):
    """git_noten: (start, ende, ton, staerke) der Gitarren-Spur; bass_t/bass_ton: Bass-Einsätze und -Töne;
    skank_t: Skank-Zeiten. Rückgabe: (erkannt, kandidaten, bericht)."""
    bass_t = np.asarray(bass_t)
    skank_t = np.asarray(skank_t) if len(skank_t) else np.array([-9.0])
    kand = []
    for g in gruppen(git_noten):
        t = g[0][0]
        if np.min(np.abs(skank_t - t)) < 0.06:                  # Skank (Akkord oder Rest davon) -> nicht Picking
            continue
        tief = [e for e in g if 45 <= e[2] <= 70]
        if not tief or len(g) >= 4:                            # voller Akkord -> nicht Picking
            continue
        e = max(tief, key=lambda x: x[3])
        kand.append((float(t), int(e[2]), float(e[1] - e[0]), float(e[3])))
    treffer, passend = [], 0
    for k in kand:
        j = int(np.argmin(np.abs(bass_t - k[0])))
        if abs(bass_t[j] - k[0]) < 0.05:
            iv = (k[1] - int(round(bass_ton[j]))) % 12
            treffer.append((j, k, iv))
            passend += iv in (0, 7)
    abdeckung = len({j for j, _, _ in treffer}) / max(1, len(bass_t))
    genau = passend / max(1, len(kand))
    takte = max(1.0, (beats[-1] - beats[0]) / (4 * np.median(np.diff(beats))))
    dichte = len(kand) / takte
    erkannt = bool(abdeckung >= 0.3 and genau >= 0.5 and dichte >= 1.0)
    return erkannt, kand, treffer, {"kandidaten": len(kand), "bass_abdeckung": round(abdeckung, 2),
                                    "passend_zum_bass": round(genau, 2), "einzeltoene_je_takt": round(dichte, 1)}


def reparatur_plan(kand, treffer, bass_t, bass_ton, beats):
    """Töne der vorhandenen Picking-Gitarre übernehmen:
    - Bass-Anschlag mit Gitarrenton -> Gitarrenton (Tonhöhe der Gitarre), Zeit = Bass-Anschlag
    - Bass-Anschlag ohne Gitarrenton -> Pause, solange die Gitarre im Takt spielt (bewusste Pause);
      ist der ganze Takt stumm (von der Trennung verloren), wird er aus dem Bass aufgefüllt
    - Gitarrenton ohne Bass-Anschlag (eigener Zusatzton): bleibt, wenn er zur Basslinie passt
    Rückgabe: Liste (zeit, ton, quelle, bass_index | None)"""
    bass_t = np.asarray(bass_t)
    an = {j: k for j, k, iv in treffer}
    P = float(np.median(np.diff(beats)))
    takt = lambda t: int((t - beats[0]) // (4 * P))           # noqa: E731
    gt = {}
    for k in kand:
        gt.setdefault(takt(k[0]), []).append(k)
    plan, gefuellt = [], 0
    for j, t in enumerate(bass_t):
        if j in an:
            plan.append((float(t), an[j][1], "gitarre", j))
        elif not gt.get(takt(t)) and (gt.get(takt(t) - 1) or gt.get(takt(t) + 1)):
            plan.append((float(t), None, "aufgefuellt", j))      # Ton wird später aus dem Bass gesetzt
            gefuellt += 1
    benutzt = {id(k) for _, k, _ in treffer}
    extra = 0
    for k in kand:
        if id(k) in benutzt:
            continue
        j = int(np.argmin(np.abs(bass_t - k[0])))
        # Zusatzton: kräftig, Grundton/Quinte des gerade klingenden Basstons
        if k[3] >= 0.35 and (k[1] - int(round(bass_ton[j]))) % 12 in (0, 7) and k[0] > bass_t[j]:
            plan.append((k[0], k[1], "zusatz", None))
            extra += 1
    plan.sort()
    return plan, {"von_gitarre": sum(1 for p in plan if p[2] == "gitarre"), "aufgefuellt": gefuellt, "zusatztoene": extra}


def nachpruefen(plan, git, bass_t, bass_ton, skank_t, treffer, lage):
    """Bass-Anschläge ohne erkannten Gitarrenton im Klang der Gitarren-Spur prüfen: setzt dort ein Ton auf
    Grundton oder Quinte (in Gitarrenlage) neu ein? Schwelle aus den sicher erkannten Tönen gelernt."""
    import librosa
    hop = 256
    C = np.abs(librosa.cqt(git, sr=SR, hop_length=hop, fmin=librosa.midi_to_hz(40), n_bins=48, bins_per_octave=12))
    C = np.log1p(C / (np.percentile(C, 99) + 1e-9) * 100)
    def gewinn(t, p):
        b = p - 40
        if not 0 <= b < 48:
            return -9.0
        f = int(t * SR / hop)
        nach = C[max(0, b - 0):b + 1, f + 2:f + 14].mean() if f + 14 < C.shape[1] else 0
        vor = C[b:b + 1, max(0, f - 12):max(1, f - 2)].mean()
        # Obertöne mitzählen (Oktave darüber)
        if b + 12 < 48 and f + 14 < C.shape[1]:
            nach += 0.5 * C[b + 12, f + 2:f + 14].mean(); vor += 0.5 * C[b + 12, max(0, f - 12):max(1, f - 2)].mean()
        return float(nach - vor)
    def kandidaten(j):
        r = int(round(bass_ton[j]))
        return [q for q in range(45, 71) if (q - r) % 12 in (0, 7)]
    # Schwelle: Gewinne bei sicher erkannten Tönen vs. bei Skank-Zeiten ohne Bass-Anschlag
    sicher = [gewinn(bass_t[j], k[1]) for j, k, iv in treffer]
    skank_t = np.asarray(skank_t)
    leer = []
    for t in skank_t:
        j = int(np.argmin(np.abs(np.asarray(bass_t) - t)))
        if abs(bass_t[j] - t) > 0.08:
            leer.append(max(gewinn(t, q) for q in kandidaten(j)))
    s_med = float(np.median(sicher)) if sicher else 1.0
    l_90 = float(np.percentile(leer, 90)) if leer else 0.0
    schwelle = max(l_90, 0.35 * s_med)
    im_plan = {p[3] for p in plan if p[3] is not None}
    neu, n_neu = list(plan), 0
    for j, t in enumerate(bass_t):
        if j in im_plan:
            continue
        g = [(gewinn(t, q), q) for q in kandidaten(j)]
        bg, bq = max(g)
        if bg > schwelle:
            neu.append((float(t), bq, "nachgeprueft", j)); n_neu += 1
    neu.sort()
    return neu, {"nachgeprueft_gefunden": n_neu, "schwelle": round(schwelle, 3), "sicher_median": round(s_med, 3),
                 "skank_leer_90": round(l_90, 3)}


# ------------------------------------------------------------ Klang
def eigene_proben():
    lb = K.librosa()
    proben = []
    for f in sorted(glob.glob(os.path.join(SAMPLES, "*.flac")) + glob.glob(os.path.join(SAMPLES, "*.wav"))):
        x, sr = sf.read(f)
        x = x.mean(1) if x.ndim > 1 else x
        if sr != SR:
            x = lb.resample(x, orig_sr=sr, target_sr=SR)
        a = int(np.argmax(np.abs(x) > 0.2 * np.abs(x).max()))
        x = x[max(0, a - 32):]
        f0, vf, _ = lb.pyin(x[:int(0.25 * SR)], fmin=80, fmax=600, sr=SR, frame_length=2048, hop_length=256)
        if not vf.any():
            continue
        ton = float(np.nanmedian(lb.hz_to_midi(f0[vf])))
        n = min(len(x), int(0.45 * SR))
        x = x[:n] * np.r_[np.ones(n - int(0.08 * SR)), np.linspace(1, 0, int(0.08 * SR))][:n]
        proben.append((ton, x / max(np.abs(x).max(), 1e-9)))
    if not proben:
        raise RuntimeError(f"keine eigenen Gitarren-Töne in {SAMPLES}")
    return sorted(proben, key=lambda p: p[0])


def spielen_eigene(noten, L):
    proben = eigene_proben()
    pegel = np.array([x[3] for x in noten])
    pz = (pegel - np.median(pegel)) / (np.std(pegel) + 1e-9)
    rng = np.random.default_rng(5)
    y = np.zeros(L + SR)
    cache = {}
    for (t, p, dauer, _), z in zip(noten, pz):
        if p not in cache:
            i = int(np.argmin([abs(q[0] - p) for q in proben]))
            ton, x = proben[i]
            st = p - ton
            if abs(st) > 0.01:                        # Umsampeln: Anschlag bleibt scharf
                x = np.interp(np.arange(0, len(x) - 1, 2 ** (st / 12)), np.arange(len(x)), x)
            cache[p] = x
        x = cache[p]
        n_ = min(len(x), int(min(max(dauer, 0.12), 0.4) * SR))
        seg = x[:n_] * np.r_[np.ones(max(0, n_ - 400)), np.linspace(1, 0, min(400, n_))]
        a = int(t * SR)
        y[a:a + n_] += seg * 10 ** ((np.clip(z, -1, 1) * 3 + rng.normal(0, 0.8)) / 20)
    return y[:L]


def spielen_sf(noten, L, bank, prog):
    import sf2_spieler as sp
    import piano_neu_step as P
    pegel = np.array([x[3] for x in noten])
    pz = (pegel - np.median(pegel)) / (np.std(pegel) + 1e-9)
    nn = [(t, t + min(max(d, 0.12), 0.3), p, int(np.clip(96 + 10 * z, 70, 120))) for (t, p, d, _), z in zip(noten, pz)]
    ins = sp.SoundFont(P.soundfont_sichern()).instrument(bank, prog)
    return sp.spielen(ins, nn, L / SR, nachklang_max=0.06).mean(1)[:L]


def bp(x, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(2, [lo, hi], "bandpass", fs=SR, output="sos"), x)


def effekt(y, art, P):
    """Klangfarbe/Effekt -> Stereo."""
    from scipy.signal import butter, sosfiltfilt, fftconvolve, lfilter
    t = np.arange(len(y)) / SR

    def verz(x, ms, tiefe, f, ph):
        d = (ms + tiefe * np.sin(2 * np.pi * f * t + ph)) / 1000 * SR
        return np.interp(np.arange(len(x)) - d, np.arange(len(x)), x, left=0)

    def raum(x, anteil=0.12):
        kk = int(0.3 * SR)
        tt = np.arange(kk) / SR
        ir = np.random.default_rng(1).standard_normal(kk) * np.exp(-tt / 0.07)
        ir /= np.sqrt(np.sum(ir ** 2))
        return x + anteil * fftconvolve(x, ir)[:len(x)]
    if art == "warm":
        w = sosfiltfilt(butter(2, 2800, "low", fs=SR, output="sos"), y)
        y = np.tanh(2.2 * w / max(np.abs(w).max(), 1e-9))
    elif art == "fett":
        f = K.fett(y)
        return f if f.ndim == 2 else np.stack([f, f], 1)
    elif art == "hell":
        y = bp(y, 400, 9000) + 0.6 * bp(y, 2500, 6000)
    elif art == "chorus":
        return np.stack([y + 0.7 * verz(y, 9, 2.5, 0.8, 0), y + 0.7 * verz(y, 9, 2.5, 0.8, np.pi / 2)], 1)
    elif art == "dub_echo":
        dl = int(0.75 * P * SR)
        L_, R_, e = np.zeros(len(y)), np.zeros(len(y)), y.copy()
        for k in range(1, 6):
            e = sosfiltfilt(butter(1, 2500, "low", fs=SR, output="sos"), np.r_[np.zeros(dl), e[:-dl]]) * 0.5
            (L_ if k % 2 else R_)[:] += e
        return np.stack([y + L_, y + R_], 1)
    elif art == "federhall":
        kk = int(1.6 * SR)
        tt = np.arange(kk) / SR
        ir = np.random.default_rng(4).standard_normal(kk) * np.exp(-tt / 0.45)
        ir = bp(lfilter([1], [1, *([0] * 220), -0.55], ir), 300, 4500)
        ir /= np.sqrt(np.sum(ir ** 2))
        y = y + 0.35 * fftconvolve(y, ir)[:len(y)]
    elif art == "phaser":
        # 4 Allpässe mit wanderndem Knick (0,4 Hz), blockweise (schnell), gemischt mit dem Original
        B_ = 256
        x = y.copy()
        for _ in range(4):
            out, zi = np.zeros_like(x), np.zeros(1)
            for i in range(0, len(x), B_):
                fc = 300 * 8 ** (0.5 * (1 + np.sin(2 * np.pi * 0.4 * i / SR)))
                c = (np.tan(np.pi * fc / SR) - 1) / (np.tan(np.pi * fc / SR) + 1)
                out[i:i + B_], zi = lfilter([c, 1], [1, c], x[i:i + B_], zi=zi)
            x = out
        y = 0.6 * y + 0.6 * x
    elif art == "tremolo":
        y = y * (1 - 0.45 * 0.5 * (1 + np.sin(2 * np.pi * t * (2 / P))))   # Achtel-Tremolo im Song-Tempo
    elif art == "autowah":
        from scipy.ndimage import uniform_filter1d
        env = np.sqrt(uniform_filter1d(y ** 2, int(0.01 * SR)))
        env = env / max(env.max(), 1e-9)
        B_ = 128
        out, zi = np.zeros_like(y), np.zeros(2)
        for i in range(0, len(y), B_):                   # Bandpass, Mitte folgt dem Anschlag (400…2600 Hz)
            fc = 400 + 2200 * min(1.0, 2 * float(env[i:i + B_].mean()))
            w0 = 2 * np.pi * fc / SR
            al = np.sin(w0) / (2 * 2.5)
            bb = np.array([al, 0, -al]) / (1 + al)
            aa = np.array([1, -2 * np.cos(w0) / (1 + al), (1 - al) / (1 + al)])
            out[i:i + B_], zi = lfilter(bb, aa, y[i:i + B_], zi=zi)
        y = 0.4 * y + 1.6 * out
    y = raum(y)
    return np.stack([y * 0.8, y], 1)


# ------------------------------------------------------------ MIDI
def midi_schreiben(noten, bank, prog, ziel, bpm=120.0):
    ppq = 960
    tempo = float(bpm) if bpm and bpm > 0 else 120.0
    tick = lambda s: int(round(s * tempo / 60.0 * ppq))  # noqa: E731
    pegel = np.array([x[3] for x in noten])
    pz = (pegel - np.median(pegel)) / (np.std(pegel) + 1e-9)
    ev = []
    for (t, p, d, _), z in zip(noten, pz):
        v = int(np.clip(96 + 12 * z, 60, 120))
        ev.append((tick(t), 1, 0x90, p, v))
        ev.append((max(tick(t + min(max(d, 0.12), 0.3)), tick(t) + 1), 0, 0x80, p, 0))
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
    nb = "Picking-Gitarre".encode("utf-8")
    d = b"\x00\xff\x03" + vlq(len(nb)) + nb + b"\x00" + bytes([0xB0, 0, bank & 0x7F]) + b"\x00" + bytes([0xC0, prog & 0x7F])
    letzt = 0
    for tk_, _, st_, a1, a2 in ev:
        d += vlq(tk_ - letzt) + bytes([st_, a1, a2])
        letzt = tk_
    d += b"\x00\xff\x2f\x00"
    datei = os.path.join(ziel, "picking_neu.mid")
    with open(datei, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + spur(t0) + spur(d))
    return datei


# ------------------------------------------------------------ Ablauf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe")
    ap.add_argument("ziel")
    ap.add_argument("--klang", default="eigene")
    ap.add_argument("--effekt", default="original")
    ap.add_argument("--timing", default="bass", choices=("bass", "schlagzeug"))
    ap.add_argument("--midi-bpm", type=float, default=120.0)
    ap.add_argument("--lautstaerke-db", type=float, default=0.0)
    ap.add_argument("--modelle", default=os.path.expanduser("~/HELTECH_Stems/models/"))
    ap.add_argument("--spuren-ordner", default=None)
    ap.add_argument("--modus", default="auto", choices=("auto", "neu"),
                    help="auto: vorhandene Picking-Gitarre erkennen und reparieren, sonst neu; neu: immer aus dem Bass")
    ap.add_argument("--gitarren-noten", default=None, help="(Test) schon erkannte Noten der Gitarren-Spur (json)")
    a = ap.parse_args()
    os.makedirs(a.ziel, exist_ok=True)
    try:
        lb = K.librosa()
        klang = a.klang if a.klang in KLAENGE else "eigene"
        eff = a.effekt if a.effekt in EFFEKTE else "original"
        if a.spuren_ordner:
            spuren = {s: glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))[0]
                      for s in K.STIMMEN if glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))}
            modell = "vorhanden"
        else:
            melde("… trenne den Song in 6 Spuren (dauert einige Minuten)")
            spuren, modell = K.trennen(a.eingabe, a.ziel, a.modelle)
        melde(f"✔ Spuren getrennt ({modell})")
        if "bass" not in spuren or "drums" not in spuren:
            raise RuntimeError("Trennung lieferte keine Bass- oder Schlagzeug-Spur")
        st = {s: K.lade_mono(p) for s, p in spuren.items() if s != "vocals"}
        n = min(len(v) for v in st.values())
        st = {s: v[:n] for s, v in st.items()}
        melde("… Takt, Tonart und Basslinie")
        beats = K.taktraster(st["drums"])
        _, _, ton = K.akkord_plan(beats, 3 * st.get("piano", 0) + st.get("other", 0) + st.get("guitar", 0), st["bass"])
        D = K.schlagzeug_anschlaege(st["drums"])
        N, info = basslinie(st["bass"], beats, D, ton)
        melde(f"✔ {len(N)} Töne der Basslinie ({info['einsaetze_verfeinert']} Einsätze verfeinert, "
              f"{info['tonhoehe_korrigiert']} Tonhöhen korrigiert)")
        bass_t = [x["einsatz"] / SR for x in N]
        bass_ton = [x["ton"] for x in N]
        art, erk, plan = "neu", None, None
        git = st.get("guitar")
        if a.modus == "auto" and git is not None and git.any():
            melde("… prüfe, ob schon eine Picking-Gitarre im Song spielt")
            akk_g, phase_g, _ = K.akkord_plan(beats, st.get("piano", 0) + st.get("other", 0) + 3 * git, st["bass"])
            sl_g, zt_g, _ = K.skank_stellen(beats, akk_g, phase_g, git)
            gn = [tuple(x) for x in json.load(open(a.gitarren_noten))] if a.gitarren_noten \
                else gitarren_noten(spuren["guitar"], a.ziel)
            erkannt, kand, treffer, erk = erkennen(gn, bass_t, bass_ton, zt_g, beats)
            if erkannt:
                plan, pb = reparatur_plan(kand, treffer, bass_t, bass_ton, beats)
                plan, pn = nachpruefen(plan, git, bass_t, bass_ton, zt_g, treffer, None)
                erk.update(pb)
                erk.update(pn)
                art = "repariert"
                melde(f"✔ Vorhandene Picking-Gitarre erkannt – ihre Töne werden übernommen und repariert "
                      f"({pb['von_gitarre'] + pn['nachgeprueft_gefunden']} Töne von der Gitarre, "
                      f"{pb['aufgefuellt']} Lücken aus dem Bass, {pb['zusatztoene']} Zusatztöne)")
            else:
                melde("✔ Keine Picking-Gitarre im Song gefunden – sie wird neu hinzugefügt")

        def lage(b, letzte):                              # b = Basston: Gitarre 1–3 Oktaven höher, C3…E4
            kand = [k for k in (b + 12, b + 24, b + 36) if 47 <= k <= 64] or \
                   [min((b + 12, b + 24, b + 36), key=lambda k: abs(k - 55))]
            return min(kand, key=lambda k: abs(k - (letzte if letzte else 55)))
        noten, letzte = [], None
        if plan is None:                                  # neu aus dem Bass
            for x in N:
                p = lage(int(round(x["ton"])), letzte)
                letzte = p
                noten.append([x["einsatz"] / SR, p, (x["ende"] - x["einsatz"]) / SR, x["pegel"]])
        else:                                             # Töne der vorhandenen Gitarre
            pmed = float(np.median([x["pegel"] for x in N]))
            for t0, p, quelle, j in plan:
                if p is None:
                    p = int(round(N[j]["ton"]))
                else:                                     # Gitarrenton: gleiche Lage-Regel wie neu
                    p = int(p) - 36
                p = lage(int(p), letzte)
                letzte = p
                d = (N[j]["ende"] - N[j]["einsatz"]) / SR if j is not None else 0.18
                noten.append([float(t0), p, d, N[j]["pegel"] if j is not None else pmed])
        if a.timing == "schlagzeug" and noten:
            tz = raster_zeiten(np.array([x[0] for x in noten]), beats, D)
            for x, tt in zip(noten, tz):
                x[0] = float(tt)
        noten = [tuple(x) for x in noten]
        melde(f"… spiele: {KLAENGE[klang]}, {EFFEKTE[eff]}")
        if klang == "eigene":
            y = spielen_eigene(noten, n)
            bank, prog = 0, 28
        else:
            bank, prog = (int(q) for q in klang.split(":"))
            y = bp(spielen_sf(noten, n, bank, prog), 150, 7000)
        if klang == "eigene":
            y = bp(y, 180, 6000)
        P_ = float(np.median(np.diff(beats)))
        y = effekt(y, eff, P_)
        # Lautstärke: ca. 8 dB unter der Skank-Gitarre (ohne Gitarre: 6 dB unter dem Bass)
        def akt(x):
            return np.sqrt(np.mean(x[np.abs(x) > 0.02 * np.abs(x).max()] ** 2))
        bezug = st["guitar"] if "guitar" in st and st["guitar"].any() else st["bass"]
        y *= akt(bezug) / max(akt(y[:, 1]), 1e-12) * 10 ** ((-8 if bezug is not st["bass"] else -6) / 20) \
            * 10 ** (a.lautstaerke_db / 20)
        if np.abs(y).max() > 0.98:
            y *= 0.98 / np.abs(y).max()
        out = os.path.join(a.ziel, "picking_neu.wav")
        sf.write(out, y, SR, subtype="PCM_24")
        skank_neu = None
        if art == "repariert":
            try:
                melde("… spiele auch den Skank neu (eigene Akkord-Aufnahmen, „C fett“), damit nichts doppelt klingt")
                K.SAMPLES = os.path.join(APP, "gitarre_neu", "samples")
                akk_s, phase_s, _ = K.akkord_plan(beats, st.get("piano", 0) + st.get("other", 0) + 3 * git, st["bass"])
                sl_s, zt_s, _ = K.skank_stellen(beats, akk_s, phase_s, git)
                zt_s, _ = K.am_schlagzeug(zt_s, D)
                wahl = K.griffe_waehlen(K.bibliothek(), git, sl_s, zt_s, akk_s)
                skank_neu = K.fett(K.spielen(n, sl_s, zt_s, akk_s, wahl))
                kk = int(2.8 * SR)
                a0, b0 = int(zt_s[0] * SR), int(zt_s[-1] * SR)

                def tr(x):
                    return np.array([np.sqrt(np.mean(x[i:i + kk] ** 2)) for i in range(a0, max(a0 + 1, b0 - kk), kk)])
                skank_neu *= 2.0 * np.percentile(tr(git), 75) / max(np.median(tr(skank_neu.mean(1))), 1e-12)
                sf.write(os.path.join(a.ziel, "skank_neu.wav"), skank_neu, SR, subtype="PCM_24")
            except Exception as e:
                melde(f"… Skank konnte nicht neu gespielt werden ({str(e)[:80]}) – alte Gitarren-Spur bleibt im Song")
                skank_neu = None
                art = "repariert_alte_bleibt"
        rest = None
        for s_, p_ in spuren.items():
            if s_ == "guitar" and skank_neu is not None:
                x = skank_neu
                rest = x if rest is None else rest[:min(len(rest), len(x))] + x[:min(len(rest), len(x))]
                continue
            x, _ = lb.load(p_, sr=SR, mono=False)
            x = np.atleast_2d(x).T
            if x.shape[1] == 1:
                x = np.repeat(x, 2, 1)
            rest = x if rest is None else rest[:min(len(rest), len(x))] + x[:min(len(rest), len(x))]
        L = min(len(rest), len(y))
        mix = rest[:L] + y[:L]
        mix *= min(1.0, 0.98 / max(np.abs(mix).max(), 1e-12))
        sf.write(os.path.join(a.ziel, "song_mit_picking.wav"), mix, SR, subtype="PCM_24")
        midi = midi_schreiben(noten, bank, prog, a.ziel, a.midi_bpm)
        zeilen = ["Picking-Gitarre – gedämpfte Begleit-Gitarre, spielt die Basslinie eine Oktave höher",
                  {"neu": "Keine Picking-Gitarre im Song gefunden – neu aus der Basslinie gespielt",
                   "repariert": "Vorhandene Picking-Gitarre erkannt – ihre Töne übernommen; im Song ist die ganze "
                                "Gitarren-Spur ersetzt (neuer Skank + reparierte Picking-Gitarre)",
                   "repariert_alte_bleibt": "Vorhandene Picking-Gitarre erkannt – ihre Töne übernommen (alte "
                                            "Gitarren-Spur bleibt im Song)"}[art],
                  f"Klang: {KLAENGE[klang]}, Klangfarbe: {EFFEKTE[eff]} (Effekt nur im Audio)",
                  f"Timing: {'genau mit jedem Bass-Anschlag' if a.timing == 'bass' else 'auf dem Schlagzeug-Raster'}",
                  f"Tonart: {K.NAMEN[ton[0]]}-{ton[1]}, Song-Tempo ca. {60 / P_:.1f} BPM",
                  f"MIDI-Datei mit {a.midi_bpm:g} BPM geschrieben: Logic-Projekt auf {a.midi_bpm:g} BPM stellen.",
                  "Im MIDI: Noten mit Anschlagstärke, Bank- und Programmwechsel (GM 29 Muted Guitar bei eigenen Tönen), "
                  "Tempo, 4/4-Takt.", "", "Zeit       Ton"]
        for tt, p, d, _ in noten:
            zeilen.append(f"{int(tt // 60)}:{tt % 60:06.3f}  {K.NAMEN[p % 12]}{p // 12 - 1}")
        with open(os.path.join(a.ziel, "picking_neu_Notenliste.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(zeilen) + "\n")
        bericht = {"trennung": modell, "klang": KLAENGE[klang], "effekt": EFFEKTE[eff], "timing": a.timing,
                   "toene": len(noten), "art": art, "modus": a.modus, "erkennung": erk, **info, "tonart": f"{K.NAMEN[ton[0]]}-{ton[1]}", "midi": midi}
        json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
        import shutil
        shutil.rmtree(os.path.join(a.ziel, "spuren_roh"), ignore_errors=True)
        for f in glob.glob(os.path.join(a.ziel, "_*")):
            os.remove(f)
        print(json.dumps({"ok": True, "datei": out, "ordner": a.ziel, "art": art}, ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "fehler": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
