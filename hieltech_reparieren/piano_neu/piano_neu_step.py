#!/usr/bin/env python3
"""
HELTECH Stems – Piano / Keys neu einspielen (Stand 5. Nov 2026). Funktioniert ohne Musikprogramm:
das Ergebnis klingt fertig, das MIDI ist zusätzlich für alle, die weiterarbeiten wollen.

Ablauf:
  1. Eingabe: ganzer Song (empfohlen – Trennung in 6 Spuren mit BS-Roformer-SW wie bei der Gitarre) oder Piano-Spur
     Ganzer Song + Piano spielt Skank: Akkorde aus allen Instrumenten, Skank über den ganzen Song (auch wo die
     Trennung das Piano verliert), Griffe vom Original-Piano, zusätzliche erkannte Töne, gleichmäßige Lautstärke
  2. Noten erkennen (polyphon): ByteDance „High-resolution Piano Transcription“ (Noten, Anschlag, Haltepedal),
     falls installiert – sonst Basic Pitch (wie die MIDI-Funktion der App)
  3. optional: Timing korrigieren (Raster aus der Spur selbst, wie bei Schlagzeug/Bass)
  4. Neu spielen mit einem Klang aus der SoundFont GeneralUser GS (Flügel, Rhodes, Clavinet, Orgel …),
     eigener SF2-Abspieler (sf2_spieler.py), Lautstärke wie im Original
  5. Ausgabe: piano_neu.wav, piano_neu.mid (Programmwechsel, Anschlag, Haltepedal, Tempo, Takt),
     Noten-/Akkordliste, bericht.json (+ song_mit_neuem_piano.wav bei ganzem Song)

Aufruf:
  python piano_neu_step.py <eingabe> <ziel> [--klang 0] [--ganzer-song] [--timing-korrigieren] [--midi-bpm 120]
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
sys.path.insert(0, HIER)
SR = 44100
SOUNDFONT = os.path.join(HIER, "soundfont", "GeneralUser-GS.sf2")
KLAENGE = {0: "Flügel", 1: "Heller Flügel", 2: "E-Grand (CP-70)", 3: "Honky-Tonk", 4: "Rhodes (Tine E-Piano)",
           5: "FM-E-Piano (DX)", 6: "Cembalo", 7: "Clavinet", 16: "Hammond-Orgel", 17: "Perkussive Orgel",
           18: "Rock-Orgel", 19: "Kirchenorgel", 20: "Harmonium"}
BYTEDANCE_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
BYTEDANCE_BYTES_MIN = 1.6e8
# Python mit Basic Pitch: auf dem Mac die eigene Umgebung, auf dem Server (RunPod) per HIELTECH_MIDI_PYTHON
MIDI_ENV_PYTHON = os.environ.get("HIELTECH_MIDI_PYTHON") or os.path.expanduser("~/miniforge3/envs/heltech_stems_midi/bin/python")
NAMEN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def melde(t):
    print(t, flush=True)

def _gpu():
    """GPU-Schalter der App (gpu_rechner.py): Modul zurück, wenn „GPU“ gewählt ist, sonst None."""
    try:
        sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
        import gpu_rechner as G
        return G if G.gpu_an() else None
    except Exception:  # noqa: BLE001
        return None


def _speicher():
    """Speicher der App für schon gerechnete Trennungen (gpu_rechner.py) – unabhängig vom Schalter GPU/CPU."""
    try:
        sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
        import gpu_rechner as G
        return G
    except Exception:  # noqa: BLE001
        return None


try:
    sys.path.insert(0, os.path.expanduser("~/HELTECH_Stems"))
    from merker import merke
except Exception:  # noqa: BLE001 – ohne Merker läuft alles wie bisher
    def merke(*_a, **_k):
        return lambda fn: fn



SOUNDFONT_BYTES = 32319396


def soundfont_sichern():
    """Die SoundFont liegt in 2 Teilen im Ordner (Übertragung max. 20 MB je Datei) und wird beim ersten Lauf
    zusammengesetzt und auf Vollständigkeit geprüft."""
    if os.path.exists(SOUNDFONT) and os.path.getsize(SOUNDFONT) == SOUNDFONT_BYTES:
        return SOUNDFONT
    teile = sorted(glob.glob(SOUNDFONT + ".teil*"))
    if not teile:
        raise RuntimeError(f"SoundFont fehlt: {SOUNDFONT}")
    with open(SOUNDFONT + ".tmp", "wb") as f:
        for t in teile:
            with open(t, "rb") as q:
                f.write(q.read())
    if os.path.getsize(SOUNDFONT + ".tmp") != SOUNDFONT_BYTES:
        raise RuntimeError("SoundFont-Teile unvollständig")
    os.replace(SOUNDFONT + ".tmp", SOUNDFONT)
    return SOUNDFONT


# ------------------------------------------------------------ 1. Eingabe
def piano_heraustrennen(eingabe, ziel):
    roh = os.path.join(ziel, "spuren_roh")
    G = _gpu()
    if G:
        try:
            sp = G.demucs(eingabe, "htdemucs_6s")
            if "piano" in sp:
                return {s: sp[s] for s in ("piano", "drums", "bass", "guitar", "other", "vocals") if s in sp}
        except Exception as e:  # noqa: BLE001
            melde(f"… GPU nicht verfügbar ({str(e)[:150]}) – rechne auf dem Mac")
    Sp = _speicher()
    da = Sp.gespeichert(eingabe, Sp.a_demucs("htdemucs_6s")) if Sp else None
    if da and "piano" in da:
        melde("… Trennung dieses Songs schon vorhanden (Zwischenspeicher) – wird wiederverwendet")
        return {s: da[s] for s in ("piano", "drums", "bass", "guitar", "other", "vocals") if s in da}
    p = subprocess.run([sys.executable, "-m", "demucs", "--name", "htdemucs_6s", "--out", roh, eingabe],
                       capture_output=True, text=True)
    d = os.path.join(roh, "htdemucs_6s", os.path.splitext(os.path.basename(eingabe))[0])
    if not os.path.exists(os.path.join(d, "piano.wav")):
        raise RuntimeError("Piano-Trennung fehlgeschlagen: " + (p.stderr or "")[-300:])
    out = {s: os.path.join(d, s + ".wav") for s in ("piano", "drums", "bass", "guitar", "other", "vocals")
           if os.path.exists(os.path.join(d, s + ".wav"))}
    return Sp.merken(eingabe, Sp.a_demucs("htdemucs_6s"), out) if Sp else out


# ------------------------------------------------------------ 2. Noten
def _download(url, ziel, mindest):
    import urllib.request
    os.makedirs(os.path.dirname(ziel), exist_ok=True)
    for _ in range(5):
        if os.path.exists(ziel) and os.path.getsize(ziel) >= mindest:
            return ziel
        schon = os.path.getsize(ziel) if os.path.exists(ziel) else 0
        melde(f"… lade Noten-Modell ({schon // 1_000_000} MB geladen)")
        req = urllib.request.Request(url, headers={"Range": f"bytes={schon}-"} if schon else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(ziel, "ab" if schon and r.status == 206 else "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:
            melde(f"  Download unterbrochen ({str(e)[:80]}), versuche weiter …")
    if os.path.exists(ziel) and os.path.getsize(ziel) >= mindest:
        return ziel
    raise RuntimeError("Noten-Modell konnte nicht geladen werden")


@merke("noten_bytedance", nur=(0,))
def noten_bytedance(pfad, modelle):
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate
    ck = _download(BYTEDANCE_URL, os.path.join(modelle, "piano_transcription", "note_F1=0.9677_pedal_F1=0.9186.pth"),
                   BYTEDANCE_BYTES_MIN)
    audio, _ = librosa.load(pfad, sr=sample_rate, mono=True)
    pt = PianoTranscription(device="cpu", checkpoint_path=ck)
    tmp = os.path.join(os.path.dirname(os.path.abspath(pfad)), "_tmp_piano.mid")
    erg = pt.transcribe(audio, tmp)
    noten = [(float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]), int(e["velocity"]))
             for e in erg["est_note_events"]]
    pedal = [(float(e["onset_time"]), float(e["offset_time"])) for e in erg.get("est_pedal_events", [])]
    return noten, pedal, "ByteDance Piano Transcription"


BASIC_PITCH_HELFER = r'''
import json, os, sys
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH as M
arg = dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60)
try:
    _, _, ev = predict(sys.argv[1], M, **arg)
except Exception:
    from basic_pitch.inference import Model
    _, _, ev = predict(sys.argv[1], Model(os.path.join(os.path.dirname(M), "nmp.onnx")), **arg)
json.dump([[float(a), float(b), int(p), float(v)] for a, b, p, v, *_ in ev], open(sys.argv[2], "w"))
'''


@merke("noten_basic_pitch", nur=(0,))
def noten_basic_pitch(pfad, ziel):
    out = os.path.join(ziel, "_bp_noten.json")
    helfer = os.path.join(ziel, "_bp_helfer.py")
    with open(helfer, "w") as f:
        f.write(BASIC_PITCH_HELFER)
    for py in (sys.executable, MIDI_ENV_PYTHON):
        if not os.path.exists(py):
            continue
        r = subprocess.run([py, helfer, pfad, out], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            ev = json.load(open(out))
            # Basic-Pitch-Amplitude (0..1) -> Anschlag
            noten = [(a, b, p, int(np.clip(round(30 + 100 * v ** 0.8), 1, 127))) for a, b, p, v in ev]
            return noten, [], "Basic Pitch"
    raise RuntimeError("keine Notenerkennung verfügbar (weder ByteDance noch Basic Pitch)")


def noten_erkennen(pfad, ziel, modelle):
    try:
        import piano_transcription_inference  # noqa: F401
        return noten_bytedance(pfad, modelle)
    except Exception as e:
        melde(f"… ByteDance nicht verfügbar ({str(e)[:80]}), nehme Basic Pitch")
        return noten_basic_pitch(pfad, ziel)


def midi_lesen(pfad):
    import mido
    m = mido.MidiFile(pfad)
    t, tempo, offen, out = 0.0, 500000, {}, []
    for msg in mido.merge_tracks(m.tracks):
        t += mido.tick2second(msg.time, m.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            offen[msg.note] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on") and msg.note in offen:
            a0, v = offen.pop(msg.note)
            out.append((a0, t, msg.note, v))
    return sorted(out)


def pedal_anwenden(noten, pedal):
    """Haltepedal: Note klingt bis Pedal-Ende, wenn sie bei gedrücktem Pedal losgelassen wurde."""
    if not pedal:
        return noten
    out = []
    for a, b, p, v in noten:
        for pa, pb in pedal:
            if pa <= b < pb:
                b = pb
                break
        out.append((a, b, p, v))
    return out


def anschlaege_verfeinern(noten, m):
    """Notenerkennung arbeitet in groben Zeitstufen (Basic Pitch: 11,6 ms). Jeden Anschlag (Akkord = Noten
    innerhalb 30 ms) auf den echten Anschlag in der Spur setzen: steilster Anstieg im Fenster −40…+20 ms,
    dann wie beim Schlagzeug der Punkt, an dem 10 % der dortigen Spitze erreicht sind."""
    import librosa
    hop = 64
    o = librosa.onset.onset_strength(y=m, sr=SR, hop_length=hop)
    starts = sorted({round(a, 4) for a, *_ in noten})
    gruppen = []
    for t in starts:
        if gruppen and t - gruppen[-1][0] <= 0.03:
            gruppen[-1].append(t)
        else:
            gruppen.append([t])
    karte = {}
    verschoben = []
    for g in gruppen:
        t = g[0]
        i0, i1 = max(0, int((t - 0.04) * SR / hop)), int((t + 0.02) * SR / hop)
        if i1 <= i0 or i1 > len(o):
            continue
        k = i0 + int(np.argmax(o[i0:i1]))
        a0 = max(0, k * hop - int(0.03 * SR))
        seg = np.abs(m[a0:k * hop + int(0.02 * SR)])
        if len(seg) == 0 or seg.max() <= 0:
            continue
        neu = (a0 + int(np.argmax(seg > seg.max() * 0.1))) / SR
        if abs(neu - t) <= 0.04:
            for x in g:
                karte[x] = neu - g[0]
            verschoben.append(neu - t)
    out = [(a + karte.get(round(a, 4), 0.0), b + karte.get(round(a, 4), 0.0), p, v) for a, b, p, v in noten]
    return out, {"anschlaege": len(gruppen), "verfeinert": len(verschoben),
                 "versatz_ms_median": round(float(np.median(verschoben)) * 1000, 1) if verschoben else 0.0}


# ------------------------------------------------------------ 3. Timing
def timing_korrigieren(noten, m):
    import timing_korrektur as tk
    on = np.array(sorted({round(a, 3) for a, *_ in noten}))
    # Anschläge innerhalb 30 ms = ein Akkord
    gruppen = []
    for t in on:
        if gruppen and t - gruppen[-1][-1] <= 0.03:
            gruppen[-1].append(t)
        else:
            gruppen.append([t])
    zeiten = np.array([g[0] for g in gruppen])
    schub, info = tk.raster_schub(zeiten, tk.tempo_schaetzen(m, SR))
    karte = {}
    for g, s in zip(gruppen, schub):
        for t in g:
            karte[t] = s
    return [(a + karte.get(round(a, 3), 0.0), b + karte.get(round(a, 3), 0.0), p, v) for a, b, p, v in noten], info


# ------------------------------------------------------------ 5. MIDI + Listen
def midi_schreiben(noten, pedal, programm, ziel, bpm=120.0, name="Piano repariert"):
    ppq = 960
    tempo = float(bpm) if bpm and bpm > 0 else 120.0
    tick = lambda s: int(round(s * tempo / 60.0 * ppq))  # noqa: E731
    ev = []
    for a, b, p, v in noten:
        ev.append((tick(a), 2, 0x90, p, int(np.clip(v, 1, 127))))
        ev.append((max(tick(b), tick(a) + 1), 0, 0x80, p, 0))
    for a, b in pedal:
        ev.append((tick(a), 1, 0xB0, 64, 127))
        ev.append((tick(b), 0, 0xB0, 64, 0))
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
    nb = name.encode("utf-8")
    d = b"\x00\xff\x03" + vlq(len(nb)) + nb + b"\x00" + bytes([0xC0, programm & 0x7F])
    letzt = 0
    for tk_, _, st, a1, a2 in ev:
        d += vlq(tk_ - letzt) + bytes([st, a1, a2])
        letzt = tk_
    d += b"\x00\xff\x2f\x00"
    datei = os.path.join(ziel, "piano_neu.mid")
    with open(datei, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + spur(t0) + spur(d))
    return datei


AKK = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
       "dim": (0, 3, 6), "sus4": (0, 5, 7), "sus2": (0, 2, 7)}


def akkord_liste(noten, laenge, fenster=1.0):
    """Akkord je Abschnitt aus den erkannten Noten (gewichtet nach Dauer und Anschlag)."""
    vor = []
    for r in range(12):
        for q, iv in AKK.items():
            v = np.zeros(12)
            v[[(r + i) % 12 for i in iv]] = 1
            vor.append((NAMEN[r] + q, v / np.linalg.norm(v), len(iv)))
    zeilen, letzt = [], None
    t = 0.0
    while t < laenge:
        c = np.zeros(12)
        for a, b, p, v in noten:
            ue = min(b, t + fenster) - max(a, t)
            if ue > 0:
                c[p % 12] += ue * v
        if c.sum() > 0:
            c /= np.linalg.norm(c)
            best = max(vor, key=lambda x: float(c @ x[1]) - 0.02 * (x[2] - 3))
            if best[0] != letzt:
                zeilen.append(f"{int(t // 60)}:{t % 60:04.1f}  {best[0]}")
                letzt = best[0]
        t += fenster
    return zeilen


# ------------------------------------------------------------ Skank (ganzer Song, wie die Gitarre)
def standard_griff(a, K):
    pcs = [(a[0] + iv) % 12 for iv in K.QUAL[a[1]]]
    g = sorted(55 + ((p - 55) % 12) for p in pcs)
    return g + [g[0] + 12]


def skank_plan(st, noten, timing):
    """Akkorde aus allen Instrumenten, Skank-Stellen wo Piano ODER Gitarre skankt, Griffe vom Original-Piano,
    zusätzliche erkannte Töne (nicht auf den Skank-Stellen, passend zur Tonart). Rückgabe: Skank-Noten,
    zusätzliche Noten, Infos – oder None, wenn das Piano keinen Skank spielt (dann exakt nach Noten spielen)."""
    import skank_kern as K
    pno, git = st["piano"], st.get("guitar", st["piano"] * 0)
    beats = K.taktraster(st["drums"])
    akk, phase, ton = K.akkord_plan(beats, 3 * pno + st.get("other", 0) + git, st["bass"])
    sl_p, z_p, _ = K.skank_stellen(beats, akk, phase, pno)
    d = {}
    if git.any():
        sl_g, z_g, _ = K.skank_stellen(beats, akk, phase, git)
        d = {s: z for s, z in zip(sl_g, z_g)}
    d.update({s: z for s, z in zip(sl_p, z_p)})
    slots = sorted(d)
    zeiten = np.array([d[s] for s in slots])
    if len(slots) < 16:
        return None
    # spielt das Piano überhaupt Skank? (Anteil der Piano-Anschläge auf Skank-Stellen)
    ans = np.array(sorted({round(a, 2) for a, *_ in noten}))
    if len(ans):
        auf = float(np.mean([np.min(np.abs(zeiten - t)) < 0.07 for t in ans]))
        if auf < 0.4:
            return None
    else:
        auf = 0.0
    D = K.schlagzeug_anschlaege(st["drums"])
    am = None
    if timing:
        zeiten, am = K.am_schlagzeug(zeiten, D)
    griff = {}
    for a in {akk[i] for i, _ in slots}:
        zahl, cnt = {}, 0
        for z, (i, p) in zip(zeiten, slots):
            if akk[i] != a:
                continue
            sel = {x[2] for x in noten if abs(x[0] - z) < 0.06}
            cnt += bool(sel)
            for x in sel:
                zahl[x] = zahl.get(x, 0) + 1
        pcs = {(a[0] + iv) % 12 for iv in K.QUAL[a[1]]}
        g = sorted(p for p, c in zahl.items() if cnt and c >= 0.35 * cnt and p % 12 in pcs and 50 <= p <= 79)
        griff[a] = g if len({p % 12 for p in g}) >= 3 else standard_griff(a, K)
    skala = {(ton[0] + i) % 12 for i in ((0, 2, 3, 5, 7, 8, 10) if ton[1] == "moll" else (0, 2, 4, 5, 7, 9, 11))}
    extra = [x for x in noten if np.min(np.abs(zeiten - x[0])) > 0.07 and x[2] >= 50 and x[2] % 12 in skala
             and x[1] - x[0] >= 0.08]
    if extra and timing:
        ze, _ = K.am_schlagzeug(np.array([x[0] for x in extra]), D)
        extra = [(float(z), float(z + (b - a)), p, v) for z, (a, b, p, v) in zip(ze, extra)]
    extra = [(a, b, p, int(np.clip(v, 55, 100))) for a, b, p, v in extra]
    rng = np.random.default_rng(3)
    P = float(np.median(np.diff(beats)))
    dauer = min(0.3 * P, 0.24)
    sk = []
    for z, (i, p) in zip(zeiten, slots):
        v = int(np.clip(92 + rng.normal(0, 3), 1, 127))
        for t_ in griff[akk[i]]:
            sk.append((float(z), float(z) + dauer, t_, v))
    info = {"skank_anschlaege": len(slots), "zusaetzliche_toene": len(extra), "piano_auf_skank_anteil": round(auf, 2),
            "tonart": f"{K.NAMEN[ton[0]]}-{ton[1]}", "tempo_bpm": round(60.0 / P, 2),
            "am_schlagzeug": am, "griffe": {K.akkord_name(a): [K.NAMEN[p % 12] + str(p // 12 - 1) for p in g]
                                             for a, g in griff.items()}}
    plan = ["Piano-Skank – Akkord-Plan, erkannt aus allen Instrumenten",
            f"Tonart: {info['tonart']}, Song-Tempo ca. {info['tempo_bpm']:.1f} BPM (A# = B♭, D# = E♭)",
            "Griffe (vom Original-Piano übernommen): " + ", ".join(f"{k} = {' '.join(v)}" for k, v in info["griffe"].items()), ""]
    for k, t in enumerate(range(phase, len(akk), 4)):
        s = beats[t]
        plan.append(f"{int(s // 60)}:{s % 60:05.2f}  Takt {k + 1:3d}  " + " | ".join(f"{K.akkord_name(x):6s}" for x in akk[t:t + 4]))
    return sk, extra, info, plan, zeiten


def gleichmaessig(y, ref, zeiten, faktor):
    """Lautstärke wie bei der Gitarre: Takt-Pegel an den kräftigen Stellen des Original-Pianos (75 %)."""
    k = int(2.8 * SR)
    a0, b0 = int(zeiten[0] * SR), int(zeiten[-1] * SR)

    def tr(x):
        return np.array([np.sqrt(np.mean(x[i:i + k] ** 2)) for i in range(a0, max(a0 + 1, b0 - k), k)])
    ist = float(np.median(tr(y.mean(1))))
    return y * (faktor * float(np.percentile(tr(ref), 75)) / max(ist, 1e-12))


# ------------------------------------------------------------ Ablauf
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe")
    ap.add_argument("ziel")
    ap.add_argument("--klang", type=int, default=0, help="GM-Programm aus GeneralUser GS (0 Flügel, 4 Rhodes, 7 Clavinet, 16 Orgel …)")
    ap.add_argument("--ganzer-song", action="store_true")
    ap.add_argument("--timing-korrigieren", action="store_true")
    ap.add_argument("--midi-bpm", type=float, default=120.0)
    ap.add_argument("--modelle", default=os.path.expanduser("~/HELTECH_Stems/models/"))
    ap.add_argument("--spuren-ordner", default=None, help="schon getrennte Spuren (drums/bass/guitar/piano/other)")
    ap.add_argument("--noten-midi", default=None, help="(Test) schon erkannte Noten aus einer MIDI-Datei")
    ap.add_argument("--lautstaerke-db", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.ziel, exist_ok=True)
    try:
        import librosa
        import sf2_spieler as sp
        spuren = None
        piano = a.eingabe
        if a.spuren_ordner:
            spuren = {s: glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))[0]
                      for s in ("drums", "bass", "guitar", "piano", "other", "vocals")
                      if glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))}
            piano = spuren["piano"]
        elif a.ganzer_song:
            import skank_kern as K
            melde("… trenne den Song in 6 Spuren (dauert einige Minuten)")
            spuren, modell = K.trennen(a.eingabe, a.ziel, a.modelle)
            melde(f"✔ Spuren getrennt ({modell})")
            if "piano" not in spuren:
                raise RuntimeError("Trennung lieferte keine Piano-Spur")
            piano = spuren["piano"]
        m, _ = librosa.load(piano, sr=SR, mono=True)
        melde("… erkenne die Noten (mehrstimmig)")
        if a.noten_midi:
            noten, pedal, verfahren = midi_lesen(a.noten_midi), [], "MIDI-Datei (Test)"
        else:
            noten, pedal, verfahren = noten_erkennen(piano, a.ziel, a.modelle)
        noten = [n for n in noten if 21 <= n[2] <= 108 and n[1] > n[0]]
        melde(f"✔ {len(noten)} Noten erkannt ({verfahren})")
        noten, feinzeit = anschlaege_verfeinern(noten, m)
        melde(f"✔ Anschläge auf die Spur ausgerichtet ({feinzeit['verfeinert']} von {feinzeit['anschlaege']})")
        klang = KLAENGE.get(a.klang, f"Programm {a.klang}")
        instr = sp.SoundFont(soundfont_sichern()).instrument(0, a.klang)
        timing, skank, plan = None, None, None
        if spuren and all(s in spuren for s in ("drums", "bass")):
            melde("… Takt, Akkorde aus allen Instrumenten und Skank-Stellen")
            st = {s: librosa.load(p_, sr=SR, mono=True)[0].astype(np.float64) for s, p_ in spuren.items() if s != "vocals"}
            nn = min(len(v) for v in st.values())
            st = {s: v[:nn] for s, v in st.items()}
            erg = skank_plan(st, noten, a.timing_korrigieren)
            if erg is None:
                melde("… das Piano spielt hier keinen Skank – spiele die erkannten Noten genau nach")
            else:
                sk, extra, skank, plan, zeiten = erg
                melde(f"✔ Skank: {skank['skank_anschlaege']} Anschläge über den ganzen Song, "
                      f"{skank['zusaetzliche_toene']} zusätzliche Töne")
                melde(f"… spiele neu: {klang}")
                y = sp.spielen(instr, sk, len(m) / SR, nachklang_max=0.12)
                if extra:
                    y2 = sp.spielen(instr, pedal_anwenden(extra, pedal), len(m) / SR, nachklang_max=1.5)
                    L = min(len(y), len(y2))
                    y = y[:L] + y2[:L]
                y = gleichmaessig(y, st["piano"], zeiten, 1.4 * 10 ** (a.lautstaerke_db / 20))
                noten = sorted(sk + [(x[0], x[1], x[2], x[3]) for x in pedal_anwenden(extra, pedal)])
                pedal = []          # Skank ohne Haltepedal (im MIDI schon in die Notenlängen eingerechnet)
                timing = {"am_schlagzeug": skank["am_schlagzeug"]} if a.timing_korrigieren else None
        if skank is None:
            if a.timing_korrigieren and noten:
                noten, timing = timing_korrigieren(noten, m)
                melde(f"✔ Timing: {timing.get('verschoben', 0)} Anschläge aufs Raster")
            melde(f"… spiele neu: {klang}")
            y = sp.spielen(instr, pedal_anwenden(noten, pedal), len(m) / SR)

            # Lautstärke wie das Original (über die gespielten Stellen)
            def rms(x):
                return float(np.sqrt(np.mean(x ** 2))) + 1e-12
            aktiv = np.abs(m) > np.abs(m).max() * 0.01
            g = rms(m[aktiv]) / rms(y.mean(1)[aktiv[:len(y)]]) if aktiv.any() else 1.0
            y *= g * 10 ** (a.lautstaerke_db / 20)
        spitze = float(np.abs(y).max())
        if spitze > 0.98:
            y *= 0.98 / spitze
        out = os.path.join(a.ziel, "piano_neu.wav")
        sf.write(out, y, SR, subtype="PCM_24")
        if spuren:
            rest = None
            for s_, p_ in spuren.items():
                if s_ == "piano":
                    continue
                x, _ = librosa.load(p_, sr=SR, mono=False)
                x = np.atleast_2d(x).T
                rest = x if rest is None else rest[:min(len(rest), len(x))] + x[:min(len(rest), len(x))]
            n = min(len(rest), len(y))
            mix = rest[:n] + y[:n]
            mix *= min(1.0, 0.98 / max(np.abs(mix).max(), 1e-12))
            sf.write(os.path.join(a.ziel, "song_mit_neuem_piano.wav"), mix, SR, subtype="PCM_24")
        midi = midi_schreiben(noten, pedal, a.klang, a.ziel, a.midi_bpm)
        from collections import Counter
        c = Counter(p for _, _, p, _ in noten)
        zeilen = [f"Piano repariert – Klang: {klang} (GeneralUser GS, GM-Programm {a.klang + 1})",
                  f"Noten erkannt mit: {verfahren}" + (" (inkl. Haltepedal)" if pedal else ""),
                  f"MIDI-Datei mit {a.midi_bpm:g} BPM geschrieben: Logic-Projekt auf {a.midi_bpm:g} BPM stellen.",
                  "Im MIDI: Noten mit Anschlagstärke, Programmwechsel (Klang)" + (", Haltepedal (CC64)" if pedal else "")
                  + ", Tempo, 4/4-Takt.",
                  "", f"Noten gesamt: {len(noten)}, tiefste {NAMEN[min(c) % 12]}{min(c) // 12 - 1}, "
                  f"höchste {NAMEN[max(c) % 12]}{max(c) // 12 - 1}" if c else "", "", "Akkorde (Zeit  Akkord):"]
        if plan:
            zeilen[-1:] = [f"Gespielt als Skank: {skank['skank_anschlaege']} Anschläge + "
                           f"{skank['zusaetzliche_toene']} zusätzliche Töne vom Original-Piano", ""] + plan
        else:
            zeilen += akkord_liste(noten, len(m) / SR)
        with open(os.path.join(a.ziel, "piano_neu_Notenliste.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(zeilen) + "\n")
        bericht = {"verfahren": verfahren, "noten": len(noten), "anschlaege_ausgerichtet": feinzeit, "pedal": len(pedal), "klang": klang,
                   "programm": a.klang, "timing": timing, "skank": skank, "midi": midi}
        json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
        import shutil
        shutil.rmtree(os.path.join(a.ziel, "spuren_roh"), ignore_errors=True)
        for f in glob.glob(os.path.join(a.ziel, "_bp_*")) + glob.glob(os.path.join(a.ziel, "_tmp_*")):
            os.remove(f)
        print(json.dumps({"ok": True, "datei": out, "ordner": a.ziel, "noten": len(noten), "klang": klang},
                         ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "fehler": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
