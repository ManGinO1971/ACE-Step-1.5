#!/usr/bin/env python3
"""
HELTECH Stems – Lead-Gitarre reparieren (Melodie-Gitarre: Fills, Melodien, Soli). Stand 10. Nov 2026.

Ablauf:
  1. Ganzer Song -> Trennung in 6 Spuren (BS-Roformer-SW, Rückfall htdemucs_6s)  (oder nur die Gitarren-Spur)
  2. Tonhöhen-Kurve der Gitarre (pyin, Hochpass): einstimmige, stimmhafte Abschnitte = Lead-Kandidaten;
     Skank-Akkorde fallen über die geringe Stimmhaftigkeit heraus, Picking (kurz, tief, Tonklasse = Bass
     am Bass-Anschlag) wird ausgeschlossen; einzelne verstreute Töne zählen nicht als Lead
  3. Anschläge innerhalb eines Laufs (Onset-Spitze + Tonsprung); Slides/Bends/Hammer-ons ohne neuen
     Anschlag bleiben ein durchgehender Ton – die genaue Tonhöhen-Kurve (Bends, Vibrato) wird übernommen
  4. Neu gespielt: Probe mit veränderlicher Lese-Geschwindigkeit, die genau der Kurve folgt;
     Lautstärke-Verlauf wie das Original; Klang: eigene Clean-Töne oder Gitarren aus GeneralUser GS
  5. Klangfarbe/Effekt (wie Picking + Overdrive)
  6. Alte Lead-Gitarre wird über ihre Obertöne (entlang der Kurve) aus der Gitarren-Spur genommen,
     Skank/Picking bleiben; Song = übrige Spuren + Gitarre ohne Lead + neue Lead-Gitarre
  7. Ausgabe: lead_neu.wav, gitarre_ohne_lead.wav, song_mit_lead_neu.wav, MIDI (mit Pitch-Bend), Notenliste

Aufruf:
  python lead_neu_step.py <song.wav> <ziel> [--klang eigene|0:27|…] [--effekt original|overdrive|…]
                          [--nur-gitarre] [--midi-bpm 120]
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np
import soundfile as sf

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP, "piano_neu"))
sys.path.insert(0, os.path.join(APP, "picking_neu"))
import skank_kern as K  # noqa: E402

SR = 44100
HOP = 256
PROBEN = os.path.join(HIER, "samples")
KLAENGE = {
    "eigene": "Eigene Clean-Gitarre", "0:27": "Clean-Gitarre", "8:27": "Clean-Gitarre mit Chorus",
    "12:27": "Clean-Gitarre 2", "0:26": "Jazz-Gitarre", "0:29": "Overdrive-Gitarre", "0:30": "Distortion-Gitarre",
    "11:29": "Wah-Gitarre", "8:26": "Hawaii-Gitarre", "0:25": "Steel-Gitarre", "0:24": "Nylon-Gitarre",
    "8:25": "12-saitige Gitarre", "16:25": "Mandoline"}
EFFEKTE = {"original": "Original", "overdrive": "Overdrive", "warm": "Warm (Vintage)", "fett": "Fett",
           "hell": "Hell (Twang)", "chorus": "Chorus", "dub_echo": "Dub-Echo", "federhall": "Federhall",
           "phaser": "Phaser", "tremolo": "Tremolo", "autowah": "Auto-Wah"}
GM_PROG = {"eigene": (0, 27)}


def melde(t):
    print(t, flush=True)


def lb():
    return K.librosa()


def bandpass(y, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(2, [lo, hi], "bandpass", fs=SR, output="sos"), y, axis=0)


def hochpass(y, f):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(2, f, "highpass", fs=SR, output="sos"), y, axis=0)


# ------------------------------------------------------------ 1) Tonhöhen-Kurve und Lead-Abschnitte
def tonkurve(y):
    f0, vf, vp = lb().pyin(hochpass(y, 150), fmin=160, fmax=1400, sr=SR, frame_length=2048, hop_length=HOP,
                           max_transition_rate=60, switch_prob=0.02)
    return f0, vf, vp


def bass_anschlaege(bass):
    L_ = lb()
    on = L_.onset.onset_detect(y=bass, sr=SR, units="time", hop_length=HOP, backtrack=True)
    if not len(on):
        return np.array([]), np.array([])
    bs = L_.resample(bass, orig_sr=SR, target_sr=SR // 4)
    bf, _, _ = L_.pyin(bs, fmin=35, fmax=300, sr=SR // 4, frame_length=1024, hop_length=HOP // 4)
    ton = []
    for t in on:
        k = int(t * SR / HOP); seg = bf[k:k + 8]
        ton.append(float(np.nanmedian(L_.hz_to_midi(seg))) if np.isfinite(seg).any() else -99.0)
    return np.array(on), np.array(ton)


def lead_finden(y, f0, vf, vp, bass_t, bass_ton):
    n = len(f0)
    t = np.arange(n) * HOP / SR
    rms = lb().feature.rms(y=y, frame_length=2048, hop_length=HOP)[0][:n]
    leise = rms < 0.06 * np.percentile(rms, 99)
    gut = vf & (vp > 0.15) & ~leise & np.isfinite(f0)
    laeufe, i = [], 0
    while i < n:
        if not gut[i]:
            i += 1
            continue
        j = i
        while j < n and (gut[j] or (j + 2 < n and gut[j + 1:j + 3].all())):
            j += 1
        laeufe.append((i, j))
        i = j
    kand = []
    for a, b in laeufe:
        dauer = (b - a) * HOP / SR
        if dauer < 0.12:
            continue
        mid = float(np.nanmedian(lb().hz_to_midi(f0[a:b])))
        if len(bass_t):                                  # Picking: kurz, tief, Tonklasse wie der Bass
            j = int(np.argmin(np.abs(bass_t - t[a])))
            if abs(bass_t[j] - t[a]) < 0.05 and (round(mid) - round(bass_ton[j])) % 12 == 0 and dauer < 0.35 and mid < 66:
                continue
        kand.append((a, b, dauer))
    # Lead = Phrasen: ein Lauf zählt, wenn er lang ist (>= 0,3 s) oder Nachbarn im Abstand <= 1,5 s hat
    lead = []
    for k, (a, b, d) in enumerate(kand):
        nachbarn = sum(1 for q, (a2, b2, _) in enumerate(kand) if q != k and abs(t[a2] - t[a]) <= 1.5)
        if d >= 0.3 or nachbarn >= 2:
            lead.append((a, b))
    return lead, t


def noten_aus_laeufen(lead, f0, y):
    L_ = lb()
    o = L_.onset.onset_strength(y=hochpass(y, 300), sr=SR, hop_length=HOP)
    noten = []
    for a, b in lead:
        m = L_.hz_to_midi(np.where(np.isfinite(f0[a:b]), f0[a:b], np.nan))
        # Anfang auf die Anschlag-Spitze kurz davor/danach (±3 Rahmen)
        s0 = max(0, a - 3); a2 = s0 + int(np.argmax(o[s0:min(len(o), a + 4)]))
        cuts = [min(a2, a)]
        ref = np.nanpercentile(o[a:b], 60) if b - a > 4 else 0
        for k in range(a + 4, b - 3):
            spitze = o[k] >= o[k - 1] and o[k] >= o[k + 1] and o[k] > 2.0 * ref
            vor, nach = np.nanmedian(m[max(0, k - a - 4):k - a]), np.nanmedian(m[k - a:k - a + 4])
            if spitze and k - cuts[-1] >= 6 and (abs(nach - vor) > 0.6 or o[k] > 4.0 * ref):
                cuts.append(k)
        cuts.append(b)
        for s, e in zip(cuts[:-1], cuts[1:]):
            if e - s >= 4:
                noten.append((s, e, bool(s != cuts[0])))
    return noten


def kurve_fuellen(f):
    f = np.array(f, float)
    ok = np.isfinite(f)
    if not ok.any():
        return None
    f[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), f[ok])
    return f


# ------------------------------------------------------------ 2) Klang
def eigene_toene():
    info = json.load(open(os.path.join(PROBEN, "toene.json"), encoding="utf-8"))
    out = []
    for e in info:
        x, sr = sf.read(os.path.join(PROBEN, e["datei"]))
        x = x.mean(1) if x.ndim > 1 else x
        if sr != SR:
            x = lb().resample(x, orig_sr=sr, target_sr=SR)
        out.append((float(e["hz"]), x))
    return sorted(out, key=lambda p: p[0])


def sf_toene(bank, prog):
    import sf2_spieler as sp
    import piano_neu_step as P
    ins = sp.SoundFont(P.soundfont_sichern()).instrument(bank, prog)
    out = []
    for k in range(52, 89, 3):                               # E3 … E6
        y = sp.spielen(ins, [(0.0, 4.0, k, 100)], 4.5, nachklang_max=0.5)
        y = y.mean(1) if y.ndim > 1 else y
        out.append((float(lb().midi_to_hz(k)), y / (np.abs(y).max() + 1e-9)))
    return out


def spiele_kurve(toene, kurve_hz, n):
    """Probe so lesen, dass sie genau der Tonhöhen-Kurve folgt (Bends, Vibrato, Slides)."""
    ziel = float(np.nanmedian(kurve_hz))
    f_p, x = min(toene, key=lambda p: abs(np.log2(p[0] / ziel)))
    rate = np.interp(np.arange(n), np.linspace(0, n - 1, len(kurve_hz)), kurve_hz) / f_p
    pos = np.minimum(np.cumsum(rate) - rate[0], len(x) - 2)
    return np.interp(pos, np.arange(len(x)), x)


def lead_spielen(noten, f0, y, toene, L):
    out = np.zeros(L + SR)
    rms = lb().feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    phrasen = []
    i = 0
    while i < len(noten):
        j = i + 1
        while j < len(noten) and noten[j][2] and noten[j][0] == noten[j - 1][1]:
            j += 1
        phrasen.append((noten[i][0], noten[j - 1][1]))
        i = j
    for k, (s, e) in enumerate(phrasen):
        kurve = kurve_fuellen(f0[s:e])
        if kurve is None:
            continue
        kurve = np.convolve(np.r_[kurve[0], kurve, kurve[-1]], [0.25, 0.5, 0.25], "valid")
        # Ausklang: bis zur nächsten Phrase, höchstens 0,15 s
        frei = (phrasen[k + 1][0] - e) if k + 1 < len(phrasen) else 999
        aus = int(min(0.15 * SR / HOP, max(2, frei - 1)))
        kurve = np.r_[kurve, np.full(aus, kurve[-1])]
        n = len(kurve) * HOP
        z = spiele_kurve(toene, kurve, n)
        env = np.r_[rms[s:e], np.zeros(aus)][:len(kurve)]
        env = np.convolve(env, np.ones(5) / 5, "same")
        env[-aus:] = np.linspace(env[-aus - 1] if len(env) > aus else 0, 0, aus)
        g = np.interp(np.arange(n), np.linspace(0, n - 1, len(env)), env / (env.max() + 1e-9))
        z = z * np.maximum(g, 0) ** 0.7 * float(np.percentile(rms[s:e], 90))
        a = s * HOP
        out[a:a + n] += z[:max(0, len(out) - a)]
    return out[:L], phrasen


def overdrive(y, staerke=4.0):
    from scipy.signal import butter, sosfiltfilt
    x = y / (np.abs(y).max() + 1e-9)
    z = np.tanh(staerke * x) / np.tanh(staerke)
    z = sosfiltfilt(butter(2, 5500, "lowpass", fs=SR, output="sos"), z, axis=0)
    return z * (np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(z ** 2)) + 1e-12))


def effekt(y, eff, schlag):
    if eff == "overdrive":
        z = overdrive(y)
        return np.stack([z * 0.95, z], 1)
    if eff == "original":
        return np.stack([y * 0.95, y], 1)
    import picking_neu_step as G
    return G.effekt(y, eff, schlag)


# ------------------------------------------------------------ 3) alte Lead-Gitarre herausnehmen
def lead_entfernen(git_st, f0, noten, tiefe_db=-40):
    from scipy.signal import stft, istft
    n_fft = 4096
    aktiv = np.zeros(len(f0), bool)
    for s, e, _ in noten:
        aktiv[max(0, s - 4):e + 25] = True
    if not aktiv.any():
        return git_st.copy()
    kurve = kurve_fuellen(np.where(aktiv, f0, np.nan))
    g = 10 ** (tiefe_db / 20)
    out = []
    for c in range(git_st.shape[1]):
        f, _, Z = stft(git_st[:, c], fs=SR, nperseg=n_fft, noverlap=n_fft - HOP, boundary=None, padded=True)
        M = np.ones(Z.shape)
        nf = min(Z.shape[1], len(aktiv))
        for k in np.flatnonzero(aktiv[:nf]):
            for h in range(1, 25):
                fh = kurve[k] * h
                if fh > 16000:
                    break
                M[(f >= fh * 2 ** (-1.0 / 12) - 15) & (f <= fh * 2 ** (1.0 / 12) + 15), k] = g
        M = np.minimum(M, np.minimum(np.roll(M, 1, 1), np.roll(M, -1, 1)))
        _, x = istft(Z * M, fs=SR, nperseg=n_fft, noverlap=n_fft - HOP, boundary=None)
        out.append(np.r_[x, np.zeros(max(0, len(git_st) - len(x)))][:len(git_st)])
    return np.stack(out, 1)


# ------------------------------------------------------------ MIDI mit Pitch-Bend (Bereich ±12 Halbtöne)
def midi_schreiben(phrasen, f0, rms_rel, bank, prog, ziel, bpm=120.0):
    L_ = lb()
    ppq = 960
    tempo = float(bpm) if bpm and bpm > 0 else 120.0
    tick = lambda s: int(round(s * tempo / 60.0 * ppq))  # noqa: E731
    ev = []
    zeilen = []
    for s, e in phrasen:
        k = kurve_fuellen(f0[s:e])
        if k is None:
            continue
        m = L_.hz_to_midi(k)
        note = int(round(float(np.median(m[:max(3, len(m) // 4)]))))
        t0, t1 = s * HOP / SR, e * HOP / SR
        v = int(np.clip(70 + 50 * float(np.percentile(rms_rel[s:e], 90)), 50, 120))
        ev.append((tick(t0) - 1 if tick(t0) > 0 else 0, 0, bytes([0xE0, 0x00, 0x40])))       # Bend auf Mitte
        ev.append((tick(t0), 1, bytes([0x90, note, v])))
        letzt = 8192
        for q in range(0, len(m), 4):                     # alle ~23 ms
            w = int(np.clip(8192 + (m[q] - note) / 12.0 * 8191, 0, 16383))
            if abs(w - letzt) >= 40:
                ev.append((tick((s + q) * HOP / SR), 2, bytes([0xE0, w & 0x7F, (w >> 7) & 0x7F])))
                letzt = w
        ev.append((tick(t1), 3, bytes([0x80, note, 0])))
        zeilen.append((t0, t1 - t0, note, float(np.max(m) - np.min(m))))
    ev.sort(key=lambda x: (x[0], x[1]))

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
    nb = "Lead-Gitarre".encode("utf-8")
    d = b"\x00\xff\x03" + vlq(len(nb)) + nb
    d += b"\x00" + bytes([0xB0, 0, bank & 0x7F]) + b"\x00" + bytes([0xC0, prog & 0x7F])
    # Pitch-Bend-Bereich ±12 Halbtöne (RPN 0)
    d += b"\x00\xb0\x65\x00\x00\xb0\x64\x00\x00\xb0\x06\x0c\x00\xb0\x26\x00\x00\xb0\x65\x7f\x00\xb0\x64\x7f"
    letzt = 0
    for tk, _, b in ev:
        d += vlq(max(0, tk - letzt)) + b
        letzt = max(letzt, tk)
    d += b"\x00\xff\x2f\x00"
    p = os.path.join(ziel, "lead_neu.mid")
    with open(p, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + spur(t0) + spur(d))
    return p, zeilen


def stereo(x):
    x = np.atleast_2d(x)
    if x.shape[0] < x.shape[1]:
        x = x.T
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    return x[:, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe")
    ap.add_argument("ziel")
    ap.add_argument("--klang", default="eigene")
    ap.add_argument("--effekt", default="original")
    ap.add_argument("--nur-gitarre", action="store_true", help="Eingabe ist schon die Gitarren-Spur")
    ap.add_argument("--midi-bpm", type=float, default=120.0)
    ap.add_argument("--modelle", default=os.path.expanduser("~/HELTECH_Stems/models/"))
    ap.add_argument("--spuren-ordner", default=None)
    a = ap.parse_args()
    os.makedirs(a.ziel, exist_ok=True)
    try:
        L_ = lb()
        klang = a.klang if a.klang in KLAENGE else "eigene"
        eff = a.effekt if a.effekt in EFFEKTE else "original"
        spuren, modell = {}, "keine"
        if a.nur_gitarre:
            spuren = {"guitar": a.eingabe}
        elif a.spuren_ordner:
            spuren = {s: glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))[0]
                      for s in K.STIMMEN if glob.glob(os.path.join(a.spuren_ordner, f"*{s}*.wav"))}
            modell = "vorhanden"
        else:
            melde("… trenne den Song in 6 Spuren (dauert einige Minuten)")
            spuren, modell = K.trennen(a.eingabe, a.ziel, a.modelle)
            melde(f"✔ Spuren getrennt ({modell})")
        if "guitar" not in spuren:
            raise RuntimeError("Keine Gitarren-Spur gefunden")
        git_st = stereo(L_.load(spuren["guitar"], sr=SR, mono=False)[0])
        git = git_st.mean(1)
        L = len(git)
        if "bass" in spuren:
            bass = L_.load(spuren["bass"], sr=SR, mono=True)[0][:L]
            bass_t, bass_ton = bass_anschlaege(bass)
        else:
            bass_t, bass_ton = np.array([]), np.array([])
        schlag = 0.75
        if "drums" in spuren:
            try:
                beats = K.taktraster(K.lade_mono(spuren["drums"]))
                schlag = float(np.median(np.diff(beats)))
            except Exception:
                pass
        melde("… lese die Tonhöhen-Kurve der Gitarre (Bends, Vibrato, Slides)")
        f0, vf, vp = tonkurve(git)
        lead, t = lead_finden(git, f0, vf, vp, bass_t, bass_ton)
        noten = noten_aus_laeufen(lead, f0, git)
        lead_s = sum(e - s for s, e, _ in noten) * HOP / SR
        if lead_s < 2.0:
            bericht = {"art": "keine_lead", "lead_sekunden": round(lead_s, 1), "trennung": modell}
            json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
            melde("✔ In dieser Gitarren-Spur spielt keine Lead-Gitarre (nur Skank/Begleitung) – nichts geändert")
            print(json.dumps({"ok": True, "datei": "", "ordner": a.ziel, "art": "keine_lead"}, ensure_ascii=False))
            return
        abschnitte = sorted({int(t[s] // 10) * 10 for s, _, _ in noten})
        melde(f"✔ Lead-Gitarre gefunden: {len(noten)} Töne, {lead_s:.0f} s Spielzeit "
              f"(um {', '.join(f'{x // 60}:{x % 60:02d}' for x in abschnitte[:12])})")
        melde(f"… spiele neu: {KLAENGE[klang]}, {EFFEKTE[eff]}")
        if klang == "eigene":
            toene = eigene_toene()
            bank, prog = GM_PROG["eigene"]
        else:
            bank, prog = (int(q) for q in klang.split(":"))
            toene = sf_toene(bank, prog)
        neu, phrasen = lead_spielen(noten, f0, git, toene, L)
        neu = bandpass(neu, 90, 9000)
        st = effekt(neu, eff, schlag)
        maske = np.zeros(L, bool)
        for s, e, _ in noten:
            maske[s * HOP:e * HOP] = True
        st *= np.sqrt(np.mean(git[maske] ** 2)) / (np.sqrt(np.mean(st.mean(1)[maske] ** 2)) + 1e-12)
        melde("… nehme die alte Lead-Gitarre aus der Gitarren-Spur (Skank bleibt)")
        ohne = lead_entfernen(git_st, f0, noten)
        st = st[:L]
        if np.abs(st).max() > 0.98:
            st *= 0.98 / np.abs(st).max()
        out = os.path.join(a.ziel, "lead_neu.wav")
        sf.write(out, st, SR, subtype="PCM_24")
        sf.write(os.path.join(a.ziel, "gitarre_ohne_lead.wav"), ohne, SR, subtype="PCM_24")
        if not a.nur_gitarre:
            rest = None
            for s_, p_ in spuren.items():
                x = ohne if s_ == "guitar" else stereo(L_.load(p_, sr=SR, mono=False)[0])
                rest = x if rest is None else rest[:min(len(rest), len(x))] + x[:min(len(rest), len(x))]
            n = min(len(rest), len(st))
            mix = rest[:n] + st[:n]
            mix *= min(1.0, 0.98 / max(np.abs(mix).max(), 1e-12))
            sf.write(os.path.join(a.ziel, "song_mit_lead_neu.wav"), mix, SR, subtype="PCM_24")
        rms = L_.feature.rms(y=git, frame_length=1024, hop_length=HOP)[0]
        midi, zeilen = midi_schreiben(phrasen, f0, rms / (np.percentile(rms, 99) + 1e-12), bank, prog, a.ziel, a.midi_bpm)
        txt = ["Lead-Gitarre – Melodie-Gitarre (Fills, Melodien, Soli) neu gespielt mit der Tonhöhen-Kurve des Originals",
               f"Klang: {KLAENGE[klang]}, Klangfarbe: {EFFEKTE[eff]} (Effekt nur im Audio)",
               f"MIDI-Datei mit {a.midi_bpm:g} BPM: Logic-Projekt auf {a.midi_bpm:g} BPM stellen.",
               "Im MIDI: je gespielter Phrase eine Note mit Pitch-Bend (Bereich ±12 Halbtöne) für Bends, Vibrato und "
               "Slides, Bank-/Programmwechsel (GM 28 Clean Guitar bei eigenen Tönen).", "",
               "Zeit       Dauer   Ton    Bend-Umfang"]
        for t0, d, p, umf in zeilen:
            txt.append(f"{int(t0 // 60)}:{t0 % 60:06.3f}  {d:5.2f}s  {K.NAMEN[p % 12]}{p // 12 - 1:<3}  {umf:4.1f} HT")
        with open(os.path.join(a.ziel, "lead_neu_Notenliste.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(txt) + "\n")
        bericht = {"art": "repariert", "trennung": modell, "klang": KLAENGE[klang], "effekt": EFFEKTE[eff],
                   "toene": len(noten), "phrasen": len(phrasen), "lead_sekunden": round(lead_s, 1), "midi": midi}
        json.dump(bericht, open(os.path.join(a.ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
        import shutil
        shutil.rmtree(os.path.join(a.ziel, "spuren_roh"), ignore_errors=True)
        for f in glob.glob(os.path.join(a.ziel, "_*")):
            os.remove(f)
        print(json.dumps({"ok": True, "datei": out, "ordner": a.ziel, "art": "repariert"}, ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "fehler": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
