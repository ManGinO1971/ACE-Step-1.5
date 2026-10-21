#!/usr/bin/env python3
"""
Background V2 (NEU, 21. Okt 2026) – saubere Background-Vorlage.

Hörtest/Analyse Root Nah Bow.1: die getrennte Background-Spur bestand zu ~2/3
aus Lead (durchgesickert, gleicher Ton zur gleichen Zeit) und Lead-Echo
("tape echo on the vocal ends"); nur ~14 s waren echte Background-Linie.
Solo-Background (z.B. "(Ooh-ooh)" im Intro) landet dagegen oft in der LEAD-Spur.

Dieses Skript baut daraus eine saubere Vorlage (Wunsch des Inhabers: "Weg 1 plus
die Stellen, die der Lead nicht singt, plus Kontrolle mit den Lyrics"):
  1. aus der Background-Spur raus: gleicher Ton wie der Lead zur selben Zeit
     (Durchsickern) und Wiederholung eines Lead-Tons 0,12-0,8 s später (Echo)
     – AUSSER in {…}-Zeilen (dort singt der Background mit dem Lead zusammen)
  2. dazu: Gesang aus der Lead-Spur, der zu KEINER Lead-Textzeile gehört
     (Text-Abgleich) -> Solo-Background wie "(Ooh-ooh)"
  3. Quelle fürs Neu-Singen = Instrumente + saubere Background-Linie, so laut
     wie sonst der Lead

Aufruf: python backing_v2_clean_step.py <lead_ref> <backing_ref> <song_ohne_background>
        <f0_lead.npy> <f0_backing.npy> <lead_lines.json|-> <ausgabeordner>
  lead_lines.json: [{"t0":..,"t1":..,"conf":..,"together":true|false}, ...]
Erzeugt: clean_backing.wav, source.wav. Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

E = lambda k, d: float(os.environ.get(k, d))  # noqa: E731
HOP_S = 0.01


def load(path, sr=None):
    import librosa

    y, s = sf.read(path, always_2d=True)
    y = y.T.astype(np.float32)
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    y = y[:2]
    if sr and s != sr:
        y = np.vstack([librosa.resample(c, orig_sr=s, target_sr=sr) for c in y])
        s = sr
    return y, s


def env_db(y, sr):
    m = y.mean(axis=0)
    h = int(sr * HOP_S)
    n = len(m) // h
    return 20 * np.log10(np.sqrt(np.mean(m[: n * h].reshape(n, h) ** 2, axis=1)) + 1e-9)


def midi(f):
    return np.where(f > 0, 12 * np.log2(np.maximum(f, 1e-6) / 440.0) + 69, np.nan)


def pc_dist(a, b):
    return np.abs(((a - b) + 6) % 12 - 6)


def smooth_gain(keep, hold_f, att_f, rel_f):
    g = keep.astype(float).copy()
    idx = np.flatnonzero(keep)
    for t in idx:
        g[t:t + hold_f + 1] = 1.0
    out = g.copy()
    for t in range(1, len(out)):
        tgt = g[t]
        step = 1.0 / (att_f if tgt > out[t - 1] else rel_f)
        out[t] = min(tgt, out[t - 1] + step) if tgt > out[t - 1] else max(tgt, out[t - 1] - step)
    return 0.5 - 0.5 * np.cos(np.pi * out)


def main():
    if len(sys.argv) < 8:
        print(__doc__)
        return 1
    lead_p, back_p, song_p, f0l_p, f0b_p, lines_p, out_dir = sys.argv[1:8]
    win_p = sys.argv[8] if len(sys.argv) > 8 else "-"
    os.makedirs(out_dir, exist_ok=True)
    back, sr = load(back_p)
    lead, _ = load(lead_p, sr)
    song, _ = load(song_p, sr)
    n_s = min(back.shape[1], lead.shape[1], song.shape[1])
    back, lead, song = back[:, :n_s], lead[:, :n_s], song[:, :n_s]
    eB, eL = env_db(back, sr), env_db(lead, sr)
    B, L = midi(np.load(f0b_p)), midi(np.load(f0l_p))
    n = min(len(eB), len(eL), len(B), len(L))
    eB, eL, B, L = eB[:n], eL[:n], B[:n], L[:n]
    refB, refL = np.percentile(eB, 95), np.percentile(eL, 95)

    lines = []
    if lines_p != "-" and os.path.isfile(lines_p):
        lines = [x for x in json.load(open(lines_p)) if float(x.get("conf", 0)) >= E("BV2_LINE_MIN_CONF", 0.4)]
    together = np.zeros(n, dtype=bool)
    in_lead_line = np.zeros(n, dtype=bool)
    pad = int(E("BV2_LINE_PAD_S", 0.4) / HOP_S)
    for x in lines:
        a, b = max(0, int(x["t0"] / HOP_S) - pad), min(n, int(x["t1"] / HOP_S) + pad)
        in_lead_line[a:b] = True
        if x.get("together"):
            together[max(0, int(x["t0"] / HOP_S)):min(n, int(x["t1"] / HOP_S))] = True

    # NEU (21. Okt 2026, Wunsch des Inhabers "Fenster nur zum Aufräumen, nicht zum
    # Abschalten"): Zeitfenster der (…)/{…}-Stellen aus Songtext + Text-Abgleich.
    # IN den Fenstern: normale Regeln. AUSSERHALB: Lead-Anteil strenger raus, aber
    # echter Background (klar anderer Ton als der Lead) bleibt überall erhalten.
    windows = []
    if win_p != "-" and os.path.isfile(win_p):
        windows = [w for w in json.load(open(win_p)) if len(w) >= 2 and w[1] > w[0]]
    win = np.ones(n, dtype=bool)
    if windows:
        win[:] = False
        for a, b in windows:
            win[max(0, int(a / HOP_S)):min(n, int(b / HOP_S) + 1)] = True

    # 1. Background-Spur: Durchsickern + Echo raus (nicht in {…}-Zeilen)
    act_b = (eB > refB + E("BV2_ACTIVE_DB", -25)) & ~np.isnan(B)
    same = np.zeros(n, dtype=bool)
    echo = np.zeros(n, dtype=bool)
    lo_f, hi_f = int(E("BV2_ECHO_MIN_S", 0.12) / HOP_S), int(E("BV2_ECHO_MAX_S", 0.8) / HOP_S)
    hi_strict = int(E("BV2_STRICT_ECHO_MAX_S", 1.5) / HOP_S)
    tol_same, tol_echo = 1.0, 0.5
    tol_same_s, tol_echo_s = E("BV2_STRICT_SAME_ST", 2.0), E("BV2_STRICT_ECHO_ST", 1.0)
    for i in np.flatnonzero(act_b & ~together):
        strict = windows and not win[i]
        ts, te, hf = (tol_same_s, tol_echo_s, hi_strict) if strict else (tol_same, tol_echo, hi_f)
        if not np.isnan(L[i]) and pc_dist(B[i], L[i]) < ts:
            same[i] = True
            continue
        w = L[max(0, i - hf):max(0, i - lo_f)]
        w = w[~np.isnan(w)]
        if len(w) and np.min(pc_dist(B[i], w)) < te:
            echo[i] = True
    keep_b = act_b & ~same & ~echo

    # 2. Solo-Background aus der Lead-Spur (Gesang außerhalb jeder Lead-Textzeile)
    solo = np.zeros(n, dtype=bool)
    if lines:
        solo = (eL > refL + E("BV2_SOLO_DB", -30)) & ~np.isnan(L) & ~in_lead_line

    # Mini-Schnipsel streichen
    def drop_short(mask, min_f):
        m = mask.copy()
        i = 0
        while i < len(m):
            if m[i]:
                j = i
                while j < len(m) and m[j]:
                    j += 1
                if j - i < min_f:
                    m[i:j] = False
                i = j
            else:
                i += 1
        return m

    keep_b = drop_short(keep_b, int(E("BV2_MIN_S", 0.12) / HOP_S))
    solo = drop_short(solo, int(E("BV2_MIN_S", 0.12) / HOP_S))
    hold, att, rel = int(E("BV2_HOLD_S", 0.15) / HOP_S), max(1, int(0.03 / HOP_S)), max(1, int(0.15 / HOP_S))
    gb = smooth_gain(keep_b, hold, att, rel)
    gs = smooth_gain(solo, hold, att, rel)
    t_audio = np.arange(n_s) / sr
    t_fr = np.arange(n) * HOP_S
    gb_s = np.interp(t_audio, t_fr, gb)
    gs_s = np.interp(t_audio, t_fr, gs)
    clean = back * gb_s[None, :] + lead * gs_s[None, :]
    sf.write(os.path.join(out_dir, "clean_backing.wav"), clean.T, sr, subtype="PCM_24")

    # 3. Quelle: Instrumente (Song ohne Background minus Lead) + saubere Linie, so laut wie der Lead
    instr = song - lead
    e_clean = env_db(clean, sr)[:n]
    voiced = (gb > 0.5) | (gs > 0.5)
    ref_clean = np.percentile(e_clean[voiced], 95) if voiced.any() else refL
    gain_db = float(np.clip(refL - ref_clean + E("BV2_SOURCE_EXTRA_DB", 0), 0, E("BV2_MAX_BOOST_DB", 15)))
    src = instr + clean * (10 ** (gain_db / 20))
    pk = float(np.max(np.abs(src))) if src.size else 0.0
    if pk > 0.99:
        src *= 0.99 / pk
    sf.write(os.path.join(out_dir, "source.wav"), src.T, sr)

    report = {
        "ok": True,
        "background_active_s": round(act_b.sum() * HOP_S, 1),
        "lead_bleed_s": round(same.sum() * HOP_S, 1),
        "lead_echo_s": round(echo.sum() * HOP_S, 1),
        "background_line_s": round(keep_b.sum() * HOP_S, 1),
        "solo_from_lead_s": round(solo.sum() * HOP_S, 1),
        "together_s": round(together.sum() * HOP_S, 1),
        "lead_lines_used": len(lines),
        "text_windows": len(windows),
        "text_windows_s": round(float(win.sum()) * HOP_S, 1) if windows else None,
        "kept_inside_windows_s": round(float((keep_b & win).sum() + (solo & win).sum()) * HOP_S, 1) if windows else None,
        "kept_outside_windows_s": round(float((keep_b & ~win).sum() + (solo & ~win).sum()) * HOP_S, 1) if windows else None,
        "boost_db": round(gain_db, 1),
    }
    print(f"Saubere Background-Vorlage: Linie {report['background_line_s']} s, Solo aus Lead-Spur {report['solo_from_lead_s']} s, "
          f"entfernt: Lead {report['lead_bleed_s']} s, Echo {report['lead_echo_s']} s, Quelle +{gain_db:.1f} dB", file=sys.stderr, flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
