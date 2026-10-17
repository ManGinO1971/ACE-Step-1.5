#!/usr/bin/env python3
"""
Instrumental-Mix-Balance (NEU, 13. Okt 2026, Nutzerwunsch "Instrumente
erzeugen: professionelle Lautstaerke- und Mix-Anpassung").

Ziel: Bass + Schlagzeug bilden das Fundament, Begleitinstrumente (Piano,
Orgel, Saxofon, Gitarre, Blaeser, Pads ...) bleiben klar hoerbar - OHNE
pauschal alles lauter zu machen. Dazu:

  1. Das fertige Instrumental wird mit Demucs htdemucs_6s in Spuren zerlegt:
     drums, bass, guitar, piano, other (Orgel/Sax/Blaeser/Streicher/Synths),
     vocals (im Instrumental nur Reste).
  2. Jede Spur wird einzeln gemessen (Lautheit nach ITU-R BS.1770: K-Filter
     + Gating, also wie das Ohr Lautstaerke empfindet - Bass zaehlt nicht
     kuenstlich mehr als Hoehen).
  3. Gegen das Fundament (Bass+Schlagzeug zusammen) wird je Spur ein
     genre-abhaengiges Zielfenster geprueft. NUR Spuren ausserhalb ihres
     Fensters werden angepasst (zu leise -> hoeher, max +5 dB; zu dominant
     -> tiefer, max -4 dB). Spuren, die im Song praktisch nicht vorkommen,
     werden NICHT angehoben (sonst wuerde man nur Trennungs-Reste/Rauschen
     lauter machen).
  4. Bass/Schlagzeug-Verhaeltnis wird ebenfalls ins Genre-Fenster gebracht
     (z.B. Reggae: Bass etwas vor dem Schlagzeug).
  5. Neu-Zusammensetzung: Summe der angepassten Spuren PLUS der Rest
     (Original minus Summe aller Spuren), damit durch die Trennung nichts
     verloren geht. Pro Spur EIN fester Faktor (keine Pump-Effekte),
     danach Schutz gegen Uebersteuern (gleichmaessiges Absenken, keine
     Verzerrung). Die Gesamtlautheit gleicht runpod-handler.js am Ende
     ohnehin wieder an den Original-Song an.

Aufruf: python instrumental_balance_step.py <eingang> <ausgang.wav> [genre-text]
Ausgabe auf stdout: eine JSON-Zeile mit Messwerten und Anpassungen.
Exit-Code != 0 bei Fehler (der Aufrufer liefert dann das unveraenderte
Instrumental aus).
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

MODEL_NAME = os.environ.get("HIELTECH_BALANCE_MODEL", "htdemucs_6s")
MAX_BOOST_DB = 5.0
MAX_CUT_DB = 4.0
MAX_BASS_DRUM_CORRECTION_DB = 3.0
# Spur gilt als "nicht vorhanden", wenn sie so viel leiser als das
# Fundament ist (dann nie anheben).
ABSENT_BELOW_RHYTHM_DB = 24.0

# Genre-Profile. acc: Ziel-Lautheit der Begleitspur relativ zum Fundament
# (Bass+Schlagzeug zusammen) in dB; Fenster = [Ziel, Ziel + 4 dB].
# bass_minus_drums: erlaubtes Fenster fuer (Bass - Schlagzeug) in dB.
PROFILES = {
    # Reggae/Dub/Roots: Bass fuehrt, Skank-Gitarre + Orgel-Bubble + Piano
    # gut hoerbar, aber hinter dem Riddim
    "reggae": {"bass_minus_drums": (0.0, 4.0), "acc": {"guitar": -7.0, "piano": -6.0, "other": -5.0}},
    # Rai/Maghreb: Melodie-Instrumente (Synth, Akkordeon, Geige, Ney,
    # Darbuka-Begleitung) weit vorne
    "rai": {"bass_minus_drums": (-4.0, 1.0), "acc": {"guitar": -5.0, "piano": -5.0, "other": -2.5}},
    "afro": {"bass_minus_drums": (-2.0, 3.0), "acc": {"guitar": -4.5, "piano": -6.0, "other": -4.0}},
    "pop": {"bass_minus_drums": (-4.0, 0.5), "acc": {"guitar": -5.0, "piano": -4.0, "other": -3.5}},
    "rock": {"bass_minus_drums": (-5.0, 0.0), "acc": {"guitar": -2.5, "piano": -6.0, "other": -5.0}},
    "hiphop": {"bass_minus_drums": (-2.0, 3.0), "acc": {"guitar": -8.0, "piano": -6.0, "other": -5.0}},
    "electronic": {"bass_minus_drums": (-3.0, 2.0), "acc": {"guitar": -8.0, "piano": -6.0, "other": -3.5}},
    "jazz": {"bass_minus_drums": (-4.0, 1.0), "acc": {"guitar": -4.0, "piano": -2.5, "other": -2.5}},
    "latin": {"bass_minus_drums": (-3.0, 1.5), "acc": {"guitar": -4.0, "piano": -4.0, "other": -3.0}},
    "ballad": {"bass_minus_drums": (-3.0, 2.0), "acc": {"guitar": -3.0, "piano": -2.0, "other": -2.5}},
    "default": {"bass_minus_drums": (-3.0, 2.0), "acc": {"guitar": -5.0, "piano": -4.5, "other": -4.0}},
}

GENRE_KEYWORDS = [
    ("reggae", ["reggae", "dub", "roots", "dancehall", "ska", "rocksteady", "lovers rock", "nyahbinghi"]),
    ("rai", ["raï", "rai ", "rai,", "chaabi", "gnawa", "maghreb", "arabic", "oriental", "darija", "andalus"]),
    ("afro", ["afrobeat", "afro", "highlife", "amapiano", "soukous", "makossa", "mbalax", "zouk", "kizomba"]),
    ("hiphop", ["hip hop", "hip-hop", "hiphop", "rap", "trap", "drill", "boom bap", "r&b", "rnb"]),
    ("electronic", ["edm", "house", "techno", "trance", "electro", "dubstep", "drum and bass", "synthwave", "lofi", "lo-fi"]),
    ("rock", ["rock", "metal", "punk", "grunge", "indie"]),
    ("jazz", ["jazz", "swing", "bossa", "blues", "soul", "funk", "gospel"]),
    ("latin", ["latin", "salsa", "reggaeton", "bachata", "cumbia", "flamenco", "samba", "tango"]),
    ("ballad", ["ballad", "ballade", "acoustic", "classical", "orchestral", "piano solo", "folk", "country"]),
    ("pop", ["pop", "k-pop", "dance"]),
]

ACC_LABELS = {"guitar": "Gitarre", "piano": "Piano", "other": "Orgel/Sax/Bläser/Synths"}


def detect_genre(text):
    t = f" {(text or '').lower()} "
    for name, words in GENRE_KEYWORDS:
        if any(w in t for w in words):
            return name
    return "default"


# ---------- Lautheit nach ITU-R BS.1770 (K-Filter + Gating) ----------
def _biquad_highshelf(fs, f0=1681.974450955533, gain_db=3.999843853973347, q=0.7071752369554196):
    a = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * q)
    cosw = math.cos(w0)
    b0 = a * ((a + 1) + (a - 1) * cosw + 2 * math.sqrt(a) * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cosw)
    b2 = a * ((a + 1) + (a - 1) * cosw - 2 * math.sqrt(a) * alpha)
    a0 = (a + 1) - (a - 1) * cosw + 2 * math.sqrt(a) * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cosw)
    a2 = (a + 1) - (a - 1) * cosw - 2 * math.sqrt(a) * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_highpass(fs, f0=38.13547087602444, q=0.5003270373238773):
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * q)
    cosw = math.cos(w0)
    b = np.array([(1 + cosw) / 2, -(1 + cosw), (1 + cosw) / 2])
    a = np.array([1 + alpha, -2 * cosw, 1 - alpha])
    return b / a[0], a / a[0]


def lufs(x, fs):
    """x: (channels, samples) float. Gibt integrierte Lautheit in LUFS zurueck
    (-inf bei Stille)."""
    from scipy.signal import lfilter

    x = np.atleast_2d(x).astype(np.float64)
    b1, a1 = _biquad_highshelf(fs)
    b2, a2 = _biquad_highpass(fs)
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=1), axis=1)
    block = int(0.4 * fs)
    hop = int(0.1 * fs)
    if y.shape[1] < block:
        return float("-inf")
    n = 1 + (y.shape[1] - block) // hop
    # mittlere Quadrate je 400-ms-Block, Kanaele summiert (Gewicht 1 fuer L/R)
    sq = y ** 2
    csum = np.concatenate([np.zeros((sq.shape[0], 1)), np.cumsum(sq, axis=1)], axis=1)
    starts = np.arange(n) * hop
    ms = (csum[:, starts + block] - csum[:, starts]) / block
    z = ms.sum(axis=0)
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(z)
    keep = lk > -70.0
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(z[keep].mean()) - 10.0
    keep2 = keep & (lk > rel)
    if not keep2.any():
        return float("-inf")
    return float(-0.691 + 10 * np.log10(z[keep2].mean()))


def power_sum_db(*levels):
    vals = [10 ** (l / 10) for l in levels if l is not None and math.isfinite(l)]
    return 10 * math.log10(sum(vals)) if vals else float("-inf")


# ---------- Planung der Anpassungen (rein rechnerisch, separat testbar) ----------
def plan_gains(levels, genre):
    """levels: {stem: LUFS}. Gibt (gains_db, notes) zurueck."""
    prof = PROFILES.get(genre, PROFILES["default"])
    gains = {k: 0.0 for k in levels}
    notes = []
    ld, lb = levels.get("drums", float("-inf")), levels.get("bass", float("-inf"))

    # 1) Bass/Schlagzeug-Verhaeltnis ins Genre-Fenster bringen (Korrektur
    #    halb ueber Bass, halb ueber Schlagzeug, je max. 3 dB)
    if math.isfinite(ld) and math.isfinite(lb):
        lo, hi = prof["bass_minus_drums"]
        diff = lb - ld
        target = lo if diff < lo else hi if diff > hi else diff
        corr = target - diff
        if abs(corr) >= 0.5:
            half = max(-MAX_BASS_DRUM_CORRECTION_DB, min(MAX_BASS_DRUM_CORRECTION_DB, corr / 2))
            gains["bass"] += half
            gains["drums"] -= half
            notes.append(f"Bass/Schlagzeug {diff:+.1f} dB -> {diff + 2 * half:+.1f} dB")

    rhythm = power_sum_db(ld + gains.get("drums", 0.0), lb + gains.get("bass", 0.0))
    if not math.isfinite(rhythm):
        notes.append("kein Rhythmus-Fundament erkannt - nur Bass/Schlagzeug-Check übersprungen")
        return gains, notes

    # 2) Begleitspuren in ihr Zielfenster relativ zum Fundament
    for stem, target in prof["acc"].items():
        l = levels.get(stem, float("-inf"))
        if not math.isfinite(l):
            continue
        rel = l - rhythm
        if rel < -ABSENT_BELOW_RHYTHM_DB:
            continue  # Instrument kommt praktisch nicht vor - nicht aufblasen
        lo, hi = target, target + 4.0
        g = 0.0
        if rel < lo - 1.0:
            g = min(MAX_BOOST_DB, lo - rel)
        elif rel > hi:
            g = -min(MAX_CUT_DB, rel - hi)
        if abs(g) >= 0.5:
            gains[stem] += g
            notes.append(f"{ACC_LABELS.get(stem, stem)} {rel:+.1f} dB -> {rel + g:+.1f} dB zum Fundament")
    return gains, notes


def _decode_to_wav(src, dst):
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", dst],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError("ffmpeg-Dekodierung fehlgeschlagen: " + (r.stderr or "")[-300:])


def separate(wav_path):
    """Gibt (fs, mix(2,n), {stem: (2,n)}) zurueck."""
    import torch
    from demucs.api import Separator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sep = Separator(model=MODEL_NAME, device=device, shifts=0, overlap=0.25, split=True, progress=False, jobs=0)
    origin, res = sep.separate_audio_file(wav_path)
    stems = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in res.items()}
    mix = origin.detach().cpu().numpy().astype(np.float64)
    print(f"[Instrumental-Balance] Demucs {MODEL_NAME} auf {device}", file=sys.stderr, flush=True)
    return sep.samplerate, mix, stems


def balance(mix, stems, fs, genre, drop_vocals=False):
    n = min([mix.shape[1]] + [s.shape[1] for s in stems.values()])
    mix = mix[:, :n]
    stems = {k: v[:, :n] for k, v in stems.items()}
    levels = {k: lufs(v, fs) for k, v in stems.items()}
    gains, notes = plan_gains(levels, genre)
    # NEU (17. Okt 2026, Karaoke-Test): Gesangs-Reste (das "Summen" der
    # Lead-Melodie, das flow_edit_morph manchmal stehen laesst) komplett
    # herausnehmen - nur wenn ausdruecklich angefordert.
    if drop_vocals and "vocals" in stems:
        notes.append("Gesangs-Reste (Summen) entfernt")
    residual = mix - sum(stems.values())
    out = residual.copy()
    for k, v in stems.items():
        if drop_vocals and k == "vocals":
            continue
        out += v * (10 ** (gains.get(k, 0.0) / 20))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    safety_db = 0.0
    if peak > 0.97:
        safety_db = 20 * math.log10(0.97 / peak)
        out *= 0.97 / peak
    return out, levels, gains, notes, safety_db


def main():
    if len(sys.argv) < 3:
        print("Aufruf: instrumental_balance_step.py <eingang> <ausgang.wav> [genre-text]", file=sys.stderr)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    genre_text = sys.argv[3] if len(sys.argv) > 3 else ""
    drop_vocals = len(sys.argv) > 4 and sys.argv[4] == "drop_vocals"
    genre = detect_genre(genre_text)
    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "in.wav")
        _decode_to_wav(src, wav)
        fs, mix, stems = separate(wav)
        out, levels, gains, notes, safety_db = balance(mix, stems, fs, genre, drop_vocals)
        sf.write(dst, out.T.astype(np.float32), fs, subtype="PCM_24")

    report = {
        "ok": True,
        "genre": genre,
        "levels_lufs": {k: (round(v, 1) if math.isfinite(v) else None) for k, v in levels.items()},
        "gains_db": {k: round(v, 1) for k, v in gains.items() if abs(v) >= 0.05},
        "notes": notes,
        "safety_db": round(safety_db, 1),
        "drop_vocals": bool(drop_vocals),
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
