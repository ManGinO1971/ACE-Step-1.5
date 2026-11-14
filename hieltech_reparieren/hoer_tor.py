"""Hör-Modell für alle Reparatur-Module (14. Nov 2026).

Das Hör-Modell der Instrumenten-Erkennung (EfficientAT, AudioSet) hört eine Spur alle 0,5 s ab und sagt je
Instrument-Gruppe, wie sicher es klingt. Das Ergebnis wird je Spur zwischengespeichert: dieselbe Spur wird pro Song
nur EINMAL abgehört, jede weitere Reparatur liest nur nach (kostet dann praktisch keine Zeit).

Benutzung in einem Modul:
    import hoer_tor as ht
    y = ht.tor(y, stem, SR, "bass", bericht)        # spielt nur dort, wo das Instrument im Original zu hören ist
    t, p = ht.verlauf(stem, SR, "hihat")            # Verlauf einer Gruppe (z. B. zum Ergänzen)

Fehlt das Modell (oder torch), ändert nichts das Ergebnis: tor() gibt y unverändert zurück.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HIER = os.path.dirname(os.path.abspath(__file__))
FASSUNG = 1
FENSTER_S, SCHRITT_S = 1.0, 0.5
DETECT_PYTHON = os.path.expanduser("~/miniforge3/envs/heltech_stems_detect/bin/python")
CACHE = os.environ.get("HIELTECH_HOER_CACHE") or os.path.join(
    os.path.expanduser("~"), ".hieltech_hoer_cache")

# AudioSet-Namen je Gruppe (die Gruppe zählt, wenn eines davon sicher klingt)
GRUPPEN = {
    "schlagzeug": ["Drum kit", "Snare drum", "Bass drum", "Hi-hat", "Cymbal", "Drum roll", "Drum machine"],
    "hihat": ["Hi-hat"], "snare": ["Snare drum", "Rimshot"], "kick": ["Bass drum"], "becken": ["Cymbal"],
    "trommel": ["Drum", "Tabla"],
    "percussion": ["Percussion", "Tambourine", "Maraca", "Rattle (instrument)", "Cowbell", "Wood block", "Tabla",
                   "Drum", "Gong", "Mallet percussion", "Marimba, xylophone"],
    "bass": ["Bass guitar", "Double bass"],
    "gitarre": ["Guitar", "Electric guitar", "Acoustic guitar", "Strum", "Steel guitar, slide guitar",
                "Plucked string instrument"],
    "orgel": ["Organ", "Electronic organ", "Hammond organ"],
    "piano": ["Piano", "Electric piano", "Keyboard (musical)"],
    "blaeser": ["Brass instrument", "Trumpet", "Trombone", "French horn", "Saxophone", "Wind instrument, woodwind instrument",
                "Flute", "Clarinet", "Harmonica", "Accordion"],
    "gesang": ["Singing", "Male singing", "Female singing", "Choir", "Vocal music"],
    "synth": ["Synthesizer"],
    "streicher": ["Bowed string instrument", "String section", "Violin, fiddle", "Cello"],
}

_HELFER = r'''
import sys, io, json, contextlib, numpy as np
sys.path.insert(0, sys.argv[1])
import detect_instruments_eat as d
import librosa, torch
with contextlib.redirect_stdout(io.StringIO()):
    m, mel = d.modell()
dev = next(m.parameters()).device
lab = d.ek.labels(d.EAT_DIR)
gruppen = json.loads(sys.argv[4])
idx = {g: [lab.index(n) for n in v if n in lab] for g, v in gruppen.items()}
y, _ = librosa.load(sys.argv[2], sr=32000, mono=True)
W, H = int(float(sys.argv[5]) * 32000), int(float(sys.argv[6]) * 32000)
st = np.arange(0, max(1, len(y) - W + 1), H)
P = []
n = 64 if str(dev) != "cpu" else 16
for k in range(0, len(st), n):
    x = torch.from_numpy(np.stack([np.pad(y[s:s + W], (0, max(0, W - len(y[s:s + W])))) for s in st[k:k + n]])).float().to(dev)
    with torch.no_grad():
        p, _ = m(mel(x).unsqueeze(1))
    P.append(torch.sigmoid(p.float()).cpu().numpy())
P = np.concatenate(P)
out = {"t": st / 32000 + W / 64000}
for g, ii in idx.items():
    out[g] = P[:, ii].max(1) if ii else np.zeros(len(st))
np.savez(sys.argv[3], **out)
'''


def _app_ordner():
    """Ordner mit detect_instruments_eat.py + hoermodell/ (Mac: ~/HELTECH_Stems, GPU: hieltech_reparieren)."""
    for d in (HIER, os.path.dirname(HIER), os.path.expanduser("~/HELTECH_Stems")):
        if os.path.isfile(os.path.join(d, "detect_instruments_eat.py")):
            return d
    return None


def _schluessel(mono):
    h = hashlib.sha1(f"{FASSUNG}|{FENSTER_S}|{SCHRITT_S}|".encode())
    h.update(np.round(np.asarray(mono[::7], dtype=np.float32), 4).tobytes())
    return h.hexdigest()[:24]


def zeitleiste(audio, sr):
    """Alle Gruppen-Verläufe für eine Spur (Array, mono oder stereo). Rückgabe dict {"t": …, gruppe: …} oder None."""
    if audio is None:
        return None
    mono = np.asarray(audio, dtype=np.float32)
    mono = mono.mean(1) if mono.ndim == 2 else mono
    if len(mono) < sr or not np.abs(mono).max() > 1e-5:
        return None
    os.makedirs(CACHE, exist_ok=True)
    pfad = os.path.join(CACHE, _schluessel(mono) + ".npz")
    if os.path.isfile(pfad):
        try:
            z = np.load(pfad)
            return {k: z[k] for k in z.files}
        except (OSError, ValueError):
            pass
    app = _app_ordner()
    if app is None:
        return None
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        wav, helfer, out = (os.path.join(tmp, n) for n in ("spur.wav", "hoeren.py", "out.npz"))
        sf.write(wav, mono, sr, subtype="PCM_16")
        with open(helfer, "w") as f:
            f.write(_HELFER)
        for py in (DETECT_PYTHON, sys.executable):
            if not os.path.exists(py):
                continue
            try:
                r = subprocess.run([py, helfer, app, wav, out, json.dumps(GRUPPEN), str(FENSTER_S), str(SCHRITT_S)],
                                   capture_output=True, text=True, timeout=1800)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0 and os.path.isfile(out):
                z = np.load(out)
                d = {k: z[k] for k in z.files}
                try:
                    np.savez(pfad, **d)
                except OSError:
                    pass
                return d
    return None


def verlauf(audio, sr, gruppe):
    """(t, Wahrscheinlichkeit) einer Gruppe – oder None."""
    z = zeitleiste(audio, sr)
    if z is None or gruppe not in z:
        return None
    return z["t"], z[gruppe]


def tor(y, ref, sr, gruppe, bericht=None, min_pause_s=4.0, blende_s=0.3):
    """Spielt die neue Spur nur dort, wo das Hör-Modell das Instrument in der Original-Spur (ref) hört.
    Stumm geschaltet werden nur lange, sichere Pausen (≥ min_pause_s), in denen das Modell das Instrument klar NICHT
    hört UND die Original-Spur mindestens 15 dB leiser ist als beim echten Spielen – dort spielt die Reparatur sonst
    nur Reste der Trennung nach. Sicherungen:
      * Hört das Modell das Instrument im ganzen Song kaum (unsicher für dieses Instrument), bleibt alles wie es ist.
      * Würde mehr als 40 % der gespielten Zeit wegfallen, bleibt alles wie es ist (Modell vermutlich daneben)."""
    info = {"gruppe": gruppe}
    try:
        from scipy.ndimage import uniform_filter1d
        v = verlauf(ref, sr, gruppe)
        if v is None:
            info["stand"] = "Hör-Modell hier nicht verfügbar"
            return y
        t, p = v
        p = uniform_filter1d(p.astype(float), max(1, int(round(2.0 / SCHRITT_S))))
        p90 = float(np.percentile(p, 90))
        info["sicherheit_p90"] = round(p90, 3)
        if p90 < 0.08:
            info["stand"] = "Hör-Modell unsicher bei diesem Instrument – nichts geändert"
            return y
        weg = p < max(0.03, 0.15 * p90)
        # Sicherung über die Lautstärke: nur wo die Original-Spur dort auch deutlich leiser ist als beim echten Spielen
        # (Reste der Trennung sind leise). Ist sie normal laut, irrt das Modell (z. B. Reggae-Skank) -> nicht anfassen.
        rm = np.asarray(ref, dtype=np.float32)
        rm = rm.mean(1) if rm.ndim == 2 else rm
        hw = int(SCHRITT_S * sr)
        lv = np.array([20 * np.log10(np.sqrt(np.mean(rm[max(0, int(tt * sr) - hw):int(tt * sr) + hw] ** 2)) + 1e-9)
                       for tt in t])
        oben = lv[lv > lv.max() - 40]
        aktiv_db = float(np.median(oben)) if len(oben) else float(lv.max())
        info["pegel_spielt_db"] = round(aktiv_db, 1)
        weg &= lv < aktiv_db - 15
        # nur lange Pausen
        lauf, start = [], None
        for i, w in enumerate(np.r_[weg, False]):
            if w and start is None:
                start = i
            elif not w and start is not None:
                if (i - start) * SCHRITT_S >= min_pause_s:
                    lauf.append((start, i))
                start = None
        if not lauf:
            info["stand"] = "Instrument überall zu hören – nichts geändert"
            return y
        yy = np.asarray(y, dtype=np.float32)
        n = len(yy)
        g = np.ones(n, np.float32)
        for a, b in lauf:
            s0 = int(max(0.0, t[a] - SCHRITT_S / 2) * sr); s1 = int(min(n / sr, t[min(b, len(t) - 1)] + SCHRITT_S / 2) * sr)
            g[s0:s1] = 0.0
        f = max(1, int(blende_s * sr))
        g = uniform_filter1d(g, f)                     # weiche Ein-/Ausblende
        e = np.abs(yy).mean(1) if yy.ndim == 2 else np.abs(yy)
        spielt = e > 0.01 * (e.max() + 1e-12)
        anteil = float(np.mean(g[spielt] < 0.5)) if spielt.any() else 0.0
        info["pausen"] = [[round(float(t[a]), 1), round(float(t[min(b, len(t) - 1)]), 1)] for a, b in lauf][:40]
        info["stumm_anteil_gespielt"] = round(anteil, 3)
        if anteil > 0.40:
            info["stand"] = "würde zu viel wegnehmen – nichts geändert"
            return y
        info["stand"] = f"{len(lauf)} Pausen ohne {gruppe} stumm (Reste der Trennung)"
        return (yy * (g[:, None] if yy.ndim == 2 else g)).astype(yy.dtype)
    except Exception as e:  # noqa: BLE001 – nie am Ergebnis scheitern
        info["stand"] = f"übersprungen ({str(e)[:120]})"
        return y
    finally:
        if bericht is not None:
            bericht.setdefault("hoer_modell", {})[gruppe] = info
