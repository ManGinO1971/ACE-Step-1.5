"""Hör-Modell für alle Reparatur-Module (14. Nov 2026).

Das Hör-Modell der Instrumenten-Erkennung (EfficientAT, AudioSet) hört eine Spur alle 0,5 s ab und sagt je
Instrument-Gruppe, wie sicher es klingt. Das Ergebnis wird je Spur zwischengespeichert: dieselbe Spur wird pro Song
nur EINMAL abgehört, jede weitere Reparatur liest nur nach (kostet dann praktisch keine Zeit).

Benutzung in einem Modul:
    import hoer_tor as ht
    y = ht.tor(y, stem, SR, "bass", bericht)        # spielt nur dort, wo das Instrument im Original zu hören ist
    t, p = ht.verlauf(stem, SR, "hihat")            # Verlauf einer Gruppe (z. B. zum Ergänzen)

    y = ht.pegel(y, stem, song, SR, bericht)      # 17. Nov: Lautstärke-Sicherung gegenüber dem Original-Song

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
    for d in (HIER, os.path.dirname(HIER), os.path.join(os.path.dirname(HIER), "hieltech_erkennung"),
              os.path.expanduser("~/HELTECH_Stems")):
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


def tor(y, ref, sr, gruppe, bericht=None, min_pause_s=4.0, blende_s=0.3, song=None):
    """Spielt die neue Spur nur dort, wo das Hör-Modell das Instrument in der Original-Spur (ref) hört.
    Stumm geschaltet werden nur lange, sichere Pausen (≥ min_pause_s), in denen das Modell das Instrument klar NICHT
    hört UND die Original-Spur mindestens 15 dB leiser ist als beim echten Spielen – dort spielt die Reparatur sonst
    nur Reste der Trennung nach. Sicherungen:
      * Hört das Modell das Instrument im ganzen Song kaum (unsicher für dieses Instrument), bleibt alles wie es ist.
      * Würde mehr als 40 % der gespielten Zeit wegfallen, bleibt alles wie es ist (Modell vermutlich daneben).
      * 17. Nov 2026 (song = ganzer Original-Song): stumm wird nur, wo das Instrument AUCH im ganzen Song nicht zu
        hören ist. Hat die Trennung es nur aufgefressen (im Song zu hören, in der Spur nicht), bleibt die Stelle."""
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
        if song is not None:
            ls = luecken(ref, song, sr, gruppe)
            if ls is not None:
                im_song = np.interp(t, ls["t"], ls["song"].astype(float)) > 0.5
                info["aufgefressen_s"] = ls["info"]["aufgefressen_s"]
                weg &= ~im_song
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


def _mono(x, sr):
    """Pfad oder Array -> mono float32 (Pfad wird mit sr geladen)."""
    if x is None:
        return None
    if isinstance(x, str):
        if not os.path.isfile(x):
            return None
        import librosa
        x, _ = librosa.load(x, sr=sr, mono=True)
    x = np.asarray(x, dtype=np.float32)
    return x.mean(1) if x.ndim == 2 else x


def pegel(y, ref, song, sr, bericht=None, versatz_db=0.0, toleranz_db=6.0, ersatz_unter_song_db=16.0):
    """17. Nov 2026 – Lautstärke-Sicherung für alle Reparaturen: die neue Spur soll so laut sein wie das Instrument
    im Original-Song – nie unhörbar leise, nie lauter als im Original.
    Gemessen wird nur dort, wo die neue Spur spielt (0,5-s-Abschnitte, höchstens 30 dB unter ihrer lautesten Stelle).
      * ref = getrennte Original-Spur des Instruments (None = keine brauchbare), song = ganzer Original-Song.
      * Ziel = Pegel der Original-Spur (+ versatz_db, z. B. Picking 8 dB unter der Gitarre).
      * Hat die Trennung das Instrument verloren (Original-Spur mehr als 35 dB unter dem Song), gilt als Ziel
        „ersatz_unter_song_db“ unter dem Song.
      * Die Module stellen die Lautstärke selbst ein; geändert wird nur, wenn sie mehr als toleranz_db vom Ziel
        abweicht (also klar falsch ist) oder lauter als das Original wäre (3 dB unter dem ganzen Song bzw. die Original-
        Spur +1 dB). Spitzen ≤ 0,98."""
    info = {}
    try:
        yy = np.asarray(y, dtype=np.float32)
        ym = yy.mean(1) if yy.ndim == 2 else yy
        r = _mono(ref, sr)
        s = _mono(song, sr)
        if s is None:
            s = r
        if s is None or len(ym) < sr:
            info["stand"] = "ohne Vergleich – nichts geändert"
            return y
        ohne_ref = r is None
        if ohne_ref:                    # keine brauchbare Original-Spur: Ziel nur nach dem Song
            r = np.zeros_like(s)
        n = min(len(ym), len(r), len(s))
        h = int(0.5 * sr)
        k = n // h
        if k < 2:
            info["stand"] = "zu kurz – nichts geändert"
            return y

        def db(x):
            e = np.sqrt(np.mean(x[:k * h].reshape(k, h).astype(np.float64) ** 2, 1)) + 1e-12
            return e
        ey, er, es = db(ym), db(r), db(s)
        if ey.max() < 1e-7:
            info["stand"] = "neue Spur leer – nichts geändert"
            return y
        akt = ey > ey.max() * 10 ** (-30 / 20)

        def lv(e):
            return float(20 * np.log10(np.sqrt(np.mean(e[akt] ** 2)) + 1e-12))
        L_y, L_r, L_s = lv(ey), lv(er), lv(es)
        info.update(neu_db=round(L_y, 1), original_spur_db=None if ohne_ref else round(L_r, 1), song_db=round(L_s, 1))
        if not ohne_ref and L_r >= L_s - 35:
            ziel, art = L_r + versatz_db, "wie die Original-Spur"
        else:
            ziel, art = L_s - ersatz_unter_song_db + versatz_db, "Original-Spur fast leer (Trennung) – nach dem Song"
        # nie lauter als das Original: höchstens 3 dB unter dem ganzen Song (ist „song“ selbst nur die Spur, zählt
        # die Spur +1 dB)
        deckel = max(L_s - 3.0, (L_r + 1.0) if not ohne_ref else -999.0)
        ziel = min(ziel, deckel)
        info["ziel_db"] = round(ziel, 1)
        g_db = 0.0
        if abs(L_y - ziel) > toleranz_db or L_y > deckel:
            g_db = ziel - L_y
            yy = yy * np.float32(10 ** (g_db / 20))
        sp = float(np.abs(yy).max())
        if sp > 0.98:
            yy = yy * np.float32(0.98 / sp)
        info["angepasst_db"] = round(g_db, 1)
        info["stand"] = (f"angepasst um {g_db:+.1f} dB ({art})" if g_db else f"passt ({art})")
        return yy.astype(np.asarray(y).dtype) if g_db or sp > 0.98 else y
    except Exception as e:  # noqa: BLE001 – nie am Ergebnis scheitern
        info["stand"] = f"übersprungen ({str(e)[:120]})"
        return y
    finally:
        if bericht is not None:
            bericht.setdefault("hoer_modell", {})["pegel"] = info


SONG_SCHWELLE = 0.08        # Sicherheit (über 2 s geglättet), ab der ein Instrument im ganzen Song als „zu hören“ gilt


def _putzen(b, schritt, min_s=2.0):
    """Lücken < min_s zwischen zwei True-Stellen schließen, True-Stücke < min_s weglassen."""
    b = np.asarray(b, bool).copy()
    n = max(1, int(round(min_s / schritt)))
    for wert in (False, True):
        i = 0
        while i < len(b):
            if b[i] != wert:
                i += 1
                continue
            j = i
            while j < len(b) and b[j] == wert:
                j += 1
            if j - i < n and (wert or (0 < i and j < len(b))):
                b[i:j] = not wert
            i = j
    return b


def luecken(stem, song, sr, gruppe, schwelle_spur=0.10, schwelle_song=SONG_SCHWELLE):
    """17. Nov 2026 – das Hör-Modell hört für EIN Instrument zwei Dinge ab: die getrennte Spur (stem) und den ganzen
    Original-Song (song). Daraus weiß jedes Modul, was die Trennung aufgefressen hat:
        spur  = Instrument in seiner Spur zu hören       -> reparieren
        song  = Instrument im ganzen Song zu hören
        fehlt = im Song zu hören, aber nicht in der Spur  -> ergänzen
    Beide Verläufe werden zwischengespeichert (pro Song nur einmal abgehört). Rückgabe dict oder None (Modell fehlt)."""
    try:
        from scipy.ndimage import uniform_filter1d
        st_ = _mono(stem, sr)
        so_ = _mono(song, sr)
        if st_ is None or so_ is None:
            return None
        vs = verlauf(st_, sr, gruppe)
        vg = verlauf(so_, sr, gruppe)
        if vs is None or vg is None:
            return None
        t = vg[0]
        hs = float(np.median(np.diff(t))) if len(t) > 1 else SCHRITT_S
        k = max(1, int(round(2.0 / hs)))
        pg = uniform_filter1d(vg[1].astype(float), k)
        ps = np.interp(t, vs[0], uniform_filter1d(vs[1].astype(float), k))
        spur = _putzen(ps >= schwelle_spur, hs)
        im_song = _putzen(pg >= schwelle_song, hs) | spur
        fehlt = im_song & ~spur
        return {"t": t, "spur": spur, "song": im_song, "fehlt": fehlt, "schritt": hs,
                "info": {"spur_s": round(float(spur.sum() * hs), 1), "song_s": round(float(im_song.sum() * hs), 1),
                         "aufgefressen_s": round(float(fehlt.sum() * hs), 1),
                         "sicherheit_song_p90": round(float(np.percentile(pg, 90)), 3)}}
    except Exception:  # noqa: BLE001
        return None


def anteil(ls, schluessel, a, b):
    """Anteil der Zeit [a, b] (Sekunden), in der ls[schluessel] gilt (0…1)."""
    if ls is None:
        return 0.0
    t, w = ls["t"], ls[schluessel]
    m = (t >= a) & (t <= b)
    if not m.any():
        i = int(np.clip(np.searchsorted(t, (a + b) / 2), 0, len(t) - 1))
        return float(w[i])
    return float(np.mean(w[m]))


def takte_ergaenzen(aktiv, takte, beats, ls, max_takte=8, schluessel="song"):
    """17. Nov 2026: Skank-Takte, die die Trennung aufgefressen hat, wieder einschalten: Lücken (höchstens max_takte
    Takte) ZWISCHEN zwei gespielten Takten, in denen das Hör-Modell das Instrument im ganzen Song hört (≥ 50 %).
    Anfang/Ende/Breaks bleiben, wie sie sind. Rückgabe: neues aktiv, Anzahl ergänzter Takte."""
    aktiv = np.asarray(aktiv, bool).copy()
    if ls is None or not aktiv.any():
        return aktiv, 0
    n = len(beats) - 1
    idx = np.where(aktiv)[0]
    neu = 0
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_takte + 1:
            for k in range(a + 1, b):
                t0 = beats[takte[k]]
                t1 = beats[min(takte[k] + 4, n)]
                if anteil(ls, schluessel, t0, t1) >= 0.5:
                    aktiv[k] = True
                    neu += 1
    return aktiv, neu


def ergaenzer(stem, song, sr, gruppe, bericht=None, max_takte=8):
    """17. Nov 2026: Für Skank-Module (Gitarre, Piano …): liefert eine Funktion f(aktiv, takte, beats) -> aktiv, die
    aufgefressene Takte wieder einschaltet (siehe takte_ergaenzen), und schreibt ins Bericht, wie viele es waren.
    Ohne Modell / ohne Song: None (dann bleibt alles wie bisher)."""
    if song is None:
        return None
    ls = luecken(stem, song, sr, gruppe)
    if ls is None:
        return None

    def f(aktiv, takte, beats):
        neu, k = takte_ergaenzen(aktiv, takte, beats, ls, max_takte)
        if bericht is not None:
            d = bericht.setdefault("hoer_modell", {}).setdefault("ergaenzt", {})
            d[gruppe] = {"takte": int(k), **ls["info"]}
        return neu
    return f
