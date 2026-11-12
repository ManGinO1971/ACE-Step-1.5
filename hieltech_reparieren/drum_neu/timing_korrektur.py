"""
HELTECH Stems – Timing korrigieren (für jede Spur, ohne Bezugsspur). Stand 5. Nov 2026.

Aus den Anschlägen der Spur selbst wird ein lokales Taktraster bestimmt (Tempo darf leicht schwanken):
je Anschlag die Raster-Periode (Schlag/Teilung, ±4 % um das Grundtempo) und Phase, die am besten zu den
Anschlägen im Umkreis von ±4 s passen. Jeder Anschlag wird dann auf den nächsten Rasterpunkt geschoben,
wenn er höchstens max_ms (bzw. 40 % des Rasterabstands) daneben liegt. Gerade (16tel) oder Shuffle
(Triolen) wird automatisch gewählt.

Wichtig: ohne Bezugsspur wird nur das Wackeln entfernt. Liegt die ganze Spur gleichmäßig vor/hinter dem
Beat (z. B. Bass immer 20 ms hinter dem Schlagzeug), bleibt dieser feste Versatz erhalten.
"""
import numpy as np

TEILUNGEN = {4: "gerade (16tel)", 3: "Shuffle (Triolen)", 6: "Shuffle (16tel-Triolen)"}


def tempo_schaetzen(m, sr):
    import librosa
    oenv = librosa.onset.onset_strength(y=m, sr=sr, hop_length=512)
    t = float(librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=512, start_bpm=90)[0])
    while t > 140:
        t /= 2
    while t < 60:
        t *= 2
    return t


def _ziele(t, w, P0, fenster):
    ziel = np.zeros_like(t)
    per = np.zeros_like(t)
    stufen = P0 * np.linspace(0.96, 1.04, 41)
    for i, ti in enumerate(t):
        sel = np.abs(t - ti) <= fenster
        tt, ww = t[sel], w[sel]
        z = np.array([np.sum(ww * np.exp(2j * np.pi * tt / P)) for P in stufen])
        k = int(np.argmax(np.abs(z)))
        P = stufen[k]
        phi = np.angle(z[k]) / (2 * np.pi) * P
        ziel[i] = phi + np.round((ti - phi) / P) * P
        per[i] = P
    return ziel, per


def raster_schub(t, bpm, w=None, fenster=4.0, max_ms=60.0, teilung=None):
    """t: Anschläge (s, aufsteigend), w: Gewicht je Anschlag (z. B. Lautstärke linear).
    Rückgabe: (schub in s je Anschlag, Bericht)."""
    t = np.asarray(t, float)
    if len(t) < 8 or not bpm:
        return np.zeros(len(t)), {"hinweis": "zu wenige Anschläge für ein Raster"}
    w = np.ones_like(t) if w is None else np.asarray(w, float)
    kand = [teilung] if teilung else [4, 3, 6]
    erg = {}
    for k in kand:
        P0 = 60.0 / bpm / k
        ziel, per = _ziele(t, w, P0, fenster)
        d = ziel - t
        # Güte: mittlere Abweichung relativ zum Rasterabstand (Zufall ≈ 0,25); feinere Raster passen
        # immer etwas besser, darum Strafe je Rasterpunkte
        erg[k] = (float(np.mean(np.abs(d) / per)) * (k / 4) ** 0.5, d, per)
    beste = 4 if 4 in erg else kand[0]
    for k in erg:
        if erg[k][0] < erg[beste][0] * 0.85:
            beste = k
    guete, d, per = erg[beste]
    grenze = np.minimum(max_ms / 1000.0, 0.4 * per)
    ok = np.abs(d) <= grenze
    schub = np.where(ok, d, 0.0)
    return schub, {"tempo_bpm": round(float(bpm), 2), "raster": TEILUNGEN.get(beste, str(beste)),
                   "verschoben": int(ok.sum()), "unveraendert": int((~ok).sum()),
                   "abweichung_vorher_ms_median": round(float(np.median(np.abs(d[ok])) * 1000), 1) if ok.any() else 0.0,
                   "abweichung_vorher_ms_90": round(float(np.percentile(np.abs(d[ok]), 90) * 1000), 1) if ok.any() else 0.0,
                   "guete": {TEILUNGEN.get(k, str(k)): round(v[0], 3) for k, v in erg.items()}}
