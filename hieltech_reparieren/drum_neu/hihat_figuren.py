"""hihat_figuren.py – Katalog aller Reggae-Hi-Hat-Spielarten (25. Nov 2026, Plan mit dem Inhaber).

Grundlage für das Hi-Hat-Hörmodell: Statt aus jedem Song neu zu raten, WIE die Hi-Hat gespielt wird, vergleicht das
Hörmodell jeden Abschnitt des Songs mit diesem Katalog (kurze Hörproben je Figur) und spielt dann genau die erkannte
Figur mit den vom Inhaber ausgewählten Klängen nach (samples/sfs/hihat_plan_auswahl.json). Lautstärke je Stelle und
Timing (Swing) werden weiterhin im Song gemessen und auf die Figur gelegt – das Tempo kommt immer aus dem Song.

Schreibweise: eine Figur ist EIN Takt in Sechzehnteln (16 Stellen, 4 je Zählzeit; bei Triolen 12 Stellen).
Jede Stelle ist None (nichts) oder (Spielweise, Stärke 0…1):
    A = zu geschlossen          B = halb offen, vom Fuß gestoppt     C = halb offen klingend
    D = offen ausklingend       E = nur Fuß (Pedal)
„swing“ = wie spät die schwachen Sechzehntel kommen (Anteil eines Sechzehntels; 0 = gerade).
„bis_naechster“ = offene/halboffene Schläge werden vom Fuß beim nächsten Hi-Hat-Schlag abgestoppt.

Aufruf zum Anhören:  python3 hihat_figuren.py AUSGABE_ORDNER [BPM]   -> eine kurze Hörprobe je Figur (2 Takte).
"""
import json
import os
import sys

import numpy as np

SR = 44100
HIER = os.path.dirname(os.path.abspath(__file__))


def _takt(je_zaehlzeit, akzente=None):
    """Hilfe: dasselbe Muster auf alle 4 Zählzeiten, optional einzelne Stellen (0…15) überschreiben."""
    t = list(je_zaehlzeit) * 4
    for k, v in (akzente or {}).items():
        t[k] = v
    return t


A, B, C, D, E = "A", "B", "C", "D", "E"

FIGUREN = {
    # ---- 10 Grundfiguren -------------------------------------------------------------------------------------
    "01_gerade_achtel": dict(
        name="Gerade Achtel, zu", stil="Ska, Rocksteady, Lovers Rock", swing=0.0,
        takt=_takt([(A, 0.9), None, (A, 0.6), None])),
    "02_nur_offbeat": dict(
        name="Nur Offbeat (Chick auf dem Und)", stil="Ska, frühes Rocksteady", swing=0.0,
        takt=_takt([(E, 0.5), None, (A, 1.0), None])),
    "03_viertel": dict(
        name="Viertel", stil="ruhige Roots-Stücke, Balladen", swing=0.0,
        takt=_takt([(A, 0.9), None, None, None])),
    "04_geswingte_sechzehntel": dict(
        name="Geswingte Sechzehntel, 4 verschieden laut", stil="One Drop, Roots (70er bis heute)", swing=0.55,
        takt=_takt([(A, 1.0), (A, 0.35), (A, 0.7), (A, 0.45)])),
    "05_zu_halboffen_zu_halboffen": dict(
        name="Zu – halb offen (Fuß) – zu – halb offen (Fuß)", stil="Roots, One Drop, Rockers", swing=0.55,
        takt=_takt([(A, 1.0), (B, 0.45), (A, 0.7), (B, 0.55)]), bis_naechster=True),
    "06_flying_cymbal": dict(
        name="Flying Cymbal (offen auf jedem Und, Fuß schließt)", stil="Mitte 70er (Bunny Lee)", swing=0.0,
        takt=_takt([(E, 0.6), None, (D, 1.0), None]), bis_naechster=True),
    "07_rockers_militant": dict(
        name="Rockers / Militant (dichte harte Sechzehntel, Akzent offen)", stil="Sly Dunbar, späte 70er", swing=0.25,
        takt=_takt([(A, 1.0), (A, 0.55), (A, 0.85), (A, 0.55)], {14: (C, 0.9)}), bis_naechster=True),
    "08_steppers": dict(
        name="Steppers (gerade Sechzehntel über durchgehender Kick)", stil="UK Roots, Dub, Steppers", swing=0.0,
        takt=_takt([(A, 0.9), (A, 0.55), (A, 0.8), (A, 0.55)])),
    "09_rub_a_dub": dict(
        name="Rub-a-Dub / Early Dancehall (sparsam, Offbeat-Akzent)", stil="80er", swing=0.15,
        takt=_takt([(A, 0.5), None, (A, 1.0), None], {6: (C, 0.8), 14: (C, 0.8)}), bis_naechster=True),
    "10_digital_dancehall": dict(
        name="Digital / Dancehall (gerade, gleich laut, offen als Effekt)", stil="ab 1985 bis heute", swing=0.0,
        takt=_takt([(A, 0.75), (A, 0.75), (A, 0.75), (A, 0.75)], {14: (D, 0.8)}), bis_naechster=True),
    # ---- 3 Sonderfälle -------------------------------------------------------------------------------------
    "S1_triolen_shuffle": dict(
        name="Triolen / Shuffle", stil="frühes Rocksteady, Nyabinghi-beeinflusst", swing=0.0, triolen=True,
        takt=[(A, 1.0), None, (A, 0.6)] * 4),
    "S2_nur_fuss": dict(
        name="Nur Fuß-Hi-Hat", stil="sehr sparsame Roots-Songs", swing=0.0,
        takt=_takt([None, None, (E, 0.8), None])),
    "S3_keine_hihat": dict(
        name="Keine Hi-Hat (Tamburin/Shaker übernimmt)", stil="Nyabinghi, manche Roots-Stücke", swing=0.0,
        takt=[None] * 16),
}

# Nur zum Anhören (25. Nov, Inhaber): Kick je Figur, damit man den Takt erkennt (Zählzeit 1…4 = Stelle 0, 4, 8, 12).
KICK_ZUM_HOEREN = {
    "01_gerade_achtel": [0, 8], "02_nur_offbeat": [0, 8], "03_viertel": [8], "04_geswingte_sechzehntel": [8],
    "05_zu_halboffen_zu_halboffen": [8], "06_flying_cymbal": [0, 8], "07_rockers_militant": [0, 7, 8],
    "08_steppers": [0, 4, 8, 12], "09_rub_a_dub": [0, 8], "10_digital_dancehall": [0, 6, 8, 14],
    "S1_triolen_shuffle": [6], "S2_nur_fuss": [8], "S3_keine_hihat": [8],
}

# Feste Klänge + Reihenfolge je Stil (vom Inhaber bestätigt). Rang = Lautstärke-Rang der gespielten Stellen eines
# Schlags im Song (1 = lauteste Stelle = Hauptschlag). Die Liste wird der Reihe nach abgespielt und wiederholt.
# 26. Nov 2026, Rise Up (Figur 04): „die Teile und die Reihenfolge ist perfekt – für den Style so speichern“.
FESTE_KLAENGE = {
    "04_geswingte_sechzehntel": {
        1: ["Nr0426-287"],                               # Hauptschlag: immer Nr. 1
        2: ["Nr0426-185", "Nr0426-143"],                 # zweitlauteste Stelle: Nr. 2, Nr. 3, Nr. 2, Nr. 3 …
        3: ["Nr0426-143", "Nr0426-185", "Nr0426-287"],   # drittlauteste: Nr. 3, Nr. 2, Nr. 1, …
        4: ["Nr0426-185", "Nr0426-143"],                 # (Rise Up spielt nur 3 Stellen – 4. wie Rang 2)
    },
}

# Die 3 Merkmale, die jede Figur zusätzlich ausmachen (werden im Song gemessen, nicht aus dem Katalog genommen):
MERKMALE = {
    "spielweise_je_stelle": "zu / halb offen vom Fuß gestoppt / halb offen klingend / offen / nur Fuß",
    "lautstaerke_je_stelle": "die Stellen eines Schlags sind bei Reggae fast immer verschieden laut (Groove)",
    "timing_je_stelle": "Swing: wie spät die schwachen Sechzehntel kommen",
}


def klaenge(samples_dir=None):
    """Die vom Inhaber ausgewählten Klänge je Spielweise (A…E), sortiert wie in der Liste (leise → laut)."""
    sd = samples_dir or os.path.join(HIER, "samples", "sfs")
    j = json.load(open(os.path.join(sd, "hihat_plan_auswahl.json"), encoding="utf-8"))
    import soundfile as sf
    aus = {}
    for g, liste in j["gruppen"].items():
        art = g[0]
        aus[art] = []
        for p in liste:
            x, sr = sf.read(os.path.join(sd, p), dtype="float32", always_2d=True)
            x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, 1)
            i = int(np.argmax(np.abs(x).max(1) > np.abs(x).max() * 0.1))
            aus[art].append(x[max(0, i - 20):])
    return aus


def spielen(fig, bpm, takte, K, start_s=0.2, rng=None):
    """Figur als Audio (stereo) – für die Hörproben des Katalogs."""
    rng = rng or np.random.default_rng(0)
    f = FIGUREN[fig]
    st = 3 if f.get("triolen") else 4
    schlag = 60.0 / bpm
    sech = schlag / st
    n_st = len(f["takt"])
    y = np.zeros((int(SR * (start_s + takte * 4 * schlag + 3.0)), 2), np.float32)
    zeiten = []
    for t in range(takte):
        for k, v in enumerate(f["takt"]):
            if v is None:
                continue
            spaet = f["swing"] * sech if (k % st) % 2 == 1 and st == 4 else 0.0
            zeiten.append((start_s + (t * n_st + k) * sech + spaet, v))
    for j, (z, (art, st_)) in enumerate(zeiten):
        liste = K.get(art) or K["A"]
        # Stärke wählt den Klang (leise … laut) – mit kleiner Abwechslung, damit es nicht wie eine Maschine klingt
        idx = int(np.clip(round(st_ * (len(liste) - 1) + rng.integers(-1, 2)), 0, len(liste) - 1))
        x = liste[idx].copy()
        if f.get("bis_naechster") and art in "BCD" and j + 1 < len(zeiten):
            L = int((zeiten[j + 1][0] - z) * SR)           # der Fuß schließt beim nächsten Hi-Hat-Schlag
            if 0 < L < len(x):
                fo = min(int(0.025 * SR), L)
                x = x[:L]
                x[-fo:] *= np.linspace(1, 0, fo)[:, None]
        a = int(z * SR)
        g = 0.25 + 0.75 * st_
        y[a:a + len(x)] += g * x[:len(y) - a]
    return y * (0.9 / (np.abs(y).max() + 1e-9)) if np.abs(y).max() > 0 else y


if __name__ == "__main__":
    import soundfile as sf
    ziel = sys.argv[1]
    bpm = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 75.0
    os.makedirs(ziel, exist_ok=True)
    K = klaenge()
    mit_kick = "--kick" in sys.argv
    if mit_kick:
        kx, _ = sf.read(os.path.join(HIER, "samples", "sfs", "Kick", "1_Kurz_trocken", "Kick_kurz_mittel_55Hz_Nr0458-79.flac"),
                        dtype="float32", always_2d=True)
        kx = kx[:, :2] if kx.shape[1] >= 2 else np.repeat(kx, 2, 1)
    for fig, f in FIGUREN.items():
        y = spielen(fig, bpm, 2, K)
        if mit_kick:
            st = 3 if f.get("triolen") else 4
            for t in range(2):
                for k in KICK_ZUM_HOEREN.get(fig, []):
                    a = int((0.2 + (t * len(f["takt"]) + k) * 60.0 / bpm / st) * SR)
                    y[a:a + len(kx)] += 3.0 * kx[:len(y) - a] * (0.9 / (np.abs(kx).max() + 1e-9))
            y *= 0.9 / (np.abs(y).max() + 1e-9)
        sf.write(os.path.join(ziel, f"{fig}.wav"), y, SR)
        print(f"{fig:32s} {f['name']}  ({f['stil']})")


# =====================================================================================================================
# Schritt 3 (25. Nov 2026): Hörmodell vergleicht den Song mit dem Katalog
# =====================================================================================================================
def _hf_db(x):
    import scipy.signal as sg
    sos = sg.butter(4, 6000, 'high', fs=SR, output='sos')
    y = np.abs(sg.sosfiltfilt(sos, x))
    h = int(.005 * SR)
    return 20 * np.log10(np.sqrt(np.convolve(y ** 2, np.ones(h) / h, 'same')) + 1e-9)


def stellen_messen(hh, beats, st=4):
    """Je Schlag und Stelle: Spitze (dB), Ausklang bis −15 dB (ms), Lage gegen das Raster (ms)."""
    E = _hf_db(hh)
    P = np.full((len(beats) - 1, st), -120.0); A = np.zeros_like(P); L = np.zeros_like(P)
    for i in range(len(beats) - 1):
        for k in range(st):
            s = int(beats[i] + (beats[i + 1] - beats[i]) * k / st)
            a = max(0, s - int(.04 * SR)); w = E[a:s + int(.05 * SR)]
            if not len(w):
                continue
            j = a + int(np.argmax(w)); p = E[j]
            seg = E[j:j + int(.5 * SR)]; u = np.where(seg < p - 15)[0]
            P[i, k] = p; A[i, k] = (u[0] if len(u) else len(seg)) / SR * 1000; L[i, k] = (j - s) / SR * 1000
    return P, A, L


def _vorlage(fig):
    """Je Stelle eines Schlags: (gespielt, Stärke, offen?) – Mittel über die 4 Zählzeiten der Figur."""
    f = FIGUREN[fig]; st = 3 if f.get("triolen") else 4
    T = np.array([[0.0, 0.0, 0.0]] * st)
    for k, v in enumerate(f["takt"]):
        if v is not None:
            T[k % st] += [1.0, v[1], 1.0 if v[0] in "BCD" else 0.0]
    T /= 4.0
    return T, f["swing"], st


def erkennen(song, hh, abschnitt_schlaege=8, melde=print):
    """Welche Katalog-Figur spielt die Hi-Hat in jedem Abschnitt? Rückgabe: beats, Liste je Abschnitt
    (start_schlag, figur, sicherheit, punkte) und die Messung je Stelle."""
    import librosa
    n = min(len(song), len(hh))
    _, beats = librosa.beat.beat_track(y=song[:n], sr=SR, units='samples', start_bpm=85)
    beats = np.array(beats)
    P, A, L = stellen_messen(hh[:n], beats)
    gmax = np.percentile(P, 98)
    ergebnis = []
    for b0 in range(0, len(beats) - 1, abschnitt_schlaege):
        sl = slice(b0, min(b0 + abschnitt_schlaege, len(beats) - 1))
        p = np.median(P[sl], 0) - gmax
        if p.max() < -30:                                   # hier spielt keine Hi-Hat
            ergebnis.append((b0, "S3_keine_hihat", 1.0, {})); continue
        da = (p > p.max() - 24).astype(float)               # gespielt?
        st_rel = np.clip(1 + (p - p.max()) / 24, 0, 1) * da    # Stärke 0…1
        a = np.median(A[sl], 0)
        ref = np.min(a[da > 0]) if (da > 0).any() else 1.0
        offen = ((a / max(ref, 1.0)) > 1.4).astype(float) * da
        sw = float(np.median(L[sl][:, 1::2]) - np.median(L[sl][:, 0::2])) if da[1::2].any() else 0.0
        sech = float(np.median(np.diff(beats[sl.start:sl.stop + 1]))) / 4 / SR * 1000 if sl.stop - sl.start > 1 else 50
        punkte = {}
        for fig in FIGUREN:
            if fig == "S3_keine_hihat":
                continue
            T, swing, st = _vorlage(fig)
            if st != 4:
                continue                                    # Triolen: eigene Prüfung (folgt)
            k_da = np.abs((T[:, 0] > 0.2) - da).sum()
            k_st = np.abs(np.where(T[:, 0] > 0, T[:, 1] / max(T[:, 1].max(), 1e-9), 0) - st_rel).sum()
            k_of = np.abs((T[:, 2] > 0.1) - offen).sum()
            k_sw = abs(swing * sech - sw) / max(sech, 1.0)
            punkte[fig] = round(float(2.0 * k_da + 1.0 * k_st + 1.5 * k_of + 1.0 * k_sw), 3)
        best = sorted(punkte, key=punkte.get)
        sicher = float(np.clip((punkte[best[1]] - punkte[best[0]]) / 1.0, 0, 1)) if len(best) > 1 else 1.0
        ergebnis.append((b0, best[0], round(sicher, 2), dict(sorted(punkte.items(), key=lambda kv: kv[1])[:3])))
    return beats, ergebnis, dict(P=P, A=A, L=L, gmax=gmax)


def glaetten(ergebnis):
    """Einzelne unsichere Abschnitte zwischen zwei gleichen Nachbarn bekommen deren Figur."""
    e = list(ergebnis)
    import collections
    sicher = collections.Counter(x[1] for x in e if x[2] >= 0.5 and x[1] != "S3_keine_hihat")
    haupt = sicher.most_common(1)[0][0] if sicher else None
    for i in range(len(e)):                                 # sehr unsicher: die Haupt-Figur des Songs
        if haupt and e[i][2] < 0.3 and e[i][1] != "S3_keine_hihat":
            e[i] = (e[i][0], haupt, e[i][2], e[i][3])
    for i in range(1, len(e) - 1):
        if e[i - 1][1] == e[i + 1][1] != e[i][1] and e[i][2] < 0.5:
            e[i] = (e[i][0], e[i - 1][1], e[i][2], e[i][3])
    return e


def nachbauen(beats, ergebnis, mess, K, n, abschnitt_schlaege=8, rng=None):
    """Hi-Hat nach den erkannten Figuren spielen – Tempo = die Schläge des Songs, Lautstärke und Swing je Stelle
    aus dem Song gemessen, Spielweise aus der Figur."""
    rng = rng or np.random.default_rng(1)
    P, L, gmax = mess["P"], mess["L"], mess["gmax"]
    zeiten = []
    for b0, fig, _, _ in ergebnis:
        f = FIGUREN[fig]
        if fig == "S3_keine_hihat":
            continue
        st = 3 if f.get("triolen") else 4
        sl = slice(b0, min(b0 + abschnitt_schlaege, len(beats) - 1))
        p_med = np.median(P[sl], 0) - gmax if st == 4 else np.zeros(st)
        l_med = np.median(L[sl], 0) if st == 4 else np.zeros(st)
        for i in range(sl.start, sl.stop):
            for k in range(st):
                v = f["takt"][(i % 4) * st + k]
                if v is None:
                    continue
                s = beats[i] + (beats[i + 1] - beats[i]) * k / st + l_med[k] / 1000 * SR
                if st == 4:
                    eig = float(np.clip(P[i, k] - gmax - p_med[k], -2.0, 2.0))
                    pegel = p_med[k] + eig if p_med[k] > -30 else -12.0 + 20 * np.log10(max(v[1], .05))
                else:
                    pegel = 20 * np.log10(max(v[1], .05)) - 4
                zeiten.append((int(s), v[0], pegel, v[1]))
    zeiten.sort()
    y = np.zeros((n + SR * 3, 2), np.float32)
    for j, (s, art, pegel, staerke) in enumerate(zeiten):
        liste = K.get(art) or K["A"]
        idx = int(np.clip(round(staerke * (len(liste) - 1) + rng.integers(-1, 2)), 0, len(liste) - 1))
        x = liste[idx].copy()
        if art in "BCD" and j + 1 < len(zeiten):            # der Fuß schließt beim nächsten Hi-Hat-Schlag
            Lx = zeiten[j + 1][0] - s
            if 0 < Lx < len(x):
                fo = min(int(0.025 * SR), Lx); x = x[:Lx]; x[-fo:] *= np.linspace(1, 0, fo)[:, None]
        g = 10 ** (pegel / 20)
        a = max(0, s)
        y[a:a + len(x)] += g * x[:len(y) - a]
    return y[:n], zeiten
