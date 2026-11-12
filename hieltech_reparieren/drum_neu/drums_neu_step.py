#!/usr/bin/env python3
"""
HELTECH Stems – Schlagzeug neu einspielen (Stand 4. Nov 2026, getestet mit „Rise Up Jah Children“ v9).

Eingabe: eine Schlagzeug-Spur (am besten sauber getrennt). Ablauf:
  1. Trennung in Kick / Snare / Toms / Hi-Hat / Ride / Crash (MDX23C-DrumSep über audio-separator)
  2. Schlag-Karte je Teil (Zeit + Lautstärke jedes Schlags)
  3. Neu spielen mit dem Reggae-Kit aus den eigenen Loops (Kick, Snare, Rimshot, Hi-Hat),
     Toms aus Muldjord/Shots (immer voll ausklingend), Becken Muldjord Crash L/R
     - jeder Schlag am Anschlag ausgerichtet, Lautstärke = Spitze im Original
     - je Song nur die ähnlichsten Kit-Varianten, Ausklang (Kick/Snare/HH) an das Original angeglichen
     - Hi-Hat: Becken-Übersprechen raus, Trennverlust ausgeglichen, gleichmäßige Grundlautstärke
     - Becken: Einsatz/Lautstärke aus der Becken-Spur, Variante nach Ausklang-Verlauf
  4. Ausgabe: drums_neu.wav + Einzelteile + bericht.json

Aufruf:
  python drums_neu_step.py <schlagzeug.wav> <ausgabe_ordner>
        [--hh-ref VON,BIS]          Hi-Hat-Bezugsstelle in Sekunden (sonst automatisch)
        [--fremd-trommel]           gestimmte Fremd-Trommel aus der Snare-Spur entfernen
        [--teile-ordner ORDNER]     schon getrennte Teile benutzen (überspringt Schritt 1)
        [--weg-json DATEI]          Zeiten (s) je Teil, die nicht nachgespielt werden
        [--rimshot-note 37|40]      MIDI-Note für Rimshot (37 Side Stick = Standard, 40 echter Rimshot)
        [--timing-korrigieren]      Timing-Wackeln entfernen (Raster aus der Spur selbst)
        [--midi-bpm 120]            Tempo der MIDI-Datei (Standard 120 wie ein neues Logic-Projekt)
        [--modelle ORDNER]          Modell-Ordner für audio-separator
Ausgabe auf stdout (letzte Zeile): JSON {"ok": true, "datei": ..., ...}
"""
import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIER)
import drum_ersatz as d  # noqa: E402
import kit_render as kr  # noqa: E402

SR = d.SR
TEILE = ("kick", "snare", "hihat", "toms", "becken")


def melde(text):
    print(text, flush=True)

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



# ---------------------------------------------------------------- 1. Trennung
def trennen(eingabe, ziel, modelle):
    roh = os.path.join(ziel, "teile_roh")
    os.makedirs(roh, exist_ok=True)
    dateien = None
    G = _gpu()
    if G:
        try:
            dateien = list(G.trenne(eingabe, {"art": "sep", "modell": "MDX23C-DrumSep-aufr33-jarredou.ckpt"},
                                    "trennt das Schlagzeug").values())
        except Exception as e:  # noqa: BLE001
            melde(f"… GPU nicht verfügbar ({str(e)[:150]}) – rechne auf dem Mac")
    Sp = _speicher()
    if dateien is None and Sp:
        da = Sp.gespeichert(eingabe, Sp.A_SCHLAGZEUG)
        if da and all(t in da for t in ("kick", "snare", "hh")):
            melde("… Schlagzeug-Trennung schon vorhanden (Zwischenspeicher) – wird wiederverwendet")
            dateien = list(da.values())
    if dateien is None:
        import torch
        torch.backends.mps.is_available = lambda: False          # wie im App-Server: MPS macht hier Probleme
        from audio_separator.separator import Separator
        sep = Separator(output_dir=roh, model_file_dir=modelle, output_format="WAV")
        sep.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")
        dateien = [f if os.path.isabs(f) else os.path.join(roh, f) for f in sep.separate(eingabe)]
        if Sp:
            import re as _re
            nach_tag = {}
            for f in dateien:
                tags = _re.findall(r"\(([^)]+)\)", os.path.basename(f))
                if tags:
                    nach_tag[_re.sub(r"[^a-z0-9]+", "_", tags[-1].lower()).strip("_")] = f
            if all(t in nach_tag for t in ("kick", "snare", "hh")):
                dateien = list(Sp.merken(eingabe, Sp.A_SCHLAGZEUG, nach_tag).values())
    def hol(tag):
        f = [p for p in dateien if f"({tag})" in os.path.basename(p) or os.path.basename(p).lower() == f"{tag}.wav"]
        return sf.read(f[0], always_2d=True)[0] if f else None
    teile_dir = os.path.join(ziel, "teile")
    os.makedirs(teile_dir, exist_ok=True)
    for neu, tags in {"kick": ["kick"], "snare": ["snare"], "hihat": ["hh"], "toms": ["toms"],
                      "becken": ["ride", "crash"]}.items():
        xs = [hol(t) for t in tags]
        xs = [x for x in xs if x is not None]
        if not xs:
            raise RuntimeError(f"Trennung lieferte kein '{neu}'")
        n = min(len(x) for x in xs)
        sf.write(os.path.join(teile_dir, neu + ".flac"), sum(x[:n] for x in xs), SR, subtype="PCM_24")
    for tag in ("ride", "crash"):                  # einzeln nur fürs MIDI (Ride 51 / Crash 49, 57)
        x = hol(tag)
        if x is not None:
            sf.write(os.path.join(teile_dir, tag + ".flac"), x, SR, subtype="PCM_24")
    import shutil
    shutil.rmtree(roh, ignore_errors=True)          # große Zwischen-WAVs der Trennung nicht aufheben
    return teile_dir


# ---------------------------------------------- optional: gestimmte Fremd-Trommel
def fremd_trommel_finden(teile_dir):
    """Snare-Spur: Schläge nach Klang gruppieren; die lauteste Gruppe = echte Snare, die anderen
    (leiser + gestimmt) = Fremd-Trommel. Dazu Tom-Schläge mit derselben Tonhöhe bzw. gleichzeitig."""
    import librosa
    from scipy.cluster.hierarchy import fcluster, linkage
    x = d.lade(os.path.join(teile_dir, "snare.flac"))
    m = x.mean(1)
    pk = np.abs(m).max()
    on = librosa.onset.onset_detect(y=m, sr=SR, hop_length=256, units="samples", backtrack=True, delta=0.04)
    R, F = [], []
    for s in on:
        lv = 20 * np.log10(np.abs(m[s:s + int(.05 * SR)]).max() + 1e-9)
        if lv < 20 * np.log10(pk) - 35:
            continue
        seg = m[s + int(.01 * SR):s + int(.15 * SR)]
        if len(seg) < 2048:
            continue
        f0, vf, vp = librosa.pyin(seg, fmin=50, fmax=800, sr=SR, frame_length=2048, hop_length=256, resolution=0.1)
        ok = vf & ~np.isnan(f0)
        f = float(np.nanmedian(f0[ok])) if ok.sum() >= 3 else 0.0
        p = float(np.nanmean(vp[ok])) if ok.sum() >= 3 else 0.0
        g = m[s:s + int(.25 * SR)]
        g = np.pad(g, (0, max(0, int(.25 * SR) - len(g))))
        mf = librosa.feature.mfcc(y=g[:int(.1 * SR)], sr=SR, n_mfcc=13).mean(1)
        h = int(.005 * SR)
        e = 20 * np.log10(np.sqrt((g[:len(g) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
        t20 = np.argmax(e < e.max() - 20) * 0.005
        cen = librosa.feature.spectral_centroid(y=g[:int(.05 * SR)], sr=SR).mean()
        R.append((s / SR, f, lv, p))
        F.append(np.r_[mf[1:6] / 20, lv / 10, np.log2(max(f, 40) / 100) * (p > 0.3), t20 * 10, np.log2(cen / 1000)])
    if len(F) < 10:
        return {"snare": [], "toms": []}, {"hinweis": "zu wenige Schläge"}
    lab = fcluster(linkage(np.array(F), "ward"), t=5, criterion="maxclust")
    R = np.array(R)
    med = {c: float(np.median(R[lab == c, 2])) for c in set(lab)}
    laut = max(med.values())
    haupt = {c for c, v in med.items() if v >= laut - 6.0}          # echte Snare (auch in mehreren Klanggruppen)
    snare_pegel = float(np.median(R[np.isin(lab, list(haupt)), 2]))
    # laute Schläge sind immer echte Snare, auch wenn sie klanglich in einer anderen Gruppe gelandet sind
    weg_s = [float(t) for (t, f, lv, p), l in zip(R, lab) if l not in haupt and lv < snare_pegel - 6.0]
    ton = [f for (t, f, lv, p), l in zip(R, lab) if l not in haupt and f > 0]
    ton_haupt = float(np.median(ton)) if ton else 0.0
    # Toms: gleiche Tonhöhe wie die Fremd-Trommel oder gleichzeitig
    xt = d.lade(os.path.join(teile_dir, "toms.flac"))
    mt = xt.mean(1)
    pt = np.abs(mt).max()
    weg_t = []
    ws = np.array(weg_s)
    for s in librosa.onset.onset_detect(y=mt, sr=SR, hop_length=256, units="samples", backtrack=True, delta=0.05):
        lv = np.abs(mt[s:s + int(.05 * SR)]).max()
        if lv < pt * 10 ** (-30 / 20):
            continue
        g = kr._grund(mt[s + int(.02 * SR):s + int(.3 * SR)])
        t = s / SR
        gleichzeitig = len(ws) and np.min(np.abs(ws - t)) < 0.03
        gleicher_ton_leise = ton_haupt and abs(12 * np.log2(g / ton_haupt)) < 0.7 and lv < pt * 10 ** (-10 / 20)
        if gleichzeitig or gleicher_ton_leise:
            weg_t.append(t)
    return {"snare": weg_s, "toms": weg_t}, {"snare_entfernt": len(weg_s), "toms_entfernt": len(weg_t),
                                              "ton_hz": round(ton_haupt, 1)}


# ---------------------------------------------------------- 2. Schlag-Karte
def karte_bauen(teile_dir, kit):
    import librosa
    karte = {}
    for teil in ("kick", "snare", "hihat"):
        x = d.lade(os.path.join(teile_dir, f"{teil}.flac"))
        hits = d.schlaege(x, teil)
        if teil == "kick":
            for h in hits:
                h["art"] = "kick"
        elif teil == "snare":
            def fp(m):
                m = np.pad(m[: int(0.06 * SR)], (0, max(0, int(0.06 * SR) - len(m))))
                q = np.log10(librosa.feature.melspectrogram(y=m, sr=SR, n_mels=48, n_fft=2048, hop_length=512).mean(1) + 1e-10)
                return q - q.max()
            cent = {a: np.mean([fp(t["x"].mean(1)) for st in kit[a].values() for t in st], 0) for a in ("snare", "rimshot")}
            xs = x.mean(1)
            for h in hits:
                f = fp(xs[h["start"]:])
                h["art"] = "snare" if np.linalg.norm(f - cent["snare"]) < np.linalg.norm(f - cent["rimshot"]) else "rimshot"
        else:
            lab = d.zwei_gruppen([h["ausklang"] for h in hits]) if hits else []
            lang = np.mean([h["ausklang"] for h, l in zip(hits, lab) if l == 1]) if len(lab) and lab.max() else 0
            kurz = np.mean([h["ausklang"] for h, l in zip(hits, lab) if l == 0]) if len(lab) else 0
            offen_ok = len(lab) and lab.max() == 1 and lang > 0.18 and lang > 2 * kurz
            for h, l in zip(hits, lab):
                h["art"] = "hh_offen" if (offen_ok and l == 1) else "hh_zu"
        karte[teil] = (x, hits)
    x = d.lade(os.path.join(teile_dir, "toms.flac"))
    karte["toms"] = (x, d.schlaege(x, "kick"))
    karte["becken"] = (d.lade(os.path.join(teile_dir, "becken.flac")), [])
    return karte


def hh_ref_automatisch(t, rest_db):
    """Hi-Hat-Bezugsstelle automatisch: 4-s-Fenster mit mind. 6 Schlägen, dessen Median der echten
    HH-Höhen am nächsten an den kräftigsten 3 % aller Fenster liegt (= normal/kräftig gespielt)."""
    kand = []
    for a in np.arange(0, max(0.0, t.max() - 4) + 0.01, 1.0):
        k = (t >= a) & (t < a + 4)
        if k.sum() >= 6:
            kand.append((a, float(np.median(rest_db[k]))))
    if not kand:
        return float(t.min()), float(t.min() + 4)
    ziel = np.percentile([m for _, m in kand], 97)
    a = min(kand, key=lambda am: abs(am[1] - ziel))[0]
    return float(a), float(a + 4)


# ---------------------------------------------------------------- 3. Spielen
def timing_vorbereiten(eingabe, karte, teile, bericht):
    """Timing korrigieren: ein gemeinsames Raster aus den Anschlägen aller gespielten Teile (siehe
    timing_korrektur.py). Rückgabe: Funktion zeit_in_samples -> schub_in_samples."""
    import timing_korrektur as tk
    T, W = [], []
    for teil in teile:
        x, hits = karte[teil]
        if teil == "becken":
            bh = kr.becken_karte(x)
            T += [h["einsatz"] for h in bh]
            W += [10 ** (h["pegel"] / 20) for h in bh]
        else:
            kr.einsatz_setzen(x, hits)
            T += [h["einsatz"] for h in hits]
            W += [10 ** (h["pegel"] / 20) for h in hits]
    if not T:
        return lambda e: 0
    o = np.argsort(T)
    T, W = np.array(T)[o], np.array(W)[o]
    schub, info = tk.raster_schub(T / SR, tk.tempo_schaetzen(d.lade(eingabe).mean(1), SR), W)
    bericht["timing"] = info
    melde(f"✔ Timing: {info.get('verschoben', 0)} Schläge aufs Raster ({info.get('raster', '')})")

    def zu(e):
        i = int(np.searchsorted(T, e))
        best = min((j for j in (i - 1, i) if 0 <= j < len(T)), key=lambda j: abs(T[j] - e), default=None)
        return int(round(schub[best] * SR)) if best is not None and abs(T[best] - e) <= int(0.003 * SR) else 0
    return zu


def neu_spielen(eingabe, teile_dir, ziel, hh_ref=None, weg=None, nur=None, rimshot_note=37, midi_bpm=120.0,
                timing=False):
    from scipy.signal import butter, sosfiltfilt
    bericht = {}
    KIT = kr.lade_kit()
    karte = karte_bauen(teile_dir, KIT)
    if weg:
        for t, zeiten in weg.items():
            if t in karte and zeiten:
                x, hits = karte[t]
                z = np.array(zeiten) * SR
                vorher = len(hits)
                karte[t] = (x, [h for h in hits if np.min(np.abs(z - h["start"])) > 0.03 * SR])
                bericht.setdefault("fremd_trommel_entfernt", {})[t] = vorher - len(karte[t][1])
    x, hits = karte["toms"]                       # Übersprechen in der Tom-Spur: nur nahe der lautesten
    if hits:
        top = max(h["pegel"] for h in hits)
        karte["toms"] = (x, [h for h in hits if h["pegel"] > top - 20])

    roh = {}
    noten = []                                   # (zeit_s, gm_note, pegel_db, teil) für den MIDI-Export
    schub_von = timing_vorbereiten(eingabe, karte, TEILE if not nur else (nur,), bericht) if timing else None
    for teil in (TEILE if not nur else (nur,)):
        x, hits = karte[teil]
        melde(f"… spiele {teil}")
        if teil == "toms":
            if hits:
                kr.einsatz_setzen(x, hits)
                if schub_von:
                    for h in hits:
                        h["schub"] = schub_von(h["einsatz"])
                y, bericht["toms"] = kr.toms_spielen(len(x), x, hits, kr.tom_bibliothek())
                # Tom-Gruppen wie beim Spielen (Lücke > 1,2 Halbtöne) -> GM-Toms, hoch = 50 ... tief = 41
                gruppen = []
                for h in sorted(hits, key=lambda h: h["ton2"]):
                    if gruppen and 12 * np.log2(h["ton2"] / gruppen[-1][-1]["ton2"]) <= 1.2:
                        gruppen[-1].append(h)
                    else:
                        gruppen.append([h])
                gm = [50, 48, 47, 45, 43, 41]
                for i, g in enumerate(reversed(gruppen)):
                    for h in g:
                        noten.append(((h["einsatz"] + h.get("schub", 0)) / SR, gm[min(i, len(gm) - 1)], h["pegel"], "toms"))
            else:
                y = np.zeros((len(x), 2), np.float32)
        elif teil == "becken":
            bh = kr.becken_karte(x)
            if bh:
                bh = [h for h in bh if h["pegel"] > max(q["pegel"] for q in bh) - 25]
            if schub_von:
                for h in bh:
                    h["schub"] = schub_von(h["einsatz"])
            y, ben = kr.becken_spielen2(len(x), x, bh, kr.becken_bibliothek(None, ("CrashL", "CrashR")))
            hits = bh
            for h, note in zip(bh, becken_noten(teile_dir, x, bh, bericht)):
                noten.append(((h["einsatz"] + h.get("schub", 0)) / SR, note, h["pegel"], "becken"))
            bericht["becken"] = {"anzahl": len(bh), "schlaege(zeit,shot,hoehen_db,abstand)": ben}
        else:
            kr.einsatz_setzen(x, hits)
            if teil == "hihat":
                hits, raus = kr.hh_uebersprechen_weg(x, karte["becken"][0], hits)
                bericht["hihat_becken_uebersprechen_entfernt"] = len(raus)
                if hits:
                    hits = hihat_lautstaerke(eingabe, teile_dir, x, hits, hh_ref, bericht)
            kit_song = copy.deepcopy(KIT)
            arten = {"kick": ["kick"], "snare": ["snare", "rimshot"], "hihat": ["hh_zu", "hh_offen"]}[teil]
            if hits:
                bericht.setdefault("kit_angepasst", {}).update(kr.passend_machen(kit_song, arten, x, hits))
                if schub_von:
                    for h in hits:
                        h["schub"] = schub_von(h["einsatz"])
                y, benutzt = kr.spielen(len(x), hits, kit_song)
                gmn = {"kick": 36, "snare": 38, "rimshot": rimshot_note, "hh_zu": 42, "hh_offen": 46}
                if teil == "hihat":
                    for h, (note, cc4) in zip(hits, hh_oeffnung(x, hits, bericht, karte["becken"][0])):
                        noten.append(((h["einsatz"] + h.get("schub", 0)) / SR, note, h["pegel"], teil, cc4))
                else:
                    for h in hits:
                        noten.append(((h["einsatz"] + h.get("schub", 0)) / SR, gmn.get(h.get("art"), 38), h["pegel"], teil))
            else:
                y = np.zeros((len(x), 2), np.float32)
        roh[teil] = y
        bericht.setdefault("schlaege", {})[teil] = len(hits)
        melde(f"✔ {teil}: {len(hits)} Schläge")
    n = min(len(t) for t in roh.values())
    mix = sum(t[:n, :2] for t in roh.values())
    bericht["nur"] = nur or "alles"
    mix *= min(1.0, 0.98 / max(float(np.abs(mix).max()), 1e-6))
    os.makedirs(ziel, exist_ok=True)
    out = os.path.join(ziel, "drums_neu.wav")
    sf.write(out, mix, SR, subtype="PCM_24")
    for t in roh:
        sf.write(os.path.join(ziel, f"{t}_neu.flac"), np.clip(roh[t], -1, 1), SR, subtype="PCM_24")
    try:
        bericht["midi"] = midi_schreiben(eingabe, noten, ziel, nur, rimshot_note, midi_bpm)
    except Exception as e:                         # MIDI ist eine Zugabe – Audio-Ergebnis nie daran scheitern lassen
        bericht["midi"] = {"fehler": str(e)}
    json.dump(bericht, open(os.path.join(ziel, "bericht.json"), "w"), indent=1, ensure_ascii=False)
    return out, bericht


def becken_noten(teile_dir, xb, hits, bericht):
    """Je Becken-Schlag die GM-Note: Ride 51, wenn die Ride-Spur dort in den Höhen klar lauter ist als die
    Crash-Spur; bei den Crashs 49/57 nur, wenn eindeutig zwei verschiedene Becken (Klangfarbe) zu hören sind."""
    if not hits:
        return []
    noten = [49] * len(hits)
    pr, pc = os.path.join(teile_dir, "ride.flac"), os.path.join(teile_dir, "crash.flac")
    if os.path.exists(pc):
        cr = d.lade(pc)
        ri = d.lade(pr) if os.path.exists(pr) else None
        if ri is None:                                   # becken = ride + crash
            n = min(len(xb), len(cr))
            ri = xb[:n] - cr[:n]
        hc, hr = kr._hf(cr.mean(1)), kr._hf(ri.mean(1))
        for i, h in enumerate(hits):
            w = slice(h["einsatz"], h["einsatz"] + int(0.3 * SR))
            lc = 10 * np.log10(np.mean(hc[w] ** 2) + 1e-14)
            lr = 10 * np.log10(np.mean(hr[w] ** 2) + 1e-14)
            if lr > lc + 6:
                noten[i] = 51
        # zwei Crash-Becken? Klangfarbe (Schwerpunkt) muss klar in zwei Gruppen zerfallen
        import librosa
        m = cr.mean(1)
        idx = [i for i, n_ in enumerate(noten) if n_ == 49]
        cen = []
        for i in idx:
            a = hits[i]["einsatz"]
            g = m[a + int(0.02 * SR):a + int(0.4 * SR)]
            S = np.abs(librosa.stft(g, n_fft=4096, hop_length=1024)).mean(1)
            f = librosa.fft_frequencies(sr=SR, n_fft=4096)
            cen.append(np.log2((S * f).sum() / max(S.sum(), 1e-12)))
        zwei = False
        if len(cen) >= 6:
            c = np.array(cen)
            o = np.sort(c)
            j = int(np.argmax(np.diff(o)[2:-2])) + 2 if len(o) > 5 else 0
            if 3 <= j + 1 <= len(o) - 3:
                grenze = (o[j] + o[j + 1]) / 2
                lo, hi = c[c < grenze], c[c >= grenze]
                streu = np.sqrt((lo.var() * len(lo) + hi.var() * len(hi)) / len(c)) + 1e-9
                zwei = (hi.mean() - lo.mean()) > max(np.log2(1.12), 4 * streu)
                if zwei:
                    for i, ci in zip(idx, c):
                        noten[i] = 49 if ci >= grenze else 57        # helleres = Crash 1, dunkleres = Crash 2
    bericht["becken_midi"] = {"ride": noten.count(51), "crash1": noten.count(49), "crash2": noten.count(57),
                              "ride_crash_spuren": os.path.exists(pc)}
    return noten


def hh_oeffnung(x, hits, bericht, xb=None):
    """Öffnung jeder Hi-Hat aus dem Abklingen der Höhen bis zum nächsten Schlag:
    zu ≈ −250 dB/s, offen ≈ −20 dB/s. Ergebnis je Schlag (GM-Note, CC4): CC4 127 = zu, 0 = offen
    (so lesen es EZdrummer, Superior Drummer, Addictive Drums). Note 42 bis halboffen, ab weit offen 46,
    damit auch Plugins ohne CC4 (Logic Drum Kit Designer) richtig klingen."""
    hf = kr._hf(x.mean(1))
    hb = kr._hf(xb.mean(1)) if xb is not None else None
    k = int(0.005 * SR)
    aus = []
    unsicher = 0
    for i, h in enumerate(hits):
        a = h["einsatz"]
        nx = hits[i + 1]["einsatz"] if i + 1 < len(hits) else a + SR
        seg = hf[a:min(nx - k, a + int(0.25 * SR))]
        n = len(seg) // k
        if n >= 6:
            e = 20 * np.log10(np.sqrt((seg[:n * k].reshape(n, k) ** 2).mean(1)) + 1e-9)
            p = int(np.argmax(e[:6]))
            steigung = float(np.polyfit(np.arange(n - p) * 0.005, e[p:], 1)[0])
            o = float(np.clip((steigung + 180.0) / (180.0 - 40.0), 0.0, 1.0))   # −180 dB/s zu … −40 dB/s offen
            # langer Ausklang nur glauben, wenn er wirklich von der Hi-Hat kommt: nicht wieder ansteigend
            # (= nächster/anderer Schlag) und klar (12 dB) über dem Becken, das in die HH-Spur hineinklingt
            if o > 0:
                m0 = min(int(0.04 * SR), len(seg) // 2)
                lh = 10 * np.log10(np.mean(seg[m0:] ** 2) + 1e-14)
                lb = 10 * np.log10(np.mean(hb[a + m0:a + len(seg)] ** 2) + 1e-14) if hb is not None else -999
                if steigung > 0 or lh < lb + 12:
                    o = 0.0
                    unsicher += 1
        else:
            o = 0.0                                       # zu kurz zum Messen: sehr schnell gespielt = zu
        if h.get("art") == "hh_offen":
            o = max(o, 0.8)
        aus.append((46 if o >= 0.65 else 42, int(round(127 * (1 - o)))))
    oe = np.array([127 - c for _, c in aus]) / 127 if aus else np.zeros(0)
    bericht["hihat_midi"] = {"zu": int((oe < 0.3).sum()), "halboffen": int(((oe >= 0.3) & (oe < 0.65)).sum()),
                             "offen": int((oe >= 0.65).sum()), "becken_ausklang_ignoriert": unsicher}
    return aus


GM_NAMEN = {36: "Kick", 37: "Rimshot (Side Stick)", 38: "Snare", 41: "Tom tief (Standtom)", 42: "Hi-Hat zu",
            43: "Tom tief", 45: "Tom mittel", 46: "Hi-Hat offen", 47: "Tom mittel-hoch", 48: "Tom hoch",
            49: "Crash 1", 50: "Tom sehr hoch", 57: "Crash 2", 40: "Rimshot (echt)", 51: "Ride"}


def _tempo_schaetzen(eingabe):
    import librosa
    y, sr = librosa.load(eingabe, sr=22050, mono=True, duration=240)
    t = float(librosa.feature.tempo(y=y, sr=sr, start_bpm=90)[0])
    while t > 160:
        t /= 2
    while t < 60:
        t *= 2
    return round(t, 2)


def midi_schreiben(eingabe, noten, ziel, nur=None, rimshot_note=37, midi_bpm=120.0):
    """Schreibt die gespielte Schlag-Karte als General-MIDI-Schlagzeug (Kanal 10): exakte Zeiten,
    Velocity aus der Lautstärke im Original (je Element skaliert). Ohne Zusatz-Bibliothek."""
    import struct
    if not noten:
        return {"hinweis": "keine Schläge"}
    # Logic übernimmt beim Hineinziehen das Datei-Tempo NICHT, sondern spielt die Takte im Projekttempo.
    # Darum Standard 120 BPM (= neues Logic-Projekt): dann liegt das MIDI genau unter dem Audio.
    # Wer das echte Song-Tempo einträgt (und Logic darauf stellt), bekommt zusätzlich das richtige Taktraster.
    tempo = round(float(midi_bpm), 3) if midi_bpm and midi_bpm > 0 else 120.0
    try:
        geschaetzt = _tempo_schaetzen(eingabe)
    except Exception:
        geschaetzt = None
    ppq = 960
    tick = lambda s: int(round(s * tempo / 60.0 * ppq))  # noqa: E731
    vel = {}
    for teil in {n[3] for n in noten}:
        p = np.array([n[2] for n in noten if n[3] == teil])
        p95 = float(np.percentile(p, 95))
        vel[teil] = p95
    ev = []
    for t, note, pegel, teil, *rest in noten:
        v = int(np.clip(round(127 + 3.0 * (pegel - vel[teil])), 15, 127))
        if rest:                                                    # Hi-Hat-Öffnung (CC4) direkt vor dem Schlag
            ev.append((tick(t), 0xB9, 4, int(rest[0])))
        ev.append((tick(t), 0x99, note, v))
        ev.append((tick(t) + ppq // 8, 0x89, note, 0))            # 1/32 lang – Schlagzeug klingt sowieso aus
    reihe = {0x89: 0, 0xB9: 1, 0x99: 2}
    ev.sort(key=lambda e: (e[0], reihe[e[1]]))
    def vlq(n):
        b = [n & 0x7F]
        n >>= 7
        while n:
            b.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(b)
    def spur(daten):
        return b"MTrk" + struct.pack(">I", len(daten)) + daten
    usek = int(round(60_000_000 / tempo))
    t0 = b"\x00\xff\x51\x03" + usek.to_bytes(3, "big") + b"\x00\xff\x58\x04\x04\x02\x18\x08" + b"\x00\xff\x2f\x00"
    name = "Schlagzeug repariert" + (f" ({nur})" if nur else "")
    nb = name.encode("utf-8")
    d = b"\x00\xff\x03" + vlq(len(nb)) + nb
    letzt = 0
    for tk, st, note, v in ev:
        d += vlq(tk - letzt) + bytes([st, note, v])
        letzt = tk
    d += b"\x00\xff\x2f\x00"
    datei = os.path.join(ziel, "drums_neu.mid")
    with open(datei, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + spur(t0) + spur(d))
    # Notenliste (wie bei den anderen Schlagzeug-MIDIs der App)
    from collections import Counter
    c = Counter(n[1] for n in noten)
    zeilen = [f"Schlagzeug repariert – MIDI-Notenliste (General MIDI, Kanal 10), MIDI-Tempo {tempo:g} BPM", ""]
    zeilen += [f"Note {k:3d}  {GM_NAMEN.get(k, '?'):22s} {c[k]:5d} Schläge" for k in sorted(c)]
    zeilen += ["", "Passt direkt zu EZdrummer 3 / Logic Drum Kit Designer (Kick 36, Snare 38, Hi-Hat 42/46, Crash 49/57, Ride 51).",
               "Hi-Hat-Öffnung zusätzlich als Controller CC4 vor jedem Hi-Hat-Schlag (127 = zu, 0 = offen) –",
               "EZdrummer, Superior Drummer und Addictive Drums spielen damit auch halboffen.",
               f"Rimshot liegt auf Note {rimshot_note} ({'echter Rimshot' if rimshot_note == 40 else 'GM Side Stick'}).",
               "",
               f"WICHTIG Tempo: Die Datei ist mit {tempo:g} BPM geschrieben. Logic-Projekt auf {tempo:g} BPM stellen,",
               "dann liegt das MIDI genau unter dem Audio (Logic übernimmt das Tempo der Datei beim Hineinziehen nicht).",
               (f"Automatisch geschätztes Song-Tempo: {geschaetzt:g} BPM (nur als Hinweis, kann daneben liegen)." if geschaetzt else "")]
    with open(os.path.join(ziel, "drums_neu_Notenliste.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(zeilen) + "\n")
    return {"datei": datei, "tempo_bpm": tempo, "tempo_geschaetzt": geschaetzt, "noten": len(noten)}


def hihat_lautstaerke(eingabe, teile_dir, x, hits, hh_ref, bericht):
    """Trennverlust ausgleichen (echte HH-Höhen = Gesamtspur minus alle anderen Teile) und
    gleichmäßige Grundlautstärke wie an der Bezugsstelle; wirklich leise Schläge bleiben leise,
    reines Übersprechen wird nicht gespielt."""
    from scipy.signal import butter, sosfiltfilt
    hp6 = butter(4, 6000, "high", fs=SR, output="sos")
    hf = lambda z: sosfiltfilt(hp6, z.mean(1))  # noqa: E731
    M = hf(d.lade(eingabe))
    andere = [hf(d.lade(os.path.join(teile_dir, f"{t}.flac"))) for t in ("kick", "snare", "toms", "becken")]
    H = hf(x)
    w = int(0.03 * SR)
    verlust = np.zeros(len(hits))
    rest_db = np.zeros(len(hits))
    for i, h in enumerate(hits):
        a0 = h["start"]
        P = lambda q: float(np.mean(q[a0:a0 + w] ** 2)) + 1e-14  # noqa: E731
        rest = max(P(M) - sum(P(q) for q in andere), 1e-14)
        verlust[i] = float(np.clip(10 * np.log10(rest / P(H)), 0, 12))
        rest_db[i] = 10 * np.log10(rest)
    t = np.array([h["start"] / SR for h in hits])
    p0 = np.array([h["pegel"] for h in hits])
    p = p0 + verlust
    a_s, b_s = hh_ref if hh_ref else hh_ref_automatisch(t, rest_db)
    imref = (t >= a_s - 2) & (t < b_s + 2)
    if imref.sum() < 3:
        imref = np.ones(len(t), bool)
    ref = float(np.median(p[imref]))
    ref_h = float(np.median(rest_db[imref]))
    aktiv = rest_db > ref_h - 10
    leise = (~aktiv) & (rest_db > ref_h - 20)
    lokal = np.array([np.median(p[aktiv & (np.abs(t - ti) <= 4.0)]) if np.any(aktiv & (np.abs(t - ti) <= 4.0)) else ref
                      for ti in t])
    abw = p - lokal
    k = imref & aktiv
    hi = np.percentile(abw[k], 95) if k.any() else 6.0
    neu = np.where(aktiv, ref + np.minimum(abw, hi + 1), ref + (rest_db - ref_h))
    for h, q in zip(hits, neu):
        h["pegel"] = float(q)
    bleib = aktiv | leise
    bericht["hihat_lautstaerke"] = {"bezug_s": [round(a_s, 1), round(b_s, 1)], "bezug_automatisch": hh_ref is None,
                                    "trennverlust_db_median": round(float(np.median(verlust)), 1),
                                    "leise_gespielt": int(leise.sum()), "nicht_gespielt": int((~bleib).sum())}
    return [h for h, b in zip(hits, bleib) if b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eingabe")
    ap.add_argument("ziel")
    ap.add_argument("--hh-ref", default=None)
    ap.add_argument("--fremd-trommel", action="store_true")
    ap.add_argument("--teile-ordner", default=None)
    ap.add_argument("--weg-json", default=None)
    ap.add_argument("--nur", choices=list(TEILE), default=None,
                    help="nur dieses Element reparieren (Eingabe z. B. ein Kick-Stem)")
    ap.add_argument("--rimshot-note", type=int, choices=(37, 40), default=37,
                    help="MIDI-Note für Rimshot: 37 = GM Side Stick, 40 = echter Rimshot")
    ap.add_argument("--timing-korrigieren", action="store_true",
                    help="Timing-Wackeln entfernen (Raster aus der Spur selbst, ohne Bezugsspur)")
    ap.add_argument("--midi-bpm", type=float, default=120.0,
                    help="Tempo der MIDI-Datei (Standard 120 = neues Logic-Projekt; echtes Song-Tempo für richtiges Raster)")
    ap.add_argument("--modelle", default=os.path.expanduser("~/HELTECH_Stems/models/"))
    a = ap.parse_args()
    os.makedirs(a.ziel, exist_ok=True)
    try:
        if a.teile_ordner:
            teile_dir = a.teile_ordner
        else:
            melde("… trenne Schlagzeug in Kick/Snare/Toms/Hi-Hat/Becken (dauert einige Minuten)")
            teile_dir = trennen(a.eingabe, a.ziel, a.modelle)
        weg = json.load(open(a.weg_json)) if a.weg_json else None
        info_fremd = None
        if a.fremd_trommel and not weg:
            melde("… suche gestimmte Fremd-Trommel")
            weg, info_fremd = fremd_trommel_finden(teile_dir)
        hh_ref = tuple(float(v) for v in a.hh_ref.split(",")) if a.hh_ref else None
        out, bericht = neu_spielen(a.eingabe, teile_dir, a.ziel, hh_ref, weg, a.nur, a.rimshot_note, a.midi_bpm, a.timing_korrigieren)
        if info_fremd:
            bericht["fremd_trommel"] = info_fremd
        print(json.dumps({"ok": True, "datei": out, "ordner": a.ziel,
                          "schlaege": bericht.get("schlaege"), "hihat": bericht.get("hihat_lautstaerke")},
                         ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "fehler": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
