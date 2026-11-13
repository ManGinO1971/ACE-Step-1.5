#!/usr/bin/env python3
"""
HELTECH Stems – Instrumenten-Erkennung v2 (Stand 5. Nov 2026), ersetzt die CLAP-Erkennung.
Eine Fassung für Mac und RunPod (11. Nov 2026): gleiche Logik überall. Das Hör-Modell rechnet auf der schnellsten
vorhandenen Recheneinheit (NVIDIA-GPU auf RunPod, Apple-GPU „MPS“ auf dem Mac, sonst CPU). Die Vorarbeit je Spur
(Ton-Teil/Schlag-Teil herausrechnen) läuft für alle Spuren gleichzeitig auf mehreren Prozessor-Kernen.

Hör-Modell: EfficientAT mn40_as_ext (MobileNet, trainiert auf AudioSet: >2 Mio. echte Aufnahmen,
527 Klangklassen, mAP 0,487; MIT-Lizenz). Gewichte (274 MB) werden beim ersten Lauf von GitHub geladen.

Ablauf:
  1. Die Spur wird zusätzlich in einen TON-Teil (gehaltene Instrumente) und einen SCHLAG-Teil
     (Trommeln/Anschläge) zerlegt (HPSS), damit kein Instrument ein anderes überdeckt.
  2. Alle drei Fassungen werden in 2-s-Fenstern (Schritt 1 s) abgehört -> Zeitleiste je Klasse.
  3. AudioSet-Klassen -> die 53 App-Instrumente (siehe erkennung_kern.ZUORDNUNG).
  4. Zusätzlich wird der ganze Song einmal abgehört: daraus kommt die Stilrichtung (Genre).
  5. Entscheidung (12. Nov 2026, siehe hoermodell/erkennung_regeln.py – gilt für jeden Song gleich):
     Spur-Wissen, Instrumenten-Familien (ein Name je Familie), eigene Zeit, gleiche Quelle, Genre (unübliche
     Instrumente brauchen mehr Beweis), Doppelung Gitarre/Piano/Orgel, Schlagzeug-Teile.
     Ergebnis je Instrument: stufe = sicher (vorausgewählt) / vielleicht (wählbar) / aus.
  6. Je Instrument: Zeitleiste (wann es spielt) und auf Wunsch eine 6-s-Hörprobe an der stärksten Stelle.
  7. Neben dem Ergebnis wird <ergebnis>.roh.json geschrieben (Zeitleisten + Genre-Werte): damit entscheidet die App
     nach einer anderen Genre-Wahl sofort neu, ohne noch einmal zu hören.

Aufruf (wie die alte Erkennung, gleiche JSON-Felder + neue):
  python detect_instruments_eat.py <audio> <ergebnis.json> [--hoerproben ORDNER]
"""
import argparse
import json
import os
import sys

import numpy as np

HIER = os.path.dirname(os.path.abspath(__file__))
EAT_DIR = os.path.join(HIER, "hoermodell", "EfficientAT")
RES_DIR = os.environ.get("HIELTECH_EAT_RESOURCES") or ("/runpod-volume/hieltech_eat" if os.path.isdir("/runpod-volume")
                                                       else os.path.join(EAT_DIR, "resources"))
GERAET = None
sys.path.insert(0, os.path.join(HIER, "hoermodell"))
import erkennung_kern as ek  # noqa: E402
import erkennung_regeln as er  # noqa: E402

SR = 32000
FENSTER, SCHRITT = 2.0, 1.0   # Fenster 2 s, alle 1 s
PERKUSSIV = {"drums", "kick", "snare", "hh", "percussion", "tambourine", "timpani"}
ALLE_53 = ["accordion", "acoustic-guitar", "back-vocal", "banjo", "bass", "bassoon", "bells", "bowed_strings", "brass",
           "cello", "clarinet", "congas", "digital-piano", "dobro", "double-bass", "drums", "electric-guitar", "flute",
           "french-horn", "glockenspiel", "guitar", "harmonica", "harp", "harpsichord", "hh", "keys", "kick",
           "lead-vocal", "mandolin", "marimba", "oboe", "organ", "percussion", "piano", "saxophone", "sitar", "snare",
           "strings", "synth", "tambourine", "timpani", "toms", "triangle", "trombone", "trumpet", "tuba", "ukulele",
           "viola", "violin", "vocal", "wind", "wind-chimes", "woodwind"]


def melde(t):
    print(t, flush=True)


GEWICHTE = "mn40_as_ext_mAP_487.pt"
GEWICHTE_BYTES = 274206417
GEWICHTE_URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/" + GEWICHTE


def gewichte_sichern():
    """Hör-Modell vollständig vorhanden? Abgebrochene Downloads (falsche Größe) löschen und neu laden,
    mit Fortsetzen nach Abbruch (bis zu 5 Versuche)."""
    import urllib.request
    ziel = os.path.join(RES_DIR, GEWICHTE)
    os.makedirs(os.path.dirname(ziel), exist_ok=True)
    for versuch in range(5):
        if os.path.exists(ziel) and os.path.getsize(ziel) == GEWICHTE_BYTES:
            return ziel
        if os.path.exists(ziel) and os.path.getsize(ziel) > GEWICHTE_BYTES:
            os.remove(ziel)
        schon = os.path.getsize(ziel) if os.path.exists(ziel) else 0
        melde(f"… lade Hör-Modell ({schon // 1_000_000} von {GEWICHTE_BYTES // 1_000_000} MB, Versuch {versuch + 1})")
        req = urllib.request.Request(GEWICHTE_URL, headers={"Range": f"bytes={schon}-"} if schon else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(ziel, "ab" if schon and r.status == 206 else "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
        except Exception as e:
            melde(f"  Download unterbrochen ({str(e)[:80]}), versuche weiter …")
    if os.path.exists(ziel) and os.path.getsize(ziel) == GEWICHTE_BYTES:
        return ziel
    raise RuntimeError(f"Hör-Modell konnte nicht vollständig geladen werden ({ziel}). Internet prüfen und erneut versuchen.")


def modell():
    gewichte_sichern()
    cwd = os.getcwd()
    os.chdir(EAT_DIR)                                   # EfficientAT erwartet metadata/ und resources/ relativ
    sys.path.insert(0, EAT_DIR)
    try:
        from models.mn import model as mnm
        from models.preprocess import AugmentMelSTFT
        from helpers.utils import NAME_TO_WIDTH
        mnm.model_dir = RES_DIR
        m = mnm.get_model(width_mult=NAME_TO_WIDTH("mn40_as_ext"), pretrained_name="mn40_as_ext")
        m.eval()
        mel = AugmentMelSTFT(n_mels=128, sr=SR, win_length=800, hopsize=320)
        mel.eval()
        global GERAET
        GERAET = geraet_waehlen(m, mel)
        m, mel = m.to(GERAET), mel.to(GERAET)
    finally:
        os.chdir(cwd)
    return m, mel


def geraet_waehlen(m, mel):
    """Schnellste Recheneinheit, die das Hör-Modell wirklich rechnen kann: cuda, sonst mps (Apple), sonst cpu.
    HIELTECH_EAT_GERAET=cpu erzwingt die CPU."""
    import torch
    wunsch = (os.environ.get("HIELTECH_EAT_GERAET") or "").strip().lower()
    if wunsch == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            x = torch.zeros(2, int(FENSTER * SR)).to("mps")          # kleine Probe: geht das Modell auf der Apple-GPU?
            with torch.no_grad():
                p, _ = m.to("mps")(mel.to("mps")(x).unsqueeze(1))
            if bool(torch.isfinite(p.float()).all().cpu()):
                return "mps"
    except Exception:  # noqa: BLE001 – dann eben CPU
        pass
    m.to("cpu"), mel.to("cpu")
    return "cpu"


def zeitleiste(m, mel, y):
    import torch
    global GERAET
    w, h = int(FENSTER * SR), int(SCHRITT * SR)
    starts = list(range(0, max(1, len(y) - w + 1), h))
    out = []
    n = 16 if (GERAET or "cpu") == "cpu" else 64
    for i in range(0, len(starts), n):
        batch = []
        for s in starts[i:i + n]:
            seg = y[s:s + w]
            batch.append(np.pad(seg, (0, w - len(seg))) if len(seg) < w else seg)
        x = torch.from_numpy(np.stack(batch)).float()
        try:
            with torch.no_grad():
                p, _ = m(mel(x.to(GERAET or "cpu")).unsqueeze(1))
                out.append(torch.sigmoid(p.float()).cpu().numpy())
        except Exception:  # noqa: BLE001 – Apple-GPU kann etwas nicht: auf der CPU weiter
            if (GERAET or "cpu") != "mps":
                raise
            GERAET = "cpu"
            m.to("cpu"), mel.to("cpu")
            with torch.no_grad():
                p, _ = m(mel(x).unsqueeze(1))
                out.append(torch.sigmoid(p.float()).numpy())
    return np.array(starts) / SR, np.concatenate(out)


def teile_rechnen(pfad, ansichten=("voll", "ton", "schlag"), merkmale=False):
    """Vorarbeit für eine Datei (nur Prozessor, kein Modell): ganz, Ton-Teil, Schlag-Teil. Läuft für mehrere Spuren
    gleichzeitig in eigenen Prozessen."""
    import librosa
    y, _ = librosa.load(pfad, sr=SR, mono=True)
    teile = {"voll": y}
    if "ton" in ansichten or "schlag" in ansichten:
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        if "ton" in ansichten:
            H, _ = librosa.decompose.hpss(S, margin=(1.0, 3.0))
            teile["ton"] = librosa.istft(H, hop_length=512, length=len(y)).astype(np.float32)
        if "schlag" in ansichten:
            _, P = librosa.decompose.hpss(S, margin=(3.0, 1.0))
            teile["schlag"] = librosa.istft(P, hop_length=512, length=len(y)).astype(np.float32)
    if merkmale:
        # Pegel und Anschlags-Kurve (50 Werte je Sekunde) – für die Prüfung „gleiche Anschläge, viel leiser = Abdruck“
        teile["_pegel_db"] = float(20 * np.log10(max(1e-7, float(np.sqrt(np.mean(y.astype(np.float64) ** 2))))))
        try:
            teile["_anschlag"] = librosa.onset.onset_strength(y=y, sr=SR, hop_length=640).astype(np.float32)
        except Exception:  # noqa: BLE001
            teile["_anschlag"] = None
    return teile


def anschlag_gleich(a, b, versatz=3):
    """Wie ähnlich sind die Anschläge zweier Spuren (0…1)? Bester Wert bei bis zu ±60 ms Versatz."""
    if a is None or b is None:
        return 0.0
    n = min(len(a), len(b))
    if n < 200:
        return 0.0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    na, nb = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    best = 0.0
    for v in range(-versatz, versatz + 1):
        x, y = (a[v:], b[:n - v]) if v >= 0 else (a[:n + v], b[-v:])
        best = max(best, float((x * y).sum()) / (na * nb))
    return round(max(0.0, best), 3)


def analyse(pfad, m, mel, lab, ansichten=("voll", "ton", "schlag"), teile=None):
    """Hört eine Datei ab. Rückgabe (t, K, P_voll): K = Stärke je Instrument und Sekunde, P_voll = alle 527 Klassen."""
    if teile is None:
        teile = teile_rechnen(pfad, ansichten)
    V, P_voll = {}, None
    for name in ansichten:
        t, Pr = zeitleiste(m, mel, teile[name])
        if name == "voll":
            P_voll = Pr
        V[name] = ek.slug_scores(Pr, lab)
    K = {}
    for s_ in ek.ZUORDNUNG:
        alt = "schlag" if s_ in PERKUSSIV else "ton"
        K[s_] = np.maximum(V["voll"][s_], V[alt][s_]) if alt in V else V["voll"][s_]
    return t, K, P_voll


def genre_werte(P, lab):
    """Stilrichtung aus dem ganzen Song: Mittelwert der Genre-Klassen des Hör-Modells."""
    L = {l: i for i, l in enumerate(lab)}
    mittel = {k: float(P[:, L[k]].mean()) for ks in er.GENRES.values() for k in ks if k in L}
    return er.genre_aus_klassen(mittel)


def roh_bauen(n, kandidaten, genre, paare, pegel):
    """Alles, was die Regeln brauchen (klein genug für die App): je Instrument Spur + Zeitleiste in Prozent."""
    inst = {}
    for s_, (spur, x) in kandidaten.items():
        if float(x.max()) >= 0.10:
            inst[s_] = {"spur": spur, "k": [int(round(float(v) * 100)) for v in x]}
    return {"fassung": er.FASSUNG, "schritt": SCHRITT, "fenster": FENSTER, "n": int(n), "instrumente": inst,
            "genre": genre, "paare": paare, "spuren": pegel}


def fertig_liste(roh, genre="auto"):
    """Regeln anwenden und auf alle 53 Instrumente auffüllen (wie bisher: ein Eintrag je Instrument)."""
    liste, info = er.auswerten(roh, genre)
    da = {r["instrument"] for r in liste}
    for s_ in ALLE_53:
        if s_ not in da:
            liste.append({"instrument": s_, "margin": 0.0, "stufe": "aus", "detected": False, "moeglich": False,
                          "best_window_start_sec": 0.0, "anteil_prozent": 0.0, "zeiten": [], "spur": ""})
    return [r for r in liste if r["instrument"] in ALLE_53], info


def hoerproben_schneiden(res, ordner, quelle_fuer):
    import librosa
    import soundfile as sf
    os.makedirs(ordner, exist_ok=True)
    cache = {}
    for r in res:
        if not (r["detected"] or r["moeglich"] or r["margin"] >= 0.40):   # auch knapp Aussortierte (andere Genre-Wahl)
            continue
        q = quelle_fuer(r)
        if q not in cache:
            yo, so = librosa.load(q, sr=44100, mono=False)
            cache[q] = (np.atleast_2d(yo), so)
        yo, so = cache[q]
        a = max(0.0, r["best_window_start_sec"] - 2.0)
        f = os.path.join(ordner, f"{r['instrument']}.wav")
        sf.write(f, yo[:, int(a * so):int((a + 6.0) * so)].T, so, subtype="PCM_16")
        r["hoerprobe"] = f


def _ansichten(spur):
    return ("voll", "schlag") if spur == "drums" else ("voll", "ton")


def erkennen(pfad, hoerproben=None, spuren=None, genre="auto"):
    """Rückgabe (liste, roh, info)."""
    import time
    t0 = time.time()
    vorarbeit, pool = {}, None
    if spuren and len(spuren) > 1:
        # Vorarbeit aller Spuren gleichzeitig starten (eigene Prozesse) – währenddessen lädt hier das Hör-Modell
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            kerne = max(1, min(len(spuren) + 1, (os.cpu_count() or 2) - 1, 7))
            if kerne > 1:
                pool = ProcessPoolExecutor(max_workers=kerne, mp_context=mp.get_context("spawn"))
                vorarbeit = {spur: pool.submit(teile_rechnen, sp, _ansichten(spur), True) for spur, sp in spuren.items()}
                vorarbeit["_mix"] = pool.submit(teile_rechnen, pfad, ("voll",))
        except Exception:  # noqa: BLE001 – dann nacheinander wie bisher
            vorarbeit, pool = {}, None
    melde("… lade Hör-Modell (beim ersten Mal 274 MB Download)")
    m, mel = modell()
    melde(f"… Hör-Modell rechnet auf: {GERAET}")
    lab = ek.labels(EAT_DIR)
    t_modell = time.time() - t0

    def geholt(name, sonst):
        if name in vorarbeit:
            try:
                return vorarbeit[name].result(timeout=1800)
            except Exception:  # noqa: BLE001 – dann hier rechnen
                pass
        return sonst()

    kandidaten, paare, pegel = {}, [], {}
    if not spuren:
        melde("… höre die Datei (ganz, Ton-Teil, Schlag-Teil)")
        t, K, P_mix = analyse(pfad, m, mel, lab)
        kandidaten = {s_: ("", x) for s_, x in K.items()}
        n = len(t)
    else:
        anschlag, n = {}, 0
        for spur, sp in spuren.items():
            melde(f"… höre Spur „{spur}“")
            teile = geholt(spur, lambda: teile_rechnen(sp, _ansichten(spur), True))
            t, K, _ = analyse(sp, m, mel, lab, _ansichten(spur), teile)
            n = max(n, len(t))
            pegel[spur] = {"pegel_db": round(teile.get("_pegel_db", -120.0), 1)}
            anschlag[spur] = teile.get("_anschlag")
            for s_, x in K.items():
                if not er.spur_erlaubt(spur, s_):
                    continue
                alt = kandidaten.get(s_)
                st = float(x.max()) * float(np.mean(x >= 0.15)) ** 0.5
                if alt is None or st > alt[2]:
                    kandidaten[s_] = (spur, x, st)
        kandidaten = {s_: (v[0], v[1]) for s_, v in kandidaten.items()}
        melde("… höre den ganzen Song (Stilrichtung)")
        P_mix = zeitleiste(m, mel, geholt("_mix", lambda: teile_rechnen(pfad, ("voll",)))["voll"])[1]
        for a_, b_ in (("guitar", "piano"), ("guitar", "other"), ("piano", "other")):
            if a_ in anschlag and b_ in anschlag:
                paare.append({"a": a_, "b": b_, "gleich": anschlag_gleich(anschlag[a_], anschlag[b_]),
                              "pegel_db": round(pegel[b_]["pegel_db"] - pegel[a_]["pegel_db"], 1)})
    if pool is not None:
        pool.shutdown(wait=False)
    # gleich lange Zeitleisten (Spuren können um ein Fenster abweichen)
    kandidaten = {s_: (sp, np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]) for s_, (sp, x) in kandidaten.items()}
    roh = roh_bauen(n, kandidaten, genre_werte(P_mix, lab), paare, pegel)
    res, info = fertig_liste(roh, genre)
    t_hoeren = time.time() - t0 - t_modell
    if hoerproben:
        for r in res:
            r["spur_datei"] = (spuren or {}).get(r.get("spur") or "") or pfad
        hoerproben_schneiden(res, hoerproben, lambda r: r.get("spur_datei") or pfad)
    print("ZEITEN " + json.dumps({"modell_laden": round(t_modell, 1), "hoeren": round(t_hoeren, 1),
                                  "gesamt": round(time.time() - t0, 1), "geraet": GERAET}), flush=True)
    for r in res:
        r.pop("spur_datei", None)
    return res, roh, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav")
    ap.add_argument("output_json")
    ap.add_argument("--hoerproben", default=None, help="Ordner für 6-s-Hörproben je Instrument")
    ap.add_argument("--spuren-ordner", default=None,
                    help="schon getrennte Spuren (drums/bass/guitar/piano/other/vocals.wav): jede einzeln abhören")
    ap.add_argument("--genre", default="auto", help="auto, ohne oder eine Stilrichtung (reggae, rock, jazz …)")
    a = ap.parse_args()
    spuren = None
    if a.spuren_ordner:
        import glob
        spuren = {}
        for n in ("drums", "bass", "guitar", "piano", "other", "vocals"):
            f = sorted(glob.glob(os.path.join(a.spuren_ordner, f"*{n}*.wav")))
            if f:
                spuren[n] = f[0]
    r, roh, info = erkennen(a.input_wav, a.hoerproben, spuren, a.genre)
    with open(a.output_json, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    with open(a.output_json + ".roh.json", "w") as f:
        json.dump({"roh": roh, "info": info}, f, ensure_ascii=False)
    er_ = [x["instrument"] for x in r if x["detected"]]
    mo = [x["instrument"] for x in r if x["moeglich"]]
    print(f"Genre: {info.get('genre') or '– (nicht eindeutig)'} | vermutet: {info.get('genre_auto')}")
    print(f"Fertig: {len(er_)} sicher: {', '.join(er_) or '–'} | vielleicht: {', '.join(mo) or '–'}")
    for x in r:
        if x["detected"] and x.get("spur"):
            print(f"  {x['instrument']:15s} in Spur {x['spur']:7s} {x['anteil_prozent']:5.1f} % der Zeit  {x['zeiten'][:4]}")
