#!/usr/bin/env python3
"""
HELTECH Stems – Instrumenten-Erkennung v2 (Stand 5. Nov 2026), ersetzt die CLAP-Erkennung.
GPU-Fassung für RunPod (11. Nov 2026): gleiche Logik wie auf dem Mac, rechnet auf der GPU; Gewichte auf dem Netz-Speicher.

Hör-Modell: EfficientAT mn40_as_ext (MobileNet, trainiert auf AudioSet: >2 Mio. echte Aufnahmen,
527 Klangklassen, mAP 0,487; MIT-Lizenz). Gewichte (274 MB) werden beim ersten Lauf von GitHub geladen.

Ablauf:
  1. Die Spur wird zusätzlich in einen TON-Teil (gehaltene Instrumente) und einen SCHLAG-Teil
     (Trommeln/Anschläge) zerlegt (HPSS), damit kein Instrument ein anderes überdeckt.
  2. Alle drei Fassungen werden in 2-s-Fenstern (Schritt 1 s) abgehört -> Zeitleiste je Klasse.
  3. AudioSet-Klassen -> die 53 App-Instrumente (siehe erkennung_kern.ZUORDNUNG).
  4. Korrekturen aus Hörproben des Nutzers (5. Nov 2026, „5 Other“, KI-Reggae-Song):
     - Reggae-Orgel wird vom Modell oft „Akkordeon“ genannt  -> zählt als Orgel
     - Synth-Leads werden oft „Violine“/„Mundharmonika“ genannt -> zählen als Synth, außer es ist
       zusätzlich echter Streicher-Klang da bzw. die Mundharmonika ist sehr deutlich und lange
  5. Entscheidung je Instrument:
       erkannt  = deutlich (≥ 3 % der Zeit über 0,15 und Spitze ≥ 0,30) oder kurz, aber klar (Spitze ≥ 0,50)
       möglich  = Spitze ≥ 0,25 und kurz über 0,15 (wird angezeigt, aber nicht vorausgewählt)
     Synth braucht Spitze ≥ 0,40 (Bass/Schlagzeug klingen sonst schnell „synthetisch“).
  6. Je Instrument: Zeitleiste (wann es spielt) und auf Wunsch eine 6-s-Hörprobe an der stärksten Stelle.

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
        import torch
        global GERAET
        GERAET = "cuda" if torch.cuda.is_available() else "cpu"
        m, mel = m.to(GERAET), mel.to(GERAET)
    finally:
        os.chdir(cwd)
    return m, mel


def zeitleiste(m, mel, y):
    import torch
    w, h = int(FENSTER * SR), int(SCHRITT * SR)
    starts = list(range(0, max(1, len(y) - w + 1), h))
    out = []
    for i in range(0, len(starts), 16):
        batch = []
        for s in starts[i:i + 16]:
            seg = y[s:s + w]
            batch.append(np.pad(seg, (0, w - len(seg))) if len(seg) < w else seg)
        x = torch.from_numpy(np.stack(batch)).float().to(GERAET or "cpu")
        with torch.no_grad():
            p, _ = m(mel(x).unsqueeze(1))
            out.append(torch.sigmoid(p.float()).cpu().numpy())
    return np.array(starts) / SR, np.concatenate(out)


def abschnitte(t, s, schwelle=0.15, luecke=3.0, mindest=2.0):
    an = t[s >= schwelle]
    if len(an) == 0:
        return []
    seg = [[float(an[0]), float(an[0]) + FENSTER]]
    for x in an[1:]:
        if x - seg[-1][1] <= luecke:
            seg[-1][1] = float(x) + FENSTER
        else:
            seg.append([float(x), float(x) + FENSTER])
    return [[round(a, 1), round(b, 1)] for a, b in seg if b - a >= mindest]


# Was in welche getrennte Spur gehört (Spur-Wissen wie ein Toningenieur): in der Bass-Spur zählt nur Bass usw.
SPUR_ERLAUBT = {
    "drums": PERKUSSIV | {"congas", "toms"},
    "bass": {"bass", "double-bass", "synth"},
    "vocals": {"vocal", "lead-vocal", "back-vocal"},
    "guitar": {"guitar", "electric-guitar", "acoustic-guitar", "banjo", "dobro", "mandolin", "sitar", "ukulele", "harp"},
    "piano": {"piano", "digital-piano", "keys", "harpsichord", "organ", "synth"},
    "other": set(ALLE_53) - {"bass", "double-bass", "drums", "kick", "snare", "hh", "vocal", "lead-vocal", "back-vocal"},
}


def analyse(pfad, m, mel, lab, ansichten=("voll", "ton", "schlag")):
    import librosa
    y, _ = librosa.load(pfad, sr=SR, mono=True)
    teile = {"voll": y}
    if "ton" in ansichten or "schlag" in ansichten:
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        if "ton" in ansichten:
            H, _ = librosa.decompose.hpss(S, margin=(1.0, 3.0))
            teile["ton"] = librosa.istft(H, hop_length=512, length=len(y))
        if "schlag" in ansichten:
            _, P = librosa.decompose.hpss(S, margin=(3.0, 1.0))
            teile["schlag"] = librosa.istft(P, hop_length=512, length=len(y))
    V = {}
    for name in ansichten:
        t, Pr = zeitleiste(m, mel, teile[name])
        V[name] = ek.slug_scores(Pr, lab)
    K = {}
    for s_ in ek.ZUORDNUNG:
        alt = "schlag" if s_ in PERKUSSIV else "ton"
        K[s_] = np.maximum(V["voll"][s_], V[alt][s_]) if alt in V else V["voll"][s_]
    # --- Korrekturen aus den Hörproben (5. Nov 2026) ---
    anteil = lambda s_, x=0.15: float(np.mean(K[s_] >= x))  # noqa: E731
    akk = K["accordion"].copy()
    K["organ"] = np.maximum(K["organ"], akk)            # Reggae-Orgel klingt für das Modell wie Akkordeon
    K["accordion"] = akk * 0
    if not (anteil("bowed_strings") >= 0.03 or anteil("strings") >= 0.03):
        K["synth"] = np.maximum(K["synth"], K["violin"])  # „Violine“ ohne Streicher-Klang = Synth
        K["violin"] = K["violin"] * 0
    if not (K["harmonica"].max() >= 0.7 and anteil("harmonica") >= 0.05):
        K["synth"] = np.maximum(K["synth"], K["harmonica"])
        K["harmonica"] = K["harmonica"] * 0
    K["lead-vocal"] = K["vocal"]
    return t, K


def entscheiden(t, K, erlaubt=None):
    res = {}
    for s_ in ALLE_53:
        x = K.get(s_)
        if x is None or (erlaubt is not None and s_ not in erlaubt):
            continue
        spitze, a15 = float(x.max()), float(np.mean(x >= 0.15))
        erkannt = (a15 >= 0.03 and spitze >= (0.40 if s_ == "synth" else 0.30)) or spitze >= 0.50
        if s_ == "synth" and spitze < 0.40:
            erkannt = False
        moeglich = (not erkannt) and spitze >= 0.25 and a15 > 0
        bi = int(np.argmax(x))
        res[s_] = {"instrument": s_, "margin": round(spitze, 3), "detected": bool(erkannt), "moeglich": bool(moeglich),
                   "best_window_start_sec": round(float(t[bi]), 1), "anteil_prozent": round(a15 * 100, 1),
                   "zeiten": abschnitte(t, x)}
    return res


def aufraeumen(res):
    if res.get("lead-vocal", {}).get("detected") and "vocal" in res:
        res["vocal"]["detected"] = False
    fam = [r for k, r in res.items() if k in ("guitar", "electric-guitar", "acoustic-guitar", "banjo", "dobro",
                                                "mandolin", "sitar", "ukulele") and r["detected"]]
    if len(fam) > 1:
        best = max(fam, key=lambda r: r["margin"])
        for r in fam:
            if r is not best:
                r["detected"], r["moeglich"] = False, True
    out = []
    for s_ in ALLE_53:
        out.append(res.get(s_, {"instrument": s_, "margin": 0.0, "detected": False, "moeglich": False,
                                "best_window_start_sec": 0.0, "anteil_prozent": 0.0, "zeiten": []}))
    out.sort(key=lambda r: (not r["detected"], not r["moeglich"], -r["margin"]))
    return out


def hoerproben_schneiden(res, ordner, quelle_fuer):
    import librosa
    import soundfile as sf
    os.makedirs(ordner, exist_ok=True)
    cache = {}
    for r in res:
        if not (r["detected"] or r["moeglich"]):
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


def erkennen(pfad, hoerproben=None, spuren=None):
    melde("… lade Hör-Modell (beim ersten Mal 274 MB Download)")
    m, mel = modell()
    lab = ek.labels(EAT_DIR)
    if not spuren:
        melde("… höre die Datei (ganz, Ton-Teil, Schlag-Teil)")
        t, K = analyse(pfad, m, mel, lab)
        res = aufraeumen(entscheiden(t, K))
        if hoerproben:
            hoerproben_schneiden(res, hoerproben, lambda r: pfad)
        return res
    gesamt = {}
    for spur, sp in spuren.items():
        erlaubt = SPUR_ERLAUBT.get(spur, set(ALLE_53))
        melde(f"… höre Spur „{spur}“")
        t, K = analyse(sp, m, mel, lab, ("voll", "schlag") if spur == "drums" else ("voll", "ton"))
        for k, r in entscheiden(t, K, erlaubt).items():
            r["spur"] = spur
            r["spur_datei"] = sp
            alt = gesamt.get(k)
            if alt is None or (r["detected"], r["margin"]) > (alt["detected"], alt["margin"]):
                gesamt[k] = r
    res = aufraeumen(gesamt)
    if hoerproben:
        hoerproben_schneiden(res, hoerproben, lambda r: r.get("spur_datei", pfad))
    for r in res:
        r.pop("spur_datei", None)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav")
    ap.add_argument("output_json")
    ap.add_argument("--hoerproben", default=None, help="Ordner für 6-s-Hörproben je Instrument")
    ap.add_argument("--spuren-ordner", default=None,
                    help="schon getrennte Spuren (drums/bass/guitar/piano/other/vocals.wav): jede einzeln abhören")
    a = ap.parse_args()
    spuren = None
    if a.spuren_ordner:
        import glob
        spuren = {}
        for n in ("drums", "bass", "guitar", "piano", "other", "vocals"):
            f = sorted(glob.glob(os.path.join(a.spuren_ordner, f"*{n}*.wav")))
            if f:
                spuren[n] = f[0]
    r = erkennen(a.input_wav, a.hoerproben, spuren)
    with open(a.output_json, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    er = [x["instrument"] for x in r if x["detected"]]
    mo = [x["instrument"] for x in r if x["moeglich"]]
    print(f"Fertig: {len(er)} erkannt: {', '.join(er) or '–'} | möglich: {', '.join(mo) or '–'}")
    for x in r:
        if x["detected"] and x.get("spur"):
            print(f"  {x['instrument']:15s} in Spur {x['spur']:7s} {x['anteil_prozent']:5.1f} % der Zeit  {x['zeiten'][:4]}")
