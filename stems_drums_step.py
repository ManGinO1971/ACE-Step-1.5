#!/usr/bin/env python3
"""
HIELTECH Stems – Test 6 (29. Okt 2026): Schlagzeug in Einzelteile – jedes Teil eigene Spur.

  1. Getrenntes Original-Schlagzeug (nur Vorlage) UND jede der 3 neu erzeugten Schlagzeug-Fassungen
     (Extract aus dem GANZEN Song) mit DrumSep (MDX23C, aufr33 & jarredou) zerlegen in
     Kick, Snare, Toms, Hi-Hat, Ride, Crash (Ride + Crash = Becken).
  2. JE TEIL die genaueste der 3 Fassungen wählen (Anschläge im Vergleich zum Original-Teil) –
     die Kick kann also aus Versuch 1 kommen, die Snare aus Versuch 3.
  3. Je Teil eigene Behandlung: Zeitmaske + Gate mit eigenen Werten (Becken klingen lange aus,
     Kick/Snare kurz), dann Karte -> Instrumenten-Vocoder (wie drums_B in Test 5).
KEINE Lautstärke-Anpassung, nur Übersteuerungsschutz.

LIZENZ-HINWEIS: DrumSep-Gewichte haben keine dokumentierte Lizenz (Code MIT) – vor dem Verkauf klären,
wie bei den Vocoder-C-Gewichten.

Aufruf: python stems_drums_step.py <getrennt_ordner> <kandidaten.json> <ausgabeordner>
Ausgabe je Teil: drums_<teil>_0_original.wav, drums_<teil>_A_roh.wav, drums_<teil>_B_behandelt.wav,
         dazu drums_gesamt_B.wav (alle behandelten Teile zusammen). Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import logging
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stems_karte_step as k  # noqa: E402
from stems_post_step import SR, fit, load, score  # noqa: E402

DRUMSEP = os.environ.get("HIELTECH_DRUMSEP_MODEL", "MDX23C-DrumSep-aufr33-jarredou.ckpt")
TEILE = ["kick", "snare", "toms", "hihat", "becken"]
# (Maske dB, Gate dB, Halten s, Blende s)
k.EINST.update({
    "kick": (-50.0, -60.0, 0.30, 0.03),
    "snare": (-50.0, -60.0, 0.40, 0.03),
    "toms": (-50.0, -60.0, 0.60, 0.05),
    "hihat": (-55.0, -65.0, 0.40, 0.05),
    "becken": (-62.0, -72.0, 2.00, 0.30),
})
_SEP = {}


def zerlegen(path, work):
    """-> {kick, snare, toms, hihat, becken}: (n, 2) bei 44,1 kHz."""
    from audio_separator.separator import Separator
    if "s" not in _SEP:
        s = Separator(output_dir=work, output_format="WAV", model_file_dir=os.path.join(HERE, "uvr_models"),
                      log_level=logging.WARNING)
        s.load_model(model_filename=DRUMSEP)
        _SEP["s"] = s
    s = _SEP["s"]
    roh, files, hinweis = {}, [], ""
    # Test 6: die installierte audio-separator-Fassung speichert bei Mehrspur-Modellen nur 2 Spuren
    # (Kick, Snare). Das Modell liefert intern aber alle 6 -> direkt abholen.
    try:
        mi = s.model_instance
        mi.audio_file_path = path
        mi.audio_file_base = os.path.splitext(os.path.basename(path))[0]
        mix = mi.prepare_mix(path)
        src = mi.demix(mix=mix)
        if isinstance(src, dict):
            for stem in ("kick", "snare", "toms", "hh", "ride", "crash"):
                if stem in src:
                    a = np.asarray(src[stem], dtype=np.float32)
                    roh[stem] = a.T if a.shape[0] == 2 else a
    except Exception as e:  # noqa: BLE001
        hinweis = f" (direkt: {str(e)[:150]})"
    if not all(x in roh for x in ("kick", "snare", "toms", "hh")):
        roh = {}
        files = [f if os.path.isabs(f) else os.path.join(work, f) for f in s.separate(path)]
    for f in files:
        low = os.path.basename(f).lower()
        for stem in ("kick", "snare", "toms", "hh", "ride", "crash"):
            if f"({stem})" in low:
                roh[stem] = load(f)
    fehlt = [x for x in ("kick", "snare", "toms", "hh") if x not in roh]
    if fehlt:
        raise RuntimeError(f"DrumSep: Teile fehlen {fehlt} in {[os.path.basename(f) for f in files]}{hinweis}")
    n = max(a.shape[0] for a in roh.values())
    roh = {key: fit(a, n) for key, a in roh.items()}
    becken = roh.get("ride", np.zeros((n, 2), np.float32)) + roh.get("crash", np.zeros((n, 2), np.float32))
    return {"kick": roh["kick"], "snare": roh["snare"], "toms": roh["toms"], "hihat": roh["hh"], "becken": becken}


def main():
    sep_dir, cand_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="stems_drums_")
    with open(cand_json) as f:
        cands = json.load(f)
    bericht = {"ok": True, "instrumente": {}}
    info = {"teile": {}}
    try:
        orig_path = os.path.join(sep_dir, "drums.wav")
        n = load(orig_path).shape[0]
        orig = zerlegen(orig_path, work)
        neu = []
        for i, p in enumerate(cands.get("drums", [])):
            try:
                neu.append((os.path.basename(p), zerlegen(p, work)))
            except Exception as e:  # noqa: BLE001
                info.setdefault("zerlegen_fehler", str(e)[:200])
        if not neu:
            raise RuntimeError("keine neue Schlagzeug-Fassung zerlegbar: " + info.get("zerlegen_fehler", ""))
        gesamt = np.zeros((n, 2), np.float32)
        for teil in TEILE:
            ti = {"kandidaten": []}
            try:
                o = fit(orig[teil], n)
                sf.write(os.path.join(out, f"drums_{teil}_0_original.wav"), k.schutz(o), SR, subtype="PCM_24")
                if float(np.abs(o).max()) < 1e-3:
                    ti["hinweis"] = "im Original praktisch nicht vorhanden"
                best, best_s = None, -9.0
                for name, parts in neu:
                    c = fit(parts[teil], n)
                    sc = score(o, c, "drums")
                    ti["kandidaten"].append({"datei": name, "punkte": sc[0]})
                    if sc[0] > best_s:
                        best, best_s, ti["gewaehlt"] = c, sc[0], name
                sf.write(os.path.join(out, f"drums_{teil}_A_roh.wav"), k.schutz(best), SR, subtype="PCM_24")
                mg, anteil, _nr = k.maske_gate(o, best, teil)
                ti["klingt_anteil"] = round(anteil, 3)
                try:
                    b = k.neu_aufbauen(mg, False)
                except Exception as e:  # noqa: BLE001
                    ti["vocoder_fehler"] = str(e)[:200]
                    b = mg
                gesamt += b.astype(np.float32)
                sf.write(os.path.join(out, f"drums_{teil}_B_behandelt.wav"), k.schutz(b), SR, subtype="PCM_24")
            except Exception as e:  # noqa: BLE001
                ti["fehler"] = str(e)[:300]
            info["teile"][teil] = ti
            print(f"[Stems-Test6] {teil}: {json.dumps(ti, ensure_ascii=False)}", file=sys.stderr, flush=True)
        sf.write(os.path.join(out, "drums_gesamt_B.wav"), k.schutz(gesamt), SR, subtype="PCM_24")
    except Exception as e:  # noqa: BLE001
        info["fehler"] = str(e)[:300]
    bericht["instrumente"]["drums"] = info
    print(json.dumps(bericht, ensure_ascii=False))


if __name__ == "__main__":
    main()
