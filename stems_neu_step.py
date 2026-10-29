#!/usr/bin/env python3
"""
HIELTECH Stems – "Neu erzeugen" Nachbearbeitung für EIN Instrument nach Rezept (29. Okt 2026).
(Schlagzeug läuft getrennt über stems_drums_step.py.)

  1. Von den neu erzeugten Fassungen (Extract aus dem GANZEN Song) die genaueste wählen –
     Vergleich mit der Vorlage (53er-Trennung dieses Instruments).
  2. Rezept aus stems_rezepte.py:
       bass       -> <inst>.wav = Grundton + Obertöne neu, Original-Lautstärke          (fertig)
       karte      -> <inst>.wav = Zeitmaske + Gate + Karte -> Instrumenten-Vocoder         (fertig)
       melodie    -> <inst>_B_karte.wav  und  <inst>_C_neu_aufgebaut.wav (Grundton+Obertöne) (Test)
       perkussion -> <inst>_B_karte.wav (kurze Maske, Anschläge bleiben)                    (Test)
       gesang     -> <inst>_B_karte.wav (Gesangskette folgt)
     Immer dazu <inst>_A_roh.wav (unbearbeitete beste Fassung) zum Vergleich.
  Keine Lautstärke-Anpassung außer beim Bass (Original-Lautstärke), kein Kompressor/EQ (macht der Nutzer in Logic).

Aufruf: python stems_neu_step.py <instrument> <vorlage.wav> <kandidaten.json (Liste)> <ausgabeordner>
Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stems_karte_step as k  # noqa: E402
from stems_post_step import SR, fit, load, score  # noqa: E402
from stems_rezepte import MASKE, rezept  # noqa: E402

TONHAFT = {"bass", "karte", "melodie", "gesang"}


def main():
    inst, ref_path, cand_json, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    r = rezept(inst)
    art = r["rezept"]
    info = {"rezept": art, "status": r["status"], "kandidaten": []}
    try:
        with open(cand_json) as f:
            paths = json.load(f)
        ref = load(ref_path)
        n = ref.shape[0]
        best, best_s = None, -9.0
        for p in paths:
            c = fit(load(p), n)
            sc = score(ref, c, "bass" if art in TONHAFT else "drums")   # Töne (Chroma) bzw. Anschläge
            info["kandidaten"].append({"datei": os.path.basename(p), "punkte": sc[0]})
            if sc[0] > best_s:
                best, best_s, info["gewaehlt"] = c, sc[0], os.path.basename(p)
        if best is None:
            raise RuntimeError("keine neu erzeugte Fassung")
        sf.write(os.path.join(out, f"{inst}_A_roh.wav"), k.schutz(best), SR, subtype="PCM_24")
        if art == "blaeser_satz":
            # Hörtest 29.10.: 4_brass_NEU_hall gewählt – aus der UNMASKIERTEN Fassung (ganzer Ausklang)
            from blaeser_zusammen import verarbeite
            from blaeser_trennen import satz_und_teile
            v = verarbeite(best.astype(np.float64), ref.astype(np.float64), SR, 1.0)
            satz, teile = satz_und_teile(v, SR)
            sf.write(os.path.join(out, f"{inst}.wav"), satz, SR, subtype="PCM_24")
            # Hörtest 29.10.: Aufteilung nach Stimmlage (B) – Unterordner wie beim Schlagzeug, Summe = Satz
            os.makedirs(os.path.join(out, "teile"), exist_ok=True)
            for name, spur in teile.items():
                sf.write(os.path.join(out, "teile", f"{name}.wav"), spur, SR, subtype="PCM_24")
        elif art == "bass":
            from bass_rund import bass_fertig
            fmin, fmax = r.get("ton", (30, 330))
            sf.write(os.path.join(out, f"{inst}.wav"), k.schutz(bass_fertig(best.astype(np.float64), SR, fmin, fmax)), SR, subtype="PCM_24")
        else:
            k.EINST[inst] = MASKE.get(art, MASKE["karte"])
            mg, anteil, _nr = k.maske_gate(ref, best, inst)
            info["klingt_anteil"] = round(anteil, 3)
            name_b = f"{inst}.wav" if art == "karte" else f"{inst}_B_karte.wav"
            try:
                sf.write(os.path.join(out, name_b), k.schutz(k.neu_aufbauen(mg, False)), SR, subtype="PCM_24")
            except Exception as e:  # noqa: BLE001
                info["karte_fehler"] = str(e)[:200]
            if art == "melodie":
                try:
                    from bass_rund import bass_fertig
                    fmin, fmax = r.get("ton", (80, 1200))
                    c = bass_fertig(mg.astype(np.float64), SR, fmin, fmax, K=30, obergrenze=12000.0)
                    sf.write(os.path.join(out, f"{inst}_C_neu_aufgebaut.wav"), k.schutz(c), SR, subtype="PCM_24")
                except Exception as e:  # noqa: BLE001
                    info["neu_aufgebaut_fehler"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001
        info["fehler"] = str(e)[:300]
    print(f"[Stems-Neu] {inst}: {json.dumps(info, ensure_ascii=False)}", file=sys.stderr, flush=True)
    print(json.dumps({"ok": "fehler" not in info, "instrument": inst, **info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
