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
    hilfe = sys.argv[5:]   # Ordner mit weiteren 53er-Vorlagen (für Percussion: congas, percussion, kick, snare, hh, toms)
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
        if art == "perkussion_satz":
            # Hörtest 30.10.: Stand = 9_percussion_ohne_hihat (Isolieren danach klang NICHT besser -> nicht drin)
            def vorlage(name):
                for d in hilfe + [os.path.dirname(ref_path)]:
                    for ext in (".wav", ".flac"):
                        q = os.path.join(d, name + ext)
                        if os.path.isfile(q):
                            return fit(load(q), n)
                return None
            from percussion_zusammen import verarbeite as perc_verarbeite
            from perkussion_sortieren import sortieren
            from hihat_raus import hihat_raus, hihat_raus_klang
            cg, pc = vorlage("congas"), vorlage("percussion")
            R_all = (cg if cg is not None else 0) + (pc if pc is not None else 0)
            if not isinstance(R_all, np.ndarray):
                R_all = ref
            R_cong = cg if cg is not None else R_all
            satz, _trom = perc_verarbeite(best.astype(np.float64), R_all.astype(np.float64), R_cong.astype(np.float64), SR, False)
            satz = satz.astype(np.float32)
            refs = {"congas": R_cong, "percussion": pc if pc is not None else R_all,
                    "kick": vorlage("kick"), "toms": vorlage("toms"), "snare": vorlage("snare")}
            if all(v is not None for v in refs.values()):
                satz, spuren, liste = sortieren(satz, {k: v.astype(np.float32) for k, v in refs.items()})
                info["schlagzeug_schlaege_raus"] = len(liste)
                os.makedirs(os.path.join(out, "an_schlagzeug"), exist_ok=True)
                for k_, v_ in spuren.items():
                    sf.write(os.path.join(out, "an_schlagzeug", f"{k_}.wav"), v_, SR, subtype="PCM_24")
            else:
                info["sortieren_fehlt"] = [k_ for k_, v_ in refs.items() if v_ is None]
            hh = vorlage("hh")
            if hh is not None:
                satz = hihat_raus(satz, SR, R_all.astype(np.float32), hh.astype(np.float32), 2.0)
            satz = hihat_raus_klang(satz, SR)
            # Hörtest 30.10.: Fell-Resonanz als letzter Schritt (wie 3_percussion_NEU_resonanz)
            from percussion_zusammen import resonanz
            satz = resonanz(satz, SR)
            sf.write(os.path.join(out, f"{inst}.wav"), k.schutz(satz), SR, subtype="PCM_24")
        elif art == "streicher_einzeln":
            # Hörtest 30.10.: violin_NEU_sauber_dynamik = Stand – B-Klang (Karte/Vocoder), Ausklang bis zum Ende aus A,
            # nur Töne dieses Instruments (Keys/Synth raus), Lautstärke-Verlauf wie im Original (±6 dB, 1,5 s glatt)
            from streicher_einzeln import einzeln_sauber
            from dynamik_wie_original import angleichen
            k.EINST[inst] = MASKE["melodie"]
            mg, anteil, _nr = k.maske_gate(ref, best, inst)
            info["klingt_anteil"] = round(anteil, 3)
            B = k.neu_aufbauen(mg, False)
            fmin, fmax = r.get("ton", (180, 2200))
            y = einzeln_sauber(B.astype(np.float64), best.astype(np.float64), ref.astype(np.float64), SR, fmin, fmax)
            y, gmax = angleichen(y, ref.astype(np.float64), SR)
            info["dynamik_max_db"] = round(gmax, 1)
            sf.write(os.path.join(out, f"{inst}.wav"), y, SR, subtype="PCM_24")
        elif art in ("blaeser_satz", "streicher_satz"):
            # Hörtest 29.10.: 4_brass_NEU_hall gewählt – aus der UNMASKIERTEN Fassung (ganzer Ausklang)
            # 30.10. TEST: Streicher-Satz genauso (gehaltene Töne, Hall, Lücken füllen), Teile Geige/Bratsche/Cello nach Lage
            from blaeser_zusammen import verarbeite
            from blaeser_trennen import satz_und_teile
            v = verarbeite(best.astype(np.float64), ref.astype(np.float64), SR, 1.0)
            namen = ("tief_cello", "mitte_bratsche", "hoch_geige") if art == "streicher_satz" else ("tief_posaune", "mitte_sax", "hoch_trompete")
            satz, teile = satz_und_teile(v, SR, namen)
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
