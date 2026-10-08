#!/usr/bin/env python3
"""
Standalone-Test: generische Stimmen-Resynthese ueber den WORLD-Vocoder (pyworld).

Zweck: eine UNIVERSELLE Alternative/Vorstufe zu rvc_convert_step.py, die KEIN
trainiertes, persoenliches Stimmmodell (.pth) braucht. Statt die Stimme in
eine andere Identitaet umzuwandeln (das macht RVC), zieht dieses Skript aus
dem Audio EINEN einzigen Tonhoehen-Verlauf (F0) plus Spektralhuelle
(CheapTrick) und Aperiodizitaet (D4C) heraus und baut daraus dieselbe Stimme
komplett neu auf (Synthesis). Das zwingt alle "parallel klingenden"
Toene/Obertoene (das "Synthesizer-Stapel"-Gefuehl) auf EINE kohaerente
Tonhoehen-Spur - aber universell einsetzbar, ohne eigenes Stimmmodell.

WICHTIG (Reihenfolge-Notiz fuer die spaetere App-Integration):
  - rvc_convert_step.py (echte Stimm-UMWANDLUNG ueber ein trainiertes .pth)
    bleibt der LETZTE, OPTIONALE Schritt - nur aktiv, wenn ein Nutzer sein
    eigenes Stimmmodell hinterlegt hat. Nach RVC muss Gate+Denoise NOCHMAL
    laufen, weil der RVC-Vocoder leichtes Rauschen/Hiss einbringen kann.
  - Dieses Skript (WORLD-Resynthese) ist die UNIVERSELLE Vorstufe, die auch
    ohne eigenes Stimmmodell funktioniert - direkt auf der KI-generierten
    Stimme (z.B. nach Kompression/Saettigung, VOR einem moeglichen RVC-Schritt).

WICHTIG (generisch fuer ALLE Kunden/Songs, nicht nur diesen Test-Song):
  Alle drei Mechanismen unten (automatische Tonlagen-Kalibrierung, Oktav-
  Fehler-Korrektur, Geschwindigkeitsbegrenzung, Aussetzer-Reparatur) laufen
  standardmaessig IMMER automatisch mit und stellen sich bei jedem Durchlauf
  neu auf den jeweiligen Song ein - unabhaengig von Stimmlage (maennlich/
  weiblich/Kind), Tonhoehe oder Genre/Stil. Sie arbeiten bewusst NICHT mit
  festen Hz-Werten, sondern relativ (in Halbtonschritten pro Sekunde bzw.
  pro Millisekunde) - das macht sie unabhaengig von der jeweiligen Stimme,
  vom Genre und von spaeteren Parameter-Aenderungen (z.B. Hop-Laenge). Es
  muss also fuer jeden neuen Kunden-Song NICHTS per Hand nachgestellt
  werden. --f0_floor/--f0_ceil/--no_f0_glitch_repair bleiben nur als
  manuelle Experten-Overrides erhalten, falls ein Einzelfall mal von Hand
  nachjustiert werden soll.

  Hintergrund (warum mehrere verschiedene Mechanismen noetig sind): die
  Diagnose an einem echten Testsong hat gezeigt, dass "kaputte Stellen"
  NICHT alle dieselbe Ursache haben. Manche sind reine Tracking-Aussetzer
  (kurz ausgeschlagen, kehrt zur alten Tonlage zurueck), manche sind Oktav-
  Verwechslungen (bekannter Fehler JEDES Tonhoehen-Schaetzers), und manche
  sind schlicht zu schnelle Tonhoehen-Bewegungen - einzelne abrupte Spruenge
  genauso wie mehrstufige, schnelle "Laeufe", bei denen die KI einen Ton
  nicht kontrolliert durchhaelt wie ein echter Saenger (hoert sich an wie
  eine "kaputte Welle/Vibration"). Die Geschwindigkeitsbegrenzung (Punkt 2
  unten) deckt diesen letzten, groessten Fall durchgehend ab - unabhaengig
  davon, ob es ein einzelner Sprung oder eine laengere unruhige Passage ist
  - waehrend Oktav-Korrektur und Aussetzer-Reparatur die beiden anderen,
  andersartigen Fehlerbilder gezielt abdecken, ohne normale, gewollte
  Gesangsbewegungen (Melodie, Vibrato, bewusste Rutscher) anzutasten.

Nutzung:
    python world_resynth_step.py --input vocals_warm.wav --output vocals_world.wav
    python world_resynth_step.py --input vocals_warm.wav --output vocals_world.wav --pitch_shift -2
    python world_resynth_step.py --input vocals_warm.wav --output vocals_world.wav --mix 0.6 --d4c_threshold 0.95

--mix (0.0-1.0, Standard 1.0): wie stark die Resynthese insgesamt durchschlagen
soll. 1.0 = komplett neu aufgebaute Stimme (bisheriges Verhalten). Niedriger
(z.B. 0.5) mischt wieder einen Teil des Original-Klangs dazu - gut, falls der
Effekt "zu stark"/zu vocoder-haft wirkt.

--d4c_threshold (Standard 0.85, WORLD-eigener Standard): steuert, wie viel vom
Klang als "Rauschanteil" (Atem, Zischlaute wie S/Sch) statt als reiner Ton
behandelt wird. HOEHER (z.B. 0.95) laesst mehr natuerliches Rausch-/Luftanteil
in den Hoehen durch (natuerlicher klingende S/Sch-Laute, weniger metallisch).
NIEDRIGER (z.B. 0.7) macht die Hoehen toniger/klarer, aber ggf. kuenstlicher.
Falls "die Hoehen nicht richtig" klingen, ist das der erste Hebel zum Testen.

--f0_floor/--f0_ceil (Standard: AUTOMATISCH kalibriert, siehe unten): werden
normalerweise nicht gebraucht. Nur setzen, wenn fuer einen Einzelfall von Hand
nachjustiert werden soll (z.B. ein sehr ungewoehnlicher Stimmumfang).

AUTOMATISCHE TONLAGEN-KALIBRIERUNG (laeuft standardmaessig, kein Hand-Tuning
pro Song/Kunde noetig):
  Erst ein breiter, schneller Erstdurchlauf (50-1100 Hz, deckt Bass bis
  Sopran/Kinderstimme ab), aus dem die tatsaechliche Stimmlage dieses Songs
  (5./95. Perzentil der erkannten Tonhoehe) ermittelt wird. Der eigentliche
  Durchlauf grenzt den Suchbereich dann automatisch auf diese Stimmlage
  (plus Sicherheitsabstand) ein - das reduziert Oktav-Sprung-Fehler bei
  jeder Stimme. Mit --f0_floor/--f0_ceil abschaltbar.

Die folgenden drei Reparatur-Mechanismen laufen alle unter
--no_f0_glitch_repair (schaltet alle drei zusammen ab) in dieser Reihenfolge:

1) OKTAV-FEHLER-KORREKTUR: JEDER Tonhoehen-Schaetzer kann sich auf das
   Doppelte/die Haelfte/das Drei- oder Vierfache der tatsaechlichen Frequenz
   "verschauen" (Grundton mit einer Obertonharmonischen verwechselt). Diese
   Korrektur vergleicht jeden Frame mit seinem lokalen Umfeld und zieht ihn
   zurueck in die richtige Oktave, wenn das Verhaeltnis sehr nah an 2, 1/2,
   3, 1/3, 4 oder 1/4 liegt.

2) GESCHWINDIGKEITSBEGRENZUNG ("Slew-Rate-Limiter", ersetzt die frueheren
   getrennten Mechanismen "Sprung-Glaettung" und "Instabilitaets-Glaettung"
   durch eine einfachere, durchgehend arbeitende Loesung): begrenzt, wie
   viele Halbtoene pro Sekunde die Tonhoehe sich ueberhaupt bewegen darf
   (Standard 500 Halbtoene/Sekunde - oberhalb von allem, was eine
   menschliche Stimme auch bei starkem Vibrato oder einem schnellen Lauf
   tatsaechlich schafft). Jeder zu schnelle Uebergang wird automatisch auf
   diese Maximalgeschwindigkeit ausgebremst und damit zeitlich gestreckt -
   egal ob es ein einziger abrupter Sprung ist oder ein mehrstufiger,
   schneller Lauf ueber eine laengere Passage. Das deckt genau das Symptom
   "Welle/Vibration, Ton haelt nicht durch" ab, unabhaengig davon, ob die
   Ursache ein Tracking-Fehler oder eine tatsaechlich instabile Tonhoehe in
   der KI-Stimme selbst ist. An jeder stimmlosen Luecke (Atempause) wird
   bewusst neu angesetzt, weil danach ein voellig neuer Ton legitim ist.

   Wichtige Nachjustierung (nach genauerem Hinhoeren): ein niedrigerer Wert
   (z.B. die anfangs verwendeten 150 Halbtoene/Sekunde) macht zwar jeden
   einzelnen Analyse-Schritt glatter, dehnt dafuer aber sehr grosse Spruenge
   (mehr als eine Oktave) auf bis zu 150-165ms "Gleit"-Zeit - das klingt
   dann selbst nicht mehr wie ein Sprungfehler, sondern wie ein kuenstliches
   Hochziehen/Sirenen-Rutschen, was man beim direkten Vergleich mit
   unauffaelligen Song-Stellen trotzdem als "anders" heraushoert. 500
   Halbtoene/Sekunde begrenzt dieselben Fehler auf maximal ca. 30-40ms
   Gleit-Zeit (selbst bei einem seltenen Extremfall von mehr als einer
   Oktave) - kurz genug, um wie ein normaler, schneller Ansatz/Rutscher zu
   wirken statt wie ein hoerbares Gleiten, bei weiterhin sauber begrenzten
   Einzelschritten.

3) AUSSETZER-REPARATUR: findet kurze (meist 1-3 Frame) Tracking-Aussetzer,
   die zur VORHERIGEN Tonlage zurueckspringen (anders als Mechanismus 2, wo
   die Tonlage auf einem NEUEN Niveau bleibt), und zieht nur diese per
   Interpolation glatt.

Alle drei arbeiten in Halbtonschritten pro Sekunde/Millisekunde statt in
Hz/Frames und sind damit automatisch auf jede Stimme/jeden Song
uebertragbar. Die Experten-Overrides (--f0_max_semitones_per_sec,
--f0_glitch_baseline_ms, --f0_glitch_dev_semitones, --f0_glitch_max_run_ms)
sind normalerweise nicht noetig.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def auto_calibrate_f0_range(mono, sr, f0_method, pw):
    """
    Ermittelt automatisch einen sinnvollen F0-Suchbereich fuer DIESEN Song,
    unabhaengig von Stimmlage/Geschlecht/Genre. Erst ein breiter, schneller
    Erstdurchlauf (50-1100 Hz, deckt praktisch jede menschliche Stimme ab),
    dann Eingrenzung auf das 5.-95. Perzentil der dort erkannten Tonhoehe
    (plus Sicherheitsabstand nach unten/oben) fuer den eigentlichen Durchlauf.
    Das ersetzt das fruehere Hand-Tuning pro Song.
    """
    wide_floor, wide_ceil = 50.0, 1100.0
    if f0_method == "harvest":
        f0_wide, _ = pw.harvest(mono, sr, f0_floor=wide_floor, f0_ceil=wide_ceil)
    else:
        f0_wide, _ = pw.dio(mono, sr, f0_floor=wide_floor, f0_ceil=wide_ceil)

    voiced = f0_wide[f0_wide > 0]
    if len(voiced) < 20:
        return wide_floor, wide_ceil, False

    p5, p95 = np.percentile(voiced, [5, 95])
    floor = max(wide_floor, p5 * 0.7)
    ceil = min(wide_ceil, p95 * 1.4)
    return floor, ceil, True


def octave_correct_f0(f0, window=15, tolerance=0.08):
    """
    Korrigiert den klassischen Oktav-/Harmonischen-Verwechslungsfehler, den
    JEDER Tonhoehen-Schaetzer gelegentlich macht (siehe Docstring oben).
    Arbeitet relativ (Frequenzverhaeltnis zum lokalen Median), nicht mit
    festen Hz-Werten - funktioniert dadurch gleich gut bei jeder Stimmlage.
    Gibt (korrigierte F0-Kurve, Anzahl korrigierter Frames) zurueck.
    """
    f0 = f0.copy()
    voiced = f0 > 0
    n = len(f0)
    half = window // 2
    corrected = 0
    candidates = (2.0, 0.5, 3.0, 1.0 / 3.0, 4.0, 0.25)

    for i in range(n):
        if not voiced[i]:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        neighbor_idx = [j for j in range(lo, hi) if j != i and voiced[j]]
        if len(neighbor_idx) < 5:
            continue
        local_med = np.median(f0[neighbor_idx])
        ratio = f0[i] / local_med
        best = min(candidates, key=lambda c: abs(ratio - c) / c)
        rel_err = abs(ratio - best) / best
        if rel_err < tolerance and abs(np.log2(ratio)) > 0.4:
            f0[i] = f0[i] / best
            corrected += 1

    return f0, corrected


def limit_f0_slew_rate(f0, t, max_semitones_per_sec=500.0):
    """
    Tonhoehen-Geschwindigkeitsbegrenzung ("Slew-Rate-Limiter"): begrenzt, wie
    viele Halbtoene pro Sekunde sich die Tonhoehe ueberhaupt bewegen darf -
    grosszuegig oberhalb von allem, was eine menschliche Stimme auch bei
    sehr schnellem Vibrato, einem dramatischen Rutscher oder einem schnellen
    Lauf tatsaechlich schafft (Standard 500 Halbtoene/Sekunde - mehr als eine
    Oktave in rund 30ms). Jeder Uebergang, der schneller ist, wird automatisch auf
    diese Maximalgeschwindigkeit "ausgebremst" und damit zeitlich gestreckt,
    statt als harter, abrupter Sprung bestehen zu bleiben.

    Das ist die generischste und robusteste der vier Reparaturen hier: sie
    arbeitet durchgehend (nicht nur an erkannten "Plateau-Raendern" oder in
    vorher als "unruhig" eingestuften Passagen) und faengt dadurch auch
    Faelle ab, die aus mehreren, dicht aufeinanderfolgenden Einzelspruengen
    bestehen (ein schneller, mehrstufiger Lauf statt eines einzigen
    Sprungs) - das deckt genau das Symptom ab, das sich wie eine "kaputte"
    Welle/Vibration anhoert, weil die Stimme einen Ton nicht kontrolliert
    durchhaelt. An jeder stimmlosen Luecke (z.B. Atempause) wird bewusst neu
    angesetzt, weil nach einer Pause ein voellig neuer Ton einsetzen kann -
    das ist keine Tonhoehen-Verletzung, sondern ein neuer musikalischer
    Einsatz.

    Arbeitet in Halbtonschritten pro Sekunde, also unabhaengig von Stimmlage,
    Geschlecht und Genre - der Grenzwert orientiert sich an menschlicher
    Physiologie, nicht an diesem einen Song.

    Gibt (geschwindigkeitsbegrenzte F0-Kurve, Anzahl begrenzter Frames) zurueck.
    """
    f0 = f0.copy()
    n = len(f0)
    voiced = f0 > 0
    log_f0 = np.where(voiced, np.log2(np.where(voiced, f0, 1.0)), np.nan)
    out_log = log_f0.copy()

    limited = 0
    prev_log = None
    prev_t = None
    for i in range(n):
        if not voiced[i]:
            prev_log = None
            continue
        if prev_log is None:
            prev_log = out_log[i]
            prev_t = t[i]
            continue
        dt = t[i] - prev_t
        max_step = (max_semitones_per_sec * dt) / 12.0
        raw_step = out_log[i] - prev_log
        if abs(raw_step) > max_step:
            out_log[i] = prev_log + np.sign(raw_step) * max_step
            limited += 1
        prev_log = out_log[i]
        prev_t = t[i]

    f0_out = f0.copy()
    f0_out[voiced] = 2.0 ** out_log[voiced]
    return f0_out, limited


def repair_f0_dropouts(f0, t, baseline_ms=120.0, dev_semitones=2.0,
                        max_run_ms=80.0, revert_tol_semitones=1.0):
    """
    Findet zusammenhaengende Abschnitte, die stark von einer laengerfristigen
    Basislinie abweichen UND danach wieder (fast) zur selben Basislinie
    zurueckkehren - das Muster eines kurzen Tracking-Aussetzers, der sich
    selbst wieder "einfaengt". Reine Melodie-Bewegungen, die NICHT zur alten
    Basislinie zurueckkehren, sondern auf einem neuen Niveau bleiben, werden
    bewusst nicht angetastet (das deckt Mechanismus 2 bzw. 3 ab, falls es
    sich doch um einen echten Sprung/eine echte unruhige Passage handelt).
    Alle Schwellen sind in Halbtonschritten bzw. Millisekunden angegeben,
    nicht in Hz/Frames - damit das Ergebnis unabhaengig von Stimmlage und
    Analyse-Aufloesung ist.

    Gibt (reparierte F0-Kurve, Anzahl reparierter Frames) zurueck.
    """
    f0 = f0.copy()
    n = len(f0)
    voiced = f0 > 0
    log_f0 = np.where(voiced, np.log2(np.where(voiced, f0, 1.0)), np.nan)

    hop_s = float(np.median(np.diff(t))) if len(t) > 1 else 0.005
    baseline_frames = max(5, int(round(baseline_ms / 1000.0 / hop_s)))
    if baseline_frames % 2 == 0:
        baseline_frames += 1
    max_run_frames = max(1, int(round(max_run_ms / 1000.0 / hop_s)))
    half = baseline_frames // 2

    baseline = np.full(n, np.nan)
    for i in range(n):
        if not voiced[i]:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals = log_f0[lo:hi]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 3:
            baseline[i] = np.median(vals)

    dev = np.full(n, np.nan)
    valid = voiced & ~np.isnan(baseline)
    dev[valid] = (log_f0[valid] - baseline[valid]) * 12.0

    flagged = np.zeros(n, dtype=bool)
    flagged[valid] = np.abs(dev[valid]) > dev_semitones

    repaired = 0
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j < n and flagged[j]:
            j += 1
        run_len = j - i
        if run_len <= max_run_frames:
            before_idx = [k for k in range(max(0, i - 3), i) if voiced[k] and not flagged[k]]
            after_idx = [k for k in range(j, min(n, j + 3)) if voiced[k] and not flagged[k]]
            if before_idx and after_idx:
                b_med = np.median(log_f0[before_idx])
                a_med = np.median(log_f0[after_idx])
                if abs(b_med - a_med) * 12.0 < revert_tol_semitones:
                    b, a = before_idx[-1], after_idx[0]
                    for k in range(i, j):
                        frac = (k - b) / (a - b)
                        new_log = log_f0[b] * (1 - frac) + log_f0[a] * frac
                        f0[k] = 2.0 ** new_log
                        repaired += 1
        i = j

    return f0, repaired


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Pfad zur Eingangs-Vocal-Datei.")
    p.add_argument("--output", required=True, help="Zielpfad fuer die resynthetisierte Datei.")
    p.add_argument("--pitch_shift", type=float, default=0.0,
                    help="Optionale Tonhoehen-Verschiebung in Halbtoenen (Standard 0 = unveraendert).")
    p.add_argument("--mix", type=float, default=1.0,
                    help="Staerke der Resynthese (0.0-1.0, Standard 1.0 = komplett neu). "
                         "Niedriger mischt wieder Original-Klang dazu, falls der Effekt zu stark wirkt.")
    p.add_argument("--d4c_threshold", type=float, default=0.85,
                    help="D4C-Schwelle fuer den Rausch-/Atemanteil (Standard 0.85). Hoeher = "
                         "natuerlichere, luftigere Hoehen/S-Laute; niedriger = tonigere, klarere Hoehen.")
    p.add_argument("--f0_method", choices=["dio", "harvest"], default="dio",
                    help="Tonhoehen-Tracker: 'dio' (Standard, schnell) oder 'harvest' (deutlich "
                         "langsamer, aber robuster bei lauten/geschrienen/unruhigen Passagen, wo "
                         "dio leicht die Spur verliert - 'Welle'/Tonabriss-Artefakte).")
    p.add_argument("--f0_floor", type=float, default=None,
                    help="Manueller Experten-Override fuer die untere Grenze des Tonhoehen-"
                         "Suchbereichs in Hz. Normalerweise NICHT setzen - laeuft standardmaessig "
                         "automatisch kalibriert (siehe Docstring oben). Nur fuer Einzelfaelle.")
    p.add_argument("--f0_ceil", type=float, default=None,
                    help="Manueller Experten-Override fuer die obere Grenze des Tonhoehen-"
                         "Suchbereichs in Hz. Normalerweise NICHT setzen - laeuft standardmaessig "
                         "automatisch kalibriert (siehe Docstring oben). Nur fuer Einzelfaelle.")
    p.add_argument("--no_f0_glitch_repair", action="store_true",
                    help="Schaltet ALLE DREI automatischen Reparatur-Mechanismen ab (Oktav-"
                         "Korrektur, Geschwindigkeitsbegrenzung, Aussetzer-Reparatur). "
                         "Standardmaessig AN (siehe Docstring oben).")
    p.add_argument("--f0_glitch_baseline_ms", type=float, default=120.0,
                    help="Experten-Override (Aussetzer-Reparatur): Laenge der laufenden "
                         "Basislinie in Millisekunden (Standard 120ms).")
    p.add_argument("--f0_glitch_dev_semitones", type=float, default=2.0,
                    help="Experten-Override (Aussetzer-Reparatur): ab wie vielen Halbtoenen "
                         "Abweichung von der Basislinie ein Frame als moeglicher Aussetzer gilt "
                         "(Standard 2.0).")
    p.add_argument("--f0_glitch_max_run_ms", type=float, default=80.0,
                    help="Experten-Override (Aussetzer-Reparatur): laengster Zeitraum in "
                         "Millisekunden, der noch als 'kurzer Aussetzer' gilt (Standard 80ms).")
    p.add_argument("--f0_max_semitones_per_sec", type=float, default=500.0,
                    help="Experten-Override (Geschwindigkeitsbegrenzung): wie viele Halbtoene "
                         "pro Sekunde die Tonhoehe sich maximal bewegen darf (Standard 500.0 - "
                         "oberhalb jedes natuerlichen Vibratos/Rutschers, aber kurz genug, dass "
                         "selbst sehr grosse Spruenge als schneller Ansatz statt als hoerbares "
                         "Gleiten/Sirenen-Rutschen klingen). Niedriger = laengeres, weicheres "
                         "Gleiten bei grossen Spruengen; hoeher = kuerzeres Gleiten, aber groesserer "
                         "Rest-Sprung pro Analyse-Frame.")
    args = p.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        print("FEHLER: 'soundfile' fehlt in dieser Umgebung (pip install soundfile).", flush=True)
        return 1
    try:
        import pyworld as pw
    except ImportError:
        print("FEHLER: 'pyworld' fehlt in dieser Umgebung (pip install pyworld).", flush=True)
        return 1

    print(f"Lade: {args.input}", flush=True)
    audio, sr = sf.read(args.input, always_2d=True)
    mono = np.mean(audio, axis=1).astype(np.float64)
    print(f"  {mono.shape[0]} Samples, {sr} Hz, {mono.shape[0] / sr:.1f}s "
          f"(Eingang war {audio.shape[1]}-kanalig, fuer die Analyse zu Mono zusammengefasst)", flush=True)

    f0_floor, f0_ceil = args.f0_floor, args.f0_ceil
    if f0_floor is None or f0_ceil is None:
        print("Kalibriere Tonhoehen-Suchbereich automatisch auf diese Stimme/diesen Song "
              "(Erstdurchlauf 50-1100 Hz)...", flush=True)
        auto_floor, auto_ceil, ok = auto_calibrate_f0_range(mono, sr, args.f0_method, pw)
        if f0_floor is None:
            f0_floor = auto_floor
        if f0_ceil is None:
            f0_ceil = auto_ceil
        if ok:
            print(f"  Automatisch kalibriert: {f0_floor:.0f}-{f0_ceil:.0f} Hz "
                  f"(aus 5./95. Perzentil der erkannten Tonhoehe dieses Songs)", flush=True)
        else:
            print(f"  Kalibrierung unsicher (zu wenig stimmhaftes Material) - "
                  f"Rueckfall auf breiten Standardbereich {f0_floor:.0f}-{f0_ceil:.0f} Hz", flush=True)

    if args.f0_method == "harvest":
        print(f"Extrahiere Tonhoehe (F0) ueber Harvest (robuster, langsamer), "
              f"Suchbereich {f0_floor:.0f}-{f0_ceil:.0f} Hz...", flush=True)
        f0, t = pw.harvest(mono, sr, f0_floor=f0_floor, f0_ceil=f0_ceil)
        f0 = pw.stonemask(mono, f0, t, sr)
    else:
        print(f"Extrahiere Tonhoehe (F0) ueber Dio + StoneMask-Verfeinerung, "
              f"Suchbereich {f0_floor:.0f}-{f0_ceil:.0f} Hz...", flush=True)
        raw_f0, t = pw.dio(mono, sr, f0_floor=f0_floor, f0_ceil=f0_ceil)
        f0 = pw.stonemask(mono, raw_f0, t, sr)

    if not args.no_f0_glitch_repair:
        f0, n_octave = octave_correct_f0(f0)
        if n_octave:
            print(f"  1) Oktav-Fehler-Korrektur: {n_octave} Frame(s) in die richtige Oktave zurueckgezogen",
                  flush=True)

        f0, n_slew = limit_f0_slew_rate(
            f0, t, max_semitones_per_sec=args.f0_max_semitones_per_sec,
        )
        if n_slew:
            print(f"  2) Geschwindigkeitsbegrenzung: {n_slew} Frame(s) (schneller als "
                  f"{args.f0_max_semitones_per_sec:.0f} Halbtoene/Sekunde) auf eine plausible "
                  f"Geschwindigkeit ausgebremst", flush=True)

        f0, n_dropout = repair_f0_dropouts(
            f0, t,
            baseline_ms=args.f0_glitch_baseline_ms,
            dev_semitones=args.f0_glitch_dev_semitones,
            max_run_ms=args.f0_glitch_max_run_ms,
        )
        if n_dropout:
            print(f"  3) Aussetzer-Reparatur: {n_dropout} Frame(s) (kurze Tracking-Aussetzer, "
                  f"die zur vorherigen Tonlage zurueckkehren) glattgezogen", flush=True)

        if not any([n_octave, n_slew, n_dropout]):
            print("  Oktav-/Geschwindigkeits-/Aussetzer-Pruefung: keine auffaelligen Stellen gefunden",
                  flush=True)

    if args.pitch_shift != 0.0:
        factor = 2.0 ** (args.pitch_shift / 12.0)
        f0 = f0 * factor
        print(f"  Tonhoehe verschoben: Faktor {factor:.4f} ({args.pitch_shift} Halbtoene)", flush=True)

    print(f"Extrahiere Spektralhuelle (CheapTrick) und Aperiodizitaet (D4C, threshold={args.d4c_threshold})...",
          flush=True)
    sp = pw.cheaptrick(mono, f0, t, sr)
    ap = pw.d4c(mono, f0, t, sr, threshold=args.d4c_threshold)

    print("Synthese: baue EINE kohaerente Stimme aus F0 + Huelle + Aperiodizitaet neu auf...", flush=True)
    y = pw.synthesize(f0, sp, ap, sr)

    mix = max(0.0, min(1.0, args.mix))
    if mix < 1.0:
        n = min(len(y), len(mono))
        y = mix * y[:n] + (1.0 - mix) * mono[:n]
        print(f"  Mix angewendet: {mix:.2f} Resynthese / {1.0 - mix:.2f} Original", flush=True)

    sf.write(args.output, y, sr)
    print(f"Fertig -> {args.output}", flush=True)
    print("Bitte reinhoeren: klingt die Stimme jetzt wie EINE durchgehende Stimme statt", flush=True)
    print("mehrerer gleichzeitiger/gestapelter Toene? Bleibt die Melodie/Tonhoehe noch", flush=True)
    print("sauber nachvollziehbar, oder klingt es jetzt zu 'vocoder-haft'/kuenstlich?", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
