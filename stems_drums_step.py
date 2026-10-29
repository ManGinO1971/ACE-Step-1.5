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
Ausgabe: drums_1_ohne_master.wav, drums_2_master_sanft.wav, drums_3_master_kraeftig.wav (Schlagzeug komplett)
         + drums_teile/<kick|snare|toms|hihat|becken>.wav (dieselben Teile, gleich behandelt, ohne Bus-Master).
  4. Je Teil eigene Kompression, dann Schlagzeug-Bus-Master: Parallel-Kompression + Bus-Kompressor + Limiter. Letzte stdout-Zeile: JSON-Bericht.
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


SCHAERFE = float(os.environ.get("HIELTECH_DRUMSEP_SCHAERFE", "3.0"))


def nachsortieren(roh):
    """Test 6: in der Snare hingen Hi-Hat und teils Toms. Jede Zeit-Frequenz-Zelle wird dem Teil
    zugeordnet, das dort am stärksten ist (weiche Maske mit Schärfe 3) – Summe bleibt gleich,
    nur falsch einsortierte Reste wandern ins richtige Teil."""
    import librosa
    keys = list(roh)
    n = roh[keys[0]].shape[0]
    out = {key: np.zeros_like(roh[key]) for key in keys}
    for ch in range(2):
        S = np.stack([librosa.stft(roh[key][:, ch], n_fft=2048, hop_length=512) for key in keys])
        mix = S.sum(0)
        w = np.abs(S) ** SCHAERFE
        m = w / (w.sum(0, keepdims=True) + 1e-12)
        for i, key in enumerate(keys):
            out[key][:, ch] = librosa.istft(m[i] * mix, hop_length=512, length=n)
    return out


def drumsep_roh(path, work):
    """DrumSep einmal -> {kick, snare, toms, hh, ride, crash}: (n, 2)."""
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
    return {key: fit(a, n) for key, a in roh.items()}


SNARE_DURCHGAENGE = int(os.environ.get("HIELTECH_SNARE_DURCHGAENGE", "2"))


def zerlegen(path, work):
    """-> {kick, snare, toms, hihat, becken}: (n, 2) bei 44,1 kHz.
    Test 7 (Hörtest): in der Snare-Spur waren Hi-Hat, Toms und Becken noch deutlich zu hören.
    -> Snare-Spur ein zweites (und drittes) Mal durch DrumSep: nur der Snare-Anteil bleibt in der Snare,
       Hi-Hat/Toms/Becken/Kick daraus werden NICHT weggeworfen, sondern ihrer eigenen Spur zugeschlagen."""
    roh = drumsep_roh(path, work)
    n = roh["snare"].shape[0]
    for i in range(SNARE_DURCHGAENGE):
        sp = os.path.join(work, f"snare_runde{i}_{os.path.splitext(os.path.basename(path))[0]}.wav")
        sf.write(sp, roh["snare"], SR, subtype="FLOAT")
        try:
            r = drumsep_roh(sp, work)
        except Exception as e:  # noqa: BLE001
            print(f"[Stems-Test7] Snare-Reinigung Runde {i + 1} übersprungen: {e}", file=sys.stderr, flush=True)
            break
        roh["snare"] = fit(r["snare"], n)
        for key in ("kick", "toms", "hh", "ride", "crash"):
            if key in r:
                roh[key] = roh.get(key, np.zeros((n, 2), np.float32)) + fit(r[key], n)
    roh = nachsortieren(roh)
    becken = roh.get("ride", np.zeros((n, 2), np.float32)) + roh.get("crash", np.zeros((n, 2), np.float32))
    return {"kick": roh["kick"], "snare": roh["snare"], "toms": roh["toms"], "hihat": roh["hh"], "becken": becken}


BECKEN_AUSKLANG = float(os.environ.get("HIELTECH_BECKEN_AUSKLANG", "1.6"))
KOMPRESSION = os.environ.get("HIELTECH_DRUMS_KOMPRESSION", "0") == "1"  # Hörtest 29.10.: erst ganz am Ende
# Je Teil eigene Kompression (wie im Studio: jedes Teil auf eigenem Kanal) – Tiefen-Filter gegen Rumpeln,
# Attack so lang, dass der Anschlag durchkommt.
TEIL_KETTE = {
    "kick": "highpass=f=30,acompressor=threshold=-20dB:ratio=4:attack=20:release=120:makeup=2",
    "snare": "highpass=f=80,acompressor=threshold=-20dB:ratio=4:attack=8:release=100:makeup=2",
    "toms": "highpass=f=50,acompressor=threshold=-20dB:ratio=3:attack=15:release=150:makeup=1.5",
    "hihat": "highpass=f=300,acompressor=threshold=-22dB:ratio=2:attack=5:release=80:makeup=1.2",
    "becken": "highpass=f=250,acompressor=threshold=-24dB:ratio=1.5:attack=10:release=200",
}
# Schlagzeug-Bus-Master: Parallel-Kompression (New York) + Bus-Kompressor ("Kleber") + Limiter (-1 dB).
MASTER = {
    "sanft": ("0.25", "threshold=-18dB:ratio=2:attack=30:release=150:makeup=1.5", "2dB"),
    "kraeftig": ("0.45", "threshold=-20dB:ratio=3:attack=20:release=120:makeup=2", "4dB"),
}


def ffmpeg_kette(src, dst, kette=None, komplex=None):
    import subprocess
    args = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
    args += ["-filter_complex", komplex, "-map", "[m]"] if komplex else ["-af", kette]
    r = subprocess.run(args + ["-c:a", "pcm_f32le", dst], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + (r.stderr or "")[-200:])
    a, _ = sf.read(dst, dtype="float32", always_2d=True)
    return a


def master(src, dst, art):
    nyc, bus, gain = MASTER[art]
    komplex = (f"[0:a]highpass=f=25,asplit=2[d][w];[w]acompressor=threshold=-30dB:ratio=8:attack=1:release=60:makeup=4[c];"
               f"[d][c]amix=inputs=2:weights='1 {nyc}':normalize=0,acompressor={bus},volume={gain},"
               f"alimiter=limit=0.89:attack=5:release=50:level=0[m]")
    return ffmpeg_kette(src, dst, komplex=komplex)


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
        teile = {}
        for teil in TEILE:
            ti = {"kandidaten": []}
            try:
                o = fit(orig[teil], n)
                if float(np.abs(o).max()) < 1e-3:
                    ti["hinweis"] = "im Original praktisch nicht vorhanden"
                best, best_s = None, -9.0
                for name, parts in neu:
                    c = fit(parts[teil], n)
                    sc = score(o, c, "drums")
                    ti["kandidaten"].append({"datei": name, "punkte": sc[0]})
                    if sc[0] > best_s:
                        best, best_s, ti["gewaehlt"] = c, sc[0], name
                mg, anteil, _nr = k.maske_gate(o, best, teil)
                ti["klingt_anteil"] = round(anteil, 3)
                try:
                    b = k.neu_aufbauen(mg, False)
                except Exception as e:  # noqa: BLE001
                    ti["vocoder_fehler"] = str(e)[:200]
                    b = mg
                if teil == "becken":
                    # Hörtest 29.10.: "Becken NEU lang genau was ich brauche" – Ausklang halten, Faktor 1,6
                    try:
                        from becken_halt import halten
                        b = np.stack([halten(b[:, c].astype(np.float32), SR, BECKEN_AUSKLANG) for c in range(b.shape[1])], 1)
                        ti["ausklang_gehalten"] = BECKEN_AUSKLANG
                    except Exception as e:  # noqa: BLE001
                        ti["ausklang_fehler"] = str(e)[:200]
                bp = os.path.join(work, f"teil_{teil}.wav")
                sf.write(bp, b.astype(np.float32), SR, subtype="FLOAT")
                if KOMPRESSION:
                    try:
                        b = fit(ffmpeg_kette(bp, os.path.join(work, f"teil_{teil}_k.wav"), TEIL_KETTE[teil]), n)
                    except Exception as e:  # noqa: BLE001
                        ti["kompressor_fehler"] = str(e)[:200]
                teile[teil] = b.astype(np.float32)
                gesamt += teile[teil]
            except Exception as e:  # noqa: BLE001
                ti["fehler"] = str(e)[:300]
            info["teile"][teil] = ti
            print(f"[Stems-Test6] {teil}: {json.dumps(ti, ensure_ascii=False)}", file=sys.stderr, flush=True)
        # Ordner: Schlagzeug komplett + Unterordner mit allen Teilen (gleich behandelt, gleiche Balance)
        ordner = os.path.join(out, "drums_teile")
        os.makedirs(ordner, exist_ok=True)
        spitze = max([float(np.abs(gesamt).max())] + [float(np.abs(a).max()) for a in teile.values()] + [1e-9])
        fak = min(1.0, 0.98 / spitze)
        for teil, a in teile.items():
            sf.write(os.path.join(ordner, f"{teil}.wav"), a * fak, SR, subtype="PCM_24")
        sf.write(os.path.join(out, "drums_1_komplett.wav"), gesamt * fak, SR, subtype="PCM_24")
        gp = os.path.join(work, "gesamt.wav")
        sf.write(gp, gesamt, SR, subtype="FLOAT")
        for i, art in enumerate(("sanft", "kraeftig") if KOMPRESSION else (), start=2):
            try:
                m = fit(master(gp, os.path.join(work, f"master_{art}.wav"), art), n)
                sf.write(os.path.join(out, f"drums_{i}_master_{art}.wav"), m, SR, subtype="PCM_24")
            except Exception as e:  # noqa: BLE001
                info[f"master_{art}_fehler"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001
        info["fehler"] = str(e)[:300]
    bericht["instrumente"]["drums"] = info
    print(json.dumps(bericht, ensure_ascii=False))


if __name__ == "__main__":
    main()
