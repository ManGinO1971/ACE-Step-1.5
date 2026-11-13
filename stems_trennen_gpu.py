#!/usr/bin/env python3
"""HELTECH Stems – Trennen auf der GPU für die Mac-App (11. Nov 2026).

Die Mac-App rechnet mit Schalter "GPU" alle Trennungen hier (sonst auf dem Mac). Gleiche Modelle wie auf dem Mac,
damit das Ergebnis gleich bleibt – nur schneller. Das Neu-Einspielen mit den eigenen Klängen bleibt auf dem Mac.

Aufruf: python stems_trennen_gpu.py <song> <ausgabeordner> '<auftrag-json>' [--cache <ordner>]
--cache: Ergebnis-Speicher je Song auf dem Volume (der Song liegt dort nach einmaligem Hochladen). Schon gerechnete
Aufträge kommen sofort von dort (FLAC, verlustfrei); beim 53er-Modell je Instrument einzeln, die Erkennung legt ihre
6 Spuren (htdemucs_6s) gleich mit ab. Gilt für jeden Song – der Speicher hängt nur am Inhalt der Datei.
Aufträge:
  {"art": "erkennen"}                                                   Instrumenten-Erkennung (6 Spuren + Hör-Modell)
  {"art": "sep", "modell": "BS-Roformer-SW.ckpt"}                       audio-separator, alle Spuren
  {"art": "sep", "modell": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
   "dann": {"tag": "vocals", "modell": "UVR-BVE-4B_SN-44100-1.pth", "praefix": "k_"}}   zweite Stufe auf einer Spur
  {"art": "demucs", "modell": "htdemucs_6s", "two_stems": "vocals"}      Demucs (two_stems optional)
  {"art": "s53", "instrumente": ["percussion", "congas"]}               53-Instrumente-Modell
Ausgabe: <name>.wav je Spur (name = Spur-Kennung in Kleinbuchstaben, z. B. drums, bass, vocals, instrumental).
Letzte stdout-Zeile: JSON {"ok": true, "spuren": {name: datei}, "sekunden": …, "geraet": …}
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UVR_DIR = os.environ.get("HIELTECH_UVR_DIR") or (
    "/runpod-volume/hieltech_uvr_models" if os.path.isdir("/runpod-volume") else os.path.join(HERE, "uvr_models"))
DEMUCS_MODELLE = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q", "demucs", "mdx", "mdx_q"}
NAME_OK = re.compile(r"^[A-Za-z0-9_.\-]{3,120}$")


def geraet():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def tag_von(datei):
    tags = re.findall(r"\(([^)]+)\)", os.path.basename(datei))
    return re.sub(r"[^a-z0-9]+", "_", tags[-1].lower()).strip("_") if tags else os.path.splitext(os.path.basename(datei))[0].lower()


def alle_spuren_sichern(sep_obj, eingabe, roh, dateien):
    """Manche Fassungen von audio-separator speichern bei Modellen mit mehr als zwei Spuren (z. B. DrumSep: Kick, Snare,
    Toms, Hi-Hat, Ride, Crash) nur die ersten zwei. Das Modell liefert intern aber alle – dann werden sie hier direkt
    abgeholt und gespeichert (auf RunPod so erprobt in stems_drums_step.py). Rückgabe: Liste aller Dateien."""
    import numpy as np
    import soundfile as sf
    try:
        mi = sep_obj.model_instance
        cfg = getattr(mi, "model_data_cfgdict", None)
        namen = [str(n) for n in cfg.training.instruments]
    except Exception:  # noqa: BLE001 – Modell ohne Spuren-Liste (z. B. VR-/Demucs-Modelle): nichts zu tun
        return dateien
    da = " ".join(os.path.basename(f).lower() for f in dateien)
    if len(namen) <= 2 or all(f"({n.lower()})" in da for n in namen):
        return dateien
    mi.audio_file_path = eingabe
    mi.audio_file_base = os.path.splitext(os.path.basename(eingabe))[0]
    quellen = mi.demix(mix=mi.prepare_mix(eingabe))
    if not isinstance(quellen, dict):
        return dateien
    sr = int(getattr(mi, "sample_rate", 44100) or 44100)
    neu = []
    for n in namen:
        if n not in quellen:
            continue
        y = np.asarray(quellen[n], dtype=np.float32)
        y = y.T if y.ndim == 2 and y.shape[0] == 2 else y
        spitze = float(np.max(np.abs(y))) if y.size else 0.0
        if spitze > 0.9:                                   # wie audio-separator: nur zu laute Spuren absenken
            y = y * (0.9 / spitze)
        p = os.path.join(roh, f"{mi.audio_file_base}_({n})_direkt.wav")
        sf.write(p, y, sr, subtype="PCM_24")
        neu.append(p)
    return neu or dateien


def sep(eingabe, modell, roh):
    from audio_separator.separator import Separator
    import logging
    os.makedirs(UVR_DIR, exist_ok=True)
    os.makedirs(roh, exist_ok=True)
    s = Separator(output_dir=roh, model_file_dir=UVR_DIR, output_format="WAV", log_level=logging.WARNING)
    s.load_model(model_filename=modell)
    dateien = [f if os.path.isabs(f) else os.path.join(roh, f) for f in s.separate(eingabe)]
    return alle_spuren_sichern(s, eingabe, roh, dateien)


def als_flac(wav, ziel):
    """WAV -> FLAC (verlustfrei, 24 Bit). Rückgabe: Pfad der FLAC-Datei."""
    try:
        import soundfile as sf
        y, sr = sf.read(wav, always_2d=True)
        sf.write(ziel, y, sr, format="FLAC", subtype="PCM_24")
    except Exception:  # noqa: BLE001
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "flac", ziel], check=True, timeout=900)
    return ziel


SPEICHER_FASSUNG = "2"      # erhöhen, wenn alte Einträge im Ergebnis-Speicher nicht mehr gelten sollen
                            # (2: Mehrspur-Modelle wie DrumSep liefern jetzt alle Spuren statt nur zwei)


def schluessel(auftrag):
    fassung = SPEICHER_FASSUNG if auftrag.get("art") == "sep" else ""
    if auftrag.get("art") == "erkennen":
        fassung = "regeln2"          # Erkennung mit Genre/Familien/Stufen (12. Nov 2026): alte Ergebnisse gelten nicht mehr
    return hashlib.sha1((json.dumps(auftrag, sort_keys=True) + fassung).encode()).hexdigest()[:16]


def aus_speicher(cache, key):
    """(spuren, extra) aus dem Ergebnis-Speicher oder None."""
    if not cache:
        return None
    d = os.path.join(cache, key)
    try:
        info = json.load(open(os.path.join(d, "fertig.json"), encoding="utf-8"))
        spuren = {n: os.path.join(d, f) for n, f in info["spuren"].items()}
        if all(os.path.isfile(p) for p in spuren.values()):
            os.utime(d, None)
            return spuren, info.get("extra") or {}
    except (OSError, ValueError, KeyError):
        pass
    return None


def in_speicher(cache, key, spuren, extra):
    """Legt die Spuren als FLAC im Ergebnis-Speicher ab; Rückgabe: Spuren mit den Pfaden dort."""
    if not cache:
        return spuren
    d = os.path.join(cache, key)
    tmp = d + f".tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    namen = {}
    for n, p in spuren.items():
        als_flac(p, os.path.join(tmp, f"{n}.flac"))
        namen[n] = f"{n}.flac"
    with open(os.path.join(tmp, "fertig.json"), "w", encoding="utf-8") as f:
        json.dump({"spuren": namen, "extra": extra}, f, ensure_ascii=False)
    shutil.rmtree(d, ignore_errors=True)
    os.replace(tmp, d)
    return {n: os.path.join(d, f) for n, f in namen.items()}


def rechnen(song, aus, auftrag, cache=None):
    """Ein Auftrag. Rückgabe (spuren, extra). Mit cache landen Nebenergebnisse (6 Spuren der Erkennung) gleich im Speicher."""
    spuren = {}
    extra = {}
    art = auftrag.get("art")
    if art == "sep":
        modell = auftrag.get("modell", "")
        if not NAME_OK.match(modell):
            raise ValueError("ungültiges Modell")
        for f in sep(song, modell, os.path.join(aus, "_roh1")):
            name = tag_von(f)
            ziel = os.path.join(aus, f"{name}.wav")
            shutil.move(f, ziel)
            spuren[name] = ziel
        dann = auftrag.get("dann") or {}
        if dann:
            m2, tag, pre = dann.get("modell", ""), dann.get("tag", ""), re.sub(r"[^a-z_]", "", dann.get("praefix", "k_"))
            if not NAME_OK.match(m2) or tag not in spuren:
                raise ValueError(f"zweite Stufe: Modell oder Spur '{tag}' fehlt ({list(spuren)})")
            for f in sep(spuren[tag], m2, os.path.join(aus, "_roh2")):
                name = pre + tag_von(f)
                ziel = os.path.join(aus, f"{name}.wav")
                shutil.move(f, ziel)
                spuren[name] = ziel
    elif art == "demucs":
        modell = auftrag.get("modell", "htdemucs")
        if modell not in DEMUCS_MODELLE:
            raise ValueError("unbekanntes Demucs-Modell")
        roh = os.path.join(aus, "_demucs")
        cmd = [sys.executable, "-m", "demucs", "--name", modell, "--out", roh, "-d", geraet()]
        two = auftrag.get("two_stems")
        if two:
            if not re.match(r"^[a-z]{3,12}$", two):
                raise ValueError("two_stems ungültig")
            cmd += ["--two-stems", two]
        r = subprocess.run(cmd + [song], capture_output=True, text=True, timeout=3600)
        d = os.path.join(roh, modell, os.path.splitext(os.path.basename(song))[0])
        if not os.path.isdir(d):
            raise RuntimeError("Demucs: " + (r.stderr or "")[-500:])
        for f in sorted(os.listdir(d)):
            if f.endswith(".wav"):
                ziel = os.path.join(aus, f)
                shutil.move(os.path.join(d, f), ziel)
                spuren[os.path.splitext(f)[0]] = ziel
    elif art == "s53":
        namen = [i for i in auftrag.get("instrumente") or [] if re.match(r"^[a-z_-]{2,20}$", i)][:53]
        if not namen:
            raise ValueError("keine Instrumente")
        args = [sys.executable, os.path.join(HERE, "stems53_ref_step.py"), song]
        for i in namen:
            args += [os.path.join(aus, f"{i}.wav"), i]
        r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        for i in namen:
            p = os.path.join(aus, f"{i}.wav")
            if os.path.isfile(p):
                spuren[i] = p
        if not spuren:
            raise RuntimeError("53er-Modell: " + (r.stderr or r.stdout or "")[-500:])
    elif art == "erkennen":
        # Instrumenten-Erkennung komplett auf der GPU: 6 Spuren (htdemucs_6s) + Hör-Modell (EfficientAT) je Spur
        roh = os.path.join(aus, "_demucs")
        zeiten = {}
        d = None
        fertig = aus_speicher(cache, schluessel({"art": "demucs", "modell": "htdemucs_6s"}))
        if fertig:                                   # 6 Spuren liegen schon im Speicher -> nicht noch einmal trennen
            d = os.path.join(aus, "_sechs")
            os.makedirs(d, exist_ok=True)
            import soundfile as sf
            for n, p in fertig[0].items():
                y, sr = sf.read(p, always_2d=True)
                sf.write(os.path.join(d, f"{n}.wav"), y, sr, subtype="PCM_16")
        else:
            tt = time.time()
            r = subprocess.run([sys.executable, "-m", "demucs", "--name", "htdemucs_6s", "--out", roh, "-d", geraet(), song],
                               capture_output=True, text=True, timeout=3600)
            d = os.path.join(roh, "htdemucs_6s", os.path.splitext(os.path.basename(song))[0])
            zeiten["trennen_6_spuren"] = round(time.time() - tt, 1)
        tt = time.time()
        args = [sys.executable, os.path.join(HERE, "hieltech_erkennung", "detect_instruments_eat.py"), song,
                os.path.join(aus, "erkennung.json"), "--hoerproben", os.path.join(aus, "_hp")]
        if os.path.isdir(d):
            args += ["--spuren-ordner", d]
        r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        ej = os.path.join(aus, "erkennung.json")
        if not os.path.isfile(ej):
            raise RuntimeError("Erkennung: " + (r.stderr or r.stdout or "")[-500:])
        erkennung = json.load(open(ej, encoding="utf-8"))
        try:                                         # Zeitleisten + Genre: damit entscheidet die App bei anderer Genre-Wahl neu
            rj = json.load(open(ej + ".roh.json", encoding="utf-8"))
            extra["erkennung_roh"], extra["erkennung_info"] = rj.get("roh"), rj.get("info")
        except (OSError, ValueError):
            pass
        zeiten["erkennen"] = round(time.time() - tt, 1)
        for z in (r.stdout or "").splitlines():
            if z.startswith("ZEITEN "):
                try:
                    zeiten["hoer_modell"] = json.loads(z[7:])
                except ValueError:
                    pass
        extra["zeiten"] = zeiten
        for e in erkennung:
            hp = e.pop("hoerprobe", None)
            if hp and os.path.isfile(hp):
                name = "hp_" + re.sub(r"[^a-z0-9_-]", "", e["instrument"])
                ziel = os.path.join(aus, f"{name}.wav")
                shutil.move(hp, ziel)
                spuren[name] = ziel
                e["hoerprobe_spur"] = name
        extra["erkennung"] = erkennung
        extra["mit_spuren"] = os.path.isdir(d)
        if cache and os.path.isdir(d) and not fertig:
            try:
                sechs = {os.path.splitext(f)[0]: os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".wav")}
                if sechs:
                    in_speicher(cache, schluessel({"art": "demucs", "modell": "htdemucs_6s"}), sechs, {})
            except Exception:  # noqa: BLE001 – nur ein Nebenergebnis
                pass
        for x in ("_hp", "_sechs", "erkennung.json", "erkennung.json.roh.json"):
            p = os.path.join(aus, x)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.path.isfile(p) and os.remove(p))
    else:
        raise ValueError(f"unbekannte Art: {art}")
    return spuren, extra


def s53_mit_speicher(song, aus, auftrag, cache):
    """53er-Modell: je Instrument ein eigener Speicher-Eintrag – nur fehlende Instrumente werden gerechnet."""
    namen = [i for i in auftrag.get("instrumente") or [] if re.match(r"^[a-z_-]{2,20}$", i)][:53]
    if not namen:
        raise ValueError("keine Instrumente")
    spuren, fehlt = {}, []
    for i in namen:
        da = aus_speicher(cache, f"s53_{i}")
        if da:
            spuren.update(da[0])
        else:
            fehlt.append(i)
    if fehlt:
        neu, _ = rechnen(song, aus, {"art": "s53", "instrumente": fehlt})
        for i, p in neu.items():
            spuren.update(in_speicher(cache, f"s53_{i}", {i: p}, {}))
    return spuren, {"aus_speicher": [i for i in namen if i not in fehlt]}


def main():
    argv = list(sys.argv[1:])
    cache = None
    if "--cache" in argv:
        k = argv.index("--cache")
        cache = argv[k + 1]
        del argv[k:k + 2]
        os.makedirs(cache, exist_ok=True)
    song, aus, auftrag = argv[0], argv[1], json.loads(argv[2])
    os.makedirs(aus, exist_ok=True)
    t0 = time.time()
    if cache and auftrag.get("art") == "s53":
        spuren, extra = s53_mit_speicher(song, aus, auftrag, cache)
    else:
        key = schluessel(auftrag)
        da = aus_speicher(cache, key)
        if da:
            spuren, extra = da
            extra = dict(extra, aus_speicher=True)
        else:
            spuren, extra = rechnen(song, aus, auftrag, cache)
            spuren = in_speicher(cache, key, spuren, extra)
    for d in ("_roh1", "_roh2", "_demucs"):
        shutil.rmtree(os.path.join(aus, d), ignore_errors=True)
    print(json.dumps({"ok": True, "spuren": spuren, "extra": extra, "sekunden": round(time.time() - t0, 1), "geraet": geraet()},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "trennen_fehlgeschlagen", "message": str(e)[:800]}), flush=True)
        sys.exit(1)
