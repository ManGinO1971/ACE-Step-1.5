#!/usr/bin/env python3
"""HELTECH Stems – Trennen auf der GPU für die Mac-App (11. Nov 2026).

Die Mac-App rechnet mit Schalter "GPU" alle Trennungen hier (sonst auf dem Mac). Gleiche Modelle wie auf dem Mac,
damit das Ergebnis gleich bleibt – nur schneller. Das Neu-Einspielen mit den eigenen Klängen bleibt auf dem Mac.

Aufruf: python stems_trennen_gpu.py <song> <ausgabeordner> '<auftrag-json>'
Aufträge:
  {"art": "sep", "modell": "BS-Roformer-SW.ckpt"}                       audio-separator, alle Spuren
  {"art": "sep", "modell": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
   "dann": {"tag": "vocals", "modell": "UVR-BVE-4B_SN-44100-1.pth", "praefix": "k_"}}   zweite Stufe auf einer Spur
  {"art": "demucs", "modell": "htdemucs_6s", "two_stems": "vocals"}      Demucs (two_stems optional)
  {"art": "s53", "instrumente": ["percussion", "congas"]}               53-Instrumente-Modell
Ausgabe: <name>.wav je Spur (name = Spur-Kennung in Kleinbuchstaben, z. B. drums, bass, vocals, instrumental).
Letzte stdout-Zeile: JSON {"ok": true, "spuren": {name: datei}, "sekunden": …, "geraet": …}
"""
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


def sep(eingabe, modell, roh):
    from audio_separator.separator import Separator
    import logging
    os.makedirs(UVR_DIR, exist_ok=True)
    os.makedirs(roh, exist_ok=True)
    s = Separator(output_dir=roh, model_file_dir=UVR_DIR, output_format="WAV", log_level=logging.WARNING)
    s.load_model(model_filename=modell)
    return [f if os.path.isabs(f) else os.path.join(roh, f) for f in s.separate(eingabe)]


def main():
    song, aus, auftrag = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    os.makedirs(aus, exist_ok=True)
    t0 = time.time()
    spuren = {}
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
        namen = [i for i in auftrag.get("instrumente") or [] if re.match(r"^[a-z_-]{2,20}$", i)][:12]
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
    else:
        raise ValueError(f"unbekannte Art: {art}")
    for d in ("_roh1", "_roh2", "_demucs"):
        shutil.rmtree(os.path.join(aus, d), ignore_errors=True)
    print(json.dumps({"ok": True, "spuren": spuren, "sekunden": round(time.time() - t0, 1), "geraet": geraet()}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "trennen_fehlgeschlagen", "message": str(e)[:800]}), flush=True)
        sys.exit(1)
