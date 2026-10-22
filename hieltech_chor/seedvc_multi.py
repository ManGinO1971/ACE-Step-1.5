#!/usr/bin/env python3
"""
Background-Chor (NEU, 23. Okt 2026) – Seed-VC singt die Vorlage mit mehreren Stimmen.

Läuft in der eigenen Seed-VC-Umgebung (/opt/seedvc_venv, siehe Dockerfile.runpod),
im Seed-VC-Ordner (/opt/seed-vc, mit seedvc_hieltech_patch.py angepasst).
Die Modelle werden EINMAL geladen und für alle Stimmen benutzt.

Einstellungen wie im Hörtest des Inhabers (22. Okt 2026, Root Nah Bow):
  Gesangsmodell 44,1 kHz mit Tonhöhe, 50 Schritte, cfg 0.7, Stimmprobe 15 s,
  keine automatische Tonlagen-Anpassung. Je Stimme: Lage (terz/original/...),
  Oktave, Ton-Halt – über die HIELTECH_SVC_*-Variablen des Patches.

Aufruf: python seedvc_multi.py <auftrag.json>
  auftrag.json: {"source": wav, "out_dir": ordner, "tonart": "a-moll", "steps": 50,
                 "voices": [{"name": "frau2", "ref": wav, "lage": "terz", "oktave": 0, "halt": 0.3}, ...]}
Erzeugt je Stimme <name>.wav und <name>_ziel.npy (Zielmelodie, 10-ms-Raster).
Letzte stdout-Zeile: JSON-Bericht.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

SEEDVC_DIR = os.environ.get("HIELTECH_SEEDVC_DIR", "/opt/seed-vc")


def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    out_dir = os.path.abspath(cfg["out_dir"])
    source = os.path.abspath(cfg["source"])
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(SEEDVC_DIR)
    sys.path.insert(0, SEEDVC_DIR)
    import inference  # noqa: E402 - erst nach chdir (relative Modellpfade)

    cache = {}
    orig_load = inference.load_models

    def cached_load(args):
        if "m" not in cache:
            cache["m"] = orig_load(args)
        else:
            inference.fp16 = args.fp16
        return cache["m"]

    inference.load_models = cached_load
    report = {"ok": True, "voices": []}
    for v in cfg["voices"]:
        name = v["name"]
        t0 = time.time()
        os.environ["HIELTECH_SVC_LAGE"] = str(v.get("lage", "original"))
        os.environ["HIELTECH_SVC_OKTAVE"] = str(int(v.get("oktave", 0)))
        os.environ["HIELTECH_SVC_HALT"] = str(float(v.get("halt", 0.3)))
        os.environ["HIELTECH_SVC_TONART"] = str(cfg.get("tonart") or "a-moll")
        ziel = os.path.join(out_dir, f"{name}_ziel.npy")
        os.environ["HIELTECH_SVC_F0_OUT"] = ziel
        tmp = os.path.join(out_dir, f"tmp_{name}")
        shutil.rmtree(tmp, ignore_errors=True)
        args = argparse.Namespace(
            source=source, target=os.path.abspath(v["ref"]), output=tmp,
            diffusion_steps=int(cfg.get("steps", 50)), length_adjust=1.0, inference_cfg_rate=0.7,
            f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
            checkpoint=None, config=None, fp16=bool(cfg.get("fp16", False)),
        )
        print(f"Seed-VC: {name} singt (Lage {v.get('lage')}, Oktave {v.get('oktave', 0)}, Halt {v.get('halt', 0.3)}, Tonart {cfg.get('tonart')})", file=sys.stderr, flush=True)
        inference.main(args)
        files = glob.glob(os.path.join(tmp, "*.wav"))
        if not files or not os.path.isfile(ziel):
            raise RuntimeError(f"Seed-VC hat für {name} kein Ergebnis geschrieben")
        shutil.move(files[0], os.path.join(out_dir, f"{name}.wav"))
        shutil.rmtree(tmp, ignore_errors=True)
        report["voices"].append({"name": name, "seconds": round(time.time() - t0, 1)})
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
