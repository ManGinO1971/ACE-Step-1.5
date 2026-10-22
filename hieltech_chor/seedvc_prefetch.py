#!/usr/bin/env python3
"""
Background-Chor (NEU, 23. Okt 2026): lädt beim Docker-Build einmal alle Seed-VC-Gewichte
(Gesangsmodell 44,1 kHz, Whisper-small, BigVGAN, CAMPPlus, RMVPE) ins Image, indem eine
winzige Probe (2 s, 2 Schritte) auf der CPU gerechnet wird. Sonst lädt jeder frische
RunPod-Worker beim ersten Auftrag 1-2 GB aus dem Internet nach.
Bricht den Build nie ab (Exit-Code immer 0).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    with tempfile.TemporaryDirectory() as d:
        sr = 44100
        t = np.arange(2 * sr) / sr
        sf.write(os.path.join(d, "probe.wav"), 0.3 * np.sin(2 * np.pi * 220 * t), sr)
        job = {"source": os.path.join(d, "probe.wav"), "out_dir": os.path.join(d, "out"), "tonart": "a-moll", "steps": 2,
               "voices": [{"name": "probe", "ref": os.path.join(HERE, "stimme_Frau2.wav"), "lage": "original", "oktave": 0, "halt": 0.3}]}
        jf = os.path.join(d, "job.json")
        json.dump(job, open(jf, "w"))
        r = subprocess.run([sys.executable, os.path.join(HERE, "seedvc_multi.py"), jf], capture_output=True, text=True, timeout=3600)
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        print("[Vorab-Download] Seed-VC: " + ("OK" if r.returncode == 0 else f"Fehler {r.returncode}"), flush=True)
except Exception as e:  # noqa: BLE001 - darf den Build nie abbrechen
    print(f"[Vorab-Download] Seed-VC übersprungen: {e}", flush=True)
sys.exit(0)
