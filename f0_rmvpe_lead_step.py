#!/usr/bin/env python3
"""
Lead Vocals v2 (NEU, 20. Okt 2026): ruft f0_rmvpe_step.py UNVERÄNDERT auf,
nur mit der Schwelle, die im Hörtest für den Lead benutzt wurde (0.03 statt
0.02 beim Background-Melodie-Lock). Läuft wie f0_rmvpe_step.py in der RVC-venv.

Aufruf: python f0_rmvpe_lead_step.py <eingang.wav> <ausgang.npy>
"""
import os
import runpy
import sys

os.environ["HIELTECH_RMVPE_THRED"] = os.environ.get("HIELTECH_LEAD_RMVPE_THRED", "0.03")
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "f0_rmvpe_step.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
