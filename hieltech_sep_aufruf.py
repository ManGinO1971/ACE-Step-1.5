"""HIELTECH (21. Nov 2026): Trennung über die eigene Umgebung mit audio-separator 0.44.5 (wie auf dem Mac).
Gibt die Liste der Dateien zurück – oder None, wenn die Umgebung fehlt oder etwas schiefgeht (dann trennt der
Aufrufer wie bisher selbst). Auf dem Mac gibt es /opt/hieltech_sep_venv nicht: dort ändert sich nichts."""
import os
import json
import subprocess

SEP_PY = os.environ.get("HIELTECH_SEP_PYTHON", "/opt/hieltech_sep_venv/bin/python")
LAUF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hieltech_sep_lauf.py")


def trennen(eingabe, modell, roh, modelle, zeitlimit=3600):
    if not (os.path.exists(SEP_PY) and os.path.exists(LAUF)):
        return None
    try:
        p = subprocess.run([SEP_PY, LAUF, eingabe, modell, roh, modelle], capture_output=True, text=True,
                           timeout=zeitlimit)
        for z in reversed((p.stdout or "").splitlines()):
            if z.startswith("HIELTECH_SEP_JSON "):
                d = [f for f in json.loads(z[len("HIELTECH_SEP_JSON "):]) if os.path.exists(f)]
                return d or None
        print(f"… eigene Trenn-Umgebung ohne Ergebnis: {(p.stderr or p.stdout or '')[-300:]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"… eigene Trenn-Umgebung nicht nutzbar: {str(e)[:200]}", flush=True)
    return None
