#!/usr/bin/env python3
"""Startet ein Reparatur-Modul und stellt jeder Fortschritts-Zeile („… Text“) die vergangene Zeit voran.
So steht im Protokoll der App, welcher Schritt wie lange gedauert hat – ohne die Module selbst zu ändern.

Aufruf: python _lauf.py <modul-skript> <eingabe> <ausgabe-ordner> [Schalter …]

13. Nov 2026: Früher wurde dafür print() ersetzt. Das brach auf der GPU alle Module ab („module '__main__' has no
attribute '_print'“), sobald ein Modul Unter-Prozesse startete. Jetzt wird nur die Ausgabe (stdout) Zeile für Zeile
gestempelt – print() bleibt unberührt.
"""
import os
import runpy
import sys
import time


class _Stempel:
    """Reicht alles an die echte Ausgabe durch; Zeilen, die mit „…“ beginnen, bekommen die vergangene Zeit davor."""

    def __init__(self, echt):
        self._echt, self._t0, self._zeilenanfang = echt, time.time(), True

    def write(self, text):
        if not isinstance(text, str) or not text:
            return self._echt.write(text)
        aus = []
        for stueck in text.splitlines(True):
            if self._zeilenanfang and stueck.startswith("…"):
                aus.append(f"[{time.time() - self._t0:6.1f} s] ")
            aus.append(stueck)
            self._zeilenanfang = stueck.endswith("\n")
        self._echt.write("".join(aus))
        return len(text)

    def __getattr__(self, name):
        return getattr(self._echt, name)


if __name__ == "__main__":
    skript = os.path.abspath(sys.argv[1])
    sys.argv = [skript] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(skript))
    try:
        sys.stdout = _Stempel(sys.stdout)
    except Exception:  # noqa: BLE001 – ohne Zeit-Stempel läuft das Modul genauso
        pass
    runpy.run_path(skript, run_name="__main__")
