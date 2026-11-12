#!/usr/bin/env python3
"""Startet ein Reparatur-Modul und stellt jeder Fortschritts-Zeile („… Text“) die vergangene Zeit voran.
So steht im Protokoll der App, welcher Schritt wie lange gedauert hat – ohne die Module selbst zu ändern.

Aufruf: python _lauf.py <modul-skript> <eingabe> <ausgabe-ordner> [Schalter …]
"""
import builtins
import os
import runpy
import sys
import time

_T0 = time.time()
_PRINT = builtins.print


def _print(*a, **k):
    if a and isinstance(a[0], str) and a[0].startswith("…") and k.get("file") in (None, sys.stdout):
        a = (f"[{time.time() - _T0:6.1f} s] {a[0]}",) + a[1:]
    return _PRINT(*a, **k)


if __name__ == "__main__":
    builtins.print = _print
    skript = os.path.abspath(sys.argv[1])
    sys.argv = [skript] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(skript))
    runpy.run_path(skript, run_name="__main__")
