#!/usr/bin/env python3
"""
HIELTECH Stems – Lautheit wie im Original (31. Okt 2026).

Wie im Musik-Generator (runpod-handler.js matchLoudnessToOriginal: gemessene Lautheit angleichen, höchstens ±12 dB,
Spitzen begrenzen) – nur dass hier das Gegenstück die ORIGINAL-SPUR ist (getrennter Original-Lead bzw.
Original-Background), nicht der ganze Song: Stems sollen so laut sein wie im Song (wie beim Bass, Hörtest 29.10.).
Kein Kompressor, kein EQ (macht der Nutzer in Logic).

Aufruf: python stems_lautheit.py <neu.wav|flac> <original.wav|flac> <ausgabe.wav> [versatz_db]
"""
import json
import sys

import numpy as np
import soundfile as sf

MAX_DB = 12.0


def lautheit(x, sr):
    """Integrierte Lautheit (LUFS) – pyloudnorm, sonst RMS der aktiven Stellen."""
    m = x if x.ndim == 1 else x.mean(1)
    try:
        import pyloudnorm as pyln
        v = pyln.Meter(sr).integrated_loudness(x)
        if np.isfinite(v):
            return float(v)
    except Exception:  # noqa: BLE001
        pass
    h = sr // 10
    n = len(m) // h
    if n < 1:
        return float("-inf")
    e = np.sqrt((m[:n * h].reshape(n, h) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(e)
    aktiv = db[db > db.max() - 40]
    return float(10 * np.log10(np.mean(10 ** (aktiv / 10)))) if len(aktiv) else float("-inf")


def main():
    neu, orig, out = sys.argv[1:4]
    versatz = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    x, sr = sf.read(neu, always_2d=True, dtype="float32")
    r, sr_r = sf.read(orig, always_2d=True, dtype="float32")
    a, b = lautheit(r, sr_r), lautheit(x, sr)
    gain = 0.0
    if np.isfinite(a) and np.isfinite(b):
        gain = float(np.clip(a + versatz - b, -MAX_DB, MAX_DB))
    y = x * (10 ** (gain / 20))
    spitze = float(np.abs(y).max())
    if spitze > 0.89:                       # wie alimiter limit=0.86 im Generator – nur Spitzen, kein Kompressor
        y = np.where(np.abs(y) > 0.86, np.sign(y) * (0.86 + (np.abs(y) - 0.86) / (1 + (np.abs(y) - 0.86) * 8)), y)
        y = y * min(1.0, 0.89 / max(float(np.abs(y).max()), 1e-6))
    sf.write(out, y, sr, subtype="PCM_24")
    print(json.dumps({"ok": True, "original_lufs": round(a, 1), "neu_lufs": round(b, 1), "gain_db": round(gain, 1)}))


if __name__ == "__main__":
    main()
