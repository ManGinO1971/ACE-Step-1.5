#!/usr/bin/env python3
"""
HIELTECH Stems – Tempo, Tonart und Länge des Songs schätzen (31. Okt 2026).

Im Musik-Generator bekommt das Neu-Singen (Vocals V2) Tempo (bpm), Tonart (key_scale) und Länge des Songs mit –
dort sind sie bekannt, weil die App den Song selbst erzeugt hat. In der Stems-App kommt ein fremder Song,
deshalb hier geschätzt: Tonart wie beim Background-Chor (backing_chor_prepare_step.detect_key,
Krumhansl-Schmuckler), Tempo per librosa-Taktschlag.

Aufruf: python stems_takt_tonart.py <song>
Letzte stdout-Zeile: {"ok": true, "bpm": 72, "key_scale": "A minor", "tonart": "a-moll", "sicherheit": 0.81, "dauer": 241.3}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_EN = {"c": "C", "cis": "C#", "des": "Db", "d": "D", "dis": "D#", "es": "Eb", "e": "E", "f": "F", "fis": "F#", "ges": "Gb",
       "g": "G", "gis": "G#", "as": "Ab", "a": "A", "ais": "A#", "b": "Bb", "h": "B"}


def key_englisch(tonart):
    """'a-moll' -> 'A minor', 'fis-dur' -> 'F# major' (Schreibweise wie in der App)."""
    try:
        name, art = tonart.split("-")
        return f"{_EN[name]} {'minor' if art == 'moll' else 'major'}"
    except (ValueError, KeyError):
        return ""


def main():
    import librosa
    import numpy as np
    song = sys.argv[1]
    bericht = {"ok": True}
    try:
        from backing_chor_prepare_step import detect_key
        tonart, sicher = detect_key(song)
        bericht.update(tonart=tonart, sicherheit=sicher, key_scale=key_englisch(tonart) if sicher >= 0.5 else "")
    except Exception as e:  # noqa: BLE001
        bericht.update(tonart="", key_scale="", tonart_fehler=str(e)[:200])
    try:
        y, sr = librosa.load(song, sr=22050, mono=True, duration=240)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        while bpm > 160:          # Reggae/One-Drop wird oft doppelt gezählt
            bpm /= 2
        while bpm and bpm < 55:
            bpm *= 2
        bericht["bpm"] = int(round(bpm)) if bpm else None
        bericht["dauer"] = round(float(librosa.get_duration(path=song)), 2)
    except Exception as e:  # noqa: BLE001
        bericht["bpm_fehler"] = str(e)[:200]
    print(json.dumps(bericht, ensure_ascii=False))


if __name__ == "__main__":
    main()
