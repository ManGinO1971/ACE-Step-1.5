#!/usr/bin/env python3
"""
HIELTECH Stems – Rezept-Tabelle "Neu erzeugen" (29. Okt 2026).

EINE Zeile pro Instrument der 53er-Trennung (BS-Roformer MVSep Mega 53, gleiche Namen wie get_stem53.py).
Neue Instrumente später = neue Zeile, kein neues Programm.

Felder:
  familie   Ordner im Ergebnis (Gesang, Schlagzeug, Bass, Gitarre, Tasten, Bläser, Holzbläser, Streicher, Percussion, Sonstiges)
  spur      ACE-Step-Extract-Spur (nur diese 12 kennt ACE-Step: vocals, backing_vocals, drums, bass, guitar,
            keyboard, percussion, strings, synth, brass, woodwinds, fx)
  text      Beschreibung für ACE-Step ("isolated … only")
  rezept    Nachbearbeitung:
              schlagzeug   – zerlegen in Kick/Snare/Toms/Hi-Hat/Becken, Snare reinigen, Becken-Ausklang 1,6 (fertig, Hörtest 29.10.)
              bass         – Grundton + Obertöne neu, Original-Lautstärke (fertig, Hörtest 29.10.)
              karte        – Zeitmaske + Gate + Karte -> Instrumenten-Vocoder (Gitarre/Piano "B", Hörtest 29.10.)
              melodie      – einstimmig: Karte (B) UND Grundton+Obertöne neu (C) zum Vergleich (TEST)
              perkussion   – kurze Maske + Gate + Karte, Anschläge bleiben (TEST)
              gesang       – vorerst Karte; Gesangskette (Lead V2 ohne .pth) folgt
              blaeser_satz – Bläser zusammen: entrauschen, Ausklang bis zum Ende, Lücken füllen, Raum-Hall (fertig, Hörtest 29.10.)
                             + Unterordner teile/ nach Stimmlage: hoch (Trompete) / mitte (Sax) / tief (Posaune) (Hörtest: B besser)
  ton       (fmin, fmax) Hz – nur für "melodie"/"bass": Tonbereich für die Tonhöhen-Erkennung
  status    "fertig" (per Hörtest bestätigt) oder "test"
"""

REZEPTE = {
    # ---------------- Gesang ----------------
    "lead-vocal":   dict(familie="Gesang", spur="vocals", text="lead vocal", rezept="gesang", status="test"),
    "vocal":        dict(familie="Gesang", spur="vocals", text="vocals", rezept="gesang", status="test"),
    "back-vocal":   dict(familie="Gesang", spur="backing_vocals", text="backing vocals choir", rezept="gesang", status="test"),
    # ---------------- Schlagzeug (ein gemeinsamer Durchgang) ----------------
    "drums":        dict(familie="Schlagzeug", spur="drums", text="drum kit, tight live drums, punchy kick, crisp snare, clean hi-hats", rezept="schlagzeug", status="fertig"),
    "kick":         dict(familie="Schlagzeug", spur="drums", text="drum kit", rezept="schlagzeug", status="fertig"),
    "snare":        dict(familie="Schlagzeug", spur="drums", text="drum kit", rezept="schlagzeug", status="fertig"),
    "toms":         dict(familie="Schlagzeug", spur="drums", text="drum kit", rezept="schlagzeug", status="fertig"),
    "hh":           dict(familie="Schlagzeug", spur="drums", text="drum kit", rezept="schlagzeug", status="fertig"),
    # ---------------- Bass ----------------
    "bass":         dict(familie="Bass", spur="bass", text="electric bass, deep round warm bass guitar, clean DI", rezept="bass", ton=(30, 330), status="fertig"),
    "double-bass":  dict(familie="Bass", spur="bass", text="upright double bass, deep warm acoustic bass", rezept="bass", ton=(30, 330), status="test"),
    # ---------------- Gitarren ----------------
    "guitar":          dict(familie="Gitarre", spur="guitar", text="electric guitar, clean offbeat rhythm guitar", rezept="karte", status="fertig"),
    "electric-guitar": dict(familie="Gitarre", spur="guitar", text="electric guitar", rezept="karte", status="fertig"),
    "acoustic-guitar": dict(familie="Gitarre", spur="guitar", text="acoustic guitar, warm steel strings", rezept="karte", status="test"),
    "banjo":           dict(familie="Gitarre", spur="guitar", text="banjo", rezept="karte", status="test"),
    "mandolin":        dict(familie="Gitarre", spur="guitar", text="mandolin", rezept="karte", status="test"),
    "ukulele":         dict(familie="Gitarre", spur="guitar", text="ukulele", rezept="karte", status="test"),
    "dobro":           dict(familie="Gitarre", spur="guitar", text="dobro resonator slide guitar", rezept="karte", status="test"),
    "sitar":           dict(familie="Gitarre", spur="guitar", text="sitar", rezept="karte", status="test"),
    "harp":            dict(familie="Gitarre", spur="strings", text="concert harp", rezept="karte", status="test"),
    # ---------------- Tasten ----------------
    "piano":         dict(familie="Tasten", spur="keyboard", text="acoustic piano", rezept="karte", status="fertig"),
    "digital-piano": dict(familie="Tasten", spur="keyboard", text="electric piano", rezept="karte", status="test"),
    "keys":          dict(familie="Tasten", spur="keyboard", text="keyboards, organ and piano", rezept="karte", status="fertig"),
    "organ":         dict(familie="Tasten", spur="keyboard", text="hammond organ", rezept="karte", status="test"),
    "harpsichord":   dict(familie="Tasten", spur="keyboard", text="harpsichord", rezept="karte", status="test"),
    "accordion":     dict(familie="Tasten", spur="keyboard", text="accordion", rezept="karte", status="test"),
    "synth":         dict(familie="Tasten", spur="synth", text="synthesizer", rezept="karte", status="test"),
    # ---------------- Bläser (Blech) ----------------
    # Hörtest 29.10.: "4_brass_NEU_hall der bessere" – Bläser-Satz zusammen: entrauschen -> natürlicher Ausklang
    # -> Lücken füllen (wie Becken) -> leichter Raum-Hall 1,8 s
    "brass":        dict(familie="Bläser", spur="brass", text="brass section, trumpets, trombones and saxophones", rezept="blaeser_satz", status="fertig"),
    "trumpet":      dict(familie="Bläser", spur="brass", text="trumpet", rezept="melodie", ton=(160, 1100), status="test"),
    "trombone":     dict(familie="Bläser", spur="brass", text="trombone", rezept="melodie", ton=(70, 600), status="test"),
    "french-horn":  dict(familie="Bläser", spur="brass", text="french horn", rezept="melodie", ton=(60, 800), status="test"),
    "tuba":         dict(familie="Bläser", spur="brass", text="tuba", rezept="melodie", ton=(35, 400), status="test"),
    "saxophone":    dict(familie="Bläser", spur="woodwinds", text="saxophone", rezept="melodie", ton=(100, 1000), status="test"),
    # ---------------- Holzbläser ----------------
    "woodwind":     dict(familie="Holzbläser", spur="woodwinds", text="woodwind section", rezept="karte", status="test"),
    "wind":         dict(familie="Holzbläser", spur="woodwinds", text="wind instruments", rezept="karte", status="test"),
    "flute":        dict(familie="Holzbläser", spur="woodwinds", text="flute", rezept="melodie", ton=(250, 2100), status="test"),
    "clarinet":     dict(familie="Holzbläser", spur="woodwinds", text="clarinet", rezept="melodie", ton=(140, 1600), status="test"),
    "oboe":         dict(familie="Holzbläser", spur="woodwinds", text="oboe", rezept="melodie", ton=(230, 1400), status="test"),
    "bassoon":      dict(familie="Holzbläser", spur="woodwinds", text="bassoon", rezept="melodie", ton=(55, 600), status="test"),
    "harmonica":    dict(familie="Holzbläser", spur="woodwinds", text="harmonica", rezept="melodie", ton=(250, 2000), status="test"),
    # ---------------- Streicher ----------------
    "strings":       dict(familie="Streicher", spur="strings", text="string section", rezept="karte", status="test"),
    "bowed_strings": dict(familie="Streicher", spur="strings", text="bowed string section", rezept="karte", status="test"),
    "violin":        dict(familie="Streicher", spur="strings", text="violin", rezept="melodie", ton=(190, 2100), status="test"),
    "viola":         dict(familie="Streicher", spur="strings", text="viola", rezept="melodie", ton=(125, 1100), status="test"),
    "cello":         dict(familie="Streicher", spur="strings", text="cello", rezept="melodie", ton=(60, 800), status="test"),
    # ---------------- Percussion ----------------
    "percussion":   dict(familie="Percussion", spur="percussion", text="hand percussion, djembe, african drums", rezept="perkussion", status="test"),
    "congas":       dict(familie="Percussion", spur="percussion", text="congas and djembe hand drums", rezept="perkussion", status="test"),
    "tambourine":   dict(familie="Percussion", spur="percussion", text="tambourine", rezept="perkussion", status="test"),
    "triangle":     dict(familie="Percussion", spur="percussion", text="triangle", rezept="perkussion", status="test"),
    "timpani":      dict(familie="Percussion", spur="percussion", text="timpani", rezept="perkussion", status="test"),
    # ---------------- Klangfarben / Sonstiges ----------------
    "marimba":      dict(familie="Sonstiges", spur="percussion", text="marimba", rezept="karte", status="test"),
    "glockenspiel": dict(familie="Sonstiges", spur="percussion", text="glockenspiel", rezept="karte", status="test"),
    "bells":        dict(familie="Sonstiges", spur="percussion", text="bells", rezept="karte", status="test"),
    "wind-chimes":  dict(familie="Sonstiges", spur="fx", text="wind chimes", rezept="karte", status="test"),
}

# Zeitmaske/Gate je Rezept: (Maske dB, Gate dB, Halten s, Blende s) – Schlüssel für stems_karte_step.EINST
MASKE = {
    "karte": (-45.0, -55.0, 0.15, 0.02),
    "gesang": (-45.0, -55.0, 0.25, 0.03),
    "melodie": (-50.0, -60.0, 0.40, 0.05),     # Bläser/Streicher: Ausklang/Raum nicht abschneiden
    "perkussion": (-50.0, -62.0, 0.50, 0.04),  # Djembe/Congas: Bass-Ton klingt nach
}


def rezept(name):
    r = REZEPTE.get(name)
    if not r:
        raise KeyError(f"Unbekanntes Instrument: {name}")
    return r


def prompt(name, caption=""):
    t = f"isolated {rezept(name)['text']} only, clean studio recording, no other instruments"
    return f"{caption}, {t}"[:600] if caption else t


if __name__ == "__main__":
    import collections
    print(len(REZEPTE), "Instrumente")
    print(collections.Counter(r["rezept"] for r in REZEPTE.values()))
