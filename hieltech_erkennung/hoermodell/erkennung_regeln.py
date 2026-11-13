"""Regeln der Instrumenten-Erkennung (12. Nov 2026) – aus den Hör-Ergebnissen wird eine kurze, treffsichere Liste.

Das Hör-Modell liefert je Sekunde und je Instrumenten-Name eine Stärke (0…1). Diese Datei entscheidet daraus, was
„sicher“ (vorausgewählt), „vielleicht“ (wählbar, mit Hörprobe) oder „aus“ ist. Nichts ist auf einen bestimmten Song
zugeschnitten; die Regeln gelten für jeden Song:

  1. Spur-Wissen      – Gitarren zählen nur in der Gitarren-Spur, Piano nur in der Piano-Spur, in der Bass-Spur nur Bass.
  2. Familien         – Gitarren, Tasten, Orgel, Streicher, Blech, Holz …: je Familie EIN Name (der stärkste; der genaue
                        Name schlägt den Oberbegriff). Ein zweiter Name nur, wenn er eigene Zeit hat.
  3. Eigene Zeit      – ein Instrument muss in einem Teil des Songs der stärkste Name seiner Spur sein. Namen, die nur
                        „mitklingen“, sind meist derselbe Klang unter anderem Namen.
  4. Gleiche Quelle   – steigen und fallen zwei Namen in denselben Sekunden gemeinsam, ist es EIN Instrument.
  5. Genre            – für die Stilrichtung unübliche Instrumente brauchen einen deutlich höheren Beweis.
  6. Doppelung        – Gitarre / Piano / Orgel mit denselben Anschlägen, eine Spur viel leiser: Abdruck der anderen.
                        Ähnlich laut: beide bleiben, mit Hinweis zum Gegenhören. Fast stumme Spur: kein Instrument.
  7. Schlagzeug       – ist ein Schlagzeug da, gehören Kick, Snare und Hi-Hat dazu.

Die Datei braucht nur numpy und läuft überall gleich: im Erkennungs-Skript (Mac und RunPod) und im Server der App
(dort, um nach einer anderen Genre-Wahl sofort neu zu entscheiden, ohne noch einmal zu hören).
"""
import numpy as np

FASSUNG = 2
SCHWELLE = 0.15          # ab hier gilt ein Name in einer Sekunde als „hörbar“

# ------------------------------------------------------------------ Familien
# Erster Eintrag mit * = Oberbegriff (zählt nur, wenn kein genauer Name deutlich ist).
FAMILIEN = {
    "gitarre":   ["*guitar", "electric-guitar", "acoustic-guitar", "banjo", "sitar", "dobro", "mandolin", "ukulele"],
    "harfe":     ["harp"],
    "tasten":    ["*keys", "piano", "digital-piano", "harpsichord"],
    "orgel":     ["organ", "accordion"],
    "harmonika": ["harmonica"],
    "synth":     ["synth"],
    "holz":      ["*woodwind", "*wind", "clarinet", "flute", "saxophone", "oboe", "bassoon"],
    "blech":     ["*brass", "trumpet", "trombone", "french-horn", "tuba"],
    "streicher": ["*bowed_strings", "*strings", "violin", "cello", "viola"],
    "marimba":   ["marimba"], "glockenspiel": ["glockenspiel"], "glocken": ["bells"], "windspiel": ["wind-chimes"],
}
FAMILIE_VON, OBERBEGRIFF = {}, set()
for _f, _l in FAMILIEN.items():
    for _s in _l:
        if _s.startswith("*"):
            OBERBEGRIFF.add(_s[1:])
        FAMILIE_VON[_s.lstrip("*")] = _f
SCHLAGZEUG_TEILE = ("kick", "snare", "hh")
EINZELN = ("percussion", "tambourine", "timpani", "congas", "toms", "triangle", "bass", "double-bass")

# Spur-Wissen (wie ein Toningenieur): was in welcher der 6 getrennten Spuren zählen darf
_EXOTEN = {"banjo", "sitar", "dobro", "mandolin", "ukulele", "harp"}
SPUR_ERLAUBT = {
    "drums": {"drums", "kick", "snare", "hh", "percussion", "tambourine", "timpani", "congas", "toms", "triangle"},
    "bass": {"bass", "double-bass"},
    "vocals": {"vocal", "lead-vocal", "back-vocal"},
    "guitar": {"guitar", "electric-guitar", "acoustic-guitar"} | _EXOTEN,
    "piano": {"piano", "digital-piano", "keys", "harpsichord", "organ", "synth"},
}
_NICHT_IN_OTHER = {"bass", "double-bass", "drums", "kick", "snare", "hh", "vocal", "lead-vocal", "back-vocal",
                   "guitar", "electric-guitar", "acoustic-guitar", "piano"}


def spur_erlaubt(spur, slug):
    if spur == "other":
        return slug not in _NICHT_IN_OTHER
    e = SPUR_ERLAUBT.get(spur)
    return True if e is None else slug in e


# ------------------------------------------------------------------ Genre
# Stilrichtungen der App  <-  Klassen des Hör-Modells (AudioSet)
GENRES = {
    "reggae": ["Reggae", "Ska"],
    "rock": ["Rock music", "Punk rock"],
    "metal": ["Heavy metal"],
    "pop": ["Pop music"],
    "hiphop": ["Hip hop music"],
    "soul": ["Rhythm and blues", "Soul music", "Funk", "Disco", "Gospel music"],
    "jazz": ["Jazz", "Swing music"],
    "blues": ["Blues"],
    "folk": ["Country", "Bluegrass", "Folk music", "Traditional music"],
    "klassik": ["Classical music", "Opera"],
    "elektro": ["Electronic music", "House music", "Techno", "Dubstep", "Drum and bass", "Ambient music",
                "Trance music", "New-age music"],
    "latin": ["Music of Latin America", "Salsa music", "Flamenco"],
    "afro": ["Afrobeat"],
    "orient": ["Middle Eastern music"],
    "indisch": ["Carnatic music", "Music of Bollywood"],
}
_SELTEN = {"banjo", "sitar", "dobro", "harp", "harpsichord", "mandolin", "ukulele", "french-horn", "tuba", "timpani",
           "glockenspiel", "bells", "wind-chimes", "bassoon", "oboe", "viola"}
# Für die Stilrichtung UNÜBLICH (nicht verboten – braucht nur einen höheren Beweis)
UNUEBLICH = {
    "reggae": _SELTEN | {"accordion", "harmonica", "violin", "cello", "bowed_strings", "strings", "clarinet", "flute",
                         "woodwind", "wind", "double-bass", "marimba"},
    "rock": _SELTEN | {"accordion", "clarinet", "flute", "woodwind", "wind", "trombone", "trumpet", "brass", "cello",
                       "violin", "bowed_strings", "double-bass", "marimba"},
    "metal": _SELTEN | {"accordion", "clarinet", "flute", "woodwind", "wind", "trombone", "trumpet", "brass",
                        "saxophone", "cello", "violin", "bowed_strings", "double-bass", "marimba", "harmonica",
                        "digital-piano", "tambourine"},
    "pop": _SELTEN | {"accordion", "harmonica", "clarinet", "double-bass"},
    "hiphop": _SELTEN | {"accordion", "harmonica", "clarinet", "double-bass", "cello", "trombone", "acoustic-guitar"},
    "soul": _SELTEN | {"accordion", "harmonica", "clarinet", "cello", "double-bass"},
    "jazz": (_SELTEN - {"tuba"}) | {"accordion", "harmonica", "synth", "cello"},
    "blues": _SELTEN | {"accordion", "clarinet", "flute", "woodwind", "wind", "cello", "violin", "bowed_strings",
                        "strings", "synth", "marimba"},
    "folk": (_SELTEN - {"banjo", "dobro", "mandolin", "ukulele", "harp"}) | {"synth", "trombone", "trumpet", "brass",
                                                                           "saxophone", "digital-piano"},
    "klassik": (_SELTEN - {"harp", "harpsichord", "french-horn", "tuba", "timpani", "glockenspiel", "bells", "bassoon",
                           "oboe", "viola"}) | {"electric-guitar", "synth", "digital-piano", "bass", "harmonica",
                                                "accordion", "drums", "kick", "snare", "hh", "tambourine"},
    "elektro": _SELTEN | {"accordion", "harmonica", "clarinet", "double-bass", "cello", "trombone", "acoustic-guitar",
                          "saxophone", "trumpet", "flute"},
    "latin": (_SELTEN - {"timpani"}) | {"harmonica", "cello", "synth"},
    "afro": _SELTEN | {"accordion", "harmonica", "cello", "clarinet", "double-bass"},
    "orient": (_SELTEN - {"sitar"}) | {"harmonica", "trombone", "electric-guitar"},
    "indisch": (_SELTEN - {"sitar", "harp"}) | {"harmonica", "trombone", "double-bass"},
}
# Bekannte Verwechslung des Hör-Modells: Orgel wird oft „Akkordeon“ genannt. Akkordeon heißt die Orgel-Familie
# deshalb nur dort, wo ein Akkordeon üblich ist – sonst „Orgel“ (das Akkordeon bleibt als „vielleicht“ wählbar).
AKKORDEON_UEBLICH = {"folk", "latin", "orient", "klassik"}
GENRE_MIN, GENRE_ABSTAND = 0.06, 1.4      # Genre gilt als erkannt: Mittelwert ≥ MIN und ≥ ABSTAND × das zweitbeste


def genre_aus_klassen(mittel):
    """mittel: {AudioSet-Klasse: Mittelwert über den ganzen Song}. Rückgabe {genre: wert}."""
    return {g: round(float(max(mittel.get(k, 0.0) for k in ks)), 4) for g, ks in GENRES.items()}


def genre_waehlen(werte):
    """Rückgabe (genre_oder_None, sicher). Nur ein klares Ergebnis zählt – sonst lieber kein Genre-Filter."""
    r = sorted((werte or {}).items(), key=lambda kv: -kv[1])
    if not r or r[0][1] < GENRE_MIN:
        return None, False
    if len(r) > 1 and r[0][1] < GENRE_ABSTAND * r[1][1]:
        return r[0][0], False
    return r[0][0], True


# ------------------------------------------------------------------ Messwerte
def _mess(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return {"spitze": 0.0, "a15": 0.0, "bi": 0}
    return {"spitze": float(x.max()), "a15": float(np.mean(x >= SCHWELLE)), "bi": int(np.argmax(x))}


def abschnitte(x, schritt=1.0, fenster=2.0, luecke=3.0, mindest=2.0):
    an = np.nonzero(np.asarray(x) >= SCHWELLE)[0] * schritt
    if len(an) == 0:
        return []
    seg = [[float(an[0]), float(an[0]) + fenster]]
    for t in an[1:]:
        if t - seg[-1][1] <= luecke:
            seg[-1][1] = float(t) + fenster
        else:
            seg.append([float(t), float(t) + fenster])
    return [[round(a, 1), round(b, 1)] for a, b in seg if b - a >= mindest]


def _staerke(m):
    return m["spitze"] * (m["a15"] ** 0.5)


def _gleiche_quelle(a, b):
    """Steigen und fallen zwei Zeitleisten gemeinsam? Rückgabe (deckung, gleichlauf)."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    aa, bb = a >= SCHWELLE, b >= SCHWELLE
    klein = min(int(aa.sum()), int(bb.sum()))
    if klein < 3:
        return 0.0, 0.0
    deckung = float((aa & bb).sum()) / klein
    m = aa | bb
    if m.sum() < 6 or a[m].std() < 1e-6 or b[m].std() < 1e-6:
        return deckung, 0.0
    return deckung, float(np.corrcoef(a[m], b[m])[0, 1])


SICHER_A = (0.45, 0.08, 0.04)     # Spitze, Anteil der Zeit, eigene Zeit
SICHER_B = (0.65, 0.04, 0.03)     # kurz, aber sehr deutlich
SICHER_UNUEBLICH = (0.70, 0.12, 0.08)
VIELLEICHT = (0.30, 0.02)


EIGENE_SPUR = (0.35, 0.05)        # Instrument mit eigener Spur der Trennung (Bass, Gesang, Gitarre, Piano): die Spur
                                  # selbst ist schon ein Beweis – hier genügt weniger


def _stufe(m, eigen, unueblich=False, eigene_spur=False):
    sp, a = m["spitze"], m["a15"]
    if eigene_spur and not unueblich and sp >= EIGENE_SPUR[0] and a >= EIGENE_SPUR[1]:
        return "sicher"
    if unueblich:
        if sp >= SICHER_UNUEBLICH[0] and a >= SICHER_UNUEBLICH[1] and eigen >= SICHER_UNUEBLICH[2]:
            return "sicher"
        return "vielleicht" if (sp >= 0.45 and a >= 0.04) else "aus"
    if (sp >= SICHER_A[0] and a >= SICHER_A[1] and eigen >= SICHER_A[2]) or \
       (sp >= SICHER_B[0] and a >= SICHER_B[1] and eigen >= SICHER_B[2]):
        return "sicher"
    return "vielleicht" if (sp >= VIELLEICHT[0] and a >= VIELLEICHT[1]) else "aus"


# ------------------------------------------------------------------ Hauptteil
def auswerten(roh, genre="auto"):
    """roh: {"instrumente": {slug: {"spur": …, "k": [0…100 je Sekunde]}}, "genre": {…}, "paare": […], "spuren": {…},
             "schritt": 1.0, "fenster": 2.0}
    genre: "auto" (aus dem Song), "ohne" (kein Genre-Filter) oder ein Schlüssel aus GENRES.
    Rückgabe (liste, info) – liste wie bisher (ein Eintrag je Instrument, sortiert), dazu stufe / hinweis / familie."""
    schritt, fenster = float(roh.get("schritt", 1.0)), float(roh.get("fenster", 2.0))
    K, SPUR = {}, {}
    for s, e in (roh.get("instrumente") or {}).items():
        K[s] = np.asarray(e.get("k") or [], dtype=np.float32) / 100.0
        SPUR[s] = e.get("spur") or ""
    auto, auto_sicher = genre_waehlen(roh.get("genre"))
    if genre in (None, "", "auto"):
        g = auto if auto_sicher else None
    elif genre == "ohne":
        g = None
    else:
        g = genre if genre in GENRES else None
    unueblich = UNUEBLICH.get(g, set())
    info = {"genre": g, "genre_auto": auto, "genre_sicher": bool(auto_sicher), "genre_wahl": genre or "auto",
            "genre_werte": roh.get("genre") or {}}
    M = {s: _mess(x) for s, x in K.items()}
    res = {}

    def eintrag(s, stufe, hinweis=None, wie=None, familie=None):
        m, x = M[s], K[s]
        res[s] = {"instrument": s, "margin": round(m["spitze"], 3), "stufe": stufe,
                  "detected": stufe == "sicher", "moeglich": stufe == "vielleicht",
                  "best_window_start_sec": round(m["bi"] * schritt, 1), "anteil_prozent": round(m["a15"] * 100, 1),
                  "zeiten": abschnitte(x, schritt, fenster), "spur": SPUR.get(s, "")}
        if familie:
            res[s]["familie"] = familie
        if hinweis:
            res[s]["hinweis"] = hinweis
            if wie:
                res[s]["hinweis_wie"] = wie

    # ---- Familien je Spur: Einheiten (Familie, Spur) mit Familien-Zeitleiste
    einheiten = {}
    for s in K:
        f = FAMILIE_VON.get(s)
        if f and M[s]["spitze"] >= 0.15:
            einheiten.setdefault((f, SPUR[s]), []).append(s)
    E = {}
    for (f, spur), glieder in einheiten.items():
        F = np.max(np.stack([K[s] for s in glieder]), axis=0)
        E[(f, spur)] = {"familie": f, "spur": spur, "glieder": glieder, "F": F, "m": _mess(F)}
    # eigene Zeit: in wie vielen Sekunden ist die Familie die stärkste ihrer Spur?
    for spur in {sp for _, sp in E}:
        ks = [k for k in E if k[1] == spur]
        stapel = np.stack([E[k]["F"] for k in ks])
        top = stapel.argmax(axis=0)
        for i, k in enumerate(ks):
            E[k]["eigen"] = float(np.mean((top == i) & (E[k]["F"] >= SCHWELLE)))
    # Leitname je Einheit
    for k, e in E.items():
        gl = sorted(e["glieder"], key=lambda s: -_staerke(M[s]))
        genau = [s for s in gl if s not in OBERBEGRIFF]
        ober = [s for s in gl if s in OBERBEGRIFF]
        leit = gl[0]
        if leit in OBERBEGRIFF and genau and _staerke(M[genau[0]]) >= 0.8 * _staerke(M[leit]):
            leit = genau[0]                              # der genaue Name schlägt den Oberbegriff
        if g and leit in unueblich:                      # unüblicher Leitname: gibt es einen üblichen genauen?
            ok = [s for s in genau if s not in unueblich and _staerke(M[s]) >= 0.5 * _staerke(M[leit])]
            if ok:
                leit = ok[0]
        umbenannt = None
        if e["familie"] == "orgel" and leit == "accordion" and g not in AKKORDEON_UEBLICH:
            leit, umbenannt = "organ", "accordion"
            if "organ" not in K:                         # Orgel bekommt die Messwerte der Familie
                K["organ"], SPUR["organ"] = e["F"], e["spur"]
            else:
                K["organ"] = np.maximum(K["organ"], e["F"]) if SPUR.get("organ") == e["spur"] else e["F"]
                SPUR["organ"] = e["spur"]
            M["organ"] = _mess(K["organ"])
            if "organ" not in e["glieder"]:
                e["glieder"].append("organ")
        e["leit"], e["umbenannt"] = leit, umbenannt
        e["unueblich"] = bool(g) and leit in unueblich
        e["stufe"] = _stufe(e["m"], e.get("eigen", 1.0), e["unueblich"],
                            (e["familie"], e["spur"]) in (("gitarre", "guitar"), ("tasten", "piano")))
        e["hinweis"], e["wie"] = ("unueblich", g) if (e["unueblich"] and e["stufe"] != "aus") else (None, None)

    # ---- gleiche Quelle: zwei Familien derselben Spur, die gemeinsam steigen und fallen -> die schwächere „vielleicht“
    for spur in {sp for _, sp in E}:
        ks = sorted((k for k in E if k[1] == spur and E[k]["stufe"] != "aus"),
                    key=lambda k: -_staerke(E[k]["m"]) * (0.6 if E[k]["unueblich"] else 1.0))
        for i, kb in enumerate(ks):
            for ka in ks[:i]:
                if E[ka]["stufe"] != "sicher":
                    continue
                deckung, lauf = _gleiche_quelle(E[ka]["F"], E[kb]["F"])
                if deckung >= 0.6 and lauf >= 0.45:
                    if E[kb]["stufe"] == "sicher":
                        E[kb]["stufe"] = "vielleicht"
                    E[kb]["hinweis"], E[kb]["wie"] = "klingt_wie", E[ka]["leit"]
                    break

    # ---- dieselbe Familie in zwei Spuren (z. B. Piano in „piano“, Keys in „other“): nur eine zählt, außer die zweite
    #      hat einen anderen genauen Namen und ist für sich sicher
    for f in {k[0] for k in E}:
        ks = sorted((k for k in E if k[0] == f), key=lambda k: (E[k]["stufe"] != "sicher", -_staerke(E[k]["m"])))
        haupt = E[ks[0]]
        for k in ks[1:]:
            e = E[k]
            if e["stufe"] == "sicher" and e["leit"] not in OBERBEGRIFF and e["leit"] != haupt["leit"]:
                continue
            if e["leit"] == haupt["leit"] or e["stufe"] == "aus":
                e["stufe"] = "aus"
            else:
                e["stufe"], e["hinweis"], e["wie"] = "vielleicht", "gehoert_zu", haupt["leit"]

    # ---- Doppelung Gitarre / Piano / Orgel: gleiche Anschläge, eine Spur viel leiser -> Abdruck der anderen Spur
    leise, rhythmus = {}, {}
    for p in roh.get("paare") or []:
        try:
            gleich, db = float(p["gleich"]), float(p["pegel_db"])      # pegel_db = Pegel(b) − Pegel(a)
        except (KeyError, TypeError, ValueError):
            continue
        if gleich >= 0.55:
            schwach, stark = (p["b"], p["a"]) if db < 0 else (p["a"], p["b"])
            if abs(db) >= 10.0:
                leise[schwach] = stark                     # viel leiser: Abdruck der anderen Spur
            elif abs(db) >= 3.0:
                rhythmus[schwach] = stark                  # ähnlich laut: zwei Instrumente spielen denselben Rhythmus
    for k, e in E.items():
        stark = leise.get(e["spur"])
        if stark and e["stufe"] == "sicher":
            quelle = max((x for x in E.values() if x["spur"] == stark and x["stufe"] == "sicher"),
                         key=lambda x: _staerke(x["m"]), default=None)
            if quelle is not None:
                e["stufe"], e["hinweis"], e["wie"] = "vielleicht", "doppelung", quelle["leit"]
        stark = rhythmus.get(e["spur"])
        if stark and e["stufe"] == "sicher" and not e["hinweis"]:
            quelle = max((x for x in E.values() if x["spur"] == stark and x["stufe"] == "sicher"),
                         key=lambda x: _staerke(x["m"]), default=None)
            if quelle is not None:                         # bleibt „sicher“ – nur der Hinweis zum Gegenhören
                e["hinweis"], e["wie"] = "gleicher_rhythmus", quelle["leit"]
    # ---- sehr leise Spur (Gitarre / Piano / Other): weit unter der lautesten = Rest der Trennung, kein Instrument
    pegel = {sp: float(v.get("pegel_db", -120.0)) for sp, v in (roh.get("spuren") or {}).items()
             if sp in ("guitar", "piano", "other") and isinstance(v, dict)}
    if pegel:
        laut = max(pegel.values())
        for k, e in E.items():
            d = laut - pegel.get(e["spur"], laut)
            if d >= 45.0:
                e["stufe"], e["hinweis"], e["wie"] = "aus", None, None
            elif d >= 25.0 and e["stufe"] == "sicher":
                e["stufe"], e["hinweis"], e["wie"] = "vielleicht", "sehr_leise", None

    # ---- Einträge der Familien
    rang = {"sicher": 0, "vielleicht": 1, "aus": 2}
    for k, e in sorted(E.items(), key=lambda ke: (rang[ke[1]["stufe"]], -_staerke(ke[1]["m"]))):
        leit = e["leit"]
        if leit in res:
            continue
        if e["spur"] != SPUR.get(leit):                  # Messwerte der Einheit gelten (Leitname lag in anderer Spur)
            K[leit], SPUR[leit] = e["F"], e["spur"]
            M[leit] = e["m"]
        eintrag(leit, e["stufe"], e["hinweis"], e["wie"], e["familie"])
        res[leit]["eigen_prozent"] = round(e.get("eigen", 0.0) * 100, 1)
        fam_sicher = e["stufe"] == "sicher"
        # weitere Namen der Familie: eigenes Instrument nur mit eigener Zeit innerhalb der Familie
        gl = [s for s in e["glieder"] if s != leit]
        if gl:
            alle = [leit] + gl
            stapel = np.stack([K[s] if len(K[s]) == len(e["F"]) else e["F"] * 0 for s in alle])
            top = stapel.argmax(axis=0)
        for j, s in enumerate(gl, start=1):
            if s in res:
                continue
            m = M[s]
            eigen = float(np.mean((top == j) & (K[s] >= SCHWELLE))) if len(K[s]) == len(e["F"]) else 0.0
            un = bool(g) and s in unueblich
            if s == e["umbenannt"]:
                st = "vielleicht" if (fam_sicher and not un) else "aus"
                eintrag(s, st, "klingt_wie" if st != "aus" else None, leit, e["familie"])
            elif (s not in OBERBEGRIFF and leit not in OBERBEGRIFF and not un and fam_sicher
                  and m["spitze"] >= 0.5 and eigen >= 0.08):
                eintrag(s, "sicher", None, None, e["familie"])          # zweites Instrument der Familie
            elif s in OBERBEGRIFF or m["spitze"] < 0.40 or m["a15"] < 0.03 or un:
                eintrag(s, "aus", None, None, e["familie"])
            else:
                eintrag(s, "vielleicht" if e["stufe"] != "aus" else "aus", "gehoert_zu", leit, e["familie"])

    # ---- Schlagzeug: Kick, Snare, Hi-Hat gehören zu einem erkannten Schlagzeug
    if "drums" in K:
        m = M["drums"]
        st = "sicher" if (m["spitze"] >= 0.40 and m["a15"] >= 0.10) else \
             ("vielleicht" if (m["spitze"] >= 0.30 and m["a15"] >= 0.02) else "aus")
        if g and "drums" in unueblich and st == "sicher" and not (m["spitze"] >= 0.6 and m["a15"] >= 0.3):
            st = "vielleicht"
        eintrag("drums", st, None, None, "schlagzeug")
    kit = res.get("drums", {}).get("stufe") == "sicher"
    for s in SCHLAGZEUG_TEILE:
        if s not in K:
            continue
        m = M[s]
        if kit and m["spitze"] >= 0.25:
            eintrag(s, "sicher", None, None, "schlagzeug")
        else:
            eintrag(s, _stufe(m, 1.0), None, None, "schlagzeug")
    # ---- einzelne Instrumente ohne Familie (Percussion, Tamburin, Bass …)
    for s in EINZELN:
        if s in K and s not in res:
            un = bool(g) and s in unueblich
            st = _stufe(M[s], 1.0, un, s in ("bass", "double-bass") and SPUR.get(s) == "bass")
            eintrag(s, st, "unueblich" if un and st != "aus" else None, g if un else None)
    # ---- Gesang
    if "vocal" in K:
        K["lead-vocal"], SPUR["lead-vocal"], M["lead-vocal"] = K["vocal"], SPUR.get("vocal", ""), M["vocal"]
        eintrag("lead-vocal", _stufe(M["vocal"], 1.0, False, SPUR.get("vocal") == "vocals"), None, None, "gesang")
    if "back-vocal" in K:
        eintrag("back-vocal", _stufe(M["back-vocal"], 1.0), None, None, "gesang")
    for s in K:                                          # alles Übrige: nach den einfachen Schwellen
        if s not in res and s != "vocal":
            eintrag(s, _stufe(M[s], 1.0, bool(g) and s in unueblich))

    liste = sorted(res.values(), key=lambda r: ({"sicher": 0, "vielleicht": 1}.get(r["stufe"], 2), -r["margin"]))
    return liste, info
