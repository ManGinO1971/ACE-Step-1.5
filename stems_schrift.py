#!/usr/bin/env python3
"""
HIELTECH Stems – Schrift-Hilfen für den Gesang (31. Okt 2026).

1. Lead-/Background-Text aus dem Songtext (genau wie im Musik-Generator, runpod-handler.js):
     (…) = nur Background,  {…} = Lead + Background zusammen,  [Abschnitt] bleibt beim Lead
2. Arabische Schrift -> Arabizi (3/7/9/5/2) – ENTWURF für Darija, der Nutzer korrigiert.
3. Umschrift für den Text-Abgleich: das Abgleich-Modell (wav2vec2) kennt nur A–Z.
   Arabizi-Ziffern und arabische Schrift werden NUR für den Abgleich in ähnliche Buchstaben
   umgeschrieben – gesungen wird immer der Text, wie der Nutzer ihn geschrieben hat.
"""
import re
import unicodedata

_BUCHSTABE = r"A-Za-zÀ-ÿ؀-ۿЀ-ӿ一-鿿぀-ヿ가-힯ऀ-ॿ฀-๿Ͱ-Ͽ"


def lead_text(text):
    """Nur der Lead-Text: (…) raus; {…} -> Text bleibt (wie leadOnlyLyricsV2)."""
    out = []
    for raw in str(text or "").splitlines():
        t = raw.strip()
        if not t:
            if out and out[-1] != "":
                out.append("")
            continue
        if re.match(r"^\[[^\]]+\]$", t):
            out.append(t)
            continue
        line = re.sub(r"\s+", " ", re.sub(r"\{([^{}]*)\}", r"\1", re.sub(r"\([^()]*\)", " ", t))).strip()
        if line and re.search(f"[{_BUCHSTABE}]", line):
            out.append(line)
    clean = []
    for i, z in enumerate(out):
        if re.match(r"^\[[^\]]+\]$", z):
            j = i + 1
            while j < len(out) and out[j] == "":
                j += 1
            if j >= len(out) or re.match(r"^\[[^\]]+\]$", out[j]):
                continue
        clean.append(z)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(clean)).strip()


def backing_text(text):
    """Nur der Background-Text: Inhalt von (…) und {…}, Abschnitt-Marken davor (wie extractBackingLyrics)."""
    out, pending = [], None
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\[[^\]]+\]$", line):
            pending = line
            continue
        teile = [(a or b).strip() for a, b in re.findall(r"\(([^()]+)\)|\{([^{}]+)\}", line) if (a or b).strip()]
        if teile:
            if pending:
                out.append(pending)
                pending = None
            out.append(" ".join(teile))
    return "\n".join(out)


# ---------------------------------------------------------------- Arabisch -> Arabizi
_AR = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ٱ": "a", "ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "7", "خ": "5",
    "د": "d", "ذ": "d", "ر": "r", "ز": "z", "س": "s", "ش": "ch", "ص": "s", "ض": "d", "ط": "t", "ظ": "d",
    "ع": "3", "غ": "gh", "ف": "f", "ڤ": "v", "ق": "9", "ك": "k", "ڭ": "g", "گ": "g", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "ة": "a", "ى": "a", "ء": "2", "ئ": "2", "ؤ": "2", "پ": "p",
    "،": ",", "؟": "?", "؛": ";",
}
_HARAKAT = {"َ": "a", "ِ": "i", "ُ": "ou", "ْ": "", "ً": "an", "ٍ": "in", "ٌ": "oun"}
_VOK = set("aeiou")


def arabizi(text):
    """Arabische Schrift -> Arabizi (Entwurf). Lateinische Zeichen bleiben wie sie sind."""
    zeilen = []
    for zeile in str(text or "").splitlines():
        woerter = []
        for wort in zeile.split(" "):
            out = ""
            for i, ch in enumerate(wort):
                vor = out[-1:] if out else ""
                if ch == "ّ":                      # Schadda: Buchstabe doppelt
                    if out and out[-1] not in _VOK:
                        out += out[-1]
                    continue
                if ch in _HARAKAT:
                    out += _HARAKAT[ch]
                    continue
                if ch == "و":                           # am Wortanfang/zwischen Vokalen "w", sonst "ou"
                    out += "w" if (not out or vor in _VOK) else "ou"
                    continue
                if ch == "ي":
                    out += "y" if (not out or vor in _VOK) else "i"
                    continue
                if ch == "ل" and i + 1 < len(wort) and wort[i + 1] in "اأإآ" and not out:
                    out += "l"
                    continue
                out += _AR.get(ch, ch)
            woerter.append(out)
        zeilen.append(" ".join(woerter))
    return "\n".join(zeilen)


# ---------------------------------------------------------------- Umschrift für den Abgleich
_ZIFFER = {"3": "a", "7": "h", "9": "k", "5": "kh", "2": "", "8": "gh", "6": "t"}


def fuer_abgleich(text):
    """Text so umschreiben, dass das A–Z-Abgleich-Modell ihn lesen kann. Klammern ( ) { } [ ] und Zeilen bleiben."""
    t = arabizi(text) if re.search(r"[؀-ۿ]", str(text or "")) else str(text or "")
    out = []
    for zeile in t.splitlines():
        if re.match(r"^\s*\[[^\]]+\]\s*$", zeile):
            out.append(zeile)
            continue
        # Ziffern nur IN Wörtern (Arabizi: "3la", "s7a"), alleinstehende Zahlen bleiben
        zeile = re.sub(r"(?<=[A-Za-z])[2356789]|[2356789](?=[A-Za-z])", lambda m: _ZIFFER.get(m.group(0), m.group(0)), zeile)
        zeile = unicodedata.normalize("NFKD", zeile)
        zeile = "".join(c for c in zeile if not unicodedata.combining(c))
        out.append(zeile)
    return "\n".join(out)


if __name__ == "__main__":
    probe = "Seeds inna stony ground (seeds inna stony ground)\n{Jah guide} mi bredren"
    print(lead_text(probe), "|", backing_text(probe))
    print(arabizi("واش نتا مزيان؟ بغيت نمشي معاك للبحر"))
    print(fuer_abgleich("Rami ya weldi, lyoum far7ana bik\nKol l3ayla tghanni"))
    print(fuer_abgleich("واش نتا مزيان"))
