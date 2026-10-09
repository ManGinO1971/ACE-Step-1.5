"""
RunPod-Load-Balancing-Einstiegspunkt fuer den bestehenden ACE-Step-1.5
REST-API-Server.

Diese Datei aendert das originale acestep-Paket NICHT - sie macht nur:
  1) importiert das bestehende FastAPI-"app"-Objekt (unveraendert),
  2) fuegt eine zusaetzliche Route "/ping" hinzu, die RunPods
     Load-Balancing-Endpoints fuer Gesundheitschecks brauchen,
  3) fuegt eine zusaetzliche Route "/v1/hieltech_lyrics" hinzu: generiert
     Reggae/Rasta-Songtexte per LLM direkt auf der RunPod-GPU (CUDA) -
     inhaltlich identisch zum lokalen generate_lyrics.py (gleiches Modell,
     gleicher System-Prompt), nur eben auf der schon warmen GPU statt auf
     dem Mac. Reiner HIELTECH-Zusatz, kein Teil des originalen ACE-Step-
     Projekts,
  4) fuegt eine zusaetzliche Route "/v1/hieltech_translate" hinzu:
     uebersetzt Songtext (inkl. Jamaica-Patois-Sonderbehandlung) per
     TranslateGemma-4B direkt auf der RunPod-GPU - inhaltlich identisch
     zum lokalen translate_lyrics.py (gleiches Modell, gleiche
     Patois-Sonderlogik/Bereinigung), nur eben auf der schon warmen GPU
     statt auf dem Mac. Braucht die Umgebungsvariable HF_TOKEN (Hugging-
     Face-Zugriffstoken mit akzeptierter Lizenz fuer das gated Modell),
     die ueber die RunPod-Endpoint-Einstellungen gesetzt wird, NICHT im
     Code steht,
  5) startet uvicorn auf dem Port, den RunPod erwartet,
  6) fuegt eine zusaetzliche Route "/v1/hieltech_vocals_postprocess" hinzu
     (7. Okt, "Vocals erzeugen" fuer die PWA): nimmt das Ergebnis des
     bereits per /release_task erzeugten flow_edit_morph-Schritts (Schritt 2
     der 7-Schritte-Kette, siehe runpod-handler.js buildVocalsReleaseTaskBody)
     entgegen und haengt Schritt 3-7 (Demucs-Nachtrennung, ffmpeg-Waermekette,
     WORLD-Resynthese, optional RVC+Denoise+Gate, optional gemessener
     Studio-Reverb) daran - 1:1 dieselben, bereits am Mac einen ganzen Testtag
     lang bestaetigten Skripte (reseparate_vocal_step.py/vocal_warmth_step.py/
     world_resynth_step.py/rvc_convert_step.py/denoise_step.py/
     gate_vocal_stem.py/stem_clean_step.py/apply_measured_reverb.py), hier nur
     per subprocess auf der RunPod-GPU statt auf dem Mac ausgefuehrt. Jeder
     Teilschritt, dessen Python-Paket auf diesem RunPod-Image (noch) fehlt
     (pyworld/rvc-python+fairseq/DeepFilterNet/pedalboard - siehe deren
     eigene requirements), wird NICHT hart abgebrochen, sondern ueberspringt
     sich selbst mit einer Warnung in der Antwort ("warnings") - das Ergebnis
     bleibt dadurch nutzbar, auch bevor/falls das Image noch nicht alle
     optionalen Pakete enthaelt.

Nichts hier hat mit Preisen/Lizenzen/Geschaeftslogik zu tun - es startet
nur den bestehenden, quelloffenen REST-Server so, dass RunPod damit reden
kann.
"""
import base64
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import asyncio
import concurrent.futures

from fastapi import Response, Request, UploadFile, Form
from acestep.api_server import app


@app.get("/ping")
async def _runpod_ping():
    return Response(status_code=200)


# =============================================================================
# HIELTECH: eigene Lyrics-Generierung auf der RunPod-GPU
# =============================================================================

_LYRICS_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
_lyrics_lock = threading.Lock()
_lyrics_model = None
_lyrics_tokenizer = None
_lyrics_device = None

_SECTION_NAMES = {
    "intro": "Intro",
    "verse": "Verse 1",
    "verse2": "Verse 2",
    "chorus2": "Chorus 2",
    "verse3": "Verse 3",
    "prechorus": "Pre-Chorus",
    "chorus": "Chorus",
    "bridge": "Bridge",
    "outro": "Outro",
}

_META_KEYWORDS = [
    "this is just", "snippet", "continue or end", "using the provided",
    "feel free", "let me know", "hope this", "here is", "here's the",
    "(no title)", "no title", "untitled",
]


def _get_lyrics_model():
    """Laedt Tokenizer/Modell beim ersten Aufruf und behaelt sie im
    Speicher (Singleton) - danach ist jeder weitere Aufruf schnell, weil
    kein erneutes Laden noetig ist, solange der Worker warm bleibt."""
    global _lyrics_model, _lyrics_tokenizer, _lyrics_device
    with _lyrics_lock:
        if _lyrics_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _lyrics_device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if _lyrics_device == "cuda" else torch.float32
            _lyrics_tokenizer = AutoTokenizer.from_pretrained(_LYRICS_MODEL_NAME)
            _lyrics_model = AutoModelForCausalLM.from_pretrained(
                _LYRICS_MODEL_NAME, dtype=dtype
            ).to(_lyrics_device)
    return _lyrics_model, _lyrics_tokenizer, _lyrics_device


def _is_meta_line(line):
    stripped = line.strip().lower().rstrip(".")
    if stripped in ("end", "the end", "fin"):
        return True
    return any(kw in stripped for kw in _META_KEYWORDS)


def _parse_and_reorder(raw_text, sections):
    """Findet Abschnitte per Label im Rohtext (auch mit Markdown-Sternchen),
    ordnet sie in die gewuenschte Reihenfolge. Identische Logik zum lokalen
    generate_lyrics.py, damit das Ergebnis gleich formatiert ist."""
    lines = raw_text.split("\n")
    section_content = {}
    current_key = sections[0] if sections else None
    buffer = []

    paren_line_pattern = re.compile(r'^[\(\[].*[\)\]]$')

    def flush():
        nonlocal buffer, current_key
        if current_key and buffer:
            clean_lines = [
                l for l in buffer
                if l.strip() and not _is_meta_line(l) and not paren_line_pattern.match(l.strip())
            ]
            text = "\n".join(clean_lines)
            if text.strip():
                section_content.setdefault(current_key, text.strip())
        buffer = []

    label_pattern = re.compile(r'^[\*_]{0,2}\s*[\(\[]?\s*([A-Za-z0-9\- ]+?)\s*[\)\]]?\s*[\*_]{0,2}\s*:?\s*$')

    for line in lines:
        stripped = line.strip()
        match = label_pattern.match(stripped)
        if match and len(match.group(1)) < 20:
            candidate = match.group(1).lower().replace(" ", "").replace("-", "")
            found_key = None
            for key in sections:
                section_label_norm = _SECTION_NAMES.get(key, key).lower().replace(" ", "").replace("-", "")
                if candidate == section_label_norm:
                    found_key = key
                    break
            if not found_key:
                for key in sections:
                    key_norm = key.lower().replace(" ", "").replace("-", "")
                    if key_norm in candidate or candidate in key_norm:
                        found_key = key
                        break
            if found_key:
                flush()
                current_key = found_key
                continue
        buffer.append(line)
    flush()

    ordered_parts = []
    for key in sections:
        if key in section_content:
            ordered_parts.append(section_content[key])

    return "\n---\n".join(ordered_parts)


@app.post("/v1/hieltech_lyrics")
async def _hieltech_generate_lyrics(request: Request):
    """Generiert Reggae/Rasta-Songtexte per LLM auf der GPU.

    Erwartet JSON-Body: {"theme": "...", "sections": ["intro","verse",...]}
    Antwort: {"lyrics": "...mit '---' zwischen den Abschnitten getrennt..."}
    """
    import torch

    body = await request.json()
    theme = (body.get("theme") or "freedom and unity").strip()
    sections = body.get("sections") or ["intro", "verse", "chorus", "outro"]

    model, tokenizer, device = _get_lyrics_model()

    section_labels = [_SECTION_NAMES.get(s, s) for s in sections]
    section_list = ", ".join(section_labels)

    system_prompt = (
        "You are a songwriter specializing in conscious African roots reggae, "
        "in the style of Alpha Blondy, Lucky Dube, and Tiken Jah Fakoly. "
        "Themes often include: corruption, police accountability, slavery and resistance, "
        "Pan-Africanism, youth struggles, spirituality, and social justice. "
        "Write plain lyrics only, no chords, no explanations, no comments about the song. "
        "Label each section clearly on its own line using parentheses, e.g. (Verse), (Chorus). "
        "Do not use markdown bold or asterisks. "
        "Only write the sections requested, nothing extra, and stop immediately after the last section."
    )
    user_prompt = (
        f"Write song lyrics about: {theme}\n"
        f"Write EXACTLY these sections, in this exact order, each labeled: {section_list}\n"
        f"Each section should be 2-3 lines. Keep sentences short and simple, 5-8 words per line. Do not include a title. Do not wrap lines in parentheses."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=550, temperature=0.85, do_sample=True, top_p=0.9
        )
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    formatted = _parse_and_reorder(raw, sections)

    return {"lyrics": formatted}


# =============================================================================
# HIELTECH: eigene Uebersetzung (inkl. Jamaica-Patois) auf der RunPod-GPU
# =============================================================================

_TRANSLATE_MODEL_NAME = "google/translategemma-4b-it"
_translate_lock = threading.Lock()
_translate_processor = None
_translate_model = None

# Codepoints 0 bis 591 (dezimal, 0x24F hex) decken Basic Latin, Latin-1
# Supplement und Latin Extended-A/B ab - reicht fuer Englisch/Patois
# inkl. gaengiger Sonderzeichen; alles darueber (Devanagari, Kyrillisch,
# CJK, Arabisch, Thai usw.) zaehlt als Schrift-Ausreisser (siehe
# _translate_looks_like_wrong_script unten). Identische Logik zum lokalen
# translate_lyrics.py.
_TRANSLATE_LATIN_MAX_CODEPOINT = 591


def _get_translate_model():
    """Laedt Prozessor/Modell beim ersten Aufruf und behaelt sie im
    Speicher (Singleton), genau wie bei der Lyrics-Generierung oben.
    HF_TOKEN kommt bewusst aus der Umgebung (RunPod-Endpoint-Einstellungen),
    nicht aus dem Code - das gated Google-Modell braucht einen Hugging-Face-
    Zugriffstoken mit akzeptierter Lizenz."""
    global _translate_processor, _translate_model
    with _translate_lock:
        if _translate_model is None:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            hf_token = os.environ.get("HF_TOKEN")
            # Debug-Ausgabe (nur Laenge/erste-letzte Zeichen, NIE der volle
            # Wert) - hilft zu erkennen ob RunPod versehentlich ein
            # Leerzeichen/Zeilenumbruch mit reinkopiert hat.
            # Bewusst auf sys.stderr statt print()/stdout, weil stdout im
            # RunPod-Log-Viewer offenbar nicht zuverlaessig ankommt,
            # waehrend unbehandelte Tracebacks (die auch ueber stderr
            # laufen) sichtbar sind.
            if hf_token:
                print(
                    f"[DEBUG hieltech_translate] HF_TOKEN vorhanden, "
                    f"Laenge={len(hf_token)}, "
                    f"repr_anfang={hf_token[:6]!r}, repr_ende={hf_token[-4:]!r}",
                    file=sys.stderr, flush=True,
                )
                hf_token = hf_token.strip()
            else:
                print("[DEBUG hieltech_translate] HF_TOKEN ist LEER/None in os.environ!", file=sys.stderr, flush=True)
            token_debug = (
                f"HF_TOKEN-Debug: vorhanden={bool(hf_token)}, "
                f"Laenge={len(hf_token) if hf_token else 0}, "
                f"anfang={(hf_token[:6] if hf_token else None)!r}, "
                f"ende={(hf_token[-4:] if hf_token else None)!r}"
            )
            try:
                _translate_processor = AutoProcessor.from_pretrained(_TRANSLATE_MODEL_NAME, token=hf_token)
                # Update (12. Sept): laeuft jetzt wieder bewusst auf der GPU,
                # in bfloat16 statt float32 (halbiert den Speicherbedarf auf
                # ca. 8GB statt ca. 16GB). Vorher (CPU/float32) war nur eine
                # Interimsloesung, weil die ACE-Step-Musikmodelle (DiT+5Hz-LM,
                # ~19GB) den alten 19,6GB-GPU-Tier (A4500) bereits komplett
                # ausgelastet hatten - CPU-Inferenz eines 4B-Modells war fuer
                # echte (v.a. mobile) Nutzer unzumutbar langsam. Loesung:
                # Umstellung des gesamten RunPod-Endpoints auf einen 48GB-
                # GPU-Tier (A6000/A40), dadurch ist jetzt genug Platz fuer
                # beide Modelle gleichzeitig auf der GPU (~19GB + ~8GB =
                # ~27GB von 48GB, komfortabler Puffer fuer KV-Cache etc.).
                _translate_model = AutoModelForCausalLM.from_pretrained(
                    _TRANSLATE_MODEL_NAME, dtype=torch.bfloat16, token=hf_token
                )
                if torch.cuda.is_available():
                    _translate_model = _translate_model.to("cuda")
            except Exception as e:
                # token_debug wird bewusst der Fehlermeldung angehaengt,
                # damit die Info auch dann in der RunPod-Traceback-Ausgabe
                # landet, wenn separate print()/stderr-Zeilen im Log-Viewer
                # aus irgendeinem Grund nicht ankommen.
                raise RuntimeError(f"{token_debug} | Original-Fehler: {e}") from e
    return _translate_processor, _translate_model


def _translate_to_model_device(model, inputs):
    if next(model.parameters()).is_cuda:
        return {k: v.to("cuda") for k, v in inputs.items()}
    return inputs


def _translate_normal(processor, model, text, target_lang_code, source_lang_code="en", sample=False, temperature=0.7):
    import torch
    messages = [
        {'role': 'user', 'content': [
            {'type': 'text', 'source_lang_code': source_lang_code, 'target_lang_code': target_lang_code, 'text': text}
        ]}
    ]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
    inputs = _translate_to_model_device(model, inputs)
    gen_kwargs = {"max_new_tokens": 200}
    if sample:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    result = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return result.strip()


def _translate_lang_code_candidates(code):
    """Nutzerfund Sept 2026 (identisch zum lokalen translate_lyrics.py):
    manche Sprachcodes (z.B. "zh-CN") brechen im Modell-Chat-Template ab,
    obwohl andere im exakt gleichen Format klappen - deshalb mehrere
    plausible Schreibweisen der Reihe nach probieren."""
    seen = []

    def add(c):
        if c and c not in seen:
            seen.append(c)

    add(code)
    if "-" in code:
        add(code.replace("-", "_"))
    if "_" in code:
        add(code.replace("_", "-"))
    base = code.replace("_", "-").split("-")[0]
    add(base)
    if base == "zh":
        add("zh-Hans")
        add("zh_Hans")
        add("zh-Hans-CN")
    return seen


def _translate_normal_with_fallback(processor, model, text, target_lang_code, source_lang_code="en"):
    last_err = None
    for code in _translate_lang_code_candidates(target_lang_code):
        try:
            result = _translate_normal(processor, model, text, code, source_lang_code)
            if result:
                return result
        except Exception as e:
            last_err = e
            continue
    try:
        result = _translate_normal(processor, model, text, target_lang_code, source_lang_code, sample=True, temperature=0.7)
        if result:
            return result
    except Exception as e:
        last_err = e
    # Letzte Sicherheit: lieber unuebersetzten Text als ein kompletter
    # Absturz - identisch zum lokalen Verhalten am Mac.
    return text


def _translate_patois(processor, model, text, sample=False, temperature=0.8, num_beams=1, source_lang_name="English"):
    import torch
    # source_lang_name (Runde 75): Standard bleibt "English", da die
    # Songtext-Uebersetzung (Haupt-Anwendungsfall dieser Funktion) den
    # Songtext immer zuerst auf Englisch erzeugt. Der Admin-Systemhinweis
    # (siehe /v1/hieltech_translate_text) schreibt dagegen auf Deutsch und
    # gibt hier "German" durch, damit der Prompt korrekt beschreibt, aus
    # welcher Sprache tatsaechlich uebersetzt wird.
    prompt = (
        f"<start_of_turn>user\nTranslate this {source_lang_name} text into authentic Jamaican Patois "
        f"(Jamaican Creole), using real Patois vocabulary and grammar, not just {source_lang_name} with "
        f"an accent. Output ONLY the translated lines, nothing else -- no explanation, no "
        f"commentary, no intro phrase:\n{text}<end_of_turn>\n<start_of_turn>model\n"
    )
    inputs = processor.tokenizer(prompt, return_tensors='pt')
    inputs = _translate_to_model_device(model, inputs)
    gen_kwargs = {"max_new_tokens": 200}
    if sample:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    elif num_beams > 1:
        gen_kwargs.update({"num_beams": num_beams, "early_stopping": True})
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    result = processor.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return _clean_patois_output(result)


def _clean_patois_output(text):
    lines = text.split("\n")
    cleaned = []
    skip_rest = False
    meta_markers = [
        "explanation of choices", "explanation:", "note:", "alright, listen",
        "this translation aims", "**explanation", "here's the translation",
        "translated text:", "patois translation:",
    ]
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped:
            continue
        if any(m in lower for m in meta_markers):
            skip_rest = True
            continue
        if skip_rest:
            continue
        if stripped.startswith("*") or stripped.startswith("#"):
            continue
        if len(stripped) > 1 and stripped[0] in ('"', chr(39)) and stripped[-1] == stripped[0]:
            stripped = stripped[1:-1].strip()
        cleaned.append(stripped)
    result = "\n".join(cleaned)
    if len(result) > 1 and result[0] in ('"', chr(39)) and result[-1] == result[0]:
        result = result[1:-1].strip()
    return result


# Nutzerwunsch (21. Sept, Runde 73): Darija ("ary") hatte bisher UEBERHAUPT
# keine Sonderbehandlung (lief einfach durch den generischen
# _translate_normal_with_fallback-Zweig unten) - das produziert bestenfalls
# formelles Hocharabisch in arabischer Schrift, aber niemals die im Norden
# Afrikas beim Schreiben (Chat/SMS/Songtexte) tatsaechlich gebraeuchliche
# "Arabizi"-Umschrift mit lateinischen Buchstaben + Ziffern fuer Laute ohne
# lateinische Entsprechung (3=Ain, 2=Hamza, 7=Ha, 9=Qaf, 5=Kha). Nutzer
# wollte hier ausdruecklich eine Wahlmoeglichkeit - script_style ("latin"
# Standard laut Nutzerentscheidung, oder "arabic") kommt vom Frontend durch
# server.js/runpod-handler.js bis hierher durchgereicht. Gleiches
# Prompt-basiertes Vorgehen wie bei Jamaica-Patois oben (TranslateGemma kennt
# den ary-Sprachcode zwar, aber weder Darija-Umgangssprache noch erst recht
# nicht die Arabizi-Konvention zuverlaessig als eigenstaendiges Sprachziel).
def _translate_darija(processor, model, text, script_style="latin", sample=False, temperature=0.8, num_beams=1):
    import torch
    if script_style == "arabic":
        prompt = (
            f"<start_of_turn>user\nTranslate this English text into authentic Moroccan/Algerian "
            f"Darija (North African Arabic dialect), written in Arabic script - real spoken Darija "
            f"vocabulary and grammar, not formal Modern Standard Arabic. Output ONLY the translated "
            f"lines, nothing else -- no explanation, no commentary, no intro phrase:\n{text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    else:
        prompt = (
            f"<start_of_turn>user\nTranslate this English text into authentic Moroccan/Algerian "
            f"Darija (North African Arabic dialect), written in the Latin \"Arabizi\" chat alphabet - "
            f"the way North Africans actually write it online and in song lyrics: plain Latin letters, "
            f"using the digit 3 for the letter ain, 2 for hamza, 7 for ha, 9 for qaf, and 5 for kha "
            f"wherever those sounds occur. Do not use Arabic script. Use real Darija vocabulary and "
            f"grammar, not just French or English with an accent. Match this exact style:\n"
            f"Rami ya weldi, lyoum far7ana bik\nKol l3ayla tghanni w tfar7 m3ak\nKbert chwiya, w zadt "
            f"lfar7a fik\nAllah y7afdek w ykhalik lya\n"
            f"Output ONLY the translated lines, nothing else -- no explanation, no commentary, no intro "
            f"phrase:\n{text}<end_of_turn>\n<start_of_turn>model\n"
        )
    inputs = processor.tokenizer(prompt, return_tensors='pt')
    inputs = _translate_to_model_device(model, inputs)
    gen_kwargs = {"max_new_tokens": 200}
    if sample:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    elif num_beams > 1:
        gen_kwargs.update({"num_beams": num_beams, "early_stopping": True})
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    result = processor.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return _clean_patois_output(result)


# Arabische Schrift liegt im Unicode-Block U+0600-U+06FF (Basic Arabic -
# deckt fuer diese Pruefung ausreichend ab, Praesentationsformen/Supplement
# nicht extra noetig).
_ARABIC_SCRIPT_MIN = 0x0600
_ARABIC_SCRIPT_MAX = 0x06FF


def _translate_darija_wrong_script(text, script_style):
    """Wie _translate_looks_like_wrong_script, aber script-bewusst: im
    Arabizi-Modus (script_style='latin') zaehlt zu VIEL Nicht-Latein als
    Fehler (wie beim Patois-Check); im Arabisch-Modus ist es genau
    umgekehrt - dort zaehlt zu WENIG arabische Schrift als Fehler (Modell
    hat die Anweisung ignoriert und z.B. auf Englisch/Franzoesisch
    geantwortet)."""
    stripped = re.sub(r'\s+', '', text)
    if not stripped:
        return False
    if script_style == "arabic":
        arabic_chars = sum(1 for ch in stripped if _ARABIC_SCRIPT_MIN <= ord(ch) <= _ARABIC_SCRIPT_MAX)
        return (arabic_chars / len(stripped)) < 0.3
    non_latin = sum(1 for ch in stripped if ord(ch) > _TRANSLATE_LATIN_MAX_CODEPOINT)
    return non_latin / len(stripped) > 0.15


def _translate_looks_like_wrong_script(text):
    """Identische Logik zum lokalen translate_lyrics.py: Jamaica-Patois
    laeuft ueber einen rohen Prompt statt das strukturierte Uebersetzungs-
    Template und driftet dadurch gelegentlich in eine andere Sprache/
    Schrift ab - ein hoher Anteil nicht-lateinischer Zeichen ist ein
    zuverlaessiges Anzeichen dafuer."""
    stripped = re.sub(r'\s+', '', text)
    if not stripped:
        return False
    non_latin = sum(1 for ch in stripped if ord(ch) > _TRANSLATE_LATIN_MAX_CODEPOINT)
    return non_latin / len(stripped) > 0.15


@app.post("/v1/hieltech_translate")
async def _hieltech_translate(request: Request):
    """Uebersetzt Songtext (Abschnitte per '---' getrennt) auf der GPU -
    identische Logik zum lokalen translate_lyrics.py (Mac).

    Erwartet JSON-Body: {"text": "...---...", "target_lang_code": "jam"}
    Antwort: {"translated_lyrics": "...uebersetzt, '---'-Struktur bleibt erhalten..."}
    """
    body = await request.json()
    text = body.get("text") or ""
    target_lang_code = (body.get("target_lang_code") or "en").strip()
    # script_style (Runde 73): nur fuer target_lang_code=="ary" relevant,
    # "latin" (Arabizi mit 3/2/7/9/5) oder "arabic" (arabische Schrift).
    # Unbekannter/leerer Wert faellt sicherheitshalber auf "latin" zurueck
    # (Nutzerentscheidung fuer den Standard), statt den Request abzulehnen -
    # so bleibt auch ein aelteres Frontend, das das Feld noch gar nicht
    # mitschickt, funktionsfaehig.
    script_style = (body.get("script_style") or "latin").strip().lower()
    if script_style not in ("latin", "arabic"):
        script_style = "latin"

    if target_lang_code == "en" or not text.strip():
        return {"translated_lyrics": text}

    processor, model = _get_translate_model()

    sections = text.split("---")
    translated_sections = []
    for section in sections:
        lines = [l for l in section.split("\n") if l.strip()]
        section_text = "\n".join(lines)
        if not section_text.strip():
            continue
        if target_lang_code == "jam":
            # num_beams=4 (Update 12. Sept, zurueckgestellt): jetzt wieder
            # auf der GPU (siehe _get_translate_model), Beam-Search mit
            # mehreren Kandidaten ist dort schnell genug und liefert
            # bessere Patois-Qualitaet als num_beams=1 (das war nur die
            # CPU-Interimsloesung).
            translated = _translate_patois(processor, model, section_text, num_beams=4)
            if _translate_looks_like_wrong_script(translated):
                retry = _translate_patois(processor, model, section_text, sample=True, temperature=0.8)
                translated = retry if not _translate_looks_like_wrong_script(retry) else section_text
        elif target_lang_code == "ary":
            # Nutzerwunsch (21. Sept, Runde 73): siehe _translate_darija-
            # Kommentar oben - gleiches Retry-Muster wie bei Jamaica-Patois,
            # nur script-bewusst statt immer "erwarte Latein".
            translated = _translate_darija(processor, model, section_text, script_style=script_style, num_beams=4)
            if _translate_darija_wrong_script(translated, script_style):
                retry = _translate_darija(processor, model, section_text, script_style=script_style, sample=True, temperature=0.8)
                translated = retry if not _translate_darija_wrong_script(retry, script_style) else section_text
        else:
            translated = _translate_normal_with_fallback(processor, model, section_text, target_lang_code)
        translated_sections.append(translated)

    return {"translated_lyrics": "\n---\n".join(translated_sections)}


# Menschlicher Name je Sprachcode, NUR fuer die Prompt-Formulierung unten
# (_translate_patois braucht "Translate this <X> text..." - das eigentliche
# NLLB-artige Modell (_translate_normal_with_fallback) nutzt stattdessen den
# rohen source_lang_code direkt, keine Namen noetig).
_PROMPT_LANG_NAME = {
    "de": "German", "en": "English", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
    "pl": "Polish", "el": "Greek", "ru": "Russian", "ar": "Arabic",
    "hi": "Hindi", "sw": "Swahili", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "vi": "Vietnamese", "th": "Thai",
}


@app.post("/v1/hieltech_translate_text")
async def _hieltech_translate_text(request: Request):
    """Einfache, generische Text-Uebersetzung - KEIN Songtext (keine '---'-
    Abschnitte, keine Patois/Darija-Sonderpfade fuer kreative Songtext-
    Formulierung). Eingefuehrt fuer den Admin-Systemhinweis (Nutzerwunsch,
    21. Sept, Runde 75): der Admin schreibt IMMER auf Deutsch, der Text wird
    hier in jede App-UI-Sprache uebersetzt (inkl. "jam"/Jamaica Patois, das
    als einzige der 20 UI-Sprachen eine eigene kreative Vorlage statt des
    normalen Uebersetzungsmodells braucht - "ary"/Darija ist dagegen KEINE
    App-UI-Sprache und taucht hier nie als Zielsprache auf).

    Erwartet JSON-Body: {"text": "...", "source_lang_code": "de", "target_lang_code": "fr"}
    Antwort: {"translated_text": "..."}
    """
    body = await request.json()
    text = (body.get("text") or "").strip()
    source_lang_code = (body.get("source_lang_code") or "de").strip().lower()
    target_lang_code = (body.get("target_lang_code") or "en").strip().lower()

    if not text or target_lang_code == source_lang_code:
        return {"translated_text": text}

    processor, model = _get_translate_model()

    if target_lang_code == "jam":
        source_name = _PROMPT_LANG_NAME.get(source_lang_code, "English")
        translated = _translate_patois(processor, model, text, num_beams=4, source_lang_name=source_name)
        if _translate_looks_like_wrong_script(translated):
            retry = _translate_patois(processor, model, text, sample=True, temperature=0.8, source_lang_name=source_name)
            translated = retry if not _translate_looks_like_wrong_script(retry) else text
    else:
        translated = _translate_normal_with_fallback(processor, model, text, target_lang_code, source_lang_code=source_lang_code)

    return {"translated_text": translated}


# =============================================================================
# HIELTECH: "Vocals erzeugen" Schritt 1 (Vortrennung+Bereinigung, _hieltech_
# vocals_preprocess) + Schritt 3-7 (Nachbearbeitung der Schritt-2-flow_edit_
# morph-Ausgabe, _hieltech_vocals_postprocess) auf der RunPod-GPU (7.-8. Okt)
# =============================================================================
#
# Projekt-Wurzel: dieselbe wie diese Datei (runpod_entrypoint.py liegt immer
# im ACE-Step-1.5-Projektordner, genau wie reseparate_vocal_step.py usw. auf
# dem Mac) - damit funktionieren alle relativen Pfade (z.B. rvc_convert_step.py
# -> rvc_models/voices/<modell>.pth) unveraendert, sofern diese Dateien beim
# Docker-Image-Build/Deploy in denselben Ordner wie diese Datei kopiert werden.
_HIELTECH_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Separate, isolierte Python-3.10-venv NUR fuer rvc_convert_step.py (siehe
# Dockerfile.runpod) - durch einen isolierten Abhaengigkeits-Test (8. Okt,
# ohne jede GPU-Erzeugung, siehe Projektnotiz Runde 148) bestaetigt: der von
# "rvc-python" verlangte Stack (fairseq==0.12.2 + hydra-core==1.0.7 +
# omegaconf==2.0.6) importiert unter Python 3.11 NICHT - hydra-core 1.0.7
# nutzt ein Dataclass-Feld mit "mutable default", das Python 3.11 strikter
# prueft als 3.10 (ValueError beim Import von "fairseq", weil fairseq beim
# Laden automatisch Hydra initialisiert - sys.executable ist hier aber
# Python 3.11, siehe pyproject.toml requires-python). fairseq selbst baut
# unter 3.11 einwandfrei (das urspruenglich vermutete "version.txt"-Problem
# betraf nur die veraltete Version 0.12.1, NICHT die hier benoetigte
# 0.12.2) - das eigentliche Problem ist ausschliesslich diese
# Hydra/Python-3.11-Inkompatibilitaet. Deshalb bewusst eine zweite, komplett
# getrennte venv mit Python 3.10 statt die ganze Hauptumgebung
# herunterzustufen. Existiert diese venv auf einem (noch nicht
# neugebauten) Image nicht, faellt _hieltech_run_step() automatisch auf
# sys.executable zurueck - das schlaegt dann wie bisher nur mit der
# bestehenden Warnung fehl, bricht die Anfrage nicht ab.
_HIELTECH_RVC_VENV_PYTHON = "/opt/acestep_rvc_venv/bin/python"

# NEU (9. Okt 2026, Nutzerfrage "gibt es auch Kunden mit 5-Minuten-Songs
# oder sogar mehr - passt FLAC trotzdem?"): FLAC alleine (siehe
# _hieltech_wav_to_flac_file unten) verschiebt RunPod's 30-MB-Grenze nur auf
# ca. 7-8 Minuten, loest das Problem fuer laengere Songs also NICHT
# zuverlaessig. Dieses Verzeichnis ist die Grundlage fuer die echte Loesung:
# "Worker Affinity" (von RunPod's eigenem Load-Balancer dokumentiert,
# Header "X-Runpod-Worker-Id: strict-resume <id>") erlaubt es, einen
# Folge-Request an GENAU denselben Worker zu pinnen, der einen vorherigen
# Request bearbeitet hat. Dadurch kann Schritt 1 sein Ergebnis einfach HIER
# lokal ablegen und nur einen (winzigen) Dateipfad zurueckgeben, statt die
# Audio-Bytes selbst ueber das Netzwerk zu schicken - Schritt 2
# (/release_task, ACE-Step's EIGENE, gemeinsam genutzte Route) liest diesen
# Pfad direkt wieder ein, weil sie bereits einen "src_audio_path"-Parameter
# unterstuetzt (siehe release_task_audio_paths.py - bewusst NICHT
# veraendert, siehe AGENTS.md "keep edits minimal"). Das funktioniert nur,
# weil validate_audio_path() dort NUR Pfade innerhalb von
# tempfile.gettempdir() akzeptiert - deshalb MUSS dieser Ordner dort drunter
# liegen, nicht z.B. auf dem (Netzwerk-, aber fuer diese Pruefung
# ausserhalb-von-temp liegenden) RunPod-Checkpoints-Volume.
_HIELTECH_HANDOFF_DIR = os.path.join(tempfile.gettempdir(), "hieltech_vocals_handoff")
os.makedirs(_HIELTECH_HANDOFF_DIR, exist_ok=True)
# Nur fuer _hieltech_fetch_chunk unten: schliesst Pfad-Ausbrueche ueber einen
# manipulierten "token"-Query-Parameter aus, BEVOR er in einen Dateipfad
# eingesetzt wird - gueltige Tokens sind IMMER uuid.uuid4().hex (32 Hex-
# Zeichen, siehe _hieltech_handoff_store unten).
_HIELTECH_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")


def _hieltech_handoff_cleanup_stale(max_age_s=3600):
    """Loescht Handoff-Dateien, die aelter als max_age_s sind - Sicherheitsnetz
    gegen ein Festplatten-Leck, falls Node nach einem Fehler nie den
    Abschluss-Aufruf macht, der eine Ergebnisdatei regulaer entfernt (siehe
    done=true bei _hieltech_fetch_chunk). Wirft nie, rein best effort."""
    try:
        now = time.time()
        for name in os.listdir(_HIELTECH_HANDOFF_DIR):
            path = os.path.join(_HIELTECH_HANDOFF_DIR, name)
            try:
                if now - os.path.getmtime(path) > max_age_s:
                    os.remove(path)
            except OSError:
                pass
    except Exception:  # noqa: BLE001 - rein defensiv, darf eine Anfrage nie stoeren
        pass


def _hieltech_handoff_store(source_path, ext):
    """Kopiert source_path (innerhalb eines work_dir, das die aufrufende
    Route per `finally: shutil.rmtree(...)` gleich wieder loescht) in
    _HIELTECH_HANDOFF_DIR unter einem frischen, zufaelligen Token und gibt
    (handoff_path, token) zurueck - die Kopie ueberlebt also das Aufraeumen
    des work_dir und bleibt fuer einen per Worker-Affinitaet gepinnten
    Folge-Request auffindbar."""
    _hieltech_handoff_cleanup_stale()
    token = uuid.uuid4().hex
    handoff_path = os.path.join(_HIELTECH_HANDOFF_DIR, f"{token}.{ext}")
    shutil.copyfile(source_path, handoff_path)
    return handoff_path, token


def _hieltech_run_step(args, timeout=1800, python_path=None):
    """Fuehrt ein Schritt-Skript per subprocess aus. Nutzt standardmaessig
    denselben Python-Interpreter wie dieser Server (kein Conda-
    Umgebungswechsel wie auf dem Mac noetig - ein Docker-Image hat fuer die
    meisten Schritte nur eine Umgebung); `python_path` erlaubt optional
    einen ANDEREN Interpreter fuer einzelne Schritte mit eigener,
    inkompatibler Abhaengigkeit (siehe _HIELTECH_RVC_VENV_PYTHON oben) - ist
    der angegebene Pfad nicht vorhanden, wird transparent auf
    sys.executable zurueckgefallen. Gibt (erfolgreich, stdout+stderr)
    zurueck, wirft NIE eine Exception - ein fehlendes optionales Paket
    (ImportError in dem jeweiligen Skript) soll diesen Teilschritt nur
    ueberspringen, nicht die ganze Anfrage abbrechen.
    """
    interpreter = python_path if python_path and os.path.isfile(python_path) else sys.executable
    cmd = [interpreter] + args
    # NEU (9. Okt 2026, Nutzerfrage "nehmen die 4 CPU-Schritte soviel Zeit"):
    # bisher gab es GAR KEINE Zeitmessung pro Teilschritt - jede bisherige
    # Einschaetzung dazu war geraten, nicht gemessen. Jeder Aufruf landet
    # jetzt mit Name+Dauer im RunPod-eigenen Worker-Log (print auf stderr,
    # dort landen auch die Teilskript-Ausgaben selbst), damit der naechste
    # echte Testlauf eine ECHTE Zeitaufschluesselung liefert statt einer
    # Schaetzung.
    step_name = os.path.basename(args[0]) if args else "?"
    started_at = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, cwd=_HIELTECH_PROJECT_ROOT, capture_output=True, text=True, timeout=timeout
        )
        ok = result.returncode == 0
        duration_s = time.perf_counter() - started_at
        print(f"[HIELTECH-Zeitmessung] {step_name}: {duration_s:.1f}s (ok={ok})", file=sys.stderr, flush=True)
        # NEU (10. Okt 2026): wichtige Hinweise der Teilskripte (welches
        # Trennmodell lief, Schnellweg/Rueckfall, Parallel-Kerne) zusaetzlich
        # ins RunPod-Worker-Log durchreichen - die Skript-Ausgabe selbst wird
        # sonst nur im Fehlerfall zurueckgegeben und war im Log unsichtbar.
        for line in ((result.stdout or "") + (result.stderr or "")).splitlines():
            if any(m in line for m in ("[Demucs-Schnellweg]", "Trenne Gesang", "Rückfall", "parallel auf")):
                print(f"[HIELTECH-Info] {step_name}: {line.strip()[:300]}", file=sys.stderr, flush=True)
        return ok, (result.stdout or "") + (result.stderr or "")
    except Exception as e:  # noqa: BLE001 - bewusst breit, siehe Docstring
        duration_s = time.perf_counter() - started_at
        print(f"[HIELTECH-Zeitmessung] {step_name}: {duration_s:.1f}s (ok=False, Ausnahme={e})", file=sys.stderr, flush=True)
        return False, str(e)


def _hieltech_wav_to_flac_file(wav_path):
    """Wandelt eine WAV-Datei VERLUSTFREI in FLAC um und gibt (pfad, "flac")
    zurueck - bei jedem Fehlschlag faellt die Funktion transparent auf die
    ungewandelte Original-WAV zurueck (wav_path, "wav").

    Grund (9. Okt 2026, Nutzerwunsch "wav mit 30mb konvertieren vor die
    bearbeitung, um das problem zu beseitigen, oder die qualitaet wird
    schlecht... nur fuer vocal erzeugen"): RunPod's Load-Balancer verwirft
    Anfragen/Antworten ueber 30 MB STILL, bevor sie diesen FastAPI-Server
    ueberhaupt erreichen (siehe runpod-handler.js - dort landet das nur als
    HTTP 502/524, ohne jede eigene Fehlermeldung von uns). FLAC ist - anders
    als MP3 - bit-exakt verlustfrei rekonstruierbar, erfuellt also genau die
    vom Nutzer geforderte Bedingung "qualitaet wird nicht schlechter". Per
    echtem Test (9. Okt) gemessen: reduziert eine 5-Minuten-Vocals-Spur von
    ~26 MB auf ~19-22 MB, je nach Rauschanteil - das verschiebt RunPod's
    30-MB-Grenze auf ca. 7-8 Minuten, beseitigt sie aber NICHT fuer beliebig
    lange Songs. Die eigentliche, laengenunabhaengige Loesung dafuer ist die
    Worker-Affinitaet-Weitergabe (_HIELTECH_HANDOFF_DIR oben) bzw. das
    Chunk-Abholen (_hieltech_fetch_chunk unten) - diese Funktion bleibt
    trotzdem sinnvoll, weil eine kleinere Datei dort schneller zu kopieren/
    abzuholen ist. Nutzt ffmpeg per subprocess (im Docker-Image bereits per
    apt installiert, siehe Dockerfile.runpod) statt einer zusaetzlichen
    Python-Bibliothek. Bewusst NUR in der Vocals-Kette eingesetzt
    (_hieltech_vocals_preprocess/_postprocess) - die Song-/Instrumental-
    Erzeugung (generate_song/generate_instrumental, downloadAudioBuffer) ist
    davon unberuehrt, die liefern ohnehin schon komprimiertes mp3/flac
    direkt vom Modell, kein WAV ueber diese Routen.
    """
    flac_path = os.path.splitext(wav_path)[0] + "_hieltech.flac"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-compression_level", "8", flac_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and os.path.isfile(flac_path):
            return flac_path, "flac"
        print(
            "[HIELTECH] FLAC-Konvertierung fehlgeschlagen, verwende WAV unverändert: " + (result.stderr or "")[-300:],
            file=sys.stderr, flush=True,
        )
    except Exception as e:  # noqa: BLE001 - siehe Docstring, nie die Anfrage abbrechen
        print(f"[HIELTECH] FLAC-Konvertierung fehlgeschlagen ({e}), verwende WAV unverändert.", file=sys.stderr, flush=True)

    return wav_path, "wav"


@app.post("/v1/hieltech_vocals_preprocess")
async def _hieltech_vocals_preprocess(audio: UploadFile):
    """Schritt 1 der "Vocals erzeugen"-Kette auf dem ROHEN Original-Song,
    BEVOR Schritt 2 (flow_edit_morph, per /release_task) ueberhaupt laeuft.

    Zweite Nutzer-Korrektur (8. Okt 2026, Projektnotiz Runde 150, ersetzt die
    erste Korrektur aus Runde 149): nicht Demucs+DeepFilterNet allein,
    sondern Demucs htdemucs_ft (--two-stems=vocals) -> UVR-DeEcho-DeReverb
    (Hall raus) -> UVR-De-Echo-Aggressive (Echo raus) -> Noise-Gate liefert
    die tatsaechlich am Testtag (7. Okt, vor dieser Session) bestaetigte
    Eingabe fuer Schritt 2 (vocal_separate_uvr_clean_step.py+gate_vocal_
    stem.py, identisch zur Kette in ace_step_web_server.py auf dem Mac,
    siehe _isolate_vocals_flow_edit). Woertlich vom Nutzer: "was gut ist
    soll kein reverb im vocals sein, damit den reconstruktion besser
    klappt, und nicht was schief laeuft wegen den hall." stem_clean_step.py
    (Demucs htdemucs PLAIN+DeepFilterNet) laeuft weiterhin parallel, nur
    noch fuer instrumental_full/residue (optional, aktuell von keinem
    PWA-Schritt ausgewertet) bzw. als zweite Rueckfallstufe, falls die
    UVR-Kette fehlschlaegt.

    Erwartet multipart/form-data: audio - der rohe Original-Song (noch kein
    flow_edit_morph gelaufen).

    Antwort (JSON) bei Erfolg: {"ok": true, "handoff_path": "...",
      "audio_ext": "flac"|"wav", "instrumental_full_base64": null,
      "residue_base64": null}. NEU (9. Okt 2026, Nutzerfrage "passt das auch
      bei 5 Minuten oder mehr?"): statt die UVR-gegatete Spur (bzw.
      ersatzweise die DeepFilterNet-Spur aus stem_clean_step.py) als Bytes
      ueber das Netzwerk zu schicken (was bei langen Songs selbst als FLAC
      irgendwann wieder an RunPod's 30-MB-Antwortgrenze stoesst), liegt sie
      jetzt lokal auf DIESEM Worker unter handoff_path (siehe
      _HIELTECH_HANDOFF_DIR oben) - runpod-handler.js muss den naechsten
      Aufruf (/release_task) per "Worker Affinity"
      (X-Runpod-Worker-Id: strict-resume <id>) an GENAU diesen Worker
      pinnen und src_audio_path=handoff_path statt einer Datei-Upload
      mitschicken. instrumental_full_base64/residue_base64 bleiben NEU
      (9. Okt 2026) immer null: runpod-handler.js liest sie ohnehin nie aus,
      reiner Ballast.
    Bei Fehlschlag (UVR+DeepFilterNet ODER die Handoff-Ablage selbst):
      {"ok": false, "error": "...", "log": "..."} - der Aufrufer
      (runpod-handler.js) faellt dann auf das rohe Original als Schritt-2-
      Eingabe zurueck, bricht die Anfrage NICHT ab (gleiche "graceful
      degradation" wie bei Schritt 3-7 unten).
    """
    # NEU (9. Okt 2026, Fehler "Verbindung zum Server fehlgeschlagen"/HTTP 524):
    # die eigentliche Arbeit laeuft jetzt in einem Hintergrund-Thread statt
    # direkt im asyncio-Event-Loop - siehe _hieltech_vocals_preprocess_sync.
    src_bytes = await audio.read()
    return await asyncio.to_thread(_hieltech_vocals_preprocess_sync, src_bytes, audio.filename)


def _hieltech_vocals_preprocess_sync(src_bytes, filename):
    """Synchroner Teil von /v1/hieltech_vocals_preprocess.

    WARUM ein eigener Thread (9. Okt 2026, Ursache fuer HTTP 524 + die
    "kompletten Worker-Neustarts" in den RunPod-Logs): die Route war
    "async def", rief darin aber das BLOCKIERENDE subprocess.run() auf. Damit
    stand der komplette Server (ein einziger uvicorn-Prozess, ein einziger
    Event-Loop) fuer die ganze Dauer von Schritt 1 bzw. Schritt 3-7 still -
    auch /ping hat in dieser Zeit NICHT mehr geantwortet. RunPod's Load-
    Balancer haelt einen Worker, der /ping nicht beantwortet, fuer
    ungesund und startet ihn neu (= der beobachtete volle Modell-Reload),
    und jede gerade an diesen Worker gepinnte Anfrage (Worker-Affinitaet)
    haengt dann, bis Cloudflare mit 524 aufgibt. Im Thread bleibt der
    Event-Loop frei: /ping, /query_result und /v1/hieltech_fetch_chunk
    antworten waehrenddessen ganz normal.
    """
    work_dir = tempfile.mkdtemp(prefix="hieltech_vocals_pre_")
    try:
        src_ext = os.path.splitext(filename or "source.wav")[1] or ".wav"
        src_path = os.path.join(work_dir, f"source{src_ext}")
        with open(src_path, "wb") as f:
            f.write(src_bytes)

        # NEU (9. Okt 2026, Geschwindigkeit): stem_clean_step.py (Demucs
        # htdemucs + DeepFilterNet) lief bisher IMMER ZUERST und komplett,
        # obwohl sein Ergebnis nur gebraucht wird, wenn die UVR-Kette
        # scheitert (instrumental_full/residue wertet seit dem 9. Okt kein
        # PWA-Schritt mehr aus, siehe Antwort unten: beide immer null). Es
        # laeuft jetzt nur noch als echte Rueckfallstufe NACH einem
        # UVR-Fehlschlag - gleiche Kette, gleiches Ergebnis, eine komplette
        # Demucs+DeepFilterNet-Runde weniger Wartezeit.
        clean_log = ""
        vocals_clean_fallback = None

        uvr_dir = os.path.join(work_dir, "uvr_clean")
        uvr_ok, uvr_log = _hieltech_run_step([
            os.path.join(_HIELTECH_PROJECT_ROOT, "vocal_separate_uvr_clean_step.py"),
            src_path, uvr_dir,
        ], timeout=2400)
        vocals_uvr_clean = os.path.join(uvr_dir, "vocals_uvr_clean.wav")
        final_vocals = None
        if uvr_ok and os.path.isfile(vocals_uvr_clean):
            vocals_gated = os.path.join(uvr_dir, "vocals_gated.wav")
            gate_ok, gate_log = _hieltech_run_step([
                os.path.join(_HIELTECH_PROJECT_ROOT, "gate_vocal_stem.py"),
                "--input", vocals_uvr_clean, "--output", vocals_gated,
            ], timeout=300)
            final_vocals = vocals_gated if gate_ok and os.path.isfile(vocals_gated) else vocals_uvr_clean

        if not final_vocals:
            # Zweite Rueckfallstufe (nur noch hier, siehe Kommentar oben).
            clean_ok, clean_log = _hieltech_run_step([
                os.path.join(_HIELTECH_PROJECT_ROOT, "stem_clean_step.py"),
                src_path, work_dir,
            ])
            candidate = os.path.join(work_dir, "vocals_clean.wav")
            if clean_ok and os.path.isfile(candidate):
                vocals_clean_fallback = candidate
            final_vocals = vocals_clean_fallback

        if not final_vocals:
            return {
                "ok": False,
                "error": "vocal_preprocess_failed",
                "message": "Schritt 1 (UVR-Dereverb/Deecho und DeepFilterNet-Bereinigung) fehlgeschlagen.",
                "log": (uvr_log[-1000:] + "\n---\n" + clean_log[-1000:]),
            }

        # NEU (9. Okt 2026, Nutzerfrage "passt FLAC auch bei 5 Minuten oder
        # mehr?"): statt die Bytes inline zurueckzuschicken (frueher per
        # _hieltech_wav_to_flac_b64, stoesst bei langen Songs selbst als FLAC
        # irgendwann wieder an die 30-MB-Grenze), landet das Ergebnis jetzt
        # im Handoff-Ordner - siehe Routen-Docstring oben und
        # _HIELTECH_HANDOFF_DIR weiter oben in der Datei fuer die
        # Begruendung ("Worker Affinity"). Schlaegt die Ablage ausnahmsweise
        # fehl (z.B. Festplatte voll), graceful degradation wie ueberall
        # sonst in dieser Datei: ok=false statt Absturz, der Aufrufer faellt
        # dann auf das rohe Original zurueck.
        try:
            converted_path, vocals_clean_ext = _hieltech_wav_to_flac_file(final_vocals)
            handoff_path, _token = _hieltech_handoff_store(converted_path, vocals_clean_ext)
        except OSError as e:
            return {
                "ok": False,
                "error": "vocal_preprocess_handoff_failed",
                "message": f"Schritt 1 fertig, aber Weitergabe an Schritt 2 fehlgeschlagen: {e}",
            }

        return {
            "ok": True,
            "handoff_path": handoff_path,
            "audio_ext": vocals_clean_ext,
            # NEU (9. Okt 2026, Nutzerwunsch "wav mit 30mb konvertieren..."):
            # bisher wurden instrumental_full/residue IMMER mitgeschickt,
            # obwohl runpod-handler.js (preprocessVocalsOnRunPod) sie nie
            # ausliest - reiner Ballast, der die Antwort unnoetig Richtung
            # der 30-MB-Grenze draengt. Die Dateien existieren bei Erfolg
            # weiterhin lokal in work_dir (bis zum finally-Cleanup unten),
            # falls spaeter doch mal ein Verbraucher dafuer gebraucht wird.
            "instrumental_full_base64": None,
            "residue_base64": None,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/v1/hieltech_vocals_postprocess")
async def _hieltech_vocals_postprocess(
    audio: UploadFile,
    voice_model: str = Form(default=""),
    apply_reverb: str = Form(default="false"),
):
    """Schritt 3-7 der "Vocals erzeugen"-Kette, siehe Modul-Docstring oben.

    Erwartet multipart/form-data:
      audio         - die bereits per flow_edit_morph erzeugte Schritt-2-Datei
      voice_model   - Dateiname (ohne .pth) eines bereits auf dieses RunPod-
                      Volume hochgeladenen RVC-Stimmmodells (rvc_models/voices/).
                      Leer = kein Stimmwechsel (nur WORLD-Resynthese) - das ist
                      der Pfad fuer normale Kunden, Admin-only in der PWA
                      (server.js prueft is_owner, BEVOR dieses Feld ueberhaupt
                      hier ankommt - diese Route selbst vertraut server.js).
      apply_reverb  - "true"/"false" (als String, multipart kennt kein echtes
                      Bool) - gemessener Studio-Reverb (Schritt 7), siehe unten.

    Antwort (JSON): {"ok": true, "result_token": "...", "size_bytes": N,
                      "audio_ext": "flac"|"wav", "steps_applied": [...],
                      "warnings": [...]}. NEU (9. Okt 2026, Nutzerfrage
    "passt das auch bei 5 Minuten oder mehr?"): das fertige Ergebnis kommt
    NICHT mehr als ein einziges audio_base64-Feld zurueck (das bei langen
    Songs selbst als FLAC irgendwann wieder RunPod's 30-MB-Antwortgrenze
    sprengen wuerde, siehe _hieltech_wav_to_flac_file), sondern liegt lokal
    auf DIESEM Worker - runpod-handler.js muss die Bytes per "Worker
    Affinity" (X-Runpod-Worker-Id: strict-resume <id>, von der urspruenglich
    bedienenden Antwort) in sicheren Haeppchen ueber
    GET /v1/hieltech_fetch_chunk?token=...&offset=...&length=... abholen
    (siehe dortigen Docstring) - das funktioniert fuer JEDE Songlaenge,
    unabhaengig von der 30-MB-Grenze.
    """
    # NEU (9. Okt 2026): Arbeit im Hintergrund-Thread, damit /ping usw.
    # waehrenddessen antworten - siehe _hieltech_vocals_preprocess_sync.
    step2_bytes = await audio.read()
    return await asyncio.to_thread(
        _hieltech_vocals_postprocess_sync, step2_bytes, audio.filename, voice_model, apply_reverb
    )


def _hieltech_vocals_postprocess_sync(step2_bytes, filename, voice_model, apply_reverb):
    """Synchroner Teil von /v1/hieltech_vocals_postprocess (Schritt 3-7),
    laeuft in einem Thread - Begruendung siehe _hieltech_vocals_preprocess_sync."""
    steps_applied = []
    warnings = []
    reverb_wanted = str(apply_reverb).strip().lower() in {"1", "true", "yes", "y", "on"}
    voice_model = (voice_model or "").strip()
    # "Eigene Stimmmodelle" (7. Okt, Runde 144): voice_model kommt jetzt nicht
    # mehr nur vom vertrauenswuerdigen Admin-Feld, sondern (ueber server.js,
    # siehe dortigen Kommentar) auch aus vom Kunden selbst gewaehlten
    # "custom/<storage_filename>"-Werten. storage_filename ist zwar immer
    # server-generiert, trotzdem hier zusaetzlich (Verteidigung in der Tiefe,
    # Projekt-Konvention) ein billiger Schutz gegen Pfad-Ausbrueche, BEVOR der
    # Wert unten an rvc_convert_step.py weitergegeben wird.
    if ".." in voice_model or voice_model.startswith("/"):
        voice_model = ""
        warnings.append("Ungültiger Stimmmodell-Name verworfen (Sicherheitsprüfung)")

    work_dir = tempfile.mkdtemp(prefix="hieltech_vocals_")
    reverb_ref_future = None
    reverb_ref_pool = None
    try:
        src_ext = os.path.splitext(filename or "step2.wav")[1] or ".wav"
        step2_path = os.path.join(work_dir, f"step2{src_ext}")
        with open(step2_path, "wb") as f:
            f.write(step2_bytes)

        # --- Schritt 3: Demucs-Nachtrennung (reseparate_vocal_step.py,
        # htdemucs_ft, --shifts 5 --overlap 0.5, 1:1 wie auf dem Mac
        # bestaetigt) ---
        ok, log = _hieltech_run_step([
            os.path.join(_HIELTECH_PROJECT_ROOT, "reseparate_vocal_step.py"),
            step2_path, work_dir,
        ])
        vocals_reseparated = os.path.join(work_dir, "vocals_reseparated.wav")
        if not ok or not os.path.isfile(vocals_reseparated):
            return {
                "ok": False,
                "error": "demucs_reseparation_failed",
                "message": "Schritt 3 (Demucs-Nachtrennung) fehlgeschlagen - ohne dieses Ergebnis kann die Kette nicht weiterlaufen.",
                "log": log[-2000:],
            }
        steps_applied.append("demucs_reseparation")
        current = vocals_reseparated

        # NEU (9. Okt 2026, Geschwindigkeit): die Reverb-Referenzmessung
        # (stem_clean_step.py auf der Schritt-2-Datei) haengt NUR von
        # step2_path ab, nicht von Schritt 3-6. Bisher lief sie ganz am Ende
        # hinterher - jetzt startet sie direkt nach Schritt 3 im Hintergrund
        # (GPU: Demucs+DeepFilterNet), WAEHREND Schritt 4+5 (ffmpeg/WORLD) auf
        # der CPU rechnen. Gleiche Messung, gleiche Eingabe, gleiches Ergebnis
        # - nur ueberlappt statt hintereinander. Bewusst erst NACH Schritt 3
        # gestartet (nicht parallel dazu), damit die beiden Demucs-Laeufe sich
        # nicht gleichzeitig um den GPU-Speicher streiten. Eigener Unterordner,
        # damit keine Datei mit der Hauptkette kollidiert.
        reverb_ref_dir = os.path.join(work_dir, "reverb_ref")
        if reverb_wanted:
            os.makedirs(reverb_ref_dir, exist_ok=True)
            reverb_ref_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            reverb_ref_future = reverb_ref_pool.submit(
                _hieltech_run_step,
                [os.path.join(_HIELTECH_PROJECT_ROOT, "stem_clean_step.py"), step2_path, reverb_ref_dir],
            )

        # --- Schritt 4: ffmpeg-Waermekette (vocal_warmth_step.py) ---
        warm_path = os.path.join(work_dir, "vocals_warm.wav")
        ok, log = _hieltech_run_step([
            os.path.join(_HIELTECH_PROJECT_ROOT, "vocal_warmth_step.py"),
            current, warm_path,
        ])
        if ok and os.path.isfile(warm_path):
            steps_applied.append("ffmpeg_warmth")
            current = warm_path
        else:
            warnings.append("Schritt 4 (ffmpeg-Wärmekette) übersprungen: " + log[-300:])

        # --- Schritt 5: WORLD-Resynthese (world_resynth_step.py,
        # f0_method=harvest/d4c_threshold=0.7/mix=1.0, bestaetigte Werte) ---
        world_path = os.path.join(work_dir, "vocals_world.wav")
        ok, log = _hieltech_run_step([
            os.path.join(_HIELTECH_PROJECT_ROOT, "world_resynth_step.py"),
            "--input", current, "--output", world_path,
            "--f0_method", "harvest", "--d4c_threshold", "0.7", "--mix", "1.0",
        ])
        if ok and os.path.isfile(world_path):
            steps_applied.append("world_resynthesis")
            current = world_path
        else:
            warnings.append(
                "Schritt 5 (WORLD-Resynthese) übersprungen (vermutlich fehlt 'pyworld' auf "
                "diesem RunPod-Image - siehe Deployment-Hinweise): " + log[-300:]
            )

        # --- Schritt 6: RVC + Denoise + Gate (NUR falls ein Stimmmodell
        # gewaehlt wurde - Admin-only, siehe server.js) ---
        if voice_model:
            rvc_path = os.path.join(work_dir, "vocals_rvc.wav")
            ok, log = _hieltech_run_step([
                os.path.join(_HIELTECH_PROJECT_ROOT, "rvc_convert_step.py"),
                current, rvc_path, voice_model,
            ], python_path=_HIELTECH_RVC_VENV_PYTHON)
            if ok and os.path.isfile(rvc_path):
                steps_applied.append("rvc_convert")
                current = rvc_path

                denoised_path = os.path.join(work_dir, "vocals_rvc_denoised.wav")
                ok, log = _hieltech_run_step([
                    os.path.join(_HIELTECH_PROJECT_ROOT, "denoise_step.py"),
                    "--input", current, "--output", denoised_path,
                ])
                if ok and os.path.isfile(denoised_path):
                    steps_applied.append("rvc_denoise")
                    current = denoised_path
                else:
                    warnings.append(
                        "RVC-Denoise übersprungen (vermutlich fehlt 'DeepFilterNet' auf diesem "
                        "RunPod-Image): " + log[-300:]
                    )

                gated_path = os.path.join(work_dir, "vocals_rvc_gated.wav")
                ok, log = _hieltech_run_step([
                    os.path.join(_HIELTECH_PROJECT_ROOT, "gate_vocal_stem.py"),
                    "--input", current, "--output", gated_path,
                ])
                if ok and os.path.isfile(gated_path):
                    steps_applied.append("rvc_gate")
                    current = gated_path
                else:
                    warnings.append("RVC-Gate übersprungen: " + log[-300:])
            else:
                warnings.append(
                    f"RVC-Stimmwechsel ('{voice_model}') übersprungen (Modell nicht gefunden oder "
                    "'rvc-python'/fairseq fehlt auf diesem RunPod-Image - siehe Deployment-Hinweise): "
                    + log[-300:]
                )

        # --- Schritt 7: gemessener Studio-Reverb (NUR falls vom Kunden
        # aktiviert) - braucht eine zusaetzliche Referenz-Extraktion
        # (stem_clean_step.py, Demucs+DeepFilterNet) fuer die Nachhall-
        # Staerke-Messung, exakt wie bereits auf dem Mac umgesetzt. Schlaegt
        # diese Referenz-Extraktion fehl (z.B. DeepFilterNet fehlt), wird
        # Schritt 7 uebersprungen statt die ganze Anfrage abzubrechen -
        # dieselbe "graceful degradation" wie am Mac.
        if reverb_wanted:
            # Ergebnis der oben (nach Schritt 3) gestarteten Hintergrund-
            # Messung abholen - wartet nur noch auf den Rest, falls sie
            # laenger als Schritt 4-6 gedauert hat.
            if reverb_ref_future is not None:
                ref_ok, ref_log = reverb_ref_future.result()
            else:
                ref_ok, ref_log = _hieltech_run_step([
                    os.path.join(_HIELTECH_PROJECT_ROOT, "stem_clean_step.py"),
                    step2_path, reverb_ref_dir,
                ])
            vocals_clean_ref = os.path.join(reverb_ref_dir, "vocals_clean.wav")
            residue_ref = os.path.join(reverb_ref_dir, "residue.wav")
            if ref_ok and os.path.isfile(vocals_clean_ref) and os.path.isfile(residue_ref):
                reverb_path = os.path.join(work_dir, "vocals_reverb.wav")
                ok, log = _hieltech_run_step([
                    os.path.join(_HIELTECH_PROJECT_ROOT, "apply_measured_reverb.py"),
                    current, residue_ref, vocals_clean_ref, reverb_path,
                ])
                if ok and os.path.isfile(reverb_path):
                    steps_applied.append("studio_reverb")
                    current = reverb_path
                else:
                    warnings.append(
                        "Studio-Reverb übersprungen (vermutlich fehlt 'pedalboard' auf diesem "
                        "RunPod-Image): " + log[-300:]
                    )
            else:
                warnings.append(
                    "Studio-Reverb übersprungen: Referenz-Extraktion (Demucs+DeepFilterNet) "
                    "fehlgeschlagen: " + ref_log[-300:]
                )

        # NEU (9. Okt 2026, Nutzerfrage "passt FLAC auch bei 5 Minuten oder
        # mehr?") - siehe Routen-Docstring oben: Ergebnis landet im Handoff-
        # Ordner statt inline in der Antwort, Node holt es per
        # /v1/hieltech_fetch_chunk in sicheren Haeppchen ab. Schlaegt die
        # Ablage ausnahmsweise fehl (z.B. Festplatte voll), graceful
        # degradation wie ueberall sonst in dieser Datei: ok=false statt
        # Absturz.
        try:
            converted_path, final_ext = _hieltech_wav_to_flac_file(current)
            handoff_path, result_token = _hieltech_handoff_store(converted_path, final_ext)
            size_bytes = os.path.getsize(handoff_path)
        except OSError as e:
            return {
                "ok": False,
                "error": "vocal_postprocess_handoff_failed",
                "message": f"Nachbearbeitung fertig, aber Ergebnis-Uebergabe fehlgeschlagen: {e}",
            }

        return {
            "ok": True,
            "result_token": result_token,
            "size_bytes": size_bytes,
            "audio_ext": final_ext,
            "steps_applied": steps_applied,
            "warnings": warnings,
        }
    finally:
        # Laeuft die Hintergrund-Messung noch (z.B. weil Schritt 3 hier
        # vorzeitig mit Fehler zurueckkehrt - dann wurde sie gar nicht
        # gestartet - oder ein anderer Fehler auftrat), erst auf sie warten,
        # bevor work_dir geloescht wird.
        if reverb_ref_pool is not None:
            reverb_ref_pool.shutdown(wait=True)
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/v1/hieltech_fetch_chunk")
async def _hieltech_fetch_chunk(token: str, offset: int = 0, length: int = 18_000_000, done: str = "false"):
    """Liest einen Byte-Bereich aus einer per _hieltech_handoff_store
    abgelegten Ergebnisdatei (siehe _hieltech_vocals_postprocess oben) und
    liefert ihn als rohe Bytes zurueck.

    NEU (9. Okt 2026, Nutzerfrage "passt FLAC auch bei 5 Minuten oder mehr?"):
    das fertige Vocals-Ergebnis muss am Ende IMMER als echte Bytes RunPod
    verlassen (der Hoerer muss es ja abspielen koennen) - anders als beim
    Handoff zwischen Schritt 1 und Schritt 2 (siehe _HIELTECH_HANDOFF_DIR
    oben) gibt es hier also keinen Weg, die Netzwerk-Uebertragung ganz zu
    vermeiden. length ist deshalb standardmaessig 18 MB - deutlich unter
    RunPod's 30-MB-Antwortgrenze, auch bei beliebig langen Songs sicher,
    da runpod-handler.js so viele Haeppchen abholt wie noetig.

    WICHTIG: dieser Aufruf MUSS per "Worker Affinity"
    (X-Runpod-Worker-Id: strict-resume <id>, von RunPod's eigenem Load-
    Balancer dokumentiert) an GENAU den Worker gerichtet werden, der die
    Anfrage an /v1/hieltech_vocals_postprocess bearbeitet hat - die Datei
    liegt nur auf dessen lokaler Festplatte (System-Temp), NICHT auf dem
    (fuer Checkpoints genutzten) RunPod-Netzwerk-Volume. Ist der Worker
    inzwischen weg, kommt von RunPod ein eigener Fehler (404
    affinity_worker_gone) zurueck, BEVOR diese Route ueberhaupt erreicht
    wird - runpod-handler.js faellt in diesem (seltenen) Fall auf die
    rohe, nicht nachbearbeitete Audiospur zurueck.

    done=true (vom letzten abgeholten Haeppchen gesetzt) raeumt die Datei
    sofort auf - andernfalls erledigt das spaetestens nach 1 Stunde
    _hieltech_handoff_cleanup_stale() (Sicherheitsnetz, falls Node den
    letzten Aufruf z.B. wegen eines Netzwerkfehlers nie macht).
    """
    if not _HIELTECH_TOKEN_RE.match(token or ""):
        return Response(status_code=400, content=b"ungueltiges token-Format")

    matches = glob.glob(os.path.join(_HIELTECH_HANDOFF_DIR, f"{token}.*"))
    if not matches:
        return Response(status_code=404, content=b"Ergebnis nicht (mehr) gefunden - falscher Worker oder abgelaufen")
    result_path = matches[0]

    try:
        with open(result_path, "rb") as f:
            f.seek(max(0, offset))
            chunk = f.read(max(1, length))
    except OSError as e:
        return Response(status_code=500, content=str(e).encode("utf-8"))

    if str(done).strip().lower() in {"1", "true", "yes", "y"}:
        try:
            os.remove(result_path)
        except OSError:
            pass

    return Response(content=chunk, media_type="application/octet-stream")


# "Eigene Stimmmodelle" (7. Okt, Runde 144): 2 neue Routen, mit denen
# server.js .pth-Dateien in einen eigenen Unterordner
# rvc_models/voices/custom/ hochladen/wieder entfernen kann - getrennt vom
# flachen rvc_models/voices/ (dort liegt weiterhin NUR das Admin-eigene
# "Hicham.pth" direkt). rvc_convert_step.py (unveraendert) bekommt fuer
# eigene Modelle einfach "custom/<name>" als model_name uebergeben (siehe
# server.js) - os.path.join haengt das als normalen Unterordner an, kein
# Code dort musste dafuer angepasst werden.
_HIELTECH_CUSTOM_VOICES_DIR = os.path.join(_HIELTECH_PROJECT_ROOT, "rvc_models", "voices", "custom")
# Streng: nur das Muster, das server.js selbst erzeugt (u<license_id>_<16 Hex-
# Zeichen>.pth) - alles andere wird abgelehnt, BEVOR ein Dateiname den
# Dateisystem-Aufruf erreicht (Verteidigung in der Tiefe - server.js prueft
# zwar bereits Ownership/erzeugt den Namen selbst, aber diese Route soll
# auch fuer sich allein sicher sein).
_HIELTECH_STORAGE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}\.pth$")


@app.post("/v1/hieltech_vocals_upload_model")
async def _hieltech_vocals_upload_model(
    model: UploadFile,
    storage_filename: str = Form(...),
):
    """Speichert ein von einem Kunden hochgeladenes RVC-.pth-Modell unter
    rvc_models/voices/custom/<storage_filename> auf dem RunPod-Volume.
    storage_filename wird von server.js server-seitig erzeugt (nie aus einer
    Nutzereingabe abgeleitet) - hier zusaetzlich per Muster geprueft.
    """
    storage_filename = (storage_filename or "").strip()
    if not _HIELTECH_STORAGE_FILENAME_RE.match(storage_filename):
        return {"ok": False, "error": "invalid_filename", "message": "Ungültiger Dateiname"}
    os.makedirs(_HIELTECH_CUSTOM_VOICES_DIR, exist_ok=True)
    target_path = os.path.join(_HIELTECH_CUSTOM_VOICES_DIR, storage_filename)
    try:
        with open(target_path, "wb") as f:
            f.write(await model.read())
    except Exception as e:  # noqa: BLE001 - soll nie den ganzen Server crashen
        return {"ok": False, "error": "write_failed", "message": str(e)}
    return {"ok": True}


@app.post("/v1/hieltech_vocals_delete_model")
async def _hieltech_vocals_delete_model(
    storage_filename: str = Form(...),
):
    """Gegenstueck zum Upload - entfernt die Datei wieder, falls vorhanden.
    Kein Fehler, falls die Datei schon nicht (mehr) existiert (idempotent)."""
    storage_filename = (storage_filename or "").strip()
    if not _HIELTECH_STORAGE_FILENAME_RE.match(storage_filename):
        return {"ok": False, "error": "invalid_filename", "message": "Ungültiger Dateiname"}
    target_path = os.path.join(_HIELTECH_CUSTOM_VOICES_DIR, storage_filename)
    try:
        os.remove(target_path)
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": "delete_failed", "message": str(e)}
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
