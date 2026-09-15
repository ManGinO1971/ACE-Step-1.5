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
  5) startet uvicorn auf dem Port, den RunPod erwartet.

Nichts hier hat mit Preisen/Lizenzen/Geschaeftslogik zu tun - es startet
nur den bestehenden, quelloffenen REST-Server so, dass RunPod damit reden
kann.
"""
import os
import re
import sys
import threading

from fastapi import Response, Request
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


def _translate_patois(processor, model, text, sample=False, temperature=0.8, num_beams=1):
    import torch
    prompt = (
        f"<start_of_turn>user\nTranslate this English text into authentic Jamaican Patois "
        f"(Jamaican Creole), using real Patois vocabulary and grammar, not just English with "
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


def _translate_darija(processor, model, text, sample=False, temperature=0.8, num_beams=1):
    """Nutzerwunsch (15. Sept): Darija (gesprochenes Marokkanisch/Algerisch),
    wichtig fuer Raï- und Gnawa-Songtexte. Genau wie Jamaica Patois ist
    Darija KEIN offizieller Zielsprachcode, den TranslateGemma ueber das
    strukturierte Uebersetzungs-Template kennt (das liefert bestenfalls
    Hocharabisch/Fus'ha) - deshalb hier derselbe Ansatz wie bei
    _translate_patois oben: ein roher Prompt, der das Modell explizit auf
    authentisches Darija in arabischer Schrift festlegt (nicht Hocharabisch,
    nicht lateinische Umschrift/Arabizi)."""
    import torch
    prompt = (
        f"<start_of_turn>user\nTranslate this English text into authentic Moroccan/Algerian "
        f"Darija (the everyday spoken Maghrebi Arabic dialect), written in Arabic script. "
        f"Use real Darija vocabulary and grammar (including common French/Berber loanwords "
        f"where that is how Darija is actually spoken) -- NOT Modern Standard Arabic (Fus'ha) "
        f"and NOT a Latin-letter transcription. Output ONLY the translated lines, nothing "
        f"else -- no explanation, no commentary, no intro phrase:\n{text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
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


def _translate_looks_untranslated(text):
    """Gegenstueck zu _translate_looks_like_wrong_script (dort: Patois soll
    lateinisch bleiben, viel Nicht-Latein ist verdaechtig). Darija soll
    dagegen in arabischer Schrift zurueckkommen - bleibt die Ausgabe
    ueberwiegend lateinisch/ASCII, hat das Modell vermutlich nur den
    englischen Text wiederholt oder ignoriert die Sonderanweisung. Ein
    hoher Lateinanteil ist hier also das Fehler-Anzeichen, nicht Nicht-Latein."""
    stripped = re.sub(r'\s+', '', text)
    if not stripped:
        return False
    latin = sum(1 for ch in stripped if ch.isascii() and ch.isalpha())
    return latin / len(stripped) > 0.5


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
            # Darija (Nutzerwunsch 15. Sept) - gleiches Muster wie Jamaica
            # Patois oben, nur mit umgekehrter Skript-Erwartung (siehe
            # _translate_looks_untranslated): hier ist viel LATEIN das
            # Fehler-Anzeichen, nicht Nicht-Latein.
            translated = _translate_darija(processor, model, section_text, num_beams=4)
            if _translate_looks_untranslated(translated):
                retry = _translate_darija(processor, model, section_text, sample=True, temperature=0.8)
                translated = retry if not _translate_looks_untranslated(retry) else section_text
        else:
            translated = _translate_normal_with_fallback(processor, model, section_text, target_lang_code)
        translated_sections.append(translated)

    return {"translated_lyrics": "\n---\n".join(translated_sections)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
