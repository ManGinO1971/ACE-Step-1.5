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
  4) startet uvicorn auf dem Port, den RunPod erwartet.

Nichts hier hat mit Preisen/Lizenzen/Geschaeftslogik zu tun - es startet
nur den bestehenden, quelloffenen REST-Server so, dass RunPod damit reden
kann.
"""
import os
import re
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
