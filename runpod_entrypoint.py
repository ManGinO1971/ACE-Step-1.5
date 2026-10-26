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
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import faulthandler

# Bei einem harten Absturz (exit code 139) den Python-Stand aller Threads ins Log schreiben (27. Okt 2026)
faulthandler.enable(file=sys.stderr, all_threads=True)
import asyncio
import concurrent.futures


# =============================================================================
# HIELTECH (14. Okt 2026): Modell-Pruefung beim Start, BEVOR ACE-Step laedt.
# /app/checkpoints ist das RunPod-Netzwerk-Volume (siehe Dockerfile CMD). Lief
# das Volume beim Herunterladen voll ("Disk quota exceeded"), bleibt ein
# halber Modellordner liegen - ACE-Step prueft nur, ob der ORDNER existiert,
# und laedt ihn nie neu. Hier: jede .safetensors-Datei der zu pruefenden
# Modelle gegen ihren eigenen Header pruefen (erwartete Groesse = 8 + Header +
# Datenende); unvollstaendig -> Ordner loeschen, ACE-Step laedt dann neu.
# Zusaetzlich die Belegung je Modellordner ins Log, damit man sieht, wie gross
# das Volume sein muss.
# =============================================================================
def _hieltech_safetensors_complete(path):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return False
            header_len = int.from_bytes(raw, "little")
            if header_len <= 0 or header_len > 100 * 1024 * 1024:
                return False
            header = json.loads(f.read(header_len).decode("utf-8"))
        data_end = 0
        for key, meta in header.items():
            if key == "__metadata__" or not isinstance(meta, dict):
                continue
            offsets = meta.get("data_offsets") or [0, 0]
            data_end = max(data_end, int(offsets[1]))
        return size == 8 + header_len + data_end
    except Exception:  # noqa: BLE001
        return False


def _hieltech_dir_size_gb(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / (1024 ** 3)


def _hieltech_check_models_at_startup():
    ckpt = os.environ.get("HIELTECH_CHECKPOINTS_DIR", "/app/checkpoints")
    if not os.path.isdir(ckpt):
        return
    try:
        sizes = []
        for name in sorted(os.listdir(ckpt)):
            full = os.path.join(ckpt, name)
            if os.path.isdir(full):
                sizes.append(f"{name}={_hieltech_dir_size_gb(full):.1f} GB")
        usage = shutil.disk_usage(ckpt)
        print(f"[HIELTECH-Info] Volume-Belegung: {', '.join(sizes) or '(leer)'}; "
              f"laut System {usage.used / 1024 ** 3:.1f} GB belegt, {usage.free / 1024 ** 3:.1f} GB frei", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[HIELTECH-Info] Volume-Belegung nicht messbar: {e}", file=sys.stderr, flush=True)

    names = [n.strip() for n in os.environ.get("HIELTECH_VERIFY_MODELS", "acestep-v15-base").split(",") if n.strip()]
    for name in names:
        if "/" in name or ".." in name:
            continue
        model_dir = os.path.join(ckpt, name)
        if not os.path.isdir(model_dir):
            continue
        files = glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True)
        broken = [os.path.basename(f) for f in files if not _hieltech_safetensors_complete(f)]
        if files and not broken:
            print(f"[HIELTECH-Info] Modell {name}: vollständig ({len(files)} Gewichtsdatei(en))", file=sys.stderr, flush=True)
            continue
        reason = f"unvollständige Datei(en): {', '.join(broken)}" if broken else "keine Gewichtsdatei gefunden"
        # Mehrere Worker teilen sich das Volume: laeuft gerade bei einem
        # anderen Worker der Download, NICHT dazwischenfunken (Dateien in den
        # letzten 15 Minuten veraendert -> stehen lassen).
        newest = 0.0
        for root, _dirs, fnames in os.walk(model_dir):
            for fn in fnames:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, fn)))
                except OSError:
                    pass
        if newest and time.time() - newest < 900:
            print(f"[HIELTECH] Modell {name}: {reason} - wird gerade noch geschrieben (anderer Worker?), bleibt stehen", file=sys.stderr, flush=True)
            continue
        print(f"[HIELTECH] Modell {name}: {reason} - Ordner wird gelöscht und neu heruntergeladen", file=sys.stderr, flush=True)
        shutil.rmtree(model_dir, ignore_errors=True)


try:
    _hieltech_check_models_at_startup()
except Exception as _e:  # noqa: BLE001
    print(f"[HIELTECH] Modell-Prüfung übersprungen: {_e}", file=sys.stderr, flush=True)

from fastapi import Response, Request, UploadFile, Form, File
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


def _hieltech_free_gpu_cache():
    """Gibt ungenutzten PyTorch-GPU-Zwischenspeicher DIESES Prozesses (ACE-
    Step-Server) frei. Geladene Modelle bleiben unberuehrt; laufende
    Erzeugungen in anderen Threads sind davon nicht betroffen."""
    try:
        if "torch" not in sys.modules:
            return
        import gc

        import torch

        if torch.cuda.is_available():
            gc.collect()
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            freed = (before - torch.cuda.memory_reserved()) / 1024 ** 3
            if freed > 0.05:
                print(f"[HIELTECH-Info] GPU-Zwischenspeicher freigegeben: {freed:.1f} GB", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[HIELTECH] GPU-Zwischenspeicher nicht freigebbar: {e}", file=sys.stderr, flush=True)


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
    # NEU (16. Okt 2026, gemessen: "CUDA out of memory ... Process 137 has
    # 17.80 GiB memory in use" - der ACE-Step-Server haelt seit dem Base-
    # Modell fast die ganze GPU, ueberwiegend als ungenutzten PyTorch-
    # Zwischenspeicher): vor jedem Schritt diesen Zwischenspeicher freigeben
    # (die Modelle bleiben geladen) und den Schritt mit sparsamerer
    # Speicher-Einstellung starten. Scheitert er trotzdem an Speichermangel:
    # einmal erneut freigeben und mit kleineren GPU-Paketen wiederholen.
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _hieltech_free_gpu_cache()
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
            cmd, cwd=_HIELTECH_PROJECT_ROOT, capture_output=True, text=True, timeout=timeout, env=env
        )
        ok = result.returncode == 0
        if not ok and "out of memory" in ((result.stdout or "") + (result.stderr or "")).lower():
            print(f"[HIELTECH] {step_name}: GPU-Speicher voll - gebe Zwischenspeicher frei und wiederhole sparsamer", file=sys.stderr, flush=True)
            _hieltech_free_gpu_cache()
            env_retry = dict(env)
            env_retry["HIELTECH_UVR_BATCH"] = "1"
            result = subprocess.run(
                cmd, cwd=_HIELTECH_PROJECT_ROOT, capture_output=True, text=True, timeout=timeout, env=env_retry
            )
            ok = result.returncode == 0
        duration_s = time.perf_counter() - started_at
        print(f"[HIELTECH-Zeitmessung] {step_name}: {duration_s:.1f}s (ok={ok})", file=sys.stderr, flush=True)
        # NEU (10. Okt 2026): wichtige Hinweise der Teilskripte (welches
        # Trennmodell lief, Schnellweg/Rueckfall, Parallel-Kerne) zusaetzlich
        # ins RunPod-Worker-Log durchreichen - die Skript-Ausgabe selbst wird
        # sonst nur im Fehlerfall zurueckgegeben und war im Log unsichtbar.
        for line in ((result.stdout or "") + (result.stderr or "")).splitlines():
            if any(m in line for m in ("[Demucs-Schnellweg]", "Trenne Gesang", "Rückfall", "parallel auf", "GPU-Speicher", "Stimmwechsel lief", "Melodie-Lock", "Melodie-Folgen", "Text-Maske", "Seed-VC:", "Trennung ", "dereverb fertig", "denoise fertig", "übersprungen:", "HIELTECH Lage", "HIELTECH Oktav", "HIELTECH Stabil", "RTF:")):
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
        world_args = [
            os.path.join(_HIELTECH_PROJECT_ROOT, "world_resynth_step.py"),
            "--input", current, "--output", world_path,
            "--f0_method", "harvest", "--d4c_threshold", "0.7", "--mix", "1.0",
        ]
        # NEU (10. Okt 2026): optional per RunPod-Umgebungsvariable
        # HIELTECH_WORLD_PARALLEL_CALIBRATION=1 (schneller, ca. 1% der
        # Tonhoehen-Frames anders - nur nach Hoertest). Standard: AUS.
        if (os.environ.get("HIELTECH_WORLD_PARALLEL_CALIBRATION") or "").strip().lower() in {"1", "true", "yes", "on"}:
            world_args.append("--parallel_calibration")
        ok, log = _hieltech_run_step(world_args)
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


# ============================================================================
# "Background Vocals erzeugen" (NEU, 13. Okt 2026, Nutzerwunsch): eigene
# Kette fuer die Background-Stimmen, nach demselben Prinzip wie "Vocals
# erzeugen" - die bestehende Lead-Kette oben bleibt komplett unveraendert.
#   1. /v1/hieltech_backing_preprocess: Gesang trennen -> Lead/Background
#      trennen (Karaoke-Modell) -> Background reinigen -> Gate. Meldet
#      has_backing=false, wenn im Song gar kein Background-Gesang ist.
#   2. (optional, runpod-handler.js) flow_edit_morph per /release_task mit
#      dem Background-Text aus den (Klammern) im Songtext.
#   3. /v1/hieltech_backing_postprocess: gemeinsame Nachbearbeitung EINMAL
#      (Demucs-Nachtrennung nur nach Neu-Erzeugung, Waermekette, WORLD), dann
#      bis zu 3 Stimmen getrennt: je eigenes RVC-Modell + Tonlage
#      (maennlich/weiblich, nur ganze Oktaven - die Harmonien bleiben in der
#      Tonart) + Denoise + Gate. Ohne Modell: System-Stimme.
# ============================================================================


def _hieltech_is_handoff_path(path):
    try:
        real = os.path.realpath(path or "")
        return real.startswith(os.path.realpath(_HIELTECH_HANDOFF_DIR) + os.sep) and os.path.isfile(real)
    except Exception:  # noqa: BLE001
        return False


def _hieltech_last_json_line(output):
    for line in reversed((output or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"ok"' in line:
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


@app.post("/v1/hieltech_backing_preprocess")
async def _hieltech_backing_preprocess(audio: UploadFile):
    """Antwort: {"ok": true, "has_backing": bool, "handoff_path": "...",
    "direct_path": "...", "audio_ext": "...", "report": {...}}.
    handoff_path ist fuer /release_task (src_audio_path), direct_path fuer
    /v1/hieltech_backing_postprocess (src_handoff_path) - zwei Kopien, damit
    ein Rueckfall ohne Neu-Erzeugung immer noch eine Datei vorfindet. Beide
    nur per Worker-Affinitaet erreichbar."""
    src_bytes = await audio.read()
    return await asyncio.to_thread(_hieltech_backing_preprocess_sync, src_bytes, audio.filename)


def _hieltech_backing_preprocess_sync(src_bytes, filename):
    work_dir = tempfile.mkdtemp(prefix="hieltech_backing_pre_")
    try:
        src_ext = os.path.splitext(filename or "source.mp3")[1].lower() or ".mp3"
        if not re.match(r"^\.[a-z0-9]{1,5}$", src_ext):
            src_ext = ".mp3"
        src_path = os.path.join(work_dir, "source" + src_ext)
        with open(src_path, "wb") as f:
            f.write(src_bytes)
        out_dir = os.path.join(work_dir, "out")
        ok, log = _hieltech_run_step(
            [os.path.join(_HIELTECH_PROJECT_ROOT, "backing_separate_step.py"), src_path, out_dir], timeout=2400
        )
        report = _hieltech_last_json_line(log)
        if not ok or not report:
            return {"ok": False, "error": "backing_preprocess_failed", "message": "Lead/Background-Trennung fehlgeschlagen.", "log": log[-1500:]}
        print(f"[HIELTECH-Info] Background-Trennung: {json.dumps(report, ensure_ascii=False)}", file=sys.stderr, flush=True)
        if not report.get("has_backing"):
            return {"ok": True, "has_backing": False, "report": report}
        backing_clean = os.path.join(out_dir, "backing_clean.wav")
        if not os.path.isfile(backing_clean):
            return {"ok": False, "error": "backing_preprocess_failed", "message": "Background-Spur fehlt nach der Trennung.", "log": log[-1500:]}
        gated = os.path.join(out_dir, "backing_gated.wav")
        gate_ok, _gate_log = _hieltech_run_step([
            os.path.join(_HIELTECH_PROJECT_ROOT, "gate_vocal_stem.py"), "--input", backing_clean, "--output", gated,
        ], timeout=300)
        final = gated if gate_ok and os.path.isfile(gated) else backing_clean
        try:
            converted, ext = _hieltech_wav_to_flac_file(final)
            handoff_path, _t1 = _hieltech_handoff_store(converted, ext)
            direct_path, _t2 = _hieltech_handoff_store(converted, ext)
        except OSError as e:
            return {"ok": False, "error": "backing_preprocess_handoff_failed", "message": str(e)}
        # NEU (15. Okt 2026): Song ohne Lead-Gesang als Vorlage fuer Lego
        # (nur per Worker-Affinitaet erreichbar, wie handoff_path).
        context_path = None
        ctx_wav = os.path.join(out_dir, "context_no_lead.wav")
        if os.path.isfile(ctx_wav):
            try:
                ctx_conv, ctx_ext = _hieltech_wav_to_flac_file(ctx_wav)
                context_path, _t3 = _hieltech_handoff_store(ctx_conv, ctx_ext)
            except OSError as e:
                print(f"[HIELTECH] Song ohne Lead nicht übergebbar: {e}", file=sys.stderr, flush=True)
        return {"ok": True, "has_backing": True, "handoff_path": handoff_path, "direct_path": direct_path, "context_path": context_path, "audio_ext": ext, "report": report}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _hieltech_median_f0(path):
    """Mittlere Tonhoehe (Hz) der gesungenen Stellen, oder None."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(path, sr=16000, mono=True)
        if y.size < sr:
            return None
        f0 = librosa.yin(y, fmin=70, fmax=900, sr=sr, frame_length=1024)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        n = min(len(f0), len(rms))
        f0, rms = f0[:n], rms[:n]
        voiced = f0[rms > (np.max(rms) * 0.1)]
        if voiced.size < 20:
            return None
        return float(np.median(voiced))
    except Exception as e:  # noqa: BLE001
        print(f"[HIELTECH] Tonhöhen-Messung nicht möglich: {e}", file=sys.stderr, flush=True)
        return None


def _hieltech_octave_shift(median_f0, gender):
    """Nur ganze Oktaven, damit Melodie und Harmonien in der Tonart bleiben.
    NEU (13. Okt 2026, Nutzer-Hoertest "die female sollen +12, auf jeden
    Fall hoeher, aber genau die vocal range treffen"): weiblich IMMER eine
    Oktave hoeher, ausser der Background liegt schon klar in Frauenlage
    (mittlere Tonhoehe ab HIELTECH_BACKING_FEMALE_KEEP_HZ, Standard 250 Hz).
    Maennlich: NEU (14. Okt 2026, Hoertest "Man-Stimme ist zu tief, soll
    hoeher") standardmaessig KEINE Transposition mehr - die Maennerstimme
    singt in der Lage des Backgrounds. Optional per
    HIELTECH_BACKING_MALE_DOWN_HZ (z.B. 330) wieder eine Oktave tiefer, wenn
    der Background sehr hoch liegt."""
    female_keep = float(os.environ.get("HIELTECH_BACKING_FEMALE_KEEP_HZ", "250"))
    male_down = float(os.environ.get("HIELTECH_BACKING_MALE_DOWN_HZ", "0") or 0)
    if gender == "female":
        return 0 if (median_f0 and median_f0 >= female_keep) else 12
    if male_down > 0 and median_f0 and median_f0 > male_down:
        return -12
    return 0


def _hieltech_apply_backing_mask(voice_path, mask_path, out_path):
    """NEU (15. Okt 2026, Hoertest "Lead-Gesang ist mit drin, soll nur die
    (Klammer)-Stellen treffen"): die neue Stimme ist nur dort hoerbar, wo im
    Original Background gesungen wird (Huellkurve der gereinigten Original-
    Background-Spur, mit Haltezeit und weichen Uebergaengen)."""
    try:
        import librosa
        import numpy as np
        import soundfile as sf

        y, sr = sf.read(voice_path, always_2d=True)
        m, _ = librosa.load(mask_path, sr=sr, mono=True)
        hop = 512
        rms = librosa.feature.rms(y=m, frame_length=2048, hop_length=hop)[0]
        if rms.size == 0 or float(np.max(rms)) <= 0:
            return False
        thr_db = float(os.environ.get("HIELTECH_BACKING_MASK_DB", "-32"))
        hold_s = float(os.environ.get("HIELTECH_BACKING_MASK_HOLD_S", "0.25"))
        fade_s = float(os.environ.get("HIELTECH_BACKING_MASK_FADE_S", "0.06"))
        active = rms > float(np.max(rms)) * (10 ** (thr_db / 20.0))
        hold = max(1, int(hold_s * sr / hop))
        kernel = np.ones(2 * hold + 1)
        active = np.convolve(active.astype(float), kernel, mode="same") > 0
        fade = max(1, int(fade_s * sr / hop))
        env = np.convolve(active.astype(float), np.ones(fade) / fade, mode="same")
        env = np.clip(env, 0.0, 1.0)
        t_frames = np.arange(env.size) * hop
        gain = np.interp(np.arange(y.shape[0]), t_frames, env, right=0.0)
        sf.write(out_path, y * gain[:, None], sr)
        print(f"[HIELTECH-Info] Zeitmaske: Stimme an {100.0 * float(np.mean(active)):.0f} % der Songdauer hörbar", file=sys.stderr, flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[HIELTECH] Zeitmaske nicht möglich: {e}", file=sys.stderr, flush=True)
        return False


@app.post("/v1/hieltech_backing_postprocess")
async def _hieltech_backing_postprocess(
    audio: UploadFile | None = File(default=None),
    src_handoff_path: str = Form(default=""),
    regenerated: str = Form(default="false"),
    voices: str = Form(default="[]"),
    world_mix: str = Form(default="1.0"),
    reseparate: str = Form(default="1"),
    mask_path: str = Form(default=""),
    melody_lock: str = Form(default="0"),
):
    """voices: JSON-Liste bis 3 Eintraege {"slot": 1-3, "model": "custom/..."
    oder "" (System-Stimme), "gender": "male"|"female"}.
    Antwort: {"ok": true, "voices": [{"slot", "result_token", "size_bytes",
    "audio_ext", "rvc_applied", "pitch_shift"}], "warnings": [...]}"""
    src_bytes = await audio.read() if audio is not None else None
    filename = audio.filename if audio is not None else ""
    return await asyncio.to_thread(
        _hieltech_backing_postprocess_sync, src_bytes, filename, src_handoff_path, regenerated, voices, world_mix, reseparate, mask_path, melody_lock
    )


def _hieltech_backing_postprocess_sync(src_bytes, filename, src_handoff_path, regenerated, voices_json, world_mix="1.0", reseparate="1", mask_path="", melody_lock="0"):
    warnings = []
    try:
        voices = json.loads(voices_json or "[]")
    except ValueError:
        voices = []
    voices = [v for v in voices if isinstance(v, dict)][:3] or [{"slot": 1, "model": "", "gender": "female"}]
    work_dir = tempfile.mkdtemp(prefix="hieltech_backing_post_")
    try:
        if src_bytes:
            ext = os.path.splitext(filename or "")[1].lower() or ".flac"
            if not re.match(r"^\.[a-z0-9]{1,5}$", ext):
                ext = ".flac"
            src_path = os.path.join(work_dir, "backing_src" + ext)
            with open(src_path, "wb") as f:
                f.write(src_bytes)
        elif _hieltech_is_handoff_path(src_handoff_path):
            src_path = os.path.realpath(src_handoff_path)
        else:
            return {"ok": False, "error": "backing_source_missing", "message": "Keine Background-Quelle (weder Datei noch gültiger Handoff-Pfad)."}

        # NEU (14. Okt 2026, Hoertest "Roboter-Klang auf allen 3 Spuren"):
        # WORLD (nur EINE Tonhoehe gleichzeitig -> mehrstimmige Harmonien
        # klingen roboterhaft) und die Demucs-Nachtrennung sind jetzt pro
        # Anfrage abschaltbar (VPS-.env BACKING_WORLD_MIX / BACKING_RESEP).
        try:
            world_mix_f = max(0.0, min(1.0, float(world_mix)))
        except (TypeError, ValueError):
            world_mix_f = 1.0
        resep_on = str(reseparate).strip().lower() not in {"0", "false", "no", "off"}
        print(f"[HIELTECH-Info] Background-Nachbearbeitung: Nachtrennung {'an' if resep_on else 'aus'}, WORLD-Anteil {world_mix_f:.2f}, Melodie-Lock {melody_lock}", file=sys.stderr, flush=True)

        current = src_path
        if resep_on and str(regenerated).strip().lower() in {"1", "true", "yes"}:
            ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "reseparate_vocal_step.py"), current, work_dir])
            resep = os.path.join(work_dir, "vocals_reseparated.wav")
            if ok and os.path.isfile(resep):
                current = resep
            else:
                warnings.append("Background: Demucs-Nachtrennung übersprungen: " + log[-300:])

        warm = os.path.join(work_dir, "backing_warm.wav")
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "vocal_warmth_step.py"), current, warm])
        if ok and os.path.isfile(warm):
            current = warm
        else:
            warnings.append("Background: Wärmekette übersprungen: " + log[-300:])

        # NEU (16. Okt 2026, Nutzerwunsch "De-Harmonizer: die richtige Melodie
        # holen - auch an leisen Stellen -, einstimmig neu singen, dann
        # Vocoder"): RMVPE holt EINE Hauptmelodie aus der mehrstimmigen Spur,
        # WORLD baut damit eine einstimmige Stimme neu auf. Ersetzt dann den
        # normalen WORLD-Schritt (der seine Tonhoehe selbst suchte und bei
        # Harmonien zwischen den Stimmen sprang -> Roboter-Klang).
        locked = False
        if str(melody_lock).strip().lower() in {"1", "true", "yes", "on"}:
            f0_file = os.path.join(work_dir, "backing_melody_f0.npy")
            ok, log = _hieltech_run_step(
                [os.path.join(_HIELTECH_PROJECT_ROOT, "f0_rmvpe_step.py"), current, f0_file],
                python_path=_HIELTECH_RVC_VENV_PYTHON,
            )
            for line in (log or "").splitlines():
                if line.startswith("Melodie-Linie"):
                    print(f"[HIELTECH-Info] {line.strip()}", file=sys.stderr, flush=True)
            if ok and os.path.isfile(f0_file):
                mono_out = os.path.join(work_dir, "backing_melody_lock.wav")
                ok2, log2 = _hieltech_run_step([
                    os.path.join(_HIELTECH_PROJECT_ROOT, "world_resynth_step.py"),
                    "--input", current, "--output", mono_out, "--f0_file", f0_file,
                    "--d4c_threshold", os.environ.get("HIELTECH_MELODY_D4C", "0.85"), "--mix", "1.0",
                ])
                if ok2 and os.path.isfile(mono_out):
                    current = mono_out
                    locked = True
                    print("[HIELTECH-Info] Melodie-Lock: einstimmige Background-Stimme neu aufgebaut", file=sys.stderr, flush=True)
                else:
                    warnings.append("Background: Melodie-Lock (Neusynthese) übersprungen: " + (log2 or "")[-300:])
            else:
                warnings.append("Background: Melodie-Lock (RMVPE) übersprungen: " + (log or "")[-300:])

        if world_mix_f > 0 and not locked:
            world = os.path.join(work_dir, "backing_world.wav")
            world_args = [
                os.path.join(_HIELTECH_PROJECT_ROOT, "world_resynth_step.py"),
                "--input", current, "--output", world,
                "--f0_method", "harvest", "--d4c_threshold", "0.7", "--mix", f"{world_mix_f:.2f}",
            ]
            if (os.environ.get("HIELTECH_WORLD_PARALLEL_CALIBRATION") or "").strip().lower() in {"1", "true", "yes", "on"}:
                world_args.append("--parallel_calibration")
            ok, log = _hieltech_run_step(world_args)
            if ok and os.path.isfile(world):
                current = world
            else:
                warnings.append("Background: WORLD-Resynthese übersprungen: " + log[-300:])

        median_f0 = _hieltech_median_f0(current)
        print(f"[HIELTECH-Info] Background: mittlere Tonhöhe {median_f0 and round(median_f0)} Hz", file=sys.stderr, flush=True)

        results = []
        for i, v in enumerate(voices):
            slot = int(v.get("slot") or (i + 1))
            model = str(v.get("model") or "").strip()
            gender = "male" if str(v.get("gender") or "").lower() == "male" else "female"
            if ".." in model or model.startswith("/"):
                model = ""
                warnings.append(f"Stimme {slot}: ungültiger Modellname verworfen")
            voice_path = current
            rvc_applied = False
            shift = _hieltech_octave_shift(median_f0, gender)
            if model:
                rvc_out = os.path.join(work_dir, f"backing_{slot}_rvc.wav")
                # NEU (13. Okt 2026): genaueres Tonhoehen-Verfahren RMVPE fuer
                # Background (weniger falsche Toene); Rueckfall auf Standard.
                f0m = (os.environ.get("HIELTECH_BACKING_F0_METHOD") or "rmvpe").strip()
                ok, log = _hieltech_run_step([
                    os.path.join(_HIELTECH_PROJECT_ROOT, "rvc_convert_step.py"), current, rvc_out, model, str(shift), f0m,
                ], python_path=_HIELTECH_RVC_VENV_PYTHON)
                if not (ok and os.path.isfile(rvc_out)) and f0m != "harvest":
                    warnings.append(f"Stimme {slot}: {f0m} nicht nutzbar, Rückfall auf harvest")
                    ok, log = _hieltech_run_step([
                        os.path.join(_HIELTECH_PROJECT_ROOT, "rvc_convert_step.py"), current, rvc_out, model, str(shift),
                    ], python_path=_HIELTECH_RVC_VENV_PYTHON)
                if ok and os.path.isfile(rvc_out):
                    rvc_applied = True
                    voice_path = rvc_out
                    print(f"[HIELTECH-Info] Background Stimme {slot}: {gender}, Transposition {shift:+d}", file=sys.stderr, flush=True)
                    # NEU (13. Okt 2026, Nutzer-Hoertest "nicht alle 3 Spuren
                    # haben gleich viel gesungen"): das Noise-Gate (und die
                    # Rauschentfernung) PRO Stimme schnitten leise Stellen je
                    # Stimme verschieden weg. Die Spur ist vor dem Stimmwechsel
                    # bereits gereinigt+gegatet - deshalb standardmaessig aus.
                    if (os.environ.get("HIELTECH_BACKING_VOICE_CLEANUP") or "0").strip().lower() in {"1", "true", "yes", "on"}:
                        den = os.path.join(work_dir, f"backing_{slot}_den.wav")
                        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "denoise_step.py"), "--input", voice_path, "--output", den])
                        if ok and os.path.isfile(den):
                            voice_path = den
                        gated = os.path.join(work_dir, f"backing_{slot}_gate.wav")
                        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "gate_vocal_stem.py"), "--input", voice_path, "--output", gated])
                        if ok and os.path.isfile(gated):
                            voice_path = gated
                else:
                    warnings.append(f"Stimme {slot}: Stimmwechsel ('{model}') übersprungen, System-Stimme verwendet: " + log[-300:])
            if mask_path and _hieltech_is_handoff_path(mask_path):
                masked = os.path.join(work_dir, f"backing_{slot}_masked.wav")
                if _hieltech_apply_backing_mask(voice_path, os.path.realpath(mask_path), masked):
                    voice_path = masked
                else:
                    warnings.append(f"Stimme {slot}: Zeitmaske übersprungen")
            try:
                converted, ext = _hieltech_wav_to_flac_file(voice_path) if voice_path.endswith(".wav") else (voice_path, os.path.splitext(voice_path)[1].lstrip(".") or "flac")
                _hp, token = _hieltech_handoff_store(converted, ext)
                results.append({
                    "slot": slot,
                    "result_token": token,
                    "size_bytes": os.path.getsize(_hp),
                    "audio_ext": ext,
                    "rvc_applied": rvc_applied,
                    "pitch_shift": shift if rvc_applied else 0,
                })
            except OSError as e:
                warnings.append(f"Stimme {slot}: Ergebnis-Übergabe fehlgeschlagen: {e}")
        if not results:
            return {"ok": False, "error": "backing_postprocess_failed", "message": "Keine Background-Stimme konnte erzeugt werden.", "warnings": warnings}
        return {"ok": True, "voices": results, "warnings": warnings, "median_f0": median_f0}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/v1/hieltech_instrumental_balance")
async def _hieltech_instrumental_balance(
    audio: UploadFile,
    genre_hint: str = Form(default=""),
    drop_vocals: str = Form(default="0"),
):
    """NEU (13. Okt 2026, Nutzerwunsch "Instrumente erzeugen: professionelle
    Lautstaerke- und Mix-Anpassung"): zerlegt das fertige Instrumental per
    Demucs htdemucs_6s in Spuren, gleicht Begleitinstrumente genre-abhaengig
    an das Fundament aus Bass+Schlagzeug an und setzt alles verlustfrei
    wieder zusammen - siehe instrumental_balance_step.py.

    Antwort wie /v1/hieltech_vocals_postprocess: {"ok": true, "result_token",
    "size_bytes", "audio_ext", "report": {...}} - die Bytes holt
    runpod-handler.js per /v1/hieltech_fetch_chunk (Worker-Affinitaet) ab.
    """
    src_bytes = await audio.read()
    return await asyncio.to_thread(_hieltech_instrumental_balance_sync, src_bytes, audio.filename, genre_hint, drop_vocals)


def _hieltech_instrumental_balance_sync(src_bytes, filename, genre_hint, drop_vocals="0"):
    work_dir = tempfile.mkdtemp(prefix="hieltech_balance_")
    try:
        ext = os.path.splitext(filename or "")[1].lower() or ".mp3"
        if not re.match(r"^\.[a-z0-9]{1,5}$", ext):
            ext = ".mp3"
        src_path = os.path.join(work_dir, "instrumental" + ext)
        with open(src_path, "wb") as f:
            f.write(src_bytes)
        out_wav = os.path.join(work_dir, "balanced.wav")
        ok, output = _hieltech_run_step(
            [os.path.join(_HIELTECH_PROJECT_ROOT, "instrumental_balance_step.py"), src_path, out_wav, (genre_hint or "")[:500]]
            + (["drop_vocals"] if str(drop_vocals).strip().lower() in ("1", "true", "yes") else []),
            timeout=900,
        )
        report = None
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("{") and '"ok"' in line:
                try:
                    report = json.loads(line)
                except ValueError:
                    report = None
                break
        if not ok or not os.path.isfile(out_wav):
            return {"ok": False, "error": "instrumental_balance_failed", "message": output[-600:]}
        if report:
            print(f"[HIELTECH-Info] Instrumental-Balance: {json.dumps(report, ensure_ascii=False)[:600]}", file=sys.stderr, flush=True)
        try:
            converted_path, final_ext = _hieltech_wav_to_flac_file(out_wav)
            handoff_path, result_token = _hieltech_handoff_store(converted_path, final_ext)
            size_bytes = os.path.getsize(handoff_path)
        except OSError as e:
            return {"ok": False, "error": "instrumental_balance_handoff_failed", "message": str(e)}
        return {
            "ok": True,
            "result_token": result_token,
            "size_bytes": size_bytes,
            "audio_ext": final_ext,
            "report": report or {},
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================================
# NEU (14. Okt 2026): Hintergrund-Auftraege fuer lange Schritte.
# RunPod's Load-Balancer bricht JEDE einzelne Anfrage nach 5,5 Minuten ab
# (docs.runpod.io/serverless/load-balancing/overview, "Processing timeout")
# -> HTTP 524, die Arbeit war verloren und wurde per Wiederholung sogar
# doppelt gestartet. Jetzt: eine kurze Start-Anfrage legt den Auftrag an und
# antwortet sofort mit job_id; runpod-handler.js fragt per
# /v1/hieltech_job_status (Worker-Affinitaet!) nach, bis er fertig ist. Das
# Ergebnis ist exakt dasselbe JSON wie bei den bisherigen Routen, die
# weiterhin unveraendert bestehen bleiben (Rueckfall fuer aeltere Aufrufer).
# ============================================================================
_HIELTECH_JOBS = {}
_HIELTECH_JOBS_LOCK = threading.Lock()
_HIELTECH_JOB_KEEP_S = 2 * 3600


def _hieltech_jobs_cleanup():
    now = time.time()
    with _HIELTECH_JOBS_LOCK:
        for jid in [j for j, v in _HIELTECH_JOBS.items() if now - v.get("started", now) > _HIELTECH_JOB_KEEP_S]:
            _HIELTECH_JOBS.pop(jid, None)


def _hieltech_job_start(label, fn, *args):
    _hieltech_jobs_cleanup()
    job_id = uuid.uuid4().hex
    with _HIELTECH_JOBS_LOCK:
        _HIELTECH_JOBS[job_id] = {"status": "running", "label": label, "started": time.time()}

    def _run():
        try:
            result = fn(*args)
        except Exception as e:  # noqa: BLE001 - Fehler als Ergebnis melden, nie den Server stoeren
            result = {"ok": False, "error": "job_exception", "message": f"{type(e).__name__}: {e}"}
        with _HIELTECH_JOBS_LOCK:
            job = _HIELTECH_JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = result
                job["finished"] = time.time()
        print(f"[HIELTECH-Zeitmessung] Auftrag {label}: {time.time() - _HIELTECH_JOBS.get(job_id, {}).get('started', time.time()):.1f}s", file=sys.stderr, flush=True)

    threading.Thread(target=_run, name=f"hieltech-{label}", daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/v1/hieltech_job_status")
async def _hieltech_job_status(job_id: str = ""):
    if not _HIELTECH_TOKEN_RE.match(job_id or ""):
        return Response(status_code=400, content=b"ungueltige job_id")
    with _HIELTECH_JOBS_LOCK:
        job = dict(_HIELTECH_JOBS.get(job_id) or {})
    if not job:
        return Response(status_code=404, content=b"Auftrag unbekannt - falscher Worker oder abgelaufen")
    elapsed = round(time.time() - job.get("started", time.time()), 1)
    if job.get("status") != "done":
        return {"ok": True, "status": "running", "label": job.get("label"), "elapsed": elapsed}
    return {"ok": True, "status": "done", "label": job.get("label"), "elapsed": elapsed, "result": job.get("result")}


@app.post("/v1/hieltech_vocals_preprocess_start")
async def _hieltech_vocals_preprocess_start(audio: UploadFile):
    src_bytes = await audio.read()
    return _hieltech_job_start("vocals_preprocess", _hieltech_vocals_preprocess_sync, src_bytes, audio.filename)


@app.post("/v1/hieltech_vocals_postprocess_start")
async def _hieltech_vocals_postprocess_start(
    audio: UploadFile,
    voice_model: str = Form(default=""),
    apply_reverb: str = Form(default="false"),
):
    step2_bytes = await audio.read()
    return _hieltech_job_start("vocals_postprocess", _hieltech_vocals_postprocess_sync, step2_bytes, audio.filename, voice_model, apply_reverb)


@app.post("/v1/hieltech_backing_preprocess_start")
async def _hieltech_backing_preprocess_start(audio: UploadFile):
    src_bytes = await audio.read()
    return _hieltech_job_start("backing_preprocess", _hieltech_backing_preprocess_sync, src_bytes, audio.filename)


@app.post("/v1/hieltech_backing_postprocess_start")
async def _hieltech_backing_postprocess_start(
    audio: UploadFile | None = File(default=None),
    src_handoff_path: str = Form(default=""),
    regenerated: str = Form(default="false"),
    voices: str = Form(default="[]"),
    world_mix: str = Form(default="1.0"),
    reseparate: str = Form(default="1"),
    mask_path: str = Form(default=""),
    melody_lock: str = Form(default="0"),
):
    src_bytes = await audio.read() if audio is not None else None
    filename = audio.filename if audio is not None else ""
    return _hieltech_job_start("backing_postprocess", _hieltech_backing_postprocess_sync, src_bytes, filename, src_handoff_path, regenerated, voices, world_mix, reseparate, mask_path, melody_lock)


@app.post("/v1/hieltech_instrumental_balance_start")
async def _hieltech_instrumental_balance_start(audio: UploadFile, genre_hint: str = Form(default=""), drop_vocals: str = Form(default="0")):
    src_bytes = await audio.read()
    return _hieltech_job_start("instrumental_balance", _hieltech_instrumental_balance_sync, src_bytes, audio.filename, genre_hint, drop_vocals)


def _hieltech_lyrics_align_sync(src_bytes, filename, lines_json):
    """NEU (18. Okt 2026, "Text korrigieren"): prueft, welche Zeilen des
    eigenen Songtexts wo im Song gesungen werden (Abgleich, kein freies
    Raten) - siehe lyrics_align_step.py."""
    work_dir = tempfile.mkdtemp(prefix="hieltech_lyrics_align_")
    try:
        ext = os.path.splitext(filename or "")[1].lower() or ".mp3"
        if not re.match(r"^\.[a-z0-9]{1,5}$", ext):
            ext = ".mp3"
        src_path = os.path.join(work_dir, "song" + ext)
        with open(src_path, "wb") as f:
            f.write(src_bytes)
        try:
            lines = json.loads(lines_json or "[]")
            assert isinstance(lines, list) and all(isinstance(l, list) for l in lines)
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "invalid_lines", "message": "Zeilen fehlen oder sind ungültig."}
        lines_path = os.path.join(work_dir, "lines.json")
        with open(lines_path, "w", encoding="utf-8") as f:
            json.dump(lines, f, ensure_ascii=False)
        out_json = os.path.join(work_dir, "align.json")
        ok, log = _hieltech_run_step(
            [os.path.join(_HIELTECH_PROJECT_ROOT, "lyrics_align_step.py"), src_path, lines_path, out_json], timeout=900
        )
        for line in (log or "").splitlines():
            if line.startswith("Text-Abgleich:"):
                print(f"[HIELTECH-Info] {line.strip()[:300]}", file=sys.stderr, flush=True)
        if not ok or not os.path.isfile(out_json):
            return {"ok": False, "error": "lyrics_align_failed", "message": "Text-Abgleich fehlgeschlagen.", "log": log[-1500:]}
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["ok"] = True
        return data
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/v1/hieltech_lyrics_align_start")
async def _hieltech_lyrics_align_start(audio: UploadFile, lines: str = Form(default="[]")):
    src_bytes = await audio.read()
    return _hieltech_job_start("lyrics_align", _hieltech_lyrics_align_sync, src_bytes, audio.filename, lines)


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
#
# NEU (10. Okt 2026, Fehler "Stimmmodell laedt nicht"): bisher lag dieser
# Ordner unter _HIELTECH_PROJECT_ROOT (= /app im Docker-Container) - also auf
# der FLUECHTIGEN Platte des jeweiligen Workers, NICHT auf dem Netzwerk-
# Volume. Ein hochgeladenes Modell war damit nach jedem Worker-Neustart weg
# und fuer alle anderen Worker nie sichtbar. Jetzt liegt alles auf dem
# RunPod-Netzwerk-Volume (/runpod-volume, dasselbe Volume wie die ACE-Step-
# Checkpoints), sofern vorhanden - sonst (Mac/lokal) wie bisher im Projekt.
# rvc_convert_step.py bekommt denselben Ordner ueber die Umgebungsvariable
# HIELTECH_RVC_VOICES_DIR (wird an alle Teilskripte vererbt).
def _hieltech_rvc_voices_root():
    volume = "/runpod-volume"
    if os.path.isdir(volume) and os.access(volume, os.W_OK):
        return os.path.join(volume, "hieltech_rvc", "voices")
    return os.path.join(_HIELTECH_PROJECT_ROOT, "rvc_models", "voices")


_HIELTECH_RVC_VOICES_DIR = _hieltech_rvc_voices_root()
os.environ.setdefault("HIELTECH_RVC_VOICES_DIR", _HIELTECH_RVC_VOICES_DIR)
_HIELTECH_CUSTOM_VOICES_DIR = os.path.join(_HIELTECH_RVC_VOICES_DIR, "custom")
# NEU (10. Okt 2026): RunPod's Load-Balancer verwirft jede Anfrage ueber
# 30 MB - ein typisches .pth (z.B. 55 MB) kam deshalb NIE an. Groessere
# Modelle kommen jetzt in Stuecken (siehe _hieltech_vocals_upload_model_chunk).
_HIELTECH_MAX_MODEL_BYTES = 300 * 1024 * 1024
# Streng: nur das Muster, das server.js selbst erzeugt (u<license_id>_<16 Hex-
# Zeichen>.pth) - alles andere wird abgelehnt, BEVOR ein Dateiname den
# Dateisystem-Aufruf erreicht (Verteidigung in der Tiefe - server.js prueft
# zwar bereits Ownership/erzeugt den Namen selbst, aber diese Route soll
# auch fuer sich allein sicher sein).
# NEU (20. Okt 2026): auch die passende .index-Datei (gleicher Name) ist erlaubt.
_HIELTECH_STORAGE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}\.(pth|index)$")


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
        model_bytes = await model.read()

        def _write():
            with open(target_path, "wb") as f:
                f.write(model_bytes)

        await asyncio.to_thread(_write)
    except Exception as e:  # noqa: BLE001 - soll nie den ganzen Server crashen
        return {"ok": False, "error": "write_failed", "message": str(e)}
    return {"ok": True}


@app.post("/v1/hieltech_vocals_upload_model_chunk")
async def _hieltech_vocals_upload_model_chunk(
    chunk: UploadFile,
    storage_filename: str = Form(...),
    offset: int = Form(...),
    total_size: int = Form(...),
):
    """NEU (10. Okt 2026): Upload eines .pth in Stuecken (je < 30 MB, z.B.
    20 MB), weil RunPod's Load-Balancer groessere Anfragen still verwirft.

    Die Stuecke werden direkt auf das Netzwerk-Volume geschrieben (siehe
    _HIELTECH_CUSTOM_VOICES_DIR) - deshalb ist KEINE Worker-Affinitaet
    noetig: jedes Stueck darf auf einem anderen Worker landen, alle schreiben
    in dieselbe Datei. Erst wenn alle Bytes da sind (offset + Stueckgroesse
    == total_size), wird die Teildatei in einem Schritt zum fertigen Modell
    umbenannt - ein halb hochgeladenes Modell ist also nie auswaehlbar.

    Antwort: {"ok": true, "received": <bytes bisher>, "complete": bool}
    """
    storage_filename = (storage_filename or "").strip()
    if not _HIELTECH_STORAGE_FILENAME_RE.match(storage_filename):
        return {"ok": False, "error": "invalid_filename", "message": "Ungültiger Dateiname"}
    if total_size <= 0 or total_size > _HIELTECH_MAX_MODEL_BYTES or offset < 0 or offset >= total_size:
        return {"ok": False, "error": "invalid_range", "message": "Ungültige Größe/Position"}
    data = await chunk.read()
    if not data or offset + len(data) > total_size:
        return {"ok": False, "error": "invalid_chunk", "message": "Ungültiges Teilstück"}

    def _write():
        os.makedirs(_HIELTECH_CUSTOM_VOICES_DIR, exist_ok=True)
        target_path = os.path.join(_HIELTECH_CUSTOM_VOICES_DIR, storage_filename)
        part_path = target_path + ".part"
        flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if offset == 0 else 0)
        fd = os.open(part_path, flags, 0o644)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        received = os.path.getsize(part_path)
        complete = offset + len(data) == total_size
        if complete:
            if received != total_size:
                return {"ok": False, "error": "size_mismatch",
                        "message": f"Datei unvollständig ({received} von {total_size} Bytes)"}
            os.replace(part_path, target_path)
        return {"ok": True, "received": received, "complete": complete}

    try:
        return await asyncio.to_thread(_write)
    except Exception as e:  # noqa: BLE001 - soll nie den ganzen Server crashen
        return {"ok": False, "error": "write_failed", "message": str(e)}


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


# ===========================================================================
# NEU (20. Okt 2026): "Lead Vocals v2" – neue Kette für den Mikrofon-Knopf
# (per Hörtest gewählt, Root Nah Bow T7/v2). Rein ZUSÄTZLICH – alle bisherigen
# Routen bleiben unverändert; der VPS wählt per LEAD_VOCALS_MODE (v2/alt).
#   1. /v1/hieltech_lead_v2_prepare_start: Song -> RoFormer -> Karaoke ->
#      "Song ohne Background" (Quelle für Extract) + Lead-/Background-Referenz
#   2. (VPS) /release_task Extract mit dem Base-Modell, auf denselben Worker
#      gepinnt, Quelle = "Song ohne Background" (src_audio_path)
#   3. /v1/hieltech_lead_v2_post_start: Zeitmaske -> Gate -> RMVPE ->
#      Vocoder C -> optional .pth (RVC, f0 rmvpe). KEIN Denoise (Hörtest:
#      DeepFilterNet schnitt Gesang weg), KEINE Wärmekette, KEIN WORLD.
# ===========================================================================
_HIELTECH_VOICE_MODEL_RE = re.compile(r"^(custom/)?[A-Za-z0-9_-]{1,120}$")


def _hieltech_lead_v2_prepare_sync(src_bytes, filename):
    work_dir = tempfile.mkdtemp(prefix="hieltech_lead_v2_pre_")
    try:
        src_ext = os.path.splitext(filename or "source.mp3")[1].lower() or ".mp3"
        if not re.match(r"^\.[a-z0-9]{1,5}$", src_ext):
            src_ext = ".mp3"
        src_path = os.path.join(work_dir, "source" + src_ext)
        with open(src_path, "wb") as f:
            f.write(src_bytes)
        out_dir = os.path.join(work_dir, "out")
        ok, log = _hieltech_run_step(
            [os.path.join(_HIELTECH_PROJECT_ROOT, "lead_v2_prepare_step.py"), src_path, out_dir], timeout=2400
        )
        report = _hieltech_last_json_line(log)
        if not ok or not report:
            return {"ok": False, "error": "lead_v2_prepare_failed", "message": "Lead-v2-Vorbereitung (Trennung) fehlgeschlagen.", "log": log[-1500:]}
        paths = {}
        for key, name in (("song_path", "song_ohne_background.wav"), ("lead_ref_path", "lead_ref.wav"), ("backing_ref_path", "backing_ref.wav")):
            p = os.path.join(out_dir, name)
            if not os.path.isfile(p):
                return {"ok": False, "error": "lead_v2_prepare_failed", "message": f"{name} fehlt nach der Trennung."}
            conv, ext = _hieltech_wav_to_flac_file(p)
            paths[key], _tok = _hieltech_handoff_store(conv, ext)
        # NEU (21. Okt 2026, Background V2): Song ohne Lead / mit lautem Background (optional)
        for key, name in (("song_ohne_lead_path", "song_ohne_lead.wav"), ("song_backing_laut_path", "song_backing_laut.wav")):
            p = os.path.join(out_dir, name)
            if os.path.isfile(p):
                conv, ext = _hieltech_wav_to_flac_file(p)
                paths[key], _tok = _hieltech_handoff_store(conv, ext)
        print(f"[HIELTECH-Info] Lead v2 Vorbereitung: {json.dumps(report, ensure_ascii=False)}", file=sys.stderr, flush=True)
        return {"ok": True, **paths, "report": report}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/v1/hieltech_lead_v2_prepare_start")
async def _hieltech_lead_v2_prepare_start(audio: UploadFile):
    """Antwort (über /v1/hieltech_job_status): {"ok", "song_path", "lead_ref_path",
    "backing_ref_path", "report"} – alle Pfade nur per Worker-Affinität nutzbar."""
    src_bytes = await audio.read()
    return _hieltech_job_start("lead_v2_prepare", _hieltech_lead_v2_prepare_sync, src_bytes, audio.filename)


def _hieltech_lead_v2_source_path(path):
    """Extract-Ergebnis: nur Dateien aus ACE-Steps eigenem Ausgabeordner oder
    aus dem Handoff-Ordner werden angenommen (kein beliebiger Pfad)."""
    try:
        real = os.path.realpath(path or "")
        allowed = os.path.realpath(getattr(app.state, "temp_audio_dir", "") or "")
        if allowed and real.startswith(allowed + os.sep) and os.path.isfile(real):
            return real
    except Exception:  # noqa: BLE001
        pass
    if _hieltech_is_handoff_path(path):
        return os.path.realpath(path)
    return None


def _hieltech_lead_v2_post_sync(src_bytes, filename, extract_path, lead_ref_path, backing_ref_path, voice_model, extract_paths="", lead_text="", experiment=""):
    warnings, steps = [], []
    # Verbesserungen aus den Hörtests 20./21. Okt 2026 (Rock pon rock, Fassung F):
    #   melody_lock   - Oktav-Sprünge an die Oktave des Original-Leads (Höhen)
    #   lyrics_mask   - Stellen weit weg von jeder Lead-Textzeile stumm ("(Ooh)"-Intro)
    #   melody_follow - kurze schiefe Stellen beim Tonwechsel folgen dem Original (Tiefen)
    # SEIT 21. Okt 2026 STANDARD für jeden Song (vom Inhaber nach Hörtest F freigegeben).
    # Abschalten: Umgebungsvariable HIELTECH_LEAD_V2_FEATURES="none" (oder eigene Liste);
    # ein mitgeschicktes Formfeld "experiment" hat Vorrang ("none" = alles aus, wie 20. Okt).
    feat = (experiment or "").strip() or os.environ.get("HIELTECH_LEAD_V2_FEATURES", "melody_lock,lyrics_mask,melody_follow")
    exp = set() if feat.strip().lower() == "none" else {x.strip() for x in feat.split(",") if x.strip()}
    print(f"[HIELTECH-Info] Lead v2 Verbesserungen: {', '.join(sorted(exp)) or 'keine'}", file=sys.stderr, flush=True)
    pick_info = None
    work_dir = tempfile.mkdtemp(prefix="hieltech_lead_v2_post_")
    try:
        # NEU (20. Okt): "automatisch die beste von 2" - mehrere Extract-
        # Varianten, Auswahl per Text-Abgleich + Lückenzeit (lead_v2_pick_step.py)
        try:
            cand = [p for p in json.loads(extract_paths or "[]") if isinstance(p, str)][:4]
        except ValueError:
            cand = []
        cand = [c for c in (_hieltech_lead_v2_source_path(p) for p in cand) if c]
        if not src_bytes and len(cand) >= 2:
            text_file = os.path.join(work_dir, "lead_text.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(lead_text or "")
            ref = os.path.realpath(lead_ref_path) if _hieltech_is_handoff_path(lead_ref_path) else "-"
            ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "lead_v2_pick_step.py"), text_file, ref] + cand, timeout=1200)
            pick_info = _hieltech_last_json_line(log)
            if ok and pick_info and isinstance(pick_info.get("chosen"), int) and 0 <= pick_info["chosen"] < len(cand):
                extract_path = cand[pick_info["chosen"]]
                steps.append(f"pick_variant_{pick_info['chosen'] + 1}")
                print(f"[HIELTECH-Info] Lead v2 Auswahl: Variante {pick_info['chosen'] + 1} {json.dumps(pick_info.get('scores'), ensure_ascii=False)}", file=sys.stderr, flush=True)
            else:
                extract_path = cand[0]
                warnings.append("Variantenauswahl fehlgeschlagen, Variante 1 genommen: " + (log or "")[-300:])
        elif not src_bytes and len(cand) == 1:
            extract_path = cand[0]
        if src_bytes:
            ext = os.path.splitext(filename or "")[1].lower() or ".flac"
            if not re.match(r"^\.[a-z0-9]{1,5}$", ext):
                ext = ".flac"
            current = os.path.join(work_dir, "extract" + ext)
            with open(current, "wb") as f:
                f.write(src_bytes)
        else:
            current = _hieltech_lead_v2_source_path(extract_path)
            if not current:
                return {"ok": False, "error": "lead_v2_source_missing", "message": "Extract-Ergebnis nicht gefunden (falscher Worker?)."}
        wav = os.path.join(work_dir, "extract.wav")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", current, wav], capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.isfile(wav):
            return {"ok": False, "error": "lead_v2_decode_failed", "message": (r.stderr or "")[-300:]}
        current = wav

        # 1. Zeitmaske: Stille, wo im Original nur Background singt
        if _hieltech_is_handoff_path(lead_ref_path):
            masked = os.path.join(work_dir, "masked.wav")
            args = [os.path.join(_HIELTECH_PROJECT_ROOT, "lead_mask_step.py"), current, os.path.realpath(lead_ref_path), masked]
            if _hieltech_is_handoff_path(backing_ref_path):
                args.append(os.path.realpath(backing_ref_path))
            if "lyrics_mask" in exp and pick_info and isinstance(pick_info.get("scores"), list):
                segs = next((sc.get("segments") for sc in pick_info["scores"] if sc.get("variant") == pick_info.get("chosen")), None)
                if segs:
                    keep_file = os.path.join(work_dir, "lead_lines.json")
                    with open(keep_file, "w", encoding="utf-8") as f:
                        json.dump(segs, f)
                    args.append(keep_file)
                    steps.append("lyrics_mask")
            ok, log = _hieltech_run_step(args, timeout=600)
            if ok and os.path.isfile(masked):
                current = masked
                steps.append("time_mask")
            else:
                warnings.append("Zeitmaske übersprungen: " + log[-300:])
        else:
            warnings.append("Zeitmaske übersprungen: Lead-Referenz fehlt")

        # 2. Gate (live-Werte)
        gated = os.path.join(work_dir, "gated.wav")
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "gate_vocal_stem.py"), "--input", current, "--output", gated], timeout=300)
        if ok and os.path.isfile(gated):
            current = gated
            steps.append("gate")
        else:
            warnings.append("Gate übersprungen: " + log[-300:])

        # 3. RMVPE-Melodie + Vocoder C
        f0_file = os.path.join(work_dir, "f0.npy")
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "f0_rmvpe_lead_step.py"), current, f0_file],
                                     timeout=600, python_path=_HIELTECH_RVC_VENV_PYTHON)
        if ok and os.path.isfile(f0_file):
            voc = os.path.join(work_dir, "vocoder_c.wav")
            voc_args = [os.path.join(_HIELTECH_PROJECT_ROOT, "nsf_vocoder_step.py"), current, f0_file, voc]
            if "melody_lock" in exp and _hieltech_is_handoff_path(lead_ref_path):
                ref_f0 = os.path.join(work_dir, "f0_lead_ref.npy")
                ok_r, log_r = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "f0_rmvpe_lead_step.py"), os.path.realpath(lead_ref_path), ref_f0],
                                                 timeout=600, python_path=_HIELTECH_RVC_VENV_PYTHON)
                if ok_r and os.path.isfile(ref_f0):
                    voc_args.append(ref_f0)
                    steps.append("melody_lock")
                    if "melody_follow" in exp:
                        voc_args.append("--follow")
                        steps.append("melody_follow")
                else:
                    warnings.append("Melodie-Lock übersprungen: " + (log_r or "")[-300:])
            ok, log = _hieltech_run_step(voc_args, timeout=900)
            if ok and os.path.isfile(voc):
                current = voc
                steps.append("vocoder_c")
            else:
                warnings.append("Vocoder C übersprungen: " + log[-400:])
        else:
            warnings.append("RMVPE-Melodie fehlgeschlagen, Vocoder C übersprungen: " + log[-400:])

        # 4. optional .pth (RVC) – f0 rmvpe wie im Hörtest
        rvc_applied = False
        voice_model = (voice_model or "").strip()
        if voice_model:
            if not _HIELTECH_VOICE_MODEL_RE.match(voice_model):
                warnings.append("Stimmmodell-Name ungültig – Stimmwechsel übersprungen")
            else:
                rvc_out = os.path.join(work_dir, "rvc.wav")
                ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "rvc_convert_step.py"), current, rvc_out, voice_model, "0", "rmvpe"],
                                             python_path=_HIELTECH_RVC_VENV_PYTHON)
                if ok and os.path.isfile(rvc_out):
                    current = rvc_out
                    rvc_applied = True
                    steps.append("rvc_convert")
                else:
                    warnings.append(f"Stimmwechsel ('{voice_model}') übersprungen: " + log[-400:])

        conv, ext = _hieltech_wav_to_flac_file(current)
        hp, token = _hieltech_handoff_store(conv, ext)
        print(f"[HIELTECH-Info] Lead v2 Nachbearbeitung: {', '.join(steps)}" + (f" | Warnungen: {len(warnings)}" if warnings else ""), file=sys.stderr, flush=True)
        return {"ok": True, "result_token": token, "size_bytes": os.path.getsize(hp), "audio_ext": ext,
                "steps_applied": steps, "warnings": warnings, "rvc_applied": rvc_applied,
                "pick": pick_info}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===========================================================================
# NEU (21. Okt 2026): "Background V2" (.pth-Knopf) – dieselbe Kette wie Vocals V2,
# nur für den Background: Quelle = Song OHNE Lead, Melodie-Vorlage = Original-
# Background-Spur, je Stimme eigenes .pth (z.B. 1 Mann + 2 Frauen).
#   Vorbereitung: /v1/hieltech_lead_v2_prepare_start (liefert song_ohne_lead_path)
#   Neu singen:   (VPS) /release_task Extract, track backing_vocals, Base-Modell
#   Nachbearbeitung: /v1/hieltech_backing_v2_post_start
#     Auswahl beste Variante -> Zeitmaske (Background-Spur) -> Gate -> RMVPE ->
#     Vocoder C mit Melodie-Lock + Melodie folgen (Original-Background) ->
#     je Stimme: .pth (mit Index, Oktave je Geschlecht) bzw. ohne .pth Vocoder
#     mit Transposition. KEIN WORLD, KEINE Wärmekette, KEINE Demucs-Nachtrennung.
# ===========================================================================
def _hieltech_backing_v2_post_sync(extract_paths, backing_text, backing_ref_path, lead_ref_path, voices_json, experiment=""):
    warnings, steps = [], []
    feat = (experiment or "").strip() or os.environ.get("HIELTECH_BACKING_V2_FEATURES", "melody_lock,melody_follow")
    exp = set() if feat.strip().lower() == "none" else {x.strip() for x in feat.split(",") if x.strip()}
    print(f"[HIELTECH-Info] Background v2 Verbesserungen: {', '.join(sorted(exp)) or 'keine'}", file=sys.stderr, flush=True)
    try:
        voices = [v for v in json.loads(voices_json or "[]") if isinstance(v, dict)][:3]
    except ValueError:
        voices = []
    voices = voices or [{"slot": 1, "model": "", "gender": "female"}]
    if not _hieltech_is_handoff_path(backing_ref_path):
        return {"ok": False, "error": "backing_v2_ref_missing", "message": "Background-Referenz fehlt (falscher Worker?)."}
    bref = os.path.realpath(backing_ref_path)
    work_dir = tempfile.mkdtemp(prefix="hieltech_backing_v2_post_")
    pick_info = None
    try:
        try:
            cand = [p for p in json.loads(extract_paths or "[]") if isinstance(p, str)][:4]
        except ValueError:
            cand = []
        cand = [c for c in (_hieltech_lead_v2_source_path(p) for p in cand) if c]
        if not cand:
            return {"ok": False, "error": "backing_v2_source_missing", "message": "Neu gesungener Background nicht gefunden (falscher Worker?)."}
        current = cand[0]
        if len(cand) >= 2:
            text_file = os.path.join(work_dir, "backing_text.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(backing_text or "")
            ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "lead_v2_pick_step.py"), text_file, bref] + cand, timeout=1200)
            pick_info = _hieltech_last_json_line(log)
            if ok and pick_info and isinstance(pick_info.get("chosen"), int) and 0 <= pick_info["chosen"] < len(cand):
                current = cand[pick_info["chosen"]]
                steps.append(f"pick_variant_{pick_info['chosen'] + 1}")
            else:
                warnings.append("Variantenauswahl fehlgeschlagen, Variante 1 genommen: " + (log or "")[-300:])
        wav = os.path.join(work_dir, "extract.wav")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", current, wav], capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.isfile(wav):
            return {"ok": False, "error": "backing_v2_decode_failed", "message": (r.stderr or "")[-300:]}
        current = wav

        # 1. Zeitmaske: nur wo im Original der Background singt (Lead NICHT als
        #    Gegenspieler - unter {…} singt der Lead lauter, der Background soll bleiben)
        masked = os.path.join(work_dir, "masked.wav")
        args = [os.path.join(_HIELTECH_PROJECT_ROOT, "lead_mask_step.py"), current, bref, masked]
        if "lyrics_mask" in exp and pick_info and isinstance(pick_info.get("scores"), list):
            segs = next((sc.get("segments") for sc in pick_info["scores"] if sc.get("variant") == pick_info.get("chosen")), None)
            if segs:
                keep_file = os.path.join(work_dir, "backing_lines.json")
                with open(keep_file, "w", encoding="utf-8") as f:
                    json.dump(segs, f)
                args.append(keep_file)
                steps.append("lyrics_mask")
        ok, log = _hieltech_run_step(args, timeout=600)
        if ok and os.path.isfile(masked):
            current = masked
            steps.append("time_mask")
        else:
            warnings.append("Zeitmaske übersprungen: " + (log or "")[-300:])

        # 2. Gate
        gated = os.path.join(work_dir, "gated.wav")
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "gate_vocal_stem.py"), "--input", current, "--output", gated], timeout=300)
        if ok and os.path.isfile(gated):
            current = gated
            steps.append("gate")
        else:
            warnings.append("Gate übersprungen: " + (log or "")[-300:])

        # 3. Melodie (neu gesungen) + Melodie der Original-Background-Spur
        f0_file = os.path.join(work_dir, "f0.npy")
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "f0_rmvpe_lead_step.py"), current, f0_file],
                                     timeout=600, python_path=_HIELTECH_RVC_VENV_PYTHON)
        if not (ok and os.path.isfile(f0_file)):
            return {"ok": False, "error": "backing_v2_f0_failed", "message": "Melodie-Erkennung fehlgeschlagen: " + (log or "")[-300:], "warnings": warnings}
        ref_f0 = None
        if "melody_lock" in exp:
            ref_f0 = os.path.join(work_dir, "f0_backing_ref.npy")
            ok_r, log_r = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "f0_rmvpe_lead_step.py"), bref, ref_f0],
                                             timeout=600, python_path=_HIELTECH_RVC_VENV_PYTHON)
            if ok_r and os.path.isfile(ref_f0):
                steps.append("melody_lock")
                if "melody_follow" in exp:
                    steps.append("melody_follow")
                if "melody_follow_all" in exp:
                    steps.append("melody_follow_all")
            else:
                ref_f0 = None
                warnings.append("Melodie-Lock übersprungen: " + (log_r or "")[-300:])

        def vocode(out_path, shift=0):
            a = [os.path.join(_HIELTECH_PROJECT_ROOT, "nsf_vocoder_step.py"), current, f0_file, out_path]
            if ref_f0:
                a.append(ref_f0)
                if "melody_follow" in exp:
                    a.append("--follow")
                if "melody_follow_all" in exp:
                    a.append("--follow-all")
            if shift:
                a.append(f"--shift={int(shift)}")
            return _hieltech_run_step(a, timeout=900)

        base_voc = os.path.join(work_dir, "vocoder_c.wav")
        ok, log = vocode(base_voc)
        if not (ok and os.path.isfile(base_voc)):
            return {"ok": False, "error": "backing_v2_vocoder_failed", "message": "Vocoder C fehlgeschlagen: " + (log or "")[-300:], "warnings": warnings}
        steps.append("vocoder_c")
        median_f0 = _hieltech_median_f0(base_voc)
        print(f"[HIELTECH-Info] Background v2: mittlere Tonhöhe {median_f0 and round(median_f0)} Hz", file=sys.stderr, flush=True)

        # 4. je Stimme: .pth (Oktave je Geschlecht wie bisher) oder Vocoder mit Transposition
        results = []
        for i, v in enumerate(voices):
            slot = int(v.get("slot") or (i + 1))
            model = str(v.get("model") or "").strip()
            gender = "male" if str(v.get("gender") or "").lower() == "male" else "female"
            shift = _hieltech_octave_shift(median_f0, gender)
            voice_path, rvc_applied = base_voc, False
            if model and not _HIELTECH_VOICE_MODEL_RE.match(model):
                warnings.append(f"Stimme {slot}: ungültiger Modellname verworfen")
                model = ""
            if model:
                rvc_out = os.path.join(work_dir, f"backing_{slot}_rvc.wav")
                ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "rvc_convert_step.py"), base_voc, rvc_out, model, str(shift), "rmvpe"],
                                             python_path=_HIELTECH_RVC_VENV_PYTHON)
                if ok and os.path.isfile(rvc_out):
                    voice_path, rvc_applied = rvc_out, True
                else:
                    warnings.append(f"Stimme {slot}: Stimmwechsel ('{model}') übersprungen, System-Stimme verwendet: " + (log or "")[-300:])
            if not rvc_applied and shift:
                shifted = os.path.join(work_dir, f"backing_{slot}_shift.wav")
                ok, log = vocode(shifted, shift)
                if ok and os.path.isfile(shifted):
                    voice_path = shifted
                else:
                    warnings.append(f"Stimme {slot}: Transposition übersprungen: " + (log or "")[-300:])
            print(f"[HIELTECH-Info] Background v2 Stimme {slot}: {gender}, {'Modell ' + model if rvc_applied else 'System-Stimme'}, Transposition {shift:+d}", file=sys.stderr, flush=True)
            conv, ext = _hieltech_wav_to_flac_file(voice_path)
            hp, token = _hieltech_handoff_store(conv, ext)
            results.append({"slot": slot, "gender": gender, "result_token": token, "size_bytes": os.path.getsize(hp), "audio_ext": ext,
                            "rvc_applied": rvc_applied, "pitch_shift": shift})
        print(f"[HIELTECH-Info] Background v2 Nachbearbeitung: {', '.join(steps)}" + (f" | Warnungen: {len(warnings)}" if warnings else ""), file=sys.stderr, flush=True)
        return {"ok": True, "voices": results, "steps_applied": steps, "warnings": warnings, "pick": pick_info, "median_f0": median_f0}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _hieltech_bv2_lead_lines(lyrics):
    """Lead-Zeilen wie im VPS (leadOnlyLyrics): (…) raus, {…} bleibt; together = Zeile hatte {…}."""
    out = []
    for raw in (lyrics or "").splitlines():
        t = raw.strip()
        if not t or (t.startswith("[") and t.endswith("]")):
            continue
        lead = re.sub(r"\s+", " ", re.sub(r"\{([^{}]*)\}", r"\1", re.sub(r"\([^()]*\)", " ", t))).strip()
        words = lead.split()
        if words and re.search(r"[A-Za-zÀ-ÿ]", lead):
            out.append({"words": words, "together": "{" in t})
    return out


def _hieltech_backing_v2_clean_sync(lead_ref_path, backing_ref_path, song_path, lyrics):
    """NEU (21. Okt 2026): saubere Background-Vorlage (backing_v2_clean_step.py) +
    Quelle fürs Neu-Singen. Läuft zwischen Vorbereitung und Neu-Singen."""
    for pth in (lead_ref_path, backing_ref_path, song_path):
        if not _hieltech_is_handoff_path(pth):
            return {"ok": False, "error": "backing_v2_clean_input_missing", "message": "Vorbereitungs-Dateien fehlen (falscher Worker?)."}
    lead_ref, back_ref, song = (os.path.realpath(x) for x in (lead_ref_path, backing_ref_path, song_path))
    work_dir = tempfile.mkdtemp(prefix="hieltech_bv2_clean_")
    try:
        f0l, f0b = os.path.join(work_dir, "f0_lead.npy"), os.path.join(work_dir, "f0_backing.npy")
        for src, dst in ((lead_ref, f0l), (back_ref, f0b)):
            ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "f0_rmvpe_lead_step.py"), src, dst],
                                         timeout=600, python_path=_HIELTECH_RVC_VENV_PYTHON)
            if not (ok and os.path.isfile(dst)):
                return {"ok": False, "error": "backing_v2_clean_f0_failed", "message": "Melodie-Erkennung fehlgeschlagen: " + (log or "")[-300:]}
        # Text-Abgleich der Lead-Zeilen auf der Lead-Spur -> wo singt der Lead seinen Text
        lines = _hieltech_bv2_lead_lines(lyrics)
        lines_out = "-"
        found = 0
        if lines:
            lj = os.path.join(work_dir, "lead_words.json")
            with open(lj, "w", encoding="utf-8") as f:
                json.dump([x["words"] for x in lines], f, ensure_ascii=False)
            aj = os.path.join(work_dir, "lead_align.json")
            ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "lyrics_align_step.py"), "--align", lead_ref, lj, aj], timeout=900)
            if ok and os.path.isfile(aj):
                inst = json.load(open(aj, encoding="utf-8")).get("instances", [])
                segs = [{"t0": float(i["t0"]), "t1": float(i["t1"]), "conf": float(i.get("conf", 0)),
                         "together": bool(lines[int(i["k"])]["together"]) if 0 <= int(i.get("k", -1)) < len(lines) else False}
                        for i in inst if "t0" in i and "t1" in i]
                found = len(segs)
                lines_out = os.path.join(work_dir, "lead_lines.json")
                with open(lines_out, "w", encoding="utf-8") as f:
                    json.dump(segs, f)
        out_dir = os.path.join(work_dir, "out")
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "backing_v2_clean_step.py"), lead_ref, back_ref, song, f0l, f0b, lines_out, out_dir], timeout=900)
        report = _hieltech_last_json_line(log)
        if not ok or not report:
            return {"ok": False, "error": "backing_v2_clean_failed", "message": "Saubere Background-Vorlage fehlgeschlagen: " + (log or "")[-500:]}
        paths = {}
        for key, name in (("clean_backing_path", "clean_backing.wav"), ("source_path", "source.wav")):
            conv, ext = _hieltech_wav_to_flac_file(os.path.join(out_dir, name))
            paths[key], tok = _hieltech_handoff_store(conv, ext)
            if key == "clean_backing_path":  # zum Anhören (Terminal-Test)
                paths["clean_backing_token"], paths["clean_backing_size"], paths["clean_backing_ext"] = tok, os.path.getsize(paths[key]), ext
        report["lead_lines_found"] = found
        report["lead_lines_total"] = len(lines)
        print(f"[HIELTECH-Info] Background v2 Vorlage: {json.dumps(report, ensure_ascii=False)}", file=sys.stderr, flush=True)
        return {"ok": True, **paths, "report": report}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/v1/hieltech_backing_v2_clean_start")
async def _hieltech_backing_v2_clean_start(
    lead_ref_path: str = Form(default=""),
    backing_ref_path: str = Form(default=""),
    song_path: str = Form(default=""),
    lyrics: str = Form(default=""),
):
    """Antwort (über /v1/hieltech_job_status): {"ok", "clean_backing_path", "source_path", "report"}."""
    return _hieltech_job_start("backing_v2_clean", _hieltech_backing_v2_clean_sync, lead_ref_path, backing_ref_path, song_path, lyrics)


@app.post("/v1/hieltech_backing_v2_post_start")
async def _hieltech_backing_v2_post_start(
    extract_paths: str = Form(default=""),
    backing_text: str = Form(default=""),
    backing_ref_path: str = Form(default=""),
    lead_ref_path: str = Form(default=""),
    voices: str = Form(default=""),
    experiment: str = Form(default=""),
):
    """Antwort (über /v1/hieltech_job_status): {"ok", "voices": [{slot, gender,
    result_token, size_bytes, audio_ext, rvc_applied, pitch_shift}], "steps_applied",
    "warnings", "pick"} – je Stimme abholen über /v1/hieltech_fetch_chunk (selber Worker)."""
    return _hieltech_job_start("backing_v2_post", _hieltech_backing_v2_post_sync, extract_paths, backing_text, backing_ref_path, lead_ref_path, voices, experiment)


@app.post("/v1/hieltech_lead_v2_post_start")
async def _hieltech_lead_v2_post_start(
    audio: UploadFile | None = File(default=None),
    extract_path: str = Form(default=""),
    lead_ref_path: str = Form(default=""),
    backing_ref_path: str = Form(default=""),
    voice_model: str = Form(default=""),
    extract_paths: str = Form(default=""),
    lead_text: str = Form(default=""),
    experiment: str = Form(default=""),
):
    """Antwort (über /v1/hieltech_job_status): {"ok", "result_token", "size_bytes",
    "audio_ext", "steps_applied", "warnings", "rvc_applied", "pick"} – abholen über
    /v1/hieltech_fetch_chunk (selber Worker). extract_paths (JSON-Liste) +
    lead_text: mehrere Varianten, die beste wird automatisch gewählt."""
    src_bytes = await audio.read() if audio is not None else None
    filename = audio.filename if audio is not None else ""
    return _hieltech_job_start("lead_v2_post", _hieltech_lead_v2_post_sync, src_bytes, filename, extract_path, lead_ref_path, backing_ref_path, voice_model, extract_paths, lead_text, experiment)



# =====================================================================
# NEU (23. Okt 2026): BACKGROUND-CHOR (Knopf mit zwei Menschen)
# Vom Inhaber per Hörtest freigegeben (22. Okt 2026, Root Nah Bow):
#   1. backing_chor_prepare_step.py: Mischung 8 (Kim + BS-Viperx + FT2 Bleedless,
#      Min-Spec) -> Hall raus -> Rauschen raus = saubere Vorlage (Lead + Background)
#   2. hieltech_chor/seedvc_multi.py (eigene Seed-VC-Umgebung): jede Stimme singt
#      die GANZE Vorlage nach - Melodie/Vibrato/Timing vom Original, Klang aus der
#      Stimmprobe; Oktav-Sperre, Ton-Halt 0.3, Lage je Stimme (Frauen Terz, Mann original)
#   3. je Stimme die Lead-V2-Vocoder-Serie: Gate -> RMVPE -> (Ton-Halt/Vibrato) ->
#      Vocoder C mit Melodie-Lock + Melodie folgen gegen die Zielmelodie
#   4. optional .pth (RVC) je Stimme, optional Humanizer (kleine Zeitversätze je Phrase)
# Der Kunde schneidet danach mit der Schere weg, was er nicht braucht.
# =====================================================================
_HIELTECH_CHOR_DIR = os.path.join(_HIELTECH_PROJECT_ROOT, "hieltech_chor")
# Seed-VC liegt auf dem RunPod-Netzwerk-Volume (einmal eingerichtet, für alle Worker).
# Grund (23. Okt 2026): der RunPod-Build mit Seed-VC im Docker-Image brach jedes Mal
# ohne Meldung ab. Gibt es doch eine Image-Installation (/opt/seed-vc), hat sie Vorrang.
_HIELTECH_SEEDVC_HOME = os.environ.get("HIELTECH_SEEDVC_HOME", "/runpod-volume/hieltech_seedvc")
_HIELTECH_SEEDVC_LOCK = threading.Lock()


def _hieltech_seedvc_paths():
    if os.path.isfile("/opt/seedvc_venv/bin/python") and os.path.isfile("/opt/seed-vc/inference.py"):
        return "/opt/seedvc_venv/bin/python", "/opt/seed-vc", True
    py = os.path.join(_HIELTECH_SEEDVC_HOME, "venv", "bin", "python")
    return py, os.path.join(_HIELTECH_SEEDVC_HOME, "seed-vc"), os.path.isfile(os.path.join(_HIELTECH_SEEDVC_HOME, ".hieltech_ok")) and os.path.isfile(py)


def _hieltech_seedvc_ensure():
    """Richtet Seed-VC einmal auf dem Volume ein (dauert beim ersten Mal einige Minuten).
    Gibt (python, seedvc_ordner, ok, log) zurück."""
    with _HIELTECH_SEEDVC_LOCK:
        py, d, ok = _hieltech_seedvc_paths()
        if ok:
            return py, d, True, ""
        lock_path = os.path.join(os.path.dirname(_HIELTECH_SEEDVC_HOME.rstrip("/")) or "/tmp", ".hieltech_seedvc.lock")
        try:
            import fcntl
            lf = open(lock_path, "w")
            fcntl.flock(lf, fcntl.LOCK_EX)  # ein anderer Worker richtet evtl. gerade ein -> warten
        except Exception:  # noqa: BLE001
            lf = None
        try:
            py, d, ok = _hieltech_seedvc_paths()
            if ok:
                return py, d, True, ""
            print(f"[HIELTECH-Info] Seed-VC wird auf dem Volume eingerichtet ({_HIELTECH_SEEDVC_HOME}) – einmalig, dauert einige Minuten", file=sys.stderr, flush=True)
            t0 = time.time()
            r = subprocess.run(["bash", os.path.join(_HIELTECH_CHOR_DIR, "seedvc_setup.sh"), _HIELTECH_SEEDVC_HOME],
                               cwd=_HIELTECH_PROJECT_ROOT, capture_output=True, text=True, timeout=3600)
            log = (r.stdout or "") + (r.stderr or "")
            print(f"[HIELTECH-Zeitmessung] Seed-VC-Einrichtung: {time.time() - t0:.1f}s (ok={r.returncode == 0})", file=sys.stderr, flush=True)
            py, d, ok = _hieltech_seedvc_paths()
            return py, d, ok, log
        finally:
            if lf is not None:
                try:
                    lf.close()
                except Exception:  # noqa: BLE001
                    pass
# Rollen = die drei gewählten Stimmen (Stimmproben im Ordner hieltech_chor/)
_HIELTECH_CHOR_ROLES = {
    "mann": {"ref": "stimme_Mann.wav", "lage": "original", "oktave": 0, "halt": 0.3, "voc": (0.3, 1.0), "humanize_seed": 303},
    "frau2": {"ref": "stimme_Frau2.wav", "lage": "terz", "oktave": 0, "halt": 0.3, "voc": None, "humanize_seed": 101},
    "frau1": {"ref": "stimme_Frau1.wav", "lage": "terz", "oktave": 0, "halt": 0.3, "voc": (0.3, 1.4), "humanize_seed": 202},
}
_HIELTECH_KEY_NAMES = {"c": "c", "c#": "cis", "db": "des", "d": "d", "d#": "dis", "eb": "es", "e": "e", "f": "f", "f#": "fis",
                       "gb": "ges", "g": "g", "g#": "gis", "ab": "as", "a": "a", "a#": "ais", "bb": "b", "b": "h", "h": "h",
                       "cis": "cis", "des": "des", "dis": "dis", "es": "es", "fis": "fis", "ges": "ges", "gis": "gis", "as": "as", "ais": "ais"}


def _hieltech_key_to_tonart(key_scale):
    """'A minor' / 'Am' / 'F# Major' / 'a-moll' -> 'a-moll' / 'fis-dur'; unbekannt -> ''."""
    k = (key_scale or "").strip().lower().replace("♯", "#").replace("♭", "b")
    if not k or k in ("auto", "none"):
        return ""
    m = re.match(r"^([a-h](?:#|b|is|es|s)?)\s*[- ]?\s*(major|minor|maj|min|dur|moll|m)?$", k)
    if not m:
        return ""
    root = _HIELTECH_KEY_NAMES.get(m.group(1))
    if not root:
        return ""
    mode = m.group(2) or "major"
    return f"{root}-{'moll' if mode in ('minor', 'min', 'moll', 'm') else 'dur'}"


def _hieltech_chor_role_plan(voices):
    """Stimme männlich -> Mann; weiblich: erste -> Frau 2 (Favorit), zweite -> Frau 1, dritte -> Frau 2."""
    female_order = ["frau2", "frau1", "frau2"]
    fi = 0
    plan = []
    for i, v in enumerate(voices):
        gender = "male" if str(v.get("gender") or "").lower() == "male" else "female"
        if gender == "male":
            role = "mann"
        else:
            role = female_order[min(fi, 2)]
            fi += 1
        plan.append({"slot": int(v.get("slot") or (i + 1)), "gender": gender, "model": str(v.get("model") or "").strip(), "role": role})
    return plan


def _hieltech_backing_chor_sync(src_bytes, filename, key_scale, voices_json, humanize):
    warnings, steps = [], []
    try:
        voices = [v for v in json.loads(voices_json or "[]") if isinstance(v, dict)][:3]
    except ValueError:
        voices = []
    voices = voices or [{"slot": 1, "gender": "male"}, {"slot": 2, "gender": "female"}, {"slot": 3, "gender": "female"}]
    plan = _hieltech_chor_role_plan(voices)
    do_humanize = str(humanize or "").strip().lower() in ("1", "true", "on", "yes")
    work_dir = tempfile.mkdtemp(prefix="hieltech_chor_")
    try:
        src_ext = os.path.splitext(filename or "source.mp3")[1].lower() or ".mp3"
        if not re.match(r"^\.[a-z0-9]{1,5}$", src_ext):
            src_ext = ".mp3"
        src_path = os.path.join(work_dir, "source" + src_ext)
        with open(src_path, "wb") as f:
            f.write(src_bytes)

        # 1. saubere Vorlage (Mischung 8 -> Hall raus -> Rauschen raus)
        prep_dir = os.path.join(work_dir, "prep")
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "backing_chor_prepare_step.py"), src_path, prep_dir], timeout=3600)
        prep = _hieltech_last_json_line(log)
        vorlage = os.path.join(prep_dir, "chor_vorlage.wav")
        if not ok or not prep or not prep.get("ok") or not os.path.isfile(vorlage):
            return {"ok": False, "error": "chor_prepare_failed", "message": "Gesangs-Vorlage (Mischung 8) fehlgeschlagen.", "log": (log or "")[-1500:]}
        steps.append(f"mischung8({len(prep.get('ensemble', []))})")
        if prep.get("dereverb"):
            steps.append("dereverb")
        if prep.get("denoise"):
            steps.append("denoise")
        tonart = _hieltech_key_to_tonart(key_scale) or prep.get("key_detected") or "a-moll"
        print(f"[HIELTECH-Info] Background-Chor: Vorlage {prep.get('seconds')} s, Tonart {tonart} (App: '{key_scale}', geschätzt: {prep.get('key_detected')})", file=sys.stderr, flush=True)

        # 2. Seed-VC: jede benötigte Stimme einmal
        roles = []
        for p in plan:
            if p["role"] not in roles:
                roles.append(p["role"])
        svc_py, svc_dir_code, svc_ok, svc_log = _hieltech_seedvc_ensure()
        if not svc_ok:
            return {"ok": False, "error": "chor_seedvc_setup_failed", "message": "Seed-VC-Einrichtung fehlgeschlagen.", "log": (svc_log or "")[-1500:]}
        os.environ["HIELTECH_SEEDVC_DIR"] = svc_dir_code
        svc_dir = os.path.join(work_dir, "svc")
        job = {"source": vorlage, "out_dir": svc_dir, "tonart": tonart, "steps": int(os.environ.get("HIELTECH_CHOR_STEPS", "50")),
               "voices": [{"name": r, "ref": os.path.join(_HIELTECH_CHOR_DIR, _HIELTECH_CHOR_ROLES[r]["ref"]), "lage": _HIELTECH_CHOR_ROLES[r]["lage"],
                           "oktave": _HIELTECH_CHOR_ROLES[r]["oktave"], "halt": _HIELTECH_CHOR_ROLES[r]["halt"]} for r in roles]}
        job_file = os.path.join(work_dir, "seedvc_job.json")
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(job, f)
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_CHOR_DIR, "seedvc_multi.py"), job_file], timeout=5400, python_path=svc_py)
        svc_rep = _hieltech_last_json_line(log)
        if not ok or not svc_rep or not svc_rep.get("ok"):
            return {"ok": False, "error": "chor_seedvc_failed", "message": "Seed-VC (Neu-Singen) fehlgeschlagen.", "log": (log or "")[-1500:]}
        steps.append("seedvc")

        # 3./4. je Stimme: Vocoder-Serie, optional .pth und Humanizer
        results = []
        for p in plan:
            slot, role = p["slot"], p["role"]
            cfg = _HIELTECH_CHOR_ROLES[role]
            sung = os.path.join(svc_dir, f"{role}.wav")
            ziel = os.path.join(svc_dir, f"{role}_ziel.npy")
            vdir = os.path.join(work_dir, f"stimme{slot}")
            os.makedirs(vdir, exist_ok=True)
            current = sung
            gated = os.path.join(vdir, "gated.wav")
            ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "gate_vocal_stem.py"), "--input", current, "--output", gated], timeout=600)
            if ok and os.path.isfile(gated):
                current = gated
            else:
                warnings.append(f"Stimme {slot}: Gate übersprungen")
            f0_file = os.path.join(vdir, "f0.npy")
            ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "f0_rmvpe_lead_step.py"), current, f0_file],
                                         timeout=900, python_path=_HIELTECH_RVC_VENV_PYTHON)
            if ok and os.path.isfile(f0_file):
                f0_use = f0_file
                if cfg["voc"]:
                    f0_mod = os.path.join(vdir, "f0_mod.npy")
                    ok2, log2 = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "f0_halt_vib.py"), f0_file, f0_mod, str(cfg["voc"][0]), str(cfg["voc"][1])], timeout=300)
                    if ok2 and os.path.isfile(f0_mod):
                        f0_use = f0_mod
                    else:
                        warnings.append(f"Stimme {slot}: Ton-Halt im Vocoder übersprungen")
                voc = os.path.join(vdir, "vocoder.wav")
                a = [os.path.join(_HIELTECH_PROJECT_ROOT, "nsf_vocoder_step.py"), current, f0_use, voc]
                if os.path.isfile(ziel):
                    a += [ziel, "--follow"]
                ok3, log3 = _hieltech_run_step(a, timeout=1800)
                if ok3 and os.path.isfile(voc):
                    current = voc
                else:
                    warnings.append(f"Stimme {slot}: Vocoder übersprungen: " + (log3 or "")[-200:])
            else:
                warnings.append(f"Stimme {slot}: Melodie-Erkennung fehlgeschlagen, Vocoder übersprungen")
            model, rvc_applied = p["model"], False
            if model and not _HIELTECH_VOICE_MODEL_RE.match(model):
                warnings.append(f"Stimme {slot}: ungültiger Modellname verworfen")
                model = ""
            if model:
                rvc_out = os.path.join(vdir, "rvc.wav")
                ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "rvc_convert_step.py"), current, rvc_out, model, "0", "rmvpe"],
                                             python_path=_HIELTECH_RVC_VENV_PYTHON)
                if ok and os.path.isfile(rvc_out):
                    current, rvc_applied = rvc_out, True
                else:
                    warnings.append(f"Stimme {slot}: Stimmmodell ('{model}') übersprungen, Chor-Stimme verwendet: " + (log or "")[-200:])
            if do_humanize:
                hum = os.path.join(vdir, "humanized.wav")
                ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "humanizer.py"), current, hum, str(cfg["humanize_seed"] + slot), "25", "1.0", "80"], timeout=600)
                if ok and os.path.isfile(hum):
                    current = hum
                else:
                    warnings.append(f"Stimme {slot}: Humanizer übersprungen")
            print(f"[HIELTECH-Info] Background-Chor Stimme {slot}: {role}, {'Modell ' + model if rvc_applied else 'Chor-Stimme'}, Humanizer {'an' if do_humanize else 'aus'}", file=sys.stderr, flush=True)
            conv, ext = _hieltech_wav_to_flac_file(current)
            hp, token = _hieltech_handoff_store(conv, ext)
            results.append({"slot": slot, "gender": p["gender"], "role": role, "result_token": token, "size_bytes": os.path.getsize(hp),
                            "audio_ext": ext, "rvc_applied": rvc_applied, "pitch_shift": 0})
        if do_humanize:
            steps.append("humanizer")
        print(f"[HIELTECH-Info] Background-Chor fertig: {', '.join(steps)}" + (f" | Warnungen: {len(warnings)}" if warnings else ""), file=sys.stderr, flush=True)
        return {"ok": True, "voices": results, "steps_applied": steps, "warnings": warnings, "tonart": tonart, "prepare": prep}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _hieltech_vocals_clean_sync(src_bytes, filename):
    """NEU (23. Okt 2026, Nutzerwunsch): alter Mikrofon-Knopf = Original-Gesang SAUBER getrennt,
    genau wie die Vorlage des Background-Chors: Mischung 8 (3 Modelle, Min-Spec) -> Hall raus -> Rauschen raus.
    Kein Neu-Singen, kein Stimmmodell - der echte Gesang des Songs (Lead + Background)."""
    work_dir = tempfile.mkdtemp(prefix="hieltech_vclean_")
    try:
        src_ext = os.path.splitext(filename or "source.mp3")[1].lower() or ".mp3"
        if not re.match(r"^\.[a-z0-9]{1,5}$", src_ext):
            src_ext = ".mp3"
        src_path = os.path.join(work_dir, "source" + src_ext)
        with open(src_path, "wb") as f:
            f.write(src_bytes)
        prep_dir = os.path.join(work_dir, "prep")
        ok, log = _hieltech_run_step([os.path.join(_HIELTECH_PROJECT_ROOT, "backing_chor_prepare_step.py"), src_path, prep_dir], timeout=3600)
        prep = _hieltech_last_json_line(log)
        vocals = os.path.join(prep_dir, "chor_vorlage.wav")
        if not ok or not prep or not prep.get("ok") or not os.path.isfile(vocals):
            return {"ok": False, "error": "vocals_clean_failed", "message": "Saubere Gesangstrennung (Mischung 8) fehlgeschlagen.", "log": (log or "")[-1500:]}
        steps = [f"mischung8({len(prep.get('ensemble', []))})"]
        if prep.get("dereverb"):
            steps.append("dereverb")
        if prep.get("denoise"):
            steps.append("denoise")
        conv, ext = _hieltech_wav_to_flac_file(vocals)
        hp, token = _hieltech_handoff_store(conv, ext)
        print(f"[HIELTECH-Info] Vocals sauber fertig: {', '.join(steps)}, {prep.get('seconds')} s", file=sys.stderr, flush=True)
        return {"ok": True, "result_token": token, "size_bytes": os.path.getsize(hp), "audio_ext": ext, "steps_applied": steps, "prepare": prep}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/v1/hieltech_vocals_clean_start")
async def _hieltech_vocals_clean_start(audio: UploadFile):
    """Antwort (über /v1/hieltech_job_status): {"ok", "result_token", "size_bytes", "audio_ext",
    "steps_applied"} – abholen über /v1/hieltech_fetch_chunk (selber Worker)."""
    src_bytes = await audio.read()
    return _hieltech_job_start("vocals_clean", _hieltech_vocals_clean_sync, src_bytes, audio.filename)


# ===========================================================================
# NEU (24. Okt 2026): eigenes LoKr/LoRA-Training (HIELTECH-Instrumente, Gnawa, Reggae …)
# Daten liegen auf dem Netzwerk-Volume: <HOME>/data/<datensatz>/ (Audio + <name>.json mit caption)
# Ablauf im Auftrag: /v1/dataset/scan -> /v1/dataset/preprocess_async -> /v1/training/start_lokr
# (die offiziellen ACE-Step-Routen, hier über localhost aufgerufen). Ergebnis: <HOME>/out/<name>/final/
# Test: Adapter laden -> Songs erzeugen -> Adapter entladen -> dieselben Songs ohne Adapter (Vergleich).
# ===========================================================================
_HIELTECH_LORA_HOME = os.environ.get("HIELTECH_LORA_HOME", "/runpod-volume/hieltech_lora")
_HIELTECH_LORA_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_HIELTECH_LORA_FILE_RE = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_.\-]{0,127}$")
_HIELTECH_LORA_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".json", ".txt")
_HIELTECH_LORA_PROGRESS = {}


def _hieltech_lora_local(method, path, payload=None, timeout=600):
    """Ruft eine Route dieses Servers über localhost auf (JSON). Gibt (http_status, json) zurück."""
    import urllib.request
    import urllib.error
    url = f"http://127.0.0.1:{os.environ.get('PORT', '5000')}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    key = os.environ.get("ACESTEP_API_KEY", "").strip()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            body = {}
        return e.code, body


def _hieltech_lora_data(body):
    """ACE-Step-Antworten: {"data": …, "code": …, "error": …}."""
    if isinstance(body, dict) and "data" in body:
        return body.get("data") or {}, body.get("error")
    return body or {}, None


def _hieltech_lora_set_progress(name, **kw):
    cur = dict(_HIELTECH_LORA_PROGRESS.get(name) or {})
    cur.update(kw)
    cur["updated"] = time.time()
    _HIELTECH_LORA_PROGRESS[name] = cur


@app.post("/v1/hieltech_lora_upload")
async def _hieltech_lora_upload(file: UploadFile, dataset: str = Form(...), filename: str = Form(...)):
    """Eine Trainingsdatei aufs Volume legen: <HOME>/data/<dataset>/<filename>."""
    if not _HIELTECH_LORA_NAME_RE.match(dataset or "") or not _HIELTECH_LORA_FILE_RE.match(filename or ""):
        return Response(status_code=400, content=b"ungueltiger Name")
    if not filename.lower().endswith(_HIELTECH_LORA_EXT):
        return Response(status_code=400, content=b"Dateityp nicht erlaubt")
    d = os.path.join(_HIELTECH_LORA_HOME, "data", dataset)
    os.makedirs(d, exist_ok=True)
    body = await file.read()
    tmp = os.path.join(d, "." + filename + ".part")
    with open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, os.path.join(d, filename))
    return {"ok": True, "dataset": dataset, "filename": filename, "bytes": len(body)}


@app.get("/v1/hieltech_lora_list")
async def _hieltech_lora_list(dataset: str = ""):
    """Dateien eines Datensatzes bzw. vorhandene Adapter auflisten."""
    out = {"ok": True, "datasets": {}, "adapters": []}
    base = os.path.join(_HIELTECH_LORA_HOME, "data")
    if os.path.isdir(base):
        for ds in sorted(os.listdir(base)):
            if dataset and ds != dataset:
                continue
            p = os.path.join(base, ds)
            if os.path.isdir(p):
                fs = [f for f in sorted(os.listdir(p)) if not f.startswith(".")]
                out["datasets"][ds] = {"files": len(fs), "audio": len([f for f in fs if not f.endswith((".json", ".txt"))]),
                                       "bytes": sum(os.path.getsize(os.path.join(p, f)) for f in fs)}
    ob = os.path.join(_HIELTECH_LORA_HOME, "out")
    if os.path.isdir(ob):
        for n in sorted(os.listdir(ob)):
            w = os.path.join(ob, n, "final", "lokr_weights.safetensors")
            if os.path.isfile(w) or os.path.isdir(os.path.join(ob, n, "final")):
                out["adapters"].append(n)
    return out


def _hieltech_lora_running_job():
    """Job-ID eines laufenden LoRA-Trainings/-Tests auf diesem Worker (nur einer gleichzeitig erlaubt)."""
    with _HIELTECH_JOBS_LOCK:
        for jid, job in _HIELTECH_JOBS.items():
            if job.get("label") in ("lora_train", "lora_test") and job.get("status") == "running":
                return jid
    return None


def _hieltech_lora_train_sync(dataset, name, epochs, learning_rate, save_every):
    t0 = time.time()
    try:
        from acestep.training.path_safety import set_safe_root
        os.makedirs(_HIELTECH_LORA_HOME, exist_ok=True)
        set_safe_root(_HIELTECH_LORA_HOME)  # Volume liegt außerhalb von /app (sonst lehnt ACE-Step die Pfade ab); "/" ginge nicht (Prüfung auf root + "/")
    except Exception as e:  # noqa: BLE001
        print(f"[HIELTECH-Info] LoRA: set_safe_root fehlgeschlagen: {e}", file=sys.stderr, flush=True)
    audio_dir = os.path.join(_HIELTECH_LORA_HOME, "data", dataset)
    tensor_dir = os.path.join(_HIELTECH_LORA_HOME, "tensors", name)
    out_dir = os.path.join(_HIELTECH_LORA_HOME, "out", name)
    if not os.path.isdir(audio_dir):
        return {"ok": False, "error": "no_dataset", "message": f"Datensatz fehlt: {audio_dir}"}
    st, body = _hieltech_lora_local("GET", "/v1/training/status")
    data, _ = _hieltech_lora_data(body)
    if (data or {}).get("is_training"):
        return {"ok": False, "error": "already_training", "message": "Auf diesem Worker läuft schon ein Training – erst mit 'stop' beenden."}
    # Sprachmodell (LM) für die ganze Trainingsdauer entladen: ACE-Step entlädt es sonst bei der
    # Vorverarbeitung und lädt es danach wieder – genau dabei stürzte der Worker am 26. Okt ab
    # (exit code 139, L4 mit 31 GB RAM). Ist es schon entladen, lädt ACE-Step es auch nicht neu.
    try:
        llm = getattr(app.state, "llm_handler", None)
        if llm is not None and getattr(llm, "llm_initialized", False):
            llm.unload()
            app.state._llm_initialized = False
            print("[HIELTECH-Info] LoRA: Sprachmodell für das Training entladen", file=sys.stderr, flush=True)
        if llm is not None:
            # verhindert das automatische Neuladen bei späteren Anfragen (Test-Songs) – bis zum nächsten Worker-Start
            app.state._llm_init_error = "Sprachmodell wegen LoRA-Training entladen (Worker neu starten für normalen Betrieb)"
    except Exception as e:  # noqa: BLE001
        print(f"[HIELTECH-Info] LoRA: Sprachmodell entladen fehlgeschlagen: {e}", file=sys.stderr, flush=True)
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    # Vorverarbeitete Dateien NICHT löschen: nach einem Absturz macht ein neuer Start dort weiter,
    # wo er war (feste Datei-IDs je Audiodatei, skip_existing). Nur die zuletzt geschriebene Datei
    # wird verworfen, weil sie beim Absturz halb geschrieben sein kann.
    os.makedirs(tensor_dir, exist_ok=True)
    try:
        pts = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")), key=os.path.getmtime)
        if pts:
            os.remove(pts[-1])
            print(f"[HIELTECH-Info] LoRA {name}: {len(pts) - 1} vorverarbeitete Dateien vorhanden, mache dort weiter", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[HIELTECH-Info] LoRA {name}: Aufräumen fehlgeschlagen: {e}", file=sys.stderr, flush=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    _hieltech_lora_set_progress(name, phase="scan", started=t0)

    st, body = _hieltech_lora_local("POST", "/v1/dataset/scan", {"audio_dir": audio_dir, "dataset_name": name, "all_instrumental": True})
    data, err = _hieltech_lora_data(body)
    n = int((data or {}).get("num_samples") or 0)
    labeled = sum(1 for x in (data or {}).get("samples", []) if x.get("labeled") or x.get("caption"))
    if st >= 400 or err or n == 0:
        return {"ok": False, "error": "scan_failed", "message": f"Scan: HTTP {st} {err or ''}"}
    print(f"[HIELTECH-Info] LoRA {name}: {n} Dateien, {labeled} mit Beschreibung", file=sys.stderr, flush=True)
    # feste IDs (statt Zufall) -> vorhandene .pt-Dateien werden beim Neustart wiedererkannt
    try:
        import hashlib
        ids = set()
        for smp in getattr(getattr(app.state, "dataset_builder", None), "samples", []) or []:
            smp.id = hashlib.md5(os.path.basename(smp.audio_path).encode("utf-8")).hexdigest()[:8]
            ids.add(smp.id)
        for f in glob.glob(os.path.join(tensor_dir, "*.pt")):
            if os.path.splitext(os.path.basename(f))[0] not in ids:
                os.remove(f)  # Reste mit alten Zufalls-IDs
    except Exception as e:  # noqa: BLE001
        print(f"[HIELTECH-Info] LoRA {name}: feste IDs fehlgeschlagen: {e}", file=sys.stderr, flush=True)
    _hieltech_lora_set_progress(name, phase="preprocess", samples=n, labeled=labeled)

    st, body = _hieltech_lora_local("POST", "/v1/dataset/preprocess_async", {"output_dir": tensor_dir, "skip_existing": True})
    data, err = _hieltech_lora_data(body)
    task = (data or {}).get("task_id")
    if st >= 400 or err or not task:
        return {"ok": False, "error": "preprocess_failed", "message": f"Vorverarbeitung Start: HTTP {st} {err or ''}"}
    while True:
        time.sleep(5)
        st, body = _hieltech_lora_local("GET", f"/v1/dataset/preprocess_status/{task}")
        data, err = _hieltech_lora_data(body)
        _hieltech_lora_set_progress(name, phase="preprocess", current=data.get("current"), total=data.get("total"))
        if data.get("status") == "completed":
            break
        if data.get("status") == "failed" or st >= 400:
            return {"ok": False, "error": "preprocess_failed", "message": str(data.get("error") or err or st)}
        if time.time() - t0 > 2 * 3600:
            return {"ok": False, "error": "preprocess_timeout", "message": "Vorverarbeitung > 2 Std."}

    _hieltech_lora_set_progress(name, phase="train", epoch=0, epochs=epochs)
    st, body = _hieltech_lora_local("POST", "/v1/training/start_lokr", {
        "tensor_dir": tensor_dir, "output_dir": out_dir, "train_epochs": int(epochs), "learning_rate": float(learning_rate),
        "save_every_n_epochs": int(save_every), "train_batch_size": 1, "gradient_accumulation": 4,
    })
    data, err = _hieltech_lora_data(body)
    if st >= 400 or err:
        return {"ok": False, "error": "train_start_failed", "message": f"Training Start: HTTP {st} {err or body}"}
    last = {}
    while True:
        time.sleep(15)
        st, body = _hieltech_lora_local("GET", "/v1/training/status")
        data, err = _hieltech_lora_data(body)
        last = data or {}
        _hieltech_lora_set_progress(name, phase="train", epoch=last.get("current_epoch"), epochs=epochs, step=last.get("current_step"),
                                    loss=last.get("current_loss"), status=str(last.get("status"))[:200],
                                    eta_s=last.get("estimated_time_remaining"))
        if not last.get("is_training"):
            break
    final_dir = os.path.join(out_dir, "final")
    if last.get("error") or not os.path.isdir(final_dir):
        return {"ok": False, "error": "train_failed", "message": str(last.get("error") or last.get("status"))[:500]}
    result = {"ok": True, "name": name, "dataset": dataset, "samples": n, "epochs": int(epochs), "final": final_dir,
              "files": sorted(os.listdir(final_dir)), "last_loss": last.get("current_loss"), "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(out_dir, "hieltech_ergebnis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    _hieltech_lora_set_progress(name, phase="done", result=result)
    print(f"[HIELTECH-Info] LoRA {name} fertig: {result}", file=sys.stderr, flush=True)
    return result


@app.post("/v1/hieltech_lora_train_start")
async def _hieltech_lora_train_start(dataset: str = Form(...), name: str = Form(...), epochs: str = Form(default="300"),
                                     learning_rate: str = Form(default="0.03"), save_every: str = Form(default="50")):
    if not _HIELTECH_LORA_NAME_RE.match(dataset or "") or not _HIELTECH_LORA_NAME_RE.match(name or ""):
        return Response(status_code=400, content=b"ungueltiger Name")
    try:
        ep, lr, se = max(1, min(5000, int(epochs))), float(learning_rate), max(1, int(save_every))
    except ValueError:
        return Response(status_code=400, content=b"ungueltige Zahl")
    laufend = _hieltech_lora_running_job()
    if laufend:  # doppelte Start-Anfrage (z. B. Wiederholung nach 502 beim Kaltstart) -> denselben Auftrag zurückgeben
        return {"ok": True, "job_id": laufend, "already_running": True}
    _HIELTECH_LORA_PROGRESS.pop(name, None)
    return _hieltech_job_start("lora_train", _hieltech_lora_train_sync, dataset, name, ep, lr, se)


@app.get("/v1/hieltech_lora_progress")
async def _hieltech_lora_progress(name: str = ""):
    return {"ok": True, "progress": _HIELTECH_LORA_PROGRESS.get(name) or {}}


@app.post("/v1/hieltech_lora_stop")
async def _hieltech_lora_stop():
    st, body = _hieltech_lora_local("POST", "/v1/training/stop", {})
    return {"ok": st < 400, "http": st}


def _hieltech_lora_generate(prompt, lyrics, duration, seed):
    """Ein Song über die normale ACE-Step-Warteschlange dieses Workers; gibt den Dateipfad zurück."""
    import urllib.parse
    payload = {"prompt": prompt, "lyrics": lyrics or "[Instrumental]", "audio_duration": float(duration), "seed": int(seed),
               "use_random_seed": False, "batch_size": 1, "inference_steps": 64, "guidance_scale": 7.0, "thinking": False,  # ohne Sprachmodell (bleibt nach dem Training entladen)
               "audio_format": "mp3", "mp3_bitrate": "320k", "task_type": "text2music"}
    st, body = _hieltech_lora_local("POST", "/release_task", payload)
    data, err = _hieltech_lora_data(body)
    tid = (data or {}).get("task_id")
    if not tid:
        raise RuntimeError(f"release_task: HTTP {st} {err or body}")
    t0 = time.time()
    while time.time() - t0 < 1200:
        time.sleep(3)
        st, body = _hieltech_lora_local("POST", "/query_result", {"task_id_list": [tid]})
        items = (body or {}).get("data") or []
        it = items[0] if items else {}
        if it.get("status") == 1:
            res = json.loads(it.get("result") or "[]")
            first = res[0] if isinstance(res, list) and res else res
            q = urllib.parse.urlparse(first.get("file") or "")
            path = urllib.parse.parse_qs(q.query).get("path", [""])[0]
            if not path or not os.path.isfile(path):
                raise RuntimeError(f"Audiodatei nicht gefunden: {first.get('file')}")
            return path, first.get("dit_model")
        if it.get("status") == 2:
            raise RuntimeError("Erzeugung fehlgeschlagen")
    raise RuntimeError("Erzeugung > 20 Min")


def _hieltech_lora_test_sync(name, items_json, scale):
    try:
        from acestep.training.path_safety import set_safe_root
        set_safe_root(_HIELTECH_LORA_HOME)
    except Exception:  # noqa: BLE001
        pass
    final_dir = os.path.join(_HIELTECH_LORA_HOME, "out", name, "final")
    w = os.path.join(final_dir, "lokr_weights.safetensors")
    adapter = w if os.path.isfile(w) else final_dir
    if not os.path.exists(adapter):
        return {"ok": False, "error": "no_adapter", "message": f"Adapter fehlt: {adapter}"}
    try:
        items = [x for x in json.loads(items_json or "[]") if isinstance(x, dict)][:8]
    except ValueError:
        items = []
    if not items:
        return {"ok": False, "error": "no_items"}
    results = []
    for mode in ("mit", "ohne"):
        if mode == "mit":
            st, body = _hieltech_lora_local("POST", "/v1/lora/load", {"lora_path": adapter})
            data, err = _hieltech_lora_data(body)
            if st >= 400 or err:
                return {"ok": False, "error": "lora_load_failed", "message": f"HTTP {st} {err or body}"}
            if scale is not None:
                _hieltech_lora_local("POST", "/v1/lora/scale", {"scale": float(scale)})
        else:
            _hieltech_lora_local("POST", "/v1/lora/unload", {})
        for i, it in enumerate(items, 1):
            try:
                path, dit = _hieltech_lora_generate(it.get("prompt", ""), it.get("lyrics", ""), it.get("duration", 45), it.get("seed", 42))
                hp, token = _hieltech_handoff_store(path, "mp3")
                results.append({"nr": i, "mode": mode, "label": str(it.get("label") or i)[:40], "result_token": token,
                                "size_bytes": os.path.getsize(hp), "audio_ext": "mp3", "dit_model": dit})
            except Exception as e:  # noqa: BLE001
                results.append({"nr": i, "mode": mode, "label": str(it.get("label") or i)[:40], "error": str(e)[:300]})
    _hieltech_lora_local("POST", "/v1/lora/unload", {})  # immer ohne Adapter zurücklassen (App!)
    return {"ok": True, "adapter": adapter, "results": results}


@app.post("/v1/hieltech_lora_test_start")
async def _hieltech_lora_test_start(name: str = Form(...), items: str = Form(...), scale: str = Form(default="")):
    if not _HIELTECH_LORA_NAME_RE.match(name or ""):
        return Response(status_code=400, content=b"ungueltiger Name")
    sc = None
    if scale.strip():
        try:
            sc = max(0.0, min(1.0, float(scale)))
        except ValueError:
            return Response(status_code=400, content=b"ungueltige Staerke")
    laufend = _hieltech_lora_running_job()
    if laufend:
        return {"ok": True, "job_id": laufend, "already_running": True}
    return _hieltech_job_start("lora_test", _hieltech_lora_test_sync, name, items, sc)


def _hieltech_seedvc_setup_sync():
    py, d, ok, log = _hieltech_seedvc_ensure()
    return {"ok": ok, "seedvc_dir": d, "python": py, "log": (log or "")[-2500:]}


@app.post("/v1/hieltech_seedvc_setup_start")
async def _hieltech_seedvc_setup_start():
    """Seed-VC einmal auf dem Netzwerk-Volume einrichten (Terminal-Test vor dem ersten Chor)."""
    return _hieltech_job_start("seedvc_setup", _hieltech_seedvc_setup_sync)


@app.post("/v1/hieltech_backing_chor_start")
async def _hieltech_backing_chor_start(
    audio: UploadFile,
    key_scale: str = Form(default=""),
    voices: str = Form(default=""),
    humanize: str = Form(default="0"),
):
    """Antwort (über /v1/hieltech_job_status): {"ok", "voices": [{slot, gender, role, result_token,
    size_bytes, audio_ext, rvc_applied}], "steps_applied", "warnings", "tonart"} – je Stimme
    abholen über /v1/hieltech_fetch_chunk (selber Worker)."""
    src_bytes = await audio.read()
    return _hieltech_job_start("backing_chor", _hieltech_backing_chor_sync, src_bytes, audio.filename, key_scale, voices, humanize)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
