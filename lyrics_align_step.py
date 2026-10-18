#!/usr/bin/env python3
"""
"Text korrigieren" per Abgleich gegen den EIGENEN Songtext (NEU, 18. Okt 2026).

Warum: Whisper (frei mithoeren) war langsam und hoerte Patois falsch, ACE-Steps
"Musik verstehen" erfand einen ganz anderen Text. Hier wird NICHT geraten, WAS
gesungen wird - es wird nur geprueft, WELCHE Zeilen des Nutzers WO im Song zu
hoeren sind (auch mehrfach, auch gar nicht) und welche Woerter einer Zeile
fehlen. Erfinden ist damit unmoeglich.

Ablauf:
  1. --separate: Demucs htdemucs_ft -> Gesangsspur (eigener Prozess, gibt den
     GPU-Speicher danach komplett frei)
  2. --align: CTC-Modell (wav2vec2, buchstabenbasiert; es prueft nur
     bekannten Text, raet nicht frei -> Patois-Schreibweise kein Problem) liefert pro 20 ms die Wahrscheinlichkeit jedes
     Buchstabens; ein eigener Viterbi-Abgleich ueber einen "Zeilen-Graphen"
     (jede Zeile = Buchstabenkette; nach jeder Zeile darf jede andere Zeile
     folgen = Wiederholung/Auslassung; einzelne Woerter duerfen fehlen;
     dazwischen "Fueller" fuer Instrumental-/Ad-lib-Stellen).

Aufruf: python lyrics_align_step.py <song> <zeilen.json> <ausgabe.json>
  zeilen.json: [["Mama","plant","a","mango",...], [...], ...]  (Lead-Zeilen)
Ausgabe: {"ok":true,"instances":[{"k":0,"words":[1,1,0,...],"t0":12.3,"t1":15.8,"conf":0.9}],...}
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FRAME_S = 0.02  # 50 Bilder pro Sekunde (wav2vec2/MMS)
ALPHA = float(os.environ.get("HIELTECH_ALIGN_FILLER_ALPHA", "1.5"))
P_JUMP = float(os.environ.get("HIELTECH_ALIGN_JUMP", "-15"))
P_SKIP = float(os.environ.get("HIELTECH_ALIGN_SKIP_WORD", "-6"))
# Wort gilt als "nicht gesungen", wenn seine Buchstaben im Schnitt so viel
# schlechter passen als der Fueller (Log-Einheiten pro Bild)
WORD_DROP_MARGIN = float(os.environ.get("HIELTECH_ALIGN_WORD_DROP", "-1.2"))
# Lizenz beachten: die MMS-Modelle von Meta sind nur fuer NICHT-kommerzielle
# Nutzung freigegeben (CC-BY-NC) - die App ist kommerziell. Deshalb
# wav2vec2-large-960h-lv60-self (Apache-2.0, Buchstaben A-Z und '), das
# fuer einen Abgleich gegen bekannten Text (kein freies Raten) auch bei
# Patois gut genug ist.
HF_MODEL = os.environ.get("HIELTECH_ALIGN_MODEL") or "facebook/wav2vec2-large-960h-lv60-self"
CHUNK_S = 30.0
OVERLAP_S = 1.0


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def norm_word(w):
    w = unicodedata.normalize("NFKD", str(w)).lower()
    w = "".join(c for c in w if not unicodedata.combining(c))
    return "".join(c for c in w if ("a" <= c <= "z") or c == "'")


# ----------------------------------------------------------------------------
# Viterbi ueber den Zeilen-Graphen (reines numpy, ohne GPU)
# ----------------------------------------------------------------------------
def build_graph(lines_tokens, blank, delim=None):
    """lines_tokens: Liste von Zeilen, jede Zeile = Liste von Woertern, jedes
    Wort = Liste von Token-IDs. Gibt Zustands-Arrays zurueck."""
    tok, line_of, word_of, is_blank = [], [], [], []
    line_start, line_end = [], []
    allow2 = []
    skip_from, skip_to = [], []
    for k, words in enumerate(lines_tokens):
        seq = []  # (token, word_index)
        for w, ids in enumerate(words):
            for c in ids:
                seq.append((c, w))
            if delim is not None and w < len(words) - 1:
                seq.append((delim, w))
        base = len(tok)
        line_start.append(base)
        # erweiterte CTC-Folge: b c1 b c2 ... cn b
        word_first_blank = {}
        for i, (c, w) in enumerate(seq):
            if w not in word_first_blank:
                word_first_blank[w] = len(tok)
            tok.append(blank); line_of.append(k); word_of.append(-1); is_blank.append(True); allow2.append(False)
            prev_c = seq[i - 1][0] if i > 0 else None
            tok.append(c); line_of.append(k); word_of.append(w); is_blank.append(False); allow2.append(i > 0 and prev_c != c)
        tok.append(blank); line_of.append(k); word_of.append(-1); is_blank.append(True); allow2.append(False)
        line_end.append(len(tok) - 1)
        # Wort ueberspringen: Blank vor Wort w -> Blank vor Wort w+1 (bzw. Zeilenende)
        nw = len(words)
        for w in range(nw):
            frm = word_first_blank[w]
            to = word_first_blank[w + 1] if w + 1 < nw else line_end[-1]
            skip_from.append(frm)
            skip_to.append(to)
    return {
        "tok": np.array(tok, dtype=np.int64),
        "line_of": np.array(line_of, dtype=np.int32),
        "word_of": np.array(word_of, dtype=np.int32),
        "is_blank": np.array(is_blank, dtype=bool),
        "allow2": np.array(allow2, dtype=bool),
        "line_start": np.array(line_start, dtype=np.int64),
        "line_end": np.array(line_end, dtype=np.int64),
        "skip_from": np.array(skip_from, dtype=np.int64),
        "skip_to": np.array(skip_to, dtype=np.int64),
    }


def viterbi(E, g, blank):
    """E: (T, V) Log-Wahrscheinlichkeiten. Gibt (Pfad, Fueller) zurueck.
    Pfad[t] = ("L", zustand) oder ("F", j)."""
    T = E.shape[0]
    S = len(g["tok"])
    L = len(g["line_start"])
    NEG = -1e18
    ls, le = g["line_start"], g["line_end"]
    start_mask = np.zeros(S, dtype=bool)
    start_mask[ls] = True
    shift1_ok = ~start_mask
    shift2_ok = g["allow2"].copy()
    nonblank = np.ones(E.shape[1], dtype=bool)
    nonblank[blank] = False
    filler = np.maximum(E[:, blank], E[:, nonblank].max(axis=1) - ALPHA)
    emit_tok = g["tok"]
    entry_states = np.concatenate([ls, ls + 1])
    entry_line = np.concatenate([np.arange(L), np.arange(L)])
    skip_from, skip_to = g["skip_from"], g["skip_to"]
    skip_src_of = np.full(S, -1, dtype=np.int64)
    skip_src_of[skip_to] = skip_from  # je Ziel genau eine Quelle

    V = np.full(S, NEG)
    F = np.full(L + 1, NEG)  # F[0] = Start, F[j+1] = nach Zeile j
    F[0] = filler[0]
    V[entry_states] = np.where(entry_line == 0, 0.0, P_JUMP) + E[0, emit_tok[entry_states]]

    bp = np.zeros((T, S), dtype=np.int8)          # 0 bleiben, 1 s-1, 2 s-2, 3 Wortsprung, 4 Eintritt
    ent_src = np.zeros((T, L), dtype=np.int32)    # >=0: Ende Zeile j ; <0: F[-1-src]
    bpF = np.zeros((T, L + 1), dtype=np.int8)     # 0 bleiben, 1 aus Zeilenende
    endsel = np.zeros((T, L), dtype=bool)         # True: Zeilenende war der letzte Buchstabe (le-1)
    bp[0, entry_states] = 5                       # 5 = Start

    idx = np.arange(S)
    for t in range(1, T):
        cand1 = np.full(S, NEG); cand1[1:] = V[:-1]; cand1[~shift1_ok] = NEG
        cand2 = np.full(S, NEG); cand2[2:] = V[:-2]; cand2[~shift2_ok] = NEG
        cand3 = np.full(S, NEG)
        cand3[skip_to] = V[skip_from] + P_SKIP
        stack = np.stack([V, cand1, cand2, cand3])
        arg = stack.argmax(axis=0)
        best = stack[arg, idx]
        es_last = V[le - 1] > V[le]
        endsel[t] = es_last
        endv = np.where(es_last, V[le - 1], V[le])
        # Eintritt in Zeile k
        allsrc = np.concatenate([F, endv])
        ib = int(allsrc.argmax())
        jump_val = allsrc[ib] + P_JUMP
        jump_src = (-1 - ib) if ib < L + 1 else (ib - (L + 1))
        seqF = F[:L]                                   # F[k] = nach Zeile k-1 (k=0: Start)
        seqE = np.concatenate([[NEG], endv[:-1]])      # Ende Zeile k-1
        seq_best = np.maximum(seqF, seqE)
        seq_src = np.where(seqF >= seqE, -1 - np.arange(L), np.arange(L) - 1)
        use_seq = seq_best >= jump_val
        ent_val = np.where(use_seq, seq_best, jump_val)
        ent_src[t] = np.where(use_seq, seq_src, jump_src)
        ent_full = np.concatenate([ent_val, ent_val])
        better = ent_full > best[entry_states]
        es = entry_states[better]
        best[es] = ent_full[better]
        arg[es] = 4
        bp[t] = arg
        fromE = np.concatenate([[NEG], endv])
        bpF[t] = (fromE > F).astype(np.int8)
        F = np.maximum(F, fromE) + filler[t]
        V = best + E[t, emit_tok]

    # Ende
    es_last = V[le - 1] > V[le]
    endv = np.where(es_last, V[le - 1], V[le])
    if endv.max() > F.max():
        k = int(endv.argmax())
        cur = ("L", int(le[k] - 1) if es_last[k] else int(le[k]))
    else:
        cur = ("F", int(F.argmax()))
    path = [None] * T
    t = T - 1
    while t >= 0:
        path[t] = cur
        if t == 0:
            break
        if cur[0] == "F":
            j = cur[1]
            if bpF[t, j] == 1:
                kk = j - 1
                cur = ("L", int(le[kk] - 1) if endsel[t, kk] else int(le[kk]))
            t -= 1
            continue
        s = cur[1]
        code = bp[t, s]
        if code == 0:
            nxt = ("L", s)
        elif code == 1:
            nxt = ("L", s - 1)
        elif code == 2:
            nxt = ("L", s - 2)
        elif code == 3:
            nxt = ("L", int(skip_src_of[s]))
        elif code == 4:
            k = int(g["line_of"][s])
            src = int(ent_src[t, k])
            if src >= 0:
                nxt = ("L", int(le[src] - 1) if endsel[t, src] else int(le[src]))
            else:
                nxt = ("F", -1 - src)
        else:
            break
        cur = nxt
        t -= 1
    return path, filler


def path_to_instances(path, E, g, filler, lines_tokens):
    insts = []
    cur = None
    prev = None
    for t, st in enumerate(path):
        if st is None:
            continue
        if st[0] == "L":
            s = st[1]
            k = int(g["line_of"][s])
            new = cur is None or prev is None or prev[0] != "L" or int(g["line_of"][prev[1]]) != k or s < prev[1]
            if new:
                cur = {"k": k, "t0": t * FRAME_S, "t1": t * FRAME_S, "words": set(), "score": 0.0, "fill": 0.0, "n": 0, "wm": {}}
                insts.append(cur)
            cur["t1"] = (t + 1) * FRAME_S
            w = int(g["word_of"][s])
            if w >= 0:
                cur["words"].add(w)
                d = float(E[t, g["tok"][s]]) - float(filler[t])
                cur["score"] += float(E[t, g["tok"][s]])
                cur["fill"] += float(filler[t])
                cur["n"] += 1
                a = cur["wm"].setdefault(w, [0.0, 0])
                a[0] += d
                a[1] += 1
        prev = st
    out = []
    for c in insts:
        nw = len(lines_tokens[c["k"]])
        if not c["words"]:
            continue
        margin = (c["score"] - c["fill"]) / max(1, c["n"])  # >0: besser als Fueller
        out.append({
            "k": c["k"],
            "words": [1 if (w in c["words"] and c["wm"][w][0] / c["wm"][w][1] > WORD_DROP_MARGIN) else 0 for w in range(nw)],
            "wordMargin": [round(c["wm"][w][0] / c["wm"][w][1], 2) if w in c["wm"] else None for w in range(nw)],
            "t0": round(c["t0"], 2),
            "t1": round(c["t1"], 2),
            "conf": round(1.0 / (1.0 + math.exp(-margin)), 3),
        })
    return out


# ----------------------------------------------------------------------------
# Modell + Emissionen
# ----------------------------------------------------------------------------
def _cache_dir():
    for c in (os.environ.get("HF_HOME"), "/runpod-volume/hf_cache", "/app/checkpoints/hf_cache"):
        if c and os.path.isdir(c):
            return c
    return None


def load_ctc():
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"cache_dir": _cache_dir()} if _cache_dir() else {}
    proc = AutoProcessor.from_pretrained(HF_MODEL, **kw)
    model = Wav2Vec2ForCTC.from_pretrained(HF_MODEL, **kw).to(dev).eval()
    vocab = proc.tokenizer.get_vocab()
    return model, vocab, dev


def emissions(model, dev, y16):
    import torch

    sr = 16000
    chunk, ov = int(CHUNK_S * sr), int(OVERLAP_S * sr)
    outs = []
    pos = 0
    with torch.inference_mode():
        while pos < len(y16):
            a = max(0, pos - ov)
            b = min(len(y16), pos + chunk + ov)
            x = torch.from_numpy(y16[a:b]).float().unsqueeze(0).to(dev)
            x = (x - x.mean()) / (x.std() + 1e-7)
            lg = model(x).logits[0].float().log_softmax(-1).cpu().numpy()
            fpers = lg.shape[0] / ((b - a) / sr)
            cut0 = int(round((pos - a) / sr * fpers))
            keep = int(round(min(chunk, len(y16) - pos) / sr * fpers))
            outs.append(lg[cut0:cut0 + keep])
            pos += chunk
    return np.concatenate(outs, axis=0)


def tokenize_lines(lines, vocab):
    lower = {k.lower(): v for k, v in vocab.items() if len(k) == 1}
    blank = vocab.get("<pad>", vocab.get("<blank>", vocab.get("[PAD]", 0)))
    delim = vocab.get("|")
    out, kept = [], []
    for words in lines:
        tw, kw = [], []
        for w in words:
            ids = [lower[c] for c in norm_word(w) if c in lower]
            if ids:
                tw.append(ids)
                kw.append(True)
            else:
                kw.append(False)
        out.append(tw)
        kept.append(kw)
    return out, kept, blank, delim


def align(vocals_path, lines):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    t0 = time.time()
    y, sr = sf.read(vocals_path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 16000:
        g_ = gcd(int(sr), 16000)
        y = resample_poly(y, 16000 // g_, int(sr) // g_).astype(np.float32)
    model, vocab, dev = load_ctc()
    _log(f"Text-Abgleich: Modell geladen ({HF_MODEL}, {dev}) nach {time.time() - t0:.1f}s")
    E = emissions(model, dev, y)
    del model
    _log(f"Text-Abgleich: {E.shape[0]} Bilder berechnet nach {time.time() - t0:.1f}s")
    toks, kept, blank, delim = tokenize_lines(lines, vocab)
    # Zeilen ohne erkennbare Buchstaben bekommen einen Platzhalter, damit die Nummern stimmen
    usable = [i for i, tw in enumerate(toks) if tw]
    g = build_graph([toks[i] for i in usable], blank, delim)
    path, filler = viterbi(E, g, blank)
    inst = path_to_instances(path, E, g, filler, [toks[i] for i in usable])
    # zurueck auf Original-Zeilen/-Woerter abbilden
    result = []
    for it in inst:
        k = usable[it["k"]]
        mask_tok = iter(it["words"])
        words = [next(mask_tok) if kept[k][w] else 1 for w in range(len(lines[k]))]
        result.append({**it, "k": k, "words": words})
    _log(f"Text-Abgleich: {len(result)} Zeilen-Auftritte gefunden nach {time.time() - t0:.1f}s")
    return result, E.shape[0] * FRAME_S


def separate(song, work):
    import soundfile as sf
    from vocal_separate_uvr_clean_step import separate_vocals_htdemucs_ft

    vocals, sr = separate_vocals_htdemucs_ft(song, work)
    out = os.path.join(work, "vocals_for_align.wav")
    sf.write(out, vocals, sr)
    return out


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--separate":
        p = separate(sys.argv[2], sys.argv[3])
        print(json.dumps({"vocals": p}), flush=True)
        return
    if len(sys.argv) >= 5 and sys.argv[1] == "--align":
        with open(sys.argv[3], "r", encoding="utf-8") as f:
            lines = json.load(f)
        inst, dur = align(sys.argv[2], lines)
        with open(sys.argv[4], "w", encoding="utf-8") as f:
            json.dump({"ok": True, "instances": inst, "duration": round(dur, 2), "model": HF_MODEL}, f)
        return
    if len(sys.argv) < 4:
        print("Nutzung: python lyrics_align_step.py <song> <zeilen.json> <ausgabe.json>")
        sys.exit(1)
    song, lines_json, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with tempfile.TemporaryDirectory() as work:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--separate", song, work],
                           capture_output=True, text=True, cwd=HERE, env=env)
        if r.returncode != 0:
            _log(((r.stdout or "") + (r.stderr or ""))[-2000:])
            sys.exit(2)
        vocals = json.loads(r.stdout.strip().splitlines()[-1])["vocals"]
        _log(f"Text-Abgleich: Gesang getrennt nach {time.time() - t0:.1f}s")
        tmp_out = os.path.join(work, "align.json")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--align", vocals, lines_json, tmp_out],
                           capture_output=True, text=True, cwd=HERE, env=env)
        sys.stderr.write((r.stderr or "")[-3000:])
        if r.returncode != 0 or not os.path.isfile(tmp_out):
            _log(((r.stdout or "") + (r.stderr or ""))[-2000:])
            sys.exit(3)
        with open(tmp_out, "r", encoding="utf-8") as f:
            data = json.load(f)
    data["seconds"] = round(time.time() - t0, 1)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(json.dumps({"ok": True, "instances": len(data["instances"]), "seconds": data["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
