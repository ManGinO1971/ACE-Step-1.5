#!/usr/bin/env python3
"""
HIELTECH Stems – Test 4 (29. Okt 2026): Instrumente wie den Gesang (Lead V2) neu aufbauen.

Gesang-Kette (Lead V2): Extract aus dem ganzen Song -> Zeitmaske -> Gate -> Melodie-Karte -> Vocoder C.
Für Instrumente dasselbe Prinzip:
  1. Von den neu erzeugten Fassungen (Extract aus dem GANZEN Song) die genaueste wählen
     (Töne/Rhythmus im Vergleich zur getrennten Spur – die getrennte Spur ist NUR Vorlage).
  2. Zeitmaske: wo das Instrument im Original schweigt, schweigt auch die neue Spur.
  3. Gate: leises Grundrauschen der neuen Spur weg.
  4. "Karte" (Mel-Spektrum) der neuen Spur -> Instrumenten-Vocoder BigVGAN v2 (NVIDIA, MIT,
     auf Musik/Instrumente trainiert) baut den Klang komplett neu auf.
       B = Karte unverändert neu aufgebaut
       C = Karte vorher gesäubert (Rauschboden je Frequenzband abgezogen), dann neu aufgebaut
  (Bass zusätzlich D = genau die Gesangs-Kette RMVPE + Vocoder C – macht runpod_entrypoint.py.)
KEINE Lautstärke-Anpassung (kommt erst ganz zum Schluss), nur Übersteuerungsschutz.

Aufruf: python stems_karte_step.py <getrennt_ordner> <kandidaten.json> <ausgabeordner>
Ausgabe: {inst}_A_neu_roh.wav, {inst}_B_karte.wav, {inst}_C_karte_sauber.wav,
         _zwischen/{inst}_maske.wav (Eingang für Vocoder C beim Bass). Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stems_post_step import SR, fit, load, score  # noqa: E402

HOP = 512
BIGVGAN_DIR = os.environ.get("HIELTECH_BIGVGAN_DIR", "/runpod-volume/bigvgan_repo")
BIGVGAN_COMMIT = "7d2b454564a6c7d014227f635b7423881f14bdac"
BIGVGAN_MODEL = "nvidia/bigvgan_v2_44khz_128band_512x"
MASKE_DB = float(os.environ.get("HIELTECH_STEMS_MASKE_DB", "-45"))   # Original unter max-45 dB = Instrument schweigt
GATE_DB = float(os.environ.get("HIELTECH_STEMS_GATE_DB", "-55"))     # neue Spur unter max-55 dB = Rauschen


def frame_db(x):
    y = x.mean(1)
    n = int(np.ceil(len(y) / HOP))
    y = np.pad(y, (0, n * HOP - len(y)))
    r = np.sqrt((y.reshape(n, HOP) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(r)


def weich(mask, halte_s=0.15, blende_s=0.02):
    """Maske aufweiten (Ausklang/Anschlag nicht abschneiden) und weich überblenden -> je Sample."""
    h = max(1, int(halte_s * SR / HOP))
    m = np.convolve(mask.astype(float), np.ones(2 * h + 1), mode="same") > 0
    k = max(1, int(blende_s * SR / HOP))
    m = np.convolve(m.astype(float), np.ones(k) / k, mode="same")
    return np.clip(m, 0, 1)


def maske_gate(sep, neu):
    n = neu.shape[0]
    ds, dn = frame_db(sep), frame_db(neu)
    m = min(len(ds), len(dn))
    an = (ds[:m] > ds.max() + MASKE_DB) & (dn[:m] > dn.max() + GATE_DB)
    g = weich(an)
    gain = np.interp(np.arange(n), np.arange(m) * HOP + HOP / 2, g)
    return neu * gain[:, None], float(an.mean())


_BV = {}


def bigvgan():
    if "m" in _BV:
        return _BV["m"], _BV["h"], _BV["mel"], _BV["dev"]
    import torch
    if not os.path.isfile(os.path.join(BIGVGAN_DIR, "bigvgan.py")):
        subprocess.run(["git", "clone", "-q", "https://github.com/NVIDIA/BigVGAN.git", BIGVGAN_DIR], check=True, timeout=300)
        subprocess.run(["git", "-C", BIGVGAN_DIR, "checkout", "-q", BIGVGAN_COMMIT], check=True, timeout=60)
    try:
        import matplotlib  # noqa: F401  (utils.py im BigVGAN-Repo importiert matplotlib)
    except ImportError:
        mpl = types.ModuleType("matplotlib")
        mpl.use = lambda *a, **k: None
        sys.modules["matplotlib"] = mpl
        sys.modules["matplotlib.pylab"] = types.ModuleType("matplotlib.pylab")
    sys.path.insert(0, BIGVGAN_DIR)
    import bigvgan as bv
    from meldataset import get_mel_spectrogram
    from huggingface_hub import hf_hub_download
    cfg = hf_hub_download(repo_id=BIGVGAN_MODEL, filename="config.json")
    wts = hf_hub_download(repo_id=BIGVGAN_MODEL, filename="bigvgan_generator.pt")
    h = bv.load_hparams_from_json(cfg)
    m = bv.BigVGAN(h, use_cuda_kernel=False)
    sd = torch.load(wts, map_location="cpu")["generator"]
    try:
        m.load_state_dict(sd)
        m.remove_weight_norm()
    except RuntimeError:
        m.remove_weight_norm()
        m.load_state_dict(sd)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.eval().to(dev)
    if int(h.sampling_rate) != SR or int(h.hop_size) != HOP:
        raise RuntimeError(f"BigVGAN-Einstellung unerwartet: {h.sampling_rate} Hz / hop {h.hop_size}")
    _BV.update(m=m, h=h, mel=get_mel_spectrogram, dev=dev)
    return m, h, get_mel_spectrogram, dev


def saeubern(mel_log):
    """Rauschboden je Mel-Band (leiseste 10 % der klingenden Frames) abziehen, Rest behalten."""
    lin = np.exp(mel_log)
    energie = lin.sum(0)
    aktiv = energie > energie.max() * 10 ** (GATE_DB / 10)
    if aktiv.sum() < 20:
        return mel_log
    boden = np.percentile(lin[:, aktiv], 10, axis=1, keepdims=True)
    sauber = np.maximum(lin - 1.5 * boden, 0.03 * lin)
    return np.log(np.maximum(sauber, 1e-5))


def neu_aufbauen(x, sauber):
    """x: (n, 2) -> Klang aus der Karte neu aufgebaut, je Kanal, in Stücken mit Überblendung."""
    import torch
    m, h, mel_fn, dev = bigvgan()
    n = x.shape[0]
    out = np.zeros_like(x)
    chunk, ctx = 1024, 32  # Frames (~12 s) + Rand
    for ch in range(x.shape[1]):
        y = torch.from_numpy(np.clip(x[:, ch], -1, 1).astype(np.float32)).unsqueeze(0)
        mel = mel_fn(y, h)[0].numpy()  # (128, T)
        if sauber:
            mel = saeubern(mel)
        T = mel.shape[1]
        wav = np.zeros(T * HOP, dtype=np.float32)
        wsum = np.zeros(T * HOP, dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, T, chunk):
                a, b = max(0, s - ctx), min(T, s + chunk + ctx)
                g = m(torch.from_numpy(mel[:, a:b]).unsqueeze(0).to(dev)).squeeze().float().cpu().numpy()
                g = g[: (b - a) * HOP]
                w = np.ones(len(g), dtype=np.float32)
                r = ctx * HOP
                if a > 0:
                    w[:r] = np.linspace(0, 1, r)
                if b < T:
                    w[-r:] = np.linspace(1, 0, r)
                wav[a * HOP:a * HOP + len(g)] += g * w
                wsum[a * HOP:a * HOP + len(g)] += w
        wav = wav / np.maximum(wsum, 1e-6)
        out[:, ch] = fit(wav[:, None], n)[:, 0]
    return out


def schutz(x):
    p = float(np.abs(x).max()) or 1.0
    return x / max(p / 0.98, 1.0)


def main():
    sep_dir, cand_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
    zw = os.path.join(out, "_zwischen")
    os.makedirs(zw, exist_ok=True)
    with open(cand_json) as f:
        cands = json.load(f)
    bericht = {"ok": True, "instrumente": {}}
    for inst, paths in cands.items():
        info = {"kandidaten": []}
        try:
            sep = load(os.path.join(sep_dir, inst + ".wav"))
            n = sep.shape[0]
            best, best_s = None, -9.0
            for p in paths:
                c = fit(load(p), n)
                sc = score(sep, c, inst)
                info["kandidaten"].append({"datei": os.path.basename(p), "punkte": sc[0], "toene_rhythmus": sc[1], "lautstaerke": sc[2]})
                if sc[0] > best_s:
                    best, best_s = c, sc[0]
            info["gewaehlt"] = max(info["kandidaten"], key=lambda k: k["punkte"])["datei"]
            sf.write(os.path.join(out, f"{inst}_A_neu_roh.wav"), schutz(best), SR, subtype="PCM_24")
            mg, anteil = maske_gate(sep, best)
            info["klingt_anteil"] = round(anteil, 3)
            sf.write(os.path.join(zw, f"{inst}_maske.wav"), schutz(mg), SR, subtype="PCM_24")
            for tag, sauber in (("B_karte", False), ("C_karte_sauber", True)):
                try:
                    sf.write(os.path.join(out, f"{inst}_{tag}.wav"), schutz(neu_aufbauen(mg, sauber)), SR, subtype="PCM_24")
                except Exception as e:  # noqa: BLE001
                    info[f"{tag}_fehler"] = str(e)[:200]
        except Exception as e:  # noqa: BLE001
            info["fehler"] = str(e)[:300]
        bericht["instrumente"][inst] = info
        print(f"[Stems-Test4] {inst}: {json.dumps(info, ensure_ascii=False)}", file=sys.stderr, flush=True)
    print(json.dumps(bericht, ensure_ascii=False))


if __name__ == "__main__":
    main()
