#!/usr/bin/env python3
"""
HIELTECH Stems – 53er-Trennung auf der GPU (29. Okt 2026): EIN Instrument aus dem Song als Vorlage.

Gleiches Modell wie die Mac-App (get_stem53.py): BS-Roformer MVSep Mega 53 Stems (MIT,
huggingface noblebarkrr/BS-Roformer-MVSep-Mega-53-stems, je Instrument ein eigener Checkpoint ~78 MB).
Hier OHNE inference.py des Trainings-Repos (das zieht viele Zusatzpakete) – nur die Modell-Klasse
BSRoformer aus ZFTurbo/Music-Source-Separation-Training (fester Stand) + eigene Stück-für-Stück-Rechnung
(20-s-Stücke, halbe Überlappung, weiche Ränder) wie dort.

Aufruf: python stems53_ref_step.py <song> <ausgabe.wav> <instrument> [<instrument2> <ausgabe2.wav> ...]
        [--sechs <ordner mit drums/bass/guitar/piano/vocals/other>]   (14. Nov 2026: Fassung 2, s53_saeubern.py)
        (mehrere Paare: Song wird nur einmal geladen)
Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

MSST_DIR = os.environ.get("HIELTECH_MSST_DIR", "/runpod-volume/mss_train_repo")
MSST_COMMIT = "6fdad13686bb7e84d029ec66f9dbe76e65c93ebf"
HF_REPO = "noblebarkrr/BS-Roformer-MVSep-Mega-53-stems"
SR = 44100


def repo():
    if not os.path.isfile(os.path.join(MSST_DIR, "models", "bs_roformer", "bs_roformer.py")):
        subprocess.run(["git", "clone", "-q", "https://github.com/ZFTurbo/Music-Source-Separation-Training.git", MSST_DIR],
                       check=True, timeout=600)
        subprocess.run(["git", "-C", MSST_DIR, "checkout", "-q", MSST_COMMIT], check=True, timeout=60)
    if MSST_DIR not in sys.path:
        sys.path.insert(0, MSST_DIR)


def lade_config(path):
    import yaml
    with open(path) as f:
        txt = f.read()
    for loader in ("FullLoader", "UnsafeLoader"):
        try:
            return yaml.load(txt, Loader=getattr(yaml, loader))
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("Config nicht lesbar: " + path)


def modell(inst, dev):
    import torch
    from huggingface_hub import hf_hub_download
    repo()
    from models.bs_roformer.bs_roformer import BSRoformer
    # 30.10.: "Disk quota exceeded" auf dem Worker (jede Vorlage 78 MB im normalen Zwischenspeicher).
    # -> eigener Speicherort auf dem Netz-Volume; ist der voll/nicht da: temporär laden und danach löschen.
    cache = os.environ.get("HIELTECH_53_CACHE") or ("/runpod-volume/hf_mega53" if os.path.isdir("/runpod-volume") else None)
    tmp = None
    try:
        cfg_p = hf_hub_download(HF_REPO, f"v1/bs_mega_53stem_{inst}_mvsep_config.yaml", cache_dir=cache)
        ckpt_p = hf_hub_download(HF_REPO, f"v1/bs_mega_53stem_{inst}_mvsep.ckpt", cache_dir=cache)
    except OSError:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="mega53_", dir=os.environ.get("TMPDIR", "/tmp"))
        cfg_p = hf_hub_download(HF_REPO, f"v1/bs_mega_53stem_{inst}_mvsep_config.yaml", local_dir=tmp)
        ckpt_p = hf_hub_download(HF_REPO, f"v1/bs_mega_53stem_{inst}_mvsep.ckpt", local_dir=tmp)
    cfg = lade_config(cfg_p)
    kw = dict(cfg["model"])
    for k in ("freqs_per_bands", "multi_stft_resolutions_window_sizes"):
        if k in kw:
            kw[k] = tuple(kw[k])
    m = BSRoformer(**kw)
    sd = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    for k in ("state_dict", "state", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd)
    chunk = int(cfg.get("inference", {}).get("chunk_size", cfg["audio"]["chunk_size"]))
    if tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return m.to(dev).eval(), chunk


def trennen(m, chunk, mix, dev):
    """mix: (2, n) float32 -> (2, n). 20-s-Stücke, halbe Überlappung, lineare Ränder (wie MSST demix)."""
    import torch
    n = mix.shape[1]
    step = chunk // 2
    fade = chunk // 10
    win = np.ones(chunk, np.float32)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    pad_l = fade
    total = pad_l + n + fade
    total = chunk if total <= chunk else chunk + -(-(total - chunk) // step) * step
    x = np.zeros((2, total), np.float32)
    x[:, pad_l:pad_l + n] = mix
    out = np.zeros_like(x)
    w = np.zeros(total, np.float32)
    use_amp = dev == "cuda"
    with torch.inference_mode():
        for s in range(0, total - chunk + 1, step):
            t = torch.from_numpy(x[:, s:s + chunk]).unsqueeze(0).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                y = m(t)
            y = y.float().squeeze(0).cpu().numpy()
            if y.ndim == 3:
                y = y[0]
            out[:, s:s + chunk] += y * win
            w[s:s + chunk] += win
    return (out / np.maximum(w, 1e-6))[:, pad_l:pad_l + n]


def sechs_laden(ordner, n):
    """6-Spuren-Trennung des Songs (für Stufe 2) als {name: (2, n)} – fehlende Spuren werden ausgelassen."""
    import glob
    out = {}
    for name in ("drums", "bass", "guitar", "piano", "vocals", "other"):
        f = sorted(glob.glob(os.path.join(ordner, f"{name}.*")))
        if not f:
            continue
        a, sr = sf.read(f[0], dtype="float32", always_2d=True)
        if sr != SR:
            import librosa
            a = np.stack([librosa.resample(a[:, c], orig_sr=sr, target_sr=SR) for c in range(a.shape[1])], 1)
        if a.shape[1] == 1:
            a = np.repeat(a, 2, 1)
        y = np.zeros((2, n), np.float32)
        k = min(n, len(a))
        y[:, :k] = a[:k, :2].T
        out[name] = y
    return out


def main():
    import torch
    argv = list(sys.argv)
    sechs_dir = None
    if "--sechs" in argv:
        k = argv.index("--sechs")
        sechs_dir = argv[k + 1]
        del argv[k:k + 2]
    song = argv[1]
    paare = list(zip(argv[3::2], argv[2::2]))  # (instrument, ausgabe)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tmp = song + ".44k.wav"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", song, "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", tmp],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + r.stderr[-300:])
    a, _ = sf.read(tmp, dtype="float32", always_2d=True)
    os.remove(tmp)
    mix = np.ascontiguousarray(a.T)
    bericht = {"ok": True, "geraet": dev, "instrumente": {}, "fassung": 2 if sechs_dir else 1}
    sechs, eltern_cache = {}, {}
    if sechs_dir:
        try:
            import s53_saeubern as S2
            sechs = sechs_laden(sechs_dir, mix.shape[1])
        except Exception as e:  # noqa: BLE001
            bericht["sechs_fehler"] = str(e)[:200]
    for inst, ziel in paare:
        t0 = time.time()
        try:
            m, chunk = modell(inst, dev)
            y = trennen(m, chunk, mix, dev)
            info = {}
            if sechs:
                eltern = [e for e in S2.ELTERN.get(inst, ()) if e in sechs]
                if eltern:
                    kz = tuple(eltern)
                    if kz not in eltern_cache:                  # Eltern-Spur + ihr Spektrum nur einmal je Kombination
                        P_ = np.ascontiguousarray(sum(sechs[e] for e in eltern))
                        eltern_cache.clear()
                        eltern_cache[kz] = (P_, S2._stft(P_))
                    P, SP = eltern_cache[kz]
                    try:
                        B = trennen(m, chunk, np.ascontiguousarray(P), dev)
                        y, info["stufe2"] = S2.verbessern(inst, y, B, P, SR, SP)
                        info["stufe2"]["eltern"] = eltern
                    except Exception as e:  # noqa: BLE001 – nie schlechter als Fassung 1
                        info["stufe2"] = {"fehler": str(e)[:200]}
                try:
                    y, info["saeubern"] = S2.saeubern(inst, y, SR)
                except Exception as e:  # noqa: BLE001
                    info["saeubern"] = {"fehler": str(e)[:200]}
            sf.write(ziel, y.T, SR, subtype="FLOAT")
            bericht["instrumente"][inst] = {"ok": True, "sekunden": round(time.time() - t0, 1),
                                            "pegel_db": round(float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)), 1),
                                            **info}
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            bericht["instrumente"][inst] = {"ok": False, "fehler": str(e)[:300]}
        print(f"[Stems53] {inst}: {bericht['instrumente'][inst]}", file=sys.stderr, flush=True)
    print(json.dumps(bericht, ensure_ascii=False))


if __name__ == "__main__":
    main()
