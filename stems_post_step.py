#!/usr/bin/env python3
"""
HIELTECH Stems – Test 3 (29. Okt 2026): Nachbearbeitung der neu erzeugten Instrumente.

Je Instrument:
  1. Von 3 neu erzeugten Fassungen die genaueste wählen (Vergleich mit der getrennten Spur:
     Töne per Chroma, Rhythmus per Anschlag-Hüllkurve, Lautstärkeverlauf).
  2. Lautstärkeverlauf der getrennten Spur übertragen (Dynamik wie im Original).
  3. Drei Fassungen zum Hörvergleich: A ohne Entrauschen, B Mel-RoFormer-Denoise, C Apollo.

Aufruf: python stems_post_step.py <getrennt_ordner> <kandidaten.json> <ausgabeordner>
  kandidaten.json = {"bass": ["/pfad/seed1.flac", ...], "drums": [...], ...}
Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SR = 44100
PITCHED = {"bass", "guitar", "piano", "other"}
APOLLO_DIR = os.environ.get("HIELTECH_APOLLO_DIR", "/runpod-volume/apollo_repo")
APOLLO_COMMIT = "e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6"


def load(path):
    """-> (n, 2) float32 mit 44,1 kHz (ffmpeg wandelt um)."""
    tmp = tempfile.mktemp(suffix=".wav")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", tmp],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + (r.stderr or "")[-200:])
    a, _ = sf.read(tmp, dtype="float32", always_2d=True)
    os.remove(tmp)
    return a


def fit(a, n):
    return a[:n] if a.shape[0] >= n else np.pad(a, ((0, n - a.shape[0]), (0, 0)))


def score(sep, cand, inst):
    import librosa
    s = librosa.resample(sep.mean(1), orig_sr=SR, target_sr=22050)
    c = librosa.resample(cand.mean(1), orig_sr=SR, target_sr=22050)
    hop = 512
    rs = librosa.feature.rms(y=s, hop_length=hop)[0]
    rc = librosa.feature.rms(y=c, hop_length=hop)[0]
    m = min(len(rs), len(rc))
    env = float(np.corrcoef(rs[:m], rc[:m])[0, 1]) if m > 10 else 0.0
    if inst in PITCHED:
        cs = librosa.feature.chroma_cqt(y=s, sr=22050, hop_length=hop)
        cc = librosa.feature.chroma_cqt(y=c, sr=22050, hop_length=hop)
        m2 = min(cs.shape[1], cc.shape[1], len(rs))
        num = (cs[:, :m2] * cc[:, :m2]).sum(0)
        den = np.linalg.norm(cs[:, :m2], axis=0) * np.linalg.norm(cc[:, :m2], axis=0) + 1e-9
        w = rs[:m2] / (rs[:m2].sum() + 1e-9)
        haupt = float((num / den * w).sum())
    else:
        os_ = librosa.onset.onset_strength(y=s, sr=22050, hop_length=hop)
        oc = librosa.onset.onset_strength(y=c, sr=22050, hop_length=hop)
        m2 = min(len(os_), len(oc))
        haupt = float(np.corrcoef(os_[:m2], oc[:m2])[0, 1]) if m2 > 10 else 0.0
    return round(0.7 * haupt + 0.3 * env, 4), round(haupt, 4), round(env, 4)


def match_envelope(sep, cand):
    """Lautstärkeverlauf der getrennten Spur auf die neue Spur übertragen (geglättet, begrenzt)."""
    hop, win = 512, 2048
    def rms(x):
        y = x.mean(1)
        pad = np.pad(y, (win // 2, win // 2))
        frames = np.lib.stride_tricks.sliding_window_view(pad, win)[::hop]
        return np.sqrt((frames ** 2).mean(1) + 1e-12)
    rs, rc = rms(sep), rms(cand)
    m = min(len(rs), len(rc))
    k = max(1, int(0.3 * SR / hop))  # 0,3 s glätten
    ker = np.ones(k) / k
    rs = np.convolve(rs[:m], ker, mode="same")
    rc = np.convolve(rc[:m], ker, mode="same")
    g_db = np.clip(20 * np.log10((rs + 1e-9) / (rc + 1e-9)), -18.0, 12.0)
    g = 10 ** (g_db / 20)
    t_frames = np.arange(m) * hop
    gain = np.interp(np.arange(cand.shape[0]), t_frames, g)
    out = cand * gain[:, None]
    peak = float(np.abs(out).max()) or 1.0
    return out / max(peak / 0.98, 1.0)


def denoise(path, work):
    from backing_chor_prepare_step import DENOISE_MODEL, run_model
    return run_model(path, DENOISE_MODEL, os.path.join(work, "dn_" + os.path.basename(path)[:-4]), "dry")


_APOLLO = {}


def apollo(path, out_path):
    import torch
    if "m" not in _APOLLO:
        if not os.path.isdir(os.path.join(APOLLO_DIR, "look2hear")):
            subprocess.run(["git", "clone", "-q", "https://github.com/JusperLee/Apollo.git", APOLLO_DIR], check=True, timeout=300)
            subprocess.run(["git", "-C", APOLLO_DIR, "checkout", "-q", APOLLO_COMMIT], check=True, timeout=60)
        sys.path.insert(0, APOLLO_DIR)
        from huggingface_hub import hf_hub_download
        import look2hear.models
        ckpt = hf_hub_download(repo_id="JusperLee/Apollo", filename="pytorch_model.bin")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _APOLLO["dev"] = dev
        _APOLLO["m"] = look2hear.models.BaseModel.from_pretrain(ckpt, sr=SR, win=20, feature_dim=256, layer=6).to(dev).eval()
    m, dev = _APOLLO["m"], _APOLLO["dev"]
    a, _ = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(np.ascontiguousarray(a.T)).unsqueeze(0)
    n = x.shape[-1]
    chunk, pad = 10 * SR, SR
    out = torch.zeros_like(x)
    with torch.inference_mode():
        for s in range(0, n, chunk):
            a0, b0 = max(0, s - pad), min(n, s + chunk + pad)
            y = m(x[..., a0:b0].to(dev)).detach().cpu()
            out[..., s:min(n, s + chunk)] = y[..., s - a0:s - a0 + min(chunk, n - s)]
    sf.write(out_path, out.squeeze(0).numpy().T, SR, subtype="PCM_24")


def main():
    sep_dir, cand_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="stems_post_")
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
            roh = match_envelope(sep, best)
            a_path = os.path.join(out, f"{inst}_A_ohne_entrauschen.wav")
            sf.write(a_path, roh, SR, subtype="PCM_24")
            try:
                d = denoise(a_path, work)
                dd, dsr = sf.read(d, always_2d=True)
                sf.write(os.path.join(out, f"{inst}_B_entrauscht.wav"), fit(dd, n), dsr, subtype="PCM_24")
            except Exception as e:  # noqa: BLE001
                info["entrauschen_fehler"] = str(e)[:200]
            try:
                apollo(a_path, os.path.join(out, f"{inst}_C_apollo.wav"))
            except Exception as e:  # noqa: BLE001
                info["apollo_fehler"] = str(e)[:200]
        except Exception as e:  # noqa: BLE001
            info["fehler"] = str(e)[:300]
        bericht["instrumente"][inst] = info
        print(f"[Stems-Test3] {inst}: {json.dumps(info, ensure_ascii=False)}", file=sys.stderr, flush=True)
    print(json.dumps(bericht, ensure_ascii=False))


if __name__ == "__main__":
    main()
