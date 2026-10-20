#!/usr/bin/env python3
"""
Lead Vocals v2 (NEU, 20. Okt 2026): "Vocoder C" – baut die Stimme mit EINER
Tonhöhe neu auf (ersetzt WORLD in der neuen Lead-Kette).

Per Hörtest am 20.10. (Root Nah Bow, T7/v2) gewählt: klingt natürlicher als
WORLD und drückt die "mehreren parallelen Töne" der KI-Stimme auf eine Stimme.

Ablauf:
  1. Melodielinie (f0, 10 ms, aus f0_rmvpe_step.py) mit DENSELBEN Reparaturen
     wie world_resynth_step.py glätten (Oktave, Geschwindigkeit, Aussetzer).
  2. Nur kurze Lücken (bis NSF_GAP_MS, Standard 200 ms – Konsonant mitten im
     Wort) füllen; echte Pausen und leise Stellen (NSF_QUIET_DB, Standard
     −45 dB unter dem lautesten Frame) bleiben stimmlos -> kein Nachklingen
     zwischen Wörtern (Hörtest 20.10.: "wwwooon" bis zum nächsten Wort).
  3. PC-NSF-HiFiGAN (OpenVPI, 44,1 kHz, 128 Mel) erzeugt die Stimme aus
     Mel-Spektrum + Melodie. Gerechnet in Stücken, deren Schnittstellen in
     Pausen liegen (keine hörbare Naht), mit 10 ms Überblendung.

LIZENZ-HINWEIS: Die Vocoder-GEWICHTE stehen unter CC BY-NC-SA 4.0 (nicht
kommerziell). Vor dem Verkauf an zahlende Kunden klären/ersetzen. Der Code
(hieltech_nsf/) ist MIT.

Aufruf: python nsf_vocoder_step.py <eingang.wav> <f0.npy> <ausgang.wav>
Gewichte: HIELTECH_NSF_DIR (Standard /opt/hieltech_nsf_vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_DIR = "/opt/hieltech_nsf_vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"


def _env(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def repair_f0(f0):
    """Dieselben drei Reparaturen wie world_resynth_step.py (live bestätigt)."""
    import world_resynth_step as w

    t = np.arange(f0.size) * 0.01
    f0, n1 = w.octave_correct_f0(f0)
    f0, n2 = w.limit_f0_slew_rate(f0, t, max_semitones_per_sec=500.0)
    f0, n3 = w.repair_f0_dropouts(f0, t, baseline_ms=120.0, dev_semitones=2.0, max_run_ms=80.0)
    print(f"Melodie-Reparatur: Oktave {n1}, Geschwindigkeit {n2}, Aussetzer {n3} Frames", file=sys.stderr, flush=True)
    return f0


def main():
    if len(sys.argv) < 4:
        print("Nutzung: python nsf_vocoder_step.py <eingang.wav> <f0.npy> <ausgang.wav>")
        return 1
    src, f0_path, dst = sys.argv[1:4]

    import librosa
    import soundfile as sf
    import torch

    from hieltech_nsf.models import AttrDict, Generator
    from hieltech_nsf.wav2mel import PitchAdjustableMelSpectrogram

    voc_dir = os.environ.get("HIELTECH_NSF_DIR") or DEFAULT_DIR
    if not os.path.isfile(os.path.join(voc_dir, "model.ckpt")):
        print(f"FEHLER: Vocoder-Gewichte fehlen in {voc_dir}", file=sys.stderr)
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(voc_dir, "config.json")))
    cfg.setdefault("noise_sigma", None)
    h = AttrDict(cfg)
    SR, HOP = h.sampling_rate, h.hop_size
    gen = Generator(h)
    ckpt = torch.load(os.path.join(voc_dir, "model.ckpt"), map_location="cpu", weights_only=False)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()
    gen.remove_weight_norm()
    gen.to(device)

    y_orig, sr_orig = sf.read(src, always_2d=True)
    y = librosa.resample(y_orig.mean(axis=1).astype(np.float32), orig_sr=sr_orig, target_sr=SR)
    y = np.clip(y, -1.0, 1.0)
    melx = PitchAdjustableMelSpectrogram(sample_rate=SR, n_fft=h.n_fft, win_length=h.win_size,
                                         hop_length=HOP, f_min=h.fmin, f_max=h.fmax, n_mels=h.num_mels)
    with torch.no_grad():
        mel = torch.log(torch.clamp(melx(torch.from_numpy(y).unsqueeze(0)), min=1e-5))
    n_frames = mel.shape[-1]

    f0 = repair_f0(np.load(f0_path).astype(np.float64))
    # NEU (20. Okt 2026, Test "hohe Töne"): optional Oktav-Lock gegen die
    # Melodie der ORIGINAL-Lead-Spur (4. Argument, .npy). Springt die neue
    # Stimme um ~1-2 Oktaven weg vom Original (z.B. kurz C4 statt C5 mitten im
    # gehaltenen Ton), wird sie auf die Oktave des Originals zurückgesetzt.
    if len(sys.argv) > 4 and os.path.isfile(sys.argv[4]):
        ref = np.load(sys.argv[4]).astype(np.float64)
        n = min(len(ref), len(f0))
        both = (f0[:n] > 0) & (ref[:n] > 0)
        d = np.zeros(n)
        d[both] = 12.0 * np.log2(f0[:n][both] / ref[:n][both])
        k = np.round(d / 12.0)
        fix = both & (k != 0) & (np.abs(d - 12.0 * k) < _env("NSF_LOCK_TOL_ST", 1.5)) & (np.abs(k) <= 2)
        f0[:n][fix] = f0[:n][fix] / np.power(2.0, k[fix])
        print(f"Melodie-Lock (Original-Lead): {int(fix.sum())} Frames auf die Oktave des Originals gesetzt", file=sys.stderr, flush=True)
    voiced = f0 > 0
    if voiced.sum() < 2:
        print("FEHLER: zu wenig Melodie erkannt", file=sys.stderr)
        return 3
    gap = int(_env("NSF_GAP_MS", 200) / 10)
    filled = f0.copy()
    idx = np.flatnonzero(voiced)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            la, lb = np.log(f0[a]), np.log(f0[b])
            filled[a + 1:b] = np.exp(la + (lb - la) * (np.arange(a + 1, b) - a) / (b - a))
    t_src = np.arange(len(f0)) * 0.01
    t_dst = np.arange(n_frames) * HOP / SR
    vm = filled > 0
    lf0 = np.where(vm, np.log(np.maximum(filled, 1e-6)), 0.0)
    f0_dst = np.exp(np.interp(t_dst, t_src, lf0))
    f0_dst = np.where(np.interp(t_dst, t_src, vm.astype(float)) > 0.5, f0_dst, 0.0)
    frame_db = 20 * np.log10(np.maximum(np.exp(mel[0].numpy()).sum(axis=0), 1e-9))
    f0_dst[frame_db < frame_db.max() + _env("NSF_QUIET_DB", -45)] = 0.0

    started = time.time()
    chunk = int(_env("NSF_CHUNK_FRAMES", 800))
    search, ctx = chunk // 4, 48
    cuts = [0]
    while cuts[-1] + chunk < n_frames:
        lo, hi = cuts[-1] + chunk - search, min(n_frames - 1, cuts[-1] + chunk + search)
        cand = np.arange(lo, hi)
        cost = frame_db[cand] + np.where(f0_dst[cand] > 0, 200.0, 0.0)
        cuts.append(int(cand[np.argmin(cost)]))
    cuts.append(n_frames)
    xf = int(0.01 * SR)
    fix = chunk + search + 2 * ctx
    out = np.zeros(n_frames * HOP + xf, dtype=np.float32)
    f0_t = torch.from_numpy(f0_dst).float().unsqueeze(0)
    with torch.no_grad():
        for s, e in zip(cuts[:-1], cuts[1:]):
            a, b = max(0, s - ctx), min(n_frames, e + ctx)
            m_in, f_in = mel[:, :, a:b], f0_t[:, a:b]
            if b - a < fix:
                m_in = torch.nn.functional.pad(m_in, (0, fix - (b - a)), mode="replicate")
                f_in = torch.nn.functional.pad(f_in, (0, fix - (b - a)))
            wav = gen(m_in.to(device), f_in.to(device)).squeeze().float().cpu().numpy()[: (b - a) * HOP]
            seg = wav[(s - a) * HOP:(e - a) * HOP + xf]
            w = np.ones(len(seg), dtype=np.float32)
            if s > 0:
                w[:xf] = np.linspace(0, 1, xf)
            if e < n_frames and len(seg) > xf:
                w[-xf:] = np.linspace(1, 0, xf)
            out[s * HOP:s * HOP + len(seg)] += seg * w
    out = out[: n_frames * HOP]
    print(f"Vocoder C: {len(out) / SR:.1f}s Audio in {time.time() - started:.1f}s auf {device}, "
          f"{len(cuts) - 2} Schnittstellen in Pausen, {np.mean(f0_dst > 0) * 100:.0f} % mit Ton", file=sys.stderr, flush=True)
    out = librosa.resample(out, orig_sr=SR, target_sr=sr_orig)
    n = y_orig.shape[0]
    out = np.pad(out, (0, max(0, n - len(out))))[:n]
    sf.write(dst, np.stack([out, out], axis=1), sr_orig, subtype="PCM_24")
    print(f"OK: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
