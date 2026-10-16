#!/usr/bin/env python3
"""
NEU (16. Okt 2026, "Melodie-Lock" fuer Background Vocals): holt mit RMVPE
(Robust Model for Vocal Pitch Estimation in Polyphonic Music, dasselbe
Modell wie im RVC-Stimmwechsel) EINE Hauptmelodie-Linie aus einer
mehrstimmigen Gesangsspur - auch an leisen Stellen (niedrigere Schwelle als
der RVC-Standard). Laeuft in der RVC-venv (dort liegen rvc_python und
rmvpe.pt).

Aufruf: python f0_rmvpe_step.py <eingang.wav> <ausgang.npy>
Ausgang: numpy-Array F0 in Hz, ein Wert pro 10 ms (0 = stimmlos).
"""
import os
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python f0_rmvpe_step.py <eingang.wav> <ausgang.npy>")
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    import soundfile as sf
    import torch
    import torchaudio

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data, sr = sf.read(src, always_2d=True)
    mono = torch.from_numpy(np.mean(data, axis=1).astype(np.float32))
    if sr != 16000:
        mono = torchaudio.functional.resample(mono, sr, 16000)
    audio = mono.numpy()
    thred = float(os.environ.get("HIELTECH_RMVPE_THRED", "0.02"))
    method = "RMVPE"
    try:
        import rvc_python
        from rvc_python.lib.rmvpe import RMVPE

        model_path = os.path.join(os.path.dirname(rvc_python.__file__), "base_model", "rmvpe.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        model = RMVPE(model_path, is_half=False, device=device)
        with torch.no_grad():
            f0 = model.infer_from_audio(audio, thred=thred)
    except Exception as e:  # noqa: BLE001 - Rueckfall auf torchcrepe (ebenfalls in der RVC-venv)
        print(f"RMVPE nicht nutzbar ({e}) - Rückfall auf torchcrepe", file=sys.stderr)
        import torchcrepe

        method = "torchcrepe"
        x = torch.from_numpy(audio).unsqueeze(0).to(device)
        pitch, periodicity = torchcrepe.predict(
            x, 16000, 160, 50.0, 1100.0, "full", batch_size=512, device=device, return_periodicity=True,
        )
        pitch = pitch.squeeze(0).cpu().numpy()
        periodicity = periodicity.squeeze(0).cpu().numpy()
        f0 = np.where(periodicity > float(os.environ.get("HIELTECH_CREPE_THRED", "0.3")), pitch, 0.0)
    f0 = np.asarray(f0, dtype=np.float64)
    np.save(dst, f0)
    voiced = float(np.mean(f0 > 0)) * 100 if f0.size else 0.0
    print(f"Melodie-Linie ({method}, Schwelle {thred}): {f0.size} Frames, {voiced:.0f} % stimmhaft, Gerät {device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
