#!/usr/bin/env python3
"""
Schritt 3 der ACE-Step -> RVC Pipeline: Stimmumwandlung.
Läuft in der acestep_rvc Umgebung (fairseq + rvc-python).

Aufruf:
  python rvc_convert_step.py <input_vocals_clean.wav> <output.wav> <model_name> [pitch_shift]

model_name muss einer .pth-Datei in rvc_models/voices/ entsprechen
(z.B. "Hicham" fuer rvc_models/voices/Hicham.pth).
"""
import sys
import os


def main():
    if len(sys.argv) < 4:
        print("Nutzung: python rvc_convert_step.py <input.wav> <output.wav> <model_name> [pitch_shift]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_name = sys.argv[3]
    pitch_shift = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "rvc_models", "voices")
    model_path = os.path.join(models_dir, f"{model_name}.pth")

    if not os.path.exists(model_path):
        print(f"FEHLER: Modell nicht gefunden: {model_path}")
        sys.exit(1)

    index_path = ""
    for f in os.listdir(models_dir):
        if model_name in f and f.endswith(".index"):
            index_path = os.path.join(models_dir, f)
            break

    hubert_path = os.path.join(base_dir, "rvc_models", "hubert_base.pt")

    from rvc_python.infer import RVCInference

    rvc = RVCInference(device="cpu")
    rvc.load_model(model_path)
    rvc.f0_up_key = pitch_shift
    if index_path:
        rvc.index_path = index_path

    rvc.infer_file(input_path, output_path)
    print(f"OK: {output_path}")


if __name__ == "__main__":
    main()
