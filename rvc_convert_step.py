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

# NEU (10. Okt 2026, Fehler "RVC übersprungen: 'tuple' object has no
# attribute 'dtype'"): ab torch 2.6 laedt torch.load standardmaessig im
# "weights_only"-Modus. fairseq (laedt hubert_base.pt fuer RVC) ruft torch.load
# OHNE diesen Parameter auf, und hubert_base.pt enthaelt neben den Gewichten
# auch Einstellungs-Objekte -> "Weights only load failed". rvc-python faengt
# das intern ab und gibt statt Audio einen Fehlertext zurueck, was erst beim
# Speichern als 'tuple' ... 'dtype' auffiel. Dieser Schalter stellt fuer
# DIESEN Prozess das alte Ladeverhalten wieder her. Muss VOR dem ersten
# "import torch" gesetzt werden. Geladen werden nur die bekannten RVC-
# Basismodelle und die eigenen .pth-Modelle der jeweiligen Lizenz.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main():
    if len(sys.argv) < 4:
        print("Nutzung: python rvc_convert_step.py <input.wav> <output.wav> <model_name> [pitch_shift]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_name = sys.argv[3]
    pitch_shift = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # NEU (10. Okt 2026): auf RunPod liegen die Stimmmodelle auf dem
    # Netzwerk-Volume (runpod_entrypoint.py setzt HIELTECH_RVC_VOICES_DIR),
    # am Mac ohne diese Variable wie bisher im Projektordner.
    models_dir = os.environ.get("HIELTECH_RVC_VOICES_DIR") or os.path.join(base_dir, "rvc_models", "voices")
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

    # NEU (9. Okt 2026, Nutzerwunsch "soll alles moeglicher auf gpu laufen"):
    # vorher stand hier fest device="cpu" (unveraendert von der Mac-Version
    # uebernommen, wo es ohnehin keine CUDA-GPU gibt) - auf RunPod lief RVC
    # dadurch bisher IMMER auf der CPU, obwohl eine GPU bereitstand und
    # bezahlt wurde. Siehe Dockerfile.runpod fuer den dazugehoerigen Wechsel
    # der RVC-venv von CPU- auf CUDA-Torch.
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    rvc = RVCInference(device=device)
    rvc.load_model(model_path)
    rvc.f0_up_key = pitch_shift
    if index_path:
        rvc.index_path = index_path

    # NEU (10. Okt 2026): rvc-python gibt bei einem internen Fehler statt Audio
    # ein Tupel (Fehlertext, ...) zurueck - dann den ECHTEN Fehler melden statt
    # des irrefuehrenden Folgefehlers beim Speichern.
    _orig_vc_single = rvc.vc.vc_single

    def _checked_vc_single(*args, **kwargs):
        result = _orig_vc_single(*args, **kwargs)
        if isinstance(result, tuple):
            raise RuntimeError("RVC-Umwandlung fehlgeschlagen:\n" + str(result[0])[-1500:])
        return result

    rvc.vc.vc_single = _checked_vc_single
    rvc.infer_file(input_path, output_path)
    print(f"OK: {output_path}")


if __name__ == "__main__":
    main()
