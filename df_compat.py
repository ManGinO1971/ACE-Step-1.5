"""
Kompatibilitaets-Bruecke fuer DeepFilterNet (NEU, 10. Okt 2026).

Problem (auf RunPod nachgestellt mit torch/torchaudio 2.10.0, exakt wie in
pyproject.toml gepinnt): DeepFilterNet 0.5.6 importiert in df/io.py
    from torchaudio.backend.common import AudioMetaData
Dieses Modul gibt es in neueren torchaudio-Versionen nicht mehr ->
"ModuleNotFoundError: No module named 'torchaudio.backend'" schon beim
Import von df.enhance. Folge: stem_clean_step.py (Hall-Messung fuer den
Studio-Reverb, Schritt 7) und denoise_step.py (Entrauschen nach RVC,
Schritt 6) sind auf RunPod IMMER gescheitert - der Reverb-Schalter hatte
deshalb keine Wirkung.

Zusaetzlich nutzt df.io zum Laden/Speichern torchaudio.load/save, die ab
torchaudio 2.9 ein weiteres Paket (torchcodec + passende ffmpeg-Bibliotheken)
brauchen. Deshalb laedt/speichert enhance_array() unten selbst (soundfile +
librosa) und nutzt von DeepFilterNet nur das eigentliche Modell.

Benutzung: VOR jedem "from df..." einmal
    import df_compat; df_compat.install_torchaudio_shim()
"""
import sys
import types


def install_torchaudio_shim():
    """Legt ein leeres Ersatz-Modul torchaudio.backend.common mit der Klasse
    AudioMetaData an, falls es fehlt. DeepFilterNet nutzt die Klasse nur als
    Typ-Hinweis - sie wird von unserem Code nie aufgerufen."""
    try:
        import torchaudio.backend.common  # noqa: F401 - existiert in alten Versionen
        return
    except Exception:
        pass
    import torchaudio  # noqa: F401 - muss existieren, sonst ist das ein anderes Problem

    class AudioMetaData:  # minimaler Platzhalter
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0, bits_per_sample=0, encoding=""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    backend = sys.modules.get("torchaudio.backend") or types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


_MODEL = None


def get_df_model():
    """DeepFilterNet einmal pro Prozess laden (model, df_state, df_sr)."""
    global _MODEL
    if _MODEL is None:
        install_torchaudio_shim()
        from df.enhance import init_df
        model, df_state, _ = init_df()
        _MODEL = (model, df_state, df_state.sr())
    return _MODEL


def enhance_array(mono, sr):
    """Entrauscht ein 1-D-Signal (beliebige Samplerate) mit DeepFilterNet und
    gibt es bei DeepFilterNet-Samplerate (48 kHz) zurueck: (signal, df_sr).
    Gleiche Kernfunktion wie bisher (df.enhance.enhance) - nur Laden/
    Resampling laufen ueber librosa statt ueber torchaudio.load."""
    import numpy as np
    import torch
    import librosa

    install_torchaudio_shim()
    from df.enhance import enhance

    model, df_state, df_sr = get_df_model()
    x = np.asarray(mono, dtype=np.float32)
    if sr != df_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=df_sr).astype(np.float32)
    audio = torch.from_numpy(x).unsqueeze(0)
    out = enhance(model, df_state, audio)
    out = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
    return out.reshape(-1).astype(np.float64), df_sr
