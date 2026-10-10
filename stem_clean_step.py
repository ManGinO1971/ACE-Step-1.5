#!/usr/bin/env python3
"""
Schritt 1+2 der ACE-Step -> RVC Pipeline: Stimmtrennung + Bereinigung.
Läuft in der acestep_stems Umgebung (Demucs + DeepFilterNet).

Instrumente UND Restspur bleiben in Stereo (Original-Breite wie bei
der LoRA-Erzeugung). Nur der Gesang wird fuer die Bereinigung und
spaetere RVC-Umwandlung zusaetzlich als Mono-Version erzeugt (RVC und
DeepFilterNet arbeiten mono).

Aufruf:
  python stem_clean_step.py <input_song.wav> <output_dir>

Erzeugt in <output_dir>:
  vocals_clean.wav       - bereinigte Gesangsspur, MONO, bereit fuer RVC
  instrumental_full.wav  - Instrumente, STEREO, unveraendert/Original
  residue.wav            - beim Bereinigen entferntes Material, STEREO
"""
import sys
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf


def separate_stems(input_path, work_dir):
    """Nutzt Demucs (htdemucs), liefert vocals und instrumental, beide STEREO."""
    cmd = [sys.executable, "-m", "demucs", "--name", "htdemucs", "--out", work_dir, input_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0 and "Separated tracks" not in result.stderr:
        raise RuntimeError(f"Demucs-Trennung fehlgeschlagen: {result.stderr[-500:]}")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    stem_dir = os.path.join(work_dir, "htdemucs", base_name)

    vocals_path = os.path.join(stem_dir, "vocals.wav")
    other_stems = ["drums.wav", "bass.wav", "other.wav"]

    vocals, sr = sf.read(vocals_path)  # bleibt Stereo (shape: samples x 2)
    instrumental = None
    for stem_file in other_stems:
        path = os.path.join(stem_dir, stem_file)
        audio, _ = sf.read(path)
        if instrumental is None:
            instrumental = audio.astype(np.float64)
        else:
            instrumental = instrumental + audio.astype(np.float64)

    return vocals.astype(np.float64), instrumental, sr


def to_mono(audio):
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def clean_vocals_deepfilter_stereo(vocals_stereo, sr):
    """Bereinigt Gesang pro Kanal einzeln, damit Ergebnis + Restspur
    Stereo bleiben. Gibt (vocals_clean_stereo, residue_stereo) zurueck,
    beide bei Original-Samplerate."""
    # NEU (10. Okt 2026): ueber df_compat statt df.io - df.io bricht mit
    # torchaudio 2.10 (RunPod) schon beim Import ab. Genau deshalb ist
    # dieses Skript auf RunPod bisher IMMER gescheitert ("ok=False" in den
    # Logs) und der Studio-Reverb-Schalter (Schritt 7) hatte keine Wirkung.
    # Gleiche DeepFilterNet-Bereinigung, siehe df_compat.py.
    import librosa
    from df_compat import enhance_array

    if vocals_stereo.ndim == 1:
        channels = [vocals_stereo]
    else:
        channels = [vocals_stereo[:, i] for i in range(vocals_stereo.shape[1])]

    cleaned_channels = []
    residue_channels = []

    for ch in channels:
        cleaned_at_dfsr, df_sr = enhance_array(ch, sr)

        ch_at_dfsr = librosa.resample(ch, orig_sr=sr, target_sr=df_sr) if sr != df_sr else ch
        min_len = min(len(ch_at_dfsr), len(cleaned_at_dfsr))
        residue_at_dfsr = ch_at_dfsr[:min_len] - cleaned_at_dfsr[:min_len]

        if df_sr != sr:
            cleaned = librosa.resample(cleaned_at_dfsr, orig_sr=df_sr, target_sr=sr)
            residue = librosa.resample(residue_at_dfsr, orig_sr=df_sr, target_sr=sr)
        else:
            cleaned = cleaned_at_dfsr
            residue = residue_at_dfsr

        cleaned_channels.append(cleaned)
        residue_channels.append(residue)

    min_len_all = min(len(c) for c in cleaned_channels)
    cleaned_channels = [c[:min_len_all] for c in cleaned_channels]
    residue_channels = [r[:min_len_all] for r in residue_channels]

    if len(cleaned_channels) == 1:
        return cleaned_channels[0], residue_channels[0]
    return np.stack(cleaned_channels, axis=1), np.stack(residue_channels, axis=1)


def match_length_multi(*arrays):
    """Bringt mehrere Arrays (Stereo oder Mono) auf gleiche Laenge."""
    min_len = min(len(a) for a in arrays)
    return tuple(a[:min_len] for a in arrays)


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python stem_clean_step.py <input.wav> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    print("Trenne Stems (Demucs)...")
    with tempfile.TemporaryDirectory() as work_dir:
        vocals, instrumental, sr = separate_stems(input_path, work_dir)

    vocals, instrumental = match_length_multi(vocals, instrumental)

    print("Bereinige Gesangsspur (DeepFilter, Stereo)...")
    vocals_clean_stereo, residue_stereo = clean_vocals_deepfilter_stereo(vocals, sr)
    vocals_clean_stereo, residue_stereo, vocals, instrumental = match_length_multi(
        vocals_clean_stereo, residue_stereo, vocals, instrumental
    )

    # Nur fuer RVC: Gesang auf Mono herunterrechnen
    vocals_clean_mono = to_mono(vocals_clean_stereo)

    vocals_out = os.path.join(output_dir, "vocals_clean.wav")
    instrumental_out = os.path.join(output_dir, "instrumental_full.wav")
    residue_out = os.path.join(output_dir, "residue.wav")
    sf.write(vocals_out, vocals_clean_mono, sr)
    sf.write(instrumental_out, instrumental, sr)  # bleibt Stereo
    sf.write(residue_out, residue_stereo, sr)      # bleibt Stereo

    print(f"OK: {vocals_out}")
    print(f"OK: {instrumental_out}")
    print(f"OK: {residue_out}")


if __name__ == "__main__":
    main()
