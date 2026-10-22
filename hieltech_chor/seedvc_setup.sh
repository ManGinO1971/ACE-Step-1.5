#!/bin/bash
# Background-Chor (NEU, 23. Okt 2026): Seed-VC EINMAL auf dem RunPod-Netzwerk-Volume
# einrichten (statt im Docker-Image – der RunPod-Build mit Seed-VC im Image brach ab).
# Danach nutzen alle Worker dieselbe Einrichtung. Aufruf: bash seedvc_setup.sh <zielordner>
set -euo pipefail
H="${1:-/runpod-volume/hieltech_seedvc}"
HIER="$(cd "$(dirname "$0")" && pwd)"
COMMIT=51383efd921027683c89e5348211d93ff12ac2a8
mkdir -p "$H"
cd "$H"
echo "Seed-VC-Einrichtung in $H"
if [ ! -d seed-vc/.git ]; then
  rm -rf seed-vc
  git clone -q https://github.com/Plachtaa/seed-vc.git seed-vc
fi
git -C seed-vc checkout -q "$COMMIT"
if [ ! -x venv/bin/python ]; then
  python3.10 -m venv venv
fi
venv/bin/pip install -q --no-cache-dir "pip==24.0"
venv/bin/python -c "import torch" 2>/dev/null || \
  venv/bin/pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1" "torchaudio==2.7.1" || \
  venv/bin/pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio
venv/bin/pip install -q --no-cache-dir \
  accelerate "scipy==1.13.1" "librosa==0.10.2" "huggingface-hub>=0.28.1" \
  "munch==4.0.0" "einops==0.8.0" "descript-audio-codec==1.0.0" "pydub==0.25.1" \
  "transformers==4.46.3" "soundfile==0.12.1" "modelscope==1.18.1" "funasr==1.1.5" \
  "numpy==1.26.4" "hydra-core==1.3.2" pyyaml python-dotenv
cd "$H/seed-vc"
"$H/venv/bin/python" "$HIER/seedvc_hieltech_patch.py"
HIELTECH_SEEDVC_DIR="$H/seed-vc" HIELTECH_PREFETCH_STRICT=1 "$H/venv/bin/python" "$HIER/seedvc_prefetch.py"
touch "$H/.hieltech_ok"
echo "Seed-VC-Einrichtung fertig"
