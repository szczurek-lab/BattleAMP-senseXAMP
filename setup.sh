#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SenseXAMP setup ==="

# Ensure conda's libstdc++ is found before the system's old one.
CONDA_LIB=$(python -c "import sys; print(sys.prefix + '/lib')")
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

# Install pip dependencies, skip torch and mkl packages (provided by conda).
grep -v -E '^(torch==|torchaudio|torchvision|mkl-fft|mkl-service)' requirements.txt > /tmp/sensexamp_requirements.txt
echo "biopython" >> /tmp/sensexamp_requirements.txt
echo "numpy<2" >> /tmp/sensexamp_requirements.txt
pip install -r /tmp/sensexamp_requirements.txt
rm -f /tmp/sensexamp_requirements.txt

# Create dataset directories.
mkdir -p datasets/ori_datasets
mkdir -p datasets/stc_datasets
mkdir -p datasets/esm_embeddings/all
mkdir -p datasets/stc_info

# Download model checkpoints if not already present.
if [ ! -f weights/amp_cls/sensexamp_cls_balanced.ckpt ] || \
   [ ! -f weights/amp_reg/sensexamp_reg_ecoli.ckpt ] || \
   [ ! -f weights/amp_reg/sensexamp_reg_saureus.ckpt ]; then
    echo "Downloading model checkpoints from Google Drive..."
    mkdir -p weights/amp_cls weights/amp_reg
    python -c "
import gdown, os

# Classifier checkpoint
gdown.download_folder(
    url='https://drive.google.com/drive/folders/1JKNH-3gUD2qVkhWiUqXNF1Ef-a4CPzvS',
    output='weights/amp_cls', quiet=False)

# Regression checkpoints (ecoli + saureus in same folder)
gdown.download_folder(
    url='https://drive.google.com/drive/folders/1QWrv0P2mo85AazilU7bzQrf8rWHM0lt-',
    output='weights/amp_reg', quiet=False)
"
fi

# Verify model checkpoints.
MISSING=0
for CKPT in \
    weights/amp_cls/sensexamp_cls_balanced.ckpt \
    weights/amp_reg/sensexamp_reg_ecoli.ckpt \
    weights/amp_reg/sensexamp_reg_saureus.ckpt; do
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Checkpoint not found: $CKPT" >&2
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "Download checkpoints from Google Drive. See weights/README.md" >&2
    exit 1
fi
echo "Model checkpoints: OK"

# Test critical imports.
python -c "
import torch
print(f'PyTorch {torch.__version__} OK')
print(f'CUDA available: {torch.cuda.is_available()}')
import torch.distributed
print('torch.distributed OK')
import tools.esm_project as esm
print('ESM project OK')
from Ampmm_base.runner import Runner
print('SenseXAMP runner OK')
"

echo "=== SenseXAMP setup complete ==="