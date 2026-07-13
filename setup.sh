#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SenseXAMP setup (uv) ==="

# 1. Build the isolated environment from the lockfile.
uv sync

# 2. Download model checkpoints from Google Drive (skipped if already present).
mkdir -p weights/amp_cls weights/amp_reg
if [ ! -f weights/amp_cls/sensexamp_cls_balanced.ckpt ] \
   || [ ! -f weights/amp_reg/sensexamp_reg_ecoli.ckpt ] \
   || [ ! -f weights/amp_reg/sensexamp_reg_saureus.ckpt ]; then
    echo "Downloading model checkpoints from Google Drive..."
    uv run python - <<'PY'
import gdown

# Classifier checkpoints.
gdown.download_folder(
    url="https://drive.google.com/drive/folders/1JKNH-3gUD2qVkhWiUqXNF1Ef-a4CPzvS",
    output="weights/amp_cls", quiet=False)

# Regression checkpoints (E. coli + S. aureus in the same folder).
gdown.download_folder(
    url="https://drive.google.com/drive/folders/1QWrv0P2mo85AazilU7bzQrf8rWHM0lt-",
    output="weights/amp_reg", quiet=False)
PY
fi

# 3. Verify the checkpoints required for inference.
missing=0
for ckpt in \
    weights/amp_cls/sensexamp_cls_balanced.ckpt \
    weights/amp_reg/sensexamp_reg_ecoli.ckpt \
    weights/amp_reg/sensexamp_reg_saureus.ckpt; do
    [ -f "$ckpt" ] || { echo "ERROR: checkpoint not found: $ckpt" >&2; missing=1; }
done
if [ "$missing" -eq 1 ]; then
    echo "See weights/README.md for the Google Drive links." >&2
    exit 1
fi
echo "Checkpoints: OK"

# 4. Sanity-check imports in the uv environment.
uv run python -c "
import torch
print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
import predict
from Ampmm_base.runner import Runner
import tools.esm_project
print('SenseXAMP imports OK')
"

echo "=== Setup complete. ESM-1b (~2.5 GB) downloads automatically on first prediction. ==="
