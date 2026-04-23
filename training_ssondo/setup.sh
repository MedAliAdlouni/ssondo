#!/usr/bin/env bash
# SSONDO Training Pipeline - One-command setup
# Usage: cd training_ssondo && ./setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/AudioSet"
MODELS_DIR="$SCRIPT_DIR/models"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
download() {
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  Already exists: $(basename "$dest")"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  Downloading $(basename "$dest")..."
    curl -L --fail --progress-bar -o "$dest" "$url"
}

echo "========================================================================"
echo "  SSONDO Training Pipeline Setup"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Step 1: Check uv
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Checking uv package manager..."
if command -v uv &>/dev/null; then
    echo "  uv found: $(uv --version)"
else
    echo "  uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "  uv installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# Step 2: Install Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Installing Python dependencies..."
cd "$SCRIPT_DIR"
uv sync --frozen
echo "  Dependencies installed."

# ---------------------------------------------------------------------------
# Step 3: Download AudioSet metadata
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Downloading AudioSet metadata CSVs..."
mkdir -p "$DATA_DIR"

AUDIOSET_CSV_BASE="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv"
download "$AUDIOSET_CSV_BASE/eval_segments.csv" "$DATA_DIR/eval_segments.csv"
download "$AUDIOSET_CSV_BASE/balanced_train_segments.csv" "$DATA_DIR/balanced_train_segments.csv"
download "$AUDIOSET_CSV_BASE/unbalanced_train_segments.csv" "$DATA_DIR/unbalanced_train_segments.csv"

download "https://raw.githubusercontent.com/audioset/ontology/refs/heads/master/ontology.json" \
         "$DATA_DIR/ontology.json"
download "$AUDIOSET_CSV_BASE/class_labels_indices.csv" \
         "$DATA_DIR/class_labels_indices.csv"

echo "  AudioSet metadata ready."

# ---------------------------------------------------------------------------
# Step 4: Generate unified metadata.csv
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Generating metadata.csv..."
if [[ -f "$DATA_DIR/metadata.csv" ]]; then
    echo "  Already exists: metadata.csv"
else
    uv run python "$SCRIPT_DIR/scripts/generate_metadata.py" --data-dir "$DATA_DIR"
fi

# ---------------------------------------------------------------------------
# Step 5: Download teacher model checkpoints
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Downloading teacher model checkpoints..."

# MATPAC
download "https://github.com/aurianworld/matpac/releases/download/MATPAC%2B%2B/matpac_plus_6s_2048_enconly.pt" \
         "$MODELS_DIR/teachers/MATPAC_MCL/matpac_plus_6s_2048_enconly.pt"

# M2D (zip archive — extract and rename checkpoint to M2D_ssl.pth)
M2D_DIR="$MODELS_DIR/teachers/M2D/m2d_vit_base-80x608p16x16-221006-mr7_enconly"
if [[ -f "$M2D_DIR/M2D_ssl.pth" ]]; then
    echo "  Already exists: M2D_ssl.pth"
else
    M2D_ZIP="$MODELS_DIR/teachers/M2D/m2d.zip"
    download "https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-221006-mr7_enconly.zip" \
             "$M2D_ZIP"
    echo "  Extracting M2D checkpoint..."
    mkdir -p "$M2D_DIR"
    unzip -qo "$M2D_ZIP" -d "$MODELS_DIR/teachers/M2D/"
    mv "$M2D_DIR/checkpoint-300.pth" "$M2D_DIR/M2D_ssl.pth"
    rm -f "$M2D_ZIP"
fi

echo "  Teacher models ready."

# ---------------------------------------------------------------------------
# Step 6: Download student model checkpoints
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Downloading student model checkpoints..."

EFFICIENTAT_BASE="https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1"
download "$EFFICIENTAT_BASE/mn10_im.pt" \
         "$MODELS_DIR/students/MobileNetV3/pretrained_models/mn10_im.pt"
download "$EFFICIENTAT_BASE/dymn10_im.pt" \
         "$MODELS_DIR/students/DyMN/dymn10_im.pt"

echo "  Student models ready."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  Setup complete!"
echo ""
echo "  Next step: download AudioSet audio clips (Step 1 of the pipeline):"
echo ""
echo "    uv run python -m training_ssondo.download_subset_of_audioset.download_audioset \\"
echo "        --metadata-csv data/AudioSet/eval_segments.csv \\"
echo "        --subset-name eval --n-clips 512"
echo ""
echo "  See readme.md for the full pipeline workflow."
echo "========================================================================"
