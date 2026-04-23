#!/usr/bin/env bash
# SSONDO Training Pipeline - End-to-end demo
# Runs all 4 pipeline steps on a small subset to verify everything works.
# Usage: cd training_ssondo && ./run_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "  SSONDO Pipeline - End-to-End Demo"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Step 1: Download a small subset of AudioSet
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Downloading AudioSet eval clips..."
uv run python -m training_ssondo.download_subset_of_audioset.download_audioset \
    --metadata-csv data/AudioSet/eval_segments.csv \
    --subset-name eval \
    --n-clips 64 \
    --max-workers 5

# ---------------------------------------------------------------------------
# Step 2: Extract teacher knowledge
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Extracting teacher embeddings (MATPAC)..."
uv run -m training_ssondo.extract_teachers_knowledge.audioset_feature_extraction \
    --conf_id matpac_mcl_eval

# ---------------------------------------------------------------------------
# Step 3: Cluster teacher embeddings
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Clustering teacher embeddings (10 clusters)..."
uv run -m training_ssondo.cluster_teachers_embeddings.learn_kmeans \
    --conf_id 10_clusters_fit_matpac
uv run -m training_ssondo.cluster_teachers_embeddings.label_prediction \
    --conf_id 10_clusters_fit_matpac
uv run -m training_ssondo.cluster_teachers_embeddings.evaluate_clustering \
    --conf_id 10_clusters_fit_matpac

# ---------------------------------------------------------------------------
# Step 4: Train student model (1 epoch)
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Training student model (MobileNetV3, 1 epoch)..."
uv run -m training_ssondo.knowledge_distillation_training.main \
    --conf_id matpac_mn_cosine_random

echo ""
echo "========================================================================"
echo "  Pipeline demo complete!"
echo "========================================================================"
