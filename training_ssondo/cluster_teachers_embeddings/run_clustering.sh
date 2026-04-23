#!/usr/bin/env bash
# Run the full clustering pipeline (train, predict, evaluate) for a given conf_id.
# Usage: ./cluster_teachers_embeddings/run_clustering.sh <conf_id>
# Example: ./cluster_teachers_embeddings/run_clustering.sh 50_clusters_fit_matpac
set -euo pipefail

CONF_ID="${1:?Usage: $0 <conf_id>}"

echo "========================================================================"
echo "  Clustering Pipeline: $CONF_ID"
echo "========================================================================"

echo ""
echo "[1/3] Training clustering model..."
uv run -m training_ssondo.cluster_teachers_embeddings.learn_kmeans --conf_id "$CONF_ID"

echo ""
echo "[2/3] Predicting cluster labels..."
uv run -m training_ssondo.cluster_teachers_embeddings.label_prediction --conf_id "$CONF_ID"

echo ""
echo "[3/3] Evaluating clustering..."
uv run -m training_ssondo.cluster_teachers_embeddings.evaluate_clustering --conf_id "$CONF_ID"

echo ""
echo "========================================================================"
echo "  Clustering pipeline complete for: $CONF_ID"
echo "========================================================================"
