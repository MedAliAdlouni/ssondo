# S-SONDO Evaluation Notebooks

Evaluates S-SONDO embedding quality on [ESC-50](https://github.com/karolpiczak/ESC-50) (2000 environmental sounds, 50 classes). Everything auto-downloads — no manual setup needed.

## Quick Start

```bash
cd notebooks
uv sync
uv run python ssondo_clustering_evaluation.py   # clustering analysis
uv run python ssondo_linear_probe_esc50.py       # linear probe accuracy
```

## Notebooks

### 1. Clustering Evaluation (`ssondo_clustering_evaluation.py`)

Unsupervised evaluation: extracts embeddings, clusters with KMeans, and measures how well clusters align with ground-truth classes.

- t-SNE and UMAP visualizations
- Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- NMI, ARI, Cluster Purity
- Hungarian-matched classification accuracy

### 2. Linear Probe (`ssondo_linear_probe_esc50.py`)

Supervised evaluation: freezes the S-SONDO backbone, trains a linear classifier on top, and reports 5-fold cross-validation accuracy on ESC-50. This is the standard protocol for evaluating audio embedding models.

- Frozen backbone + linear head (only ~48K trainable params)
- 5-fold CV following ESC-50's predefined folds
- Reports per-fold and mean accuracy

## Outputs

Saved to `outputs/`:
- `tsne_by_class.png` — t-SNE colored by 50 ESC-50 classes
- `tsne_by_major_category.png` — t-SNE colored by 5 major categories
- `umap_by_class.png` — UMAP colored by class
- `purity_distribution.png` — histogram of cluster purity
- `confusion_matrix.png` — Hungarian-mapped cluster predictions
