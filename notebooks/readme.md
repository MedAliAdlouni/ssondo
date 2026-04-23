# S-SONDO Evaluation Notebooks

Evaluates S-SONDO embedding quality using the [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sound dataset (2000 samples, 50 classes).

## Quick Start

```bash
cd notebooks
pip install -r <(python -c "import tomllib; f=open('pyproject.toml','rb'); print('\n'.join(tomllib.load(f)['project']['dependencies']))")
python ssondo_clustering_evaluation.py
```

Or with `uv`:

```bash
cd notebooks
uv sync
uv run python ssondo_clustering_evaluation.py
```

The script auto-downloads the S-SONDO model from PyPI/HF Hub and the ESC-50 dataset from HuggingFace — no manual setup needed.

## What It Does

1. Loads the `matpac-mobilenetv3` model via `pip install ssondo`
2. Downloads ESC-50 (2000 environmental sounds, 50 classes)
3. Extracts 960-dim embeddings for all samples
4. Visualizes embeddings with t-SNE and UMAP
5. Runs KMeans clustering (k=50 and k=5)
6. Evaluates: Silhouette, Calinski-Harabasz, Davies-Bouldin, NMI, ARI, Cluster Purity

## Outputs

Saved to `outputs/`:
- `tsne_by_class.png` — t-SNE colored by 50 ESC-50 classes
- `tsne_by_major_category.png` — t-SNE colored by 5 major categories
- `umap_by_class.png` — UMAP colored by class
- `purity_distribution.png` — histogram of cluster purity
