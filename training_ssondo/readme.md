# S-SONDO Training Pipeline

Official training code for **S-SONDO: Self-Supervised Knowledge Distillation for General Audio Foundation Models** (ICASSP 2026).

S-SONDO is the first framework for self-supervised knowledge distillation of general audio foundation models. It distills large teacher models (MATPAC, M2D) into lightweight students (MobileNetV3, ERes2Net, DyMN) that are up to 61x smaller while retaining up to 96% of teacher performance -- using only output embeddings, no logits or layer-level alignment required.

## Pipeline Overview

![S-SONDO Training Pipeline](./assets/ssondo_training_pipeline.png)

| Step | Module | Description |
|------|--------|-------------|
| 1 | `download_subset_of_audioset/` | Download AudioSet audio clips from YouTube |
| 2 | `extract_teachers_knowledge/` | Extract embeddings from teacher models |
| 3 | `cluster_teachers_embeddings/` | Cluster embeddings for balanced data sampling |
| 4 | `knowledge_distillation_training/` | Train student models via knowledge distillation |

## Quick Start

```bash
git clone https://github.com/MedAliAdlouni/ssondo_temp.git
cd ssondo_temp/training_ssondo
./setup.sh          # install deps, download metadata + model checkpoints
./run_pipeline.sh   # run all 4 steps end-to-end on a small subset
```

## Installation

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager (installed automatically by `setup.sh` if missing)

### Automated Setup (Recommended)

```bash
./setup.sh
```

The script is idempotent (re-running skips already-downloaded files) and performs:

1. Checks/installs `uv`
2. Installs Python dependencies (`uv sync --frozen`)
3. Downloads AudioSet metadata CSVs from Google (~100MB)
4. Generates the unified `metadata.csv` index
5. Downloads teacher model checkpoints (~652MB)
6. Downloads student model checkpoints (~60MB)

### Environment Variables (Optional)

All environment variables are optional. Defaults are relative to `training_ssondo/`.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA` | `training_ssondo/data/` | Root data directory |
| `OUTPUTS` | `training_ssondo/outputs/` | Output directory |
| `SLURM_JOB_ID` | random 8-char string | Job identification |
| `SLURM_GPUS_ON_NODE` | `1` | Number of GPUs |
| `SLURM_NNODES` | `1` | Number of nodes |

### Manual Setup

<details>
<summary>Click to expand manual setup instructions</summary>

1. **Install dependencies:**
```bash
pip install uv
uv sync --frozen
```

2. **Download AudioSet metadata** into `data/AudioSet/`:
   - [eval_segments.csv](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv)
   - [balanced_train_segments.csv](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv)
   - [unbalanced_train_segments.csv](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv)
   - [ontology.json](https://raw.githubusercontent.com/audioset/ontology/refs/heads/master/ontology.json)
   - [class_labels_indices.csv](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv)

3. **Generate metadata.csv:**
```bash
uv run python scripts/generate_metadata.py
```

4. **Download teacher models** into `models/teachers/`:
   - [MATPAC](https://github.com/aurianworld/matpac) -- we use `matpac_plus_6s_2048_enconly.pt` -> `MATPAC_MCL/matpac_plus_6s_2048_enconly.pt`
   - [M2D](https://github.com/nttcslab/m2d) -- we use `m2d_vit_base-80x608p16x16-221006-mr7` -> extract to `M2D/`

5. **Download student models** into `models/students/`:
   - [MobileNetV3](https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_im.pt) -> `MobileNetV3/pretrained_models/mn10_im.pt`
   - [DyMN](https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/dymn10_im.pt) -> `DyMN/dymn10_im.pt`
   - ERes2Net -- no pretrained weights (random initialization)

</details>

## Pipeline Workflow

### Step 1: Download AudioSet

```bash
uv run python -m training_ssondo.download_subset_of_audioset.download_audioset \
    --metadata-csv data/AudioSet/eval_segments.csv \
    --subset-name eval --n-clips 1000 --max-workers 5
```

Downloads audio clips from YouTube via `yt-dlp`, extracts segments, converts to mono 16kHz WAV. See [download_subset_of_audioset/README.md](download_subset_of_audioset/README.md).

### Step 2: Extract Teacher Knowledge

```bash
uv run -m training_ssondo.extract_teachers_knowledge.audioset_feature_extraction --conf_id matpac_mcl_eval
```

Runs audio through teacher models and saves embeddings as `.npz` files. Available configs: `matpac_mcl_train`, `matpac_mcl_eval`, `m2d_train`, `m2d_eval`. See [extract_teachers_knowledge/README.md](extract_teachers_knowledge/README.md).

### Step 3: Cluster Teacher Embeddings

```bash
# Run all 3 steps at once
./cluster_teachers_embeddings/run_clustering.sh 50_clusters_fit_matpac
```

Or run each step individually:

```bash
uv run -m training_ssondo.cluster_teachers_embeddings.learn_kmeans --conf_id 50_clusters_fit_matpac
uv run -m training_ssondo.cluster_teachers_embeddings.label_prediction --conf_id 50_clusters_fit_matpac
uv run -m training_ssondo.cluster_teachers_embeddings.evaluate_clustering --conf_id 50_clusters_fit_matpac
```

Trains MiniBatchKMeans on teacher embeddings, predicts cluster labels, and evaluates with silhouette/Calinski-Harabasz/Davies-Bouldin metrics. See [cluster_teachers_embeddings/README.md](cluster_teachers_embeddings/README.md).

### Step 4: Knowledge Distillation Training

```bash
uv run -m training_ssondo.knowledge_distillation_training.main --conf_id matpac_mn_cosine_50c
```

Trains a student model to match teacher embeddings. See [knowledge_distillation_training/README.md](knowledge_distillation_training/README.md).

**Available configurations:**

| `conf_id` | Teacher | Student | Sampler |
|-----------|---------|---------|---------|
| `baseline_mn_weighted_random_sampling` | -- | MobileNetV3 | WeightedRandom |
| `matpac_mn_cosine_random` | MATPAC | MobileNetV3 | Random |
| `matpac_mn_cosine_50c` | MATPAC | MobileNetV3 | ClusterAware (50c) |
| `matpac_eres2net_cosine_50c` | MATPAC | ERes2Net | ClusterAware (50c) |
| `matpac_dymn_cosine_50c` | MATPAC | DyMN | ClusterAware (50c) |
| `m2d_mn_cosine_50c` | M2D | MobileNetV3 | ClusterAware (50c) |
| `m2d_eres2net_cosine_50c` | M2D | ERes2Net | ClusterAware (50c) |
| `m2d_dymn_cosine_50c` | M2D | DyMN | ClusterAware (50c) |

## Architecture

### Teacher Models
- **[MATPAC](https://github.com/aurianworld/matpac)**: Masked Latent Prediction and Classification SSL audio model
 (checkpoint: `matpac_plus_6s_2048_enconly.pt`)
- **[M2D](https://github.com/nttcslab/m2d)**: Masked Modeling Duo self-supervised model (checkpoint: `m2d_vit_base-80x608p16x16-221006-mr7`)

### Student Models
- **[MobileNetV3](https://github.com/fschmid56/EfficientAT)**: Lightweight CNN (2.9M params)
- **ERes2Net**: Efficient Res2Net variant
- **[DyMN](https://github.com/fschmid56/EfficientAT)**: Dynamic MobileNet with adaptive computation

### Loss Functions
- **Standard**: MSE, L1, Cosine Similarity, KL Divergence
- **Contrastive**: Vanilla, Cluster-aware, Negative-clusters-only, Hybrid
- **Combined**: `loss = lambda * pred_loss + (1 - lambda) * kd_loss`

### Data Augmentation
- **Mixup** with configurable alpha
- **SpecAugment** (time/frequency masking)

## Directory Structure

```
training_ssondo/
├── setup.sh                         # One-command setup
├── run_pipeline.sh                  # End-to-end demo
├── pyproject.toml
├── __init__.py                      # DATA and OUTPUTS defaults
├── scripts/
│   └── generate_metadata.py         # Generates metadata.csv from AudioSet CSVs
├── data/
│   ├── AudioSet/                    # Audio files + metadata
│   └── teachers_knowledge/          # Extracted teacher embeddings
├── outputs/
│   ├── clustering/                  # Clustering results
│   └── knowledge_distillation/      # Trained student models
├── models/
│   ├── teachers/                    # Teacher checkpoints (MATPAC, M2D)
│   └── students/                    # Student checkpoints (MobileNetV3, DyMN)
├── download_subset_of_audioset/     # Step 1: Download AudioSet
├── extract_teachers_knowledge/      # Step 2: Extract teacher embeddings
├── cluster_teachers_embeddings/     # Step 3: Cluster embeddings
├── knowledge_distillation_training/ # Step 4: Train student models
└── utils/
    ├── audioset_loader.py           # AudioSet metadata management
    ├── preprocess.py                # LogMelSpectrogram, SliceAudio transforms
    ├── portable_m2d.py              # M2D runtime
    └── student_models/              # MobileNetV3, ERes2Net, DyMN architectures
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{eladlouni2026ssondo,
  title={S-SONDO: Self-Supervised Knowledge Distillation for General Audio Foundation Models},
  author={El Adlouni, Mohammed Ali and Quelennec, Aurian and Chouteau, Pierre and Peeters, Geoffroy and Essid, Slim},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.
