# SSONDO Training Pipeline

Self-Supervised Knowledge distillation framework for training efficient general audio representation models from large teacher models.

## Overview

The pipeline implements a four-step knowledge distillation workflow:

![SSONDO Training Pipeline](./assets/ssondo_training_pipeline.png)

1. **Download AudioSet** - Downloads audio clips from YouTube
2. **Extract Teacher Knowledge** - Extracts embeddings from teacher models (MATPAC, M2D)
3. **Cluster Embeddings** - Clusters teacher embeddings for structured knowledge representation
4. **Train Student Models** - Trains lightweight models (MobileNetV3, ERes2Net, DyMN) to match teacher embeddings

## Quick Start

```bash
cd training_ssondo
./setup.sh
```

This single script installs dependencies, downloads AudioSet metadata, teacher/student model checkpoints, and generates the metadata index. After setup, proceed to [Pipeline Workflow](#pipeline-workflow).

## Installation

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager (installed automatically by `setup.sh` if missing)

### Automated Setup (Recommended)

```bash
./setup.sh
```

The script performs these steps:
1. Checks/installs `uv`
2. Installs Python dependencies (`uv sync`)
3. Downloads AudioSet metadata CSVs from Google (~100MB)
4. Generates the unified `metadata.csv` index
5. Downloads teacher model checkpoints (~652MB)
6. Downloads student model checkpoints (~60MB)

The script is idempotent — running it again skips already-downloaded files.

### Environment Variables (Optional)

By default, `DATA` points to `training_ssondo/data/` and `OUTPUTS` points to `training_ssondo/outputs/`. Override them if your data lives elsewhere:

```bash
export DATA=/path/to/your/data
export OUTPUTS=/path/to/your/outputs
```

### Manual Setup

If you prefer to set up manually instead of using `setup.sh`:

1. **Install dependencies:**
```bash
pip install uv
uv sync
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
   - [MATPAC](https://github.com/aurianworld/matpac) — we use the `matpac_plus_6s_2048_enconly.pt` checkpoint → `MATPAC_MCL/matpac_plus_6s_2048_enconly.pt`
   - [M2D](https://github.com/nttcslab/m2d) — we use the `m2d_vit_base-80x608p16x16-221006-mr7` checkpoint → extract to `M2D/`

5. **Download student models** into `models/students/`:
   - [MobileNetV3](https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_im.pt) → `MobileNetV3/pretrained_models/mn10_im.pt`
   - [DyMN](https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/dymn10_im.pt) → `DyMN/dymn10_im.pt`
   - ERes2Net — no pretrained weights (random initialization)

## Data Format

- **Audio**: Mono, 16kHz WAV files
- **Spectrograms**: Log-mel, 128 bands, 0-8000 Hz, 25ms window, 10ms hop
- **Directory structure:**
```
data/AudioSet/
├── balanced_train/
├── eval/
├── unbalanced_train/
├── balanced_train_segments.csv
├── eval_segments.csv
├── unbalanced_train_segments.csv
└── ontology.json
```

## Pipeline Workflow

### Step 1: Download AudioSet Subset

```bash
uv run python -m training_ssondo.download_subset_of_audioset.download_audioset \
    --metadata-csv data/AudioSet/eval_segments.csv \
    --n-clips 1000 \
    --subset-name eval \
    --random-state 42 \
    --max-workers 5
```

**Output:** `data/AudioSet/{subset_name}/`

**See:** [download_subset_of_audioset/README.md](download_subset_of_audioset/README.md)

### Step 2: Extract Teacher Knowledge

```bash
uv run -m training_ssondo.extract_teachers_knowledge.audioset_feature_extraction --conf_id matpac_mcl_eval
```

**Configuration:** Edit `extract_teachers_knowledge/config.py`

**Output:** `data/teachers_knowledge/{teacher_model}/{window_length}/{output_type}/{subset}/{filename}.npz`

**See:** [extract_teachers_knowledge/README.md](extract_teachers_knowledge/README.md)

### Step 3: Cluster Teacher Embeddings

1. **Train clustering model:**
```bash
uv run -m training_ssondo.cluster_teachers_embeddings.learn_kmeans --conf_id 50_clusters_fit_matpac
```

2. **Predict cluster labels:**
```bash
uv run -m training_ssondo.cluster_teachers_embeddings.label_prediction --conf_id 50_clusters_fit_matpac
```

3. **Evaluate clustering:**
```bash
uv run -m training_ssondo.cluster_teachers_embeddings.evaluate_clustering --conf_id 50_clusters_fit_matpac
```

**Configuration:** Edit `cluster_teachers_embeddings/config.py`

**Output:** `outputs/clustering/{teacher_model}/{n_clusters}_clusters/`

**See:** [cluster_teachers_embeddings/README.md](cluster_teachers_embeddings/README.md)

### Step 4: Knowledge Distillation Training

```bash
uv run -m training_ssondo.knowledge_distillation_training.main --conf_id matpac_mn_cosine_50c
```

**Configuration:** Edit `knowledge_distillation_training/config.py`

**Key parameters:**
- `teacher_knowledge_path`: Path to teacher embeddings
- `cluster_labels_path`: Path to cluster assignments (optional, only for cluster-aware sampling)
- `classification_head.n_classes`: Must match teacher embedding dimension
- `knowledge_distillation.loss`: Loss type (MSE, L1, cosine_similarity, contrastive_loss, etc.)
- `knowledge_distillation.lambda`: Weight between prediction and distillation loss

**Output:** `outputs/knowledge_distillation/{teacher_model}/{student_model}/{conf_id}/{job_id}/`

**See:** [knowledge_distillation_training/README.md](knowledge_distillation_training/README.md)

## Configuration

### Student Models
- **MobileNetV3**: Lightweight CNN architecture
- **ERes2Net**: Efficient Res2Net variant
- **DyMN**: Dynamic MobileNet with adaptive computation

### Teacher Models
- **MATPAC_MCL**: Multi-level contrastive learning model
- **M2D**: Masked Modeling Duo

### Loss Functions
- MSE, L1, Cosine Similarity
- Contrastive Loss (vanilla, cluster-aware, hybrid)
- KL Divergence (with temperature scaling)

### Data Augmentation
- Mixup
- SpecAugment (time/frequency masking)
- Normalization

## Directory Structure

```
training_ssondo/
├── readme.md
├── pyproject.toml
├── __init__.py                      # DATA and OUTPUTS defaults
├── data/
│   ├── AudioSet/
│   └── teachers_knowledge/
├── outputs/
│   ├── clustering/
│   └── knowledge_distillation/
├── models/
│   ├── teachers/
│   └── students/
├── download_subset_of_audioset/
├── extract_teachers_knowledge/
├── cluster_teachers_embeddings/
├── knowledge_distillation_training/
└── utils/
    ├── audioset_loader.py
    ├── preprocess.py
    ├── portable_m2d.py
    └── student_models/
```

## Environment Variables

All environment variables are optional. Defaults are relative to the `training_ssondo/` directory.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA` | `training_ssondo/data/` | Root data directory |
| `OUTPUTS` | `training_ssondo/outputs/` | Output directory |
| `SLURM_JOB_ID` | random 8-char string | Job identification |
| `SLURM_GPUS_ON_NODE` | `1` | Number of GPUs |
| `SLURM_NNODES` | `1` | Number of nodes |
