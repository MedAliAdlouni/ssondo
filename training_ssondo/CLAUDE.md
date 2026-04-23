# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SSONDO is a four-step knowledge distillation pipeline that trains lightweight audio models (students) from large teacher models (MATPAC, M2D) using AudioSet data. Built with PyTorch + PyTorch Lightning, managed by `uv`.

## Build & Run

```bash
# Install dependencies (requires Python 3.12 exact, uv package manager)
uv sync

# All modules are run as: uv run -m training_ssondo.<module> [args]

# Step 1: Download AudioSet audio clips from YouTube
uv run python -m training_ssondo.download_subset_of_audioset.download_audioset \
    --metadata-csv data/AudioSet/eval_segments.csv --subset-name eval --n-clips 1000

# Step 2: Extract teacher embeddings
uv run -m training_ssondo.extract_teachers_knowledge.audioset_feature_extraction --conf_id matpac_mcl_eval

# Step 3: Cluster embeddings (3 sub-steps)
uv run -m training_ssondo.cluster_teachers_embeddings.learn_kmeans --conf_id 50_clusters_fit_matpac
uv run -m training_ssondo.cluster_teachers_embeddings.label_prediction --conf_id 50_clusters_fit_matpac
uv run -m training_ssondo.cluster_teachers_embeddings.evaluate_clustering --conf_id 50_clusters_fit_matpac

# Step 4: Train student model
uv run -m training_ssondo.knowledge_distillation_training.main --conf_id matpac_mn_cosine_50c
```

**Linting:** `uv run ruff check .` / `uv run ruff format .`

## Environment Variables

All optional — defaults are relative to `training_ssondo/`:

- `DATA` - root data directory, defaults to `training_ssondo/data/` (contains `AudioSet/` and `teachers_knowledge/`)
- `OUTPUTS` - output directory, defaults to `training_ssondo/outputs/` (clustering results, trained models)
- SLURM variables (`SLURM_JOB_ID`, `SLURM_GPUS_ON_NODE`, `SLURM_NNODES`) auto-detected for HPC

Defaults are defined in `training_ssondo/__init__.py`.

## Architecture

The repo root doubles as the `training_ssondo` package (see `pyproject.toml` `package-dir` mapping: `"training_ssondo" = "."`). Each pipeline step is a sub-package:

### Pipeline Steps (sequential)

1. **`download_subset_of_audioset/`** - Parallel YouTube downloads via `yt-dlp` + `ThreadPoolExecutor`. Output: WAV files at 16kHz mono.

2. **`extract_teachers_knowledge/`** - Runs audio through teacher models (MATPAC or M2D), saves embeddings as `.npz` files. Teacher models are wrapped in `models_wrappers.py` with a common `ModelWrapper` interface. Feature save logic is dispatched via model-category sets (`LOGIT_EMBED_MODELS`, `EMBED_ONLY_MODELS`, `FINETUNED_MODELS`).

3. **`cluster_teachers_embeddings/`** - Three-stage pipeline: train MiniBatchKMeans on embeddings, predict cluster labels for all samples, evaluate with silhouette/Calinski-Harabasz/Davies-Bouldin metrics. Config uses a factory function `_make_cluster_conf()` to generate all experiment configs.

4. **`knowledge_distillation_training/`** - The main training module. Key files:
   - `main.py` - orchestrates: seed -> data pipeline -> model -> training components -> train
   - `system.py` - `KnowledgeDistillationSystem` (PyTorch Lightning module): handles training/validation steps, mixup augmentation, loss computation
   - `data_pipeline.py` - builds preprocessing chain (SliceAudio -> LogMelSpectrogram -> Normalize), datasets, samplers, dataloaders. Only loads cluster labels when sampler requires them.
   - `model.py` - builds student backbone + classification head via `ModelWrapper`
   - `training_components.py` - sets up optimizer, scheduler, losses, Lightning Trainer
   - `config.py` - all experiment configs keyed by `conf_id`

### Shared (`utils/`)

- `audioset_loader.py` - `AudioSet` class for metadata management (uses `pdf` as DataFrame variable name - legacy convention)
- `preprocess.py` - `LogMelSpectrogram` and `SliceAudio` transforms (nn.Module)
- `student_models/` - three student architectures: `MobileNetV3/`, `ERes2Net/`, `dymn/` (Dynamic MobileNet), plus pluggable classification heads in `model_utils.py` (Linear, MLP, RNN, AttentionRNN)

## Key Design Patterns

- **Configuration-driven**: Every step uses a `config.py` with dict-based configs selected by `--conf_id`. No hardcoded paths or hyperparameters in logic files.
- **Model composition**: Student models are `ModelWrapper(backbone, classification_head)` returning `(logits, embeddings)`.
- **Loss combination**: `loss = lambda * prediction_loss + (1 - lambda) * kd_loss`. KD losses: MSE, L1, cosine similarity, contrastive (vanilla/cluster-aware/hybrid). When `lambda=0`, pure distillation.
- **Sampler strategies**: `RandomSampler`, `WeightedRandomSampler`, `WeightedRandomSamplerSSL` (cluster-aware) - selected per config.

## Naming Conventions

- `conf` - configuration dictionary
- `pdf` - pandas DataFrame (not PDF files)
- `sr` - sample rate
- `emb`/`embed` - embeddings
- `kd` - knowledge distillation
