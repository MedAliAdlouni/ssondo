# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

S-SONDO (Self-Supervised Knowledge Distillation for General Audio Foundation Models) is a four-step pipeline that distills large audio teacher models (MATPAC, M2D) into lightweight students (MobileNetV3, ERes2Net, DyMN) using only output embeddings. ICASSP 2026. Built with PyTorch + PyTorch Lightning, managed by `uv`.

## Setup & Run

```bash
./setup.sh          # install deps, download metadata + model checkpoints
./run_pipeline.sh   # end-to-end demo (download, extract, cluster, train)

# Or run steps individually:
uv run python -m training_ssondo.download_subset_of_audioset.download_audioset \
    --metadata-csv data/AudioSet/eval_segments.csv --subset-name eval --n-clips 1000
uv run -m training_ssondo.extract_teachers_knowledge.audioset_feature_extraction --conf_id matpac_mcl_eval
./cluster_teachers_embeddings/run_clustering.sh 50_clusters_fit_matpac
uv run -m training_ssondo.knowledge_distillation_training.main --conf_id matpac_mn_cosine_50c
```

**Linting:** `uv run ruff check . --exclude '*.ipynb'` / `uv run ruff format . --exclude '*.ipynb'`

9 pre-existing lint warnings in `utils/student_models/` (third-party EfficientAT code) are intentionally left unfixed.

## Environment Variables

All optional -- defaults defined in `training_ssondo/__init__.py`:

- `DATA` -- defaults to `training_ssondo/data/`
- `OUTPUTS` -- defaults to `training_ssondo/outputs/`
- `SLURM_JOB_ID`, `SLURM_GPUS_ON_NODE`, `SLURM_NNODES` -- auto-detected for HPC

## Architecture

Package root doubles as the `training_ssondo` package (`pyproject.toml` mapping: `"training_ssondo" = "."`).

### Pipeline Steps (sequential)

1. **`download_subset_of_audioset/`** -- Parallel YouTube downloads via `yt-dlp` + `ThreadPoolExecutor`. Output: WAV at 16kHz mono.

2. **`extract_teachers_knowledge/`** -- Runs audio through teacher models, saves embeddings as `.npz`. Feature save logic dispatched via model-category sets (`LOGIT_EMBED_MODELS`, `EMBED_ONLY_MODELS`, `FINETUNED_MODELS`) in `audioset_feature_extraction.py`.

3. **`cluster_teachers_embeddings/`** -- MiniBatchKMeans clustering: train, predict labels, evaluate (silhouette/CH/DB). Config uses factory function `_make_cluster_conf()`. Convenience script: `run_clustering.sh`.

4. **`knowledge_distillation_training/`** -- PyTorch Lightning training:
   - `main.py` -- orchestrator
   - `system.py` -- `KnowledgeDistillationSystem` (training/val steps, mixup, loss)
   - `data_pipeline.py` -- preprocessing + dataloaders. Only loads cluster labels when sampler is `WeightedRandomSamplerSSL`.
   - `model.py` -- student backbone + classification head via `ModelWrapper`
   - `training_components.py` -- optimizer, scheduler, losses, Trainer
   - `config.py` -- all experiment configs keyed by `--conf_id`

### Scripts

- `setup.sh` -- one-command setup (deps, metadata, models). Patches matpac dataclass bug for Python 3.12.
- `run_pipeline.sh` -- end-to-end demo on small subset (64 clips, 50 clusters, 1 epoch)
- `scripts/generate_metadata.py` -- builds `metadata.csv` from AudioSet segment CSVs + ontology

### Shared (`utils/`)

- `audioset_loader.py` -- `AudioSet` metadata class (`pdf` = DataFrame, legacy convention)
- `preprocess.py` -- `LogMelSpectrogram` and `SliceAudio` transforms (nn.Module)
- `portable_m2d.py` -- M2D inference runtime
- `student_models/` -- MobileNetV3, ERes2Net, DyMN architectures + classification heads (Linear, MLP, RNN, AttentionRNN) in `model_utils.py`

## Key Design Patterns

- **Configuration-driven**: `config.py` with dict configs selected by `--conf_id`. No hardcoded paths in logic.
- **Model composition**: `ModelWrapper(backbone, classification_head)` -> `(logits, embeddings)`.
- **Loss combination**: `loss = lambda * pred_loss + (1 - lambda) * kd_loss`. When `lambda=0`, pure distillation.
- **Sampler strategies**: `RandomSampler`, `WeightedRandomSampler`, `WeightedRandomSamplerSSL` (cluster-aware).

## Naming Conventions

- `conf` -- configuration dictionary
- `pdf` -- pandas DataFrame (not PDF files)
- `sr` -- sample rate
- `emb`/`embed` -- embeddings
- `kd` -- knowledge distillation
