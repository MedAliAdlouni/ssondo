<div align="center">

# S-SONDO

### Self-Supervised Knowledge Distillation for General Audio Foundation Models

**ICASSP 2026**

[![Paper](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org/)
[![PyPI](https://img.shields.io/pypi/v/ssondo.svg)](https://pypi.org/project/ssondo/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/mohammedali2501/ssondo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-blue.svg)](https://python.org)

[Paper](https://arxiv.org/) | [Models](https://huggingface.co/mohammedali2501/ssondo) | [PyPI Package](https://pypi.org/project/ssondo/) | [Training Code](#training) | [Notebooks](#notebooks)

</div>

---

S-SONDO is the first framework for **self-supervised knowledge distillation** of general audio foundation models. It distills large teacher models into lightweight students that are **up to 61x smaller** while retaining **up to 96% of teacher performance** — using only output embeddings, no logits or layer-level alignment required.

<div align="center">
<img src="assets/paper_figures/ssondo_framework.png" width="600" alt="S-SONDO Architecture">
</div>

## Key Results

Downstream evaluation across **7 audio tasks** (4 music + 3 environmental sound). Students retain up to **96.4%** of teacher performance while being up to **61x smaller**.

<div align="center">
<img src="assets/paper_figures/downstream_evaluation.png" width="800" alt="Downstream Evaluation Results">

*Table 1 from the paper. Bold = best per student. Green = % of teacher performance retained.*
</div>

### Loss Function Comparison

<div align="center">
<img src="assets/paper_figures/loss_comparison.png" width="400" alt="Loss Comparison">
</div>

### Balanced Data Sampling (BDS) Ablation

<div align="center">
<img src="assets/paper_figures/bds_cluster_ablation.png" width="500" alt="BDS Cluster Ablation">
</div>

## Quick Start — Inference

```bash
pip install ssondo
```

```python
from ssondo import get_ssondo

# Load model (auto-downloads from Hugging Face Hub)
model = get_ssondo("matpac-mobilenetv3")

# Extract embeddings from audio
embeddings = model(audio)  # (batch, n_segments, 960)
```

### Finetuning (Linear Probe)

```python
model = get_ssondo("matpac-mobilenetv3")
model.freeze_backbone()

head = torch.nn.Linear(model.embedding_dim, num_classes)  # 960 -> your classes
emb = model.get_embeddings(audio)  # (batch, 960)
logits = head(emb)
```

### Available Models

```python
from ssondo import list_models
for name, desc in list_models().items():
    print(f"{name}: {desc}")
```

| Model | Teacher | Student | Status |
|:---:|:---:|:---:|:---:|
| `matpac-mobilenetv3` | MATPAC++ | MobileNetV3 | ✅ |
| `matpac-dymn` | MATPAC++ | DyMN | 🔜 |
| `matpac-eres2net` | MATPAC++ | ERes2Net | 🔜 |
| `m2d-mobilenetv3` | M2D | MobileNetV3 | 🔜 |
| `m2d-dymn` | M2D | DyMN | 🔜 |
| `m2d-eres2net` | M2D | ERes2Net | 🔜 |

## Training

Full 4-step training pipeline: download data → extract teacher embeddings → cluster → distill.

```bash
cd training_ssondo
./setup.sh          # install deps, download metadata + model checkpoints
./run_pipeline.sh   # run all 4 steps end-to-end
```

See [training_ssondo/readme.md](training_ssondo/readme.md) for the full pipeline documentation.

### Pipeline Overview

| Step | Module | Description |
|:---:|--------|-------------|
| 1 | `download_subset_of_audioset/` | Download AudioSet audio clips from YouTube |
| 2 | `extract_teachers_knowledge/` | Extract embeddings from teacher models (MATPAC, M2D) |
| 3 | `cluster_teachers_embeddings/` | Cluster embeddings for balanced data sampling |
| 4 | `knowledge_distillation_training/` | Train student models via knowledge distillation |

### Teacher Models

| Model | Source | Checkpoint |
|-------|--------|------------|
| MATPAC++ | [aurianworld/matpac](https://github.com/aurianworld/matpac) | `matpac_plus_6s_2048_enconly.pt` |
| M2D | [nttcslab/m2d](https://github.com/nttcslab/m2d) | `m2d_vit_base-80x608p16x16-221006-mr7` |

### Student Models

| Model | Source | Params |
|-------|--------|--------|
| MobileNetV3 | [fschmid56/EfficientAT](https://github.com/fschmid56/EfficientAT) | 2.9M |
| DyMN | [fschmid56/EfficientAT](https://github.com/fschmid56/EfficientAT) | — |
| ERes2Net | — | — |

## Notebooks

| Notebook | Description |
|----------|-------------|
| [Clustering Evaluation](notebooks/ssondo_clustering_evaluation.ipynb) | t-SNE, UMAP, KMeans clustering metrics on ESC-50 |
| [Linear Probe / Finetuning](notebooks/ssondo_linear_probe_esc50.ipynb) | Frozen backbone + linear head, or full finetuning on ESC-50 |

## Repository Structure

```
ssondo/
├── README.md
├── LICENSE
├── CITATION.cff
├── training_ssondo/          # Training pipeline (4 steps)
│   ├── setup.sh              # One-command setup
│   ├── run_pipeline.sh       # End-to-end demo
│   └── ...
├── inference_ssondo/         # pip install ssondo
│   ├── ssondo/               # PyPI package source
│   └── ...
├── notebooks/                # Evaluation notebooks
│   ├── ssondo_clustering_evaluation.ipynb
│   └── ssondo_linear_probe_esc50.ipynb
└── assets/
```

## Citation

If you use S-SONDO in your research, please cite:

```bibtex
@inproceedings{eladlouni2026ssondo,
  title={S-SONDO: Self-Supervised Knowledge Distillation for General Audio Foundation Models},
  author={El Adlouni, Mohammed Ali and Quelennec, Aurian and Chouteau, Pierre and Peeters, Geoffroy and Essid, Slim},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [MATPAC](https://github.com/aurianworld/matpac) — Teacher model
- [M2D](https://github.com/nttcslab/m2d) — Teacher model
- [EfficientAT](https://github.com/fschmid56/EfficientAT) — Student architectures (MobileNetV3, DyMN)
- [AudioSet](https://research.google.com/audioset/) — Training data
