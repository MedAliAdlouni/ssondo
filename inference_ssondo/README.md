# S-SONDO

Lightweight audio embeddings from self-supervised knowledge distillation.

S-SONDO provides compact audio models (MobileNetV3, DyMN, ERes2Net) trained via knowledge distillation from large audio foundation models (MATPAC, M2D). Extract general-purpose audio embeddings with a single function call.

**Paper:** *S-SONDO: Self-Supervised Knowledge Distillation for General Audio Foundation Models* (ICASSP 2026)

## Installation

```bash
pip install ssondo
```

## Quick Start

```python
import torchaudio
from ssondo import get_ssondo

# Load a pretrained model (auto-downloads from Hugging Face Hub)
model = get_ssondo("matpac-mobilenetv3")

# Load audio (mono, 32kHz)
x, sr = torchaudio.load("audio.wav")
x = x.mean(dim=0, keepdim=True)  # mono

# Extract embeddings
embeddings = model(x)  # (1, n_segments, 960)
```

## Available Models

```python
from ssondo import list_models

for name, description in list_models().items():
    print(f"{name}: {description}")
```

| Model | Teacher | Student | Embedding Size |
|-------|---------|---------|---------------|
| `matpac-mobilenetv3` | MATPAC++ | MobileNetV3 | 960 |
| `matpac-dymn` | MATPAC++ | DyMN | 960 |
| `matpac-eres2net` | MATPAC++ | ERes2Net | varies |
| `m2d-mobilenetv3` | M2D | MobileNetV3 | 960 |
| `m2d-dymn` | M2D | DyMN | 960 |
| `m2d-eres2net` | M2D | ERes2Net | varies |

## Usage

### Extract Embeddings

```python
model = get_ssondo("matpac-mobilenetv3")
embeddings = model(audio)  # (batch, n_segments, emb_size)
```

### Get Logits Too

```python
model = get_ssondo("matpac-mobilenetv3", return_logits=True)
embeddings, logits = model(audio)
```

### GPU Inference

```python
model = get_ssondo("matpac-mobilenetv3", device="cuda")
embeddings = model(audio.cuda())
```

### Load from Local Checkpoint

```python
model = get_ssondo("path/to/checkpoint.ckpt")
```

### Finetuning with Frozen Backbone (Linear Probe)

```python
import torch
from ssondo import get_ssondo

model = get_ssondo("matpac-mobilenetv3")
model.freeze_backbone()  # freeze all backbone params
model.train()

# Add a linear classifier for your task
head = torch.nn.Linear(model.embedding_dim, num_classes)

# Extract embeddings (backbone frozen, no grad)
emb = model.get_embeddings(audio)  # (batch, 960)
logits = head(emb)
loss = criterion(logits, labels)
loss.backward()  # only head parameters are updated
```

### Full Finetuning

```python
model = get_ssondo("matpac-mobilenetv3")
model.train()  # all parameters trainable by default
```

### Useful Properties

```python
model.embedding_dim   # 960 — size of backbone embeddings
model.backbone        # the raw backbone nn.Module (e.g., MobileNetV3)
```

## Input Requirements

- **Mono audio** (single channel)
- **Sample rate**: 32,000 Hz
- Audio is internally sliced into 10-second segments and converted to 128-band log-mel spectrograms

## How It Works

`get_ssondo()` auto-detects everything from the checkpoint: student backbone, preprocessing parameters, and classification head. No manual configuration needed.

When you pass a model name (e.g., `"matpac-mobilenetv3"`), the checkpoint is automatically downloaded from [Hugging Face Hub](https://huggingface.co/mohammedali2501/ssondo) and cached locally.

## Citation

```bibtex
@inproceedings{eladlouni2026ssondo,
  title={S-SONDO: Self-Supervised Knowledge Distillation for General Audio Foundation Models},
  author={El Adlouni, Mohammed Ali and Quelennec, Aurian and Chouteau, Pierre and Peeters, Geoffroy and Essid, Slim},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## License

MIT
