# SSONDO Inference Code

Lightweight pip-installable package for running SSONDO student models at inference.
These are compact audio models (MobileNetV3, DyMN, ERes2Net) trained via knowledge
distillation from large teacher models (MATPAC, M2D).

## Installation

```bash
pip install -e .
```

## Available Model Architectures

| Student Model | Typical Embedding Size |
|---------------|----------------------|
| MobileNetV3 (`mn10_im`) | 960 |
| DyMN (`dymn10_im`) | 960 |
| ERes2Net | varies with config |

## Usage

### Basic Usage -- Extract Embeddings

```python
import torchaudio
from ssondo.model import get_ssondo

# Load audio (must be mono at 32kHz)
x, sr = torchaudio.load("my_file.wav")
x = x.mean(dim=0, keepdim=True)  # mono (1, n_samples)

# Load model from a training checkpoint
model = get_ssondo(checkpoint_path="path/to/checkpoint.ckpt")

# Extract embeddings
embeddings = model(x)  # (1, n_segments, emb_size)
```

### Get Logits Too

```python
model = get_ssondo(
    checkpoint_path="path/to/checkpoint.ckpt",
    return_logits=True,
)
embeddings, logits = model(x)
# embeddings: (1, n_segments, emb_size)
# logits: (1, n_classes)
```

### GPU Inference

```python
model = get_ssondo(
    checkpoint_path="path/to/checkpoint.ckpt",
    device="cuda",
)
x = x.to("cuda")
embeddings = model(x)
```

## Input Requirements

- **Mono audio** (single channel)
- **Sample rate**: 32000 Hz (read from checkpoint config)
- Audio is internally sliced into 10-second segments and converted to 128-band
  log-mel spectrograms

## How It Works

`get_ssondo()` auto-detects everything from the checkpoint:
- Student backbone type (MobileNetV3, DyMN, or ERes2Net)
- Preprocessing parameters (mel spectrogram settings)
- Classification head configuration (MLP, Linear, RNN, etc.)

No manual configuration is needed -- just provide the checkpoint path.
