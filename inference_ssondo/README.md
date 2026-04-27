<div align="center">

# S-SONDO

**Lightweight audio embeddings from self-supervised knowledge distillation.**

Up to **61x smaller** than teacher models, retaining up to **96% performance**.

<a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-preprint-b31b1b?style=for-the-badge&logo=arxiv" alt="Paper"></a>&nbsp;
<a href="https://huggingface.co/mohammedali2501/ssondo"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Models-yellow?style=for-the-badge" alt="HuggingFace"></a>&nbsp;
<a href="https://github.com/MedAliAdlouni/ssondo_temp"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github" alt="GitHub"></a>

*ICASSP 2026*

</div>

---

## Install

```bash
pip install ssondo
```

## Quick Start

```python
from ssondo import get_ssondo

model = get_ssondo()
embeddings = model(audio)  # (batch, n_segments, 960)
```

No preprocessing, no config files, no manual downloads. Pass raw mono audio at 32 kHz and get embeddings.

## Pretrained Classifiers

7 ready-to-use classifiers trained on standard audio benchmarks:

```python
model = get_ssondo(head="esc50")
logits = model(audio)  # (batch, 50)
```

| Head | Task | Classes |
|------|------|:-------:|
| `esc50` | Environmental sound | 50 |
| `us8k` | Urban sound | 10 |
| `fsd50k` | Sound events | 200 |
| `gtzan` | Music genre | 10 |
| `openmic` | Instrument recognition | 20 |
| `nsynth` | Instrument family | 11 |
| `magna-tag-a-tune` | Music auto-tagging | 50 |

## Custom Heads

```python
# Linear
model = get_ssondo(head="linear", n_classes=10)

# MLP
model = get_ssondo(head="mlp", n_classes=10, hidden_sizes=[512, 256])
```

## Finetuning

```python
# Linear probing (frozen backbone)
model = get_ssondo(head="linear", n_classes=10)
model.freeze_backbone()
model.train()

logits = model(audio)
loss = criterion(logits, labels)
loss.backward()  # only head parameters update

# Full finetuning
model.unfreeze_backbone()
```

## API at a Glance

```python
from ssondo import get_ssondo, list_models, list_heads

model = get_ssondo()                          # default backbone
model = get_ssondo("matpac-dymn")             # specific backbone
model = get_ssondo(head="esc50")              # pretrained classifier
model = get_ssondo(head="linear", n_classes=10)  # custom head
model = get_ssondo(device="cuda")             # GPU
model = get_ssondo("path/to/checkpoint.ckpt") # local checkpoint

embeddings = model(audio)                     # (batch, n_segments, 960)
emb = model.get_embeddings(audio)             # (batch, 960) mean-pooled
model.embedding_dim                           # 960
model.backbone                                # raw nn.Module

list_models()                                 # available backbones
list_heads()                                  # available classifiers
```

## Available Models

The **matpac-mobilenetv3** combination achieved the best downstream performance across all 7 benchmarks (96.4% of teacher performance at 61x fewer parameters). This is the model provided in the package and on Hugging Face, along with all 7 pretrained classification heads.

| Model | Teacher | Student | Params | Emb. | Avg. Score | Status |
|-------|---------|---------|:------:|:----:|:----------:|:------:|
| **`matpac-mobilenetv3`** | **MATPAC++** | **MobileNetV3** | **2.9M** | **960** | **73.0** | **Available** |
| `matpac-dymn` | MATPAC++ | DyMN | 8.7M | 960 | 72.6 | Coming soon |
| `matpac-eres2net` | MATPAC++ | ERes2Net | 1.4M | 10240 | 70.8 | Coming soon |
| `m2d-mobilenetv3` | M2D | MobileNetV3 | 2.9M | 960 | 69.2 | Coming soon |
| `m2d-dymn` | M2D | DyMN | 8.7M | 960 | 68.7 | Coming soon |
| `m2d-eres2net` | M2D | ERes2Net | 1.4M | 10240 | 69.2 | Coming soon |

## Input

- **Mono audio**, single channel
- **Sample rate:** 32,000 Hz
- Internally sliced into 10 s segments and converted to 128-band log-mel spectrograms

## Links

- **Paper:** [arXiv](https://arxiv.org/)
- **Models:** [Hugging Face Hub](https://huggingface.co/mohammedali2501/ssondo)
- **Code & Training:** [GitHub](https://github.com/MedAliAdlouni/ssondo_temp)

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
