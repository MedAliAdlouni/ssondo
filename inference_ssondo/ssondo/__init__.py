"""S-SONDO: Lightweight audio embeddings from self-supervised knowledge distillation.

Extract general-purpose audio embeddings using compact student models (MobileNetV3,
DyMN, ERes2Net) trained via knowledge distillation from large audio foundation
models (MATPAC, M2D).

Usage:
    >>> from ssondo import get_ssondo, list_models
    >>> model = get_ssondo("matpac-mobilenetv3")
    >>> embeddings = model(audio_tensor)
"""

__version__ = "0.1.0"

from ssondo.model import get_ssondo, list_models, SsondoWrapper

__all__ = ["get_ssondo", "list_models", "SsondoWrapper", "__version__"]
