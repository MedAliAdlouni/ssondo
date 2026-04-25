"""S-SONDO: Lightweight audio embeddings from self-supervised knowledge distillation.

Extract general-purpose audio embeddings using compact student models (MobileNetV3,
DyMN, ERes2Net) trained via knowledge distillation from large audio foundation
models (MATPAC, M2D).

Usage:
    >>> from ssondo import get_ssondo, list_models
    >>> model = get_ssondo()  # defaults to matpac-mobilenetv3
    >>> embeddings = model(audio_tensor)

    # With a built-in classification head:
    >>> model = get_ssondo(head="linear", n_classes=50)
    >>> model.freeze_backbone()
    >>> logits = model(audio_tensor)

    # MLP head with custom hidden layers:
    >>> model = get_ssondo(head="mlp", n_classes=50, hidden_sizes=[512, 256])
"""

__version__ = "0.2.0"

from ssondo.model import get_ssondo, list_models, SsondoWrapper

__all__ = ["get_ssondo", "list_models", "SsondoWrapper", "__version__"]
