"""S-SONDO: Lightweight audio embeddings from self-supervised knowledge distillation.

Extract general-purpose audio embeddings using compact student models (MobileNetV3,
DyMN, ERes2Net) trained via knowledge distillation from large audio foundation
models (MATPAC, M2D).

Usage:
    >>> from ssondo import get_ssondo, list_models, list_heads
    >>> model = get_ssondo()  # defaults to matpac-mobilenetv3
    >>> embeddings = model(audio_tensor)

    # Pretrained classifier (e.g., ESC-50):
    >>> model = get_ssondo(head="esc50")
    >>> logits = model(audio_tensor)  # (batch, 50)

    # Custom head:
    >>> model = get_ssondo(head="linear", n_classes=50)
    >>> model = get_ssondo(head="mlp", n_classes=50, hidden_sizes=[512, 256])
"""

__version__ = "0.3.1"

from ssondo.model import get_ssondo, list_models, list_heads, SsondoWrapper

__all__ = ["get_ssondo", "list_models", "list_heads", "SsondoWrapper", "__version__"]
