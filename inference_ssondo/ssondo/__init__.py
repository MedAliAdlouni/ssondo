"""S-SONDO: Lightweight audio embeddings from self-supervised knowledge distillation.

Extract general-purpose audio embeddings using compact student models (MobileNetV3,
DyMN, ERes2Net) trained via knowledge distillation from large audio foundation
models (MATPAC, M2D).

Usage:
    >>> from ssondo import get_ssondo, list_models
    >>> model = get_ssondo("matpac-mobilenetv3")
    >>> embeddings = model(audio_tensor)

    # Finetuning with frozen backbone:
    >>> model.freeze_backbone()
    >>> head = torch.nn.Linear(model.embedding_dim, num_classes)
    >>> emb = model.get_embeddings(audio)
    >>> logits = head(emb)
"""

__version__ = "0.1.1"

from ssondo.model import get_ssondo, list_models, SsondoWrapper

__all__ = ["get_ssondo", "list_models", "SsondoWrapper", "__version__"]
