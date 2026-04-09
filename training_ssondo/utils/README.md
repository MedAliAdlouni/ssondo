# Utils Module

This directory contains utility modules and helper functions used throughout the SSONDO training pipeline.

## Modules

### `audioset_loader.py`

**Purpose**: Minimal AudioSet dataset loader class. Provides functionality to load AudioSet metadata and manage file paths.

**Key Components**:
- `AudioSet` class: Loads metadata CSV files and provides path resolution utilities


---

### `portable_m2d.py`

**Purpose**: Portable runtime implementation of Masked Modeling Duo (M2D) - a self-supervised audio representation learning model. Provides a standalone implementation that only requires `timm`, `einops`, and `nnAudio` dependencies.

**Key Components**:
- `PortableM2D` class: Complete M2D model with audio encoding, feature extraction, and classification capabilities
- `LocalViT`: Vision Transformer adapted for audio processing
- Helper functions for loading checkpoints and processing audio


---

### `preprocess.py`

**Purpose**: Audio preprocessing modules for converting raw audio signals into mel-spectrograms and slicing audio into segments.

**Key Components**:
- `LogMelSpectrogram`: PyTorch module that creates log-mel spectrograms from raw audio
- `SliceAudio`: PyTorch module for slicing audio waveforms into smaller segments with configurable window length and step size

---

### `student_models/`

**Purpose**: Collection of lightweight student model architectures used for knowledge distillation. These models are designed to be efficient and suitable for deployment while learning from teacher models.

**Subdirectories**:

#### `MobileNetV3/`
- MobileNetV3 architecture variants for audio classification
- Used in: `knowledge_distillation_training/model.py`

#### `dymn/`
- Dynamic MobileNet (DyMN) architecture with adaptive computation
- Used in: `knowledge_distillation_training/model.py`

#### `ERes2Net/`
- ERes2Net architecture for audio feature extraction
- Used in: `knowledge_distillation_training/model.py`

#### `quantizable_models/`
- Quantization-ready versions of student models (currently MobileNetV3)
- Enables model quantization for deployment

**Shared Utilities**:
- `model_utils.py`: Model wrapper classes and classification heads (Linear, MLP, RNN, AttentionRNN)
- `pooling_layers.py`: Pooling layer implementations (Mean, Attention)
- `utils.py`: Utility functions (parameter counting, etc.)
- `flop_count.py`: FLOP counting utilities
- `receptive_field.py`: Receptive field calculation utilities

**Used in**:
- `knowledge_distillation_training/model.py` - Building student models for knowledge distillation training

---

