"""SSONDO inference module.

Provides a factory function `get_ssondo()` to load a trained student model
checkpoint and return an nn.Module ready for inference on raw audio.
"""

import torch
import torch.nn as nn

from ssondo.preprocess import SliceAudio, LogMelSpectrogram
from ssondo.models.MobileNetV3.model import get_model as get_mobilenet
from ssondo.models.dymn.model import get_model as get_dymn
from ssondo.models.ERes2Net.model import ERes2Net
from ssondo.models.model_utils import (
    LinearClassifer,
    MLPClassifer,
    RNNClassifer,
    AttentionRNNClassifer,
    ModelWrapper,
)


def _build_student_model(conf: dict) -> ModelWrapper:
    """Build the complete student model with backbone and classification head
    from a training configuration dictionary.

    Parameters
    ----------
    conf : dict
        Configuration dictionary (extracted from checkpoint["training_config"]).

    Returns
    -------
    ModelWrapper
        Complete student model (backbone + classification head).
    """

    # -------------------------------------------------------------------------
    # 1. Build Backbone Network
    # -------------------------------------------------------------------------
    model_name = conf["student_model"]["model_name"]

    # MobileNet variants (excluding DyMN)
    if "mn" in model_name and "dy" not in model_name:
        model = get_mobilenet(
            pretrained_name=None,  # weights come from checkpoint
            width_mult=conf["student_model"]["width_mult"],
            reduced_tail=conf["student_model"]["reduced_tail"],
            dilated=conf["student_model"]["dilated"],
            strides=conf["student_model"]["strides"],
            relu_only=conf["student_model"]["relu_only"],
            input_dim_f=conf["student_model"]["input_dim_f"],
            input_dim_t=conf["student_model"]["input_dim_t"],
            se_dims=conf["student_model"]["se_dims"],
            se_agg=conf["student_model"]["se_agg"],
            se_r=conf["student_model"]["se_r"]
        )

    # Dynamic MobileNet (DyMN)
    elif "dy" in model_name:
        model = get_dymn(
            pretrained_name=None,  # weights come from checkpoint
            width_mult=conf["student_model"]["width_mult"],
            strides=conf["student_model"]["strides"],
            pretrain_final_temp=conf["student_model"].get("pretrain_final_temp", 1.0),
        )

    # ERes2Net
    else:
        model = ERes2Net(
            m_channels=conf["student_model"]["m_channels"],
            feat_dim=conf["student_model"]["feat_dim"],
            num_blocks=conf["student_model"]["num_blocks"],
            pooling_func=conf["student_model"]["pooling_func"],
            add_layer=conf["student_model"]["add_layer"]
        )

    # -------------------------------------------------------------------------
    # 2. Build Classification Head
    # -------------------------------------------------------------------------
    head_type = conf["classification_head"]["head_type"]

    # MLP Head
    if head_type == "mlp":
        try:
            hidden_features_size = model.last_channel
        except AttributeError:
            hidden_features_size = conf["classification_head"]["hidden_features_size"]

        class_head = MLPClassifer(
            emb_size=model.emb_size,
            n_classes=conf["classification_head"]["n_classes"],
            hidden_features_size=hidden_features_size,
            pooling=conf["classification_head"]["pooling"],
            activation_att=conf["classification_head"]["activation_att"],
            last_activation=conf["classification_head"]["last_activation"]
        )

    # RNN-based Heads (LSTM, GRU, RNN)
    elif head_type in ["lstm", "gru", "rnn"]:
        class_head = RNNClassifer(
            rnn_type=head_type,
            emb_size=model.emb_size,
            hidden_size=conf["classification_head"]["hidden_size"],
            n_classes=conf["classification_head"]["n_classes"],
            num_layers=conf["classification_head"]["num_layers"],
            bidirectional=conf["classification_head"]["bidirectional"],
            n_last_elements=conf["classification_head"]["n_last_elements"],
            last_activation=conf["classification_head"]["last_activation"]
        )

    # Attention RNN Heads
    elif head_type in ["attention_lstm", "attention_gru", "attention_rnn"]:
        rnn_type = head_type.split("_")[1]
        class_head = AttentionRNNClassifer(
            rnn_type=rnn_type,
            emb_size=model.emb_size,
            hidden_size=conf["classification_head"]["hidden_size"],
            n_classes=conf["classification_head"]["n_classes"],
            num_layers=conf["classification_head"]["num_layers"],
            bidirectional=conf["classification_head"]["bidirectional"],
            n_last_elements=conf["classification_head"]["n_last_elements"],
            last_activation=conf["classification_head"]["last_activation"]
        )

    # Linear Head (default)
    else:
        class_head = LinearClassifer(
            emb_size=model.emb_size,
            n_classes=conf["classification_head"]["n_classes"],
            pooling=conf["classification_head"]["pooling"],
            activation_att=conf["classification_head"]["activation_att"],
            last_activation=conf["classification_head"]["last_activation"]
        )

    # -------------------------------------------------------------------------
    # 3. Combine Backbone + Head
    # -------------------------------------------------------------------------
    student_model = ModelWrapper(
        model=model,
        classification_head=class_head
    )

    return student_model


class SsondoWrapper(nn.Module):
    """End-to-end inference wrapper for SSONDO student models.

    Takes raw audio waveforms and returns embeddings (and optionally logits).

    Parameters
    ----------
    student_model : ModelWrapper
        The student model (backbone + classification head).
    slicer : SliceAudio
        Audio slicing module.
    logmel : LogMelSpectrogram
        Log-mel spectrogram module.
    return_logits : bool
        If True, also return classification logits.
    """

    def __init__(self, student_model, slicer, logmel, return_logits=False):
        super(SsondoWrapper, self).__init__()

        self.student_model = student_model
        self.slicer = slicer
        self.logmel = logmel
        self.return_logits = return_logits

    def preprocess(self, x):
        """Convert raw audio to log-mel spectrogram segments.

        Parameters
        ----------
        x : torch.Tensor
            Raw audio waveform of shape (bs, n_samples) or (1, n_samples).
            Must be mono at the model's expected sample rate.

        Returns
        -------
        torch.Tensor
            Log-mel spectrogram of shape (bs, n_segments, n_mels, time_frames).
        """
        # SliceAudio expects (bs, 1, n_samples) for batch
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)  # (bs, 1, n_samples)
        x = self.slicer(x)   # (bs, n_segments, segment_samples)
        x = self.logmel(x)   # (bs, n_segments, n_mels, time_frames)
        return x

    def forward(self, x):
        """Run inference on raw audio.

        Parameters
        ----------
        x : torch.Tensor
            Raw audio waveform of shape (bs, n_samples) at the expected
            sample rate (typically 32000 Hz). Must be mono.

        Returns
        -------
        torch.Tensor or tuple
            If return_logits is False: embeddings of shape
            (bs, n_segments, emb_size).
            If return_logits is True: tuple of (embeddings, logits).
        """
        x = self.preprocess(x)
        logits, embeddings = self.student_model(x)

        if self.return_logits:
            return embeddings, logits
        return embeddings


def get_ssondo(
    checkpoint_path: str,
    return_logits: bool = False,
    device: str = "cpu",
) -> SsondoWrapper:
    """Load a trained SSONDO student model and return a ready-to-use inference
    wrapper.

    The checkpoint must be a PyTorch Lightning `.ckpt` file from the
    `KnowledgeDistillationSystem` training pipeline. The model architecture,
    preprocessing parameters, and classification head configuration are all
    auto-detected from the saved training config.

    Parameters
    ----------
    checkpoint_path : str
        Path to a `.ckpt` checkpoint file.
    return_logits : bool, optional
        If True, the wrapper returns (embeddings, logits). If False, returns
        only embeddings (default is False).
    device : str, optional
        Device to load the model on (default is "cpu").

    Returns
    -------
    SsondoWrapper
        An nn.Module in eval mode, ready for inference.

    Example
    -------
    >>> import torchaudio
    >>> from ssondo.model import get_ssondo
    >>> model = get_ssondo("path/to/checkpoint.ckpt")
    >>> x, sr = torchaudio.load("audio.wav")
    >>> x = x.mean(dim=0, keepdim=True)  # mono
    >>> embeddings = model(x)
    """

    # 1. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Extract training config
    conf = checkpoint["training_config"]

    # 3. Build preprocessing from config
    sr = conf["student_model"]["sr"]
    logmel_params = conf["preprocess"]["logmelspec"]
    slice_params = conf["preprocess"]["slice_audio"]

    slicer = SliceAudio(
        sr=sr,
        window_length=slice_params["win_len"],
        step_size=slice_params["step_size"],
    )

    logmel = LogMelSpectrogram(
        sample_rate=sr,
        win_length=logmel_params["win_len"],
        hop_length=logmel_params["hop_len"],
        n_mels=logmel_params["n_mels"],
        f_min=logmel_params["f_min"],
        f_max=logmel_params["f_max"],
    )

    # 4. Build student model from config
    student_model = _build_student_model(conf)

    # 5. Load state dict (strip "student_model." prefix from Lightning keys)
    state_dict = checkpoint["state_dict"]
    prefix = "student_model."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v

    student_model.load_state_dict(new_state_dict, strict=True)

    # 6. Wrap and set to eval
    wrapper = SsondoWrapper(
        student_model=student_model,
        slicer=slicer,
        logmel=logmel,
        return_logits=return_logits,
    )

    wrapper = wrapper.to(device)
    wrapper.eval()

    return wrapper
