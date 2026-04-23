"""S-SONDO inference module.

Provides `get_ssondo()` to load a trained student model and return an
nn.Module ready for inference on raw audio waveforms.
"""

from __future__ import annotations

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

HF_REPO_ID = "MedAliAdlouni/ssondo"

AVAILABLE_MODELS = {
    "matpac-mobilenetv3": {
        "filename": "matpac_mobilenetv3.ckpt",
        "description": "MobileNetV3 distilled from MATPAC++ (cosine similarity, 50 clusters)",
    },
    "matpac-dymn": {
        "filename": "matpac_dymn.ckpt",
        "description": "DyMN distilled from MATPAC++ (cosine similarity, 50 clusters)",
    },
    "matpac-eres2net": {
        "filename": "matpac_eres2net.ckpt",
        "description": "ERes2Net distilled from MATPAC++ (cosine similarity, 50 clusters)",
    },
    "m2d-mobilenetv3": {
        "filename": "m2d_mobilenetv3.ckpt",
        "description": "MobileNetV3 distilled from M2D (cosine similarity, 50 clusters)",
    },
    "m2d-dymn": {
        "filename": "m2d_dymn.ckpt",
        "description": "DyMN distilled from M2D (cosine similarity, 50 clusters)",
    },
    "m2d-eres2net": {
        "filename": "m2d_eres2net.ckpt",
        "description": "ERes2Net distilled from M2D (cosine similarity, 50 clusters)",
    },
}


def list_models() -> dict[str, str]:
    """List available pretrained S-SONDO models.

    Returns
    -------
    dict[str, str]
        Mapping of model name to description.

    Example
    -------
    >>> from ssondo import list_models
    >>> for name, desc in list_models().items():
    ...     print(f"{name}: {desc}")
    """
    return {name: info["description"] for name, info in AVAILABLE_MODELS.items()}


def _resolve_checkpoint(checkpoint: str, device: str = "cpu") -> dict:
    """Resolve a checkpoint path or model name to a loaded checkpoint dict.

    Supports:
    - Local file path (e.g., "path/to/checkpoint.ckpt")
    - Pretrained model name (e.g., "matpac-mobilenetv3") — auto-downloads from HF Hub
    """
    import os

    if os.path.isfile(checkpoint):
        return torch.load(checkpoint, map_location=device, weights_only=False)

    if checkpoint in AVAILABLE_MODELS:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=AVAILABLE_MODELS[checkpoint]["filename"],
        )
        return torch.load(local_path, map_location=device, weights_only=False)

    raise ValueError(
        f"'{checkpoint}' is not a valid file path or model name. "
        f"Available models: {list(AVAILABLE_MODELS.keys())}"
    )


def _build_student_model(conf: dict) -> ModelWrapper:
    """Build the student model from a training config dict."""
    model_name = conf["student_model"]["model_name"]

    if "mn" in model_name and "dy" not in model_name:
        model = get_mobilenet(
            pretrained_name=None,
            width_mult=conf["student_model"]["width_mult"],
            reduced_tail=conf["student_model"]["reduced_tail"],
            dilated=conf["student_model"]["dilated"],
            strides=conf["student_model"]["strides"],
            relu_only=conf["student_model"]["relu_only"],
            input_dim_f=conf["student_model"]["input_dim_f"],
            input_dim_t=conf["student_model"]["input_dim_t"],
            se_dims=conf["student_model"]["se_dims"],
            se_agg=conf["student_model"]["se_agg"],
            se_r=conf["student_model"]["se_r"],
        )
    elif "dy" in model_name:
        model = get_dymn(
            pretrained_name=None,
            width_mult=conf["student_model"]["width_mult"],
            strides=conf["student_model"]["strides"],
            pretrain_final_temp=conf["student_model"].get("pretrain_final_temp", 1.0),
        )
    else:
        model = ERes2Net(
            m_channels=conf["student_model"]["m_channels"],
            feat_dim=conf["student_model"]["feat_dim"],
            num_blocks=conf["student_model"]["num_blocks"],
            pooling_func=conf["student_model"]["pooling_func"],
            add_layer=conf["student_model"]["add_layer"],
        )

    head_type = conf["classification_head"]["head_type"]

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
            last_activation=conf["classification_head"]["last_activation"],
        )
    elif head_type in ["lstm", "gru", "rnn"]:
        class_head = RNNClassifer(
            rnn_type=head_type,
            emb_size=model.emb_size,
            hidden_size=conf["classification_head"]["hidden_size"],
            n_classes=conf["classification_head"]["n_classes"],
            num_layers=conf["classification_head"]["num_layers"],
            bidirectional=conf["classification_head"]["bidirectional"],
            n_last_elements=conf["classification_head"]["n_last_elements"],
            last_activation=conf["classification_head"]["last_activation"],
        )
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
            last_activation=conf["classification_head"]["last_activation"],
        )
    else:
        class_head = LinearClassifer(
            emb_size=model.emb_size,
            n_classes=conf["classification_head"]["n_classes"],
            pooling=conf["classification_head"]["pooling"],
            activation_att=conf["classification_head"]["activation_att"],
            last_activation=conf["classification_head"]["last_activation"],
        )

    return ModelWrapper(model=model, classification_head=class_head)


class SsondoWrapper(nn.Module):
    """End-to-end inference wrapper for S-SONDO student models.

    Takes raw mono audio waveforms and returns embeddings (and optionally logits).
    """

    def __init__(self, student_model, slicer, logmel, return_logits=False):
        super().__init__()
        self.student_model = student_model
        self.slicer = slicer
        self.logmel = logmel
        self.return_logits = return_logits

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw audio to log-mel spectrogram segments.

        Parameters
        ----------
        x : torch.Tensor
            Raw mono audio of shape ``(batch, n_samples)`` or ``(n_samples,)``.

        Returns
        -------
        torch.Tensor
            Log-mel spectrogram of shape ``(batch, n_segments, n_mels, time_frames)``.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        x = self.slicer(x)
        x = self.logmel(x)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run inference on raw audio.

        Parameters
        ----------
        x : torch.Tensor
            Raw mono audio at the expected sample rate (typically 32 kHz).
            Shape: ``(batch, n_samples)`` or ``(n_samples,)``.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Embeddings of shape ``(batch, n_segments, emb_size)``.
            If ``return_logits=True``, returns ``(embeddings, logits)``.
        """
        x = self.preprocess(x)
        logits, embeddings = self.student_model(x)
        if self.return_logits:
            return embeddings, logits
        return embeddings


def get_ssondo(
    checkpoint: str,
    return_logits: bool = False,
    device: str = "cpu",
) -> SsondoWrapper:
    """Load a pretrained S-SONDO model ready for inference.

    Parameters
    ----------
    checkpoint : str
        Either a model name (e.g., ``"matpac-mobilenetv3"``) which
        auto-downloads from Hugging Face Hub, or a local path to a
        ``.ckpt`` file.
    return_logits : bool, optional
        If True, forward returns ``(embeddings, logits)`` instead of
        just embeddings.
    device : str, optional
        Device to load the model on (default: ``"cpu"``).

    Returns
    -------
    SsondoWrapper
        An ``nn.Module`` in eval mode.

    Example
    -------
    >>> import torchaudio
    >>> from ssondo import get_ssondo
    >>> model = get_ssondo("matpac-mobilenetv3")
    >>> x, sr = torchaudio.load("audio.wav")
    >>> embeddings = model(x.mean(0, keepdim=True))  # mono
    """
    ckpt = _resolve_checkpoint(checkpoint, device)
    conf = ckpt["training_config"]

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

    student_model = _build_student_model(conf)

    state_dict = ckpt["state_dict"]
    prefix = "student_model."
    new_state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    student_model.load_state_dict(new_state_dict, strict=True)

    wrapper = SsondoWrapper(
        student_model=student_model,
        slicer=slicer,
        logmel=logmel,
        return_logits=return_logits,
    )
    wrapper = wrapper.to(device)
    wrapper.eval()
    return wrapper
