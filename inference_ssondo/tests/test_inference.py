"""Tests for the ssondo inference package.

Validates end-to-end inference with a real MATPAC++ MobileNetV3 checkpoint
and both synthetic and real audio inputs.
"""

import os
import glob

import pytest
import torch
import torchaudio

from ssondo.model import get_ssondo, SsondoWrapper


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT_PATH = os.environ.get(
    "SSONDO_CKPT_PATH",
    os.path.join(
        os.path.dirname(__file__), "..", "models", "matpac++_mobilenetv3_last.ckpt"
    ),
)

AUDIO_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "training_ssondo",
    "data",
    "AudioSet",
    "eval",
)

SR = 32_000  # expected sample rate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def model():
    """Load the checkpoint once for the entire test session."""
    return get_ssondo(CKPT_PATH)


@pytest.fixture(scope="session")
def model_with_logits():
    """Load the checkpoint with return_logits=True."""
    return get_ssondo(CKPT_PATH, return_logits=True)


@pytest.fixture(scope="session")
def training_config():
    """Extract training config from checkpoint for assertions."""
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    return ckpt["training_config"]


@pytest.fixture(scope="session")
def real_audio():
    """Load the first available AudioSet eval WAV file."""
    wav_files = glob.glob(os.path.join(AUDIO_DIR, "**", "*.wav"), recursive=True)
    assert len(wav_files) > 0, f"No WAV files found in {AUDIO_DIR}"

    wav_path = wav_files[0]
    waveform, sr = torchaudio.load(wav_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 32kHz if needed
    if sr != SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR)
        waveform = resampler(waveform)

    return waveform


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCheckpointLoading:
    def test_returns_ssondo_wrapper(self, model):
        assert isinstance(model, SsondoWrapper)

    def test_eval_mode(self, model):
        assert not model.training

    def test_has_student_model(self, model):
        assert hasattr(model.student_model, "model")
        assert hasattr(model.student_model.model, "emb_size")
        assert model.student_model.model.emb_size == 960

    def test_loading_with_logits_flag(self, model_with_logits):
        assert model_with_logits.return_logits is True


class TestPreprocessing:
    def test_preprocess_shape(self, model):
        """10s audio at 32kHz -> 1 segment of 128-band mel spectrogram."""
        x = torch.randn(1, SR * 10)
        with torch.no_grad():
            out = model.preprocess(x)
        assert out.ndim == 4
        assert out.shape[0] == 1  # batch
        assert out.shape[1] == 1  # 1 segment (10s window)
        assert out.shape[2] == 128  # mel bands


class TestInferenceSynthetic:
    def test_embeddings_shape(self, model):
        """10s synthetic audio -> (1, 1, 960) embeddings."""
        x = torch.randn(1, SR * 10)
        with torch.no_grad():
            emb = model(x)
        assert emb.shape == (1, 1, 960)

    def test_with_logits(self, model_with_logits, training_config):
        """With return_logits=True, should return (embeddings, logits)."""
        x = torch.randn(1, SR * 10)
        with torch.no_grad():
            result = model_with_logits(x)

        assert isinstance(result, tuple)
        assert len(result) == 2

        emb, logits = result
        assert emb.shape == (1, 1, 960)

        n_classes = training_config["classification_head"]["n_classes"]
        assert logits.shape == (1, n_classes)

    def test_batch_inference(self, model):
        """Batch of 2 synthetic audios."""
        x = torch.randn(2, SR * 10)
        with torch.no_grad():
            emb = model(x)
        assert emb.shape == (2, 1, 960)

    def test_longer_audio_multiple_segments(self, model):
        """25s audio with 10s window / 10s step -> 2 segments."""
        x = torch.randn(1, SR * 25)
        with torch.no_grad():
            emb = model(x)
        assert emb.shape == (1, 2, 960)


class TestInferenceRealAudio:
    def test_real_audio_runs(self, model, real_audio):
        """Inference on a real AudioSet WAV file should produce valid embeddings."""
        with torch.no_grad():
            emb = model(real_audio)

        assert emb.ndim == 3
        assert emb.shape[0] == 1  # batch
        assert emb.shape[2] == 960  # embedding dim
        assert not torch.isnan(emb).any(), "Output contains NaN"
        assert not torch.isinf(emb).any(), "Output contains Inf"


class TestDeterminism:
    def test_same_input_same_output(self, model):
        """Eval mode should produce identical outputs for the same input."""
        x = torch.randn(1, SR * 10)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2)
