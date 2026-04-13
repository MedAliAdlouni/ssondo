"""File containing the modules used to process raw audio signals."""

import numpy as np
import torch
import torch.nn as nn
import torchaudio


class LogMelSpectrogram(nn.Module):
  r"""Create MelSpectrogram for a raw audio signal.

  .. devices:: CPU CUDA

  .. properties:: Autograd TorchScript

  This is a composition of :py:func:`torchaudio.transforms.Spectrogram` and
  and :py:func:`torchaudio.transforms.MelScale`.

  Sources
      * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
      * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
      * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

  Args:
      sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
      n_fft (int or None, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``None``)
      win_length (float or None, optional): Window size. (Default: ``0.025``)
      hop_length (float or None, optional): Length of hop between STFT windows. (Default: ``0.01``)
      f_min (float, optional): Minimum frequency. (Default: ``0.0``)
      f_max (float or None, optional): Maximum frequency. (Default: ``None``)
      pad (int, optional): Two sided padding of signal. (Default: ``0``)
      n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
      window_fn (Callable[..., Tensor], optional): A function to create a window tensor
          that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
      power (float, optional): Exponent for the magnitude spectrogram,
          (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2.0``)
      normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
      wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
      center (bool, optional): whether to pad :attr:`waveform` on both sides so
          that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
          (Default: ``False``)
      pad_mode (string, optional): controls the padding method used when
          :attr:`center` is ``True``. (Default: ``"reflect"``)
      onesided: Deprecated and unused.
      norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
          (area normalization). (Default: ``slaney``)
      mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``slaney``)
  """

  def __init__(self,
               sample_rate=16000,
               n_fft=None,
               win_length=0.025,
               hop_length=0.01,
               f_min=0.0,
               f_max=None,
               log_offset=0.001,
               pad=0,
               n_mels=128,
               window_fn=torch.hann_window,
               power=2.0,
               normalized=False,
               wkwargs=None,
               center=False,
               pad_mode="reflect",
               onesided=None,
               norm="slaney",
               mel_scale="slaney",
               ) -> None:
    super(LogMelSpectrogram, self).__init__()

    if f_max is None:
      f_max = sample_rate // 2

    win_length = int(np.round(sample_rate * win_length))

    if n_fft is None:
      n_fft = win_length

    if hop_length is None:
      hop_length = win_length // 2
    else:
      hop_length = int(np.round(sample_rate * hop_length))

    self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        pad=pad,
        n_mels=n_mels,
        window_fn=window_fn,
        power=power,
        normalized=normalized,
        wkwargs=wkwargs,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
        norm=norm,
        mel_scale=mel_scale)

    self.log_offset = log_offset

  def forward(self, waveform):
    mel_specgram = self.MelSpectrogram(waveform)
    log_melspecgram = torch.log(mel_specgram + self.log_offset)
    return log_melspecgram


class SliceAudio(nn.Module):
  """A PyTorch module for slicing audio waveforms into smaller segments.

  Parameters
  ----------
  sr : int, optional
    Sample rate of the audio (default is 16000).
  window_length : float, optional
    Length of each audio segment in seconds (default is 1).
  step_size : float, optional
    Step size between consecutive segments in seconds (default is 1).
  add_last : bool, optional
    Whether to add the last segment even if it is shorter than window_length
    (default is False).

  Methods
  -------
  process_audio(waveform)
    Process a single audio waveform and slice it into segments.
  process_audio_batch(waveforms)
    Process a batch of audio waveforms and slice them into segments.
  forward(x)
    Applies the appropriate slicing method based on the input dimensions.

  Raises
  ------
  ValueError
    If the input waveform is not mono or if the input dimensions are incorrect.
  """

  def __init__(self, sr=16000,
               window_length=1,
               step_size=1,
               add_last=False) -> None:
    super(SliceAudio, self).__init__()

    self.sr = sr
    self.window_length = window_length
    self.step_size = step_size
    self.add_last = add_last

  def process_audio(self, waveform):
    """
    Function to process a single audio waveform.

    Parameters
    ----------
    waveform : torch.Tensor
      A mono audio waveform tensor with shape [1, n_samples].

    Returns
    -------
    audio_segments: torch.Tensor
      A tensor containing the segmented audio waveform with shape
      [n_segments, segment_size].
      If `self.add_last` is True, an additional segment containing the
      remaining samples is included.

    Raises
    ------
    ValueError
      If the input waveform is not mono (i.e., does not have shape [1, n_samples]).
    """

    if waveform.shape[0] != 1:
      raise ValueError(
          "The audio waveform must be mono, with a shape of [1, n_samples]")

    _, n_samples = waveform.shape
    segment_size = int(self.window_length * self.sr)
    step_size = int(self.step_size * self.sr)

    if self.add_last:
      n_segments = int(1 + (n_samples - (segment_size + 1)) // step_size)
      audio_segments = torch.zeros((n_segments + 1, segment_size))
    else:
      n_segments = int(1 + (n_samples - segment_size) // step_size)
      audio_segments = torch.zeros((n_segments, segment_size))
    device = waveform.device
    audio_segments = audio_segments.to(device=device)

    for i in range(n_segments):
      start = i * step_size
      end = start + segment_size
      audio_segments[i] = waveform[:, start:end]

    if self.add_last:

      # If waveform shorter than the slice len
      if n_segments == 0:
        len_last = waveform.shape[-1]
        audio_segments[0][:len_last] = waveform
      else:
        len_last = waveform[:, end:].shape[-1]
        audio_segments[i + 1][:len_last] = waveform[:, end:]

    return audio_segments

  def process_audio_batch(self, waveforms):
    """
    Function to process a batch of audio waveforms.

    Parameters
    ----------
    waveforms : torch.Tensor
      A batch of mono audio waveforms with shape [bs, 1, n_samples].

    Returns
    -------
    audio_segments: torch.Tensor
      A tensor containing the segmented audio waveforms with shape
      [bs, n_segments, segment_size].
      If `self.add_last` is True, an additional segment containing the
      remaining samples is included.

    Raises
    ------
    ValueError
      If the input waveforms are not mono (i.e., shape is not [bs, 1, n_samples]).
    """

    if waveforms.shape[1] != 1:
      raise ValueError(
          "The batch of audio waveforms must be in mono, with a shape of [bs, 1, n_samples]")

    # batch of mono audio waveforms, with shape [bs, 1, n_samples]
    bs, _, n_samples = waveforms.shape
    waveforms = waveforms.squeeze(dim=1)  # shape [bs, n_samples]

    segment_size = int(self.window_length * self.sr)
    step_size = int(self.step_size * self.sr)

    if self.add_last:
      n_segments = int(1 + (n_samples - (segment_size + 1)) // step_size)
      audio_segments = torch.zeros((bs, n_segments + 1, segment_size))
    else:
      n_segments = int(1 + (n_samples - segment_size) // step_size)
      audio_segments = torch.zeros((bs, n_segments, segment_size))
    device = waveforms.device
    audio_segments = audio_segments.to(device=device)

    for i in range(n_segments):
      start = i * step_size
      end = start + segment_size
      audio_segments[:, i, :] = waveforms[:, start:end]

    if self.add_last:

      # If waveform shorter than the slice len
      if n_segments == 0:
        len_last = waveforms.shape[-1]
        audio_segments[:, :len_last] = waveforms
      else:
        len_last = waveforms[:, end:].shape[-1]
        audio_segments[:, i + 1, :len_last] = waveforms[:, end:]

    return audio_segments

  def forward(self, x):
    """
    Applies the appropriate slicing method based on the input dimensions.

    Parameters
    ----------
    x : torch.Tensor
      The input audio waveform. It must be either a 2D array of shape
      [1, n_samples] for a single audio waveform or a 3D array of shape
      [bs, 1, n_samples] for a batch of audio waveforms.

    Returns
    -------
    torch.Tensor
      The processed audio waveform. It is either a 2D array of shape
      [n_segments, segment_size] for a single processed audio waveform, or a 3D
      array of shape [bs, n_segments, segment_size] for a batch of processed
      audio waveforms.

    Raises
    ------
    ValueError
      If the input audio waveform does not have 2 or 3 dimensions.
    """

    if x.ndim == 3:
      return (self.process_audio_batch(x))
    elif x.ndim == 2:
      return (self.process_audio(x))
    else:
      raise ValueError(
          "The audio waveform must be of shape [1, n_samples] or [bs, 1, n_samples] for batch audio waveforms")
