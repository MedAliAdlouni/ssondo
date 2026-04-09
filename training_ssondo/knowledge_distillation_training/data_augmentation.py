"""Data augmentation techniques to increase the diversity of the training set."""

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


class Normalize(nn.Module):
  """
  A PyTorch module for normalizing input data.

  Parameters
  ----------
  mean : float, optional
    The mean value to use for normalization (default is 0).
  std : float, optional
    The standard deviation value to use for normalization (default is 1).

  Methods
  -------
  forward(x)
    Normalizes the input tensor `x` using the specified mean and standard deviation.
  """

  def __init__(self,
               mean=0,
               std=1) -> None:
    super(Normalize, self).__init__()

    self.mean = mean
    self.std = std

  def forward(self, x):
    """
    Normalize the input tensor using the mean and standard deviation.

    Parameters
    ----------
    x : torch.Tensor
      The input tensor to be normalized.

    Returns
    -------
    x: torch.Tensor
      The normalized tensor.
    """

    x = (x - self.mean) / self.std
    return x


class Mixup(nn.Module):
  """
  A PyTorch module for applying Mixup data augmentation.

  Parameters
  ----------
  alpha : float, optional
    The parameter for the Beta distribution used to sample the mixup ratio
    (default is 0.3).

  Methods
  -------
  get_mixup_params(bs, device="cpu")
    Generates the mixup parameters (random indices and lambda values) for a
    batch of size `bs`.

  forward(sources, labels)
    Applies the mixup augmentation to the given sources and labels.
  """

  def __init__(self,
               alpha=0.3) -> None:
    super(Mixup, self).__init__()

    self.alpha = alpha

  def get_mixup_params(self, bs, device="cpu"):
    """
    Generates parameters for the mixup data augmentation technique.

    Parameters
    ----------
    bs : int
      Batch size, the number of samples in the batch.
    device : str, optional
      The device on which to place the generated parameters (default is "cpu").

    Returns
    -------
    rn_indices: torch.Tensor
      A tensor of randomly permuted indices of shape (bs,).
    lam: torch.Tensor
      A tensor of mixup lambda values of shape (bs,) on the specified device.
    """

    rn_indices = torch.randperm(bs)

    lambd = np.random.beta(self.alpha, self.alpha, bs).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.from_numpy(lambd).to(device)

    return rn_indices, lam

  def forward(self, sources, labels=None):
    """
    Applies mixup augmentation to the input sources and labels.

    Parameters
    ----------
    sources : torch.Tensor
      The input sources tensor.
    labels : torch.Tensor
      The input labels tensor.

    Returns
    -------
    mixed_sources : torch.Tensor
      The mixed sources tensor.
    mixed_labels : torch.Tensor
      The mixed labels tensor.
    rn_indices : torch.Tensor
      The random indices used for mixing.
    lam : torch.Tensor
      The lambda values used for mixing.
    """

    bs = sources.size(0)
    rn_indices, lam = self.get_mixup_params(bs, sources.device)

    # if spectrogram lam_shape = (bs, 1, 1, 1)
    # if waveform lam_shape = (bs, 1, 1)
    lam_shape = ((bs,) + tuple([1] * (sources.ndim - 1)))
    mixed_sources = sources * lam.reshape(lam_shape) + \
        sources[rn_indices] * (1. - lam.reshape(lam_shape))

    if labels is not None:
      lam_shape = ((bs,) + tuple([1] * (labels.ndim - 1)))
      mixed_labels = labels * lam.reshape(lam_shape) + \
          labels[rn_indices] * (1. - lam.reshape(lam_shape))
    else:
      mixed_labels = None

    return mixed_sources, mixed_labels, rn_indices, lam


def get_spec_augment_pipeline(conf):
  """
  Creates a sequential pipeline for SpecAugment data augmentation.

  Parameters
  ----------
  conf : dict
    Configuration dictionary containing the parameters for SpecAugment.
    Expected keys are:
    - "data_augmentation": dict
      - "spec_augment_args": dict
        - "time_masking": dict
          Parameters for the TimeMasking transformation.
        - "frequency_masking": dict
          Parameters for the FrequencyMasking transformation.

  Returns
  -------
  transform: nn.Sequential
    A sequential container of the SpecAugment transformations including
    time masking and frequency masking.
  """

  transform = nn.Sequential()

  time_masking = T.TimeMasking(
      **conf["data_augmentation"]["spec_augment_args"]["time_masking"])
  transform.append(time_masking)

  freq_masking = T.FrequencyMasking(
      **conf["data_augmentation"]["spec_augment_args"]["frequency_masking"])
  transform.append(freq_masking)

  return transform
