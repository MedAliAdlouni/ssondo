"""Wrapper classes for different model architectures."""

import torch
import torch.nn as nn
from typing import Tuple

from training_ssondo.utils.preprocess import SliceAudio


class ModelWrapper(nn.Module):
  """
  A wrapper class for different model architectures.

  Parameters
  ----------
  conf : dict
    Configuration dictionary containing model specifications.

  Attributes
  ----------
  model : nn.Module
    The wrapped model instance

  Methods
  -------
  forward(feats)
    Forward pass of the wrapped model.
  """

  def __init__(self, conf: dict):
    super().__init__()

    if "MATPAC" in conf["model"]["name"]:
      self.model = MATPAC(conf=conf)

    elif "M2D" == conf["model"]["name"]:
      self.model = m2d(conf=conf)

    else:
      raise NotImplementedError("Model wrapper not implemented yet.")

  def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass of the wrapped model, it only calls the forward of the model.

    Parameters
    ----------
    feats: torch.Tensor
      Input audio tensor of shape (bs, 1, n_samples)

    Returns
    -------
    features: torch.Tensor
      Features of the audio obtained from the model, with shape
      (bs, n_slices, features_dim).
    layer_results: torch.Tensor
      Output of each intermediate layer of shape (bs, n_layers, n_slices emb_dim)
      if they exist else same as emb.
    """

    features, layer_results = self.model(feats)
    return features, layer_results


class m2d(nn.Module):
  def __init__(self, conf: dict):
    super(m2d, self).__init__()
    from training_ssondo.utils.portable_m2d import PortableM2D

    self.model = PortableM2D(
        conf["model"]["ckpt_path"])

    self.pull_time_dimension = conf["model"]["pull_time_dimension"]
    self.target_sr = conf["model"]["sr"]

    self.max_len = conf["slice_audio"]["win_len"]
    self.slice_audio = SliceAudio(
        sr=conf["model"]["sr"],
        window_length=conf["slice_audio"]["win_len"],
        step_size=conf["slice_audio"]["step_size"],
        add_last=conf["slice_audio"]["add_last"])

  def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    emb, layer_results = self.forward_ds(feats)

    return emb, layer_results

  def forward_ds(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    if feats.shape[-1] > self.target_sr * self.max_len:
      feats = self.slice_audio(feats)  # shape (bs, n_slices, n_samples)

    bs, n_seg, n_samples = feats.shape

    feats = feats.reshape(bs*n_seg, n_samples)

    layer_results = self.model(feats)

    _, n_layer, n_emb, emb_dim = layer_results.shape

    layer_results = layer_results.reshape(bs, n_seg, n_layer, n_emb, emb_dim)

    layer_results = layer_results.mean(dim=1)
    if self.pull_time_dimension:
      layer_results = layer_results.mean(dim=-2)
    emb = layer_results[:, -1]

    return emb, layer_results


class MATPAC(nn.Module):
  def __init__(self, conf: dict):
    super().__init__()
    from matpac.model import get_matpac

    self.model = get_matpac(
        checkpoint_path=conf["model"]["ckpt_path"],
        pull_time_dimension=conf["model"]["pull_time_dimension"],
    )
    self.pull_time_dimension = conf["model"]["pull_time_dimension"]
    self.target_sr = conf["model"]["sr"]

    self.max_len = conf["slice_audio"]["win_len"]
    self.slice_audio = SliceAudio(
        sr=conf["model"]["sr"],
        window_length=conf["slice_audio"]["win_len"],
        step_size=conf["slice_audio"]["step_size"],
        add_last=conf["slice_audio"]["add_last"])

  def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # feats.shape: (bs, 1, n_samples)
    if feats.shape[-1] > self.target_sr * self.max_len:
      feats = self.slice_audio(feats)  # shape (bs, n_slices, n_samples)

    # reshaping feats into (bs * n_slices, n_samples)
    bs, n_slices, n_samples = feats.shape
    feats = feats.reshape(bs * n_slices, n_samples)

    emb, layer_results = self.model(feats)

    if self.pull_time_dimension:
      # reshaping emb
      _, emb_dim = emb.shape
      emb = emb.reshape(bs, n_slices, emb_dim)

      # reshaping layer_results
      _, n_layers, emb_dim = layer_results.shape
      layer_results = layer_results.reshape(bs, n_slices, n_layers, emb_dim)
      layer_results = layer_results.transpose(1, 2)  # shape (bs, n_layers, n_slices, emb_dim) # nopep8

    else:
      # reshaping emb
      _, t, emb_dim = emb.shape
      emb = emb.reshape(bs, n_slices, t, emb_dim)

      # reshaping layer_results
      _, n_layers, t, emb_dim = layer_results.shape
      layer_results = layer_results.reshape(bs, n_slices, n_layers, t, emb_dim)
      layer_results = layer_results.transpose(1, 2)  # shape (bs, n_layers, n_slices, t, emb_dim) # nopep8

    return emb, layer_results