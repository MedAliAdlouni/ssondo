"""Utility functions for knowledge distillation."""

# Standard library imports
import numpy as np
from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import random
import pytorch_lightning as pl
import os
import json
import yaml


def seed_worker(worker_id: int) -> None:
  """Worker initialization function for DataLoader reproducibility."""
  worker_seed = int(torch.initial_seed() % 2**32)
  np.random.seed(worker_seed)
  random.seed(worker_seed)

# ----------------------------------------------------------------------------------------------------------------------
# sampling functions
# ----------------------------------------------------------------------------------------------------------------------
def compute_inverse_cluster_weights(cluster_labels: np.ndarray) -> np.ndarray:
  """
  Compute inverse weights for each sample based on cluster frequency.

  Args:
  cluster_labels : array-like
      Cluster assignments for each sample

  Returns:
  np.ndarray
      Weights for each sample (inverse of cluster frequency)
  """
  cluster_counts = Counter(cluster_labels)

  weights = np.zeros(len(cluster_labels))

  for i, cluster_id in enumerate(cluster_labels):
    weights[i] = 1.0 / cluster_counts[cluster_id]

  # Normalize weights so they sum to the number of samples
  weights = weights / weights.sum()

  return weights


def get_ft_weighted_sampler_SSL(
        train_dataset,
        num_samples: int = 100000,
        replacement=False,
        generator=None
) -> WeightedRandomSampler:
  """
  Generates a weighted random sampler using clustering of embeddings.

  Parameters
  ----------
  pdf : pandas.DataFrame
    DataFrame containing the data to be sampled.
  num_classes : int
      Number of classes in the dataset.
  num_samples : int, optional
      Number of samples to draw in each epoch (default is 100000).
  replacement : bool, optional
      Whether to sample with replacement (default is False).
  generator : torch.Generator, optional
      Generator for reproducible sampling (default is None).
  n_clusters : int, optional
      Number of clusters used in the clustering (default is 25).
  conf : dict
      Training configuration dictionary.

  Returns
  -------
  WeightedRandomSampler
      A PyTorch WeightedRandomSampler object configured with the specified parameters.
  """
  cluster_labels = np.zeros(len(train_dataset), dtype=int)

  for i in range(len(train_dataset)):
    cluster_label = train_dataset.get_cluster_label(i)
    cluster_labels[i] = cluster_label

    if i % 100_000 == 0 and i != 0:
      print(f"Processed {i} samples...")

  samples_weights = compute_inverse_cluster_weights(cluster_labels)

  return WeightedRandomSampler(weights=samples_weights,
                               num_samples=num_samples,
                               replacement=replacement,
                               generator=generator)


def get_ft_random_sampler(
    length_dataset: int,
    replacement: bool = False,
    num_samples: int = 100000,
    generator=None,
) -> WeightedRandomSampler:
  """
  Generates a custom random sampler that prevents PyTorch Lightning
  from replacing it with DistributedSampler.
  """
  sample_weights = torch.ones(length_dataset)

  return WeightedRandomSampler(
      weights=sample_weights,
      num_samples=num_samples,
      replacement=replacement,
      generator=generator,
  )


def get_ft_weighted_sampler(
        pdf: pd.DataFrame,
        num_classes: int,
        num_samples: int = 100000,
        replacement=False,
        generator=None) -> WeightedRandomSampler:
  """
  Generates a weighted random sampler for training.

  Parameters
  ----------
  pdf : pandas.DataFrame
    DataFrame containing the data to be sampled.
  num_classes : int
    Number of classes in the dataset.
  num_samples : int, optional
    Number of samples to draw in each epoch (default is 100000).
  replacement : bool, optional
    Whether to sample with replacement (default is False).
  generator : torch.Generator, optional
    Generator for reproducible sampling (default is None).

  Returns
  -------
  WeightedRandomSampler
    A PyTorch WeightedRandomSampler object configured with the specified parameters.
  """

  samples_weights = get_ft_cls_balanced_sample_weights(pdf,
                                                       num_classes)

  return WeightedRandomSampler(weights=samples_weights,
                               num_samples=num_samples,
                               replacement=replacement,
                               generator=generator)


def get_ft_cls_balanced_sample_weights(
        pdf: pd.DataFrame,
        num_classes: int,
        sample_weight_offset: int = 100,
        sample_weight_sum: bool = True) -> torch.Tensor:
  """
  Calculates class-balanced sample weights for a dataset.
  Memory-efficient version that avoids creating large intermediate matrices.

  Parameters
  ----------
  pdf : pandas.DataFrame
    DataFrame containing the dataset with a column 'label_idx' representing
    class indices (can be a list for multi-label cases).
  num_classes : int
    The number of unique classes in the dataset.
  sample_weight_offset : int, optional
    Offset added to the frequency of each class to avoid zero weights for
    low-frequency classes (default is 100).
  sample_weight_sum : bool, optional
    If True, sum the weights for each sample; otherwise, take the maximum
    weight (default is True).

  Returns
  -------
  all_weight : torch.Tensor
    A tensor containing the calculated sample weights.
  """

  # First, compute class frequencies without creating the full matrix
  per_class_counts = torch.zeros(num_classes, dtype=torch.long)
  
  # Process label_idx column to count class frequencies
  for label_idx in pdf["label_idx"]:
    if isinstance(label_idx, (list, np.ndarray)):
      # Multi-label case: label_idx is a list/array of class indices
      indices = torch.tensor(label_idx, dtype=torch.long)
      per_class_counts[indices] += 1
    else:
      # Single-label case: label_idx is a single integer
      per_class_counts[int(label_idx)] += 1
  
  # Convert to float and add offset
  per_class = per_class_counts.float().reshape(1, -1)  # shape: (1, num_classes)
  per_class = sample_weight_offset + per_class  # offset low freq classes
  
  if sample_weight_offset > 0:
    print(f"Warning: sample_weight_offset={sample_weight_offset} minnow={per_class.min()}")  # nopep8

  # Compute per-class weights
  per_class_weights = 1000. / per_class  # shape: (1, num_classes)
  
  # Now compute sample weights without creating the full matrix
  all_weight = torch.zeros(len(pdf), dtype=torch.float32)
  
  for i, label_idx in enumerate(pdf["label_idx"]):
    if isinstance(label_idx, (list, np.ndarray)):
      # Multi-label case
      indices = torch.tensor(label_idx, dtype=torch.long)
      sample_weights = per_class_weights[0, indices]  # Get weights for this sample's classes
      
      if sample_weight_sum:
        all_weight[i] = sample_weights.sum().item()
      else:
        all_weight[i] = sample_weights.max().item()
    else:
      # Single-label case
      idx = int(label_idx)
      all_weight[i] = per_class_weights[0, idx].item()
  
  return all_weight


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def set_random_seeds(seed: int) -> torch.Generator:
    """
    Set random seeds for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Random seed value.

    Returns
    -------
    torch.Generator
        PyTorch random generator with the specified seed.
    """
    print("Setting random seeds for reproducibility...")
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    
    # Create generator for dataloader reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    print(f"  ✓ All random seeds set to: {seed}\n")
    return generator


def save_config_file(conf: dict, log_dir: str) -> None:
    """
    Save configuration to YAML file in the log directory.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    log_dir : str
        Directory path for saving the config file.
    """
    print("Saving configuration file...")
    os.makedirs(log_dir, exist_ok=True)
    conf_path = os.path.join(log_dir, "conf.yml")
    
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)
    
    print(f"  ✓ Config saved to: {conf_path}\n")


def save_best_models(checkpoint_callback, log_dir: str) -> None:
    """
    Save the top-5 best model paths to JSON file.

    Parameters
    ----------
    checkpoint_callback : ModelCheckpoint
        The ModelCheckpoint callback with best model tracking.
    log_dir : str
        Directory path for saving the best models JSON.
    """
    print("\nSaving best model records...")
    best_k = {k: v.item() for k, v in checkpoint_callback.best_k_models.items()}
    best_k_path = os.path.join(log_dir, "best_k_models.json")
    
    with open(best_k_path, "w") as f:
        json.dump(best_k, f, indent=2)
    
    print(f"  ✓ Best models saved to: {best_k_path}")
    print(f"  Top model: {list(best_k.keys())[0] if best_k else 'None'}\n")

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def exp_warmup_linear_down(
        warmup,
        rampdown_length,
        start_rampdown,
        last_value):
  """
  Combines an exponential warmup phase with a linear rampdown phase.

  Parameters
  ----------
  warmup : int
    The number of epochs for the exponential warmup phase.
  rampdown_length : int
    The number of epochs over which the linear rampdown occurs.
  start_rampdown : int
    The epoch at which the linear rampdown starts.
  last_value : float
    The multiplier to obtain the final value after the rampdown is complete.

  Returns
  -------
  function
    A function that takes an epoch as input and returns the combined
    warmup and rampdown value for that epoch.
  """

  rampup = exp_rampup(warmup)
  rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)

  def wrapper(epoch):
    return rampup(epoch) * rampdown(epoch)

  return wrapper


def exp_rampup(rampup_length):
  """
  Generates an exponential ramp-up function.

  Parameters
  ----------
  rampup_length : int
    The length of the ramp-up period.

  Returns
  -------
  function
    A function that takes an epoch number as input and returns a float
    representing the ramp-up value for that epoch. The value is calculated
    using an exponential function if the epoch is within the ramp-up period,
    otherwise it returns 1.0.

  Notes
  -----
  It correspond to the Exponential rampup from : https://arxiv.org/abs/1610.02242
  """
  def wrapper(epoch):
    if epoch < rampup_length:
      epoch = np.clip(epoch, 0.5, rampup_length)
      phase = 1.0 - epoch / rampup_length
      return float(np.exp(-5.0 * phase * phase))
    else:
      return 1.0
  return wrapper


def linear_rampdown(rampdown_length, start=0, last_value=0):
  """
  Creates a linear rampdown function.

  Parameters
  ----------
  rampdown_length : int
    The number of epochs over which the rampdown occurs.
  start : int, optional
    The epoch at which the rampdown starts (default is 0).
  last_value : float, optional
    The multiplier to obtain the final value after the rampdown is complete
    (default is 0).

  Returns
  -------
  function
    A function that takes an epoch number as input and returns the
    corresponding rampdown value.
  """
  def wrapper(epoch):
    if epoch <= start:
      return 1.
    elif epoch - start < rampdown_length:
      return last_value + (1. - last_value) * (
          rampdown_length - epoch + start) / rampdown_length
    else:
      return last_value
  return wrapper


def merge_dicts(dict1: dict, dict2: dict) -> dict:
  """ Recursively merge two dictionaries, with values from dict2
  taking precedence.

  Parameters
  ----------
  dict1 : dict
    First dictionary.
  dict2 : dict
    Second dictionary.

  Returns
  -------
  dict
    The merged dictionary.

  Author: Joris Cosentino
  """
  merged = dict1.copy()
  for key, value in dict2.items():
    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):  # nopep8
      merged[key] = merge_dicts(merged[key], value)
    else:
      merged[key] = value
  return merged
