"""A Pytorch Dataset class for loading and processing the AudioSet dataset."""

# Standard library imports
import os
import pickle
from typing import Optional, Tuple

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

# # Local application/library imports
from training_ssondo.utils.audioset_loader import AudioSet


class AudiosetDataset(Dataset):
  """
  A dataset class for loading and processing the AudioSet dataset.

  Parameters
  ----------
  subset : str
    The subset of the dataset to load between ("train", "eval", "all").
  sr : int, optional
    Sample rate at which the model works (default is 16000).
  audio_duration : int or None, optional
    Duration of the audio in seconds (default is None).
    If audio_duration is `None`, audio will not be cropped or padded, and
    the metadata will be filtered to retain only 10s audio.

  Methods
  -------
  __init_aggregated_labels(pdf)
    Aggregates labels and label indices by file_id in the given DataFrame.
  __len__()
    Returns the length of the dataset.
  __getitem__(index)
    Retrieve the file path and corresponding audio tensor for a given index.
  load_set(pdf, subset)
    Loads metadata and useful lists from the general dataframe and the
    specified subset.
  load_audio_tensor(file_path)
    Loads an audio file, converts it to mono, and resamples it if necessary.
  """

  def __init__(
          self,
          subset: str,
          sr: int = 16000,
          audio_duration: Optional[int] = None) -> None:

    audioset = AudioSet(
        root_dir=os.path.join(os.environ["DATA"], "AudioSet")  # nopep8
    )

    pdf = audioset._pdf_metadata.copy(deep=True)
    pdf = self.__init_aggregated_labels(pdf)  # Assign the returned aggregated DataFrame

    self.sr = sr
    self.audio_duration = audio_duration

    # Load the metadata and useful lists from the general dataframe and the
    # specified subset
    metadata_df, file_path_list, labels_list, labels_idx_list = self.load_set(
        pdf=pdf,
        subset=subset)

    self.metadata_df = metadata_df
    self.file_path_list = file_path_list
    self.labels_list = labels_list
    self.labels_idx_list = labels_idx_list


  def __init_aggregated_labels(self, pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates labels and label indices by file_id in the given DataFrame.
    Uses caching to avoid recomputing on subsequent runs.

    Parameters
    ----------
    pdf : pd.DataFrame
      The input DataFrame containing 'file_id', 'label', and 'label_idx'
      columns.
    
    Returns
    -------
    pd.DataFrame
      The aggregated DataFrame with one row per file_id.
    """

    # Determine cache file path based on metadata file location
    metadata_file_path = os.path.join(
        os.environ.get("DATA", ""), 
        "AudioSet", 
        "metadata.csv"
    )
    
    cache_file = os.path.join(os.path.dirname(metadata_file_path),
                                   "metadata_aggregated_cache.pkl")
    use_cache = False

    if os.path.exists(cache_file):
          try:
              print("Loading aggregated labels from cache...")
              with open(cache_file, 'rb') as f:
                  pdf_aggregated = pickle.load(f)
              if isinstance(pdf_aggregated, type(pdf)) and 'file_id' in pdf_aggregated.columns:
                  use_cache = True
                  print("Labels loaded from cache.")
                  return pdf_aggregated  # Return the cached version
          except Exception as e:
              print(f"Cache loading failed ({e}), recomputing...")
    
    if not use_cache:
        print("Aggregating labels...")        # Aggregate label and label_idx into lists by file_id
        aggregated = pdf.groupby("file_id").agg({
            "label": list,
            "label_idx": list
        }).reset_index()

        # Create dictionaries for the aggregated labels and labels_idx
        labels_dict = aggregated.set_index("file_id")["label"].to_dict()
        label_idx_dict = aggregated.set_index("file_id")["label_idx"].to_dict()

        # Replace individual labels and label_idx with aggregated values
        pdf["label"] = pdf["file_id"].map(labels_dict)
        pdf["label_idx"] = pdf["file_id"].map(label_idx_dict)

        # Drop duplicates to keep only one row per id
        pdf.drop_duplicates(subset=["file_id"], inplace=True)
        pdf.reset_index(drop=True, inplace=True)
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(pdf.copy(), f)
            print("Labels aggregated and cached.")
        except Exception as e:
            print(f"Warning: Could not save cache ({e})")
            print("Labels aggregated.")
        
        return pdf


  def __len__(self) -> int:
    """
    Returns the length of the dataset.

    Returns
    -------
    int
      The number of items in the file_path_list.
    """
    return len(self.file_path_list)


  def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
    """
    Retrieve the file path and corresponding audio tensor for a given index.

    Parameters
    ----------
    index : int
      Index of the item to retrieve.

    Returns
    -------
    Tuple[str, torch.Tensor]
      A tuple containing the file path and audio tensor.
    """

    file_path = self.file_path_list[index]
    audio_tensor = self.load_audio_tensor(file_path=file_path)

    return file_path, audio_tensor


  def load_set(
      self, 
      pdf: pd.DataFrame, 
      subset: str
  ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads metadata and useful lists from the general dataframe and the
    specified subset.

    Parameters
    ----------
    pdf : pd.DataFrame
      The dataframe containing all the metadata of audioset.
    subset : str
      The subset of the dataset to load between ("train", "eval", "all").

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
      A tuple containing (metadata_df, file_path_list, labels_list, labels_idx_list).

    Raises
    ------
    ValueError
      If the subset is not one of ["train", "eval", "all"].

    Notes
    -----
    - For the "train" subset, both "unbalanced_train" and "balanced_train" sets
      are concatenated.
    - Only metadata entries with a duration of 10.0 seconds are included if
      `self.audio_duration` is None.
    """

    if subset == "train":
      metadata_df = pdf[pdf["set"].isin(["unbalanced_train", "balanced_train"])]  # nopep8

    elif subset == "eval":
      metadata_df = pdf[pdf["set"] == "eval"]

    elif subset == "all":
      metadata_df = pdf

    else:
      raise ValueError(
          "Invalid subset. Choose between ['train', 'eval', 'all'].")

    if self.audio_duration is None:
      # Filter metadata to keep only 10s audio
      metadata_df = metadata_df[metadata_df["duration"] == 10.0]

    metadata_df = metadata_df.reset_index(drop=True)

    # Filter to only include files that actually exist on disk
    print("Filtering files to only include those that exist on disk...")
    file_exists_mask = metadata_df["file_path"].apply(os.path.exists)
    num_existing = file_exists_mask.sum()
    num_total = len(metadata_df)
    print(f"Found {num_existing}/{num_total} files that exist on disk.")
    
    metadata_df = metadata_df[file_exists_mask].reset_index(drop=True)

    file_path_list = metadata_df["file_path"].values
    labels_list = metadata_df["label"].values
    labels_idx_list = metadata_df["label_idx"].values

    return metadata_df, file_path_list, labels_list, labels_idx_list


  def central_crop_or_pad_audio_tensor(
          self,
          audio_tensor: torch.Tensor,
          audio_duration: int,
          sample_rate: int,
  ) -> torch.Tensor:
    """
    Central crops or pads the audio tensor to the desired duration.

    Parameters
    ----------
    audio_tensor : torch.Tensor
      The audio tensor to crop or pad.
    audio_duration : int
      The desired duration of the audio in seconds.
    sample_rate : int
      The sample rate of the audio.

    Returns
    -------
    torch.Tensor
      The audio tensor cropped or padded to the desired duration.
    """

    n_samples = audio_duration * sample_rate

    # Central crop the audio tensor if it is longer than the desired duration
    if audio_tensor.shape[-1] > n_samples:
      start = (audio_tensor.shape[-1] - n_samples) // 2
      audio_tensor = audio_tensor[:, start:start + n_samples]

    # Pad the audio tensor if it is shorter than the desired duration
    elif audio_tensor.shape[-1] < n_samples:
      padding_size = n_samples - audio_tensor.shape[-1]

      # Pad the audio tensor on the right with zeros to reach desired length
      audio_tensor = torch.nn.functional.pad(audio_tensor,
                                             pad=(0, padding_size))

    return audio_tensor


  def load_audio_tensor(self, file_path: str) -> torch.Tensor:
    """
    Loads an audio file, converts it to mono, and resamples it if necessary.

    Parameters
    ----------
    file_path : str
      Path to the audio file to be loaded.

    Returns
    -------
    torch.Tensor
      A tensor containing the audio data with shape (1, n_samples).

    Notes
    -----
    - The audio is loaded using torchaudio.
    - If the audio has more than one channel, it is converted to mono by
      averaging the channels.
    - If the sample rate of the audio does not match the desired sample rate
      (`self.sr`), the audio is resampled.
    """
    audio_tensor, sr = torchaudio.load(file_path, normalize=True)

    # Resample if the sample rate of the audio does not match the desired sample rate
    if sr != self.sr:
      audio_tensor = torchaudio.functional.resample(waveform=audio_tensor,
                                                    orig_freq=sr,
                                                    new_freq=self.sr)

    if self.audio_duration is not None:
      # Pad or truncate the audio to a fixed length
      audio_tensor = self.central_crop_or_pad_audio_tensor(
          audio_tensor,
          audio_duration=self.audio_duration,
          sample_rate=self.sr,
      )
    else:
      # Even when audio_duration is None, ensure consistent length for 10s audio files
      # This prevents variable-length tensor issues during batching
      expected_samples = int(10.0 * self.sr)  # 10 seconds at target sample rate
      if audio_tensor.shape[-1] != expected_samples:
        audio_tensor = self.central_crop_or_pad_audio_tensor(
            audio_tensor,
            audio_duration=int(10.0),
            sample_rate=self.sr,
        )

    # Ensure the tensor is contiguous, detached, and owns its data
    # This is critical for DataLoader with multiple workers on Windows
    audio_tensor = audio_tensor.contiguous()
    if audio_tensor.requires_grad:
      audio_tensor = audio_tensor.detach()
    # Create a new tensor that fully owns its data
    return audio_tensor.clone()