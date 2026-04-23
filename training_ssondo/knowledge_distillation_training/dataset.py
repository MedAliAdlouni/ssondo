"""PyTorch Dataset classes for loading AudioSet data with teacher knowledge."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio


class AudiosetDatasetKD(Dataset):
    """
    PyTorch Dataset to load AudioSet audios, teacher knowledge and labels.

    Parameters
    ----------
    audioset_loader : object
      The loader object that contains the metadata of AudioSet.
    subset : str
      The subset of the dataset to load between ("train", "eval", "all").
    teacher_knowledge_path : str or None
      Path to the directory containing the teacher knowledge files.
      If None, teacher knowledge will not be loaded.
    sr : int, optional
      Sample rate at which the model works (default is 16000).
    temperature : int, optional
      Temperature for scaling teacher predictions if teacher_knowledge_path is
      not None (default is 1).
    preprocess : callable, optional
      A preprocessing function to apply to the audio data (default is None).

    Methods
    -------
    __init_aggregated_labels(pdf)
      Aggregates labels and label indices by file_id in the given DataFrame.
    __len__()
      Returns the length of the dataset.
    __getitem__(index)
      Retrieves the audio tensor, labels tensor, and teacher predictions for a
      given index.
    load_subset(pdf, subset)
      Loads metadata and useful lists from the general DataFrame and the
      specified subset.
    labels_idx_to_tensor(labels_idx)
      Converts a list of label indices to a tensor with one-hot encoding.
    load_audio_tensor(file_path)
      Loads an audio file, converts it to mono, and resamples it if necessary.
    """

    def __init__(
        self,
        audioset_loader,
        subset,
        teacher_knowledge_path,
        cluster_labels_path,
        sr=16000,
        temperature=1,
        preprocess=None,
    ) -> None:

        pdf = audioset_loader._pdf_metadata.copy(deep=True)
        self.__init_aggregated_labels(pdf)

        metadata_df, file_path_list, labels_list, labels_idx_list = self.load_subset(
            pdf=pdf, subset=subset
        )

        self.subset = subset
        self.metadata_df = metadata_df
        self.file_path_list = file_path_list
        self.labels_list = labels_list
        self.labels_idx_list = labels_idx_list
        self.sr = sr
        self.preprocess = preprocess
        self.teacher_knowledge_path = teacher_knowledge_path
        self.cluster_labels_path = cluster_labels_path
        self.temperature = temperature

        if self.cluster_labels_path:
            cluster_labels = pd.read_csv(self.cluster_labels_path)

            # Normalize audio_id paths to match normalized file paths
            cluster_labels["audio_id"] = cluster_labels["audio_id"].apply(
                lambda x: os.path.normpath(x) if isinstance(x, str) else x
            )

            self.id_to_cluster = dict(
                zip(cluster_labels["audio_id"], cluster_labels["cluster_id"])
            )

    def __init_aggregated_labels(self, pdf):
        """
        Aggregates labels and label indices by file_id in the given DataFrame.

        This method performs the following steps:
        1. Groups the DataFrame by 'file_id' and aggregates 'label' and 'label_idx'
        into lists.
        2. Creates dictionaries for the aggregated labels and label indices.
        3. Maps the aggregated values back to the DataFrame.
        4. Drops duplicate rows to keep only one row per 'file_id'.
        5. Resets the DataFrame index.

        Parameters
        ----------
        pdf : pandas.DataFrame
          The input DataFrame containing 'file_id', 'label', and 'label_idx'
          columns.
        """
        # Drop duplicates to keep only one row per id
        pdf.drop_duplicates(subset=["file_id"], inplace=True)
        pdf.reset_index(drop=True, inplace=True)
        print("Labels aggregated.")

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
          The number of items in the file_path_list.
        """
        return len(self.file_path_list)

    def __getitem__(self, index):
        """
        Retrieves the audio tensor, label tensor, and teacher predictions for a
        given index.

        Parameters
        ----------
        index : int
          Index of the item to retrieve.

        Returns
        -------
        output : torch.Tensor
          The preprocessed or raw audio tensor.
        labels_tensor : torch.Tensor
          The tensor of labels corresponding to the audio.
        teacher_preds : torch.Tensor
          The tensor of teacher predictions, binarized and scaled by temperature.
          If teacher_knowledge_path is None, this will be a zero tensor.
        cluster_labels : torch.Tensor
          The tensor of pre-computed cluster labels for teacher embeddings.
          If cluster_labels_path is None, this will be a zero tensor.
        Raises
        ------
        FileNotFoundError
          If the teacher knowledge file does not exist.
        """

        file_path = self.file_path_list[index]
        labels_idx = self.labels_idx_list[index]
        labels_tensor = self.labels_idx_to_tensor(labels_idx)

        # load audio with error handling
        try:
            audio_tensor = self.load_audio_tensor(file_path=file_path)
        except Exception as e:
            # Re-raise with more context about which file failed
            raise RuntimeError(
                f"Failed to load audio at index {index}, file_path: {file_path}"
            ) from e

        if self.teacher_knowledge_path is None:
            # no teacher knowledge
            teacher_preds = torch.zeros_like(labels_tensor)

        else:
            # load stored teacher knowledge
            # Use os.path.basename() for cross-platform compatibility
            filename = os.path.basename(file_path)
            # Remove extension (.wav) and add .npz
            filename_without_ext = os.path.splitext(filename)[0]
            teacher_knowledge = np.load(
                os.path.join(
                    self.teacher_knowledge_path,
                    f"{self.subset}",
                    f"{filename_without_ext}.npz",
                ),
            )

            teacher_preds = torch.from_numpy(teacher_knowledge["embed"])
            teacher_preds = teacher_preds.squeeze(dim=0)  # shape : (emb_dim)

            # scale teacher_preds by temperature
            teacher_preds = teacher_preds / self.temperature

        if self.cluster_labels_path is None:
            cluster_label = torch.tensor(0, dtype=torch.long)
        else:
            # file_id is the file path without the .wav extension
            # file_path is already normalized, so file_id will be too
            file_id = file_path[:-4] if file_path.endswith(".wav") else file_path

            # Look up cluster label, with fallback if not found
            if file_id in self.id_to_cluster:
                cluster_label = torch.tensor(
                    self.id_to_cluster[file_id], dtype=torch.long
                )
            else:
                # If not found, try with normalized path again (in case of edge cases)
                file_id_normalized = os.path.normpath(file_id)
                if file_id_normalized in self.id_to_cluster:
                    cluster_label = torch.tensor(
                        self.id_to_cluster[file_id_normalized], dtype=torch.long
                    )
                else:
                    # If still not found, use default cluster 0 and warn
                    print(
                        f"Warning: Cluster label not found for file_id: {file_id}, using default cluster 0"
                    )
                    cluster_label = torch.tensor(0, dtype=torch.long)

        if self.preprocess is not None:
            output = self.preprocess(audio_tensor)
        else:
            output = audio_tensor

        return output, labels_tensor, teacher_preds, cluster_label

    def get_cluster_label(self, index):
        """
        Retrieves the cluster label for the given index.

        Parameters
        ----------
        index : int
            Index of the item in the dataset.

        Returns
        -------
        int or torch.Tensor
            The cluster label associated with the audio file at the given index.
            If cluster_labels_path is None, returns a zero tensor like in __getitem__.
        """
        file_path = self.file_path_list[index]
        if self.cluster_labels_path is None:
            raise KeyError("cluster_labels_path not found.")
        else:
            # file_id is the file path without the .wav extension
            file_id = file_path[:-4] if file_path.endswith(".wav") else file_path

            # Look up cluster label, with fallback if not found
            if file_id in self.id_to_cluster:
                cluster_id = self.id_to_cluster[file_id]
            else:
                # If not found, try with normalized path again
                file_id_normalized = os.path.normpath(file_id)
                if file_id_normalized in self.id_to_cluster:
                    cluster_id = self.id_to_cluster[file_id_normalized]
                else:
                    raise KeyError(f"Cluster label not found for file_id: {file_id}")
            return cluster_id

    def load_subset(self, pdf, subset):
        """
        Loads metadata and useful lists from the general dataframe and the
        specified subset.

        Parameters
        ----------
        pdf : pandas dataframe
          The dataframe containing all the metadata of audioset.
        subset : str
          The subset of the dataset to load between ("train", "eval", "all").

        Returns
        -------
        metadata_df : pandas.DataFrame
          The metadata of the specified subset.
        file_path_list : numpy.ndarray
          Array containing all the file paths of the metadata.
        labels_list : numpy.ndarray
          Array containing all labels for all audio files.
        labels_idx_list : numpy.ndarray
          Array containing all label indices for all audio files.

        Notes
        -----
        - For the "train" subset, both "unbalanced_train" and "balanced_train" sets
          are concatenated.
        - Only metadata entries with a duration of 10.0 seconds are included.
        """

        if subset == "train":
            metadata_df = pdf[pdf["set"].isin(["unbalanced_train", "balanced_train"])]
        elif subset == "eval":
            metadata_df = pdf[pdf["set"] == "eval"]
        elif subset == "all":
            metadata_df = pdf

        else:
            raise ValueError("Invalid subset. Choose between ['train', 'eval', 'all'].")

        AUDIOSET_CLIP_DURATION_S = 10.0
        metadata_df = metadata_df[metadata_df["duration"] == AUDIOSET_CLIP_DURATION_S]
        metadata_df = metadata_df.reset_index(drop=True)

        file_path_list = metadata_df["file_path"].values

        # Normalize file paths for Windows compatibility
        # Convert forward slashes to OS-appropriate path separators
        file_path_list = np.array([os.path.normpath(path) for path in file_path_list])

        # Filter to only include files that actually exist
        print(f"Filtering {subset} subset to only include existing audio files...")
        initial_count = len(file_path_list)

        # Create a mask for existing files
        file_exists_mask = np.array([os.path.exists(path) for path in file_path_list])

        # Filter all arrays using the mask
        file_path_list = file_path_list[file_exists_mask]
        labels_list = metadata_df["label"].values[file_exists_mask]
        labels_idx_list = metadata_df["label_idx"].values[file_exists_mask]
        metadata_df = metadata_df[file_exists_mask].reset_index(drop=True)

        final_count = len(file_path_list)
        print(
            f"  {subset} subset: {initial_count} files in metadata, {final_count} files exist ({final_count / initial_count * 100:.1f}%)"
        )

        if final_count == 0:
            raise ValueError(f"No existing audio files found for {subset} subset!")

        return metadata_df, file_path_list, labels_list, labels_idx_list

    def labels_idx_to_tensor(self, labels_idx):
        """
        Converts a list of label indices to a tensor with one-hot encoding.

        Parameters
        ----------
        labels_idx : list of int
          List of indices representing the labels.

        Returns
        -------
        torch.Tensor
          A tensor of shape (527,) with ones at the positions specified by
          labels_idx and zeros elsewhere.
        """

        labels_tensor = torch.zeros((527))
        labels_tensor[labels_idx] = 1
        return labels_tensor

    def load_audio_tensor(self, file_path: str) -> torch.Tensor:
        """
        Loads an audio file, converts it to mono, and resamples it if necessary.

        Parameters
        ----------
        file_path : str
          Path to the audio file to be loaded.

        Returns
        -------
        audio : torch.Tensor
          A tensor containing the audio data with shape (1, n_samples).

        Notes
        -----
        - The audio is loaded using torchaudio.
        - If the audio has more than one channel, it is converted to mono by
          averaging the channels.
        - If the sample rate of the audio does not match the desired sample rate
          (`self.sr`), the audio is resampled.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            audio_tensor, sr = torchaudio.load(file_path, normalize=True)
        except Exception as e:
            # Provide more informative error message
            error_msg = f"Error loading audio file: {file_path}\n"
            error_msg += f"Error type: {type(e).__name__}\n"
            error_msg += f"Error message: {str(e)}"
            raise RuntimeError(error_msg) from e

        # To Mono
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        # Resample
        if sr != self.sr:
            audio_tensor = torchaudio.functional.resample(
                waveform=audio_tensor, orig_freq=sr, new_freq=self.sr
            )

        # Pad or trim to exactly 10 seconds to ensure consistent tensor sizes
        expected_samples = 10 * self.sr
        if audio_tensor.shape[-1] < expected_samples:
            padding = expected_samples - audio_tensor.shape[-1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        elif audio_tensor.shape[-1] > expected_samples:
            audio_tensor = audio_tensor[:, :expected_samples]

        return audio_tensor
