"""PyTorch Dataset class for loading teacher knowledge (embeddings) for AudioSet data."""

# Standard library imports
import os
from typing import Tuple

# Third-party library imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

AUDIOSET_CLIP_DURATION_S = 10.0


class TeacherKnowledgeDataset(Dataset):
    """
    PyTorch Dataset to load only teacher knowledge embeddings.

    Parameters
    ----------
    audioset_loader : object
        The loader object that contains the metadata of AudioSet.
    subset : str
        The subset of the dataset to load between ("train", "eval", "all").
    teacher_knowledge_path : str
        Path to the directory containing the teacher knowledge files.
    temperature : float, optional
        Temperature for scaling teacher predictions (default is 1.0).
    """

    def __init__(
        self, audioset_loader, subset, teacher_knowledge_path, temperature=1.0
    ):

        pdf = audioset_loader._pdf_metadata.copy(deep=True)
        # Drop duplicates to keep only one row per id
        pdf.drop_duplicates(subset=["file_id"], inplace=True)
        pdf.reset_index(drop=True, inplace=True)

        metadata_df, file_path_list = self.load_subset(pdf=pdf, subset=subset)

        # Filter out samples where teacher knowledge files don't exist on disk
        metadata_df, file_path_list = self.filter_existing_files(
            metadata_df=metadata_df,
            file_path_list=file_path_list,
            subset=subset,
            teacher_knowledge_path=teacher_knowledge_path,
        )

        self.subset = subset
        self.metadata_df = metadata_df
        self.file_path_list = file_path_list
        self.teacher_knowledge_path = teacher_knowledge_path
        self.temperature = temperature

    def load_subset(self, pdf, subset) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Loads metadata from the general dataframe and the specified subset.

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
            metadata_df_train = pdf[
                pdf["set"].isin(["unbalanced_train", "balanced_train"])
            ]
            metadata_df_eval = pdf[pdf["set"] == "eval"]
            metadata_df = pd.concat(
                [metadata_df_eval, metadata_df_train], ignore_index=True
            )

        else:
            raise ValueError("Invalid subset. Choose between ['train', 'eval', 'all'].")

        metadata_df = metadata_df[metadata_df["duration"] == AUDIOSET_CLIP_DURATION_S]
        metadata_df = metadata_df.reset_index(drop=True)

        file_path_list = metadata_df["file_path"].values

        return metadata_df, file_path_list

    def filter_existing_files(
        self, metadata_df, file_path_list, subset, teacher_knowledge_path
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Filters out samples where the teacher knowledge file doesn't exist on disk.

        Parameters
        ----------
        metadata_df : pandas.DataFrame
            The metadata dataframe to filter.
        file_path_list : numpy.ndarray
            Array of file paths.
        subset : str
            The subset of the dataset ("train", "eval", "all").
        teacher_knowledge_path : str
            Path to the directory containing the teacher knowledge files.

        Returns
        -------
        filtered_metadata_df : pandas.DataFrame
            The filtered metadata dataframe containing only existing files.
        filtered_file_path_list : numpy.ndarray
            Array of file paths for existing files.
        """
        existing_indices = []

        print(
            f"Filtering {len(metadata_df)} samples to check for existing teacher knowledge files..."
        )

        for idx, file_path in enumerate(file_path_list):
            file_name = os.path.basename(file_path).replace(".wav", ".npz")

            if subset == "all":
                full_path_train = os.path.join(
                    teacher_knowledge_path, "train", file_name
                )
                full_path_eval = os.path.join(teacher_knowledge_path, "eval", file_name)

                if os.path.exists(full_path_train) or os.path.exists(full_path_eval):
                    existing_indices.append(idx)
            else:
                full_path = os.path.join(teacher_knowledge_path, subset, file_name)
                if os.path.exists(full_path):
                    existing_indices.append(idx)

        filtered_metadata_df = metadata_df.iloc[existing_indices].reset_index(drop=True)
        filtered_file_path_list = filtered_metadata_df["file_path"].values

        print(
            f"Found {len(existing_indices)} existing files out of {len(metadata_df)} samples."
        )

        return filtered_metadata_df, filtered_file_path_list

    def __len__(self) -> int:
        return len(self.file_path_list)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        file_path = self.file_path_list[index]
        file_name = os.path.basename(file_path).replace(".wav", ".npz")

        if self.subset == "all":
            full_path_train = os.path.join(
                self.teacher_knowledge_path, "train", file_name
            )
            full_path_eval = os.path.join(
                self.teacher_knowledge_path, "eval", file_name
            )

            if os.path.exists(full_path_train):
                full_path = full_path_train
            else:
                full_path = full_path_eval

        else:
            full_path = os.path.join(
                self.teacher_knowledge_path, self.subset, file_name
            )

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Teacher knowledge file not found: {full_path}")

        teacher_knowledge = np.load(full_path)
        teacher_preds = torch.from_numpy(teacher_knowledge["embed"]).squeeze(0)
        teacher_preds = teacher_preds / self.temperature

        return file_path[:-4], teacher_preds
