"""Data pipeline setup for knowledge distillation training."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import (
    seed_worker,
    get_ft_weighted_sampler,
    get_ft_random_sampler,
    get_ft_weighted_sampler_SSL,
)
from .dataset import AudiosetDatasetKD
from .data_augmentation import Normalize

from training_ssondo import DATA
from training_ssondo.utils.audioset_loader import AudioSet
from training_ssondo.utils.preprocess import LogMelSpectrogram, SliceAudio


def setup_data_pipeline(conf: dict, generator: torch.Generator) -> tuple:
    """
    Set up the complete data pipeline including preprocessing, datasets, and dataloaders.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing dataset and preprocessing settings.
    generator : torch.Generator
        Random generator for reproducibility.

    Returns
    -------
    tuple
        (train_loader, val_loader) - PyTorch DataLoaders for training and validation.
    """
    print("=" * 80)
    print("SETTING UP DATA PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Preprocessing Pipeline
    # -------------------------------------------------------------------------
    print("\n[1/4] Preparing pre-processing pipeline...")
    preprocess = nn.Sequential(
        SliceAudio(
            sr=conf["student_model"]["sr"],
            window_length=conf["preprocess"]["slice_audio"]["win_len"],
            step_size=conf["preprocess"]["slice_audio"]["step_size"],
        ),
        LogMelSpectrogram(
            sample_rate=conf["student_model"]["sr"],
            win_length=conf["preprocess"]["logmelspec"]["win_len"],
            hop_length=conf["preprocess"]["logmelspec"]["hop_len"],
            n_mels=conf["preprocess"]["logmelspec"]["n_mels"],
            f_min=conf["preprocess"]["logmelspec"]["f_min"],
            f_max=conf["preprocess"]["logmelspec"]["f_max"],
        ),
        Normalize(
            mean=conf["preprocess"]["normalize"]["mean"],
            std=conf["preprocess"]["normalize"]["std"],
        ),
    )
    print("✓ Preprocessing pipeline configured")

    # -------------------------------------------------------------------------
    # 2. Load Datasets
    # -------------------------------------------------------------------------
    print("\n[2/4] Loading AudioSet datasets...")

    if conf["dataset"]["name"] != "audioset":
        raise ValueError(f"Dataset {conf['dataset']['name']} is not supported")

    # Set up AudioSet paths
    root_dir = os.path.join(DATA, "AudioSet")
    audioset_loader = AudioSet(root_dir=root_dir)

    cls_dir = os.path.join(
        conf["cluster_dir"], f"{conf['dataset']['sampler_args']['n_clusters']}_clusters"
    )
    cluster_df_path = os.path.join(cls_dir, "predicted_labels.csv")

    print(f"  Metadata file: {audioset_loader.metadata_file_path}")
    print(f"  Cluster labels: {cluster_df_path}")

    # Training dataset
    print("  Loading TRAIN dataset...")
    train_dataset = AudiosetDatasetKD(
        audioset_loader=audioset_loader,
        subset="train",
        teacher_knowledge_path=conf["dataset"]["teacher_knowledge_path"],
        cluster_labels_path=cluster_df_path,
        sr=conf["student_model"]["sr"],
        preprocess=preprocess,
    )
    print(f"  ✓ Train dataset loaded: {len(train_dataset)} samples")

    # Validation dataset
    print("  Loading EVAL dataset...")
    val_dataset = AudiosetDatasetKD(
        audioset_loader=audioset_loader,
        subset="eval",
        teacher_knowledge_path=conf["dataset"]["teacher_knowledge_path"],
        cluster_labels_path=cluster_df_path,
        sr=conf["student_model"]["sr"],
        preprocess=preprocess,
    )
    print(f"  ✓ Validation dataset loaded: {len(val_dataset)} samples")

    # -------------------------------------------------------------------------
    # 3. Configure Sampler
    # -------------------------------------------------------------------------
    print("\n[3/4] Configuring sampler...")

    # Create sampler based on configuration
    sampler_type = conf["dataset"]["sampler"]
    print(f"  Sampler type: {sampler_type}")

    if sampler_type == "WeightedRandomSampler":
        sampler = get_ft_weighted_sampler(
            pdf=train_dataset.metadata_df,
            num_classes=conf["dataset"]["n_classes"],
            num_samples=conf["dataset"]["sampler_args"]["num_samples"],
            replacement=conf["dataset"]["sampler_args"]["replacement"],
            generator=generator,
        )
    elif sampler_type == "RandomSampler":
        sampler = get_ft_random_sampler(
            length_dataset=len(train_dataset),
            replacement=conf["dataset"]["sampler_args"]["replacement"],
            num_samples=conf["dataset"]["sampler_args"]["num_samples"],
            generator=generator,
        )
    elif sampler_type == "WeightedRandomSamplerSSL":
        sampler = get_ft_weighted_sampler_SSL(
            train_dataset=train_dataset,
            num_samples=conf["dataset"]["sampler_args"]["num_samples"],
            replacement=conf["dataset"]["sampler_args"]["replacement"],
            generator=generator,
        )
    else:
        raise ValueError(f"Sampler {sampler_type} is not implemented")

    print(f"  ✓ Sampler configured: {len(sampler)} samples")

    # -------------------------------------------------------------------------
    # 4. Create DataLoaders
    # -------------------------------------------------------------------------
    print("\n[4/4] Creating DataLoaders...")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=conf["batch_size"],
        shuffle=conf["dataset"]["train_shuffle"],
        num_workers=conf["process"]["num_workers"],
        persistent_workers=conf["process"]["persistent_workers"],
        pin_memory=conf["process"]["pin_memory"],
        prefetch_factor=conf["process"]["prefetch"],
        worker_init_fn=seed_worker,
        generator=generator,
        sampler=sampler,
    )
    print(f"  ✓ Train loader: {len(train_loader)} batches")

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["process"]["num_workers"],
        persistent_workers=conf["process"]["persistent_workers"],
        pin_memory=conf["process"]["pin_memory"],
        prefetch_factor=conf["process"]["prefetch"],
        worker_init_fn=seed_worker,
        generator=generator,
    )
    print(f"  ✓ Validation loader: {len(val_loader)} batches")

    print("\n" + "=" * 80)
    print("DATA PIPELINE SETUP COMPLETE")
    print("=" * 80 + "\n")

    return train_loader, val_loader
