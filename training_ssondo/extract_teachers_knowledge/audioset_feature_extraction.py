"""Script to extract features from the AudioSet dataset."""

# Standard library imports
import os
import yaml

# Third-party library imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local application/library imports
from .dataset import AudiosetDataset
from .models_wrappers import ModelWrapper

LOGIT_EMBED_MODELS = {
    "PaSST",
    "Ensemble5PaSST",
    "Ensemble9PaSST",
    "HTSAT",
    "EnsembleHTSAT",
}
EMBED_ONLY_MODELS = {
    "BEATs_iter3+",
    "CLAP",
    "MATPAC",
    "MATPAC_MCL",
    "MATPAC_CLS_MCL",
    "M2D",
}
FINETUNED_MODELS = {"BEATs_iter3+_finetuned", "EnsembleBEATs_finetuned"}


def _save_features(
    save_dir, file_names, features, layer_outputs, model_name, feature_type
):
    """Save extracted features to .npz files based on model category and feature type."""
    for i in range(features.shape[0]):
        save_path = os.path.join(save_dir, f"{file_names[i]}.npz")

        if model_name in LOGIT_EMBED_MODELS:
            logits = features[i, :, :527]
            embed = features[i, :, 527:]
            if feature_type == "all":
                np.savez(
                    file=save_path,
                    logits=logits,
                    embed=embed,
                    layer_outputs=layer_outputs[i],
                )
            elif feature_type == "logits":
                np.savez(file=save_path, logits=logits)
            elif feature_type == "embed":
                np.savez(file=save_path, embed=embed)

        elif model_name in EMBED_ONLY_MODELS:
            if feature_type == "all":
                np.savez(
                    file=save_path, embed=features[i], layer_outputs=layer_outputs[i]
                )
            elif feature_type == "embed":
                np.savez(file=save_path, embed=features[i])
            else:
                raise ValueError(
                    f"Model '{model_name}' does not support feature_type='{feature_type}'. "
                    "Use 'embed' or 'all'."
                )

        elif model_name in FINETUNED_MODELS:
            if feature_type == "all":
                np.savez(
                    file=save_path,
                    logits=features[i],
                    embed=layer_outputs[i, -1],
                    layer_outputs=layer_outputs[i],
                )
            elif feature_type == "logits":
                np.savez(file=save_path, logits=features[i])
            elif feature_type == "embed":
                np.savez(file=save_path, embed=layer_outputs[i, -1])

        else:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported: {LOGIT_EMBED_MODELS | EMBED_ONLY_MODELS | FINETUNED_MODELS}"
            )


def main(conf) -> None:
    """
    Main function for extracting features from an audio dataset using a
    specified model configuration.

    Parameters
    ----------
    conf : dict
      Configuration dictionary.

    Raises
    ------
    NotImplementedError
      If the specified dataset is not implemented.
    ValueError
      If the specified model name or feature type is not supported.
    """

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preparing data
    if conf["dataset"]["name"] == "AudioSet":
        dataset = AudiosetDataset(
            subset=conf["dataset"]["set"],
            sr=conf["model"]["sr"],
            audio_duration=conf["dataset"]["audio_duration"],
        )

    else:
        raise NotImplementedError(f"{conf['dataset']} has not been implemented yet.")

    num_workers = conf["process"]["num_workers"]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=conf["dataset"]["batch_size"],
        shuffle=conf["dataset"]["shuffle"],
        num_workers=num_workers,
        persistent_workers=False
        if num_workers == 0
        else conf["process"]["persistent_workers"],
        pin_memory=conf["process"]["pin_memory"],
        prefetch_factor=conf["process"]["prefetch_factor"] if num_workers > 0 else None,
        drop_last=conf["dataset"]["drop_last"],
    )

    # Setup the model
    model = ModelWrapper(conf)
    model.eval()
    model.to(device)

    # Define the save dir
    save_dir = os.path.join(
        conf["save_dir"],
        conf["model"]["name"],
        f"window_length_{str(conf['slice_audio']['win_len'])}s",
        conf["model"]["feature_type"],
        conf["dataset"]["set"],
    )
    os.makedirs(save_dir, exist_ok=True)

    # saving conf
    conf_path = os.path.join(
        conf["save_dir"],
        conf["model"]["name"],
        f"window_length_{str(conf['slice_audio']['win_len'])}s",
        conf["model"]["feature_type"],
        f"{conf['dataset']['set']}_conf.yml",
    )

    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Extract and save the feature of each file of the dataset
    num_batches = len(dataloader)
    print(
        f"Processing {num_batches} batch(es) with batch_size={conf['dataset']['batch_size']}"
    )
    for file_paths, audios in tqdm(
        dataloader, total=num_batches, desc="Extracting features"
    ):
        # Normalize file paths and extract just the filename
        file_names = [
            os.path.splitext(os.path.basename(str(fp)))[0] for fp in file_paths
        ]
        with torch.no_grad():
            audios = audios.to(device)
            features, layer_outputs = model(audios)

        features = features.cpu()
        layer_outputs = layer_outputs.cpu()

        feature_type = conf["model"]["feature_type"]
        if feature_type not in ("all", "logits", "embed"):
            raise ValueError(
                f"feature_type must be 'all', 'logits', or 'embed', got '{feature_type}'."
            )

        _save_features(
            save_dir,
            file_names,
            features,
            layer_outputs,
            conf["model"]["name"],
            feature_type,
        )


if __name__ == "__main__":
    import argparse
    from pprint import pprint
    from .config import conf, common_parameters

    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_id", required=True, help="Conf tag, used to get the right config."
    )

    args = parser.parse_args()
    args = vars(args)

    conf = {**conf[args["conf_id"]], **common_parameters}
    conf["conf_id"] = args["conf_id"]

    pprint(conf)
    main(conf)
