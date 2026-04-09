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
        audio_duration=conf["dataset"]["audio_duration"])

  else:
    raise NotImplementedError(
        f"{conf['dataset']} has not been implemented yet.")

  num_workers = conf["process"]["num_workers"]
  
  dataloader = DataLoader(
      dataset=dataset,
      batch_size=conf["dataset"]["batch_size"],
      shuffle=conf["dataset"]["shuffle"],
      num_workers=num_workers,
      persistent_workers=False if num_workers == 0 else conf["process"]["persistent_workers"],
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
      conf["dataset"]["set"])
  os.makedirs(save_dir, exist_ok=True)

  # saving conf
  conf_path = os.path.join(
      conf["save_dir"],
      conf["model"]["name"],
      f"window_length_{str(conf['slice_audio']['win_len'])}s",
      conf["model"]["feature_type"],
      f"{conf['dataset']['set']}_conf.yml")

  with open(conf_path, "w") as outfile:
    yaml.safe_dump(conf, outfile)

  # Extract and save the feature of each file of the dataset
  num_batches = len(dataloader)
  print(f"Processing {num_batches} batch(es) with batch_size={conf['dataset']['batch_size']}")
  for (file_paths, audios) in tqdm(dataloader, total=num_batches, desc="Extracting features"):
    # Normalize file paths and extract just the filename
    file_names = [os.path.splitext(os.path.basename(str(fp)))[0] for fp in file_paths]
    with torch.no_grad():
      audios = audios.to(device)
      features, layer_outputs = model(audios)

    features = features.cpu()
    layer_outputs = layer_outputs.cpu()

    # To extract everything from the model (logits and/or embeddings AND layer outputs) # nopep8
    if conf["model"]["feature_type"] == "all":

      if conf["model"]["name"] in ["PaSST",
                                   "Ensemble5PaSST",
                                   "Ensemble9PaSST",
                                   "HTSAT",
                                   "EnsembleHTSAT",
                                   ]:
        logits = features[:, :, :527]
        embed = features[:, :, 527:]

        for i in range(features.shape[0]):
          np.savez(file=os.path.join(save_dir, f"{file_names[i]}.npz"),
                   logits=logits[i],
                   embed=embed[i],
                   layer_outputs=layer_outputs[i])

      elif conf["model"]["name"] in ["BEATs_iter3+",
                                     "CLAP",
                                     "MATPAC",
                                     "MATPAC_MCL",
                                     "MATPAC_CLS_MCL",
                                     "M2D"]:
        for i in range(features.shape[0]):
          np.savez(file=os.path.join(save_dir, f"{file_names[i]}.npz"),
                   embed=features[i],
                   layer_outputs=layer_outputs[i])

      elif conf["model"]["name"] in ["BEATs_iter3+_finetuned",
                                     "EnsembleBEATs_finetuned"]:
        for i in range(features.shape[0]):
          np.savez(file=os.path.join(save_dir, f"{file_names[i]}.npz"),
                   logits=features[i],
                   embed=layer_outputs[i, -1],
                   layer_outputs=layer_outputs[i])

      else:
        raise ValueError

    # To extract only logits from the model
    elif conf["model"]["feature_type"] == "logits":

      if conf["model"]["name"] in ["PaSST",
                                   "Ensemble5PaSST",
                                   "Ensemble9PaSST",
                                   "HTSAT",
                                   "EnsembleHTSAT"]:
        logits = features[:, :, :527]

        for i in range(features.shape[0]):
          np.savez(file=os.path.join(save_dir, f"{file_names[i]}.npz"),
                   logits=logits[i])

      elif conf["model"]["name"] in ["BEATs_iter3+_finetuned",
                                     "EnsembleBEATs_finetuned"]:
        for i in range(features.shape[0]):
          np.savez(file=os.path.join(save_dir, f"{file_names[i]}.npz"),
                   logits=features[i])

      else:
        raise ValueError

    # To extract only embeddings from the model
    elif conf["model"]["feature_type"] == "embed":

      if conf["model"]["name"] in ["PaSST",
                                   "Ensemble5PaSST",
                                   "Ensemble9PaSST",
                                   "HTSAT",
                                   "EnsembleHTSAT"]:
        embed = features[:, :, 527:]

        for i in range(features.shape[0]):
          np.savez(file=os.path.join(save_dir, f"{file_names[i]}.npz"),
                   embed=embed[i])

      elif conf["model"]["name"] in ["BEATs_iter3+",
                                     "CLAP",
                                     "MATPAC",
                                     "MATPAC_MCL",
                                     "MATPAC_CLS_MCL",
                                     "M2D"]:
        for i in range(features.shape[0]):
          np.savez(file=os.path.join(save_dir, f"{file_names[i]}.npz"),
                   embed=features[i])

      elif conf["model"]["name"] in ["BEATs_iter3+_finetuned",
                                     "EnsembleBEATs_finetuned"]:
        for i in range(features.shape[0]):
          np.savez(file=os.path.join(save_dir, f"{file_names[i]}.npz"),
                   embed=layer_outputs[i, -1])

      else:
        raise ValueError

    else:
      raise ValueError("The feature_type argument should be in ['all', 'logits', 'embed'].")  # nopep8


if __name__ == "__main__":

  import argparse
  from pprint import pprint
  from .config import conf, common_parameters

  import warnings
  warnings.filterwarnings("ignore")

  parser = argparse.ArgumentParser()
  parser.add_argument("--conf_id", required=True,
                      help="Conf tag, used to get the right config.")

  args = parser.parse_args()
  args = vars(args)

  conf = {**conf[args["conf_id"]], **common_parameters}
  conf["conf_id"] = args["conf_id"]

  pprint(conf)
  main(conf)