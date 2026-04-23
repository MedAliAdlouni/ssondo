""" Clustering training with MiniBatchKMeans from sklearn."""
# Standard library imports
import os
import time
import random
import argparse
from pprint import pprint

# Third-party library imports
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local application/library imports
from training_ssondo import DATA
from training_ssondo.utils.audioset_loader import AudioSet
from .utils import merge_dicts, plot_inertia, save_clustering_results
from .dataset import TeacherKnowledgeDataset
from .config import conf, common_parameters



def main(conf) -> None:
  """Main function to run the clustering.

  Parameters
  ----------
  conf : dict
      Configuration dictionary.
  """
  # Reproducibility
  random.seed(conf["seed"])
  np.random.seed(conf["seed"])

  print(f"Starting clustering experiment with : {conf['n_clusters']}_clusters")

  # Load teacher embeddings
  root_dir = os.path.join(DATA, "AudioSet")
  print(f"Loading AudioSet from: {root_dir}")
  audioset_loader = AudioSet(root_dir=root_dir)

  print(' TeacherKnowledgeDataset to load "TRAIN" teacher knowledge of AudioSet')
  train_dataset = TeacherKnowledgeDataset(
      audioset_loader=audioset_loader,
      subset=conf["dataset"]["subset"],
      teacher_knowledge_path=conf["dataset"]["teacher_knowledge_path"],
  )
  print(f"Number of training samples: {len(train_dataset)}")
  print(f"dimension of embedding : {len(train_dataset[0])}")

  # Use DataLoader to stream dataset in batches
  dataloader = DataLoader(
      train_dataset, batch_size=conf["batch_size"], shuffle=True)
  print(f"Number of batches: {len(dataloader)}")

  # Initialize MiniBatchKMeans with parameters from config
  print("Initializing MiniBatchKMeans...")
  kmeans = MiniBatchKMeans(
      n_clusters=conf["n_clusters"],
      init=conf["init_method"],
      max_iter=conf["max_iter"],
      batch_size=conf["batch_size"],
      verbose=conf["verbose"],
      compute_labels=conf["compute_labels"],
      random_state=conf["seed"],
      tol=conf["tol"],
      max_no_improvement=conf["max_no_improvement"],
      init_size=conf["init_size"],
      n_init=conf["n_init"],
      reassignment_ratio=conf["reassignment_ratio"]
  )

  print("Training clustering model...")
  inertia_history = []

  if conf["use_fit"]:
    subset_size = int(conf["dataset"]["sample_size"] * len(train_dataset))
    if subset_size != len(train_dataset):
      print(
          f"Using .fit() with a dataset sample portion of {conf['dataset']['sample_size']}.")
      indices = np.random.choice(
          len(train_dataset), size=subset_size, replace=False)
    else:
      print(
          f"use whole dataset because subset size is equal to {conf['dataset']['sample_size']}")
      indices = range(len(train_dataset))

    print(f"Sampling {subset_size} embeddings out of {len(train_dataset)}...")
    sampled_embeddings = []
    for i in tqdm(indices, desc="Sampling subset of dataset"):
      # log_mem(i)
      _, emb = train_dataset[i]
      emb = emb.numpy()
      sampled_embeddings.append(emb)

    sample_array = np.vstack(sampled_embeddings)
    print("Start training : kmeans.fit(sample_array) ...")
    start_time = time.time()
    kmeans.fit(sample_array)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"TOTAL TRAINING TIME IS EQUAL TO {training_time} seconds ")
    print(f"Inertia after fit: {kmeans.inertia_:.4f}")
    inertia_history.append(kmeans.inertia_)

  else:
    for i, (_, embeddings) in enumerate(tqdm(dataloader, desc="Processing batches for clustering", unit="batch")):
      batch = embeddings.numpy()
      kmeans.partial_fit(batch)
      print(f"Inertia after batch {i+1}: {kmeans.inertia_:.4f}")
      inertia_history.append(kmeans.inertia_)

  print("Saving inertia history plot")
  plot_inertia(inertia_history, conf)

  # Save results
  print("\nSaving results...")
  save_clustering_results(conf, kmeans, inertia_history)

  print("\nClustering training experiment completed successfully!")


if __name__ == "__main__":

  import warnings
  warnings.filterwarnings("ignore")

  parser = argparse.ArgumentParser()
  parser.add_argument("--conf_id", required=False, default="None",
                      help="Conf tag, used to get the right config.")

  args = parser.parse_args()
  args = vars(args)

  conf = merge_dicts(common_parameters, conf[args["conf_id"]])
  conf = {**conf, **args}
  pprint(conf)

  main(conf)
