"""Cluster-Based Teacher Label Prediction"""

# Standard library imports
import os
import argparse
import pickle
import random
from pprint import pprint

# Third-party library imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

# Local application/library imports
from training_ssondo import DATA
from training_ssondo.utils.audioset_loader import AudioSet
from .config import conf, common_parameters
from .dataset import TeacherKnowledgeDataset
from .utils import merge_dicts, save_labels, log_mem


def main(conf) -> None:
    random.seed(conf["seed"])
    np.random.seed(conf["seed"])

    parent_dir = os.path.join(conf["exp_dir"], f"{conf['n_clusters']}_clusters")
    model_path = os.path.join(parent_dir, "kmeans_model.pkl")

    with open(model_path, "rb") as f:
        kmeans: MiniBatchKMeans = pickle.load(f)

    # Load data
    root_dir = os.path.join(DATA, "AudioSet")
    print(f"Loading AudioSet from: {root_dir}")
    audioset_loader = AudioSet(root_dir=root_dir)
    print("Loading teacher knowledge...")
    train_dataset = TeacherKnowledgeDataset(
        audioset_loader=audioset_loader,
        subset=conf["dataset"]["subset"],
        teacher_knowledge_path=conf["dataset"]["teacher_knowledge_path"],
    )

    dataloader = DataLoader(train_dataset, batch_size=conf["batch_size"], shuffle=False)
    print(f"Number of batches: {len(dataloader)}")

    print("Predicting labels...")
    all_ids = []
    all_labels = []
    for i, (ids, embeddings) in enumerate(
        tqdm(dataloader, desc="Predicting", unit="batch")
    ):
        log_mem(i)
        embeddings = embeddings.numpy()
        labels = kmeans.predict(embeddings)
        all_ids.extend(ids)
        all_labels.extend(labels)

    df = pd.DataFrame({"audio_id": all_ids, "cluster_id": all_labels})
    all_labels_arr = np.array(all_labels)
    print(
        f"Total: {len(all_ids)} ids, {len(all_labels)} labels, shape: {all_labels_arr.shape}"
    )
    print("Saving predicted labels!")
    save_labels(df, conf)
    print("Label Prediction completed successfully!")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_id", required=True, help="Conf tag")
    args = vars(parser.parse_args())
    conf = merge_dicts(common_parameters, conf[args["conf_id"]])
    conf = {**conf, **args}
    pprint(conf)
    main(conf)
