"""Evaluate and visualize clustering for teacher model embeddings.
This module computes clustering metrics (inertia, silhouette score, etc.),
and saves t-SNE and UMAP visualization data for the clustered teacher embeddings.
"""

# Standard library imports
import os
import random
import argparse
from pprint import pprint

# Third-party library imports
import numpy as np
import pandas as pd

# Local application/library imports
from training_ssondo import DATA
from training_ssondo.utils.audioset_loader import AudioSet
from .config import conf, common_parameters
from .dataset import TeacherKnowledgeDataset
from .utils import (
    merge_dicts,
    save_clustering_metrics,
    save_tsne_data,
    save_umap_data,
    compute_clustering_metrics,
)


def main(conf) -> None:
    random.seed(conf["seed"])
    np.random.seed(conf["seed"])

    parent_dir = os.path.join(conf["exp_dir"], f"{conf['n_clusters']}_clusters")
    df_path = os.path.join(parent_dir, "predicted_labels.csv")

    print(f"Loading predicted labels from {df_path}")
    df = pd.read_csv(df_path)

    root_dir = os.path.join(DATA, "AudioSet")
    audioset_loader = AudioSet(root_dir=root_dir)
    train_dataset = TeacherKnowledgeDataset(
        audioset_loader=audioset_loader,
        subset=conf["dataset"]["subset"],
        teacher_knowledge_path=conf["dataset"]["teacher_knowledge_path"],
    )

    print("Computing metrics...")
    metrics = compute_clustering_metrics(train_dataset, df, conf)

    print("\nCLUSTERING METRICS")
    print("=" * 40)
    print(f"Inertia: {metrics.get('inertia', 'N/A')}")
    if metrics.get("silhouette_score") is not None:
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    if metrics.get("calinski_harabasz_score") is not None:
        print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
    if metrics.get("davies_bouldin_score") is not None:
        print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")

    save_clustering_metrics(conf, metrics)
    save_tsne_data(train_dataset, df, conf)
    save_umap_data(train_dataset, df, conf)

    print("Evaluation completed successfully!")


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
