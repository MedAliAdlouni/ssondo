"""Utility functions for the clustering training."""

# Standard library imports
import os
import json
import pickle
import yaml
from typing import Tuple

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.manifold import TSNE
import umap.umap_ as umap
import psutil


def merge_dicts(dict1, dict2) -> dict:
    """Recursively merge two dictionaries, with values from dict2 taking precedence."""
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


# -----------------------  01_learn_kmeans.py  ----------------------------------------------
def plot_inertia(inertia_history, conf) -> None:
    """
    Save inertia history and plot it.

    Parameters
    ----------
    inertia_history : list or np.ndarray
        List of inertia values recorded after each batch.
    conf : dict
        Configuration dictionary, must contain keys 'exp_dir', 'n_clusters', and 'job_id'.
    """
    inertia_history = np.array(inertia_history)
    exp_dir = os.path.join(conf["exp_dir"], f"{conf['n_clusters']}_clusters")
    os.makedirs(exp_dir, exist_ok=True)

    # Save inertia history
    inertia_path = os.path.join(exp_dir, "inertia_history.npy")
    np.save(inertia_path, inertia_history)
    print(f"Saved inertia history to: {inertia_path}")

    # Plot inertia
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, len(inertia_history) + 1), inertia_history, marker="o", linestyle="-"
    )
    plt.xlabel("Batch Number")
    plt.ylabel("Inertia")
    plt.title("MiniBatchKMeans Inertia over Batches")
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(exp_dir, "inertia_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved inertia plot to: {plot_path}")


def save_clustering_results(conf, kmeans_model, inertia_history) -> None:
    """Save clustering results and metrics.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    kmeans_model : MiniBatchKMeans
        Trained clustering model.
    metrics : dict
        Clustering metrics.
    """
    exp_dir = os.path.join(conf["exp_dir"], f"{conf['n_clusters']}_clusters")
    os.makedirs(exp_dir, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(exp_dir, "kmeans_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(kmeans_model, f)
    print(f"Saved model to: {model_path}")

    # Save cluster centers
    centers_path = os.path.join(exp_dir, "cluster_centers.npy")
    np.save(centers_path, kmeans_model.cluster_centers_)
    print(f"Saved cluster centers to: {centers_path}")

    # Save inertia history
    inertia_path = os.path.join(exp_dir, "inertia_history.csv")
    np.savetxt(inertia_path, np.array(inertia_history), delimiter=",")
    print(f"Saved inertia history to: {inertia_path}")

    # Save configuration
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)
    print(f"Saved config to: {config_path}")


# -----------------------  02_label_prediction.py  ----------------------------------------------
def save_labels(df, conf) -> None:
    """Save clustering results and metrics.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    df : pd.DataFrame
        DataFrame with audio_id and cluster_id columns.
    """
    exp_dir = os.path.join(conf["exp_dir"], f"{conf['n_clusters']}_clusters")
    os.makedirs(exp_dir, exist_ok=True)

    # Save ids and labels
    cluster_labels_path = os.path.join(exp_dir, "predicted_labels.csv")
    df.to_csv(cluster_labels_path, index=False)


def log_mem(i) -> None:
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1e9  # Resident Set Size in GB
    mem = psutil.virtual_memory()
    total = mem.total / 1e9
    available = mem.available / 1e9
    print(
        f"[MEM] Batch {i:03d} RAM used: {rss:.2f} GB | Available: {available:.2f} GB / Total: {total:.2f} GB"
    )


# -----------------------  03_compute_metrics.py  ----------------------------------------------
def sample_embeddings_and_labels(
    train_dataset, df, sample_size
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = len(train_dataset)
    sample_size = min(sample_size, n_samples)
    sampled_indices = np.random.choice(n_samples, sample_size, replace=False)

    sampled_embeddings = []
    sampled_labels = []

    label_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    for idx in sampled_indices:
        sample_id, sample_embedding = train_dataset[idx]
        sampled_embeddings.append(sample_embedding)
        sampled_labels.append(label_dict[sample_id])

    sampled_embeddings = np.stack(sampled_embeddings)
    sampled_labels = np.array(sampled_labels)

    return sampled_embeddings, sampled_labels


def compute_clustering_metrics(train_dataset, df, conf) -> dict:
    """Compute clustering evaluation metrics on a sample of the data.

    Parameters
    ----------
    train_dataset : TeacherKnowledgeDataset dataset. __getitem__ returns back file_path[:-4], embedding
        Input embeddings (shape: [num_samples, num_features]).
    df : panda dataframe
        It has 2 columns. First is for IDs of embeddings (full paths). Second is for the cluster labels
        Cluster labels (shape: [num_samples]).
    n_clusters : int
        Number of clusters.
    sample_size : int, optional
        Number of samples to randomly select for metric computation.

    Returns
    -------
    dict
        Dictionary containing clustering metrics.
    """
    metrics = {}

    n_samples = len(train_dataset)
    sample_size = min(conf["clustering_metrics"]["sample_size"], n_samples)

    sampled_embeddings, sampled_labels = sample_embeddings_and_labels(
        train_dataset, df, sample_size
    )

    print("Computing clustering metrics...")
    if conf["n_clusters"] > 1:
        if conf["n_clusters"] < len(sampled_embeddings):
            print("Computing silhouette score...")
            metrics["silhouette_score"] = float(
                silhouette_score(sampled_embeddings, sampled_labels)
            )

        print("Computing Calinski-Harabasz score...")
        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(sampled_embeddings, sampled_labels)
        )

        print("Computing Davies-Bouldin score...")
        metrics["davies_bouldin_score"] = float(
            davies_bouldin_score(sampled_embeddings, sampled_labels)
        )

    return metrics


def save_clustering_metrics(conf, metrics) -> None:
    """Save clustering results and metrics.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    metrics : np.ndarray
        Clustering metrics.
    """
    exp_dir = os.path.join(conf["exp_dir"], f"{conf['n_clusters']}_clusters")
    os.makedirs(exp_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


def save_tsne_data(train_dataset, df, conf) -> None:
    print("Sampling embeddings for t-SNE...")
    sample_size = conf["visualization"]["tsne"]["n_samples"]
    sampled_embeddings, sampled_labels = sample_embeddings_and_labels(
        train_dataset, df, sample_size
    )

    tsne = TSNE(
        n_components=conf["visualization"]["tsne"]["n_components"],
        perplexity=conf["visualization"]["tsne"]["perplexity"],
        random_state=conf["seed"],
    )
    print("Fit transform tSNE on sampled embeddings ...")
    reduced = tsne.fit_transform(sampled_embeddings)

    # Create DataFrame with reduced dimensions and labels
    tsne_df = pd.DataFrame(
        {
            "tsne_x": reduced[:, 0],
            "tsne_y": reduced[:, 1],
            "cluster_label": sampled_labels,
        }
    )

    exp_dir = os.path.join(conf["exp_dir"], f"{conf['n_clusters']}_clusters")
    os.makedirs(exp_dir, exist_ok=True)
    tsne_path = os.path.join(exp_dir, "tsne_data.csv")

    tsne_df.to_csv(tsne_path, index=False)
    print(f"Saved t-SNE data to: {tsne_path}")


def save_umap_data(train_dataset, df, conf) -> None:
    print("Sampling embeddings for UMAP...")
    sample_size = conf["visualization"]["umap"]["n_samples"]
    sampled_embeddings, sampled_labels = sample_embeddings_and_labels(
        train_dataset, df, sample_size
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=conf["visualization"]["umap"]["n_neighbors"],
        min_dist=conf["visualization"]["umap"]["min_dist"],
        random_state=conf["seed"],
    )
    print("Fit transform UMAP on sampled embeddings ...")
    reduced = reducer.fit_transform(sampled_embeddings)

    # Create DataFrame with reduced dimensions and labels
    umap_df = pd.DataFrame(
        {
            "umap_x": reduced[:, 0],
            "umap_y": reduced[:, 1],
            "cluster_label": sampled_labels,
        }
    )

    exp_dir = os.path.join(conf["exp_dir"], f"{conf['n_clusters']}_clusters")
    os.makedirs(exp_dir, exist_ok=True)
    umap_path = os.path.join(exp_dir, "umap_data.csv")

    umap_df.to_csv(umap_path, index=False)
    print(f"Saved UMAP data to: {umap_path}")
