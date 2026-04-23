"Config file to run teacher AudioSet embeddings clustering experiments."

import os
import string
import random

from training_ssondo import DATA, OUTPUTS

slurm = "SLURM_JOB_ID" in os.environ

if slurm:
    JOB_ID = os.environ["SLURM_JOB_ID"]
else:
    JOB_ID = "".join(random.choices(string.ascii_letters + string.digits, k=8))

common_parameters = {
    "exp_dir": os.path.join(OUTPUTS, "clustering"),
    "dataset": {
        "teacher_knowledge_path": os.path.join(
            DATA,
            "teachers_knowledge",
            "MATPAC_MCL",
            "window_length_10s",
            "embed",
        ),
        "subset": "all",
        "sample_size": 1,
    },
    "job_id": JOB_ID,
    "seed": 42,
    # Clustering parameters
    "n_clusters": 50,
    "init_method": "k-means++",
    "max_iter": 100,
    "batch_size": 10000,
    "verbose": 1,
    "compute_labels": True,
    "tol": 0,
    "max_no_improvement": 100,
    "init_size": None,
    "n_init": 10,
    "reassignment_ratio": 0.0,
    "early_stopping": False,
    "clustering_metrics": {
        "sample_size": 2000,
    },
    "visualization": {
        "tsne": {
            "perplexity": 30,
            "n_samples": 2000,
            "n_components": 2,
        },
        "umap": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_samples": 2000,
        },
    },
}


def _make_cluster_conf(n_clusters: int, teacher: str) -> dict:
    """Generate a clustering config for a given cluster count and teacher model."""
    return {
        "n_clusters": n_clusters,
        "use_fit": True,
        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                teacher,
                "window_length_10s",
                "embed",
            ),
            "subset": "all",
            "sample_size": 1,
        },
        "exp_dir": os.path.join(OUTPUTS, "clustering", teacher),
    }


_CLUSTER_COUNTS = [10, 25, 50, 100, 200]
_TEACHERS = {"matpac": "MATPAC_MCL", "m2d": "M2D"}

conf = {
    f"{n}_clusters_fit_{tag}": _make_cluster_conf(n, model)
    for tag, model in _TEACHERS.items()
    for n in _CLUSTER_COUNTS
}
