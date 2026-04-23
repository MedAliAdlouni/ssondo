"Config file to run teacher AudioSet embeddings clustering experiments."

import os
import string
import random

from training_ssondo import DATA, OUTPUTS

slurm = "SLURM_JOB_ID" in os.environ

if slurm:
  JOB_ID = os.environ["SLURM_JOB_ID"]
else:
  JOB_ID = "".join(random.choices(
      string.ascii_letters + string.digits, k=8))

common_parameters = {
    # Experiment directories
    "exp_dir": os.path.join(
        OUTPUTS,
        "clustering",
        "Placeholder_for_teacher_model_name",
    ),

    # Dataset settings
    "dataset": {
        "teacher_knowledge_path": os.path.join(
            DATA,
            "teachers_knowledge",
            "MATPAC_MCL",
            "window_length_10s",
            "embed",
        ),
        "subset": "all",          # Dataset subset: "train", "eval", or "all"
        "sample_size": 1,        # Fraction of dataset used if using fit()
    },

    # Training options


    # SLURM job id
    "job_id": JOB_ID,

    # Reproducibility
    "seed": 42,

    # Clustering parameters
    # Number of clusters (uncomment and set as needed)
    "n_clusters": 50,
    # Initialization method: "k-means++", "random", or array-like
    "init_method": "k-means++",
    "max_iter": 100,               # Maximum iterations per run
    "batch_size": 10000,           # Mini-batch size for partial_fit
    # Verbosity level, either 0 for non-verbosity or 1 for verbosity
    "verbose": 1,
    "compute_labels": True,        # Compute labels & inertia after convergence
    "tol": 0,                     # Tolerance for early stopping (0 disables)
    "max_no_improvement": 100,     # Early stopping patience
    "init_size": None,             # Size of init sample; heuristic if None
    "n_init": 10,                  # Number of random initializations
    "reassignment_ratio": 0.0,     # Fraction of centers to reassign
    "early_stopping": False,       # Whether to stop early when converged


    # Clustering evaluation metrics
    "clustering_metrics": {
        "sample_size": 2000,      # Number of samples for metric computation
    },

    # Visualization parameters
    "visualization": {
        "tsne": {
            "perplexity": 30,
            "n_samples": 2000,   # Number of samples for t-SNE
            "n_components": 2,
        },
        "umap": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_samples": 2000,  # Number of samples for UMAP
        },
    },
}


conf = {

# ----------------------- Teacher: MATPAC++  ------------------------------------------------------------
    "10_clusters_fit_matpac": {
        "n_clusters": 10,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "MATPAC_MCL",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "MATPAC_MCL",
        ),
    },

    "25_clusters_fit_matpac": {
        "n_clusters": 25,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "MATPAC_MCL",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "MATPAC_MCL",
        ),
    },

    "50_clusters_fit_matpac": {
        "n_clusters": 50,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "MATPAC_MCL",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "MATPAC_MCL",
        ),
    },

    "100_clusters_fit_matpac": {
        "n_clusters": 100,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "MATPAC_MCL",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "MATPAC_MCL",
        ),
    },

    "200_clusters_fit_matpac": {
        "n_clusters": 200,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "MATPAC_MCL",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "MATPAC_MCL",
        ),
    },



# ----------------------- Teacher: M2D ------------------------------------------------------------
    "10_clusters_fit_m2d": {
        "n_clusters": 10,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "M2D",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "M2D",
        ),
    },

    "25_clusters_fit_m2d": {
        "n_clusters": 25,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "M2D",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "M2D",
        ),
    },

    "50_clusters_fit_m2d": {
        "n_clusters": 50,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "M2D",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "M2D",
        ),
    },

    "100_clusters_fit_m2d": {
        "n_clusters": 100,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "M2D",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "M2D",
        ),
    },

    "200_clusters_fit_m2d": {
        "n_clusters": 200,
        "use_fit": True,

        "dataset": {
            "teacher_knowledge_path": os.path.join(
                DATA,
                "teachers_knowledge",
                "M2D",
                "window_length_10s",
                "embed",
            ),
            "subset": "all",          # Dataset subset: "train", "eval", or "all"
            "sample_size": 1,        # Fraction of dataset used if using fit()
        },

        "exp_dir": os.path.join(
            OUTPUTS,
            "clustering",
            "M2D",
        ),
    },

}