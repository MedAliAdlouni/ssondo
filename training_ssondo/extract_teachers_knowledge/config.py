"""Config file to extract the features (logits, embeddings) of a model"""

import os

# slurm = "SLURM_JOB_ID" in os.environ

common_parameters = {
    "process": {
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "pin_memory": False,
    },
    "save_dir": os.path.join(
        os.environ["DATA"],
        "teachers_knowledge",
    ),
}

conf = {

# ----------------------- Teacher: MATPAC_MCL ------------------------------------------------------------
    "matpac_mcl_train": {
        "dataset": {
            "name": "AudioSet",
            "set": "train",
            "audio_duration": None,
            "batch_size": 8,
            "shuffle": False,
            "drop_last": False
        },
        "model": {
            "name": "MATPAC_MCL",
            "sr": 16000,
            "pull_time_dimension": True,
            "feature_type": "embed",  # embed, all
            "ckpt_path": os.path.join("models",
                                      "teachers",
                                      "MATPAC_MCL",
                                      "matpac_plus_6s_2048_enconly.pt"),
        },
        "slice_audio": {
            "win_len": 10,
            "step_size": 10,
            "add_last": True,
        },
    },

    "matpac_mcl_eval": {
        "dataset": {
            "name": "AudioSet",
            "set": "eval",
            "audio_duration": None,
            "batch_size": 8,
            "shuffle": False,
            "drop_last": False
        },
        "model": {
            "name": "MATPAC_MCL",
            "sr": 16000,
            "pull_time_dimension": True,
            "feature_type": "embed",  # embed, all
            "ckpt_path": os.path.join("models",
                                      "teachers",
                                      "MATPAC_MCL",
                                      "matpac_plus_6s_2048_enconly.pt"),
        },
        "slice_audio": {
            "win_len": 10,
            "step_size": 10,
            "add_last": True,
        },
    },

# ----------------------- Teacher: M2D ------------------------------------------------------------
    "m2d_train": {
        "dataset": {
            "name": "AudioSet",
            "set": "train",
            "audio_duration": None,
            "batch_size": 4,
            "shuffle": False,
            "drop_last": False
        },
        "model": {
            "name": "M2D",
            "sr": 16000,
            "pull_time_dimension": True,
            "feature_type": "embed",  # embed, all
            "ckpt_path": os.path.join("models",
                                      "teachers",
                                      "M2D",
                                      "m2d_vit_base-80x608p16x16-221006-mr7_enconly",
                                      "M2D_ssl.pth"),
        },
        "slice_audio": {
            "win_len": 10,
            "step_size": 10,
            "add_last": True,
        },
    },

    "m2d_eval": {
        "dataset": {
            "name": "AudioSet",
            "set": "eval",
            "audio_duration": None,
            "batch_size": 4,
            "shuffle": False,
            "drop_last": False
        },
        "model": {
            "name": "M2D",
            "sr": 16000,
            "pull_time_dimension": True,
            "feature_type": "embed",  # embed, all
            "ckpt_path": os.path.join("models",
                                      "teachers",
                                      "M2D",
                                      "m2d_vit_base-80x608p16x16-221006-mr7_enconly",
                                      "M2D_ssl.pth"),
        },
        "slice_audio": {
            "win_len": 10,
            "step_size": 10,
            "add_last": True,
        },
    },


}
