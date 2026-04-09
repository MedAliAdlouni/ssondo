"Config file to run training experiments."


import os
import string
import random

slurm = "SLURM_JOB_ID" in os.environ

if slurm:
  JOB_ID = os.environ["SLURM_JOB_ID"]
else:
  JOB_ID = "".join(random.choices(
      string.ascii_letters + string.digits, k=8))

common_parameters = {
    "exp_dir": os.path.join(
        os.environ["OUTPUTS"],
        "knowledge_distillation",
        "Placeholder_for_teacher_model_name"
        "Placeholder_for_student_model_name"),

    "cluster_dir": os.path.join(
        os.environ["OUTPUTS"],
        "clustering"),

    "process": {
        "num_workers": 8,
        "prefetch": 3,
        "devices": int(
            os.environ["SLURM_GPUS_ON_NODE"]) if "SLURM_GPUS_ON_NODE" in os.environ else 1,
        "num_nodes": int(
            os.environ.get("SLURM_NNODES", 1)),
        "precision": "16-mixed",
        "persistent_workers": True,
        "pin_memory": True,
    },

    "job_id": JOB_ID,
    "seed": 42,

    # data preprocessing
    "preprocess": {
        "slice_audio": {
            "win_len": 10,    # 10 second
            "step_size": 10,  # no overlap
        },
        "logmelspec": {
            "win_len": 0.025,  # 800 frames
            "hop_len": 0.01,   # 320 frames
            "n_mels": 128,     # 128 mel bands
            "f_min": 0,        # 0 Hz
            "f_max": None,     # None, 16 kHz
        },
        "normalize": {
            "mean": 0.0,
            "std": 1.0,
        },
    },

    "dataset": {
        "name": "audioset",
        "n_classes": 527,

        # Sampler or split
        # "RandomSampler" or "WeightedRandomSampler" ou "WeightedRandomSamplerSSL"
        "sampler": "RandomSampler",
        "sampler_args": {
            "num_samples": 4,
            "replacement": True,
            "n_clusters": 50,  # number of clusters for WeightedRandomSamplerSSL
        },
        "train_shuffle": False,  # necessarily False
    },

    # batch size
    "batch_size": 2,

    # max_epochs
    "epochs": 1,

    # optimizer
    "optimizer": "Adam",
    "optimizer_args": {
        "lr": 8e-4,  # default value: if scheduler 8e-4, else 1e-3
        "betas": (0.9, 0.999),
        "weight_decay": 0,
    },

    # learning rate scheduler
    "lr_scheduler": "CustomScheduler",
    "lr_scheduler_args": {"warm_up_len": 8, "ramp_down_start": 80,
                          "ramp_down_len": 95, "last_lr_value": 0.01},

    # early_stopping
    "early_stopping": False,
    # "early_stopping_args": {
    #     "patience": 10,
    # },

    # data augmentation
    "data_augmentation": {
        "mixup": False,
        "mixup_args": {
            "alpha": 0.3,
        },

        "spec_augment": False,
    },

    # "teacher_model_name": "MATPAC_MCL",
}

conf = {
# ----------- baseline, Supervised Learning, no Knowledge Distillation ----------------------------------------------------------
    "baseline_mn_weighted_random_sampling": {
        "exp_dir": os.path.join(
            os.environ["OUTPUTS"],
            "knowledge_distillation",
            "baseline",
            "MobileNetV3"),
  
        "trainer": {
            "val_check_interval": None,    # default value
            "check_val_every_n_epoch": 1,  # default value
            "num_sanity_val_steps": None,  # default value
        },

        "preprocess": {
            # audible settings
            "logmelspec": {
                "win_len": 0.032,  # 1024 frames
                "hop_len": 0.016,  # 50% overlap, 512 frames
                "n_mels": 128,     # 128 mel bands
                "f_min": 50,       # 50 Hz
                "f_max": 16000,    # 16 kHz
            },
            "slice_audio": {
                "win_len": 10,      # 10 seconds -> no slice
                "step_size": 10,    # no overlap
            },
        },

        "dataset": {
            "teacher_knowledge_path": None,  # Sampler or split
            "sampler": "WeightedRandomSampler",  # "RandomSampler" or "WeightedRandomSampler"
            "sampler_args": {
                "num_samples": 100,
                "replacement": True,
            },
        },

        "prediction_loss": "BCEWithLogits",  # "FocalLoss", "BCEWithLogits" or None

        "student_model": {
            "model_name": "mn10_im",
            "sr": 32000,

            "pretrained_name": "mn10_im.pt",
            "width_mult": 1.0,
            "reduced_tail": False,
            "dilated": False,
            "strides": (2, 2, 2, 2),
            "relu_only": False,
            "input_dim_f": 128,
            "input_dim_t": 998,
            "se_dims": "c",
            "se_agg": "max",
            "se_r": 4,
        },
        "classification_head": {
            "head_type": "mlp",
            "n_classes": 527,
            "pooling": "mean",
            "activation_att": None,
            "last_activation": "",
        },
        "knowledge_distillation": {
            "temperature": 1,
            "lambda": 1,             # just prediction_loss :  lam * pred_loss + (1 - lam) * kd_loss # nopep8
            "loss": None             # "L1", "MSE", "BCEWithLogits" or none
        },

        "data_augmentation": {
        "mixup": True,
        "mixup_args": {
            "alpha": 0.3,
        },

        "spec_augment": False,
        },

        # To continue training from a checkpoint
        "checkpoint_path": None,
    },

# --------------------------- TEACHER MODEL : MATPAC_MCL ----------------------------------------------------------
# --------------------------- Student model : MobileNetV3 ---------------------------------------------------------
    "matpac_mn_cosine_random": {
        "exp_dir": os.path.join(
            os.environ["OUTPUTS"],
            "knowledge_distillation",
            "MATPAC_MCL",
            "MobileNetV3"),

        "trainer": {
            # "debug": {
            #     "lmt_train_bt": 0.0001,
            #     "lmt_val_bt": 0.005,
            # },

            "val_check_interval": None,    # default value
            "check_val_every_n_epoch": 1,  # default value
            "num_sanity_val_steps": None,  # default value
        },

        "preprocess": {
            # audible settings
            "logmelspec": {
                "win_len": 0.032,  # 1024 frames
                "hop_len": 0.016,  # 50% overlap, 512 frames
                "n_mels": 128,     # 128 mel bands
                "f_min": 50,       # 50 Hz
                "f_max": 16000,    # 16 kHz
            },
            "slice_audio": {
                "win_len": 10,      # 10 seconds -> no slice
                "step_size": 10,    # no overlap
            },
        },

        "dataset": {
            "teacher_knowledge_path":
                os.path.join(os.environ["DATA"],
                             "teachers_knowledge",
                             "MATPAC_MCL",
                             "window_length_10s",
                             "embed"),

            "sampler": "RandomSampler",
        },

        "prediction_loss": None,  # "FocalLoss", "BCEWithLogits" or None

        "student_model": {
            "model_name": "mn10_im",
            "sr": 32000,

            "pretrained_name": "mn10_im.pt",
            "width_mult": 1.0,
            "reduced_tail": False,
            "dilated": False,
            "strides": (2, 2, 2, 2),
            "relu_only": False,
            "input_dim_f": 128,
            "input_dim_t": 998,
            "se_dims": "c",
            "se_agg": "max",
            "se_r": 4,
        },
        "classification_head": {
            "head_type": "mlp",
            "n_classes": 3840,
            "pooling": "mean",
            "activation_att": None,
            "last_activation": "",
        },
        "knowledge_distillation": {
            "temperature": 0.0,  # temperature for KLDivLoss
            "lambda": 0,              # just kd_loss :  lam * pred_loss + (1 - lam) * kd_loss # nopep8
            "loss": "cosine_similarity",             # "L1", "MSE", "BCEWithLogits" or None
        },

        # To continue training from a checkpoint
        "checkpoint_path": None,
    },

    "matpac_mn_cosine_50c": {  # USING CLUSTERS FROM MATPAC MOBILENET
        "exp_dir": os.path.join(
            os.environ["OUTPUTS"],
            "knowledge_distillation",
            "MATPAC_MCL",
            "MobileNetV3"),

        "trainer": {
            # "debug": {
            #     "lmt_train_bt": 0.0001,
            #     "lmt_val_bt": 0.005,
            # },

            "val_check_interval": None,    # default value
            "check_val_every_n_epoch": 1,  # default value
            "num_sanity_val_steps": None,  # default value
        },

        "cluster_dir": os.path.join(
            os.environ["OUTPUTS"],
            "clustering",
            "MATPAC_MCL"),

        "preprocess": {
            # audible settings
            "logmelspec": {
                "win_len": 0.032,  # 1024 frames
                "hop_len": 0.016,  # 50% overlap, 512 frames
                "n_mels": 128,     # 128 mel bands
                "f_min": 50,       # 50 Hz
                "f_max": 16000,    # 16 kHz
            },
            "slice_audio": {
                "win_len": 10,      # 10 seconds -> no slice
                "step_size": 10,    # no overlap
            },
        },

        "dataset": {
            "teacher_knowledge_path":
                os.path.join(os.environ["DATA"],
                             "teachers_knowledge",
                             "MATPAC_MCL",
                             "window_length_10s",
                             "embed"),
            
            "sampler": "WeightedRandomSamplerSSL",

            "sampler_args": {
                "n_clusters": 50,
                # use kmeans.fit() or kmeans.pfit() to train the clustering model with MiniBatchKMeans
                "use_fit": True,
            },
        },

        "prediction_loss": None,  # "FocalLoss", "BCEWithLogits" or None

        "student_model": {
            "model_name": "mn10_im",
            "sr": 32000,

            "pretrained_name": "mn10_im.pt",
            "width_mult": 1.0,
            "reduced_tail": False,
            "dilated": False,
            "strides": (2, 2, 2, 2),
            "relu_only": False,
            "input_dim_f": 128,
            "input_dim_t": 998,
            "se_dims": "c",
            "se_agg": "max",
            "se_r": 4,
        },
        "classification_head": {
            "head_type": "mlp",
            "n_classes": 3840,
            "pooling": "mean",
            "activation_att": None,
            "last_activation": "",
        },
        "knowledge_distillation": {
            "temperature": 0.0,  # temperature for KLDivLoss
            "lambda": 0,              # just kd_loss :  lam * pred_loss + (1 - lam) * kd_loss # nopep8
            "loss": "cosine_similarity",             # "L1", "MSE", "BCEWithLogits" or None
        },

        # To continue training from a checkpoint
        "checkpoint_path": None,
    },
# --------------------------- Student model : ERes2Net ----------------------------------------------------------
    "matpac_eres2net_cosine_50c": {  # USING CLUSTERS FROM MATPAC MOBILENET
        "exp_dir": os.path.join(
            os.environ["OUTPUTS"],
            "knowledge_distillation",
            "MATPAC_MCL",
            "ERes2Net"),

        "trainer": {
            # "debug": {
            #     "lmt_train_bt": 0.0001,
            #     "lmt_val_bt": 0.005,
            # },

            "val_check_interval": None,    # default value
            "check_val_every_n_epoch": 1,  # default value
            "num_sanity_val_steps": None,  # default value
        },

        "cluster_dir": os.path.join(
            os.environ["OUTPUTS"],
            "clustering",
            "MATPAC_MCL"),

        "preprocess": {
            # audible settings
            "logmelspec": {
                "win_len": 0.032,  # 1024 frames
                "hop_len": 0.016,  # 50% overlap, 512 frames
                "n_mels": 128,     # 128 mel bands
                "f_min": 50,       # 50 Hz
                "f_max": 16000,    # 16 kHz
            },
            "slice_audio": {
                "win_len": 10,      # 10 seconds -> no slice
                "step_size": 10,    # no overlap
            },
        },

        "dataset": {
            "teacher_knowledge_path":
                os.path.join(os.environ["DATA"],
                             "teachers_knowledge",
                             "MATPAC_MCL",
                             "window_length_10s",
                             "embed"),

            "sampler": "WeightedRandomSamplerSSL",

            "sampler_args": {
                "n_clusters": 50,
                # use kmeans.fit() or kmeans.pfit() to train the clustering model with MiniBatchKMeans
                "use_fit": True,
            },
        },

        "prediction_loss": None,  # "FocalLoss", "BCEWithLogits" or None

        "student_model": {
            "model_name": "ERes2Net",
            "sr": 32000,
            "m_channels": 16,
            "feat_dim": 128,
            "num_blocks": [3, 4, 6, 3, 3],
            "pooling_func": "TSTP",
            "add_layer": True,
        },
        "classification_head": {
            "hidden_features_size": 1280,
            "head_type": "mlp",
            "n_classes": 3840,
            "pooling": "mean",
            "activation_att": None,
            "last_activation": "",
        },
        "knowledge_distillation": {
            "temperature": 0.0,  # temperature for KLDivLoss
            "lambda": 0,              # just kd_loss :  lam * pred_loss + (1 - lam) * kd_loss # nopep8
            "loss": "cosine_similarity",             # "L1", "MSE", "BCEWithLogits" or None
        },

        # To continue training from a checkpoint
        "checkpoint_path": None,
    },
# --------------------------- Student model : DyMN ----------------------------------------------------------
    "matpac_dymn_cosine_50c": {  # USING CLUSTERS FROM MATPAC MOBILENET
        "exp_dir": os.path.join(
            os.environ["OUTPUTS"],
            "knowledge_distillation",
            "MATPAC_MCL",
            "DyMN"),

        "trainer": {
            # "debug": {
            #     "lmt_train_bt": 0.0001,
            #     "lmt_val_bt": 0.005,
            # },

            "val_check_interval": None,    # default value
            "check_val_every_n_epoch": 1,  # default value
            "num_sanity_val_steps": None,  # default value
        },

        "cluster_dir": os.path.join(
            os.environ["OUTPUTS"],
            "clustering",
            "MATPAC_MCL"),

        "preprocess": {
            # audible settings
            "logmelspec": {
                "win_len": 0.032,  # 1024 frames
                "hop_len": 0.016,  # 50% overlap, 512 frames
                "n_mels": 128,     # 128 mel bands
                "f_min": 50,       # 50 Hz
                "f_max": 16000,    # 16 kHz
            },
            "slice_audio": {
                "win_len": 10,      # 10 seconds -> no slice
                "step_size": 10,    # no overlap
            },
        },

        "dataset": {
            "teacher_knowledge_path":
                os.path.join(os.environ["DATA"],
                             "teachers_knowledge",
                             "MATPAC_MCL",
                             "window_length_10s",
                             "embed"),

            "sampler": "WeightedRandomSamplerSSL",

            "sampler_args": {
                "n_clusters": 50,
                # use kmeans.fit() or kmeans.pfit() to train the clustering model with MiniBatchKMeans
                "use_fit": True,
            },
        },

        "prediction_loss": None,  # "FocalLoss", "BCEWithLogits" or None

        "student_model": {
            "model_name": "dymn10_im",
            "sr": 32000,
            "width_mult": 1,
            "pretrained_name": "dymn10_im",
            "strides": (2, 2, 2, 2),
            "pretrain_final_temp": 30.0,
        },
        "classification_head": {
            "hidden_features_size": 1280,
            "head_type": "mlp",
            "n_classes": 3840,
            "pooling": "mean",
            "activation_att": None,
            "last_activation": "",
        },
        "knowledge_distillation": {
            "temperature": 0.0,  # temperature for KLDivLoss
            "lambda": 0,              # just kd_loss :  lam * pred_loss + (1 - lam) * kd_loss # nopep8
            "loss": "cosine_similarity",             # "L1", "MSE", "BCEWithLogits" or None
        },

        # To continue training from a checkpoint
        "checkpoint_path": None,
    },

# --------------------------- TEACHER MODEL : M2D ----------------------------------------------------------
# --------------------------- Student model : MobileNetV3 ----------------------------------------------------------
    "m2d_mn_cosine_50c": {
        "exp_dir": os.path.join(
            os.environ["OUTPUTS"],
            "knowledge_distillation",
            "M2D",
            "MobileNetV3"),

            "trainer": {
            # "debug": {
            #     "lmt_train_bt": 0.0001,
            #     "lmt_val_bt": 0.005,
            # },

            "val_check_interval": None,    # default value
            "check_val_every_n_epoch": 1,  # default value
            "num_sanity_val_steps": None,  # default value
        },

        "preprocess": {
            # audible settings
            "logmelspec": {
                "win_len": 0.032,  # 1024 frames
                "hop_len": 0.016,  # 50% overlap, 512 frames
                "n_mels": 128,     # 128 mel bands
                "f_min": 50,       # 50 Hz
                "f_max": 16000,    # 16 kHz
            },
            "slice_audio": {
                "win_len": 10,      # 10 seconds -> no slice
                "step_size": 10,    # no overlap
            },
        },

        "dataset": {
            "teacher_knowledge_path":
                os.path.join(os.environ["DATA"],
                             "teachers_knowledge",
                             "M2D",
                             "window_length_10s",
                             "embed"),

            "sampler": "WeightedRandomSamplerSSL",

            "sampler_args": {
                "n_clusters": 50,
                # use kmeans.fit() or kmeans.pfit() to train the clustering model with MiniBatchKMeans
                "use_fit": True,
            },
        },

        "prediction_loss": None,  # "FocalLoss", "BCEWithLogits" or None

        "student_model": {
            "model_name": "mn10_im",
            "sr": 32000,

            "pretrained_name": "mn10_im.pt",
            "width_mult": 1.0,
            "reduced_tail": False,
            "dilated": False,
            "strides": (2, 2, 2, 2),
            "relu_only": False,
            "input_dim_f": 128,
            "input_dim_t": 998,
            "se_dims": "c",
            "se_agg": "max",
            "se_r": 4,
        },
        "classification_head": {
            "head_type": "mlp",
            "n_classes": 3840,
            "pooling": "mean",
            "activation_att": None,
            "last_activation": "",
        },
        "knowledge_distillation": {
            "temperature": 0.0,  # temperature for KLDivLoss
            "lambda": 0,              # just kd_loss :  lam * pred_loss + (1 - lam) * kd_loss # nopep8
            "loss": "cosine_similarity",             # "L1", "MSE", "BCEWithLogits" or None
        },

        # To continue training from a checkpoint
        "checkpoint_path": None,
    },
# --------------------------- Student model : ERes2Net ----------------------------------------------------------
    "m2d_eres2net_cosine_50c": {
        "exp_dir": os.path.join(
            os.environ["OUTPUTS"],
            "knowledge_distillation",
            "M2D",
            "ERes2Net"),

        "trainer": {
            # "debug": {
            #     "lmt_train_bt": 0.0001,
            #     "lmt_val_bt": 0.005,
            # },

            "val_check_interval": None,    # default value
            "check_val_every_n_epoch": 1,  # default value
            "num_sanity_val_steps": None,  # default value
        },

        "cluster_dir": os.path.join(
            os.environ["OUTPUTS"],
            "clustering",
            "M2D"),

        "preprocess": {
            # audible settings
            "logmelspec": {
                "win_len": 0.032,  # 1024 frames
                "hop_len": 0.016,  # 50% overlap, 512 frames
                "n_mels": 128,     # 128 mel bands
                "f_min": 50,       # 50 Hz
                "f_max": 16000,    # 16 kHz
            },
            "slice_audio": {
                "win_len": 10,      # 10 seconds -> no slice
                "step_size": 10,    # no overlap
            },
        },

        "dataset": {
            "teacher_knowledge_path":
                os.path.join(os.environ["DATA"],
                             "teachers_knowledge",
                             "M2D",
                             "window_length_10s",
                             "embed"),

            "sampler": "WeightedRandomSamplerSSL",

            "sampler_args": {
                "n_clusters": 50,
                # use kmeans.fit() or kmeans.pfit() to train the clustering model with MiniBatchKMeans
                "use_fit": True,
            },
        },

        "prediction_loss": None,  # "FocalLoss", "BCEWithLogits" or None

        "student_model": {
            "model_name": "ERes2Net",
            "sr": 32000,
            "m_channels": 16,
            "feat_dim": 128,
            "num_blocks": [3, 4, 6, 3, 3],
            "pooling_func": "TSTP",
            "add_layer": True,
        },
        "classification_head": {
            "hidden_features_size": 1280,
            "head_type": "mlp",
            "n_classes": 3840,
            "pooling": "mean",
            "activation_att": None,
            "last_activation": "",
        },
        "knowledge_distillation": {
            "temperature": 0.0,  # temperature for KLDivLoss
            "lambda": 0,              # just kd_loss :  lam * pred_loss + (1 - lam) * kd_loss # nopep8
            "loss": "cosine_similarity",             # "L1", "MSE", "BCEWithLogits" or None
        },

        # To continue training from a checkpoint
        "checkpoint_path": None,
    },
# --------------------------- Student model : DyMN ----------------------------------------------------------
    "m2d_dymn_cosine_50c": {
        "exp_dir": os.path.join(
            os.environ["OUTPUTS"],
            "knowledge_distillation",
            "M2D",
            "DyMN"),

        "trainer": {
            # "debug": {
            #     "lmt_train_bt": 0.0001,
            #     "lmt_val_bt": 0.005,
            # },

            "val_check_interval": None,    # default value
            "check_val_every_n_epoch": 1,  # default value
            "num_sanity_val_steps": None,  # default value
        },

        "cluster_dir": os.path.join(
            os.environ["OUTPUTS"],
            "clustering",
            "M2D"),

        "preprocess": {
            # audible settings
            "logmelspec": {
                "win_len": 0.032,  # 1024 frames
                "hop_len": 0.016,  # 50% overlap, 512 frames
                "n_mels": 128,     # 128 mel bands
                "f_min": 50,       # 50 Hz
                "f_max": 16000,    # 16 kHz
            },
            "slice_audio": {
                "win_len": 10,      # 10 seconds -> no slice
                "step_size": 10,    # no overlap
            },
        },

        "dataset": {
            "teacher_knowledge_path":
                os.path.join(os.environ["DATA"],
                             "teachers_knowledge",
                             "M2D",
                             "window_length_10s",
                             "embed"),

            "sampler": "WeightedRandomSamplerSSL",

            "sampler_args": {
                "n_clusters": 50,
                # use kmeans.fit() or kmeans.pfit() to train the clustering model with MiniBatchKMeans
                "use_fit": True,
            },
        },

        "prediction_loss": None,  # "FocalLoss", "BCEWithLogits" or None

        "student_model": {
            "model_name": "dymn10_im",
            "sr": 32000,
            "width_mult": 1,
            "pretrained_name": "dymn10_im",
            "strides": (2, 2, 2, 2),
            "pretrain_final_temp": 30.0,
        },
        "classification_head": {
            "hidden_features_size": 1280,
            "head_type": "mlp",
            "n_classes": 3840,
            "pooling": "mean",
            "activation_att": None,
            "last_activation": "",
        },
        "knowledge_distillation": {
            "temperature": 0.0,  # temperature for KLDivLoss
            "lambda": 0,              # just kd_loss :  lam * pred_loss + (1 - lam) * kd_loss # nopep8
            "loss": "cosine_similarity",             # "L1", "MSE", "BCEWithLogits" or None
        },

        # To continue training from a checkpoint
        "checkpoint_path": None,
    },
    
}
