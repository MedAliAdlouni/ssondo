"""Training components setup (optimizer, scheduler, losses)."""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .utils import exp_warmup_linear_down


def setup_optimizer(
    conf: dict,
    student_model: nn.Module
) -> torch.optim.Optimizer:
    """
    Set up optimizer for training.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing optimizer settings.
    student_model : nn.Module
        The student model to be trained.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer.
    """
    optimizer_name = conf["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params=student_model.parameters(),
            **conf["optimizer_args"]
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            params=student_model.parameters(),
            **conf["optimizer_args"]
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not implemented")

    return optimizer


def setup_learning_rate_scheduler(
    conf: dict,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """
    Set up learning rate scheduler.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing scheduler settings.
    optimizer : torch.optim.Optimizer
        The optimizer to attach the scheduler to.
    train_loader : DataLoader
        Training data loader (used for scheduler calculations).

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler | None
        Learning rate scheduler (or None if not configured).
    """
    scheduler_name = conf["lr_scheduler"]

    if scheduler_name is None:
        scheduler = None

    elif scheduler_name == "OneCycleLR":
        # Calculate total training steps
        if train_loader.sampler is not None:
            n_audios_per_epoch = len(train_loader.sampler)
        else:
            n_audios_per_epoch = len(train_loader.dataset)

        total_steps = conf["epochs"] * (
            n_audios_per_epoch // (
                conf["batch_size"] * 
                conf["process"]["devices"] * 
                conf["process"]["num_nodes"]
            ) + 1  # Add 1 if drop_last=False
        )

        # Update config with calculated steps
        conf["lr_scheduler_args"]["total_steps"] = total_steps

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            **conf["lr_scheduler_args"]
        )

    elif scheduler_name == "CustomScheduler":
        schedule_lambda = exp_warmup_linear_down(
            conf["lr_scheduler_args"]["warm_up_len"],
            conf["lr_scheduler_args"]["ramp_down_len"],
            conf["lr_scheduler_args"]["ramp_down_start"],
            conf["lr_scheduler_args"]["last_lr_value"]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    else:
        raise ValueError(f"Scheduler {scheduler_name} is not implemented")

    return scheduler


def setup_loss_fct(
    conf: dict
) -> tuple:
    """
    Set up loss functions for prediction and knowledge distillation.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing loss function settings.

    Returns
    -------
    tuple
        Tuple containing:
        - pred_loss_func: Prediction loss function (or None)
        - kd_loss_func: Knowledge distillation loss function (or None)
    """
    # Prediction Loss
    pred_loss_name = conf["prediction_loss"]

    if pred_loss_name is None:
        pred_loss_func = None

    elif pred_loss_name == "BCEWithLogits":
        pred_loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")

    else:
        raise ValueError(f"Prediction loss {pred_loss_name} is not implemented")

    # Knowledge Distillation Loss
    kd_loss_name = conf["knowledge_distillation"]["loss"]

    if kd_loss_name is None:
        kd_loss_func = None
    elif kd_loss_name == "contrastive_loss":
        # Contrastive loss uses string identifiers for custom implementation
        loss_type = conf["knowledge_distillation"]["loss_params"]["loss_type"]
        contrastive_loss_map = {
            "vanilla": "kd_loss_contrastive",
            None: "kd_loss_contrastive",
            "cluster_aware": "_kd_loss_cluster_aware_contrastive",
            "neg_clusters": "kd_loss_contrastive_neg_clusters_only",
            "hybrid": "hybrid_cluster_contrastive_loss",
            "kd_loss_real_contrastive": "kd_loss_real_contrastive",
        }
        if loss_type not in contrastive_loss_map:
            raise ValueError(f"Unknown contrastive loss type: {loss_type}")
        kd_loss_func = contrastive_loss_map[loss_type]
    else:
        # Standard PyTorch loss functions
        standard_losses = {
            "MSE": torch.nn.MSELoss,
            "BCEWithLogits": torch.nn.BCEWithLogitsLoss,
            "CrossEntropy": torch.nn.CrossEntropyLoss,
            "L1": torch.nn.L1Loss,
            "KLDivLoss": torch.nn.KLDivLoss,
            "cosine_similarity": torch.nn.CosineEmbeddingLoss,
        }
        if kd_loss_name not in standard_losses:
            raise ValueError(f"KD loss {kd_loss_name} is not implemented")
        kd_loss_func = standard_losses[kd_loss_name](reduction="none")

    return pred_loss_func, kd_loss_func


def configure_trainer(conf: dict, log_dir: str) -> tuple:
    """
    Configure PyTorch Lightning trainer with callbacks and logging.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing trainer settings.
    log_dir : str
        Directory path for logs and checkpoints.

    Returns
    -------
    tuple
        (trainer, callbacks) - Configured PyTorch Lightning Trainer and list of callbacks.
    """
    # Setup Callbacks
    callbacks = []
    checkpoint_dir = os.path.join(log_dir, "checkpoint/")

    # Determine monitoring metric
    if conf["early_stopping"]:
        monitor = "val/total_loss"
        mode = "min"
    else:
        monitor = "epoch"
        mode = "max"

    # Model Checkpoint callback
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor=monitor,
        mode=mode,
        save_last=True,
        save_top_k=5,
        verbose=True,
    )
    callbacks.append(checkpoint)

    # Early Stopping callback
    if conf["early_stopping"]:
        early_stop = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=conf["early_stopping_args"]["patience"],
            verbose=True,
        )
        callbacks.append(early_stop)

    # Determine Hardware Configuration
    if torch.cuda.is_available():
        accelerator = "gpu"
        requested_devices = conf["process"]["devices"]
        available_devices = torch.cuda.device_count()
        devices = min(requested_devices, available_devices)
    else:
        accelerator = "cpu"
        devices = 1

    # Debug settings (for fast experiments)
    try:
        lmt_train_bt = conf["trainer"]["debug"]["lmt_train_bt"]
        lmt_val_bt = conf["trainer"]["debug"]["lmt_val_bt"]
    except (KeyError, TypeError):
        lmt_train_bt = None
        lmt_val_bt = None

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=conf["epochs"],
        callbacks=callbacks,
        default_root_dir=log_dir,
        devices=devices,
        accelerator=accelerator,
        num_nodes=conf["process"]["num_nodes"],
        limit_train_batches=lmt_train_bt,
        limit_val_batches=lmt_val_bt,
        gradient_clip_val=5.0,
        logger=TensorBoardLogger(log_dir, default_hp_metric=False),
        deterministic="warn",
        precision=conf["process"]["precision"],
        val_check_interval=conf["trainer"]["val_check_interval"],
        check_val_every_n_epoch=conf["trainer"]["check_val_every_n_epoch"],
        num_sanity_val_steps=conf["trainer"]["num_sanity_val_steps"],
    )

    return trainer, callbacks


def get_checkpoint_callback(callbacks: list) -> ModelCheckpoint:
    """
    Extract ModelCheckpoint callback from list of callbacks.

    Parameters
    ----------
    callbacks : list
        List of PyTorch Lightning callbacks.

    Returns
    -------
    ModelCheckpoint
        The ModelCheckpoint callback instance.
    """
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    raise ValueError("ModelCheckpoint callback not found in callbacks list")
