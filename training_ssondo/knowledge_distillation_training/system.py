"Code that defines the training, val steps and the logging of the experiment"

import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .data_augmentation import Mixup, get_spec_augment_pipeline


class KnowledgeDistillationSystem(pl.LightningModule):
    """
    PyTorch Lightning system for training a student model with knowledge
    distillation and logging relevant metrics and information.

    Parameters
    ----------
    config : dict
      Configuration dictionary containing experiment settings, including data
      augmentation, knowledge distillation, and training hyperparameters.
    student_model : nn.Module
      The student model to be trained.
    train_loader : torch.utils.data.DataLoader
      DataLoader for loading the training dataset.
    val_loader : torch.utils.data.DataLoader
      DataLoader for loading the validation dataset.
    optimizer : torch.optim.Optimizer
      Optimizer used for training the model.
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
      Learning rate scheduler for adjusting the learning rate during training.
    pred_loss_func : callable
      Loss function used to compute the prediction loss.
    kd_loss_func : callable
      Loss function used for knowledge distillation.
    preprocess : callable, optional
      Preprocessing function to apply to the input data (e.g., audio data)
      before feeding it into the model (default is None).
    """

    default_monitor: str = "val/pred_loss"

    def __init__(
        self,
        config,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        pred_loss_func,
        kd_loss_func,
        preprocess=None,
    ) -> None:
        super().__init__()

        self.student_model = student_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pred_loss_func = pred_loss_func
        self.kd_loss_func = kd_loss_func
        self.loss_weight = config["knowledge_distillation"]["lambda"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.preprocess = preprocess
        self.conf = config

        if config["data_augmentation"]["mixup"]:
            self.mixup = Mixup(config["data_augmentation"]["mixup_args"]["alpha"])

        if config["data_augmentation"]["spec_augment"]:
            self.spec_augment = get_spec_augment_pipeline(config)

        # Accumulators, to compute metrics on val set
        self.all_y = []
        self.all_y_hat = []

    def forward(self, x):
        """Applies forward pass of the model."""
        logits, emb = self.student_model(x)
        return logits, emb

    def configure_optimizers(self):
        """Initialize optimizers"""
        if self.scheduler is None:
            return self.optimizer

        if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            interval = "step"
        else:
            interval = "epoch"

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": self.scheduler, "interval": interval},
        }

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply batch transformation to your batch after it is transferred to the
        device.
        """
        x, y, y_teacher, cluster_labels = batch

        if self.preprocess is not None:
            self.preprocess = self.preprocess.to(x.device)
            x = self.preprocess(x)

        return x, y, y_teacher, cluster_labels

    def training_step(self, train_batch, batch_idx):
        x, y, y_teacher, cluster_labels = train_batch

        if self.conf["data_augmentation"]["spec_augment"]:
            x = self.spec_augment(x)

        if self.conf["data_augmentation"]["mixup"]:
            x, y, rn_indices, lam = self.mixup(x, y)
        else:
            rn_indices, lam = None, None

        y_hat, _ = self(x)

        if y_hat.dim() == 3 and y_hat.size(1) == 1:
            y_hat = y_hat.squeeze(1)

        # Prediction loss
        if self.pred_loss_func is not None:
            pred_loss = self.pred_loss_func(y_hat, y).mean()
        else:
            pred_loss = torch.tensor(0.0, device=x.device)

        # Distillation loss
        if self.kd_loss_func is not None:
            if isinstance(self.kd_loss_func, torch.nn.KLDivLoss):
                kd_loss = self._kd_loss_kl_div(y_hat, y_teacher, None, None, x.size(0))
            elif isinstance(self.kd_loss_func, torch.nn.CosineEmbeddingLoss):
                if self.conf["data_augmentation"]["mixup"]:
                    kd_loss = self._kd_loss_cosine_embedding(
                        y_hat, y_teacher, rn_indices, lam
                    )
                else:
                    kd_loss = self._kd_loss_cosine_embedding(
                        y_hat, y_teacher, None, None
                    )
            elif self.kd_loss_func == "kd_loss_contrastive":
                kd_loss = self._kd_loss_contrastive(y_hat, y_teacher)
            elif self.kd_loss_func == "kd_loss_real_contrastive":
                kd_loss = self.kd_loss_real_contrastive(y_hat, y_teacher)
            elif self.kd_loss_func == "_kd_loss_cluster_aware_contrastive":
                kd_loss = self._kd_loss_cluster_aware_contrastive(
                    y_hat, y_teacher, cluster_labels
                )
            elif self.kd_loss_func == "kd_loss_contrastive_neg_clusters_only":
                kd_loss = self.kd_loss_contrastive_neg_clusters_only(
                    y_hat, y_teacher, cluster_labels
                )
            elif self.kd_loss_func == "hybrid_cluster_contrastive_loss":
                kd_loss = self.hybrid_cluster_contrastive_loss(
                    y_hat, y_teacher, cluster_labels, self.conf
                )
            else:
                kd_loss = self._kd_loss_default(y_hat, y_teacher, None, None, x.size(0))
        else:
            kd_loss = torch.tensor(0.0, device=x.device)

        pred_loss = self.loss_weight * pred_loss
        kd_loss = (1 - self.loss_weight) * kd_loss

        total_loss = pred_loss + kd_loss

        if self.scheduler is not None:
            cur_lr = self.scheduler.get_last_lr()[0]
            self.log("lr", cur_lr, prog_bar=True, logger=True, on_epoch=True)

        self.log("train/pred_loss", pred_loss, on_epoch=True)
        self.log("train/kd_loss", kd_loss, on_epoch=True)
        self.log("train/total_loss", total_loss, on_epoch=True)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, y, y_teacher, cluster_labels = val_batch
        y_hat, _ = self(x)
        if y_hat.dim() == 3 and y_hat.size(1) == 1:
            y_hat = y_hat.squeeze(1)

        # Prediction loss
        if self.pred_loss_func is not None:
            pred_loss = self.pred_loss_func(y_hat, y).mean()
        else:
            pred_loss = torch.tensor(0.0, device=x.device)

        # Distillation loss
        if self.kd_loss_func is not None:
            if isinstance(self.kd_loss_func, torch.nn.KLDivLoss):
                kd_loss = self._kd_loss_kl_div(y_hat, y_teacher, None, None, x.size(0))
            elif isinstance(self.kd_loss_func, torch.nn.CosineEmbeddingLoss):
                kd_loss = self._kd_loss_cosine_embedding(y_hat, y_teacher, None, None)
            elif self.kd_loss_func == "kd_loss_contrastive":
                kd_loss = self._kd_loss_contrastive(y_hat, y_teacher)
            elif self.kd_loss_func == "kd_loss_real_contrastive":
                kd_loss = self.kd_loss_real_contrastive(y_hat, y_teacher)
            elif self.kd_loss_func == "_kd_loss_cluster_aware_contrastive":
                kd_loss = self._kd_loss_cluster_aware_contrastive(
                    y_hat, y_teacher, cluster_labels
                )
            elif self.kd_loss_func == "kd_loss_contrastive_neg_clusters_only":
                kd_loss = self.kd_loss_contrastive_neg_clusters_only(
                    y_hat, y_teacher, cluster_labels
                )
            elif self.kd_loss_func == "hybrid_cluster_contrastive_loss":
                kd_loss = self.hybrid_cluster_contrastive_loss(
                    y_hat, y_teacher, cluster_labels, self.conf
                )
            else:
                kd_loss = self._kd_loss_default(y_hat, y_teacher, None, None, x.size(0))
        else:
            kd_loss = torch.tensor(0.0, device=x.device)

        pred_loss = self.loss_weight * pred_loss
        kd_loss = (1 - self.loss_weight) * kd_loss
        total_loss = pred_loss + kd_loss

        self.log(
            "val/pred_loss", pred_loss, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("val/kd_loss", kd_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val/total_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True
        )

        if self.pred_loss_func is not None:
            self.all_y.append(y.detach())
            self.all_y_hat.append(torch.sigmoid(y_hat).detach())

        return total_loss

    def on_validation_epoch_end(self) -> None:

        if self.pred_loss_func is None:
            # If no pred_loss function is defined, skip the mAP computation
            return super().on_validation_epoch_end()

        # Concatenate all tensors in the accumulators
        all_y = torch.cat(self.all_y, dim=0)
        all_y_hat = torch.cat(self.all_y_hat, dim=0)

        # Synchronize the tensors across all GPUs and convert it to numpy to
        # compute the mAP
        all_y = (
            self.all_gather(all_y).reshape(-1, all_y.shape[-1]).cpu().float().numpy()
        )
        all_y_hat = (
            self.all_gather(all_y_hat)
            .reshape(-1, all_y_hat.shape[-1])
            .cpu()
            .float()
            .numpy()
        )

        try:
            average_precision = metrics.average_precision_score(
                all_y, all_y_hat, average=None
            )
        except ValueError:
            average_precision = np.array([np.nan] * self.conf["dataset"]["n_classes"])

        self.log("val/mAP", average_precision.mean(), on_epoch=True, prog_bar=True)

        # Clear the accumulators
        self.all_y.clear()
        self.all_y_hat.clear()

        return super().on_validation_epoch_end()

    def on_validation_end(self) -> None:
        # Log model params
        hparams = self.conf
        self.logger.log_hyperparams(
            hparams, metrics=self.trainer.callback_metrics["val/pred_loss"]
        )

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.conf
        return checkpoint

    def _kd_loss_default(self, y_hat, y_teacher, rn_indices, lam, bs):
        if self.conf["data_augmentation"]["mixup"]:
            kd_loss = self.kd_loss_func(y_hat, y_teacher).mean(dim=1) * lam.reshape(bs)
            kd_loss += self.kd_loss_func(y_hat, y_teacher[rn_indices]).mean(dim=1) * (
                1 - lam.reshape(bs)
            )
        else:
            kd_loss = self.kd_loss_func(y_hat, y_teacher)

        return kd_loss.mean()

    def _kd_loss_kl_div(self, y_hat, y_teacher, rn_indices, lam, bs):
        T = self.conf["knowledge_distillation"]["temperature"]

        y_hat_T = y_hat / T
        y_teacher_T = y_teacher / T

        y_hat_log_prob = F.log_softmax(y_hat_T, dim=1)
        y_teacher_prob = F.softmax(y_teacher_T, dim=1)

        if self.conf["data_augmentation"]["mixup"]:
            kd_loss = self.kd_loss_func(y_hat_log_prob, y_teacher_prob).mean(
                dim=1
            ) * lam.reshape(bs)
            kd_loss += self.kd_loss_func(
                y_hat_log_prob, y_teacher_prob[rn_indices]
            ).mean(dim=1) * (1 - lam.reshape(bs))
        else:
            kd_loss = self.kd_loss_func(y_hat_log_prob, y_teacher_prob)

        kd_loss = (kd_loss * (T**2)).mean()
        return kd_loss

    def _kd_loss_cosine_embedding(self, y_hat, y_teacher, rn_indices, lam):
        # Handle if kd_loss_func is a string
        if isinstance(self.kd_loss_func, str):
            if self.kd_loss_func == "hybrid_cluster_contrastive_loss":
                kd_loss_func = torch.nn.CosineEmbeddingLoss(reduction="none")
            else:
                raise ValueError(f"Unknown kd_loss_func: {self.kd_loss_func}")
        else:
            # Assume it's already a callable loss
            kd_loss_func = self.kd_loss_func

        y_hat_norm = F.normalize(y_hat, dim=1)
        y_teacher_norm = F.normalize(y_teacher, dim=1)
        target = torch.ones(y_hat_norm.size(0), device=y_hat_norm.device)

        if self.conf["data_augmentation"]["mixup"] and rn_indices is not None:
            loss1 = kd_loss_func(y_hat_norm, y_teacher_norm, target)
            loss2 = kd_loss_func(y_hat_norm, y_teacher_norm[rn_indices], target)
            kd_loss = lam.view(-1) * loss1 + (1 - lam.view(-1)) * loss2
        else:
            kd_loss = kd_loss_func(y_hat_norm, y_teacher_norm, target)

        return kd_loss.mean()

    def _kd_loss_contrastive(self, y_hat, y_teacher):
        y_hat_norm = F.normalize(y_hat, dim=1)
        y_teacher_norm = F.normalize(y_teacher, dim=1)

        logits_st = torch.matmul(y_hat_norm, y_teacher_norm.T)
        logits_ts = torch.matmul(y_teacher_norm, y_hat_norm.T)

        labels = torch.arange(logits_st.size(0), device=logits_st.device)
        T = self.conf["knowledge_distillation"]["temperature"]

        logits_st /= T
        logits_ts /= T

        loss_st = F.cross_entropy(logits_st, labels)
        loss_ts = F.cross_entropy(logits_ts, labels)

        kd_loss = 0.5 * (loss_st + loss_ts)
        return kd_loss

    def kd_loss_real_contrastive(self, y_hat, y_teacher):
        y_hat = F.normalize(y_hat, dim=1)
        y_teacher = F.normalize(y_teacher, dim=1)

        logits = (
            torch.matmul(y_hat, y_teacher.T)
            / self.conf["knowledge_distillation"]["temperature"]
        )

        # Diagonal = positives
        positives = logits.diag()

        # Mask out positives
        negatives = logits - torch.eye(logits.size(0), device=logits.device) * 1e9

        # Denominator: log-sum-exp over negatives only
        denom = torch.logsumexp(negatives, dim=1)

        return -(positives - denom).mean()

    def kd_loss_contrastive_neg_clusters_only(self, y_hat, y_teacher, cluster_labels):
        """
        Contrastive KD loss where negatives are restricted to different clusters only.
        This avoids penalizing student/teacher pairs from the same cluster.
        Vectorized implementation.
        """
        y_hat_norm = F.normalize(y_hat, dim=1)
        y_teacher_norm = F.normalize(y_teacher, dim=1)
        T = self.conf["knowledge_distillation"]["temperature"]

        # Compute all pairwise similarities
        logits = torch.matmul(y_hat_norm, y_teacher_norm.T) / T

        # Create masks
        cluster_labels_expanded = cluster_labels.unsqueeze(1)
        same_cluster_mask = (
            cluster_labels_expanded == cluster_labels_expanded.T
        ).float()

        # Different cluster mask (excluding diagonal)
        different_cluster_mask = 1 - same_cluster_mask

        # Positive logits (diagonal elements only)
        pos_logits = torch.diag(logits)

        # Negative logits (different cluster, excluding diagonal)
        neg_logits = logits * different_cluster_mask

        # Check which samples have valid negatives
        has_negatives = different_cluster_mask.sum(dim=1) > 0

        if has_negatives.sum() == 0:
            return torch.tensor(0.0, device=y_hat.device, requires_grad=True)

        # Compute InfoNCE loss for all samples
        pos_exp = torch.exp(pos_logits)
        neg_exp = torch.exp(neg_logits).sum(dim=1)

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(negs))))
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))

        # Only average over samples that have valid negatives
        return loss[has_negatives].mean()

    def _kd_loss_cluster_aware_contrastive(self, y_hat, y_teacher, cluster_labels):
        y_hat_norm = F.normalize(y_hat, dim=1)
        y_teacher_norm = F.normalize(y_teacher, dim=1)

        logits_st = torch.matmul(y_hat_norm, y_teacher_norm.T)
        logits_ts = torch.matmul(y_teacher_norm, y_hat_norm.T)

        T = self.conf["knowledge_distillation"]["temperature"]
        logits_st /= T
        logits_ts /= T

        # Compute masks once
        cluster_labels_expanded = cluster_labels.unsqueeze(1)
        same_cluster_mask = (
            cluster_labels_expanded == cluster_labels_expanded.T
        ).float()
        diagonal_mask = torch.eye(cluster_labels.size(0), device=cluster_labels.device)

        # Remove diagonal (self-similarity)
        same_cluster_mask = same_cluster_mask - diagonal_mask
        different_cluster_mask = 1 - same_cluster_mask - diagonal_mask

        # Vectorized computation
        # For each sample, compute positive and negative exponentials
        pos_logits_st = logits_st * same_cluster_mask
        neg_logits_st = logits_st * different_cluster_mask

        pos_logits_ts = logits_ts * same_cluster_mask
        neg_logits_ts = logits_ts * different_cluster_mask

        # Use logsumexp for numerical stability
        pos_exp_st = torch.exp(pos_logits_st).sum(dim=1)
        neg_exp_st = torch.exp(neg_logits_st).sum(dim=1)

        pos_exp_ts = torch.exp(pos_logits_ts).sum(dim=1)
        neg_exp_ts = torch.exp(neg_logits_ts).sum(dim=1)

        # Compute losses (only for samples that have both positive and negative examples)
        valid_samples = (pos_exp_st > 0) & (neg_exp_st > 0)

        if valid_samples.sum() > 0:
            loss_st = -torch.log(
                pos_exp_st[valid_samples]
                / (pos_exp_st[valid_samples] + neg_exp_st[valid_samples])
            )
            loss_ts = -torch.log(
                pos_exp_ts[valid_samples]
                / (pos_exp_ts[valid_samples] + neg_exp_ts[valid_samples])
            )

            kd_loss = 0.5 * (loss_st + loss_ts).mean()
        else:
            kd_loss = torch.tensor(
                0.0, device=cluster_labels.device, requires_grad=True
            )

        return kd_loss

    def hybrid_cluster_contrastive_loss(
        self, y_hat, y_teacher, cluster_labels, rn_indices=None, lam=None
    ):
        """
        Hybrid KD loss:
        1. Cosine embedding loss (teacher prediction alignment)
        2. Cluster-aware contrastive learning (structure preservation)
        3. Cross-cluster teacher guidance (semantic spacing)
        """
        # === Component 1: Alignment loss
        alignment_loss = self._kd_loss_cosine_embedding(
            y_hat, y_teacher, rn_indices, lam
        )

        # === Component 2: Contrastive loss
        contrastive_loss = self._kd_loss_cluster_aware_contrastive(
            y_hat, y_teacher, cluster_labels
        )

        # === Component 3: Cross-cluster teacher guidance
        cross_cluster_loss = self._cross_cluster_teacher_guidance_loss(
            y_hat, y_teacher, cluster_labels
        )

        # Combine with weights
        alpha = self.conf["knowledge_distillation"]["loss_params"]["alignment_weight"]
        beta = self.conf["knowledge_distillation"]["loss_params"]["contrastive_weight"]
        gamma = self.conf["knowledge_distillation"]["loss_params"][
            "cross_cluster_weight"
        ]

        total_loss = (
            alpha * alignment_loss
            + beta * contrastive_loss
            + gamma * cross_cluster_loss
        )

        return total_loss

    def _cross_cluster_teacher_guidance_loss(self, y_hat, y_teacher, cluster_labels):
        """
        Cross-cluster teacher guidance loss to prevent cluster collapse.
        Preserves inter-cluster distances by matching student and teacher
        cluster centroid relationships.
        """
        # Normalize embeddings
        y_hat_norm = F.normalize(y_hat, dim=1)
        y_teacher_norm = F.normalize(y_teacher, dim=1)

        unique_clusters = torch.unique(cluster_labels)

        # Compute cluster centroids for both student and teacher
        student_centroids = []
        teacher_centroids = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id

            # Compute centroids (mean of embeddings in each cluster)
            student_centroid = y_hat_norm[cluster_mask].mean(dim=0)
            teacher_centroid = y_teacher_norm[cluster_mask].mean(dim=0)

            # Normalize centroids
            student_centroid = F.normalize(
                student_centroid.unsqueeze(0), dim=1
            ).squeeze(0)
            teacher_centroid = F.normalize(
                teacher_centroid.unsqueeze(0), dim=1
            ).squeeze(0)

            student_centroids.append(student_centroid)
            teacher_centroids.append(teacher_centroid)

        # Stack centroids into matrices
        student_centroids = torch.stack(
            student_centroids
        )  # [n_clusters, embedding_dim]
        teacher_centroids = torch.stack(
            teacher_centroids
        )  # [n_clusters, embedding_dim]

        # Direct cosine similarity loss between corresponding centroids
        # This ensures each student cluster centroid aligns with its teacher centroid
        cosine_similarities = F.cosine_similarity(
            student_centroids, teacher_centroids, dim=1
        )

        return -cosine_similarities.mean()
