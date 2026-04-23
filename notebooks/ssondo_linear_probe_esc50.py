"""S-SONDO Linear Probe on ESC-50

Demonstrates downstream finetuning with frozen S-SONDO backbone + linear
classifier on ESC-50 (50 environmental sound classes). This is the standard
evaluation protocol for audio embedding models.

Requirements: pip install ssondo datasets soundfile scikit-learn tqdm
"""

import os

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Audio
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from ssondo import get_ssondo

SEED = 42
TARGET_SR = 32000
MODEL_NAME = "matpac-mobilenetv3"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64


def extract_all_embeddings(model, dataset, device):
    """Extract embeddings for the entire dataset (one pass, cached)."""
    all_emb = []
    all_labels = []

    for sample in tqdm(dataset, desc="Extracting embeddings"):
        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        min_samples = TARGET_SR * 10
        if waveform.shape[0] < min_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, min_samples - waveform.shape[0])
            )

        waveform = waveform.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.get_embeddings(waveform)

        all_emb.append(emb.squeeze(0).cpu())
        all_labels.append(sample["target"])

    return torch.stack(all_emb), torch.tensor(all_labels, dtype=torch.long)


def train_linear_probe(embeddings, labels, train_idx, val_idx, emb_dim, n_classes, device):
    """Train a linear classifier on frozen embeddings."""
    X_train = embeddings[train_idx].to(device)
    y_train = labels[train_idx].to(device)
    X_val = embeddings[val_idx].to(device)
    y_val = labels[val_idx].to(device)

    head = nn.Linear(emb_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        head.train()
        # Mini-batch training
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            logits = head(X_train[idx])
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        head.eval()
        with torch.no_grad():
            val_logits = head(X_val)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds)

        if val_acc > best_acc:
            best_acc = val_acc

    return best_acc


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load S-SONDO Model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. Loading S-SONDO model...")
    print("=" * 60)
    model = get_ssondo(MODEL_NAME, device=device)
    model.freeze_backbone()
    print(f"Model: {MODEL_NAME} (embedding_dim={model.embedding_dim})")
    print("Backbone frozen for linear probing.")

    # ------------------------------------------------------------------
    # 2. Load ESC-50 Dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. Loading ESC-50 dataset...")
    print("=" * 60)
    ds = load_dataset("ashraq/esc50", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    n_classes = len(set(ds["target"]))
    print(f"ESC-50: {len(ds)} samples, {n_classes} classes")

    # ------------------------------------------------------------------
    # 3. Extract Embeddings (one-time, cached)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. Extracting embeddings (frozen backbone)...")
    print("=" * 60)
    embeddings, labels = extract_all_embeddings(model, ds, device)
    print(f"Embeddings: {embeddings.shape}  (cached in memory)")

    # ------------------------------------------------------------------
    # 4. 5-Fold Cross-Validation Linear Probe
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"4. Linear Probe ({EPOCHS} epochs, 5-fold CV)...")
    print("=" * 60)

    folds = ds["fold"]
    fold_ids = np.array(folds)
    unique_folds = sorted(set(folds))

    fold_accuracies = []
    for fold in unique_folds:
        val_idx = np.where(fold_ids == fold)[0]
        train_idx = np.where(fold_ids != fold)[0]

        acc = train_linear_probe(
            embeddings, labels, train_idx, val_idx,
            model.embedding_dim, n_classes, device,
        )
        fold_accuracies.append(acc)
        print(f"  Fold {fold}: {acc:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    # ------------------------------------------------------------------
    # 5. Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LINEAR PROBE RESULTS")
    print("=" * 60)
    print(f"Model:    {MODEL_NAME}")
    print(f"Dataset:  ESC-50 ({len(ds)} samples, {n_classes} classes)")
    print(f"Protocol: 5-fold CV, frozen backbone + linear head")
    print(f"Epochs:   {EPOCHS}")
    print()
    for fold, acc in zip(unique_folds, fold_accuracies):
        print(f"  Fold {fold}: {acc:.4f}")
    print(f"\n  Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
