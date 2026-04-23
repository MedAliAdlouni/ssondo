"""S-SONDO Embedding Quality Evaluation

Downloads ESC-50 (2000 environmental sounds, 50 classes) from HuggingFace,
extracts embeddings with the S-SONDO inference package (pip install ssondo),
clusters them, and evaluates whether clusters align with ground-truth labels.
"""

import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset, Audio
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from tqdm.auto import tqdm

from ssondo import get_ssondo

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

SEED = 42
TARGET_SR = 32000
MODEL_NAME = "matpac-mobilenetv3"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {DEVICE}')

    # ------------------------------------------------------------------
    # 1. Load S-SONDO Model (auto-downloads from Hugging Face Hub)
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('1. Loading S-SONDO model...')
    print('=' * 60)
    model = get_ssondo(MODEL_NAME, device=DEVICE)
    print(f'Model loaded: {MODEL_NAME} (emb_size={model.student_model.model.emb_size})')

    # ------------------------------------------------------------------
    # 2. Load ESC-50 Dataset
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('2. Loading ESC-50 dataset...')
    print('=' * 60)
    ds = load_dataset('ashraq/esc50', split='train')
    ds = ds.cast_column('audio', Audio(sampling_rate=TARGET_SR))
    print(f'ESC-50: {len(ds)} samples')

    class_names = sorted(set(ds['category']))
    print(f'Number of classes: {len(class_names)}')

    for i in range(3):
        sample = ds[i]
        audio = sample['audio']
        print(f"  Sample {i}: category='{sample['category']}', "
              f"sr={audio['sampling_rate']}, "
              f"duration={len(audio['array']) / audio['sampling_rate']:.1f}s")

    # ------------------------------------------------------------------
    # 3. Extract Embeddings
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('3. Extracting embeddings...')
    print('=' * 60)

    all_embeddings = []
    all_labels = []
    all_categories = []

    for sample in tqdm(ds, desc='Extracting embeddings'):
        audio = sample['audio']
        waveform = torch.tensor(audio['array'], dtype=torch.float32)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        min_samples = TARGET_SR * 10
        if waveform.shape[0] < min_samples:
            waveform = torch.nn.functional.pad(waveform, (0, min_samples - waveform.shape[0]))

        waveform = waveform.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(waveform)

        emb = emb.mean(dim=1).squeeze(0).cpu().numpy()

        all_embeddings.append(emb)
        all_labels.append(sample['target'])
        all_categories.append(sample['category'])

    embeddings = np.stack(all_embeddings)
    labels = np.array(all_labels)
    categories = np.array(all_categories)

    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Unique labels: {len(np.unique(labels))}')

    # ------------------------------------------------------------------
    # 4. Visualize Embeddings (t-SNE)
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('4. Running t-SNE...')
    print('=' * 60)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                         c=labels, cmap='tab20', s=8, alpha=0.7)
    ax.set_title('S-SONDO Embeddings - t-SNE (colored by ESC-50 class)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, label='Class ID')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_by_class.png'), dpi=150)
    print('  Saved tsne_by_class.png')
    plt.close()

    if HAS_UMAP:
        print('Running UMAP...')
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
        emb_umap = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        scatter = ax.scatter(emb_umap[:, 0], emb_umap[:, 1],
                             c=labels, cmap='tab20', s=8, alpha=0.7)
        ax.set_title('S-SONDO Embeddings - UMAP (colored by ESC-50 class)', fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, label='Class ID')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'umap_by_class.png'), dpi=150)
        print('  Saved umap_by_class.png')
        plt.close()

    # ------------------------------------------------------------------
    # 5. K-Means Clustering
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('5. Running KMeans (k=50)...')
    print('=' * 60)
    N_CLUSTERS = 50
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    print(f'Inertia: {kmeans.inertia_:.2f}')

    # ------------------------------------------------------------------
    # 6. Clustering Metrics
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('6. CLUSTERING EVALUATION RESULTS')
    print('=' * 60)

    sil = silhouette_score(embeddings, cluster_labels)
    ch = calinski_harabasz_score(embeddings, cluster_labels)
    db = davies_bouldin_score(embeddings, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)

    print(f'  {"Silhouette Score":>35s}: {sil:>10.4f}  (higher is better, max 1.0)')
    print(f'  {"Calinski-Harabasz Index":>35s}: {ch:>10.1f}  (higher is better)')
    print(f'  {"Davies-Bouldin Index":>35s}: {db:>10.4f}  (lower is better)')
    print(f'  {"Normalized Mutual Information":>35s}: {nmi:>10.4f}  (higher is better, max 1.0)')
    print(f'  {"Adjusted Rand Index":>35s}: {ari:>10.4f}  (higher is better, max 1.0)')

    # ------------------------------------------------------------------
    # 7. Cluster Purity
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('7. Cluster Purity Analysis')
    print('=' * 60)

    contingency = pd.crosstab(
        pd.Series(cluster_labels, name='Cluster'),
        pd.Series(categories, name='Category'),
    )

    cluster_info = []
    for cluster_id in range(N_CLUSTERS):
        if cluster_id not in contingency.index:
            continue
        row = contingency.loc[cluster_id]
        dominant_class = row.idxmax()
        dominant_count = row.max()
        total = row.sum()
        purity = dominant_count / total
        cluster_info.append({
            'Cluster': cluster_id,
            'Size': total,
            'Dominant Class': dominant_class,
            'Dominant Count': dominant_count,
            'Purity': purity,
        })

    df_clusters = pd.DataFrame(cluster_info).sort_values('Purity', ascending=False)
    overall_purity = df_clusters['Dominant Count'].sum() / df_clusters['Size'].sum()

    print(f'Overall Cluster Purity: {overall_purity:.4f}\n')
    print('Top 15 purest clusters:')
    print(df_clusters.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_clusters['Purity'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(overall_purity, color='red', linestyle='--', label=f'Overall purity: {overall_purity:.3f}')
    ax.set_xlabel('Cluster Purity')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Cluster Purity')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'purity_distribution.png'), dpi=150)
    print(f'\n  Saved purity_distribution.png')
    plt.close()

    # ------------------------------------------------------------------
    # 8. Per-Category Analysis (5 major ESC-50 groups)
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('8. Major Category Analysis (k=5)')
    print('=' * 60)

    MAJOR_CATEGORIES = {
        'Animals': list(range(0, 10)),
        'Natural soundscapes': list(range(10, 20)),
        'Human (non-speech)': list(range(20, 30)),
        'Domestic / Interior': list(range(30, 40)),
        'Urban / Exterior': list(range(40, 50)),
    }

    label_to_major = {}
    for major_name, label_ids in MAJOR_CATEGORIES.items():
        for lid in label_ids:
            label_to_major[lid] = major_name

    major_labels = np.array([label_to_major.get(l, 'Unknown') for l in labels])
    major_names_ordered = list(MAJOR_CATEGORIES.keys())

    for name in major_names_ordered:
        count = (major_labels == name).sum()
        print(f'  {name}: {count} samples')

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    cmap = plt.cm.Set1
    for i, name in enumerate(major_names_ordered):
        mask = major_labels == name
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap(i)] * mask.sum(), s=12, alpha=0.7, label=name)

    ax.set_title('S-SONDO Embeddings - t-SNE (colored by ESC-50 major category)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', fontsize=9, markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_by_major_category.png'), dpi=150)
    print(f'  Saved tsne_by_major_category.png')
    plt.close()

    colors_major = {name: i for i, name in enumerate(major_names_ordered)}
    major_label_ids = np.array([colors_major[m] for m in major_labels])

    kmeans_5 = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    cluster_labels_5 = kmeans_5.fit_predict(embeddings)

    nmi_5 = normalized_mutual_info_score(major_label_ids, cluster_labels_5)
    ari_5 = adjusted_rand_score(major_label_ids, cluster_labels_5)
    sil_5 = silhouette_score(embeddings, cluster_labels_5)

    print(f'\n  NMI (k=5):        {nmi_5:.4f}')
    print(f'  ARI (k=5):        {ari_5:.4f}')
    print(f'  Silhouette (k=5): {sil_5:.4f}')

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('S-SONDO EMBEDDING QUALITY EVALUATION - SUMMARY')
    print('=' * 60)
    print(f'Model:   {MODEL_NAME} (auto-downloaded from Hugging Face Hub)')
    print(f'Dataset: ESC-50 ({len(ds)} samples, 50 classes)')
    print(f'Embedding dim: {embeddings.shape[1]}')
    print()
    print('--- Fine-grained clustering (k=50) ---')
    print(f'  Silhouette Score:  {sil:>8.4f}')
    print(f'  Calinski-Harabasz: {ch:>8.1f}')
    print(f'  Davies-Bouldin:    {db:>8.4f}')
    print(f'  NMI:               {nmi:>8.4f}')
    print(f'  ARI:               {ari:>8.4f}')
    print(f'  Cluster Purity:    {overall_purity:>8.4f}')
    print()
    print('--- Coarse clustering (k=5, major categories) ---')
    print(f'  NMI:               {nmi_5:>8.4f}')
    print(f'  ARI:               {ari_5:>8.4f}')
    print(f'  Silhouette:        {sil_5:>8.4f}')
    print('=' * 60)
    print(f'\nPlots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
