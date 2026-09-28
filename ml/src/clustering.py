"""
clustering.py -- K-Means on SOM codebook for Medical SOM project.

Pipeline: X_scaled -> SOM (som_model.py) -> codebook -> K-Means (this file)
y labels NEVER influence K-Means. Used post-hoc only for validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CLUSTER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                  '#9b59b6', '#1abc9c', '#e67e22', '#34495e']


def extract_codebook_flat(som):
    """Flatten SOM weights to (n_neurons, n_features). Returns (codebook, n_rows, n_cols)."""
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape
    return weights.reshape(n_rows * n_cols, n_features), n_rows, n_cols


def select_k_elbow(codebook_flat, k_range=range(2, 9), seed=42):
    """WCSS elbow analysis to select K. Returns (wcss_results, silhouette_results, best_k)."""
    wcss_results, silhouette_results = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(codebook_flat)
        wcss = km.inertia_
        sil  = silhouette_score(codebook_flat, labels) if k > 1 else 0.0
        wcss_results.append((k, wcss))
        silhouette_results.append((k, sil))
        print(f'  K={k:2d}  WCSS={wcss:9.3f}  Silhouette={sil:.4f}')
    wcss_vals = [w for _, w in wcss_results]
    diffs2    = np.diff(np.diff(wcss_vals))
    best_k    = list(k_range)[int(np.argmax(diffs2)) + 1]
    return wcss_results, silhouette_results, best_k


def run_kmeans(codebook_flat, k, seed=42):
    """Fit K-Means on codebook vectors. Returns (kmeans, neuron_labels)."""
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=20)
    return kmeans, kmeans.fit_predict(codebook_flat)


def assign_neuron_clusters(n_rows, n_cols, neuron_labels):
    """Reshape flat neuron labels to (n_rows, n_cols) grid."""
    return neuron_labels.reshape(n_rows, n_cols)


def assign_sample_clusters(bmu_indices, cluster_grid):
    """Map each sample to a cluster via its BMU. Returns (n_samples,) array."""
    return cluster_grid[bmu_indices[:, 0], bmu_indices[:, 1]]


def cluster_sizes(sample_clusters, k):
    """Count and percentage per cluster. Returns {cluster_id: {count, pct}}."""
    n = len(sample_clusters)
    return {cid: {'count': int((sample_clusters == cid).sum()),
                  'pct':   round(int((sample_clusters == cid).sum()) / n * 100, 1)}
            for cid in range(k)}


def cluster_feature_profiles(X, sample_clusters, k, feature_names):
    """Mean feature values per cluster on un-scaled X. Returns DataFrame (k, n_features)."""
    X_np = X.values if hasattr(X, 'values') else X
    profiles = pd.DataFrame(
        [X_np[sample_clusters == cid].mean(axis=0) for cid in range(k)],
        columns=feature_names
    )
    profiles.index.name = 'cluster'
    return profiles


def cluster_diagnosis_distribution(sample_clusters, y_series, k):
    """POST-HOC ONLY: Benign/Malignant count per cluster. Returns DataFrame."""
    y_arr = y_series.values
    rows = []
    for cid in range(k):
        labels_in = y_arr[sample_clusters == cid]
        b, m = int((labels_in == 'B').sum()), int((labels_in == 'M').sum())
        total = b + m
        rows.append({'cluster': cid, 'B': b, 'M': m, 'total': total,
                     'pct_M': round(m / total * 100, 1) if total > 0 else 0.0})
    return pd.DataFrame(rows).set_index('cluster')





def plot_som_cluster_map(cluster_grid, umatrix, k, save_path=None):
    """Cluster assignments overlaid on SOM grid with U-Matrix background. Returns fig."""
    n_rows, n_cols = cluster_grid.shape
    colors = CLUSTER_COLORS[:k]
    cmap   = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(umatrix.T, cmap='bone', origin='upper',
              interpolation='nearest', aspect='equal', alpha=0.35)
    ax.imshow(cluster_grid.T, cmap=cmap, origin='upper',
              interpolation='nearest', aspect='equal', alpha=0.75,
              vmin=-0.5, vmax=k - 0.5)
    ax.set_title(f'SOM Cluster Map -- K={k} Clusters ({n_rows}x{n_cols} Grid)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Neuron column', fontsize=11)
    ax.set_ylabel('Neuron row', fontsize=11)
    ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
    ax.tick_params(labelsize=8)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i}') for i in range(k)]
    ax.legend(handles=patches, loc='upper right', fontsize=10, framealpha=0.85, title='Clusters')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_cluster_sizes(sizes, k, save_path=None):
    """Horizontal bar chart of cluster sample counts. Returns fig."""
    labels = [f'Cluster {i}' for i in range(k)]
    counts = [sizes[i]['count'] for i in range(k)]
    pcts   = [sizes[i]['pct']   for i in range(k)]
    colors = CLUSTER_COLORS[:k]
    fig, ax = plt.subplots(figsize=(9, max(4, k * 0.9 + 1.5)))
    bars = ax.barh(labels, counts, color=colors, edgecolor='white', linewidth=1.2, height=0.55)
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f'{count} ({pct}%)', va='center', ha='left',
                fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Number of Samples', fontsize=11)
    ax.set_title('Cluster Sizes (K-Means on SOM Codebook)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(counts) * 1.25)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_diagnosis_per_cluster(diag_dist, k, save_path=None):
    """Stacked bar chart: Benign/Malignant per cluster. POST-HOC. Returns fig."""
    clusters    = [f'Cluster {i}' for i in range(k)]
    benign_vals = [diag_dist.loc[i, 'B'] for i in range(k)]
    malig_vals  = [diag_dist.loc[i, 'M'] for i in range(k)]
    totals      = [diag_dist.loc[i, 'total'] for i in range(k)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.bar(clusters, benign_vals, color='#2ecc71', label='Benign (B)', edgecolor='white')
    ax1.bar(clusters, malig_vals,  color='#e74c3c', label='Malignant (M)',
            edgecolor='white', bottom=benign_vals)
    ax1.set_title('Diagnosis Distribution per Cluster\n(Post-hoc -- labels NOT used in training)',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sample Count', fontsize=11)
    ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.35)
    for i, (b, m, tot) in enumerate(zip(benign_vals, malig_vals, totals)):
        pct_m = round(m / tot * 100, 1) if tot > 0 else 0
        ax1.text(i, tot + 2, f'{pct_m}% M', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#e74c3c')
    ax2 = axes[1]
    pct_b = [diag_dist.loc[i, 'B'] / diag_dist.loc[i, 'total'] * 100
             if diag_dist.loc[i, 'total'] > 0 else 0 for i in range(k)]
    pct_m = [diag_dist.loc[i, 'pct_M'] for i in range(k)]
    x, w = np.arange(k), 0.35
    ax2.bar(x - w/2, pct_b, width=w, color='#2ecc71', label='% Benign',    edgecolor='white')
    ax2.bar(x + w/2, pct_m, width=w, color='#e74c3c', label='% Malignant', edgecolor='white')
    ax2.axhline(y=37.3, color='#e74c3c', linestyle='--', alpha=0.6,
                label='Overall % Malignant (37.3%)')
    ax2.set_xticks(x); ax2.set_xticklabels(clusters)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.set_title('Benign vs Malignant % per Cluster', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_cluster_feature_heatmap(profiles, save_path=None):
    """Heatmap of mean feature values per cluster. Returns fig."""
    import seaborn as sns  # type: ignore[import-untyped]
    profiles_norm = (profiles - profiles.mean()) / (profiles.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(20, max(3, len(profiles) * 1.2 + 2)))
    sns.heatmap(profiles_norm, cmap='RdBu_r', center=0, annot=False,
                linewidths=0.4, linecolor='white',
                cbar_kws={'label': 'Normalised mean (z-score across clusters)', 'shrink': 0.6},
                ax=ax, xticklabels=True,
                yticklabels=[f'Cluster {i}' for i in profiles.index])
    ax.set_title('Cluster Feature Profiles -- Mean Values per Cluster\n'
                 '(red = above average, blue = below average)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Feature', fontsize=10)
    ax.tick_params(axis='x', labelsize=7, rotation=45)
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig
