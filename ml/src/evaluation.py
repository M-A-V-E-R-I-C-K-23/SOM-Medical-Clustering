"""
evaluation.py -- Evaluation metrics and grid experiments for Medical SOM project.

ARI is the ONLY metric that uses y (diagnosis labels), post-hoc only.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA


def compute_all_metrics(som, X_scaled, sample_clusters, y_series):
    """Compute QE, TE, Silhouette, and ARI. Returns dict with all four metrics."""
    qe         = som.quantization_error(X_scaled)
    te         = som.topographic_error(X_scaled)
    silhouette = silhouette_score(X_scaled, sample_clusters)
    y_numeric  = (y_series.values == 'M').astype(int)
    ari        = adjusted_rand_score(y_numeric, sample_clusters)
    return {
        'quantization_error': round(qe,         6),
        'topographic_error':  round(te,         6),
        'silhouette_score':   round(silhouette, 6),
        'ari':                round(ari,        6),
    }


def run_grid_experiment(X_scaled, y_series, grid_sizes=[(9,9),(11,11),(13,13)],
                        n_iterations=10000, sigma=1.5, learning_rate=0.5,
                        seed=42, k_range=range(2, 9)):
    """Run full SOM + K-Means + evaluation pipeline for each grid size. Returns list of dicts."""
    from ml.src import som_model  as sm
    from ml.src import clustering as cl

    results = []
    for n_rows, n_cols in grid_sizes:
        label = f'{n_rows}x{n_cols}'
        print(f'\n{"="*55}\n  Grid {label}  ({n_rows*n_cols} neurons)\n{"="*55}')
        som = sm.create_som(n_rows=n_rows, n_cols=n_cols,
                            n_features=X_scaled.shape[1],
                            sigma=sigma, learning_rate=learning_rate, random_seed=seed)
        som, _ = sm.train_som(som, X_scaled, n_iterations=n_iterations,
                              random_seed=seed, record_qe_every=n_iterations)
        codebook_flat, _, _ = cl.extract_codebook_flat(som)
        print(f'  Codebook: {codebook_flat.shape}\n  Elbow analysis:')
        wcss_res, sil_res, best_k = cl.select_k_elbow(codebook_flat, k_range=k_range, seed=seed)
        print(f'  Best K = {best_k}')
        kmeans, neuron_labels = cl.run_kmeans(codebook_flat, k=best_k, seed=seed)
        cluster_grid    = cl.assign_neuron_clusters(n_rows, n_cols, neuron_labels)
        bmu_indices     = sm.get_bmu_indices(som, X_scaled)
        sample_clusters = cl.assign_sample_clusters(bmu_indices, cluster_grid)
        sizes   = cl.cluster_sizes(sample_clusters, best_k)
        metrics = compute_all_metrics(som, X_scaled, sample_clusters, y_series)
        print(f'  QE={metrics["quantization_error"]:.4f}  TE={metrics["topographic_error"]:.4f}'
              f'  Sil={metrics["silhouette_score"]:.4f}  ARI={metrics["ari"]:.4f}')
        results.append({
            'grid': label, 'n_rows': n_rows, 'n_cols': n_cols, 'n_neurons': n_rows * n_cols,
            'k': best_k, **metrics,
            'cluster_sizes':  {str(k): v for k, v in sizes.items()},
            'som_object':     som,
            'sample_clusters': sample_clusters,
            'bmu_indices':    bmu_indices,
        })
    return results


def select_best_grid(results):
    """Score each grid by composite of normalised QE, TE, Silhouette, ARI. Returns (best, scores, reason)."""
    metrics_names = ['quantization_error', 'topographic_error', 'silhouette_score', 'ari']
    lower_better  = {'quantization_error', 'topographic_error'}
    values = {m: np.array([r[m] for r in results]) for m in metrics_names}
    normed = {}
    for m, vals in values.items():
        rng = vals.max() - vals.min()
        n = np.ones_like(vals) * 0.5 if rng < 1e-10 else (vals - vals.min()) / rng
        normed[m] = (1 - n) if m in lower_better else n
    composite: np.ndarray = np.asarray(
        sum(normed[m] for m in metrics_names) / len(metrics_names)
    )
    best_idx  = int(np.argmax(composite))
    best      = results[best_idx]
    scores    = [{'grid': r['grid'], 'composite_score': round(float(s), 4)}
                 for r, s in zip(results, composite)]
    reason = (f"Grid {best['grid']} selected. Composite score: {composite[best_idx]:.4f}. "
              f"QE={best['quantization_error']:.4f}, TE={best['topographic_error']:.4f}, "
              f"Silhouette={best['silhouette_score']:.4f}, ARI={best['ari']:.4f}.")
    return best, scores, reason


def compute_pca_projection(X_scaled, n_components=2, seed=42):
    """PCA on standardised features. Returns (X_pca, pca, explained_var)."""
    pca   = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, pca.explained_variance_ratio_


def plot_grid_comparison(results, scores, best_grid_label, save_path=None):
    """Multi-panel bar chart comparing QE, TE, Silhouette, ARI across grid sizes. Returns fig."""
    grids  = [r['grid']               for r in results]
    qes    = [r['quantization_error'] for r in results]
    tes    = [r['topographic_error']  for r in results]
    sils   = [r['silhouette_score']   for r in results]
    aris   = [r['ari']                for r in results]
    comps  = [s['composite_score']    for s in scores]
    colors = ['#e74c3c' if g == best_grid_label else '#3498db' for g in grids]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'Grid Size Comparison -- SOM Quality Metrics\n(Red = selected: {best_grid_label})',
                 fontsize=14, fontweight='bold')

    def bar(ax, vals, title, ylabel, lower_better=False):
        bars = ax.bar(grids, vals, color=colors, edgecolor='white', linewidth=1.2, width=0.55)
        for bar_, v in zip(bars, vals):
            ax.text(bar_.get_x() + bar_.get_width()/2,
                    bar_.get_height() + (max(vals)-min(vals))*0.02,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('(lower = better)' if lower_better else '(higher = better)',
                      fontsize=8, color='#7f8c8d')
        ax.grid(axis='y', alpha=0.35)
        ax.set_ylim(0, max(vals) * 1.2)

    bar(axes[0,0], qes,   'Quantization Error',  'QE',    lower_better=True)
    bar(axes[0,1], tes,   'Topographic Error',   'TE',    lower_better=True)
    bar(axes[0,2], sils,  'Silhouette Score',    'Sil')
    bar(axes[1,0], aris,  'Adjusted Rand Index', 'ARI')
    bar(axes[1,1], comps, 'Composite Score',     'Score')

    ax_table = axes[1, 2]
    ax_table.axis('off')
    col_labels = ['Grid', 'QE', 'TE', 'Sil', 'ARI', 'K']
    rows_data  = [[r['grid'], f"{r['quantization_error']:.4f}", f"{r['topographic_error']:.4f}",
                   f"{r['silhouette_score']:.4f}", f"{r['ari']:.4f}", str(r['k'])]
                  for r in results]
    tbl = ax_table.table(cellText=rows_data, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.6)
    for i, r in enumerate(results):
        if r['grid'] == best_grid_label:
            for j in range(len(col_labels)):
                tbl[(i+1, j)].set_facecolor('#fde8e8')
    ax_table.set_title('Summary Table', fontsize=11, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_pca_scatter(X_pca, sample_clusters, y_series, explained_var, k, save_path=None):
    """Side-by-side PCA scatter: left=cluster colour, right=diagnosis colour. Returns fig."""
    CLUSTER_COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#34495e']
    DIAG_COLORS    = {'B': '#2ecc71', 'M': '#e74c3c'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'PCA 2D Projection (PC1={explained_var[0]*100:.1f}%, '
                 f'PC2={explained_var[1]*100:.1f}% | Total={sum(explained_var)*100:.1f}% variance)',
                 fontsize=13, fontweight='bold')
    ax1 = axes[0]
    for cid in range(k):
        mask = sample_clusters == cid
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=CLUSTER_COLORS[cid],
                    s=22, alpha=0.75, label=f'Cluster {cid}', edgecolors='none')
    ax1.set_title('Coloured by K-Means Cluster\n(Unsupervised -- no labels used)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=10)
    ax1.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=10)
    ax1.legend(fontsize=9, framealpha=0.8); ax1.grid(alpha=0.25)
    ax2 = axes[1]
    y_arr = y_series.values
    for label, color in DIAG_COLORS.items():
        mask = y_arr == label
        name = 'Benign' if label == 'B' else 'Malignant'
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color,
                    s=22, alpha=0.75, label=f'{name} ({mask.sum()})', edgecolors='none')
    ax2.set_title('Coloured by Diagnosis Label\n(Post-hoc reference only)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=10)
    ax2.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=10)
    ax2.legend(fontsize=9, framealpha=0.8); ax2.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def save_metrics_json(best_result, all_results, scores, selection_reason, save_path=None):
    """Write final metrics and experiment results to results/metrics.json."""
    def clean(r):
        return {k: v for k, v in r.items()
                if k not in ('som_object', 'sample_clusters', 'bmu_indices')}
    output = {
        'selected_grid':        best_result['grid'],
        'selected_k':           best_result['k'],
        'quantization_error':   best_result['quantization_error'],
        'topographic_error':    best_result['topographic_error'],
        'silhouette_score':     best_result['silhouette_score'],
        'ari':                  best_result['ari'],
        'cluster_sizes':        best_result['cluster_sizes'],
        'grid_selection_reason': selection_reason,
        'grid_comparison':      [{**clean(r), 'composite_score': s['composite_score']}
                                 for r, s in zip(all_results, scores)],
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
    return output
