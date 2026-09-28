"""
backend/pipeline.py
-------------------
Central service that runs and holds the full ML pipeline result in memory.

Pipeline flow (matches validated Phase 1-5 methodology exactly):
    load_data -> prepare_features -> scale_features
    -> create_som -> train_som -> get_bmu_indices
    -> extract_codebook_flat -> run_kmeans -> assign_neuron_clusters -> assign_sample_clusters
    -> compute_all_metrics
"""

import os

from ml.src import preprocessing as prep
from ml.src import som_model as sm
from ml.src import clustering as cl
from ml.src import evaluation as ev

# Global in-memory state: holds latest pipeline result
_pipeline_result: dict | None = None

# Paths relative to project root (medical-som-clustering/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV_PATH  = os.path.join(_PROJECT_ROOT, "ml", "data", "breast_cancer.csv")
_WDBC_PATH = os.path.join(_PROJECT_ROOT, "..", "wdbc.data")

# Default final configuration (selected 9x9 grid, K=3)
FINAL_ROWS    = 9
FINAL_COLS    = 9
FINAL_K       = 3
N_ITERATIONS  = 10_000
SIGMA         = 1.5
LEARNING_RATE = 0.5
SEED          = 42


def get_result() -> dict:
    """Return the latest pipeline result, or raise if not yet trained."""
    if _pipeline_result is None:
        raise RuntimeError("Pipeline has not been trained yet. Call POST /api/train first.")
    return _pipeline_result


def is_trained() -> bool:
    """Return True if the pipeline has been run at least once."""
    return _pipeline_result is not None


def run_pipeline(
    n_rows:        int   = FINAL_ROWS,
    n_cols:        int   = FINAL_COLS,
    k:             int   = FINAL_K,
    n_iterations:  int   = N_ITERATIONS,
    sigma:         float = SIGMA,
    learning_rate: float = LEARNING_RATE,
    seed:          int   = SEED,
) -> dict:
    """Run the complete ML pipeline end-to-end and cache the result in memory."""
    global _pipeline_result

    # Step 1: Load data
    X_raw, y_raw, source = prep.load_data(local_csv=_CSV_PATH, wdbc_raw=_WDBC_PATH)

    # Step 2: Validate and separate X and y
    X, y_series, feature_names = prep.prepare_features(X_raw, y_raw)

    # Step 3: Scale X only (y never touches the scaler)
    X_scaled, _ = prep.scale_features(X)

    # Step 4: Train SOM on X_scaled
    som = sm.create_som(
        n_rows=n_rows, n_cols=n_cols,
        n_features=X_scaled.shape[1],
        sigma=sigma, learning_rate=learning_rate,
        random_seed=seed,
    )
    som, qe_history = sm.train_som(
        som, X_scaled,
        n_iterations=n_iterations,
        random_seed=seed,
        record_qe_every=max(1, n_iterations // 20),
    )

    # Step 5: BMU assignment
    bmu_indices = sm.get_bmu_indices(som, X_scaled)

    # Step 6: K-Means on SOM codebook vectors (NOT on raw samples)
    codebook_flat, n_rows_g, n_cols_g = cl.extract_codebook_flat(som)
    kmeans, neuron_labels = cl.run_kmeans(codebook_flat, k=k, seed=seed)
    cluster_grid    = cl.assign_neuron_clusters(n_rows_g, n_cols_g, neuron_labels)
    sample_clusters = cl.assign_sample_clusters(bmu_indices, cluster_grid)

    # Step 7: Evaluation (y used only here - post-hoc)
    metrics = ev.compute_all_metrics(som, X_scaled, sample_clusters, y_series)

    # Step 8: Cluster analysis
    sizes     = cl.cluster_sizes(sample_clusters, k)
    diag_dist = cl.cluster_diagnosis_distribution(sample_clusters, y_series, k)
    profiles  = cl.cluster_feature_profiles(X, sample_clusters, k, feature_names)

    # Step 9: SOM visualisation data
    umatrix  = sm.get_umatrix(som)
    hit_map  = sm.get_hit_map(som, X_scaled)

    # Assemble result dict
    _pipeline_result = {
        "source":                 source,
        "sample_count":           int(X_raw.shape[0]),
        "feature_count":          int(X_raw.shape[1]),
        "feature_names":          list(feature_names),
        "diagnosis_distribution": _series_value_counts(y_series),

        "config": {
            "grid":          f"{n_rows}x{n_cols}",
            "n_rows":        n_rows,
            "n_cols":        n_cols,
            "k":             k,
            "n_iterations":  n_iterations,
            "sigma":         sigma,
            "learning_rate": learning_rate,
            "seed":          seed,
        },

        "umatrix":     umatrix.tolist(),
        "hit_map":     hit_map.tolist(),
        "bmu_indices": bmu_indices.tolist(),

        "cluster_count":    k,
        "cluster_sizes":    _serialize_sizes(sizes),
        "diag_dist":        _serialize_diag(diag_dist, k),
        "feature_profiles": _serialize_profiles(profiles),

        "quantization_error":  float(metrics["quantization_error"]),
        "topographic_error":   float(metrics["topographic_error"]),
        "silhouette_score":    metrics["silhouette_score"],
        "ari":                 float(metrics["ari"]),

        "qe_history": [
            {"step": int(step), "qe": float(qe)}
            for step, qe in qe_history
        ],
    }

    return _pipeline_result


def get_dataset_info() -> dict:
    """Load and return basic dataset info without requiring a trained model."""
    X_raw, y_raw, source = prep.load_data(local_csv=_CSV_PATH, wdbc_raw=_WDBC_PATH)
    X, y_series, feature_names = prep.prepare_features(X_raw, y_raw)
    return {
        "source":                 source,
        "sample_count":           int(X_raw.shape[0]),
        "feature_count":          int(X_raw.shape[1]),
        "feature_names":          list(feature_names),
        "diagnosis_distribution": _series_value_counts(y_series),
    }


def _series_value_counts(y_series) -> dict:
    counts = y_series.value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _serialize_sizes(sizes: dict) -> list:
    return [
        {"cluster_id": int(cid), "count": int(v["count"]), "percentage": float(v["pct"])}
        for cid, v in sizes.items()
    ]


def _serialize_diag(diag_dist, k: int) -> list:
    rows = []
    for cid in range(k):
        try:
            row = diag_dist.loc[cid]
            rows.append({
                "cluster_id":    cid,
                "benign":        int(row["B"]),
                "malignant":     int(row["M"]),
                "pct_malignant": float(row["pct_M"]),
            })
        except KeyError:
            rows.append({
                "cluster_id":    cid,
                "benign":        0,
                "malignant":     0,
                "pct_malignant": 0.0,
            })
    return rows


def _serialize_profiles(profiles) -> list:
    return [
        {
            "cluster_id": int(idx),
            "mean_features": {col: float(row[col]) for col in profiles.columns},
        }
        for idx, row in profiles.iterrows()
    ]


if __name__ == "__main__":
    print("Running ML Pipeline validation...")
    res = run_pipeline()
    print("Pipeline finished successfully!")
    print(f"  Samples: {res['sample_count']} | Features: {res['feature_count']}")
    print(f"  Grid: {res['config']['grid']} | Clusters: {res['cluster_count']}")
    print(f"  QE: {res['quantization_error']:.4f} | TE: {res['topographic_error']:.4f}")
    print(f"  Silhouette: {res['silhouette_score']:.4f} | ARI: {res['ari']:.4f}")

