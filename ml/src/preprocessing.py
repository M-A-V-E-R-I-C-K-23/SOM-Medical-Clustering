"""
preprocessing.py — Data loading and preprocessing for Medical SOM project.

X/y are always kept separate. y is NEVER passed to SOM or scaler.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst',
]

_CSV_FEATURE_NAMES = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
    'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
    'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2',
    'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2',
    'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3',
    'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3',
]


def load_data(local_csv: str | None = None, wdbc_raw: str | None = None):
    """Load WDBC dataset. Tries ucimlrepo -> local CSV -> raw wdbc.data in order.

    Returns: X (DataFrame 569x30), y (DataFrame 569x1), source (str)
    """
    try:
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=17)
        X = repo.data.features.copy().reset_index(drop=True)
        y = repo.data.targets.copy().reset_index(drop=True)
        return X, y, 'ucimlrepo (online)'
    except Exception:
        pass

    if local_csv and os.path.exists(local_csv):
        try:
            df = pd.read_csv(local_csv)
            X = df[_CSV_FEATURE_NAMES].copy().reset_index(drop=True)
            X.columns = FEATURE_NAMES
            y = df[['Diagnosis']].copy().reset_index(drop=True)
            return X, y, f'local CSV ({local_csv})'
        except Exception:
            pass

    if wdbc_raw and os.path.exists(wdbc_raw):
        try:
            raw = pd.read_csv(wdbc_raw, header=None, names=['id', 'Diagnosis'] + FEATURE_NAMES)
            X = raw[FEATURE_NAMES].copy().reset_index(drop=True)
            y = raw[['Diagnosis']].copy().reset_index(drop=True)
            return X, y, f'raw wdbc.data ({wdbc_raw})'
        except Exception:
            pass

    raise RuntimeError(
        'All data loading methods failed. Provide internet access, '
        'a local breast_cancer.csv, or a wdbc.data path.'
    )


def prepare_features(X: pd.DataFrame, y: pd.DataFrame):
    """Validate X/y and return (X, y_series, feature_names). y is never mixed into X."""
    assert X.shape[1] == 30, f'Expected 30 features, got {X.shape[1]}'
    assert X.shape[0] == y.shape[0], 'X and y row counts do not match'
    assert X.isnull().sum().sum() == 0, 'X contains missing values'
    non_numeric = [c for c in X.columns if X[c].dtype not in [np.float64, np.float32, np.int64]]
    assert len(non_numeric) == 0, f'Non-numeric columns in X: {non_numeric}'
    return X, y['Diagnosis'].copy(), X.columns.tolist()


def scale_features(X: pd.DataFrame):
    """StandardScaler on X only. Returns (X_scaled ndarray, fitted scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6), 'Column means are not ~0'
    assert np.allclose(X_scaled.std(axis=0),  1, atol=1e-6), 'Column stds are not ~1'
    return X_scaled, scaler


def get_preprocessing_summary(X, y_series, X_scaled, feature_names):
    """Return a dict summarising the preprocessing state (used by notebook)."""
    return {
        'n_samples':           X.shape[0],
        'n_features':          X.shape[1],
        'feature_names':       feature_names,
        'X_dtype':             str(X.dtypes.unique()),
        'X_scaled_dtype':      str(X_scaled.dtype),
        'missing_in_X':        int(X.isnull().sum().sum()),
        'class_distribution':  y_series.value_counts().to_dict(),
        'scaled_means_near_0': np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6),
        'scaled_stds_near_1':  np.allclose(X_scaled.std(axis=0),  1, atol=1e-6),
        'y_in_X':              'Diagnosis' in X.columns,
    }


