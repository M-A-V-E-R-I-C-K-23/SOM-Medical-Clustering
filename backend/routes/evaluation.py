"""
backend/routes/evaluation.py
-----------------------------
GET /api/evaluation — returns QE, TE, Silhouette, ARI, grid comparison, and QE history.
"""

import json
import os

from fastapi import APIRouter, HTTPException

from backend import pipeline as svc

router = APIRouter()

# Path to the pre-computed benchmark grid comparison (results/metrics.json)
_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "metrics.json",
)


def _load_grid_comparison() -> list:
    """Load the 3-grid benchmark table from results/metrics.json."""
    if os.path.exists(_METRICS_PATH):
        try:
            with open(_METRICS_PATH, encoding="utf-8") as f:
                return json.load(f).get("grid_comparison", [])
        except Exception:
            pass
    return []


@router.get("/evaluation", summary="Evaluation metrics")
def evaluation():
    """Return QE, TE, Silhouette, ARI, QE convergence history, and grid-size comparison table."""
    if not svc.is_trained():
        raise HTTPException(status_code=404, detail="Pipeline not trained yet. Call POST /api/train first.")

    result = svc.get_result()
    return {
        "selected_grid":       result["config"]["grid"],
        "quantization_error":  result["quantization_error"],
        "topographic_error":   result["topographic_error"],
        "silhouette_score":    result["silhouette_score"],
        "adjusted_rand_index": result["ari"],
        "qe_history":          result["qe_history"],
        "grid_comparison":     _load_grid_comparison(),
    }
