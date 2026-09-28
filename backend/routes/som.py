"""
backend/routes/som.py
---------------------
GET /api/som — returns SOM grid, U-Matrix, hit map, and BMU summary.
"""

from fastapi import APIRouter, HTTPException

from backend import pipeline as svc

router = APIRouter()


@router.get("/som", summary="SOM results")
def som():
    """Return SOM grid dimensions, U-Matrix, hit map, and BMU information."""
    if not svc.is_trained():
        raise HTTPException(status_code=404, detail="Pipeline not trained yet. Call POST /api/train first.")

    result = svc.get_result()
    cfg    = result["config"]

    return {
        "grid":        cfg["grid"],
        "grid_rows":   cfg["n_rows"],
        "grid_cols":   cfg["n_cols"],
        "n_neurons":   cfg["n_rows"] * cfg["n_cols"],
        "umatrix":     result["umatrix"],
        "hit_map":     result["hit_map"],
        "bmu_summary": {
            "total_samples": result["sample_count"],
            "grid_rows":     cfg["n_rows"],
            "grid_cols":     cfg["n_cols"],
        },
    }
