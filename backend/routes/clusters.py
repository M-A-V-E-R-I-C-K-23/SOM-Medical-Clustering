"""
backend/routes/clusters.py
--------------------------
GET /api/clusters — returns clustering results from the trained pipeline.
"""

from fastapi import APIRouter, HTTPException

from backend import pipeline as svc

router = APIRouter()


@router.get("/clusters", summary="Clustering results")
def clusters():
    """Return cluster count, sizes, percentages, feature profiles, and diagnosis distribution."""
    if not svc.is_trained():
        raise HTTPException(status_code=404, detail="Pipeline not trained yet. Call POST /api/train first.")

    result = svc.get_result()
    return {
        "cluster_count":          result["cluster_count"],
        "cluster_sizes":          result["cluster_sizes"],
        "diagnosis_distribution": result["diag_dist"],
        "feature_profiles":       result["feature_profiles"],
    }
