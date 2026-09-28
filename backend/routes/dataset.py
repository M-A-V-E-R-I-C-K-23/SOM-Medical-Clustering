"""
backend/routes/dataset.py
-------------------------
GET /api/dataset — returns dataset info without requiring prior training.
"""

from fastapi import APIRouter, HTTPException

from backend import pipeline as svc

router = APIRouter()


@router.get("/dataset", summary="Dataset information")
def dataset():
    """Return sample count, feature count, feature names, and diagnosis distribution."""
    try:
        return svc.get_dataset_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {exc}") from exc
