"""
backend/routes/train.py
-----------------------
POST /api/train — runs the full ML pipeline and returns key metrics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend import pipeline as svc

router = APIRouter()


class TrainRequest(BaseModel):
    n_rows:        int   = Field(9,      ge=3, le=20)
    n_cols:        int   = Field(9,      ge=3, le=20)
    k:             int   = Field(3,      ge=2, le=10)
    n_iterations:  int   = Field(10_000, ge=100, le=50_000)
    sigma:         float = Field(1.5,    gt=0)
    learning_rate: float = Field(0.5,    gt=0, le=1.0)
    seed:          int   = Field(42)


@router.post("/train", summary="Train the SOM pipeline")
def train(req: TrainRequest = TrainRequest()):
    """Run the complete ML pipeline end-to-end and store results in memory."""
    try:
        result = svc.run_pipeline(
            n_rows        = req.n_rows,
            n_cols        = req.n_cols,
            k             = req.k,
            n_iterations  = req.n_iterations,
            sigma         = req.sigma,
            learning_rate = req.learning_rate,
            seed          = req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    cfg = result["config"]
    return {
        "status":      "success",
        "message":     "Pipeline trained. Query /api/som, /api/clusters, /api/evaluation for results.",
        "grid":        cfg["grid"],
        "k":           cfg["k"],
        "n_iterations": cfg["n_iterations"],
        "sample_count":       result["sample_count"],
        "feature_count":      result["feature_count"],
        "quantization_error": result["quantization_error"],
        "topographic_error":  result["topographic_error"],
        "silhouette_score":   result["silhouette_score"],
        "ari":                result["ari"],
    }
