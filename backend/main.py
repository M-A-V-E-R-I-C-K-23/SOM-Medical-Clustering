"""
backend/main.py
----------------
FastAPI application entry point for the Medical SOM Clustering project.

Start from the project root (medical-som-clustering/) with:

    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

API documentation available at:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import train, dataset, som, clusters, evaluation

# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Medical SOM Clustering API",
    description = (
        "FastAPI backend for the B.Tech Neural Networks project: "
        "Medical Data Clustering Using Self-Organizing Maps (SOM). "
        "\n\n"
        "**Pipeline**: Call `POST /api/train` first, then query the other endpoints. "
        "\n\n"
        "**Dataset**: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository. "
        "569 samples · 30 numerical features."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS — allow the future React frontend (and any local dev origin)
# -----------------------------------------------------------------------------.
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten to e.g. ["http://localhost:3000"] in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────
API_PREFIX = "/api"

app.include_router(train.router,      prefix=API_PREFIX, tags=["Training"])
app.include_router(dataset.router,    prefix=API_PREFIX, tags=["Dataset"])
app.include_router(som.router,        prefix=API_PREFIX, tags=["SOM"])
app.include_router(clusters.router,   prefix=API_PREFIX, tags=["Clusters"])
app.include_router(evaluation.router, prefix=API_PREFIX, tags=["Evaluation"])


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "running",
        "project": "Medical SOM Clustering API",
        "docs":    "/docs",
        "endpoints": {
            "train":      "POST /api/train",
            "dataset":    "GET  /api/dataset",
            "som":        "GET  /api/som",
            "clusters":   "GET  /api/clusters",
            "evaluation": "GET  /api/evaluation",
        },
    }
