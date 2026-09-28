# Medical Data Clustering Using Self-Organizing Maps (SOM)

> **B.Tech Neural Networks Course Project**  
> *An Unsupervised SOM-Based Approach for Discovering Phenotypic Subgroups in Breast Cancer Medical Data*

---

## Executive Summary

This project implements a complete, unsupervised machine learning pipeline using **Self-Organizing Maps (SOM)** and **K-Means Clustering** on the **Breast Cancer Wisconsin (Diagnostic)** dataset. 

The pipeline projects complex high-dimensional patient profiles (30 continuous numerical features) onto a topological 2D neuron lattice. It reveals natural clinical clusters and risk stratifications **without using diagnostic labels during training**. Diagnostic labels (`Malignant` / `Benign`) are held out strictly for post-hoc clinical validation via the **Adjusted Rand Index (ARI)** and cluster purity analysis.

A production-ready **FastAPI** backend exposes the trained model, metrics, topological matrices, and cluster statistics via RESTful APIs.

---

## Core ML Pipeline & Methodology

```text
+-----------------------------------------------------------------------------+
|                           UNSUPERVISED PIPELINE                             |
+-----------------------------------------------------------------------------+
|  1. Ingestion: 569 samples x 30 numerical features                          |
|     |                                                                       |
|     +---> X (Features: 569 x 30) ----------+                                |
|     +---> y (Diagnosis: 569) [ISOLATED] ---|------------------------------+ |
|                                            v                              | |
|  2. Preprocessing: StandardScaler fitted on X only                        | |
|     |                                                                     | |
|     v                                                                     | |
|  3. SOM Training: 9x9 Grid (81 neurons), 10,000 iterations                | |
|     (Gaussian neighborhood, cosine decay learning rate)                   | |
|     |                                                                     | |
|     +---> Topological Codebook Vectors (81 x 30)                          | |
|     +---> Sample Best Matching Units (BMUs: 569 x 2)                      | |
|           |                                                               | |
|           v                                                               | |
|  4. Two-Level Clustering: K-Means (K=3) fitted on Codebook Vectors ONLY   | |
|     (Neuron codebooks clustered -> samples assigned cluster via their BMU)| |
|     |                                                                     | |
|     v                                                                     | |
|  5. Cluster Assignment: All 569 samples mapped to 3 clinical clusters     | |
+---------------------------------------------------------------------------+-+
|                          POST-HOC VALIDATION                              | |
+---------------------------------------------------------------------------+-+
|  6. Evaluation Metrics:                                                   | |
|     - Quantization Error (QE)      - Topographic Error (TE)               | |
|     - Silhouette Score (Samples)   - Adjusted Rand Index (ARI with y) <---+ |
+-----------------------------------------------------------------------------+
```

### Key Architectural Principles
1. **Strict Data Isolation**: Ground truth diagnosis labels ($y$) **never** enter feature scaling, SOM training, or K-Means clustering. They are introduced solely at step 6 for post-hoc validation.
2. **Two-Level Clustering**: K-Means is performed on the **SOM codebook vectors (81 x 30)** rather than raw sample points. This preserves the topological structure learned by the SOM, smooths local noise, and significantly improves cluster stability.
3. **Reproducibility**: All random operations utilize a fixed seed (`seed = 42`).
4. **Dual Silhouette Evaluation**:
   - **K-Selection Routine (`select_k_elbow`)**: Evaluates the **codebook silhouette score** on the flattened prototype vectors (`codebook_flat` $\in \mathbb{R}^{N_{\text{neurons}} \times 30}$) across candidate $K \in [2, 8]$ alongside WCSS inertia to identify the optimal number of clusters ($K=3$, where codebook silhouette is 0.3687 on the 9x9 lattice).
   - **Final Pipeline Evaluation (`compute_all_metrics`)**: Evaluates the **sample-space silhouette score (`0.3429`)** on standardized patient samples ($X_{\text{scaled}} \in \mathbb{R}^{569 \times 30}$) with their final cluster assignments (`sample_clusters`). This quantifies clinical cluster cohesion and separation in the original 30-dimensional continuous feature space.

---

## Validated Results & Benchmark

### 1. Final Validated Metrics (Selected 9x9 Grid, K=3)

| Metric | Validated Value | Interpretation |
| :--- | :---: | :--- |
| **Quantization Error (QE)** | `2.3434` | Average Euclidean distance between samples and their BMU. |
| **Topographic Error (TE)** | `0.1441` | Proportion of samples whose 1st and 2nd BMUs are not adjacent; measures topology preservation. |
| **Silhouette Score (Sample Space)** | `0.3429` | Cluster cohesion vs. separation evaluated on standardized patient samples ($X_{\text{scaled}}$, 569 samples) with their BMU-derived cluster assignments. *(Note: Codebook-space silhouette is computed separately during K-selection on neuron prototypes, yielding 0.3687 for K=3 on 9x9).* |
| **Adjusted Rand Index (ARI)** | `0.5070` | Post-hoc concordance with true diagnosis (chance = 0.0, perfect = 1.0). |

### 2. Grid Optimization & Selection

Three grid architectures were trained and systematically benchmarked using an equal-weight composite score across normalized QE, TE, Silhouette (sample space), and ARI:

| Grid Size | Neurons | QE (lower better) | TE (lower better) | Silhouette (Samples, higher better) | ARI (higher better) | Composite Score | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **9 x 9** | **81** | **2.3434** | **0.1441** | **0.3429** | **0.5070** | **0.5577** | **Selected Best Configuration** |
| 11 x 11 | 121 | 2.1304 | 0.1599 | 0.3217 | 0.4964 | 0.1647 | Initial Phase 3 Baseline |
| 13 x 13 | 169 | 1.9689 | 0.1599 | 0.3196 | 0.5421 | 0.5000 | Evaluated Alternative |

*Note: While larger grids achieve lower quantization error by allocating more neurons, the 9x9 grid provides superior topology preservation (lower TE), better sample-space cluster separation (Silhouette = 0.3429 on standardized samples), and achieves the highest composite rank. The K-selection routine separately analyzes codebook-space silhouette on the flattened neuron weights to identify optimal K.*

### 3. Discovered Clinical Clusters

| Cluster | Total Samples | % of Cohort | Benign (B) | Malignant (M) | % Malignant | Clinical Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cluster 0** | 385 | 67.7% | 332 | 53 | 13.8% | **Low-risk / Predominantly Benign**: Small cell nuclei, uniform texture, low concavity. |
| **Cluster 1** | 122 | 21.4% | 0 | 122 | **100.0%** | **High-risk / Pure Malignant**: Markedly enlarged nuclear radius, irregular boundaries, extreme concavity. |
| **Cluster 2** | 62 | 10.9% | 25 | 37 | 59.7% | **Intermediate / High-grade Heterogeneity**: Borderline features, moderate pleomorphism requiring closer monitoring. |

---

## Project Structure

```text
medical-som-clustering/
|-- requirements.txt              # Unified dependencies for ML pipeline & FastAPI backend
|-- pyproject.toml                # Project packaging configuration & import resolution
|-- README.md                     # Project documentation
|-- AGENTS.md                     # Development guidelines & Ponytail ruleset
|
|-- ml/                           # Machine Learning Core
|   |-- __init__.py
|   |-- data/
|   |   +-- breast_cancer.csv     # Local clean copy of UCI Wisconsin Diagnostic dataset
|   |-- notebooks/
|   |   +-- som_analysis.ipynb    # Full exploratory analysis & experiments (59/59 cells PASS)
|   +-- src/
|       |-- __init__.py
|       |-- preprocessing.py      # Dataset loading, X/y separation, StandardScaler pipeline
|       |-- som_model.py          # MiniSom initialization, training wrapper, codebook extraction
|       |-- clustering.py         # K-Means on codebook vectors, sample BMU cluster assignment
|       +-- evaluation.py         # QE, TE, Silhouette, ARI, composite scoring utilities
|
|-- backend/                      # Production FastAPI REST Backend (Clean & Consolidated)
|   |-- __init__.py
|   |-- main.py                   # FastAPI app, Pydantic response models, and 5 REST endpoints
|   +-- pipeline.py               # In-memory ML pipeline orchestration service
|
|-- results/
|   |-- metrics.json              # Canonical verified benchmark metrics & grid comparison
|   +-- figures/                  # Publication-ready visualizations
|       |-- 14_umatrix_9x9_final.png              # SOM Distance matrix (U-Matrix)
|       |-- 15_hitmap_9x9_final.png               # Sample density per neuron
|       |-- 16_cluster_map_9x9_final.png          # 2D topological cluster assignments
|       |-- 17_diagnosis_per_cluster_9x9_final.png# Post-hoc clinical distribution
|       +-- 18_pca_scatter_9x9_final.png          # PCA projection of discovered clusters
|
+-- frontend/                     # Interactive Web Dashboard (Phase 7 - Upcoming)
```

---

## Installation & Getting Started

### 1. Prerequisites
- Python 3.10 to 3.13
- Git

### 2. Environment Setup

Clone the repository and install all required dependencies:

```bash
# Navigate to the project root
cd medical-som-clustering

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Editable installation for standard package import resolution:
```bash
pip install -e .
```

---

## Running the Project

### A. Run End-to-End Pipeline Validation
To execute the complete end-to-end unsupervised pipeline and verify metrics:

```bash
python -m backend.pipeline
```
*Expected output: `Pipeline finished successfully!` with verified metrics (QE ~2.34, TE ~0.14, Silhouette ~0.34, ARI ~0.51).*

---

### B. Launch the FastAPI Backend Server
Run the REST API from the project root:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Once running, access the interactive API documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/](http://localhost:8000/)

---

### C. Run the Jupyter Notebook Analysis
Open and execute the end-to-end exploratory data analysis and SOM experiments:

```bash
jupyter notebook ml/notebooks/som_analysis.ipynb
```

---

## REST API Specification

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `POST` | `/api/train` | Executes the full pipeline with configurable parameters (`n_rows`, `n_cols`, `k`, `n_iterations`, `seed`, `sigma`, `learning_rate`) and updates in-memory state. |
| `GET` | `/api/dataset` | Returns feature statistics, row counts (569), feature names (30), and ground-truth label distributions. |
| `GET` | `/api/som` | Returns the 9x9 U-Matrix, neuron hit map, training error history, QE, and TE. |
| `GET` | `/api/clusters` | Returns codebook cluster assignments, sample cluster distributions, and post-hoc benign/malignant proportions. |
| `GET` | `/api/evaluation` | Returns the final four evaluation metrics (QE, TE, sample-space Silhouette, ARI) and full grid comparison table. |
| `GET` | `/` | Service health status and registered endpoint directory. |

### Example API Request: Train Pipeline
```bash
curl -X POST "http://localhost:8000/api/train" \
     -H "Content-Type: application/json" \
     -d '{
       "n_rows": 9,
       "n_cols": 9,
       "k": 3,
       "n_iterations": 10000,
       "seed": 42
     }'
```

---

## Key Visualizations

All high-resolution figures are automatically generated in [`results/figures/`](results/figures/):

- **U-Matrix (`14_umatrix_9x9_final.png`)**: Visualizes inter-neuron Euclidean distances. Darker boundaries indicate high topological separation between clusters.
- **Hit Map (`15_hitmap_9x9_final.png`)**: Shows sample frequency per neuron, identifying dense patient phenotypes versus sparse transition zones.
- **SOM Cluster Map (`16_cluster_map_9x9_final.png`)**: Displays the 2D neuron grid partitioned into 3 distinct codebook clusters.
- **Diagnosis Distribution (`17_diagnosis_per_cluster_9x9_final.png`)**: Post-hoc verification confirming Cluster 1 as 100% Malignant and Cluster 0 as predominantly Benign.
- **PCA Scatter (`18_pca_scatter_9x9_final.png`)**: 2D PCA projection of the original 30D patient features colored by SOM-derived cluster labels.

---

## Code Conventions & Guidelines

- **Package-style imports**: All internal imports follow the package structure:
  ```python
  from ml.src import preprocessing
  from ml.src import som_model
  from ml.src import clustering
  from ml.src import evaluation
  ```
- **Unsupervised discipline**: Diagnosis labels ($y$) are isolated upon ingestion. No pipeline component uses $y$ prior to `evaluation.py`.
- **Typing & Schemas**: All API payloads are strictly validated using Pydantic models in `backend/main.py`.

---

## Roadmap & Milestones

- [x] **Phase 1**: Dataset Ingestion, EDA, and Feature Correlation Analysis
- [x] **Phase 2**: Preprocessing Pipeline (`StandardScaler` on $X$, label separation)
- [x] **Phase 3**: Self-Organizing Map (MiniSom wrapper, decay schedules, codebook extraction)
- [x] **Phase 4**: Two-Level K-Means Clustering on Codebook Vectors & Sample BMU Mapping
- [x] **Phase 5**: Hyperparameter Optimization (9x9 vs 11x11 vs 13x13 grid benchmarking)
- [x] **Phase 6**: FastAPI Backend Implementation & Validation Test Suite (19/19 checks PASS)
- [ ] **Phase 7**: Interactive Web Frontend (React + Vite Dashboard)
