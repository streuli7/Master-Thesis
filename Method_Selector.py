"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data 
Visualization and Decision Support

Author: Nils Besson

Description:
Benchmarks clustering approaches for avalanche forecasting data by comparing
HDBSCAN variants (normal, baseline, ensemble) and K-Means against SLF bulletin
labels. The script cleans and aligns grid features with bulletins, runs each
clustering method, computes ARI scores, and outputs results as tables (CSV/LaTeX).

Note: Parts of this script (e.g., documentation and header text) were drafted
with the assistance of AI (ChatGPT). All code has been reviewed and adapted by the author.
"""

# ----------- Load libraries ----------- #
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.manifold import SpectralEmbedding
from joblib import Parallel, delayed

from scipy import sparse
import hdbscan

import os
import warnings
import random

# ------------ Disable Warnings ------------ #
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
np.seterr(all="ignore")
random.seed(42)


# ------------- Paths ------------- #
BASE_DIR         = Path(r"Path/to/data/directory")
INPUT_CSV        = BASE_DIR / "aggregated_grid_time_series_lv03.csv" #Output of Clustering_Pipeline.py
BULLETIN_CSV     = BASE_DIR / "bulletin_export_2023-2024.csv" #Output of Bulletin_Exporter.py

# Column names
GRID_DATE_COL = "date"
GRID_GRID_COL = "sector_id"
BULL_DATE_COL = "valid_from"
BULL_GRID_COL = "sector_id"
BULL_LABEL_COL = "level_numeric"   # ground truth

# Non-feature columns to exclude from clustering in the grid file:
EXCLUDE_GRID_COLS = {GRID_DATE_COL, "grid_id", "x", "y", "east", "north", "sector_id"}

# K for K-Means; 5 matches avalanche danger levels 1..5
K_FOR_KMEANS = 5

# Optional outputs
WRITE_RESULTS_CSV   = "ari_results.csv"
WRITE_RESULTS_TEX   = "ari_results.tex"

# ---------- Helper Functions ---------- #
def _to_int_id(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = np.floor(s + 0.5)
    return s.astype("Int64")

def _clean_features(df_num: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    Xdf = df_num.copy()

    # Ensure float dtype
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    # ±inf -> NaN
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    # Drop all-NaN columns
    all_nan_cols = [c for c in Xdf.columns if Xdf[c].isna().all()]
    if all_nan_cols and verbose:
        print(f"[clean] Dropping all-NaN cols: {all_nan_cols}")
    Xdf = Xdf.drop(columns=all_nan_cols, errors="ignore")

    # Drop near-constant columns
    std = Xdf.std(ddof=0).fillna(0.0)
    near_const = std[std <= 1e-12].index.tolist()
    if near_const and verbose:
        print(f"[clean] Dropping near-constant cols: {near_const}")
    Xdf = Xdf.drop(columns=near_const, errors="ignore")

    # Median impute
    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med)

    # Final guard: if any NaNs remain, drop those rows
    remaining_nan_rows = Xdf.index[Xdf.isna().any(axis=1)].tolist()
    if remaining_nan_rows:
        print(f"[clean] Dropping {len(remaining_nan_rows)} rows with residual NaNs after impute.")
        Xdf = Xdf.drop(index=remaining_nan_rows)

    # Ensure float64
    Xdf = Xdf.astype(np.float64)

    return Xdf

def _sparse_topk(G: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    G = G.tolil(copy=True)
    n = G.shape[0]
    for i in range(n):
        row = G.rows[i]
        data = G.data[i]
        if not row:
            continue
        # Drop self if present
        if i in row:
            j = row.index(i)
            row.pop(j); data.pop(j)
        if len(row) > k:
            # Keep indices of top-k by data
            top_idx = np.argpartition(data, -k)[-k:]
            new_rows = [row[j] for j in top_idx]
            new_data = [data[j] for j in top_idx]
            G.rows[i] = new_rows
            G.data[i] = new_data
    return G.tocsr()

# ------------- Load and align data from CSV files ------------- #
def load_and_align():
    # Read
    grid = pd.read_csv(INPUT_CSV)
    bull = pd.read_csv(BULLETIN_CSV)

    # Strip columns
    grid.columns = grid.columns.str.strip()
    bull.columns = bull.columns.str.strip()

    # Parse dates
    if GRID_DATE_COL not in grid.columns:
        raise ValueError(f"Grid CSV missing date col '{GRID_DATE_COL}'. Have: {list(grid.columns)[:20]}")
    if BULL_DATE_COL not in bull.columns:
        raise ValueError(f"Bulletin CSV missing date col '{BULL_DATE_COL}'. Have: {list(bull.columns)[:20]}")

    grid_dt = pd.to_datetime(grid[GRID_DATE_COL], errors="coerce")
    bull_dt = pd.to_datetime(bull[BULL_DATE_COL], errors="coerce")
    grid["_date_key"] = grid_dt.dt.date
    bull["_date_key"] = bull_dt.dt.date

    # Normalize sector_id to integers on both sides
    if GRID_GRID_COL in grid.columns:
        grid = grid.dropna(subset=[GRID_GRID_COL]).copy()
        grid[GRID_GRID_COL] = _to_int_id(grid[GRID_GRID_COL])
    else:
        raise ValueError(f"Grid CSV missing spatial key '{GRID_GRID_COL}'. Have: {list(grid.columns)[:20]}")

    if BULL_GRID_COL in bull.columns:
        bull = bull.dropna(subset=[BULL_GRID_COL]).copy()
        bull[BULL_GRID_COL] = _to_int_id(bull[BULL_GRID_COL])
    else:
        raise ValueError(f"Bulletin CSV missing spatial key '{BULL_GRID_COL}'. Have: {list(bull.columns)[:20]}")

    # Label
    if BULL_LABEL_COL not in bull.columns:
        raise ValueError(f"Bulletin CSV missing label '{BULL_LABEL_COL}'. Have: {list(bull.columns)[:20]}")
    bull = bull.dropna(subset=[BULL_LABEL_COL])

    # Collapse bulletin to one row per (date, sector)
    bull = bull.assign(_dt=bull_dt)
    bull = bull.sort_values(["_date_key", BULL_GRID_COL, "_dt"]) \
               .drop_duplicates(["_date_key", BULL_GRID_COL], keep="last")

    # Minimal bulletin view for merge
    bull_small = bull[["_date_key", BULL_GRID_COL, BULL_LABEL_COL]].copy()

    # Merge grid and bulletin
    merged = pd.merge(
        grid,
        bull_small,
        left_on=["_date_key", GRID_GRID_COL],
        right_on=["_date_key", BULL_GRID_COL],
        how="inner",
        validate="many_to_one"
    ).dropna(subset=[BULL_LABEL_COL]).reset_index(drop=True)

    if merged.empty:
        raise ValueError("After merging, no rows remained. Check matching dates/sector_id after normalization.")

    # Features: numeric columns except excluded + label
    non_features = set(EXCLUDE_GRID_COLS) | {"_date_key", BULL_LABEL_COL}
    num_cols = [c for c in merged.columns
                if c not in non_features and np.issubdtype(merged[c].dtype, np.number)]
    if not num_cols:
        raise ValueError("No numeric feature columns found in GRID after exclusions.\n"
                         f"Sample columns: {merged.columns[:25].tolist()}")

    # Clean features
    Xdf_raw = merged[num_cols].copy()
    Xdf = _clean_features(Xdf_raw, verbose=True)

    # Align labels to kept rows
    kept_idx = Xdf.index
    y_true_raw = merged.loc[kept_idx, BULL_LABEL_COL].to_numpy()
    _, y_true = np.unique(y_true_raw, return_inverse=True)

    print(f"[ok] Merged rows: {len(merged)} | features (pre-clean): {len(num_cols)} "
          f"| label='{BULL_LABEL_COL}' | keys: ['_date_key','{GRID_GRID_COL}'] | bulletin deduped per day/sector")
    print(f"[ok] After cleaning: rows={len(Xdf)} | features={Xdf.shape[1]}")

    return Xdf.reset_index(drop=True), y_true

# ---------- Clustering Variants ---------- #
@dataclass
class RunResult:
    name: str
    labels: np.ndarray
    ari: float

def hdbscan_normal(X: np.ndarray) -> np.ndarray:
    return hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False
    ).fit_predict(X)

def hdbscan_baseline(X: np.ndarray) -> np.ndarray:
    Xr = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
    ]).fit_transform(X)
    return hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False
    ).fit_predict(Xr)

def _one_hdbscan(X: np.ndarray, params: Dict) -> np.ndarray:
    return hdbscan.HDBSCAN(**params).fit_predict(X)

def hdbscan_ensemble(X: np.ndarray) -> np.ndarray:
    # Preprocessing
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
    ])
    Xr = base.fit_transform(X)

    # Parameter grid
    grid: List[Dict] = []
    for mcs in (10, 15, 20, 30):
        for ms in (None, 5, 10):
            for method in ("eom", "leaf"):
                grid.append(dict(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric="euclidean",
                    cluster_selection_method=method
                ))

    labels_list = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_one_hdbscan)(Xr, g) for g in grid
    )

    n = Xr.shape[0]
    edge_i: List[int] = []
    edge_j: List[int] = []
    edge_w: List[float] = []
    nonnoise_counts = np.zeros(n, dtype=np.int32)

    # Sparse co-association edges
    for lab in labels_list:
        clustered = np.where(lab >= 0)[0]
        nonnoise_counts[clustered] += 1
        if clustered.size == 0:
            continue
        labs = lab[clustered]
        for cid in np.unique(labs):
            idx = clustered[labs == cid]
            if idx.size < 2:
                continue
            ii, jj = np.triu_indices(idx.size, k=1)
            edge_i.extend(idx[ii]); edge_j.extend(idx[jj])
            edge_w.extend(np.ones(ii.size, dtype=np.float32))

    if len(edge_i) == 0:
        # Degenerate: no co-clustering at all, fall back to baseline HDBSCAN
        return hdbscan_baseline(X)

    # Build symmetric graph & normalize by sqrt
    G = sparse.coo_matrix((edge_w, (edge_i, edge_j)), shape=(n, n), dtype=np.float32)
    G = G + G.T
    nn = nonnoise_counts.astype(np.float32)
    d_inv_sqrt = np.reciprocal(np.sqrt(np.maximum(nn, 1.0)))
    D_left = sparse.diags(d_inv_sqrt)
    Gn = D_left @ G @ D_left  # normalized similarity

    # Keep top-K neighbors per node to denoise graph
    KNN = 30
    Gn = _sparse_topk(Gn.tocsr(), k=KNN)

    # Spectral embedding
    embed_dim = 16
    Gn = Gn.maximum(0)
    se = SpectralEmbedding(
        n_components=embed_dim,
        affinity="precomputed",
        random_state=42,
        eigen_solver="arpack",
    )
    Y = se.fit_transform(Gn)

    # Final HDBSCAN
    final = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    return final.fit_predict(Y)

def kmeans_run(X: np.ndarray, k: int) -> np.ndarray:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("km", KMeans(n_clusters=k, n_init=20, random_state=42))
    ]).fit_predict(X)

# ----------- Evaluation and Output ----------- #
def evaluate_all(X: np.ndarray, y_true: np.ndarray, k_for_kmeans: int) -> List[RunResult]:
    runs = [
        ("HDBSCAN* (normal)",   hdbscan_normal(X)),
        ("HDBSCAN* (baseline)", hdbscan_baseline(X)),
        ("HDBSCAN* (ensemble)", hdbscan_ensemble(X)),
        ("K-Means",             kmeans_run(X, k_for_kmeans)),
    ]
    return [RunResult(name, lab, adjusted_rand_score(y_true, lab)) for name, lab in runs]

def to_table(results: List[RunResult]) -> pd.DataFrame:
    order = ["HDBSCAN* (normal)", "HDBSCAN* (baseline)", "HDBSCAN* (ensemble)", "K-Means"]
    df = pd.DataFrame({"Approach":[r.name for r in results], "ARI":[float(r.ari) for r in results]})
    df = df.set_index("Approach").loc[order].reset_index()
    df["ARI"] = df["ARI"].round(3)
    return df

def to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(index=False, column_format="lr", float_format="%.3f")

# ----------- Main ----------- #
if __name__ == "__main__":
    # Load & align
    Xdf, y_true = load_and_align()
    X = Xdf.to_numpy()

    # Choose k (defaults to 5)
    k = K_FOR_KMEANS if K_FOR_KMEANS else len(np.unique(y_true))

    results = evaluate_all(X, y_true, k)
    table = to_table(results)

    print("\nAdjusted Rand Index (higher is better)\n")
    print(table.to_string(index=False))

    if WRITE_RESULTS_CSV:
        table.to_csv(WRITE_RESULTS_CSV, index=False)
    if WRITE_RESULTS_TEX:
        with open(WRITE_RESULTS_TEX, "w", encoding="utf-8") as f:
            f.write(to_latex(table))
        print(f"\nSaved LaTeX to {WRITE_RESULTS_TEX}")
