"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting

Author: Nils Besson

Description:
Loads grid-level model and bulletin data, computes rolling means, variances,
lags, first- and second-order deltas and merges labels; builds a reusable
sklearn Pipeline (impute→scale→poly→RF select→PCA→LDA); then for each date
applies the transformer, optionally adds spatial/PCA features, runs HDBSCAN
with tuned params, post-processes noise/small clusters, and saves per-date
cluster CSVs, then merges them into one file.

Note: Parts of this script (e.g., documentation and header text) were drafted
with the assistance of AI (ChatGPT). All code has been reviewed and adapted by the author.
"""

# ------------ Load libraries ------------ #
import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import hdbscan

# -------- Configuration --------#
DATA_DIR        = Path(r"Path/to/data/directory")
BYPRODUCTS_DIR  = DATA_DIR / "byproducts"
BYPRODUCTS_DIR.mkdir(exist_ok=True)
INPUT_CSV       = DATA_DIR / "aggregated_grid_time_series_lv03.csv" #output file of Grid_Summary.py
BULLETIN_CSV    = DATA_DIR / "bulletin_export_2023-2024.csv" #your bulletin file
START_DATE      = "2024-02-01"
END_DATE        = "2024-04-30"
RANDOM_STATE    = 42

# Tuned HDBSCAN hyperparameters
TUNED_PARAMS = {
    'min_cluster_size': 100,
    'min_samples': 100,
    'eps': 0.02799882442598489,
    'cluster_selection_method': 'leaf',
    'metric': 'correlation'
}

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

# -------- Utility functions -------- #
def reassign_noise(labels, X):
    labels = labels.copy()
    core = labels != -1
    if not core.any():
        return labels
    Xc, Lc = X[core], labels[core]
    nn = NearestNeighbors(n_neighbors=1).fit(Xc)
    for i in np.where(labels == -1)[0]:
        _, nbr = nn.kneighbors([X[i]])
        labels[i] = Lc[nbr[0][0]]
    return labels

def prune_small(labels, X, thresh):
    vals, cts = np.unique(labels, return_counts=True)
    small = vals[cts < thresh]
    labels[np.isin(labels, small)] = -1
    return reassign_noise(labels, X)

# -------- Data loading & pipeline building -------- #
def load_and_prepare_full(start_date, end_date):
    df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
    orig_feats = df.select_dtypes(include=[np.number])\
                  .columns.difference(['sector_id','x','y']).tolist()

    df = df.sort_values(['sector_id','date'])
    rolling = df.groupby('sector_id')[orig_feats].rolling(3, min_periods=1)
    df[orig_feats] = rolling.mean().reset_index(level=0, drop=True)
    df[[f"{c}_var3" for c in orig_feats]] = rolling.var().reset_index(level=0, drop=True)

    for lag in (1,2,3):
        df[[f"{c}_lag{lag}" for c in orig_feats]] = df.groupby('sector_id')[orig_feats].shift(lag)
    for c in orig_feats:
        df[f"{c}_delta1"] = df[c] - df[f"{c}_lag1"]
        df[f"{c}_delta2"] = df[f"{c}_lag1"] - df[f"{c}_lag2"]

    df = df[(df.date >= pd.to_datetime(start_date)) & (df.date <= pd.to_datetime(end_date))].copy()

    bull = pd.read_csv(BULLETIN_CSV, parse_dates=['valid_from','valid_to'])
    bull['date'] = bull['valid_from'].dt.normalize()
    bull = bull[['date','sector_id','level_numeric']]
    df = df.merge(bull, on=['date','sector_id'], how='inner')

    feats = df.select_dtypes(include=[np.number])\
              .columns.difference(['sector_id','level_numeric']).tolist()
    X = df[feats].values
    y = df['level_numeric'].astype(int).values
    return df, X, y, orig_feats, feats

def build_transformer(X, y):
    n_cls = len(np.unique(y))
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(2, include_bias=False)),
        ('selector', SelectFromModel(
            RandomForestClassifier(100, random_state=RANDOM_STATE),
            threshold='median'
        )),
        ('pca', PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ('lda', LDA(n_components=min(n_cls-1, X.shape[1])))
    ])
    pipe.fit(X, y)
    return pipe

# -------- Clustering per date -------- #
from sklearn.neighbors import NearestNeighbors

def batch_lda_transform(pipe, X, batch_size=5000):
    """Transform X through the full pipeline (up to LDA) in batches."""
    out = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        Xb = X[i : i + batch_size]
        out.append(pipe.transform(Xb))
    return np.vstack(out)

def cluster_date(date, orig_feats, feats, transformer,
                 min_cluster_size, min_samples, eps,
                 cluster_selection_method, metric,
                 sample_per_lvl=100, random_state=42,
                 use_spatial=True, use_pca=True):

    # Load the full grid × date series
    df_ts = pd.read_csv(INPUT_CSV, parse_dates=['date'])

    # Compute rolling means & variances on the original numeric features
    df_ts = df_ts.sort_values(['sector_id','date'])
    rolling = df_ts.groupby('sector_id')[orig_feats].rolling(3, min_periods=1)
    df_ts[orig_feats] = rolling.mean().reset_index(level=0, drop=True)
    df_ts[[f"{c}_var3" for c in orig_feats]] = (
        rolling.var().reset_index(level=0, drop=True)
    )

    # Compute 1–3 day lags and deltas
    for lag in (1, 2, 3):
        df_ts[[f"{c}_lag{lag}" for c in orig_feats]] = \
            df_ts.groupby('sector_id')[orig_feats].shift(lag)
    for c in orig_feats:
        df_ts[f"{c}_delta1"] = df_ts[c] - df_ts[f"{c}_lag1"]
        df_ts[f"{c}_delta2"] = df_ts[f"{c}_lag1"] - df_ts[f"{c}_lag2"]

    # Filter to this date
    sel_ts = pd.to_datetime(date)
    df0    = df_ts[df_ts['date'] == sel_ts].copy()
    if df0.empty:
        print(f"[SKIP {date}] no grid data")
        return

    # Merge in SLF bulletin labels so level_numeric exists
    bull = pd.read_csv(BULLETIN_CSV, parse_dates=['valid_from','valid_to'])
    bull['start'] = bull['valid_from'].dt.normalize()
    bull['end']   = bull['valid_to']  .dt.normalize()
    bull = (
        bull
        .assign(date=lambda d: d.apply(
            lambda r: pd.date_range(r['start'], r['end']), axis=1
        ))
        .explode('date')
        [['date','sector_id','level_numeric']]
    )
    bull = bull[bull['date'] == sel_ts]
    df0  = df0.merge(bull, on=['date','sector_id'], how='inner')
    if df0.empty:
        print(f"[SKIP {date}] no SLF labels")
        return

    # Class‐balanced sampling
    pieces = []
    for lvl in sorted(df0['level_numeric'].unique()):
        grp = df0[df0['level_numeric'] == lvl]
        pieces.append(
            grp.sample(min(sample_per_lvl, len(grp)),
                       random_state=random_state)
        )
    samp = pd.concat(pieces, ignore_index=True)

    # Transform the sample into LDA space
    X_samp     = samp[feats].values
    X_samp_lda = transformer.transform(X_samp)

    # Cluster the sample
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size         = min_cluster_size,
        min_samples              = min_samples,
        cluster_selection_epsilon= eps,
        cluster_selection_method = cluster_selection_method,
        metric                   = metric,
        prediction_data          = True
    ).fit(X_samp_lda)
    try:
        labels_samp, _ = hdbscan.approximate_predict(clusterer, X_samp_lda)
    except AttributeError:
        labels_samp = clusterer.labels_
    labels_samp = reassign_noise(labels_samp, X_samp_lda)
    labels_samp = prune_small(labels_samp, X_samp_lda, thresh=min_cluster_size)

    # Build 1-NN on the sample’s LDA embedding
    nn = NearestNeighbors(n_neighbors=1).fit(X_samp_lda)

    # Transform the full grid (df0) into LDA in batches
    X0     = df0[feats].values
    X0_lda = batch_lda_transform(transformer, X0, batch_size=5000)

    # Propagate labels via nearest sample
    nbr_idxs    = nn.kneighbors(X0_lda, return_distance=False)[:,0]
    grid_labels = labels_samp[nbr_idxs]

    # Re-join to a fresh full-grid slice for this date to not drop gird cells
    meta = pd.read_csv(INPUT_CSV, parse_dates=['date'],
                       usecols=['grid_id','date','x','y'])
    full_date = meta[meta['date'] == sel_ts].copy()
    # Attach cluster_id
    full_date = full_date.merge(
        pd.DataFrame({
            'grid_id': df0['grid_id'],
            'cluster_id': grid_labels
        }),
        on='grid_id', how='left'
    )
    full_date['cluster_id'] = full_date['cluster_id'].fillna(-1).astype(int)

    # Save the per-date file with every grid cell present
    out_file = BYPRODUCTS_DIR / f"grid_clusters_hdbscan_{date}.csv"
    full_date.to_csv(out_file, index=False)
    print(f"[CLUSTERED {date}] -> {out_file} (n={len(full_date)})")


# -------- Main function -------- #
def main():
    df_full, X_full, y_full, orig_feats, feats = load_and_prepare_full(
        START_DATE, END_DATE
    )
    transformer = build_transformer(X_full, y_full)

    # Per‐date clustering
    for dt in sorted(df_full['date'].dt.date.unique()):
        cluster_date(
            date=dt,
            orig_feats=orig_feats,
            feats=feats,
            transformer=transformer,
            **TUNED_PARAMS,
            use_spatial=True,
            use_pca=True
        )

    # Merge into one file
    pattern = str(BYPRODUCTS_DIR / "grid_clusters_hdbscan_*.csv")
    files   = sorted(glob.glob(pattern))
    merged  = pd.concat(
        (pd.read_csv(f, parse_dates=['date']) for f in files),
        ignore_index=True
    )
    out_all = DATA_DIR / "merged_cluster_data_all_dates.csv"
    merged.to_csv(out_all, index=False)
    print(f"[MERGED ALL DATES] -> {out_all}")

if __name__ == "__main__":
    main()
