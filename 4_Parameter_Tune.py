
"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data 
Visualization and Decision Support

Author: Nils Besson

Description: 
A clustering optimization pipeline for avalanche forecasting data.
It preprocesses grid and bulletin features, applies dimensionality reduction,
and evaluates HDBSCAN ensembles with Optuna hyperparameter tuning.
The script outputs the best parameters and ARI score for model assessment.

Note: Parts of this script (e.g., documentation and header text) were drafted
with the assistance of AI (ChatGPT). All code has been reviewed and adapted by the author.
"""

# ---------- Load libraries ---------- #
import os
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

import hdbscan
import optuna

# ------------ Disable warnings ------------ #
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
np.seterr(all="ignore")

# --------- Configuration -------- #
BASE_DIR        = Path(r"Path/to/data/directory")
INPUT_CSV       = BASE_DIR / "aggregated_grid_time_series_lv03.csv" #Output of Grid_Summary.py
BULLETIN_CSV    = BASE_DIR / "bulletin_export_2023-2024.csv" #Output of Bulletin_Export.py
START_DATE      = "2024-02-01"
END_DATE        = "2024-04-30"
RANDOM_STATE    = 42
SAMPLE_PER_LVL  = 100
USE_SPATIAL     = True

cluster_selection_methods = ['eom', 'leaf']
distance_metrics = ['euclidean', 'correlation']
ENSEMBLE_K      = 5
OPTUNA_TRIALS   = 100

# ---------- Data Loader ---------- #
def load_data(agg_path, bulletins_path, start_date, end_date,
              sample_per_lvl, random_state=RANDOM_STATE, spatial=True):
    df = pd.read_csv(agg_path, parse_dates=['date'])
    if not spatial:
        df = df.drop(columns=['x', 'y'], errors='ignore')

    # Rolling features
    numeric = df.select_dtypes(include=[np.number]).columns.difference(['sector_id', 'x', 'y'])
    df = df.sort_values(['sector_id', 'date'])
    df[numeric] = df.groupby('sector_id')[numeric].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df[[f"{c}_var3" for c in numeric]] = (
        df.groupby('sector_id')[numeric]
          .rolling(3, min_periods=1)
          .var()
          .reset_index(level=0, drop=True)
    )

    for lag in (1, 2, 3):
        df[[f"{c}_lag{lag}" for c in numeric]] = df.groupby('sector_id')[numeric].shift(lag)
    for c in numeric:
        df[f"{c}_delta1"] = df[c] - df[f"{c}_lag1"]
        df[f"{c}_delta2"] = df[f"{c}_lag1"] - df[f"{c}_lag2"]

    df = df[(df.date >= pd.to_datetime(start_date)) & (df.date <= pd.to_datetime(end_date))].copy()

    bull = pd.read_csv(bulletins_path, parse_dates=['valid_from', 'valid_to'])
    bull['start'] = bull['valid_from'].dt.normalize()
    bull['end']   = bull['valid_to'].dt.normalize()
    bull = (
        bull.assign(date=lambda d: d.apply(
            lambda row: pd.date_range(row['start'], row['end']), axis=1)
        )
        .explode('date')
        [['date', 'sector_id', 'level_numeric']]
    )

    df = df.merge(bull, on=['date', 'sector_id'], how='inner')

    # Class-balanced sampling
    pieces = []
    for date in df.date.unique():
        sub = df[df.date == date]
        for lvl in sub.level_numeric.unique():
            grp = sub[sub.level_numeric == lvl]
            pieces.append(grp.sample(min(sample_per_lvl, len(grp)), random_state=random_state))
    df = pd.concat(pieces).reset_index(drop=True)

    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num.drop(columns=['sector_id', 'level_numeric'], errors='ignore', inplace=True)
    X = df_num.values
    y = df['level_numeric'].astype(int).values
    feats = df_num.columns.tolist()
    return df, X, y, feats

# ---------- Noise handling ---------- #
def reassign_noise(labels, X):
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

# ---------- Pipeline ---------- #
def build_pipeline(X, y, use_pca=True, pca_components=0.95):
    X_imp = SimpleImputer(strategy='mean').fit_transform(X)
    X_scl = StandardScaler().fit_transform(X_imp)
    X_poly = PolynomialFeatures(2, include_bias=False).fit_transform(X_scl)
    selector = SelectFromModel(
        RandomForestClassifier(100, random_state=RANDOM_STATE),
        threshold='median'
    ).fit(X_poly, y)
    X_sel = selector.transform(X_poly)
    if use_pca:
        X_sel = PCA(n_components=pca_components, random_state=RANDOM_STATE).fit_transform(X_sel)
    n_cls = len(np.unique(y))
    X_lda = LDA(n_components=min(n_cls - 1, X_sel.shape[1])).fit_transform(X_sel, y)
    return X_lda

# ---------- HDBSCAN Ensemble ---------- #
def run_ensemble(X, y, mcs, ms, eps, method, metric, k=ENSEMBLE_K):
    all_lbls = []
    for seed in range(k):
        np.random.seed(seed)
        c = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=eps,
            cluster_selection_method=method,
            metric=metric,
            prediction_data=True
        ).fit(X)
        try:
            lbls, _ = hdbscan.approximate_predict(c, X)
        except AttributeError:
            lbls = c.labels_
        all_lbls.append(lbls)
    n = X.shape[0]
    M = np.zeros((n, n))
    for lbls in all_lbls:
        for i in range(n):
            for j in range(i+1, n):
                if lbls[i] == lbls[j]:
                    M[i, j] += 1
    M += M.T + np.eye(n) * k
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=len(np.unique(y)), metric='precomputed', linkage='average')
    final = agg.fit_predict(k - M)
    return adjusted_rand_score(y, final)

# ---------- Optuna Objective ---------- #
def objective(trial, X, y):
    mcs    = trial.suggest_int('min_cluster_size', 5, 150, step=5)
    ms     = trial.suggest_int('min_samples', 1, mcs)
    eps    = trial.suggest_float('cluster_selection_epsilon', 1e-3, 1e-1, log=True)
    method = trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf'])
    metric = trial.suggest_categorical('metric', ['euclidean', 'correlation'])
    return run_ensemble(X, y, mcs, ms, eps, method, metric)

# ---------- Main ---------- #
def main():
    df, X_raw, y, feats = load_data(
        INPUT_CSV, BULLETIN_CSV, START_DATE, END_DATE,
        SAMPLE_PER_LVL, random_state=RANDOM_STATE, spatial=USE_SPATIAL
    )
    X_lda = build_pipeline(X_raw, y)

    import optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda tr: objective(tr, X_lda, y), n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    print("Best ARI:", study.best_value)
    print("Best params:", study.best_params)

if __name__ == '__main__':
    main()

