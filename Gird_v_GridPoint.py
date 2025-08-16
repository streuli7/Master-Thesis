"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data 
Visualization and Decision Support

Author: Nils Besson

Description:
Evaluates avalanche clustering performance using grid-only versus grid+point
features. The script preprocesses data, applies dimensionality reduction,
runs HDBSCAN clustering, and compares results to SLF bulletin labels with ARI
(overall and per-day). Outputs include CSV summaries for both settings.

Note: Parts of this script (e.g., documentation and header text) were drafted
with the assistance of AI (ChatGPT). All code has been reviewed and adapted by the author.
"""

# ---------- Load libraries ------------- #
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
import hdbscan

# ---------- Configuration ---------- #
BASE_DIR        = Path(r"Path/to/data/directory")
DATA_CSV        = BASE_DIR / "aggregated_grid_time_series_lv03.csv" #Output of Grid_Summary.py
BULLETIN_CSV    = BASE_DIR / "bulletin_export_2023-2024.csv" #Output of Bulletin_Export.py
START_DATE      = "2024-02-01"
END_DATE        = "2024-04-30"
RANDOM_STATE    = 42
USE_SPATIAL     = True
SAMPLE_PER_LVL  = 100

# Define HDBSCAN parameters
HDBSCAN_PARAMS = dict(
    min_cluster_size=30,
    min_samples=5,
    cluster_selection_epsilon=0.01,
    cluster_selection_method='eom',
    metric='euclidean'
)

OUT_DIR = Path("./outputs_grid_vs_gp_samefile")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Disable warnings ---------- #
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
np.seterr(all="ignore")

# ---------- Define helper functions ---------- #
GRID_FEATURES = {
    "dlModelFx_Prob3", "elevation", "instabModelFx",
    "spontLawModelFx", "level_continuous", "sublevel"
}
KEY_KEEP = {"grid_id", "date", "sector_id", "x", "y"}  # meta columns to preserve if present

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

def load_data_both_roles():
    df = pd.read_csv(DATA_CSV, parse_dates=['date'])
    if not USE_SPATIAL:
        df = df.drop(columns=['x','y'], errors='ignore')

    # Filter period
    df = df[(df['date'] >= pd.to_datetime(START_DATE)) & (df['date'] <= pd.to_datetime(END_DATE))].copy()

    # Attach bulletin labels
    bull = pd.read_csv(BULLETIN_CSV, parse_dates=['valid_from','valid_to'])
    bull['start'] = bull['valid_from'].dt.normalize()
    bull['end']   = bull['valid_to'].dt.normalize()
    bull = bull.assign(date=lambda d: d.apply(lambda r: pd.date_range(r['start'], r['end']), axis=1)).explode('date')
    bull = bull[['date','sector_id','level_numeric']]

    df = df.merge(bull, on=['date','sector_id'], how='inner')

    # Identify numeric features; split into grid vs point
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # meta/label columns we will drop from X
    drop_from_X = {'sector_id','level_numeric'}
    num_cols_X = [c for c in num_cols if c not in drop_from_X]

    grid_cols_in_file = [c for c in num_cols_X if c in GRID_FEATURES]
    point_cols_in_file = [c for c in num_cols_X if c not in GRID_FEATURES]

    # Build two views
    cols_common = list(KEY_KEEP & set(df.columns)) + ['level_numeric']
    df_grid = df[cols_common + grid_cols_in_file].copy()
    df_gp   = df[cols_common + grid_cols_in_file + point_cols_in_file].copy()

    return df_grid, df_gp

def class_balanced_sample(df, sample_per_lvl=SAMPLE_PER_LVL, seed=RANDOM_STATE):
    pieces = []
    for d, sub in df.groupby('date'):
        for lvl, grp in sub.groupby('level_numeric'):
            pieces.append(grp.sample(min(sample_per_lvl, len(grp)), random_state=seed))
    return pd.concat(pieces).reset_index(drop=True)

def to_matrix(df):
    num = df.select_dtypes(include=[np.number]).copy()
    num.drop(columns=['sector_id','level_numeric'], errors='ignore', inplace=True)
    X = num.values
    y = df['level_numeric'].astype(int).values
    return X, y

def pipeline_to_lda(X, y):
    X = SimpleImputer(strategy='mean').fit_transform(X)
    X = StandardScaler().fit_transform(X)
    X = PolynomialFeatures(2, include_bias=False).fit_transform(X)
    selector = SelectFromModel(RandomForestClassifier(100, random_state=RANDOM_STATE), threshold='median')
    X = selector.fit_transform(X, y)
    X = PCA(n_components=0.95, random_state=RANDOM_STATE).fit_transform(X)
    n_cls = len(np.unique(y))
    X = LDA(n_components=min(n_cls-1, X.shape[1])).fit_transform(X, y)
    return X

def overall_ari(X, y):
    c = hdbscan.HDBSCAN(**HDBSCAN_PARAMS, prediction_data=True).fit(X)
    try:
        labels, _ = hdbscan.approximate_predict(c, X)
    except AttributeError:
        labels = c.labels_
    labels = prune_small(reassign_noise(labels.copy(), X), X, HDBSCAN_PARAMS['min_cluster_size'])
    return adjusted_rand_score(y, labels)

def per_day_ari(df, X_lda, y):
    rows = []
    for day, g in df.groupby('date'):
        idx = g.index.values
        Xd, yd = X_lda[idx], y[idx]
        c = hdbscan.HDBSCAN(**HDBSCAN_PARAMS, prediction_data=True).fit(Xd)
        try:
            lbls, _ = hdbscan.approximate_predict(c, Xd)
        except AttributeError:
            lbls = c.labels_
        lbls = prune_small(reassign_noise(lbls.copy(), Xd), Xd, HDBSCAN_PARAMS['min_cluster_size'])
        rows.append((pd.to_datetime(day).date(), adjusted_rand_score(yd, lbls)))
    return pd.DataFrame(rows, columns=['date','ARI'])

def run_block(df):
    df_s = class_balanced_sample(df)
    X, y = to_matrix(df_s)
    X_lda = pipeline_to_lda(X, y)
    overall = overall_ari(X_lda, y)
    per_day = per_day_ari(df_s, X_lda, y)
    return overall, per_day

def main():
    df_grid, df_gp = load_data_both_roles()

    # GRID ONLY
    overall_g, per_g = run_block(df_grid)
    per_g.to_csv(OUT_DIR / "per_day_GRID.csv", index=False)

    # GRID + POINT
    overall_gp, per_gp = run_block(df_gp)
    per_gp.to_csv(OUT_DIR / "per_day_GRID_PLUS_POINT.csv", index=False)

    # Summary
    summary = pd.DataFrame({
        'Setting': ['Grid only','Grid + Point'],
        'Overall ARI': [overall_g, overall_gp],
        'Mean per-day ARI': [per_g['ARI'].mean(), per_gp['ARI'].mean()]
    })
    summary.to_csv(OUT_DIR / "summary_grid_vs_gp_samefile.csv", index=False)
    print(summary)

if __name__ == "__main__":
    main()
