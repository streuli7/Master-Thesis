"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data 
Visualization and Decision Support

Author: Nils Besson

Description: 
A data processing and clustering pipeline for avalanche forecasting data.
It prepares time-series features, applies dimensionality reduction, and
runs HDBSCAN to compare clusters against SLF bulletin danger levels.
The script outputs evaluation metrics, plots, and summaries for analysis.

Note: Parts of this script (e.g., documentation and header text) were drafted
with the assistance of AI (ChatGPT). All code has been reviewed and adapted by the author.
"""
# ------------ Import libraries ------------ #
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
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

# ------------------ Configuration ------------------ #
BASE_DIR        = Path(r"Path/to/data/directory")
INPUT_CSV       = BASE_DIR / "aggregated_grid_time_series_lv03_MAX.csv" # Output of Mean_v_Max.py
BULLETIN_CSV    = BASE_DIR / "bulletin_export_2023-2024.csv" # Output of Bulletin_Export.py
START_DATE      = "2024-02-01"
END_DATE        = "2024-04-30"
RANDOM_STATE    = 42
USE_SPATIAL     = True          # include x,y as features
SAMPLE_PER_LVL  = 100           # class-balanced sampling

# HDBSCAN params for quick, reproducible runs
HDBSCAN_PARAMS = dict(
    min_cluster_size=30,
    min_samples=5,
    cluster_selection_epsilon=0.01,
    cluster_selection_method='eom',
    metric='euclidean'
)

OUT_DIR         = Path("./outputs_fast_report")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Disable warnings ------------------ #
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
np.seterr(all="ignore")
rng = np.random.default_rng(RANDOM_STATE)

# ------------------ Define helper functions ------------------ #
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

    # Class-balanced sampling (per date, per level)
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

def run_hdbscan(X, params, reassign=True):
    c = hdbscan.HDBSCAN(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        cluster_selection_epsilon=params['cluster_selection_epsilon'],
        cluster_selection_method=params['cluster_selection_method'],
        metric=params['metric'],
        prediction_data=True
    ).fit(X)
    try:
        labels, _ = hdbscan.approximate_predict(c, X)
    except AttributeError:
        labels = c.labels_
    if reassign:
        labels = reassign_noise(labels.copy(), X)
        labels = prune_small(labels, X, thresh=params['min_cluster_size'])
    return labels

# ------------------ MAIN ------------------ #
def main():
    # Load
    df, X_raw, y, feats = load_data(
        INPUT_CSV, BULLETIN_CSV, START_DATE, END_DATE,
        SAMPLE_PER_LVL, random_state=RANDOM_STATE, spatial=USE_SPATIAL
    )

    stage_names = []
    feature_counts = []
    representations = {}

    # Raw
    stage_names.append("Raw (numeric)")
    feature_counts.append(X_raw.shape[1])
    representations['raw'] = X_raw

    # Impute + Scale
    imp = SimpleImputer(strategy='mean').fit(X_raw)
    X1 = imp.transform(X_raw)
    scl = StandardScaler().fit(X1)
    X_s = scl.transform(X1)
    stage_names.append("Scaled")
    feature_counts.append(X_s.shape[1])
    representations['scaled'] = X_s

    # Polynomial (degree=2)
    poly = PolynomialFeatures(2, include_bias=False).fit(X_s)
    X_p = poly.transform(X_s)
    stage_names.append("Poly d=2")
    feature_counts.append(X_p.shape[1])
    representations['poly'] = X_p

    # Feature selection (RF median)
    sel = SelectFromModel(
        RandomForestClassifier(100, random_state=RANDOM_STATE),
        threshold='median'
    ).fit(X_p, y)
    X_f = sel.transform(X_p)
    stage_names.append("FeatureSelect")
    feature_counts.append(X_f.shape[1])
    representations['feat'] = X_f

    # PCA (keep 95% variance or fewer comps if needed)
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_f)
    X_pc = pca.transform(X_f)
    stage_names.append(f"PCA (95% var)")
    feature_counts.append(X_pc.shape[1])
    representations['pca'] = X_pc

    # PCA variance curve
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    plt.plot(np.arange(1, len(cumvar)+1), cumvar, marker='o')
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    pca_plot_path = OUT_DIR / "pca_cumulative_variance.png"
    plt.savefig(pca_plot_path, dpi=150)
    plt.close()

    # Stage 5: LDA
    n_cls = len(np.unique(y))
    lda = LDA(n_components=min(n_cls - 1, X_pc.shape[1])).fit(X_pc, y)
    X_l = lda.transform(X_pc)
    stage_names.append("LDA")
    feature_counts.append(X_l.shape[1])
    representations['lda'] = X_l

    # Feature counts plot
    plt.figure()
    plt.bar(range(len(feature_counts)), feature_counts)
    plt.xticks(range(len(feature_counts)), stage_names, rotation=30, ha='right')
    plt.ylabel("Feature count")
    plt.title("Feature counts per pipeline stage")
    plt.tight_layout()
    fc_plot_path = OUT_DIR / "feature_counts_per_stage.png"
    plt.savefig(fc_plot_path, dpi=150)
    plt.close()

    # ARI per stage (use same HDBSCAN params for fair comparison)
    ari_rows = []
    stage_order = ['scaled', 'poly', 'feat', 'pca', 'lda']
    stage_labels = {
        'scaled': 'Scaled',
        'poly': 'Poly d=2',
        'feat': 'FeatureSelect',
        'pca': 'PCA (95% var)',
        'lda': 'LDA'
    }
    for key in stage_order:
        Xr = representations[key]
        lbls = run_hdbscan(Xr, HDBSCAN_PARAMS)
        ari = adjusted_rand_score(y, lbls)
        ari_rows.append((stage_labels[key], ari))
    ari_df = pd.DataFrame(ari_rows, columns=['Stage', 'ARI']).sort_values('Stage')
    ari_df.to_csv(OUT_DIR / "ari_per_stage.csv", index=False)

    # ARI per stage plot
    plt.figure()
    plt.bar(ari_df['Stage'], ari_df['ARI'])
    plt.ylabel("ARI")
    plt.title("ARI per stage (fixed HDBSCAN params)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    ari_plot_path = OUT_DIR / "ari_per_stage.png"
    plt.savefig(ari_plot_path, dpi=150)
    plt.close()

    # Cluster counts vs bulletin "regions" per day
    # Approximation: number of distinct danger levels present that day (unique level_numeric) vs number of HDBSCAN clusters on that day (fit on that day only, in the LDA space for best separability).
    per_date = []
    for day, g in df.groupby('date'):
        y_d = g['level_numeric'].astype(int).values

        # Use the already-fitted pipeline parts to transform to LDA space:
        Xd_raw = g.select_dtypes(include=[np.number]).drop(columns=['sector_id','level_numeric'], errors='ignore').values
        Xd = scl.transform(imp.transform(Xd_raw))
        Xd = poly.transform(Xd)
        Xd = sel.transform(Xd)
        Xd = pca.transform(Xd)
        Xd = lda.transform(Xd)

        lbls_d = run_hdbscan(Xd, HDBSCAN_PARAMS)
        n_clusters = len(np.unique(lbls_d))

        n_levels = len(np.unique(y_d))
        ari_d = adjusted_rand_score(y_d, lbls_d)

        per_date.append((pd.to_datetime(day).date(), n_clusters, n_levels, ari_d))

    date_df = pd.DataFrame(per_date, columns=['date', 'clusters_found', 'bulletin_levels', 'ARI'])
    date_df.to_csv(OUT_DIR / "clusters_vs_bulletin_by_date.csv", index=False)

    # Scatter: clusters vs bulletin levels
    plt.figure()
    plt.scatter(date_df['bulletin_levels'], date_df['clusters_found'])
    max_ax = max(date_df['bulletin_levels'].max(), date_df['clusters_found'].max())
    plt.plot([0, max_ax], [0, max_ax])
    plt.xlabel("Bulletin distinct levels (per day)")
    plt.ylabel("HDBSCAN clusters (per day, LDA space)")
    plt.title("Clusters vs. bulletin-level count by day")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    scat_path = OUT_DIR / "clusters_vs_bulletin_scatter.png"
    plt.savefig(scat_path, dpi=150)
    plt.close()

    # Examples: best/worst days by ARI
    top_k = 3 if len(date_df) >= 3 else len(date_df)
    best_days = date_df.sort_values('ARI', ascending=False).head(top_k)
    worst_days = date_df.sort_values('ARI', ascending=True).head(top_k)

    best_days.to_csv(OUT_DIR / "examples_best_days.csv", index=False)
    worst_days.to_csv(OUT_DIR / "examples_worst_days.csv", index=False)

    # Save a compact summary text
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Pipeline Dimensionality Reduction Summary ===\n")
        for name, cnt in zip(stage_names, feature_counts):
            f.write(f"{name}: {cnt} features\n")
        f.write("\n=== PCA Variance ===\n")
        f.write(f"PCA kept components: {X_pc.shape[1]}\n")
        f.write(f"Explained variance (cumulative): {cumvar[-1]:.3f}\n")
        f.write("\n=== ARI per Stage (fixed HDBSCAN) ===\n")
        for _, row in ari_df.iterrows():
            f.write(f"{row['Stage']}: ARI={row['ARI']:.3f}\n")
        f.write("\n=== Per-day comparison (see CSV) ===\n")
        f.write("clusters_vs_bulletin_by_date.csv contains per-day clusters vs. bulletin level counts and ARI.\n")
        f.write("examples_best_days.csv / examples_worst_days.csv show concrete dates with high/low agreement.\n")

    print("Done. Outputs written to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
