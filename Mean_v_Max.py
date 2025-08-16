"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data 
Visualization and Decision Support

Author: Nils Besson

Description:
Aggregates multiple avalanche forecasting datasets onto a 5x5 km grid
(using mean or max) and merges them with SLF bulletin labels. The script
preprocesses features, reduces dimensionality, applies HDBSCAN clustering,
and evaluates clustering quality via ARI (overall and per-day). Outputs
include aggregated grid files and CSV summaries for mean vs. max comparison.

Note: Parts of this script (e.g., documentation and header text) were drafted
with the assistance of AI (ChatGPT). All code has been reviewed and adapted by the author.
"""

#------------- Load libraries ------------- #
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
import hdbscan

# ---------------- Configuration ---------------- #
GRID_SIZE = 5000  # meters
DATE_COLUMN = "date"

# LV03 grid bounds
X_MIN = 480000
X_MAX = 850000
Y_MIN = 70000
Y_MAX = 250000

# Full attribute list for each dataset
station_cols = [
    "hn24", "hn72", "aspect", "elevation", "sector_id",
    "probability_level_1", "probability_level_2", "probability_level_3", "probability_level_4",
    "level_expected", "P_s2", "P_s3", "P_s4", "P_s5", "size_expected", "hs",
    "pMax_decisive", "depth_decisive", "np_pMax", "np_depth",
    "pwl_pMax", "pwl_depth", "probability_natAval"
]
dl_cols     = ["dlModelFx_Prob3", "elevation"]
instab_cols = ["instabModelFx", "elevation"]
spont_cols  = ["spontLawModelFx", "elevation"]

#----------------- Paths ---------------- #
BASE_PATH = r"Path/to/data/directory"

STATION_FILE    = os.path.join(BASE_PATH, "stationData_models_2024-02-04_to_2024-04-30.csv") #Point Data file
DANGERLEVEL_FILE= os.path.join(BASE_PATH, "dangerlevelModelForecastInterpolations.csv") #Danger Level file
INSTAB_FILE     = os.path.join(BASE_PATH, "instabModelForecastInterpolations.csv") #Instability file
SPONTLAW_FILE   = os.path.join(BASE_PATH, "spontLawModelForecastInterpolations.csv") #Spontaneous Law file
SUBLEVELS_FILE  = os.path.join(BASE_PATH, "predictions_dangerLevelModel_subLevels.csv") #Sublevels file

# Bulletin (for ARI comparison)
BULLETIN_CSV    = os.path.join(BASE_PATH, "bulletin_export_2023-2024.csv") #Output of Bulletin_Export.py
START_DATE      = "2024-02-01"
END_DATE        = "2024-04-30"

# Fixed, quick HDBSCAN params for apples‑to‑apples comparison
HDBSCAN_PARAMS = dict(
    min_cluster_size=30,
    min_samples=5,
    cluster_selection_epsilon=0.01,
    cluster_selection_method='eom',
    metric='euclidean'
)

# ---------------- Load Data ---------------- #
station_df   = pd.read_csv(STATION_FILE,    parse_dates=[DATE_COLUMN])
dangerlevel_df = pd.read_csv(DANGERLEVEL_FILE, parse_dates=[DATE_COLUMN])
instab_df    = pd.read_csv(INSTAB_FILE,     parse_dates=[DATE_COLUMN])
spontlaw_df  = pd.read_csv(SPONTLAW_FILE,   parse_dates=[DATE_COLUMN])
sublevels_df = pd.read_csv(SUBLEVELS_FILE,  parse_dates=[DATE_COLUMN])

for df in [station_df, dangerlevel_df, instab_df, spontlaw_df, sublevels_df]:
    df.columns = df.columns.str.strip()
    df.dropna(subset=["x", "y", "date"], inplace=True)
for df in [dangerlevel_df, instab_df, spontlaw_df]:
    df.rename(columns={"ele": "elevation"}, inplace=True)

# ---------------- Create grid ----------------#
x_coords = np.arange(X_MIN, X_MAX, GRID_SIZE)
y_coords = np.arange(Y_MIN, Y_MAX, GRID_SIZE)
grid_points = np.array([(x, y) for x in x_coords for y in y_coords])
grid_ids = [f"grid_{i}" for i in range(len(grid_points))]
tree = cKDTree(grid_points)
grid_metadata = pd.DataFrame({"grid_id": grid_ids, "x": grid_points[:, 0], "y": grid_points[:, 1]})

def assign_to_grid(df, x_col="x", y_col="y"):
    coords = df[[x_col, y_col]].values
    _, indices = tree.query(coords)
    return indices

# ----------------- Aggregate to grid (parametric) ----------------- #
def aggregate_to_grid(df, numeric_cols, cat_cols, source_name, agg_func="mean"):
    """
    agg_func: "mean" or "max"
    """
    df = df.copy()
    grid_indices = assign_to_grid(df)
    df["grid_id"] = [f"grid_{i}" for i in grid_indices]
    df["x"]       = [grid_points[i][0] for i in grid_indices]
    df["y"]       = [grid_points[i][1] for i in grid_indices]

    print(f"[{source_name}:{agg_func}] → {df['grid_id'].nunique()} grid cells")

    if agg_func not in ("mean", "max"):
        raise ValueError("agg_func must be 'mean' or 'max'")
    agg_dict = {c: agg_func for c in numeric_cols}
    agg_dict.update({c: "first" for c in cat_cols})
    agg_dict.update({"x": "first", "y": "first"})

    grouped = df.groupby(["grid_id", DATE_COLUMN]).agg(agg_dict).reset_index()
    return grouped

def build_merged(agg_choice="mean"):
    agg_station  = aggregate_to_grid(station_df,      station_cols, [],   "station",    agg_choice)
    agg_dl       = aggregate_to_grid(dangerlevel_df,  dl_cols,      [],   "dangerlevel",agg_choice)
    agg_instab   = aggregate_to_grid(instab_df,       instab_cols,  [],   "instab",     agg_choice)
    agg_spontlaw = aggregate_to_grid(spontlaw_df,     spont_cols,   [],   "spontlaw",   agg_choice)
    agg_sublv = aggregate_to_grid(sublevels_df, ["level_continuous"], ["sublevel"], "sublevel", agg_choice)

    # Drop duplicates before merging
    agg_station  = agg_station.drop( columns=["x","y","elevation"], errors="ignore")
    agg_instab   = agg_instab.drop(  columns=["x","y","elevation"], errors="ignore")
    agg_spontlaw = agg_spontlaw.drop(columns=["x","y","elevation"], errors="ignore")
    agg_dl       = agg_dl.drop(      columns=["x","y"],             errors="ignore")

    merged = (
        agg_station
        .merge(agg_dl,       on=["grid_id", DATE_COLUMN], how="outer")
        .merge(agg_instab,   on=["grid_id", DATE_COLUMN], how="outer")
        .merge(agg_spontlaw, on=["grid_id", DATE_COLUMN], how="outer")
        .merge(agg_sublv,    on=["grid_id", DATE_COLUMN], how="outer")
    )

    merged = merged.drop(columns=["x","y"], errors="ignore").merge(grid_metadata, on="grid_id", how="left")

    # Full date-grid coverage (optional but keeps shape consistent)
    date_min = min(df[DATE_COLUMN].min() for df in [station_df, dangerlevel_df, instab_df, spontlaw_df, sublevels_df])
    date_max = max(df[DATE_COLUMN].max() for df in [station_df, dangerlevel_df, instab_df, spontlaw_df, sublevels_df])
    all_grid_ids = grid_metadata["grid_id"].tolist()
    all_dates = pd.date_range(start=date_min, end=date_max, freq="D")
    full_index = pd.MultiIndex.from_product([all_grid_ids, all_dates], names=["grid_id", "date"])
    full_df = pd.DataFrame(index=full_index).reset_index()
    merged_full = full_df.merge(merged, on=["grid_id", "date"], how="left")
    return merged_full

# ---------------- ARI utilities ---------------- #
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

def build_Xy_for_ari(grid_csv_path, bulletin_csv=BULLETIN_CSV):
    # Load aggregated grid file
    df = pd.read_csv(grid_csv_path, parse_dates=[DATE_COLUMN])

    # Load bulletin and expand to daily rows
    bull = pd.read_csv(bulletin_csv, parse_dates=['valid_from', 'valid_to'])
    bull['start'] = bull['valid_from'].dt.normalize()
    bull['end']   = bull['valid_to'].dt.normalize()
    bull = (
        bull.assign(date=lambda d: d.apply(lambda row: pd.date_range(row['start'], row['end']), axis=1))
            .explode('date')[['date', 'sector_id', 'level_numeric']]
    )

    # Merge + filter time range
    df = df.merge(bull, on=['date','sector_id'], how='inner')
    df = df[(df[DATE_COLUMN] >= pd.to_datetime(START_DATE)) & (df[DATE_COLUMN] <= pd.to_datetime(END_DATE))].copy()

    # Build numeric matrix
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num.drop(columns=['sector_id','level_numeric'], errors='ignore', inplace=True)
    X = df_num.values
    y = df['level_numeric'].astype(int).values
    return df, X, y

def pipeline_to_lda(X, y):
    X_imp = SimpleImputer(strategy='mean').fit_transform(X)
    X_scl = StandardScaler().fit_transform(X_imp)
    X_poly = PolynomialFeatures(2, include_bias=False).fit_transform(X_scl)
    selector = SelectFromModel(RandomForestClassifier(100, random_state=42), threshold='median').fit(X_poly, y)
    X_sel = selector.transform(X_poly)
    X_pc = PCA(n_components=0.95, random_state=42).fit_transform(X_sel)
    n_cls = len(np.unique(y))
    X_lda = LDA(n_components=min(n_cls - 1, X_pc.shape[1])).fit_transform(X_pc, y)
    return X_lda

def overall_and_per_day_ari(df, X_lda, y):
    # Overall (single HDBSCAN on all rows)
    c = hdbscan.HDBSCAN(**HDBSCAN_PARAMS, prediction_data=True).fit(X_lda)
    try:
        labels, _ = hdbscan.approximate_predict(c, X_lda)
    except AttributeError:
        labels = c.labels_
    labels = prune_small(reassign_noise(labels.copy(), X_lda), X_lda, HDBSCAN_PARAMS['min_cluster_size'])
    overall = adjusted_rand_score(y, labels)

    # Per‑day (fit HDBSCAN per day for daily ARI)
    out = []
    for day, g in df.groupby('date'):
        rows = g.index.values
        Xd, yd = X_lda[rows], y[rows]
        c = hdbscan.HDBSCAN(**HDBSCAN_PARAMS, prediction_data=True).fit(Xd)
        try:
            lbls, _ = hdbscan.approximate_predict(c, Xd)
        except AttributeError:
            lbls = c.labels_
        lbls = prune_small(reassign_noise(lbls.copy(), Xd), Xd, HDBSCAN_PARAMS['min_cluster_size'])
        out.append((pd.to_datetime(day).date(), adjusted_rand_score(yd, lbls)))
    per_day = pd.DataFrame(out, columns=['date','ARI'])
    return overall, per_day

# ----------------- Build & Export + ARI ----------------- #
if __name__ == "__main__":
    # MEAN file
    mean_df = build_merged("mean")
    out_mean = os.path.join(BASE_PATH, "aggregated_grid_time_series_lv03.csv")
    mean_df.to_csv(out_mean, index=False)
    print(f"Saved (MEAN): {out_mean}")

    # MAX file
    max_df = build_merged("max")
    out_max = os.path.join(BASE_PATH, "aggregated_grid_time_series_lv03_MAX.csv")
    max_df.to_csv(out_max, index=False)
    print(f"Saved (MAX):  {out_max}")

    # ---- ARI for MEAN ----
    df_mean, X_mean, y_mean = build_Xy_for_ari(out_mean)
    X_mean_lda = pipeline_to_lda(X_mean, y_mean)
    overall_mean, per_day_mean = overall_and_per_day_ari(df_mean, X_mean_lda, y_mean)

    # ---- ARI for MAX ----
    df_max, X_max, y_max = build_Xy_for_ari(out_max)
    X_max_lda = pipeline_to_lda(X_max, y_max)
    overall_max, per_day_max = overall_and_per_day_ari(df_max, X_max_lda, y_max)

    # ---- Summary ----
    summary = pd.DataFrame({
        'Aggregation': ['MEAN','MAX'],
        'Overall ARI': [overall_mean, overall_max],
        'Mean per-day ARI': [per_day_mean['ARI'].mean(), per_day_max['ARI'].mean()]
    })

    out_summary = os.path.join(BASE_PATH, "ari_summary_mean_vs_max.csv")
    summary.to_csv(out_summary, index=False)

    print("\n=== ARI summary (MEAN vs MAX) ===")
    print(summary)
    print(f"\nSaved summary → {out_summary}")
