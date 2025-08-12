"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data 
Visualization and Decision Support

Author: Nils Besson

Description:
This script aggregates point-based station and model forecast data onto a uniform 5 km LV03 spatial grid. 
It processes multiple datasets (e.g., station data, danger level, instability, spontaneous avalanche forecasts, and SLF sublevels),
assigns each data point to a corresponding grid cell, and computes daily grid-level averages. The output is a 
complete grid x time matrix used for further clustering and visualization in the dashboard.
"""
#------------- Load libraries ------------- #
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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

dl_cols = ["dlModelFx_Prob3", "elevation"]
instab_cols = ["instabModelFx", "elevation"]
spont_cols = ["spontLawModelFx", "elevation"]

#----------------- Load Data ---------------- #
BASE_PATH = r"C://Users//nilsb//OneDrive - Universität Zürich UZH//Universität//5 Jahr//GEO 511 Masterarbeit//03_Data"

STATION_FILE = os.path.join(BASE_PATH, "stationData_models_2024-02-04_to_2024-04-30.csv")
DANGERLEVEL_FILE = os.path.join(BASE_PATH, "dangerlevelModelForecastInterpolations.csv")
INSTAB_FILE = os.path.join(BASE_PATH, "instabModelForecastInterpolations.csv")
SPONTLAW_FILE = os.path.join(BASE_PATH, "spontLawModelForecastInterpolations.csv")
SUBLEVELS_FILE = os.path.join(BASE_PATH,"predictions_dangerLevelModel_subLevels.csv")

station_df = pd.read_csv(STATION_FILE, parse_dates=[DATE_COLUMN])
dangerlevel_df = pd.read_csv(DANGERLEVEL_FILE, parse_dates=[DATE_COLUMN])
instab_df = pd.read_csv(INSTAB_FILE, parse_dates=[DATE_COLUMN])
spontlaw_df = pd.read_csv(SPONTLAW_FILE, parse_dates=[DATE_COLUMN])
sublevels_df = pd.read_csv(SUBLEVELS_FILE, parse_dates=[DATE_COLUMN])

# Clean up column names
for df in [station_df, dangerlevel_df, instab_df, spontlaw_df, sublevels_df]:
    df.columns = df.columns.str.strip()

# Drop rows with missing x/y/date
for df in [station_df, dangerlevel_df, instab_df, spontlaw_df, sublevels_df]:
    df.dropna(subset=["x", "y", "date"], inplace=True)

# Rename ele → elevation so it works consistently
for df in [dangerlevel_df, instab_df, spontlaw_df]:
    df.rename(columns={"ele": "elevation"}, inplace=True)

# ---------------- Create grid points ----------------#
x_coords = np.arange(X_MIN, X_MAX, GRID_SIZE)
y_coords = np.arange(Y_MIN, Y_MAX, GRID_SIZE)
grid_points = np.array([(x, y) for x in x_coords for y in y_coords])
grid_ids = [f"grid_{i}" for i in range(len(grid_points))]
tree = cKDTree(grid_points)

# Create grid metadata DataFrame (for merging in coordinates later)
grid_metadata = pd.DataFrame({
    "grid_id": grid_ids,
    "x": grid_points[:, 0],
    "y": grid_points[:, 1]
})

# ----------------- Assign points to grid cells ----------------- #
def assign_to_grid(df, x_col="x", y_col="y"):
    coords = df[[x_col, y_col]].values
    _, indices = tree.query(coords)
    return indices

# ----------------- Aggregate to grid ----------------- #
def aggregate_to_grid(df, numeric_cols, cat_cols, source_name):
    df = df.copy()
    # Assign each point to a grid cell
    grid_indices = assign_to_grid(df)
    df["grid_id"] = [f"grid_{i}" for i in grid_indices]
    df["x"]       = [grid_points[i][0] for i in grid_indices]
    df["y"]       = [grid_points[i][1] for i in grid_indices]

    print(f"[{source_name}] Assigned to {df['grid_id'].nunique()} grid cells")

    # Build aggregation dict
    agg_dict = {c: "mean"  for c in numeric_cols}
    agg_dict.update({c: "first" for c in cat_cols})  # or 'mode' if you prefer
    agg_dict.update({"x": "first", "y": "first"})

    # Group and aggregate
    grouped = (
        df
        .groupby(["grid_id", DATE_COLUMN])
        .agg(agg_dict)
        .reset_index()
    )
    return grouped

# ----------------- Aggregate all datasets ----------------- #
# Aggregate the four original sources
agg_station  = aggregate_to_grid(station_df,      station_cols, [],   "station")
agg_dl       = aggregate_to_grid(dangerlevel_df,  dl_cols,      [],      "dangerlevel")
agg_instab   = aggregate_to_grid(instab_df,       instab_cols,  [],   "instab")
agg_spontlaw = aggregate_to_grid(spontlaw_df,     spont_cols,   [],    "spontlaw")

# Aggreate sublevels_df separately, as it has a different structure
agg_sublv = aggregate_to_grid(
    sublevels_df,
    numeric_cols=["level_continuous"],
    cat_cols=["sublevel"],
    source_name="sublevel"
)

# Drop the extra x/y from all but one, to avoid duplicate columns
agg_station  = agg_station.drop(columns=["x","y","elevation"], errors="ignore")
agg_instab   = agg_instab.drop( columns=["x","y","elevation"], errors="ignore")
agg_spontlaw = agg_spontlaw.drop(columns=["x","y","elevation"], errors="ignore")
agg_dl       = agg_dl.drop(      columns=["x","y"],               errors="ignore")

# Merge them all by grid_id & date
merged_df = (
    agg_station
    .merge(agg_dl,       on=["grid_id", DATE_COLUMN], how="outer")
    .merge(agg_instab,   on=["grid_id", DATE_COLUMN], how="outer")
    .merge(agg_spontlaw, on=["grid_id", DATE_COLUMN], how="outer")
    .merge(agg_sublv,    on=["grid_id", DATE_COLUMN], how="outer")
)

# remove any leftover x/y from the aggregates
merged_df = merged_df.drop(columns=["x","y"], errors="ignore")

# Reattach the grid x/y coords
merged_df = merged_df.merge(grid_metadata, on="grid_id", how="left")

# ----------------- Fill missing dates ----------------- #
all_grid_ids = [f"grid_{i}" for i in range(len(grid_points))]
all_dates = pd.date_range(start=station_df[DATE_COLUMN].min(), end=station_df[DATE_COLUMN].max(), freq="D")
full_index = pd.MultiIndex.from_product([all_grid_ids, all_dates], names=["grid_id", "date"])
full_df = pd.DataFrame(index=full_index).reset_index()

# Combine with real data
merged_full = full_df.merge(merged_df, on=["grid_id", "date"], how="right")

# ---------------- Export dataset ----------------- #
OUTPUT_PATH = os.path.join(BASE_PATH, "aggregated_grid_time_series_lv03.csv")
merged_full.to_csv(OUTPUT_PATH, index=False)
print(f" Saved: {OUTPUT_PATH}")
