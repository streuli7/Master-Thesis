"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data 
Visualization and Decision Support

Author: Nils Besson

Description:
Generates daily avalanche hazard maps for Switzerland from SLF bulletin data.
It joins bulletin danger levels with regional shapefiles, applies a color scheme,
and exports map plots as PNG files for the period Feb-Apr 2024.

Note: Parts of this script (e.g., documentation and header text) were drafted
with the assistance of AI (ChatGPT). All code has been reviewed and adapted by the author.
"""
# ---------- Load libraries ---------- #
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ----------- Configuration ----------- #
shapefile_path = r"Shapefile/of/Switzerland"  # adjust path to Shapefile fo Switzerland
bulletin_csv = 'path/to/bulletin/csv' # adjust path to Bulletin file
output_dir = 'Output/Directory' # adjust path to Output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ------------ Create map plots ------------ #
# Load regions
regions = gpd.read_file(shapefile_path)
print("Shapefile columns:", regions.columns.tolist())

# Load & filter bulletins
b = pd.read_csv(bulletin_csv, parse_dates=['valid_from'])
b = (
    b[b['drySnow'] == 1]
     .drop_duplicates(subset=['valid_from','sector_id'], keep='first')
)
# Date‐range Feb 1–Apr 30, 2024
mask = (b['valid_from'] >= '2024-02-01') & (b['valid_from'] <= '2024-04-30')
b = b.loc[mask]

# Convert numeric detail to labels
def to_label(x):
    if pd.isna(x):
        return 'No Data'
    base = int(np.floor(x))
    frac = x - base
    if frac < 0.33:
        return f"{base}-"
    elif frac < 0.66:
        return f"{base}"
    else:
        return f"{base}+"

b['detail_label'] = b['level_detail_numeric'].apply(to_label)

# Colour map
color_discrete_map = {
    "1-": "#4575b4","1": "#74add1","1+": "#abd9e9",
    "2-": "#ffffbf","2": "#fdae61","2+": "#f46d43",
    "3-": "#f03b20","3": "#e31a1c","3+": "#bd0026",
    "4-": "#800026","4": "#67001f","4+": "#49006a",
    "No Data": "lightgrey"
}

# Ensure join‐keys align
regions['GEB_ID']      = regions['GEB_ID'].astype(int)
b['sector_id']         = b['sector_id'].astype(int)

# Loop dates and plot
for date, df_day in b.groupby(b['valid_from'].dt.date):
    merged = regions.merge(
        df_day[['sector_id','detail_label']],
        left_on='GEB_ID', right_on='sector_id', how='left'
    )
    merged['detail_label'] = merged['detail_label'].fillna('No Data')
    merged['color']        = merged['detail_label'].map(color_discrete_map)

    fig, ax = plt.subplots(figsize=(10,8))
    merged.plot(
        color=merged['color'],
        edgecolor='gray', linewidth=0.5,
        ax=ax
    )
    ax.set_title(f"Avalanche Detail Level (dry snow) – {date}", fontsize=16)
    ax.axis('off')

    # Manual legend
    handles = [
        mpatches.Patch(color=color_discrete_map[k], label=k)
        for k in color_discrete_map
    ]
    ax.legend(handles=handles, title="Detail Level", loc='lower left')

    fig.savefig(
        os.path.join(output_dir, f"detail_map_{date}.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig)

# ------------ Print output ------------ #
print("Plots are in", output_dir)

