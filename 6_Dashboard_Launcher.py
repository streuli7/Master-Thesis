"""
GEO 511 Master Thesis: Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data 
Visualization and Decision Support

Author: Nils Besson

Description: 
An interactive web dashboard built with Dash for visualizing snow, weather, and avalanche model data. 
It provides experts with tools to analyze station, grid, and cluster data through maps, histograms, and radar plots.
The dashboard supports decision-making by reducing cognitive load and enabling clear, user-centered data exploration.

Note: Parts of this script (e.g., documentation and header text) were drafted
with the assistance of AI (ChatGPT). All code has been reviewed and adapted by the author.
"""

#--------------- load libraries ---------------#

import dash
import dash_bootstrap_components as dbc
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, ctx
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import unary_union
from dash.dependencies import State
from plotly.colors import sample_colorscale

# ------------------ Load data ------------------ #

BASE_PATH = r"Path/to/data/directory"
SHAPEFILE_PATH = r"Path/to/Shapefile/of/Switzerland"

STATION_FILE = os.path.join(BASE_PATH, "stationData_models_2024-02-04_to_2024-04-30.csv") # Point Data file
DANGERLEVEL_FILE = os.path.join(BASE_PATH, "dangerlevelModelForecastInterpolations.csv") # Danger Level file
INSTAB_FILE = os.path.join(BASE_PATH, "instabModelForecastInterpolations.csv") # Instability file
SPONTLAW_FILE = os.path.join(BASE_PATH, "spontLawModelForecastInterpolations.csv") # Spontaneous Avalanche file
SUBLEVELS_FILE = os.path.join(BASE_PATH,"predictions_dangerLevelModel_subLevels.csv") # Sublevels file

CLUSTER_FILE = os.path.join(BASE_PATH, "merged_cluster_data_all_dates.csv") #Output Clustering_Pipeline.py
cluster_df = pd.read_csv(CLUSTER_FILE, parse_dates=["date"], low_memory=False)
cluster_dates = cluster_df["date"].dt.normalize().sort_values().unique()

# ---------------- Define some functions used in the dashbaord ------------------ #

 # load the complete grid × coord list from your aggregated output
GRID_META = (
    pd.read_csv(
        os.path.join(BASE_PATH, "aggregated_grid_time_series_lv03.csv"),
        usecols=["grid_id", "x", "y"]
    )
    # each grid_id only needs one row
    .drop_duplicates(subset="grid_id")
 )

station_df = pd.read_csv(STATION_FILE)
dangerlevel_df = pd.read_csv(DANGERLEVEL_FILE)
instab_df = pd.read_csv(INSTAB_FILE)
spontlaw_df = pd.read_csv(SPONTLAW_FILE)
sublevels_df = pd.read_csv(SUBLEVELS_FILE, parse_dates=["date"])

for df in [station_df, dangerlevel_df, instab_df, spontlaw_df, sublevels_df]:
    df['date'] = pd.to_datetime(df['date'])


gdf = gpd.read_file(SHAPEFILE_PATH)

# Ensure CRS is EPSG:2056 (Swiss LV95)
if gdf.crs != "EPSG:2056":
    gdf = gdf.to_crs(epsg=2056)

# Simplify geometry to reduce the number of points
gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)

# Merge all geometries into a single object (avoids multiple legend entries)
merged_geometry = unary_union(gdf.geometry)

# Define LV03 → LV95
transformer = Transformer.from_crs("EPSG:21781", "EPSG:2056", always_xy=True)

# Create LV95 coords for every df that has x,y in LV03 
station_df["x_LV95"], station_df["y_LV95"] = transformer.transform(station_df["x"].to_numpy(),
                                                                   station_df["y"].to_numpy())

dangerlevel_df["x_LV95"], dangerlevel_df["y_LV95"] = transformer.transform(dangerlevel_df["x"].to_numpy(),
                                                                            dangerlevel_df["y"].to_numpy())
instab_df["x_LV95"], instab_df["y_LV95"] = transformer.transform(instab_df["x"].to_numpy(),
                                                                  instab_df["y"].to_numpy())
spontlaw_df["x_LV95"], spontlaw_df["y_LV95"] = transformer.transform(spontlaw_df["x"].to_numpy(),
                                                                      spontlaw_df["y"].to_numpy())
sublevels_df["x_LV95"], sublevels_df["y_LV95"] = transformer.transform(sublevels_df["x"].to_numpy(),
                                                                        sublevels_df["y"].to_numpy())

# Keep LV03 copies for KD-Tree, but make plotted x,y be LV95 everywhere
for df in [station_df, dangerlevel_df, instab_df, spontlaw_df, sublevels_df]:
    df["x_lv03"] = df["x"]
    df["y_lv03"] = df["y"]
    df["x"] = df["x_LV95"]
    df["y"] = df["y_LV95"]

# GRID_META is read from *_lv03.csv, so keep a lv03 copy for KD-Tree,
GRID_META["x_lv03"] = GRID_META["x"]
GRID_META["y_lv03"] = GRID_META["y"]
xm, ym = transformer.transform(GRID_META["x_lv03"].to_numpy(), GRID_META["y_lv03"].to_numpy())
GRID_META["x"] = xm     # LV95 for plotting
GRID_META["y"] = ym



def get_boundary_trace():
    """Generates a Plotly scatter trace for Swiss warning region boundaries."""
    lon, lat = [], []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                lon.extend([pt[0] for pt in poly.exterior.coords] + [None])
                lat.extend([pt[1] for pt in poly.exterior.coords] + [None])
        elif geom.geom_type == "Polygon":
            lon.extend([pt[0] for pt in geom.exterior.coords] + [None])
            lat.extend([pt[1] for pt in geom.exterior.coords] + [None])

    return go.Scatter(
        x=lon,
        y=lat,
        mode="lines",
        line=dict(width=2, color="black"),
        name="Switzerland Boundary",
        opacity=0.5,
        showlegend=False,
        hoverinfo="skip"
    )

# Build KDTree for assigning Stations to Clusters. KD-Tree MUST be on LV03 coords, and queries must use LV03 too
grid_meta = pd.read_csv(
    os.path.join(BASE_PATH, "aggregated_grid_time_series_lv03.csv"),
    usecols=["grid_id", "x", "y"]
).drop_duplicates()

grid_meta = grid_meta[np.isfinite(grid_meta["x"]) & np.isfinite(grid_meta["y"])]
grid_meta["grid_num"] = grid_meta["grid_id"].str.replace("grid_", "").astype(int)
grid_meta = grid_meta.sort_values("grid_num").reset_index(drop=True)

grid_points_lv03  = grid_meta[["x", "y"]].values     # LV03
grid_id_list      = grid_meta["grid_id"].tolist()
tree = cKDTree(grid_points_lv03)

# Function to create square polygons around cluster points
def make_cluster_polygons(df, size=5000):
    half = size / 2

    # Drop rows with missing or invalid coordinates
    df_clean = df.dropna(subset=["x", "y"]).copy()
    df_clean = df_clean[np.isfinite(df_clean["x"]) & np.isfinite(df_clean["y"])]

    # Create square polygons
    geometries = [
        box(row["x"] - half, row["y"] - half, row["x"] + half, row["y"] + half)
        for _, row in df_clean.iterrows()
    ]

    return gpd.GeoDataFrame(df_clean, geometry=geometries, crs="EPSG:2056")

GLOBAL_MIN_E = float(np.floor(station_df["elevation"].min() / 500) * 500)
GLOBAL_MAX_E = float(np.ceil (station_df["elevation"].max() / 500) * 500)
GLOBAL_ELEV_BINS = np.arange(GLOBAL_MIN_E, GLOBAL_MAX_E + 500, 500)
GLOBAL_INTERVALS = pd.IntervalIndex.from_breaks(GLOBAL_ELEV_BINS, closed="left")


# ------------------ Custom colour scales & attribute labels ------------------ #

    # Your existing attribute-to-label mapping
attribute_mapping = {
    "dlModelFx_Prob3": "Probability from danger level model",
    "instabModelFx": "Instability model result",
    "spontLawModelFx": "Spontaneous avalanche model result",
    "hn24": "Snow depth in the last 24 hours (cm)",
    "hn72": "Snow depth in the last 72 hours (cm)",
    "aspect": "Slope aspect (degrees)",
    "date": "Forecast date",
    "X,y": "Coordinates (Swiss LV03 system)",
    "slfCode": "Station identifier (SLF)",
    "elevation": "Elevation (m)",
    "sector_id": "Warning region ID",
    "probability_level_1": "Probability of danger level 1",
    "probability_level_2": "Probability of danger level 2",
    "probability_level_3": "Probability of danger level 3",
    "probability_level_4": "Probability of danger level 4",
    "level_expected": "Expected danger level (weighted average)",
    "P_s2": "Probability of avalanche size ≥ 2",
    "P_s3": "Probability of avalanche size ≥ 3",
    "P_s4": "Probability of avalanche size ≥ 4",
    "P_s5": "Probability of avalanche size ≥ 5",
    "size_expected": "Expected avalanche size",
    "hs": "Measured snow depth at station (cm)",
    "pMax_decisive": "Probability of instability in weakest snow layer",
    "depth_decisive": "Depth of weakest snow layer (cm)",
    "np_pMax": "Instability probability of weakest non-persistent layer",
    "np_depth": "Depth of weakest non-persistent layer (cm)",
    "pwl_pMax": "Instability probability of weakest persistent layer",
    "pwl_depth": "Depth of weakest persistent layer (cm)",
    "probability_natAval": "Probability of a natural dry-snow avalanche",
    "dl_substep": "SLF Danger Level Substep"
}

    # Other station‐level histograms
vars_to_plot = [
        "dlModelFx_Prob3","instabModelFx","spontLawModelFx",
        "hn24","hn72","elevation",
        "probability_level_1","probability_level_2",
        "probability_level_3","probability_level_4",
        "level_expected","P_s2","P_s3","P_s4","P_s5",
        "size_expected","hs","pMax_decisive","depth_decisive",
        "np_pMax","np_depth","pwl_pMax","pwl_depth",
        "probability_natAval"
]

colour_scale_grid = [
    [0.0, "rgba(0,0,0,0)"],  # Transparent (background)
    [0.2, "#00FF00"],  # Green
    [0.4, "#FFFF00"],  # Yellow
    [0.6, "#FFA500"],  # Orange
    [0.8, "#FF0000"],  # Red
    [1.0, "#800080"]  # Purple
]

def map_rescaled_to_label(value):

    try:
        v = float(value)
    except:
        return "No Data"

    # Clamp into [0.0, 5.0]
    v_clamped = min(max(v, 0.0), 5.0)

    mapping = {
        0.0: "1-",
        0.5: "1",
        1.0: "1+",
        1.5: "2-",
        2.0: "2",
        2.5: "2+",
        3.0: "3-",
        3.5: "3",
        4.0: "3+",
        4.5: "4-",
        5.0: "4"
    }
    return mapping.get(round(v_clamped, 1), "No Data")

ALL_SUBLEVELS = [
    "1-", "1", "1+",
    "2-", "2", "2+",
    "3-", "3", "3+",
    "4-", "4",
    "No Data"
]

# Sample Hot at exactly 12 points
scale = sample_colorscale(
    px.colors.sequential.Hot_r,
    [i / (len(ALL_SUBLEVELS) - 1) for i in range(len(ALL_SUBLEVELS))]
)
color_discrete_map = { lvl: scale[i] for i, lvl in enumerate(ALL_SUBLEVELS) }

# make “No Data” fully transparent
color_discrete_map["No Data"] = "rgba(0,0,0,0)"


# ------------------ Dash App Layout ------------------ #

COMMON_GRAPH_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["pan2d","autoScale2d","toImage"],
    "modeBarButtonsToRemove": ["zoom2d","select2d","lasso2d","resetScale2d"],
    "scrollZoom": True,
}

first_cluster = pd.to_datetime(cluster_dates.min()).date().isoformat()
last_cluster  = pd.to_datetime(cluster_dates.max()).date().isoformat()

first_point = pd.to_datetime(station_df["date"].min()).date().isoformat()
last_point  = pd.to_datetime(station_df["date"].max()).date().isoformat()

first_grid = pd.to_datetime(dangerlevel_df["date"].min()).date().isoformat()
last_grid  = pd.to_datetime(dangerlevel_df["date"].max()).date().isoformat()


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.layout = dbc.Container([
    html.H1(
    "Avalanche Forecast Dashboard",
    style={
        "textAlign": "center",
        "marginTop": "20px",
        "marginBottom": "30px",
        "fontSize": "36px",
        "fontWeight": "bold"
        }
    ),

    dcc.Tabs(id="tabs", value="cluster-tab", children=[

        # ----- Cluster Analysis Tab ----------------------------#
        dcc.Tab(
            label="Cluster Analysis",
            value="cluster-tab",
            style={'fontSize': '18px','fontWeight': 'bold','padding': '10px 20px'},
            selected_style={'fontSize': '20px','fontWeight': 'bold','padding': '10px 20px'},
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Select Date"),
                            dcc.DatePickerSingle(
                                id="cluster-date-picker",
                                min_date_allowed=first_cluster,
                                max_date_allowed=last_cluster,
                                date=first_cluster,
                                style={"width": "100%"}
                            ),
                            html.Div(id="cluster-title",
                                     style={"textAlign": "center", "marginBottom": "5px"}),
                            html.Div(style={"marginBottom": "15px"}),
                            html.Label("Click on a cluster to view histograms"),
                            html.Div(id="selected-cluster-display",
                                     style={"fontWeight": "bold", "fontSize": "16px",
                                            "marginBottom": "10px", "textAlign": "center"}),
                            dcc.Loading(
                                id="loading-cluster",
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="cluster-map",
                                        config=COMMON_GRAPH_CONFIG,
                                        style={'height': '90vh','width': '100%','margin': '0 auto'}
                                    )
                                ]
                            )
                        ], style={"paddingTop": "30px", "paddingRight": "10px"})
                    ], width=9),

                    dbc.Col([
                        html.Div(id="cluster-histograms",
                                 style={"height": "85vh", "overflowY": "auto",
                                        "padding": "15px", "border": "1px solid #ccc",
                                        "backgroundColor": "#f9f9f9", "marginTop": "250px"})
                    ], width=3)
                ])
            ]
        ),

        # ----- Point-Data Tab ----------------------------#
        dcc.Tab(
            label="Point-Data",
            value="point-data",
            style={'fontSize': '18px','fontWeight': 'bold','padding': '10px 20px'},
            selected_style={'fontSize': '20px','fontWeight': 'bold','padding': '10px 20px'},
            children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-point",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id='point-data-chart',
                                    config=COMMON_GRAPH_CONFIG,
                                    style={'height': '100vh','width': '100%','margin': '0 auto'}
                                )
                            ]
                        )
                    ], width=9, style={"paddingTop": "30px"}),

                    dbc.Col([
                        html.Div([
                            html.Label("Select Date"),
                            dcc.DatePickerSingle(
                                id='point-date-picker',
                                min_date_allowed=first_point,
                                max_date_allowed=last_point,
                                date=first_point,
                                style={"width": "100%"}
                            ),
                            html.Div(style={"marginBottom": "15px"}),
                            html.Label("Select Attribute"),
                            dcc.Dropdown(
                                id='point-attribute-dropdown',
                                options=[
                                    {
                                        "label": f"{col.replace('_',' ').title()} [{col}]",
                                        "value": col
                                    } for col in station_df.columns
                                    if col not in ["date","x","y"]
                                ],
                                value="probability_level_3",
                                clearable=False,
                                style={"width": "100%"}
                            ),
                        ], style={"paddingTop": "30px", "paddingRight": "10px"})
                    ], width=3)
                ])
            ]
        ),

        # ----- Grid-Data Tab ----------------------------#
        dcc.Tab(
            label="Grid-Data",
            value="grid-data",
            style={'fontSize': '18px','fontWeight': 'bold','padding': '10px 20px'},
            selected_style={'fontSize': '20px','fontWeight': 'bold','padding': '10px 20px'},
            children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-grid",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id='grid-data-chart',
                                    config=COMMON_GRAPH_CONFIG,
                                    style={'height': '95vh','width': '100%','margin': '0 auto'}
                                )
                            ]
                        )
                    ], width=9, style={"paddingTop": "30px"}),

                    dbc.Col([
                        html.Div([
                            html.Label("Select Date"),
                            dcc.DatePickerSingle(
                                id='grid-date-picker',
                                min_date_allowed=first_grid,
                                max_date_allowed=last_grid,
                                date=first_grid,
                                style={"width": "100%"}
                            ),
                            html.Div(style={"marginBottom": "15px"}),

                            html.Label("Select Dataset"),
                            dcc.Dropdown(
                                id='grid-dataset-dropdown',
                                options=[
                                    {'label': 'Danger Level Forecast [dangerlevel]',       'value': 'dangerlevel'},
                                    {'label': 'Instability Forecast [instab]',             'value': 'instab'},
                                    {'label': 'Spont. Law Forecast [spontlaw]',            'value': 'spontlaw'},
                                    {'label': 'Substeps [sublv]',                          'value': 'sublv'},
                                    {'label': 'Continuous Level [level_continuous]',       'value': 'level_continuous'},
                                ],
                                value='dangerlevel',
                                clearable=False,
                                style={"width": "100%"}
                            ),
                            html.Div(style={"marginBottom": "15px"}),

                            html.Label("Filter Elevation Range"),
                            html.Div([
                                dcc.RangeSlider(
                                    id='elevation-slider',
                                    min=dangerlevel_df['ele'].min(),
                                    max=dangerlevel_df['ele'].max(),
                                    step=50,
                                    value=[dangerlevel_df['ele'].min(), dangerlevel_df['ele'].max()],
                                    marks={int(e): str(int(e))
                                           for e in np.linspace(dangerlevel_df['ele'].min(),
                                                               dangerlevel_df['ele'].max(), 5)}
                                ),
                                html.Div(id='elevation-message',
                                         style={"color": "gray","fontStyle": "italic",
                                                "marginTop": "5px","fontSize": "0.9em"})
                            ], 
                            id="elevation-slider-container",
                            style={"marginBottom": "20px"}),

                            html.Label("Select Aspect"),
                            html.Div([
                                html.Button("N", id="btn-0",   n_clicks=0, style={"margin":"5px"}),
                                html.Button("E", id="btn-90",  n_clicks=0, style={"margin":"5px"}),
                                html.Button("S", id="btn-180", n_clicks=0, style={"margin":"5px"}),
                                html.Button("W", id="btn-270", n_clicks=0, style={"margin":"5px"}),
                                html.Br(),
                                html.Button("Clear",
                                            id="btn-clear",
                                            n_clicks=0,
                                            style={"marginTop": "10px",
                                                   "backgroundColor": "lightgray",
                                                   "padding": "6px 12px",
                                                   "border": "1px solid black"})
                            ], style={"textAlign": "center"}),
                            dcc.Store(id="selected-aspects", data=[])
                        ], style={"paddingTop": "30px", "paddingRight": "10px"})
                    ], width=3)
                ])
            ]
        ),

    ])
], fluid=True)

# ------------------ Callbacks ------------------ #
# Define the LV95 extent for the cluster map
MINX, MINY, MAXX, MAXY = gdf.total_bounds
XPAD, YPAD = 5000, 5000
X_MIN, X_MAX = MINX - XPAD, MAXX + XPAD
Y_MIN, Y_MAX = MINY - YPAD, MAXY + YPAD


# Cache a single boundary trace so we don’t rebuild it each time:
BOUNDARY_TRACE = get_boundary_trace()

@app.callback(
    [Output("cluster-title", "children"),
     Output("cluster-map",   "figure")],
    [Input("tabs", "value"),
     Input("cluster-date-picker", "date"),
     Input("cluster-map", "clickData")]
)
def update_cluster_map(tab_value, selected_date, clickData):
    if tab_value != "cluster-tab" or not selected_date:
        return ["", dash.no_update]

    sel_dt = pd.to_datetime(selected_date).normalize()

    # Robust date filter
    today = cluster_df[cluster_df["date"].dt.normalize() == sel_dt].copy()

    # Align join keys
    GRID_META["grid_id"] = GRID_META["grid_id"].astype(str)
    today["grid_id"]     = today["grid_id"].astype(str)

    # Common layout: fixed LV95 extent
    layout_kwargs = dict(
        xaxis=dict(range=[X_MIN, X_MAX], autorange=False, constrain="domain"),
        yaxis=dict(range=[Y_MIN, Y_MAX], autorange=False, constrain="domain", scaleanchor="x"),
        dragmode="pan",
        margin=dict(l=10, r=10, t=40, b=10),
        height=800
    )

    # No clusters this date
    if today.empty or (today["cluster_id"] < 0).all():
        fig = go.Figure()
        fig.add_trace(BOUNDARY_TRACE)
        fig.update_layout(**layout_kwargs)
        title = html.Div([
            html.H3("No Clusters Found", style={"margin": "0"}),
            html.H5(f"Date: {sel_dt.date()}", style={"margin": "0", "fontStyle": "italic"})
        ], style={"textAlign": "center"})
        return [title, fig]

    # Label lookup (mode substep -> label)
    mode_num = (
        today[today["cluster_id"] >= 0]
        .groupby("cluster_id")["dl_substep_numeric"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index(name="mode_substep_num")
    )
    mode_num["danger_label"] = mode_num["mode_substep_num"].apply(map_rescaled_to_label)
    label_map = dict(zip(mode_num["cluster_id"], mode_num["danger_label"]))

    # Merge coords with today's cluster ids
    full = GRID_META[["grid_id","x","y"]].copy().merge(
        today[["grid_id","cluster_id"]], on="grid_id", how="left"
    )
    full["cluster_id"]   = full["cluster_id"].fillna(-1).astype(int)
    full["danger_label"] = full["cluster_id"].map(label_map).fillna("No Data")
    cluster_cells = full[full["cluster_id"] >= 0]

    # If merge produced nothing, show diagnostic
    if cluster_cells.empty:
        fig = go.Figure()
        fig.add_trace(BOUNDARY_TRACE)
        fig.update_layout(**layout_kwargs)
        title = html.Div([
            html.H3("No Cluster Cells Found (grid_id mismatch)", style={"margin": "0"}),
            html.H5(f"Date: {sel_dt.date()}", style={"margin": "0", "fontStyle": "italic"})
        ], style={"textAlign": "center"})
        return [title, fig]

    # Figure + boundary
    fig = go.Figure()
    fig.add_trace(BOUNDARY_TRACE)

    # Highlight if clicked
    selected_cid = None
    if clickData and clickData.get("points"):
        selected_cid = clickData["points"][0]["customdata"][0]

    # Scatter points by cluster id
    for cid, subdf in cluster_cells.groupby("cluster_id"):
        danger = label_map.get(cid, "No Data")
        mstyle = dict(size=6, opacity=0.9, color=color_discrete_map.get(danger, "rgba(0,0,0,0.2)"))
        if cid == selected_cid:
            mstyle.update(size=8, line=dict(width=0.6, color="black"))

        fig.add_trace(go.Scatter(
            x=subdf["x"], y=subdf["y"],
            mode="markers",
            marker=mstyle,
            customdata=np.stack([subdf["cluster_id"], subdf["danger_label"]], axis=1),
            hovertemplate="Cluster ID: %{customdata[0]}<br>Danger: %{customdata[1]}<extra></extra>",
            showlegend=False
        ))

    # One legend entry per used sublevel
    used = [lvl for lvl in ALL_SUBLEVELS if lvl in set(label_map.values()) and lvl != "No Data"]
    for danger in used:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=6, color=color_discrete_map[danger]),
            name=f"Danger {danger}", showlegend=True
        ))

    fig.update_layout(**layout_kwargs)

    # Title
    title_div = html.Div([
        html.H3("Cluster Danger Levels", style={
            "margin":"0","fontFamily":"Arial, sans-serif","fontSize":"18px",
            "fontWeight":"bold","color":"#324857"
        }),
        html.H5(f"Date: {sel_dt.date()}", style={
            "margin":"0","fontFamily":"Arial, sans-serif","fontSize":"18px",
            "fontStyle":"italic","color":"#324857"
        })
    ], style={"textAlign":"center"})

    return [title_div, fig]



@app.callback(
    [
      Output("selected-cluster-display", "children"),
      Output("cluster-histograms",       "children")
    ],
    [
      Input("cluster-map",        "clickData"),
      Input("cluster-date-picker","date")
    ]
)
def generate_histograms(clickData, selected_date):

    # Guard clauses
    if not clickData or not selected_date:
        return (
          "No Cluster Selected",
          html.Div("Click on a cluster point and select a date to view charts.")
        )

    sel_dt = pd.to_datetime(selected_date).normalize()

    # Use already-loaded cluster_df
    today = cluster_df[cluster_df["date"].dt.normalize() == sel_dt]

    # Extract the clicked cluster ID and its sub-level label
    cid, sub_level = clickData["points"][0]["customdata"]
    if cid < 0 or sub_level == "No Data":
        return (
          "No Cluster Selected",
          html.Div("You clicked outside of any cluster.")
        )

    header = f"Cluster Sub-Level: {sub_level}"
    main_level = sub_level[0]
    target_main = int(main_level)

    # Get station data for this date and assign to grid/cluster 
    sdf = station_df.copy()
    sdf["date"] = sdf["date"].dt.normalize()
    sdf = sdf[sdf["date"] == sel_dt]
    if sdf.empty:
        return header, html.Div("No station data for this date.")

    # Assign each station to its nearest grid cell (prebuilt KDTree & mapping)
    coords = sdf[["x_lv03","y_lv03"]].values 
    _, idxs = tree.query(coords)
    sdf["grid_id"] = [grid_id_list[i] for i in idxs]

    # Attach cluster_id and substep numeric for those grids on this date
    sdf = sdf.merge(
        today[["grid_id", "cluster_id", "dl_substep_numeric"]],
        on="grid_id", how="left"
    )

    # Keep only stations that belong to the clicked cluster
    sdf = sdf[sdf["cluster_id"] == cid]
    if sdf.empty:
        return header, html.Div("No station data in this cluster.")

    # Fixed global elevation bins
    # Use globally fixed bins so the radar range is constant across dates/clusters
    sdf["elev_bin"] = pd.cut(
        sdf["elevation"],
        bins=GLOBAL_ELEV_BINS,
        right=False,
        include_lowest=True
    )
    # Ensure categories exactly match global intervals
    sdf["elev_bin"] = sdf["elev_bin"].cat.set_categories(GLOBAL_INTERVALS, ordered=True)

    # Aspect binning into cardinal sectors
    def asp_bin(a):
        if (a>=315) | (a<45): return "N"
        if 45<=a<135:         return "E"
        if 135<=a<225:        return "S"
        return "W"
    sdf["asp_bin"] = sdf["aspect"].apply(asp_bin)

    # Proportion-by-aspect for the target level
    def main_from_numeric(v):
        label = map_rescaled_to_label(v)
        if label == "No Data":
            return None
        return int(label[0])

    sdf["station_main"] = sdf["dl_substep_numeric"].apply(main_from_numeric)

    # Totals and matches per (elev_bin, asp_bin) inside cluster
    totals = (
        sdf.groupby(["elev_bin","asp_bin"], observed=True)
           .size()
           .rename("n_total")
    )
    matches = (
        sdf[sdf["station_main"] == target_main]
           .groupby(["elev_bin","asp_bin"], observed=True)
           .size()
           .rename("n_match")
    )
    agg = pd.concat([totals, matches], axis=1).fillna(0).reset_index()
    agg["fraction"] = np.where(agg["n_total"] > 0, agg["n_match"] / agg["n_total"], 0.0)

    # Threshold + opacity mapping for intuitive prevalence
    PROPORTION_THRESHOLD = 0.20 
    def frac_to_opacity(p):
        if p < PROPORTION_THRESHOLD: return 0.0
        if p < 0.33: return 0.3
        if p < 0.66: return 0.6
        if p < 0.90: return 0.85
        return 1.0

    # Build equal-thickness radar
    angle_map = {"N":90, "E":0, "S":270, "W":180}
    cats = list(GLOBAL_INTERVALS)                         
    bin_index = {cat: i for i, cat in enumerate(cats)}  
    n_bins = len(cats)

    # Radial ticks at global band outer edges 
    tick_vals  = list(range(1, n_bins))
    tick_texts = [str(int(b)) for b in GLOBAL_ELEV_BINS[1:]]

    fig = go.Figure()
    for row in agg.itertuples():
        if row.fraction <= 0:
            continue
        i = bin_index.get(row.elev_bin)
        if i is None:
            continue
        op = frac_to_opacity(row.fraction)
        if op == 0.0:
            continue
        fig.add_trace(go.Barpolar(
            r    = [1],                         # uniform band thickness
            base = [i],                        
            theta= [angle_map[row.asp_bin]],
            width= [90],
            marker=dict(
                color=color_discrete_map[main_level],
                line_color="white", line_width=1,
                opacity=op
            ),
            hovertemplate=(
                f"Elev: {int(row.elev_bin.left)}–{int(row.elev_bin.right)} m<br>"
                f"Aspect: {row.asp_bin}<br>"
                f"Level {target_main} here: "
                f"{int(row.n_match)}/{int(row.n_total)} ({row.fraction*100:.0f}%)<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title = f"<b>Radar — Level {target_main}</b>",
        polar = dict(
            # keep room so 'S' at 270° is visible
            domain=dict(x=[0.02, 0.98], y=[0.10, 0.95]),
            radialaxis=dict(
                range=[0, n_bins],          
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_texts,
                ticks="outside",
                ticklen=8,
                tickfont=dict(size=11),
                showline=True,
                gridcolor="rgba(0,0,0,0.15)"
            ),
            angularaxis=dict(
                rotation=0, direction="counterclockwise",
                tickmode="array",
                tickvals=[90, 0, 270, 180], 
                ticktext=["N", "E", "S", "W"],
                ticks="outside",
                ticklen=8,
                tickfont=dict(size=12),
                showline=True
            )
        ),
        margin=dict(t=40, b=60, l=20, r=20),
        showlegend=False,
        height=420,
        uirevision="radar"
    )

    # Non-rotatable radar component
    radar = dcc.Graph(
        figure=fig,
        config={"staticPlot": True, "displayModeBar": False, "displaylogo": False},
        style={"marginBottom": "20px"}
    )

    # Station-level histograms
    hist_list = []
    for var in vars_to_plot:
        if var not in sdf.columns:
            continue
        clean = sdf[var].dropna()
        if clean.empty:
            hfig = px.histogram(title=f"<b>{attribute_mapping.get(var,var)} — No Data</b>")
        else:
            med = clean.median()
            hfig = px.histogram(
                sdf, x=var, nbins=30,
                title=f"<b>{attribute_mapping.get(var,var)}</b>"
            )
            hfig.add_vline(x=med, line_color="red", line_width=2)
            hfig.add_annotation(
                text=f"<b>Median: {med:.2f}</b>",
                x=0.98, y=0.96, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12), align="right"
            )
        hfig.update_layout(title_font_size=12, margin=dict(t=30,l=10,r=10,b=10))
        hist_list.append(
            dcc.Graph(
                figure=hfig,
                config={"displayModeBar": False, "displaylogo": False},
                style={"height":"300px"}
            )
        )

    return header, [radar] + hist_list


@app.callback(
    Output('point-data-chart', 'figure'),
    Input('point-date-picker', 'date'),
    Input('point-attribute-dropdown', 'value')
)
def update_point_chart(selected_date, selected_attribute):
    selected_date = pd.Timestamp(selected_date)
    filtered_df = station_df[station_df['date'] == selected_date].copy()

    # Clean and ensure the attribute column is numeric
    filtered_df[selected_attribute] = pd.to_numeric(filtered_df[selected_attribute], errors='coerce')
    filtered_df = filtered_df.dropna(subset=[selected_attribute])

    readable_label = attribute_mapping.get(selected_attribute, selected_attribute)


    # Create the scatter plot
    fig = px.scatter(
       filtered_df,
       x="x",
       y="y",
       color=selected_attribute,
       color_continuous_scale="hot_r",
       template="plotly_white",
       labels={selected_attribute: selected_attribute},
       opacity=0.99
    )

   # Build a two-line title 
    main_title = readable_label
    subtitle   = f"Date: {selected_date.strftime('%Y-%m-%d')}"
    full_title = f"<b>{main_title}</b><br><i>{subtitle}</i>"

    fig.update_layout(
    title={
        'text': full_title,
        'x': 0.5,
        'xanchor': 'center'
    },
    title_font=dict(family="Arial, sans-serif", size=18, color="#324857")
)

    # Add a dark circle around each point for visibility
    fig.update_traces(
        marker=dict(
            size=8,
            line=dict(width=0.5, color='black')
        )
    )

    # Collect all boundary points first
    boundary_lon = []
    boundary_lat = []

    for _, row in gdf.iterrows():
        if row.geometry.geom_type == "MultiPolygon":
            for polygon in row.geometry.geoms:
                boundary_lon.extend([point[0] for point in polygon.exterior.coords] + [None])  # Add None to separate shapes
                boundary_lat.extend([point[1] for point in polygon.exterior.coords] + [None])
        elif row.geometry.geom_type == "Polygon":
            boundary_lon.extend([point[0] for point in row.geometry.exterior.coords] + [None])
            boundary_lat.extend([point[1] for point in row.geometry.exterior.coords] + [None])

    # Add only one trace for all boundaries
    fig.add_trace(go.Scatter(
        x=boundary_lon,
        y=boundary_lat,
        mode="lines",
        line=dict(width=2, color="black"),
        name="Switzerland Boundary",
        opacity=0.5,
        showlegend=False
    ))

    fig.update_layout(dragmode="pan")
    return fig

@app.callback(
    # Outputs: the new store data, then each button’s style
    [Output("selected-aspects", "data")] +
    [Output(f"btn-{angle}", "style") for angle in [0, 90, 180, 270]] +
    [Output("btn-clear", "style")],
    # Inputs: clicks on each aspect button and clear
    [Input(f"btn-{angle}", "n_clicks") for angle in [0, 90, 180, 270]] +
    [Input("btn-clear", "n_clicks")],
    # Add store as State here:
    [State("selected-aspects", "data")],
    prevent_initial_call=True
)
def update_aspect_selection(n0, n90, n180, n270, clear_clicks, prev_selected):
    triggered = ctx.triggered_id

    # Use the prev_selected state
    current = set(prev_selected or [])

    # Map IDs to aspect values
    aspect_values = {"btn-0": 0, "btn-90": 90, "btn-180": 180, "btn-270": 270}

    # Handle clear button
    if triggered == "btn-clear":
        current.clear()

    # Handle aspect buttons
    elif triggered in aspect_values:
        val = aspect_values[triggered]
        if val in current:
            current.remove(val)
        else:
            # Up to 3, otherwise clear
            if len(current) < 3:
                current.add(val)
            else:
                current.clear()

    # Build the styles
    base_btn_style = {"margin": "5px"}
    btn_styles = [
        {**base_btn_style, "backgroundColor": ("lightblue" if angle in current else "white")}
        for angle in [0, 90, 180, 270]
    ]
    base_clear_style = {"marginTop": "10px", "padding": "6px 12px", "border": "1px solid black"}
    clear_style = {**base_clear_style, "backgroundColor": ("red" if current else "lightgray")}

    # Return new store value + styles
    return [list(current), *btn_styles, clear_style]

@app.callback(
    [
      Output("elevation-slider",           "disabled"),
      Output("elevation-slider-container", "style"),
      Output("elevation-message",          "children"),
    ],
    [Input("grid-dataset-dropdown", "value")]
)
#Define the callback to toggle the elevation slider
def toggle_elevation_slider(dataset):
    if dataset == "sublv" or dataset == "level_continuous":
        return (
            True,
            {"opacity": 0.9, "pointerEvents": "none"},
            "Elevation filter disabled: no elevation data for sub-level dataset."
        )
    else:
        return (
            False,
            {"opacity": 1.0, "pointerEvents": "auto"},
            ""
        )
    
@app.callback(
    Output('grid-data-chart','figure'),
    [
      Input('grid-date-picker','date'),
      Input('grid-dataset-dropdown','value'),
      Input('elevation-slider','value'),
      Input('selected-aspects','data')
    ]
)
def update_grid_chart(selected_date, selected_dataset, elevation_range, selected_aspects):
    # Parse date
    dt = pd.Timestamp(selected_date)

    # Title lookup
    title_map = {
        "dangerlevel":   "The probability that the regional avalanche danger level is ≥ danger level 3",
        "instab":        "The probability that the simulated snow cover will be classified as unstable",
        "spontlaw":      "The probability of spontaneous dry snow avalanches",
        "sublv":         "The SLF sub-levels of the avalanche danger level",
        "level_continuous": "The SLF continuous avalanche danger level",
    }

    # Pick df & zcol per dataset
    if selected_dataset == 'dangerlevel':
        df = dangerlevel_df[dangerlevel_df.date == dt].copy()
        zcol = 'dlModelFx_Prob3'
    elif selected_dataset == 'instab':
        df = instab_df[instab_df.date == dt].copy()
        zcol = 'instabModelFx'
    elif selected_dataset == 'spontlaw':
        df = spontlaw_df[spontlaw_df.date == dt].copy()
        zcol = 'spontLawModelFx'
    elif selected_dataset == 'sublv':
        df = sublevels_df[sublevels_df.date == dt].copy()
        zcol = 'sublevel'
    elif selected_dataset == 'level_continuous':
        df = sublevels_df[sublevels_df.date == dt].copy()
        zcol = 'level_continuous'
        # Bring in ele/aspect for continuous from dangerlevel_df
        coords_sub = df[['x','y']].values
        coords_dl  = dangerlevel_df[['x','y']].values
        tree_dl    = cKDTree(coords_dl)
        _, idxs    = tree_dl.query(coords_sub)
        df['ele']    = dangerlevel_df.iloc[idxs]['ele'].values
        df['aspect'] = dangerlevel_df.iloc[idxs]['aspect'].values
    else:
        return go.Figure()

    # Elevation filter
    min_ele, max_ele = elevation_range
    if 'ele' in df.columns:
        df = df[(df.ele >= min_ele) & (df.ele <= max_ele)]
    # Aspect filter
    if selected_aspects and 'aspect' in df.columns:
        df = df[df.aspect.isin(selected_aspects)]

    # Build the heatmap with the same "hot_r" scale as Point-Data
    fig = px.density_heatmap(
        df,
        x='x',
        y='y',
        z=zcol,
        nbinsx=350,
        nbinsy=350,
        histfunc="avg",
        color_continuous_scale="hot_r",                   
        labels={zcol: attribute_mapping.get(zcol, zcol)}, 
        template="plotly_white"
    )

    # Draw the basemap boundary underneath
    boundary_lon, boundary_lat = [], []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                boundary_lon.extend([pt[0] for pt in poly.exterior.coords] + [None])
                boundary_lat.extend([pt[1] for pt in poly.exterior.coords] + [None])
        elif geom.geom_type == "Polygon":
            boundary_lon.extend([pt[0] for pt in geom.exterior.coords] + [None])
            boundary_lat.extend([pt[1] for pt in geom.exterior.coords] + [None])

    fig.add_trace(go.Scatter(
        x=boundary_lon,
        y=boundary_lat,
        mode="lines",
        line=dict(width=2, color="black"),
        name="Switzerland Boundary",
        opacity=0.2,
        showlegend=False
    ))

    # Title & Layout
    custom_title = title_map.get(selected_dataset, "")
    subtitle     = f"{selected_dataset} on {dt.date()}"
    full_title   = f"<b>{custom_title}</b><br><i>{subtitle}</i>"

    fig.update_layout(
        title={'text': full_title, 'x':0.5, 'xanchor':'center'},
        coloraxis_colorbar=dict(title=attribute_mapping.get(zcol, zcol)),
        margin=dict(l=10, r=10, t=60, b=10),
        title_font=dict(family="Arial, sans-serif", size=18, color="#324857")
    )

    fig.update_layout(dragmode="pan")
    return fig


# ------------------ Run Dash App ------------------ #
if __name__ == "__main__":
    app.run(debug=True)