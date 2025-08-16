# Avalanche Forecasting Clustering & Dashboard

This repository contains the Python code developed for the Master's Thesis  
**“Enhancing Snow Avalanche Forecasting: Developing User-Centered Dashboards for Data Visualization and Decision Support”**.  

The project implements a full pipeline from data preprocessing and clustering to an interactive dashboard for avalanche forecasters.  

---

## Repository Structure

### Core Workflow
1. **`1_Grid_Summary.py`**  
   Aggregates multiple avalanche-related datasets (station data, model forecasts, sublevels) onto a 5×5 km LV03 grid.  
   Outputs the unified grid time series used in later steps.

2. **`2_Bulletin_Extraction.py`**  
   Extracts and cleans SLF avalanche bulletin data, producing ground-truth danger level labels.

3. **`3_Parameter_Analysis.py`**  
   Builds the preprocessing + clustering pipeline and evaluates feature transformations and dimensionality reduction stages.

4. **`4_Parameter_Tune.py`**  
   Uses **Optuna** to optimize HDBSCAN* hyperparameters through ensemble evaluation.

5. **`5_Clustering_Pipeline.py`**  
   Applies the clustering pipeline (with tuned parameters) to the grid dataset.  
   Prepares outputs for the **Clustering Tab** of the dashboard.

6. **`6_Dashboard_Launcher.py`**  
   Launches the interactive **Dash** application to visualize snow, weather, and avalanche data.  
   Provides expert users with maps, histograms, radar plots, and clustering outputs.

---

### Method Validation (Helper Scripts)
These scripts are not part of the dashboard but were used to validate methodological choices documented in the thesis:

- **`Grid_v_GridPoint.py`**  
  Compares clustering performance using only gridded data vs. gridded + point data.  
  → Justifies the exclusive use of grid data for clustering.

- **`Mean_v_Max.py`**  
  Compares mean vs. max aggregation strategies when summarizing data onto the grid.  
  → Shows that mean aggregation yields better ARI than the max (worst-case) approach used operationally today.

- **`Method_Selector.py`**  
  Benchmarks clustering methods (ensemble HDBSCAN*, baseline HDBSCAN, K-Means) across days.  
  → Justifies ensemble HDBSCAN* as the clustering algorithm of choice.

---
