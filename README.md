# MaxEnt Studio Pro 🌍
### Modern Species Distribution & Ecological Niche Modeling Suite

> **Concept & Design by K S Chandra 2026 (Source: MaxEnt Java App)**

[![Version](https://img.shields.io/badge/MaxEnt-v3.4.4%20Integrated-success.svg)](#)
[![File Upload](https://img.shields.io/badge/Raster%20Upload-Up%20to%202%20GB-blue.svg)](#)
[![Engines](https://img.shields.io/badge/Dual%20Engine-Java%20%2B%20Python-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Welcome to **MaxEnt Studio Pro**, a modern, user-friendly, and enterprise-grade Species Distribution Modeling (SDM) application. 

This platform transforms the traditional 2006-era MaxEnt software into an interactive, error-free GIS modeling suite that directly integrates the official AMNH binary (`maxent.jar`), supports modern **GeoTIFF rasters up to 2 GB**, provides interactive satellite web maps, and delivers publication-ready figures.

---

## 📑 Table of Contents

- [🚀 Quickstart: How to Launch](#-quickstart-how-to-launch)
- [✨ Key Features](#-key-features)
- [📂 Folder Contents & Architecture](#-folder-contents--architecture)
- [🗺️ Supported Data Formats & 2 GB Uploads](#️-supported-data-formats--2-gb-uploads)
- [⚙️ Modeling Engines (Java vs. Python)](#️-modeling-engines-java-vs-python)
- [🎛️ Parameters & Ecological Presets](#️-parameters--ecological-presets)
- [🖥️ Step-by-Step User Guide](#️-step-by-step-user-guide)
- [📊 Evaluation Metrics & Diagnostics](#-evaluation-metrics--diagnostics)
- [📥 Export & Publication Suite](#-export--publication-suite)
- [🔧 Troubleshooting & FAQ](#-troubleshooting--faq)
- [📚 Scientific References](#-scientific-references)

---

## 🚀 Quickstart: How to Launch

You have three easy ways to launch the application:

### Option 1: 1-Click Desktop Launcher (Recommended)
Simply **double-click** the file located in your downloads folder or project directory:
```text
c:\Users\DuDu-PC\Downloads\maxent\Launch_MaxEnt_Studio.bat
```
or in this folder:
```text
run_studio.bat
```
*This starts the server and automatically opens the interactive studio in your default web browser (`http://localhost:8501`).*

### Option 2: Launch via Terminal (Streamlit)
```powershell
streamlit run app.py --server.maxUploadSize 2048
```

### Option 3: Traditional MaxEnt Java GUI
If you ever need the legacy 2006 Java Swing interface, double-click:
```text
c:\Users\DuDu-PC\Downloads\maxent\maxent.bat
```
*(This has been modernized to allocate 4 GB of RAM `-Xmx4096m` and check for 64-bit Java automatically).*

---

## ✨ Key Features

- **2 GB Raster Upload Limit**: Directly upload massive, high-resolution environmental GIS layers (WorldClim, CHELSA, Copernicus, Bio-ORACLE, USGS DEM).
- **Direct GeoTIFF (`.tif`) Support**: No need to manually convert files to ASCII. The app reads GeoTIFF, verifies projections, and handles alignment in the background.
- **Zero-Error Raster Aligner**: Automatically resamples and snaps slight extent/resolution differences between layers, preventing the notorious *"Layers do not have the same extent"* crash.
- **Smart Coordinate Detection**: Automatically identifies latitude and longitude columns regardless of casing or format (`lat`, `latitude`, `decimalLatitude`, `y`, `lon`, `longitude`, `decimalLongitude`, `x`).
- **Dual-Engine Architecture**:
  - **Official MaxEnt (`maxent.jar` v3.4.4)**: The gold-standard Phillips et al. algorithm executed in headless batch mode.
  - **Native Python MaxEnt Engine**: Scikit-learn compatible dual L-BFGS solver with Continuous Boyce Index (runs completely offline without Java).
  - **Comparative Benchmarking**: Run both engines side-by-side on the same data.
- **Interactive GIS Web Map**: Zoom, pan, and inspect your species distribution over **Esri High-Resolution Satellite**, **Esri World Topographic**, and **Street** base layers with dynamic colormaps and opacity sliders.
- **Interactive Plotly Analytics**: Hover-enabled ROC curve with optimal threshold marker, variable contribution bar charts, Jackknife tests, and marginal response curves.
- **1-Click Test Drive**: Pre-loaded with a complete 4-variable bioclimatic dataset and species occurrences so you can test the entire workflow with zero setup.

---

## 📂 Project Structure

```text
maxent_presence_model/
├── maxent/
│   ├── __init__.py           # Package exports
│   ├── java_engine.py        # Official maxent.jar bridge & data pipeline
│   ├── model.py              # Native Python MaxEnt estimator (scikit-learn compatible)
│   ├── features.py           # Feature transformer (linear, quadratic, product, hinge)
│   ├── metrics.py            # SDM metrics (AUC, Continuous Boyce Index, TSS, Cutoffs)
│   ├── data.py               # Raster aligner, extractor, GeoTIFF/ASCII/ZIP exporters
│   ├── visualizer.py         # Matplotlib diagnostics & Folium GIS mapping
│   └── plotly_viz.py         # Interactive Plotly ROC, Importance & Response curves
├── sample_data/              # Pre-loaded demo datasets
│   ├── dem_elevation.tif     # Elevation GeoTIFF
│   ├── distance_to_water.tif # Distance to water GeoTIFF
│   ├── elevation_contour_slope.tif # Slope GeoTIFF
│   ├── canopy_cover.tif      # Forest canopy cover GeoTIFF
│   └── sample_species_occurrences_latlon.csv # Species presence points
├── .streamlit/
│   └── config.toml           # 2 GB max upload size & UI styling config
├── app.py                    # Interactive Streamlit Web Studio
├── cli.py                    # Terminal CLI (supports --engine java / python)
├── demo.py                   # 1-command runnable test script
├── run_studio.bat            # 1-click Studio launcher
└── README.md                 # Complete documentation
```

---

## 🗺️ Supported Data Formats & 2 GB Uploads

### 1. Environmental Predictor Rasters
- **Supported Formats**: GeoTIFF (`.tif`, `.tiff`), ESRI ASCII (`.asc`), and BIL (`.bil`).
- **File Size Limit**: **Up to 2 GB per file** (configurable up to 8 GB).
- **Spatial Reference**: EPSG:4326 (WGS84 Latitude/Longitude) recommended.
- **Automatic Snapping**: If rasters have slight cell dimension or bounding box mismatches, the application automatically resamples them to match the first reference layer.

### 2. Species Occurrence Records
- **Supported Format**: Comma-Separated Values (`.csv`).
- **Required Columns**: Any recognized latitude and longitude columns.
- **Common Header Formats Supported**:
  - `species, longitude, latitude`
  - `scientificName, decimalLongitude, decimalLatitude` (GBIF standard)
  - `lat, lon`
  - `x, y`
- **Coordinate Validation**: The app automatically detects swapped coordinates, drops invalid/NaN entries, and filters points falling outside raster bounds with transparent user warnings.

---

## ⚙️ Modeling Engines (Java vs. Python)

You can switch engines directly from the sidebar dropdown:

| Feature | 🟢 Official MaxEnt (`maxent.jar`) | 🔵 Native Python MaxEnt Engine |
| :--- | :--- | :--- |
| **Algorithm** | Exact AMNH Phillips et al. v3.4.4 | Scikit-learn regularized dual L-BFGS |
| **Runtime Requirement** | 64-bit Java (JRE/JDK 8 to 26+) | Pure Python (zero Java required) |
| **Features Supported** | Linear, Quadratic, Product, Threshold, Hinge | Linear, Quadratic, Product, Threshold, Hinge |
| **Validation Metrics** | Training/Test AUC, Gain, Thresholds | AUC, Continuous Boyce Index (CBI), TSS |
| **Official Files** | Generates `.lambdas`, omission curves, HTML report | Generates weights, probability arrays |
| **Speed** | Highly optimized C/Java multi-threaded | Accelerated NumPy / SciPy optimization |

---

## 🎛️ Parameters & Ecological Presets

To make modeling accessible for both beginners and experts, four curated presets are provided:

1. **Standard Ecological SDM (Recommended)**:
   - Regularization Multiplier: $\beta = 1.0$
   - Feature Classes: Linear, Quadratic, Product, Hinge
   - Output: Cloglog $[0, 1]$
   - Best for general species distribution modeling with 50+ occurrence records.

2. **Conservative / High Regularization (Prevents Overfitting)**:
   - Regularization Multiplier: $\beta = 2.5$
   - Feature Classes: Linear, Quadratic, Hinge
   - Best when sample sizes are small ($<30$ points) or occurrences suffer from spatial sampling bias.

3. **Complex / High-Resolution Niche**:
   - Regularization Multiplier: $\beta = 0.5$
   - Feature Classes: Linear, Quadratic, Product, Hinge, Threshold
   - Best for high-density GPS telemetry or fine-scale local environmental gradients.

4. **Custom Parameters**:
   - Full manual control over $\beta$, feature classes, background sample points ($1,000 - 10,000$), random seed, and Java heap RAM ($1 - 8\text{ GB}$).

---

## 🖥️ Step-by-Step User Guide

### Step 1: Select Your Data Source
In the left sidebar, choose your data mode:
- **🌟 Pre-loaded Bioclimatic Demo**: Instantly loads 4 aligned GeoTIFFs (DEM Elevation, Distance to Water, Slope, Canopy Cover) and 120 species occurrences.
- **🗺️ Upload Environmental Rasters + Occurrences**: Upload your custom GeoTIFF/ASCII files and occurrence CSV.
- **📊 Upload Point CSV with Predictors**: Upload SWD tabular data (coordinates + extracted environmental columns).

### Step 2: Choose Your Engine & Preset
- Select **Official MaxEnt (maxent.jar)** or **Native Python Engine**.
- Choose a preset or customize parameters in the sidebar.

### Step 3: Run the Model
Click the large blue **🚀 Train Species Distribution Model** button.
- A live progress indicator will track raster alignment and model fitting.
- The **Live Engine Logs** tab displays real-time stdout messages directly from the solver.

### Step 4: Explore the Results Tabs
- **🗺️ Interactive Habitat Suitability Map**:
  - View the predicted suitability heatmap overlaid on satellite imagery.
  - Change colormaps (`Viridis`, `Turbo`, `Plasma`, `YlOrRd`, etc.).
  - Adjust layer opacity from 10% to 100%.
  - Toggle **Binary Habitat (Above Optimal Cutoff)** to highlight suitable vs. unsuitable zones.
  - Click any red marker to view GPS coordinates.
- **📈 Model Diagnostics & ROC**:
  - Interactive ROC curve with AUC metric and optimal decision cutoff.
  - Threshold comparison table (10th percentile, Maximum Sensitivity + Specificity, Equal Sens/Spec).
- **⭐ Variable Importance & Jackknife**:
  - Horizontal bar charts showing which environmental predictors drive the species distribution.
  - Official Jackknife test charts showing gain with each variable alone and without each variable.
- **📉 Environmental Response Curves**:
  - Explore how species probability changes along each environmental gradient.
- **📋 Predictions & Export Suite**:
  - Inspect individual occurrence predictions.
  - Download all results in your preferred GIS or publication format.

---

## 📊 Evaluation Metrics & Diagnostics

The suite reports standard ecological niche validation statistics:

- **Area Under ROC Curve (AUC)**:
  - $0.9 - 1.0$: Excellent discriminatory ability
  - $0.8 - 0.9$: Good model performance
  - $0.7 - 0.8$: Fair / acceptable
  - $0.5$: Random prediction (no predictive power)
- **Regularized Training Gain**: Measures how much more concentrated the distribution is around presence sites compared to a uniform background distribution.
- **Continuous Boyce Index (CBI)**: Range $[-1, 1]$. The gold standard for presence-only evaluation ($>0.7$ indicates high model fidelity).
- **Optimal Decision Cutoffs**:
  - **Maximum Training Sensitivity plus Specificity (Max TSS)**: Minimizes both omission and commission errors.
  - **10 Percentile Training Presence**: Rejects the 10% most extreme/marginal presence observations as potential errors or sink habitats.
  - **Minimum Training Presence**: Conservative threshold including 100% of training occurrences.

---

## 📥 Export & Publication Suite

From the **Predictions & Export Suite** tab, download files with 1 click:

| Button | Format | Purpose |
| :--- | :--- | :--- |
| **📥 GeoTIFF** | `.tif` (EPSG:4326) | Standard GIS raster ready for QGIS, ArcGIS, or GDAL. |
| **📥 ESRI ASCII** | `.asc` | Legacy raster format for traditional spatial packages. |
| **📥 Predictions** | `.csv` | Table of all presence points with predicted suitability scores. |
| **📥 Map Image** | `.png` (300 DPI) | Publication-ready high-resolution suitability figure. |
| **📥 Standalone Map** | `.html` | Self-contained Leaflet web map to open full-screen in any browser. |
| **📦 Full Bundle** | `.zip` | Complete archive containing all rasters, CSVs, lambdas, and figures. |

---

## 🔧 Troubleshooting & FAQ

### Q1: The file uploader shows a limit of 200 MB instead of 2 GB?
Ensure you launched the application using [**`Launch_MaxEnt_Studio.bat`**](file:///c:/Users/DuDu-PC/Downloads/maxent/Launch_MaxEnt_Studio.bat). The `.streamlit/config.toml` file automatically overrides Streamlit's default 200 MB limit to **2048 MB (2 GB)**.

### Q2: I get `java.lang.OutOfMemoryError` on massive rasters?
In the left sidebar, expand **🛠️ Advanced Settings** and increase the **Java Heap Memory** slider to **4096 MB (4 GB)** or **6144 MB (6 GB)**. Ensure you have 64-bit Java installed.

### Q3: Why were some presence records omitted?
If any GPS presence coordinates fall outside the bounding box of your environmental rasters or land in a NoData cell (e.g. water bodies for terrestrial species), the application automatically filters them out and reports the exact count in a warning banner.

### Q4: Can I run this offline without internet access?
Yes! Both the official Java engine and the Python engine run 100% locally on your computer. Web maps cache base tiles or use standard offline canvas rendering.

---

## 👤 Credits & Attribution

- **Concept & Design**: **K S Chandra 2026** *(Source: MaxEnt Java App)*
- **Official MaxEnt Core Algorithm**: Steven J. Phillips, Miroslav Dudík, and Robert E. Schapire (AMNH, AT&T Labs, Princeton University).

---

## 📚 Scientific References

If you use MaxEnt Studio Pro in scientific research or publications, please cite:

1. **Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006)**. *Maximum entropy modeling of species geographic distributions*. Ecological Modelling, 190(3-4), 231-259.
2. **Phillips, S. J., & Dudík, M. (2008)**. *Modeling of species distributions with Maxent: new extensions and a comprehensive evaluation*. Ecography, 31(2), 161-175.
3. **Phillips, S. J., et al. (2017)**. *Opening the black box: an open-source release of Maxent*. Ecography, 40(7), 887-893.
4. **Hirzel, A. H., et al. (2006)**. *Evaluating the ability of habitat suitability models to predict species presences: Continuous Boyce Index*. Ecological Modelling, 199(2), 142-152.
