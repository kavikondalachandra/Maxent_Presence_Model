"""
app.py - MaxEnt Studio Pro: Next-Generation Species Distribution & Ecological Niche Modeling Suite.

Integrates:
- Official AMNH MaxEnt (maxent.jar v3.4.4, Phillips et al.) with automated data prep & modern Java runtime
- Native High-Speed Python MaxEnt Engine (scikit-learn compatible dual L-BFGS solver)
- Comparative Side-by-Side Model Benchmarking
- Automated Multi-layer GIS Raster Alignment (GeoTIFF, ASCII, BIL)
- Interactive Leaflet/Folium GIS Map with Dynamic Colormaps & Esri Satellite/Terrain Basemaps
- Interactive Plotly ROC Curves, Variable Contribution Charts, and Response Curves
- Comprehensive Export Suite (GeoTIFF, ESRI ASCII, 300-DPI PNG, Standalone HTML, ZIP Bundle)
"""

import os
import sys
import site
import io
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure PROJ environment points to rasterio's bundled database to avoid third-party GIS conflicts
for _p in site.getsitepackages():
    _cand = os.path.join(_p, 'rasterio', 'proj_data')
    if os.path.exists(_cand):
        os.environ['PROJ_DATA'] = _cand
        os.environ['PROJ_LIB'] = _cand
        break

# Auto-bootstrap: If executed with bare 'python app.py', automatically relaunch using Streamlit CLI
if __name__ == '__main__':
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if not in_streamlit:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__), "--server.maxUploadSize", "2048"] + sys.argv[1:]
        sys.exit(stcli.main())

# --- STREAMLIT APP CODE ---
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maxent import (
    MaxEnt,
    MaxEntJarRunner,
    find_maxent_jar,
    check_java_runtime,
    generate_synthetic_landscape,
    load_raster_layers,
    extract_values_at_points,
    sample_background_from_rasters,
    create_spatial_grid_from_points,
    export_geotiff,
    export_ascii_grid_bytes,
    create_export_zip_bundle,
    plot_model_diagnostics,
    plot_response_curves,
    plot_spatial_prediction,
    plot_spatial_difference,
    plot_spatial_comparison,
    create_interactive_map,
    export_benchmark_report,
    export_gis_map_report,
    export_roc_diagnostics_report,
    export_variable_importance_report,
    export_response_curves_report,
    export_official_response_curves_report,
    export_occurrence_predictions_report,
    export_execution_logs_report,
    export_comprehensive_report
)
from maxent.plotly_viz import (
    plot_plotly_roc,
    plot_plotly_dual_roc,
    plot_plotly_variable_importance,
    plot_plotly_grouped_variable_importance,
    plot_plotly_response_curve,
    plot_plotly_dual_response_curve
)

st.set_page_config(
    page_title="MaxEnt Studio Pro | Species Distribution Modeling",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #0d9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .badge-container {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-info {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .badge-warning {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .credits-banner {
        font-size: 0.95rem;
        color: #0f766e;
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 6px;
        padding: 0.35rem 0.85rem;
        display: inline-block;
        margin-bottom: 0.9rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def detect_coordinate_columns(columns):
    """
    Smartly detect latitude and longitude column names, handling casing variations,
    Darwin Core / GBIF standards, and GIS abbreviations.
    """
    lat_names = ['latitude', 'lat', 'decimallatitude', 'y', 'coord_y', 'point_y', 'northing']
    lon_names = ['longitude', 'lon', 'long', 'decimallongitude', 'x', 'coord_x', 'point_x', 'easting']

    detected_lat = None
    detected_lon = None

    col_map = {c.lower().strip().replace('_', '').replace(' ', ''): c for c in columns}

    for cand in lat_names:
        clean_cand = cand.replace('_', '')
        if clean_cand in col_map:
            detected_lat = col_map[clean_cand]
            break

    for cand in lon_names:
        clean_cand = cand.replace('_', '')
        if clean_cand in col_map:
            detected_lon = col_map[clean_cand]
            break

    return detected_lat, detected_lon


def safe_render_folium(folium_map, height=540):
    """Render folium map safely using native HTML component."""
    map_html = folium_map._repr_html_()
    components.html(map_html, height=height, scrolling=False)


def safe_render_df(df):
    """Render dataframe safely across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        try:
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.dataframe(df)


def main():
    # --- ENVIRONMENT DETECTION ---
    java_status = check_java_runtime()
    jar_path = find_maxent_jar()

    # Title & Header
    st.markdown('<div class="main-title">🌍 MaxEnt Studio Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Next-Generation Ecological Niche & Species Geographic Distribution Modeling Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="credits-banner">✨ <b>Concept & Design by K S Chandra 2026</b> &nbsp;(Source: MaxEnt Java App)</div>', unsafe_allow_html=True)

    # Badges
    badge_html = '<div class="badge-container">'
    if java_status['available']:
        badge_html += f'<span class="status-badge badge-success">🟢 Java {java_status["version"]} (64-Bit)</span>'
    else:
        badge_html += '<span class="status-badge badge-warning">⚠️ Java Runtime Not Found (Python Engine Active)</span>'

    if jar_path:
        badge_html += f'<span class="status-badge badge-info">📦 Official MaxEnt v3.4.4 (maxent.jar) Linked</span>'
    else:
        badge_html += '<span class="status-badge badge-warning">📦 maxent.jar Not Found (Pure Python Mode)</span>'

    badge_html += '<span class="status-badge badge-success">🗺️ Rasterio / GDAL Aligner Active</span>'
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.markdown(
        "<div style='font-size:0.86rem; color:#0f766e; background:#f0fdfa; border:1px solid #99f6e4; border-radius:6px; padding:7px 10px; margin-bottom:12px; text-align:center; font-weight:600;'>"
        "Concept & Design by K S Chandra 2026<br><span style='font-size:0.75rem; font-weight:normal; color:#475569;'>(Source: MaxEnt Java App)</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.sidebar.header("🎨 Appearance & Theme")
    theme_choice = st.sidebar.selectbox(
        "Application Background Color",
        options=[
            "🌙 Dark Slate (#0f172a)",
            "🌌 Midnight Navy (#0a1128)",
            "🌲 Forest Dark (#06231d)",
            "🌑 Modern Charcoal (#18181b)",
            "☀️ Soft Slate Light (#f8fafc)",
            "⚪ Clean White (#ffffff)",
            "🎨 Custom Background Color..."
        ],
        index=0,
        help="Change the application background color and visual styling."
    )

    if theme_choice == "🎨 Custom Background Color...":
        bg_color = st.sidebar.color_picker("Pick Custom Background Color", value="#111827")
        card_bg = "#1f2937"
        text_color = "#f9fafb"
        border_color = "#374151"
        is_dark = True
    elif "Dark Slate" in theme_choice:
        bg_color = "#0f172a"
        card_bg = "#1e293b"
        text_color = "#f8fafc"
        border_color = "#334155"
        is_dark = True
    elif "Midnight Navy" in theme_choice:
        bg_color = "#0a1128"
        card_bg = "#101f42"
        text_color = "#f1f5f9"
        border_color = "#1e3a8a"
        is_dark = True
    elif "Forest Dark" in theme_choice:
        bg_color = "#06231d"
        card_bg = "#0c3b31"
        text_color = "#ecfdf5"
        border_color = "#14532d"
        is_dark = True
    elif "Modern Charcoal" in theme_choice:
        bg_color = "#18181b"
        card_bg = "#27272a"
        text_color = "#fafafa"
        border_color = "#3f3f46"
        is_dark = True
    elif "Soft Slate Light" in theme_choice:
        bg_color = "#f8fafc"
        card_bg = "#ffffff"
        text_color = "#0f172a"
        border_color = "#e2e8f0"
        is_dark = False
    else: # Clean White
        bg_color = "#ffffff"
        card_bg = "#ffffff"
        text_color = "#1e293b"
        border_color = "#e5e7eb"
        is_dark = False

    plotly_template = "plotly_dark" if is_dark else "plotly_white"

    # Inject dynamic CSS based on chosen background color
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {card_bg} !important;
        }}
        .metric-card {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            color: {text_color} !important;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        }}
        .subtitle {{
            color: {"#94a3b8" if is_dark else "#4b5563"} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {text_color} !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {"#94a3b8" if is_dark else "#4b5563"} !important;
        }}
        div[data-testid="stExpander"] {{
            background-color: {card_bg} !important;
            border-color: {border_color} !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding-top: 8px;
            padding-bottom: 8px;
            color: {"#cbd5e1" if is_dark else "#475569"};
        }}
        .stTabs [aria-selected="true"] {{
            font-weight: 700;
            color: {"#38bdf8" if is_dark else "#0d9488"} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.header("📁 1. Data Source")

    data_source = st.sidebar.radio(
        "Select Data Mode",
        [
            "🌟 Pre-loaded Bioclimatic Demo (1-Click Test Drive)",
            "🗺️ Upload Environmental Rasters (.tif, .asc) + Occurrences (CSV)",
            "📊 Upload Point CSV with Predictors (SWD or Coordinates)",
            "🧪 Simulated Realistic Bioclimatic Landscape"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 2. Modeling Engine")

    # Always expose all engines in the Dual-Engine Architecture
    engine_options = [
        "Official MaxEnt Java Engine (maxent.jar v3.4.4)",
        "Native High-Speed Python MaxEnt Engine",
        "Comparative Side-by-Side (Both Engines)"
    ]

    selected_engine = st.sidebar.selectbox(
        "Choose SDM Engine",
        options=engine_options,
        index=0,
        help="Official MaxEnt (maxent.jar) runs Phillips et al. v3.4.4 Java release in batch mode. Native Python runs the regularized dual L-BFGS solver."
    )

    # Allow custom maxent.jar path if user has jar in custom directory
    if "Official" in selected_engine or "Comparative" in selected_engine:
        if jar_path:
            st.sidebar.success(f"✅ maxent.jar Linked: `{os.path.basename(jar_path)}`")
            with st.sidebar.expander("📍 Location of maxent.jar", expanded=False):
                st.caption(f"Full Path: `{jar_path}`")
                user_jar = st.text_input("Override maxent.jar Path", value="", placeholder="Enter custom path to maxent.jar if needed")
                if user_jar.strip():
                    resolved_jar = find_maxent_jar(user_jar.strip())
                    if resolved_jar:
                        jar_path = resolved_jar
                        st.success(f"Updated: {jar_path}")
                    else:
                        st.error("Specified file not found or is not a valid .jar file.")
        else:
            st.sidebar.warning("⚠️ maxent.jar not automatically detected.")
            user_jar = st.sidebar.text_input("Enter Path to maxent.jar", value="", placeholder=r"G:\...\maxent.jar")
            if user_jar.strip():
                resolved_jar = find_maxent_jar(user_jar.strip())
                if resolved_jar:
                    jar_path = resolved_jar
                    st.sidebar.success(f"✅ maxent.jar Linked: `{os.path.basename(jar_path)}`")

        if java_status['available']:
            st.sidebar.caption(f"🟢 Java {java_status['version']} (64-Bit) active")
        else:
            st.sidebar.error("⚠️ 64-bit Java runtime not found in system PATH. Install Java or switch to the Native Python engine.")

    st.sidebar.markdown("---")
    st.sidebar.header("🏷️ Target Species")
    species_label = st.sidebar.text_input(
        "Target Species Name",
        value="",
        placeholder="e.g., Panthera tigris (leave blank to auto-detect)",
        help="Leave blank to automatically detect species name from your data, or enter a custom name manually."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ 3. Parameters & Presets")

    preset = st.sidebar.selectbox(
        "Ecology Preset",
        [
            "Standard Ecological SDM (Recommended)",
            "Conservative / High Regularization (Prevents Overfitting)",
            "Complex / High-Resolution Niche",
            "Custom Parameters"
        ],
        index=0
    )

    if preset == "Standard Ecological SDM (Recommended)":
        default_beta = 1.0
        default_features = ['linear', 'quadratic', 'product', 'hinge']
    elif preset == "Conservative / High Regularization (Prevents Overfitting)":
        default_beta = 2.5
        default_features = ['linear', 'quadratic', 'hinge']
    elif preset == "Complex / High-Resolution Niche":
        default_beta = 0.5
        default_features = ['linear', 'quadratic', 'product', 'hinge', 'threshold']
    else:
        default_beta = 1.0
        default_features = ['linear', 'quadratic', 'product', 'hinge']

    reg_mult = st.sidebar.slider(
        "Regularization Multiplier (β)",
        min_value=0.1,
        max_value=5.0,
        value=default_beta,
        step=0.1,
        help="Higher values impose heavier L1 penalty, producing smoother response curves and reducing overfitting."
    )

    feature_types = st.sidebar.multiselect(
        "Feature Transformations",
        options=['linear', 'quadratic', 'product', 'hinge', 'threshold'],
        default=default_features,
        help="Feature classes allow capturing non-linear ecological niche responses."
    )

    output_format = st.sidebar.selectbox(
        "Output Transform",
        options=['cloglog', 'logistic', 'cumulative', 'raw'],
        index=0,
        help="Cloglog: Complementary log-log [0, 1] (standard in MaxEnt 3.4+). Logistic: Traditional logistic transform [0, 1]. Cumulative: Percentile rank [0, 100]."
    )

    with st.sidebar.expander("🛠️ Advanced Settings", expanded=False):
        n_bg_samples = st.slider("Background Sample Points", min_value=1000, max_value=10000, value=10000 if "Official" in selected_engine else 3000, step=1000)
        java_mem_mb = st.slider("Java Heap Memory (MB)", min_value=1024, max_value=8192, value=4096, step=512, help="Memory allocated to maxent.jar. 4096MB (4GB) is recommended for large 1-2GB rasters.")
        random_seed = st.checkbox("Random Seed", value=True, help="Use random seed for reproducible sampling")
        run_jackknife = st.checkbox("Jackknife Test of Variable Importance", value=True)
        gen_curves = st.checkbox("Generate Response Curves", value=True)
        esri_api_key = st.text_input(
            "🔑 ArcGIS / Esri API Key (Optional)",
            value="",
            type="password",
            help="Default basemaps (OpenStreetMap, USGS Topo, USGS Satellite, OpenTopoMap) are 100% keyless and immune to 403 Forbidden errors. Enter an ArcGIS API key only if you wish to use authenticated Esri layers."
        )

    # Variables to populate
    presence_df = None
    background_df = None
    raster_dict = None
    extent = None
    features = None
    transform = None
    nodata_mask = None

    # =========================================================================
    # DATA SOURCE 1: PRE-LOADED BIOCLIMATIC DEMO
    # =========================================================================
    if data_source.startswith("🌟"):
        st.info("💡 **Pre-loaded Bioclimatic Demo Dataset Active**: Using Mountain Forest Specialist occurrences with 4 aligned environmental GeoTIFF rasters (Elevation DEM, Distance to Water, Slope Contour, Canopy Cover).")

        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
        tif_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".tif")]

        try:
            with st.spinner("Loading pre-aligned bioclimatic raster layers..."):
                r_info = load_raster_layers(tif_files)
                raster_dict = r_info['raster_dict']
                extent = r_info['extent']
                features = r_info['feature_names']
                transform = r_info['transform']
                nodata_mask = r_info['nodata_mask']
                ref_shape = r_info['shape']

            pres_csv = os.path.join(sample_dir, "sample_species_occurrences_latlon.csv")
            raw_presence = pd.read_csv(pres_csv)
            presence_df, omitted = extract_values_at_points(
                raster_dict=raster_dict,
                transform=transform,
                points_df=raw_presence,
                lat_col='latitude',
                lon_col='longitude',
                nodata_mask=nodata_mask
            )

            background_df = sample_background_from_rasters(
                raster_dict=raster_dict,
                transform=transform,
                nodata_mask=nodata_mask,
                n_samples=min(n_bg_samples, 5000),
                random_state=42
            )

        except Exception as e:
            st.error(f"Error loading demo data: {e}")
            st.stop()

    # =========================================================================
    # DATA SOURCE 2: UPLOAD RASTERS + OCCURRENCES
    # =========================================================================
    elif data_source.startswith("🗺️"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗺️ Upload Rasters")
        raster_files = st.sidebar.file_uploader(
            "Upload Environmental Rasters (.tif, .asc, .bil)",
            type=['tif', 'tiff', 'asc', 'bil'],
            accept_multiple_files=True,
            help="Upload GIS rasters representing environmental predictors (Elevation DEM, climate, land cover, etc.)."
        )

        st.sidebar.subheader("📍 Upload Occurrences")
        pres_file = st.sidebar.file_uploader(
            "Upload Species Occurrences (CSV)",
            type=['csv'],
            help="CSV containing species GPS occurrence locations (latitude & longitude)."
        )

        if not raster_files or pres_file is None:
            st.info("👈 **Please upload your Environmental Rasters and Occurrence CSV in the sidebar to begin.**")
            st.markdown("""
            ### 📌 How MaxEnt Studio Pro Works:
            1. **Upload Rasters**: Upload any GIS raster format (GeoTIFF, ASCII, BIL). The app automatically aligns, resamples, and verifies matching spatial extents!
            2. **Upload Occurrences**: Upload a CSV with latitude and longitude coordinates. Column names are auto-detected.
            3. **Automatic Extraction**: Environmental predictor values are automatically sampled at each GPS occurrence site.
            4. **Dual Engine Execution**: Run the official AMNH MaxEnt Java binary (`maxent.jar`) or native Python engine with 1 click!
            """)
            st.stop()

        try:
            with st.spinner("Aligning and validating environmental raster layers..."):
                r_info = load_raster_layers(raster_files)
                raster_dict = r_info['raster_dict']
                extent = r_info['extent']
                features = r_info['feature_names']
                transform = r_info['transform']
                nodata_mask = r_info['nodata_mask']
                ref_shape = r_info['shape']

            raw_presence = pd.read_csv(pres_file)
            det_lat, det_lon = detect_coordinate_columns(list(raw_presence.columns))
            cols = list(raw_presence.columns)

            lat_idx = cols.index(det_lat) if det_lat in cols else 0
            lon_idx = cols.index(det_lon) if det_lon in cols else 0

            st.sidebar.markdown("---")
            sel_lat = st.sidebar.selectbox("Latitude Column", options=cols, index=lat_idx)
            sel_lon = st.sidebar.selectbox("Longitude Column", options=cols, index=lon_idx)

            raw_presence['latitude'] = pd.to_numeric(raw_presence[sel_lat], errors='coerce')
            raw_presence['longitude'] = pd.to_numeric(raw_presence[sel_lon], errors='coerce')

            presence_df, omitted = extract_values_at_points(
                raster_dict=raster_dict,
                transform=transform,
                points_df=raw_presence,
                lat_col='latitude',
                lon_col='longitude',
                nodata_mask=nodata_mask
            )

            if len(presence_df) == 0:
                st.error("❌ No presence points fell within the spatial bounding box of the rasters. Please verify coordinate systems (WGS84 EPSG:4326 expected).")
                st.stop()

            if omitted > 0:
                st.warning(f"⚠️ {omitted} presence records were outside raster boundaries or on NoData cells and were omitted.")

            background_df = sample_background_from_rasters(
                raster_dict=raster_dict,
                transform=transform,
                nodata_mask=nodata_mask,
                n_samples=min(n_bg_samples, 5000),
                random_state=42
            )

        except Exception as e:
            st.error(f"Error processing files: {e}")
            st.stop()

    # =========================================================================
    # DATA SOURCE 3: POINT CSV WITH PREDICTORS
    # =========================================================================
    elif data_source.startswith("📊"):
        pres_file = st.sidebar.file_uploader("Upload Presence CSV", type=['csv'])
        bg_file = st.sidebar.file_uploader("Upload Background CSV (Optional)", type=['csv'])

        if pres_file is None:
            st.info("👈 Upload a CSV file containing coordinates and environmental predictor columns.")
            st.stop()

        raw_presence = pd.read_csv(pres_file)
        background_df = pd.read_csv(bg_file) if bg_file is not None else None

        det_lat, det_lon = detect_coordinate_columns(list(raw_presence.columns))
        all_cols = list(raw_presence.columns)

        lat_options = ["(None)"] + all_cols
        lon_options = ["(None)"] + all_cols

        lat_idx = lat_options.index(det_lat) if det_lat in lat_options else 0
        lon_idx = lon_options.index(det_lon) if det_lon in lon_options else 0

        st.sidebar.markdown("---")
        sel_lat = st.sidebar.selectbox("Latitude Column", options=lat_options, index=lat_idx)
        sel_lon = st.sidebar.selectbox("Longitude Column", options=lon_options, index=lon_idx)

        has_coords = False
        if sel_lat != "(None)" and sel_lon != "(None)":
            raw_presence['latitude'] = pd.to_numeric(raw_presence[sel_lat], errors='coerce')
            raw_presence['longitude'] = pd.to_numeric(raw_presence[sel_lon], errors='coerce')
            has_coords = True

        exclude = {'latitude', 'longitude', 'species', 'id', sel_lat.lower(), sel_lon.lower()}
        avail_preds = [c for c in raw_presence.columns if c.lower() not in exclude and pd.api.types.is_numeric_dtype(raw_presence[c])]

        features = st.sidebar.multiselect("Select Predictor Columns", options=avail_preds, default=avail_preds)
        if not features:
            st.error("Please select at least one numeric predictor variable.")
            st.stop()

        presence_df = raw_presence

        if has_coords:
            with st.spinner("Interpolating environmental spatial layers across study bounding box..."):
                raster_dict, extent = create_spatial_grid_from_points(
                    df=presence_df,
                    features=features,
                    lat_col='latitude',
                    lon_col='longitude',
                    grid_res=80,
                    buffer_ratio=0.08
                )

    # =========================================================================
    # DATA SOURCE 4: SIMULATED BIOCLIMATIC LANDSCAPE
    # =========================================================================
    else:
        sim_data = generate_synthetic_landscape(width=100, height=100, n_presence=150, n_background=2500, seed=42)
        presence_df = sim_data['presence_df']
        background_df = sim_data['background_df']
        raster_dict = sim_data['raster_dict']
        features = sim_data['feature_names']
        extent = sim_data['extent']

    # --- TOP DATA SUMMARY BANNER ---
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("📍 Presence Records", len(presence_df) if presence_df is not None else 0)
    col_s2.metric("🌿 Predictor Variables", len(features) if features else 0)
    col_s3.metric("📐 Grid Shape", f"{raster_dict[features[0]].shape[0]} × {raster_dict[features[0]].shape[1]}" if raster_dict else "Point SWD")
    col_s4.metric("⚙️ Selected Engine", "Java maxent.jar" if "Official" in selected_engine else ("Comparison" if "Comparative" in selected_engine else "Python Engine"))

    st.markdown("---")

    # --- EXECUTION BUTTON ---
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run_model = st.button("🚀 Train Species Distribution Model", type="primary", use_container_width=True)

    # Store results in session_state so user can interact without retraining
    if run_model or "last_results" not in st.session_state:
        if not run_model and "last_results" not in st.session_state:
            # First load auto-run
            pass

        results_data = {}
        console_logs = []

        def log_stream(msg):
            console_logs.append(msg)

        # Resolve target species name cleanly (handles blank manual entry)
        active_species_name = species_label.strip() if species_label and species_label.strip() else ""
        if not active_species_name and presence_df is not None:
            for cand_col in ['species', 'taxon', 'scientificname', 'scientific_name', 'sp']:
                matched_col = next((c for c in presence_df.columns if c.lower().replace('_', '') == cand_col), None)
                if matched_col and not presence_df[matched_col].dropna().empty:
                    active_species_name = str(presence_df[matched_col].dropna().iloc[0]).strip()
                    break
        if not active_species_name:
            active_species_name = "Target_Species"

        # -------------------------------------------------------------
        # RUN 1: OFFICIAL MAXENT JAVA ENGINE
        # -------------------------------------------------------------
        if "Official" in selected_engine or "Comparative" in selected_engine:
            if not jar_path or not java_status['available']:
                st.error("Official MaxEnt Java Engine requires 64-bit Java and maxent.jar. Please verify Java is installed or specify the maxent.jar path in the sidebar.")
                st.stop()

            with st.spinner("Running Official MaxEnt (AMNH v3.4.4 Java Engine)..."):
                runner = MaxEntJarRunner(jar_path=jar_path, memory_mb=java_mem_mb)
                java_res = runner.run(
                    raster_dict=raster_dict,
                    extent=extent,
                    presence_df=presence_df,
                    lat_col='latitude',
                    lon_col='longitude',
                    species_name=active_species_name,
                    output_format=output_format,
                    reg_mult=reg_mult,
                    feature_types=feature_types,
                    jackknife=run_jackknife,
                    response_curves=gen_curves,
                    random_seed=random_seed,
                    background_points=n_bg_samples,
                    log_callback=log_stream
                )
                results_data['java'] = java_res

        # -------------------------------------------------------------
        # RUN 2: NATIVE PYTHON MAXENT ENGINE
        # -------------------------------------------------------------
        if "Native" in selected_engine or "Comparative" in selected_engine:
            with st.spinner("Fitting Native Python MaxEnt Model (Regularized Dual Solver)..."):
                X_p = presence_df[features].dropna()
                X_b = background_df[features].dropna() if background_df is not None else None

                t0 = time.time()
                py_model = MaxEnt(
                    reg_mult=reg_mult,
                    feature_types=feature_types,
                    hinge_knots=5,
                    default_output=output_format,
                    random_state=42
                )
                py_model.fit(X_p, X_b, feature_names=features)
                elapsed_py = time.time() - t0

                # Spatial prediction
                suit_grid = None
                if raster_dict is not None:
                    suit_grid = py_model.predict_raster(raster_dict, output_type=output_format)

                # Feature importance
                imp_df = py_model.get_feature_importance(X_p, X_b if X_b is not None else X_p)

                # Predictions
                p_copy = presence_df.copy()
                p_copy['Predicted_Suitability'] = py_model.predict(X_p, output_type=output_format)

                # ROC Curve points
                suit_p = py_model.predict(X_p, output_type='cloglog')
                suit_b = py_model.predict(X_b if X_b is not None else X_p, output_type='cloglog')
                y_true = np.concatenate([np.ones_like(suit_p), np.zeros_like(suit_b)])
                y_pred = np.concatenate([suit_p, suit_b])
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)

                # Generate Python response curves
                py_resp_data = {}
                for v in features:
                    try:
                        xv, yv = py_model.get_response_curve(v)
                        py_resp_data[v] = pd.DataFrame({'x': xv, 'y': yv})
                    except Exception:
                        pass

                results_data['python'] = {
                    "engine": "Native Python MaxEnt Engine",
                    "model": py_model,
                    "suitability_grid": suit_grid,
                    "extent": extent,
                    "metrics": py_model.training_metrics_,
                    "variable_importance": imp_df,
                    "sample_predictions": p_copy,
                    "roc_data": {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                        "auc": py_model.training_metrics_.get('auc', 0.5)
                    },
                    "response_curve_data": py_resp_data,
                    "elapsed_seconds": elapsed_py,
                    "log": f"Trained scikit-learn compatible MaxEnt with {len(py_model.weights_)} features in {elapsed_py:.2f}s."
                }

        results_data['active_engine'] = "java" if "Official" in selected_engine else ("python" if "Native" in selected_engine else "comparison")
        results_data['console_logs'] = console_logs
        st.session_state['last_results'] = results_data

    # Retrieve current active results
    res_bundle = st.session_state.get('last_results', {})
    if not res_bundle:
        st.stop()

    has_java = 'java' in res_bundle
    has_python = 'python' in res_bundle
    is_comparison = has_java and has_python

    res_java = res_bundle.get('java', {})
    res_py = res_bundle.get('python', {})

    # Compute comparative analytics if both engines were run
    comparison_info = {}
    if is_comparison:
        grid_j = res_java.get('suitability_grid')
        grid_p = res_py.get('suitability_grid')
        diff_grid = None
        sp_r = np.nan
        pe_r = np.nan
        mean_diff = np.nan
        max_diff = np.nan

        if grid_j is not None and grid_p is not None:
            valid_m = (~np.isnan(grid_j)) & (~np.isnan(grid_p))
            if np.any(valid_m):
                from scipy.stats import spearmanr, pearsonr
                v_j = grid_j[valid_m].ravel()
                v_p = grid_p[valid_m].ravel()
                sp_r, _ = spearmanr(v_j, v_p)
                pe_r, _ = pearsonr(v_j, v_p)
                diff_grid = np.abs(grid_j - grid_p)
                mean_diff = float(np.nanmean(diff_grid))
                max_diff = float(np.nanmax(diff_grid))

        t_j = float(res_java.get('elapsed_seconds', 0.0))
        t_p = float(res_py.get('elapsed_seconds', 0.0))
        speedup = (t_j / max(t_p, 1e-4)) if t_p > 0 else 1.0

        auc_j = float(res_java.get('metrics', {}).get('auc', 0.0))
        auc_p = float(res_py.get('metrics', {}).get('auc', 0.0))
        gain_j = float(res_java.get('metrics', {}).get('training_gain', 0.0))
        gain_p = float(res_py.get('metrics', {}).get('training_gain', 0.0))
        cbi_j = float(res_java.get('metrics', {}).get('boyce_index', np.nan))
        cbi_p = float(res_py.get('metrics', {}).get('boyce_index', np.nan))
        th_j = float(res_java.get('metrics', {}).get('threshold', 0.5))
        th_p = float(res_py.get('metrics', {}).get('threshold', 0.5))

        comparison_info = {
            "grid_java": grid_j,
            "grid_python": grid_p,
            "diff_grid": diff_grid,
            "spearman_r": sp_r,
            "pearson_r": pe_r,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "speedup": speedup,
            "t_java": t_j,
            "t_py": t_p,
            "auc_java": auc_j,
            "auc_py": auc_p,
            "gain_java": gain_j,
            "gain_py": gain_p,
            "cbi_java": cbi_j,
            "cbi_py": cbi_p,
            "threshold_java": th_j,
            "threshold_py": th_p
        }

    # View Mode Selector
    if is_comparison:
        st.markdown("### ⚖️ Multi-Engine Result Inspector")
        col_v1, col_v2 = st.columns([2.3, 1.2])
        with col_v1:
            view_mode = st.radio(
                "Active Results View:",
                options=[
                    "⚖️ Comparative Benchmark (Side-by-Side)",
                    "🟢 Official MaxEnt Java Results",
                    "🔵 Native Python MaxEnt Results"
                ],
                horizontal=True,
                index=0,
                help="Switch between comparative side-by-side benchmarking and individual engine outputs."
            )
        with col_v2:
            st.info(f"⚡ Python Engine: **{comparison_info['speedup']:.1f}× faster** | Concordance $r_s$: **{comparison_info['spearman_r']:.3f}**")
    else:
        view_mode = "🟢 Official MaxEnt Java Results" if has_java else "🔵 Native Python MaxEnt Results"

    # Set active engine bundle for single-engine tabs or baseline
    if view_mode.startswith("🔵"):
        curr_res = res_py
    elif view_mode.startswith("🟢"):
        curr_res = res_java
    else:
        curr_res = res_java if has_java else res_py

    suitability_grid = curr_res['suitability_grid']
    grid_extent = curr_res['extent']
    metrics = curr_res['metrics']
    var_importance = curr_res['variable_importance']
    sample_preds = curr_res['sample_predictions']
    roc_info = curr_res.get('roc_data')

    # Pre-render matplotlib raster figure for display and export
    png_bytes = None
    fig_map = None
    if suitability_grid is not None:
        fig_map = plot_spatial_prediction(
            suitability_grid,
            extent=grid_extent,
            presence_coords=presence_df,
            threshold=metrics.get('threshold', 0.5),
            title=f"Suitability Raster ({output_format.upper()})"
        )
        buf = io.BytesIO()
        fig_map.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        png_bytes = buf.getvalue()

    # Prepare centralized reporting bundle for PDF & MS Word exports
    clean_sp_name = species_label.strip() if (species_label and species_label.strip()) else "Target_Species"
    clean_sp_filename = clean_sp_name.replace(" ", "_").lower()

    report_data = {
        'species_name': clean_sp_name,
        'active_engine': res_bundle.get('active_engine', 'java'),
        'view_mode': view_mode,
        'is_comparison': is_comparison,
        'comparison_info': comparison_info if is_comparison else {},
        'res_java': res_java,
        'res_py': res_py,
        'curr_res': curr_res,
        'metrics': metrics,
        'extent': grid_extent,
        'presence_df': presence_df,
        'background_df': background_df,
        'raster_dict': raster_dict,
        'var_importance': var_importance,
        'sample_preds': sample_preds,
        'roc_info': roc_info,
        'output_format': output_format,
        'reg_mult': reg_mult,
        'feature_types': feature_types,
        'n_bg_samples': n_bg_samples,
        'java_status': java_status,
        'jar_path': jar_path,
        'plots': (res_java.get('plots') if (res_java and 'plots' in res_java) else curr_res.get('plots', {})),
        'console_logs': res_bundle.get('console_logs', '')
    }

    # --- TOP METRIC CARDS ---
    if view_mode.startswith("⚖️"):
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        m_col1.metric(
            "Java AUC vs Python",
            f"{comparison_info['auc_java']:.3f} | {comparison_info['auc_py']:.3f}",
            delta=f"{comparison_info['auc_py'] - comparison_info['auc_java']:+.3f} Δ",
            help="Area Under ROC Curve: Official Java vs Native Python"
        )
        m_col2.metric(
            "Spatial Concordance (rs)",
            f"{comparison_info['spearman_r']:.3f}" if not np.isnan(comparison_info['spearman_r']) else "N/A",
            help="Spearman rank correlation across all cells (>0.9 indicates strong spatial agreement)"
        )
        m_col3.metric(
            "Python Speedup",
            f"{comparison_info['speedup']:.1f}× Faster",
            delta=f"-{comparison_info['t_java'] - comparison_info['t_py']:.2f}s",
            help=f"Java runtime: {comparison_info['t_java']:.2f}s vs Python runtime: {comparison_info['t_py']:.2f}s"
        )
        m_col4.metric(
            "Mean Discrepancy",
            f"±{comparison_info['mean_diff']:.3f}" if not np.isnan(comparison_info['mean_diff']) else "N/A",
            help="Mean absolute error across all prediction pixels (|Java - Python|)"
        )
        m_col5.metric(
            "Pearson Correlation (r)",
            f"{comparison_info['pearson_r']:.3f}" if not np.isnan(comparison_info['pearson_r']) else "N/A",
            help="Linear correlation coefficient between suitability rasters"
        )
        m_col6.metric(
            "Optimal Cutoffs",
            f"{comparison_info['threshold_java']:.3f} | {comparison_info['threshold_py']:.3f}",
            help="Decision thresholds maximizing sensitivity + specificity (Java | Python)"
        )
    else:
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        m_col1.metric("ROC AUC", f"{metrics.get('auc', 0.0):.3f}", help="Area Under ROC Curve (>0.7 indicates useful model)")
        m_col2.metric("Regularized Gain", f"{metrics.get('training_gain', 0.0):.3f}", help="Log-likelihood improvement over uniform distribution")
        if "boyce_index" in metrics and not np.isnan(metrics.get('boyce_index', np.nan)):
            m_col3.metric("Continuous Boyce", f"{metrics.get('boyce_index', 0.0):.3f}", help="Boyce index [-1, 1] for presence-only validation")
        else:
            m_col3.metric("Iterations", f"{metrics.get('iterations', 0)}")
        m_col4.metric("Optimal Cutoff", f"{metrics.get('threshold', 0.5):.3f}", help="Decision threshold maximizing sensitivity + specificity")
        m_col5.metric("Training Samples", f"{metrics.get('training_samples', len(presence_df))}")
        m_col6.metric("Compute Time", f"{curr_res.get('elapsed_seconds', 0.0):.2f}s")

    # --- MAIN TABS ---
    if view_mode.startswith("⚖️"):
        tab_comp, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "⚖️ Comparative Benchmark Summary & Dual Maps",
            "🗺️ Interactive Leaflet GIS Map",
            "📈 Dual ROC Curves & Diagnostics",
            "⭐ Variable Importance Comparison",
            "📉 Response Curves Comparison",
            "📋 Occurrence Predictions & Difference Table",
            "🖥️ Multi-Engine Execution Logs & Details"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Interactive Habitat Suitability Map",
            "📈 Model Diagnostics & ROC",
            "⭐ Variable Importance & Jackknife",
            "📉 Environmental Response Curves",
            "📋 Predictions & Export Suite",
            "🖥️ Live Engine Logs & Details"
        ])

    # =========================================================================
    # TAB 0 (COMPARATIVE ONLY): BENCHMARK SUMMARY & DUAL MAPS
    # =========================================================================
    if view_mode.startswith("⚖️"):
        with tab_comp:
            st.subheader("⚖️ Empirical Benchmark: Official AMNH MaxEnt (Java) vs. Native Python Engine")
            st.markdown(
                "Both modeling engines were trained using the exact same environmental rasters, occurrence records, "
                "feature classes, regularization multiplier, and background points. Below is the full side-by-side empirical evaluation."
            )

            # 1. Comparative Metrics Table
            bench_records = [
                {
                    "Evaluation Metric": "Algorithm / Implementation",
                    "Official MaxEnt (maxent.jar v3.4.4)": "Phillips et al. Exact Java Batch Binary",
                    "Native Python MaxEnt Engine": "Scikit-Learn Compatible Dual L-BFGS Solver",
                    "Concordance / Difference": "Identical Regularized MaxEnt Objective"
                },
                {
                    "Evaluation Metric": "Execution Compute Time",
                    "Official MaxEnt (maxent.jar v3.4.4)": f"{comparison_info['t_java']:.3f} s",
                    "Native Python MaxEnt Engine": f"{comparison_info['t_py']:.3f} s",
                    "Concordance / Difference": f"⚡ Python is {comparison_info['speedup']:.1f}× Faster"
                },
                {
                    "Evaluation Metric": "Training ROC AUC",
                    "Official MaxEnt (maxent.jar v3.4.4)": f"{comparison_info['auc_java']:.4f}",
                    "Native Python MaxEnt Engine": f"{comparison_info['auc_py']:.4f}",
                    "Concordance / Difference": f"Δ AUC = {comparison_info['auc_py'] - comparison_info['auc_java']:+.4f}"
                },
                {
                    "Evaluation Metric": "Regularized Training Gain",
                    "Official MaxEnt (maxent.jar v3.4.4)": f"{comparison_info['gain_java']:.4f}",
                    "Native Python MaxEnt Engine": f"{comparison_info['gain_py']:.4f}",
                    "Concordance / Difference": f"Δ Gain = {comparison_info['gain_py'] - comparison_info['gain_java']:+.4f}"
                },
                {
                    "Evaluation Metric": "Optimal Cutoff Threshold (Max TSS)",
                    "Official MaxEnt (maxent.jar v3.4.4)": f"{comparison_info['threshold_java']:.4f}",
                    "Native Python MaxEnt Engine": f"{comparison_info['threshold_py']:.4f}",
                    "Concordance / Difference": f"Δ Cutoff = {comparison_info['threshold_py'] - comparison_info['threshold_java']:+.4f}"
                },
                {
                    "Evaluation Metric": "Continuous Boyce Index (CBI)",
                    "Official MaxEnt (maxent.jar v3.4.4)": f"{comparison_info['cbi_java']:.4f}" if not np.isnan(comparison_info['cbi_java']) else "N/A",
                    "Native Python MaxEnt Engine": f"{comparison_info['cbi_py']:.4f}" if not np.isnan(comparison_info['cbi_py']) else "N/A",
                    "Concordance / Difference": "Presence-only validation fidelity"
                },
                {
                    "Evaluation Metric": "Spatial Spearman Rank Correlation (rs)",
                    "Official MaxEnt (maxent.jar v3.4.4)": "Baseline (1.000)",
                    "Native Python MaxEnt Engine": f"{comparison_info['spearman_r']:.4f}" if not np.isnan(comparison_info['spearman_r']) else "N/A",
                    "Concordance / Difference": "✅ Excellent Spatial Concordance" if comparison_info['spearman_r'] >= 0.90 else "Good Concordance"
                },
                {
                    "Evaluation Metric": "Spatial Pearson Correlation (r)",
                    "Official MaxEnt (maxent.jar v3.4.4)": "Baseline (1.000)",
                    "Native Python MaxEnt Engine": f"{comparison_info['pearson_r']:.4f}" if not np.isnan(comparison_info['pearson_r']) else "N/A",
                    "Concordance / Difference": f"r = {comparison_info['pearson_r']:.4f}" if not np.isnan(comparison_info['pearson_r']) else "N/A"
                },
                {
                    "Evaluation Metric": "Mean Absolute Pixel Discrepancy",
                    "Official MaxEnt (maxent.jar v3.4.4)": "-",
                    "Native Python MaxEnt Engine": f"{comparison_info['mean_diff']:.4f}" if not np.isnan(comparison_info['mean_diff']) else "N/A",
                    "Concordance / Difference": f"Max Discrepancy: {comparison_info['max_diff']:.4f}" if not np.isnan(comparison_info['max_diff']) else "N/A"
                }
            ]
            safe_render_df(pd.DataFrame(bench_records))

            # 2. 3-Panel Spatial Concordance
            if comparison_info['grid_java'] is not None and comparison_info['grid_python'] is not None:
                st.markdown("---")
                st.markdown("#### 🗺️ 3-Panel Spatial Concordance: Java vs. Python vs. Absolute Difference")
                fig_comp = plot_spatial_comparison(
                    grid_java=comparison_info['grid_java'],
                    grid_py=comparison_info['grid_python'],
                    extent=grid_extent
                )
                st.pyplot(fig_comp)
                plt.close(fig_comp)

            # --- Export Benchmark Summary & Maps (PDF & MS Word) ---
            st.markdown("---")
            b_exp_c1, b_exp_c2, b_exp_c3 = st.columns([1.8, 1, 1])
            with b_exp_c1:
                st.markdown("##### 📥 Export Benchmark Summary & Maps")
                st.caption("Publication-ready report with empirical metrics, solver comparison, and 3-panel spatial concordance map.")
            with b_exp_c2:
                bench_pdf = export_benchmark_report(report_data, doc_type="pdf")
                st.download_button(
                    "📄 Download PDF",
                    data=bench_pdf,
                    file_name=f"{clean_sp_filename}_benchmark_summary_and_maps.pdf",
                    mime="application/pdf",
                    help="Download Benchmark Summary & Maps as publication-ready PDF",
                    key="dl_tab_bench_pdf"
                )
            with b_exp_c3:
                bench_docx = export_benchmark_report(report_data, doc_type="docx")
                st.download_button(
                    "📝 Download Word (.docx)",
                    data=bench_docx,
                    file_name=f"{clean_sp_filename}_benchmark_summary_and_maps.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    help="Download Benchmark Summary & Maps as editable MS Word document",
                    key="dl_tab_bench_docx"
                )

    # =========================================================================
    # TAB 1: INTERACTIVE GIS MAP
    # =========================================================================
    with tab1:
        st.subheader("Geographic Occurrence & Predicted Habitat Suitability")

        # Map controls
        c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4, c_ctrl5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2])
        with c_ctrl1:
            cmap_name = st.selectbox("Suitability Color Palette", ["viridis", "turbo", "plasma", "magma", "YlOrRd", "Spectral_r", "RdYlGn_r"], index=0)
        with c_ctrl2:
            map_opacity = st.slider("Heatmap Layer Opacity", min_value=0.1, max_value=1.0, value=0.65, step=0.05)
        with c_ctrl3:
            show_binary = st.checkbox("Show Binary Habitat (Above Cutoff)", value=False)
        with c_ctrl4:
            bg_pt_color = st.color_picker("Background Samples Color", value="#94a3b8" if is_dark else "#555555", help="Customize the color of background / landscape points on the map")
        with c_ctrl5:
            show_bg_pts = st.checkbox("Show Background Sample Sites", value=False)

        with st.expander("🗺️ Basemap Settings & ArcGIS API Key (No Key Required for Default)", expanded=False):
            st.info("ℹ️ **Zero API key required for standard viewing!** All default basemaps (OpenStreetMap, USGS Topographic, USGS Satellite, OpenTopoMap) are open, free, and immune to HTTP 403 Forbidden errors. If you have an ArcGIS Developer token and prefer authenticated Esri layers, enter it below:")
            tab_esri_key = st.text_input("Optional ArcGIS / Esri Developer Token", value=esri_api_key if 'esri_api_key' in locals() else "", type="password", key="tab1_map_key")
            if tab_esri_key.strip():
                esri_api_key = tab_esri_key.strip()

        # In comparative mode, offer map mode selector
        if is_comparison:
            map_mode = st.radio(
                "GIS Map Display Layout:",
                ["Dual-Layer Interactive Map (Java + Python Switchable)", "Side-by-Side Synchronized Web Maps (Java | Python)", "Spatial Difference Overlay (|Java - Python|)"],
                horizontal=True
            )
        else:
            map_mode = "Single"

        display_grid = suitability_grid.copy() if suitability_grid is not None else None
        if show_binary and display_grid is not None:
            cutoff_val = metrics.get('threshold', 0.5)
            display_grid = np.where(display_grid >= cutoff_val, 1.0, 0.0)

        active_esri_token = esri_api_key.strip() if ('esri_api_key' in locals() and esri_api_key and esri_api_key.strip()) else None

        if is_comparison and map_mode == "Side-by-Side Synchronized Web Maps (Java | Python)":
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("**🟢 Official MaxEnt Java Suitability**")
                map_j = create_interactive_map(
                    presence_df=presence_df,
                    background_df=background_df,
                    suitability_grid=res_java['suitability_grid'],
                    extent=grid_extent,
                    colormap=cmap_name,
                    opacity=map_opacity,
                    grid_title="Official MaxEnt (Java)",
                    bg_point_color=bg_pt_color,
                    show_bg_points=show_bg_pts,
                    api_key=active_esri_token
                )
                safe_render_folium(map_j, height=540)
            with col_m2:
                st.markdown("**🔵 Native Python Engine Suitability**")
                map_p = create_interactive_map(
                    presence_df=presence_df,
                    background_df=background_df,
                    suitability_grid=res_py['suitability_grid'],
                    extent=grid_extent,
                    colormap=cmap_name,
                    opacity=map_opacity,
                    grid_title="Native Python Engine",
                    bg_point_color=bg_pt_color,
                    show_bg_points=show_bg_pts,
                    api_key=active_esri_token
                )
                safe_render_folium(map_p, height=540)
        else:
            with st.spinner("Generating interactive GIS layers..."):
                if is_comparison:
                    folium_map = create_interactive_map(
                        presence_df=presence_df,
                        background_df=background_df,
                        suitability_grid=res_java['suitability_grid'],
                        extent=grid_extent,
                        colormap=cmap_name,
                        opacity=map_opacity,
                        grid_title="🟢 Official MaxEnt Java Suitability",
                        secondary_grid=res_py['suitability_grid'],
                        secondary_name="🔵 Native Python Engine Suitability",
                        diff_grid=comparison_info.get('diff_grid'),
                        diff_name="Spatial Discrepancy (|Java - Python|)",
                        bg_point_color=bg_pt_color,
                        show_bg_points=show_bg_pts,
                        api_key=active_esri_token
                    )
                else:
                    folium_map = create_interactive_map(
                        presence_df=presence_df,
                        background_df=background_df,
                        suitability_grid=display_grid,
                        extent=grid_extent,
                        colormap=cmap_name,
                        opacity=map_opacity,
                        bg_point_color=bg_pt_color,
                        show_bg_points=show_bg_pts,
                        api_key=active_esri_token
                    )

            st.markdown("**Interactive GIS Web Map (Keyless OpenStreetMap & USGS Basemaps &bull; Multi-Layer Viewer)**")
            safe_render_folium(folium_map, height=620)
            if fig_map is not None:
                with st.expander("🖼️ View 2D Spatial Raster Grid & Bounding Coordinates (Matplotlib)", expanded=False):
                    st.pyplot(fig_map)
                    plt.close(fig_map)

        # --- Export Interactive Leaflet GIS Map (PDF & MS Word) ---
        st.markdown("---")
        m_exp_c1, m_exp_c2, m_exp_c3 = st.columns([1.8, 1, 1])
        with m_exp_c1:
            st.markdown("##### 📥 Export Interactive Leaflet GIS Map")
            st.caption("Publication-ready report with geographic bounding coordinates, raster cell dimensions, and 2D spatial suitability figure.")
        with m_exp_c2:
            map_pdf = export_gis_map_report(report_data, doc_type="pdf")
            st.download_button(
                "📄 Download PDF",
                data=map_pdf,
                file_name=f"{clean_sp_filename}_interactive_gis_map_report.pdf",
                mime="application/pdf",
                help="Download Interactive Leaflet GIS Map report as publication-ready PDF",
                key="dl_tab_map_pdf"
            )
        with m_exp_c3:
            map_docx = export_gis_map_report(report_data, doc_type="docx")
            st.download_button(
                "📝 Download Word (.docx)",
                data=map_docx,
                file_name=f"{clean_sp_filename}_interactive_gis_map_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download Interactive Leaflet GIS Map report as editable MS Word document",
                key="dl_tab_map_docx"
            )

    # =========================================================================
    # TAB 2: DIAGNOSTICS & ROC
    # =========================================================================
    with tab2:
        st.subheader("Statistical Validation & Decision Thresholds")
        d_col1, d_col2 = st.columns([1.3, 1])

        with d_col1:
            if is_comparison and res_java.get('roc_data') and res_py.get('roc_data'):
                fig_roc = plot_plotly_dual_roc(
                    fpr_java=res_java['roc_data']['fpr'],
                    tpr_java=res_java['roc_data']['tpr'],
                    auc_java=comparison_info['auc_java'],
                    fpr_py=res_py['roc_data']['fpr'],
                    tpr_py=res_py['roc_data']['tpr'],
                    auc_py=comparison_info['auc_py'],
                    thresholds_java=res_java['roc_data'].get('thresholds'),
                    thresholds_py=res_py['roc_data'].get('thresholds'),
                    opt_th_java=comparison_info['threshold_java'],
                    opt_th_py=comparison_info['threshold_py'],
                    template=plotly_template
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            elif roc_info is not None and len(roc_info['fpr']) > 0:
                fig_roc = plot_plotly_roc(
                    fpr=roc_info['fpr'],
                    tpr=roc_info['tpr'],
                    auc_val=roc_info['auc'],
                    thresholds=roc_info.get('thresholds'),
                    opt_threshold=metrics.get('threshold'),
                    template=plotly_template
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC Curve data not available.")

        with d_col2:
            st.markdown("#### 🎯 Decision Cutoff Thresholds")
            if is_comparison:
                th_j_dict = res_java.get('thresholds', {})
                th_records = []
                for k, v in th_j_dict.items():
                    th_records.append({
                        "Criterion": k,
                        "Official Java Cutoff": f"{v:.4f}"
                    })
                th_records.append({
                    "Criterion": "Python Optimal Cutoff (Max TSS)",
                    "Official Java Cutoff": f"{comparison_info['threshold_py']:.4f}"
                })
                safe_render_df(pd.DataFrame(th_records))
            else:
                thresholds_dict = curr_res.get('thresholds', {})
                if thresholds_dict:
                    th_df = pd.DataFrame([
                        {"Criterion": k, "Threshold": f"{v:.4f}"}
                        for k, v in thresholds_dict.items()
                    ])
                    safe_render_df(th_df)
                else:
                    st.write(f"**Optimal Cutoff**: `{metrics.get('threshold', 0.5):.3f}`")

            # Official ROC PNG if generated by Java
            if "plots" in curr_res and any("roc" in p for p in curr_res["plots"]):
                for k, p in curr_res["plots"].items():
                    if "roc" in k:
                        st.image(p, caption="Official MaxEnt ROC Chart", use_container_width=True)

        # --- Export ROC Curves & Diagnostics (PDF & MS Word) ---
        st.markdown("---")
        r_exp_c1, r_exp_c2, r_exp_c3 = st.columns([1.8, 1, 1])
        with r_exp_c1:
            st.markdown("##### 📥 Export ROC Curves & Diagnostics")
            st.caption("Publication-ready report with ROC curves, Area Under Curve (AUC), and decision cutoff threshold tables.")
        with r_exp_c2:
            roc_pdf = export_roc_diagnostics_report(report_data, doc_type="pdf")
            st.download_button(
                "📄 Download PDF",
                data=roc_pdf,
                file_name=f"{clean_sp_filename}_roc_curves_and_diagnostics.pdf",
                mime="application/pdf",
                help="Download ROC Curves & Diagnostics as publication-ready PDF",
                key="dl_tab_roc_pdf"
            )
        with r_exp_c3:
            roc_docx = export_roc_diagnostics_report(report_data, doc_type="docx")
            st.download_button(
                "📝 Download Word (.docx)",
                data=roc_docx,
                file_name=f"{clean_sp_filename}_roc_curves_and_diagnostics.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download ROC Curves & Diagnostics as editable MS Word document",
                key="dl_tab_roc_docx"
            )

    # =========================================================================
    # TAB 3: VARIABLE IMPORTANCE & JACKKNIFE
    # =========================================================================
    with tab3:
        st.subheader("Environmental Predictor Contributions & Jackknife Test")
        v_col1, v_col2 = st.columns([1.3, 1])

        with v_col1:
            if is_comparison:
                fig_imp = plot_plotly_grouped_variable_importance(
                    imp_java=res_java.get('variable_importance'),
                    imp_py=res_py.get('variable_importance'),
                    template=plotly_template
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            elif not var_importance.empty:
                fig_imp = plot_plotly_variable_importance(var_importance, template=plotly_template)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Variable importance calculation in progress.")

        with v_col2:
            st.markdown("#### 📊 Variable Contributions Table")
            if is_comparison:
                df_j = res_java.get('variable_importance', pd.DataFrame())
                df_p = res_py.get('variable_importance', pd.DataFrame())
                if not df_j.empty and not df_p.empty:
                    merged_imp = pd.merge(
                        df_j[['Variable', 'Percent_Contribution']].rename(columns={'Percent_Contribution': 'Java_Contribution (%)'}),
                        df_p[['Variable', 'Percent_Contribution']].rename(columns={'Percent_Contribution': 'Python_Contribution (%)'}),
                        on='Variable',
                        how='outer'
                    ).fillna(0.0)
                    merged_imp['Difference (%)'] = (merged_imp['Java_Contribution (%)'] - merged_imp['Python_Contribution (%)']).abs().round(2)
                    safe_render_df(merged_imp)
                else:
                    safe_render_df(var_importance)
            else:
                safe_render_df(var_importance)

            # Show official Jackknife chart if available
            ref_res = res_java if is_comparison else curr_res
            if "plots" in ref_res and any("jacknife" in p for p in ref_res["plots"]):
                for k, p in ref_res["plots"].items():
                    if "jacknife" in k:
                        st.image(p, caption="Official Jackknife Test of Variable Importance", use_container_width=True)

        # --- Export Variable Importance (PDF & MS Word) ---
        st.markdown("---")
        v_exp_c1, v_exp_c2, v_exp_c3 = st.columns([1.8, 1, 1])
        with v_exp_c1:
            st.markdown("##### 📥 Export Variable Importance")
            st.caption("Publication-ready report with relative percent contributions, permutation importance, and Jackknife tests.")
        with v_exp_c2:
            vi_pdf = export_variable_importance_report(report_data, doc_type="pdf")
            st.download_button(
                "📄 Download PDF",
                data=vi_pdf,
                file_name=f"{clean_sp_filename}_variable_importance.pdf",
                mime="application/pdf",
                help="Download Variable Importance as publication-ready PDF",
                key="dl_tab_vi_pdf"
            )
        with v_exp_c3:
            vi_docx = export_variable_importance_report(report_data, doc_type="docx")
            st.download_button(
                "📝 Download Word (.docx)",
                data=vi_docx,
                file_name=f"{clean_sp_filename}_variable_importance.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download Variable Importance as editable MS Word document",
                key="dl_tab_vi_docx"
            )

    # =========================================================================
    # TAB 4: RESPONSE CURVES
    # =========================================================================
    with tab4:
        st.subheader("Marginal Environmental Response Curves")
        st.markdown("Illustrates how predicted suitability changes as each environmental variable varies across its gradient, holding other variables constant.")

        resp_j = res_java.get('response_curve_data', {}) if is_comparison else curr_res.get('response_curve_data', {})
        resp_p = res_py.get('response_curve_data', {}) if is_comparison else {}

        avail_vars = [k for k in (resp_j or resp_p or {}).keys() if not k.endswith("_only")]
        if not avail_vars and features:
            avail_vars = list(features)

        if avail_vars:
            sel_var = st.selectbox("Select Environmental Predictor", avail_vars)

            if is_comparison and sel_var in resp_j and sel_var in resp_p:
                df_j = resp_j[sel_var]
                df_p = resp_p[sel_var]
                fig_rc = plot_plotly_dual_response_curve(
                    var_name=sel_var,
                    x_vals_java=df_j['x'].values,
                    y_vals_java=df_j['y'].values,
                    x_vals_py=df_p['x'].values,
                    y_vals_py=df_p['y'].values,
                    y_label=f"Suitability ({output_format.upper()})",
                    template=plotly_template
                )
                st.plotly_chart(fig_rc, use_container_width=True)
            elif sel_var in resp_j:
                df_var = resp_j[sel_var]
                fig_rc = plot_plotly_response_curve(
                    var_name=sel_var,
                    x_vals=df_var['x'].values,
                    y_vals=df_var['y'].values,
                    y_label=f"Suitability ({output_format.upper()})",
                    template=plotly_template
                )
                st.plotly_chart(fig_rc, use_container_width=True)
            elif "model" in curr_res:
                fig_resp = plot_response_curves(curr_res["model"])
                st.pyplot(fig_resp)
                plt.close(fig_resp)

        # Also show gallery of official MaxEnt response curve PNGs
        ref_res = res_java if is_comparison else curr_res
        if "plots" in ref_res:
            st.markdown("---")
            st.markdown("#### 🖼️ Official MaxEnt Response Curve Figures")
            png_plots = [v for k, v in ref_res["plots"].items() if not any(x in k for x in ["roc", "omission", "jacknife", "overview"])]
            if png_plots:
                cols = st.columns(min(3, len(png_plots)))
                for idx, ppath in enumerate(png_plots):
                    with cols[idx % len(cols)]:
                        st.image(ppath, caption=os.path.basename(ppath), use_container_width=True)

                # --- Export Official MaxEnt Response Curve Figures (PDF & MS Word) ---
                orc_exp_c1, orc_exp_c2, orc_exp_c3 = st.columns([1.8, 1, 1])
                with orc_exp_c1:
                    st.markdown("##### 📥 Export Official MaxEnt Response Curve Figures")
                    st.caption("Publication-ready document containing the complete gallery of official AMNH MaxEnt marginal and standalone response curve figures.")
                with orc_exp_c2:
                    orc_pdf = export_official_response_curves_report(report_data, doc_type="pdf")
                    st.download_button(
                        "📄 Download Official Figures (PDF)",
                        data=orc_pdf,
                        file_name=f"{clean_sp_filename}_official_response_curves.pdf",
                        mime="application/pdf",
                        help="Download Official MaxEnt Response Curve Figures as publication-ready PDF",
                        key="dl_tab_orc_pdf"
                    )
                with orc_exp_c3:
                    orc_docx = export_official_response_curves_report(report_data, doc_type="docx")
                    st.download_button(
                        "📝 Download Official Figures (Word)",
                        data=orc_docx,
                        file_name=f"{clean_sp_filename}_official_response_curves.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        help="Download Official MaxEnt Response Curve Figures as editable MS Word document",
                        key="dl_tab_orc_docx"
                    )

        # --- Export Response Curves (PDF & MS Word) ---
        st.markdown("---")
        rc_exp_c1, rc_exp_c2, rc_exp_c3 = st.columns([1.8, 1, 1])
        with rc_exp_c1:
            st.markdown("##### 📥 Export Environmental Response Curves")
            st.caption("Publication-ready report with marginal response curve plots across environmental gradients.")
        with rc_exp_c2:
            rc_pdf = export_response_curves_report(report_data, doc_type="pdf")
            st.download_button(
                "📄 Download PDF",
                data=rc_pdf,
                file_name=f"{clean_sp_filename}_response_curves.pdf",
                mime="application/pdf",
                help="Download Response Curves as publication-ready PDF",
                key="dl_tab_rc_pdf"
            )
        with rc_exp_c3:
            rc_docx = export_response_curves_report(report_data, doc_type="docx")
            st.download_button(
                "📝 Download Word (.docx)",
                data=rc_docx,
                file_name=f"{clean_sp_filename}_response_curves.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download Response Curves as editable MS Word document",
                key="dl_tab_rc_docx"
            )

    # =========================================================================
    # TAB 5: PREDICTIONS & EXPORT SUITE
    # =========================================================================
    with tab5:
        st.subheader("Occurrence Predictions & Export Suite")

        if is_comparison:
            p_j = res_java.get('sample_predictions', pd.DataFrame()).copy()
            p_p = res_py.get('sample_predictions', pd.DataFrame()).copy()
            comp_table = p_j.copy()
            if 'Predicted_Suitability' in p_p.columns:
                comp_table['Python_Suitability'] = p_p['Predicted_Suitability'].round(4)
            if 'Predicted_Suitability' in comp_table.columns:
                comp_table['Java_Suitability'] = comp_table['Predicted_Suitability'].round(4)
                comp_table = comp_table.drop(columns=['Predicted_Suitability'])
            if 'Java_Suitability' in comp_table.columns and 'Python_Suitability' in comp_table.columns:
                comp_table['Absolute_Difference'] = (comp_table['Java_Suitability'] - comp_table['Python_Suitability']).abs().round(4)
                comp_table = comp_table.sort_values('Absolute_Difference', ascending=False)
            safe_render_df(comp_table.head(100))
        else:
            safe_render_df(sample_preds.head(100))

        # --- Export Occurrence Predictions (PDF & MS Word) ---
        st.markdown("---")
        o_exp_c1, o_exp_c2, o_exp_c3 = st.columns([1.8, 1, 1])
        with o_exp_c1:
            st.markdown("##### 📥 Export Occurrence Predictions")
            st.caption("Publication-ready report with site-specific suitability values, threshold status, and occurrence histograms.")
        with o_exp_c2:
            occ_pdf = export_occurrence_predictions_report(report_data, doc_type="pdf")
            st.download_button(
                "📄 Download PDF",
                data=occ_pdf,
                file_name=f"{clean_sp_filename}_occurrence_predictions.pdf",
                mime="application/pdf",
                help="Download Occurrence Predictions as publication-ready PDF",
                key="dl_tab_occ_pdf"
            )
        with o_exp_c3:
            occ_docx = export_occurrence_predictions_report(report_data, doc_type="docx")
            st.download_button(
                "📝 Download Word (.docx)",
                data=occ_docx,
                file_name=f"{clean_sp_filename}_occurrence_predictions.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download Occurrence Predictions as editable MS Word document",
                key="dl_tab_occ_docx"
            )

        st.markdown("---")
        st.subheader("📥 Export Habitat Suitability Models & Publications")
        e_col1, e_col2, e_col3, e_col4, e_col5, e_col6 = st.columns(6)

        with e_col1:
            try:
                gtiff_bytes = export_geotiff(suitability_grid, grid_extent)
                st.download_button(
                    "📥 GeoTIFF (.tif)",
                    data=gtiff_bytes,
                    file_name="maxent_suitability_raster.tif",
                    mime="image/tiff",
                    help="Standard GIS GeoTIFF ready for QGIS, ArcGIS, or GDAL."
                )
            except Exception as e:
                st.caption(f"GeoTIFF: {e}")

        with e_col2:
            if is_comparison and comparison_info.get('diff_grid') is not None:
                try:
                    diff_tif_bytes = export_geotiff(comparison_info['diff_grid'], grid_extent)
                    st.download_button(
                        "📥 Difference GeoTIFF",
                        data=diff_tif_bytes,
                        file_name="spatial_difference_raster.tif",
                        mime="image/tiff",
                        help="GeoTIFF raster of absolute difference |Java - Python|."
                    )
                except Exception as e:
                    st.caption(f"Diff GeoTIFF: {e}")
            else:
                asc_bytes = export_ascii_grid_bytes(suitability_grid, grid_extent)
                st.download_button(
                    "📥 ESRI ASCII (.asc)",
                    data=asc_bytes,
                    file_name="maxent_suitability_prediction.asc",
                    mime="text/plain",
                    help="ESRI ASCII Grid compatible with all legacy and modern GIS tools."
                )

        with e_col3:
            pred_export_df = comp_table if is_comparison else sample_preds
            csv_bytes = pred_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Predictions (.csv)",
                data=csv_bytes,
                file_name="occurrence_predictions.csv",
                mime="text/csv",
                help="Table of occurrence records with predicted suitability values."
            )

        with e_col4:
            if png_bytes is not None:
                st.download_button(
                    "📥 Map Image (.png)",
                    data=png_bytes,
                    file_name="spatial_suitability_map.png",
                    mime="image/png",
                    help="300-DPI publication-ready habitat suitability figure."
                )

        with e_col5:
            if folium_map is not None:
                try:
                    html_bytes = folium_map.get_root().render().encode('utf-8')
                    st.download_button(
                        "🌐 Web Map (.html)",
                        data=html_bytes,
                        file_name="interactive_habitat_map.html",
                        mime="text/html",
                        help="Standalone Leaflet/Folium web map with keyless basemaps and interactive layers, ready to open in any web browser."
                    )
                except Exception as e:
                    st.caption(f"Web Map: {e}")

        with e_col6:
            # Full ZIP Bundle
            with st.spinner("Compiling full ZIP package..."):
                html_map_str = folium_map._repr_html_()
                lambdas_p = curr_res.get('lambdas_path')
                extra_f = {}
                if is_comparison:
                    try:
                        extra_f["python_suitability.tif"] = export_geotiff(res_py['suitability_grid'], grid_extent)
                        if comparison_info.get('diff_grid') is not None:
                            extra_f["spatial_difference.tif"] = export_geotiff(comparison_info['diff_grid'], grid_extent)
                        extra_f["comparative_predictions.csv"] = comp_table.to_csv(index=False).encode('utf-8')
                    except Exception:
                        pass

                # Include PDF and Word comprehensive reports inside the ZIP bundle
                try:
                    extra_f["maxent_comprehensive_report.pdf"] = export_comprehensive_report(report_data, doc_type="pdf")
                    extra_f["maxent_comprehensive_report.docx"] = export_comprehensive_report(report_data, doc_type="docx")
                    extra_f["official_maxent_response_curves.pdf"] = export_official_response_curves_report(report_data, doc_type="pdf")
                    extra_f["official_maxent_response_curves.docx"] = export_official_response_curves_report(report_data, doc_type="docx")
                except Exception:
                    pass

                zip_bytes = create_export_zip_bundle(
                    suitability_grid=suitability_grid,
                    extent=grid_extent,
                    predictions_df=sample_preds,
                    metrics=metrics,
                    html_map_str=html_map_str,
                    png_map_bytes=png_bytes,
                    lambdas_path=lambdas_p,
                    extra_files=extra_f if extra_f else None
                )
                st.download_button(
                    "📦 Full Bundle (.zip)",
                    data=zip_bytes,
                    file_name="maxent_complete_results.zip",
                    mime="application/zip",
                    help="Complete package containing GeoTIFF, ASCII grid, CSVs, map, PDF & Word reports, and metadata."
                )

        # --- Dedicated Publication-Grade Reports Center (PDF & MS Word) ---
        st.markdown("---")
        st.subheader("📑 Publication-Grade Comprehensive SDM Reports (PDF & MS Word)")
        st.markdown(
            "Generate and download comprehensive, publication-ready MaxEnt scientific reports with high-resolution figures, "
            "styled tables, statistical validations, and full academic citations."
        )

        rep_card1, rep_card2 = st.columns(2)
        with rep_card1:
            st.markdown("""
            <div style='background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 14px; margin-bottom: 10px;'>
                <div style='font-weight: 700; color: #1e3a8a; font-size: 1.05rem; margin-bottom: 4px;'>
                    📕 Comprehensive MaxEnt Model Report (PDF)
                </div>
                <div style='color: #475569; font-size: 0.85rem; margin-bottom: 10px;'>
                    Complete publication-grade document combining all 7 core analytical sections: Benchmark Summary & Maps, Interactive Leaflet GIS Map, ROC Curves & Diagnostics, Variable Importance, Response Curves, Occurrence Predictions, and Execution Logs.
                </div>
            </div>
            """, unsafe_allow_html=True)
            comp_pdf_bytes = export_comprehensive_report(report_data, doc_type="pdf")
            st.download_button(
                "📥 Download Complete Report (PDF)",
                data=comp_pdf_bytes,
                file_name=f"{clean_sp_filename}_maxent_comprehensive_report.pdf",
                mime="application/pdf",
                help="Download complete publication-grade PDF report combining all 7 sections",
                key="btn_comp_pdf_tab5"
            )

        with rep_card2:
            st.markdown("""
            <div style='background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 8px; padding: 14px; margin-bottom: 10px;'>
                <div style='font-weight: 700; color: #0f766e; font-size: 1.05rem; margin-bottom: 4px;'>
                    📘 Comprehensive MaxEnt Model Report (MS Word .docx)
                </div>
                <div style='color: #475569; font-size: 0.85rem; margin-bottom: 10px;'>
                    Fully formatted, editable Microsoft Word document with native styled tables, high-resolution figures, callout panels, running headers/footers, and references.
                </div>
            </div>
            """, unsafe_allow_html=True)
            comp_docx_bytes = export_comprehensive_report(report_data, doc_type="docx")
            st.download_button(
                "📥 Download Complete Report (Word .docx)",
                data=comp_docx_bytes,
                file_name=f"{clean_sp_filename}_maxent_comprehensive_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download complete editable MS Word (.docx) report combining all 7 sections",
                key="btn_comp_docx_tab5"
            )

        with st.expander("📂 Download Individual Section Reports in PDF & MS Word", expanded=False):
            sec_t1, sec_t2, sec_t3, sec_t4, sec_t5, sec_t6, sec_t7 = st.tabs([
                "⚖️ Benchmark Summary & Maps",
                "🗺️ Interactive Leaflet GIS Map",
                "📈 ROC Curves & Diagnostics",
                "⭐ Variable Importance",
                "📉 Response Curves",
                "📋 Occurrence Predictions",
                "🖥️ Execution Logs & Details"
            ])
            with sec_t1:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📄 Benchmark Summary & Maps (PDF)", data=export_benchmark_report(report_data, "pdf"), file_name=f"{clean_sp_filename}_benchmark_summary_and_maps.pdf", mime="application/pdf", key="exp_hub_pdf_bench")
                with c2:
                    st.download_button("📝 Benchmark Summary & Maps (Word)", data=export_benchmark_report(report_data, "docx"), file_name=f"{clean_sp_filename}_benchmark_summary_and_maps.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="exp_hub_docx_bench")
            with sec_t2:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📄 GIS Map & Spatial Extent (PDF)", data=export_gis_map_report(report_data, "pdf"), file_name=f"{clean_sp_filename}_gis_map_report.pdf", mime="application/pdf", key="exp_hub_pdf_map")
                with c2:
                    st.download_button("📝 GIS Map & Spatial Extent (Word)", data=export_gis_map_report(report_data, "docx"), file_name=f"{clean_sp_filename}_gis_map_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="exp_hub_docx_map")
            with sec_t3:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📄 ROC Curves & Diagnostics (PDF)", data=export_roc_diagnostics_report(report_data, "pdf"), file_name=f"{clean_sp_filename}_roc_curves_and_diagnostics.pdf", mime="application/pdf", key="exp_hub_pdf_roc")
                with c2:
                    st.download_button("📝 ROC Curves & Diagnostics (Word)", data=export_roc_diagnostics_report(report_data, "docx"), file_name=f"{clean_sp_filename}_roc_curves_and_diagnostics.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="exp_hub_docx_roc")
            with sec_t4:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📄 Variable Importance (PDF)", data=export_variable_importance_report(report_data, "pdf"), file_name=f"{clean_sp_filename}_variable_importance.pdf", mime="application/pdf", key="exp_hub_pdf_vi")
                with c2:
                    st.download_button("📝 Variable Importance (Word)", data=export_variable_importance_report(report_data, "docx"), file_name=f"{clean_sp_filename}_variable_importance.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="exp_hub_docx_vi")
            with sec_t5:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📄 Marginal Response Curves (PDF)", data=export_response_curves_report(report_data, "pdf"), file_name=f"{clean_sp_filename}_response_curves.pdf", mime="application/pdf", key="exp_hub_pdf_rc")
                    st.download_button("🖼️ Official MaxEnt Figures (PDF)", data=export_official_response_curves_report(report_data, "pdf"), file_name=f"{clean_sp_filename}_official_response_curves.pdf", mime="application/pdf", key="exp_hub_pdf_orc")
                with c2:
                    st.download_button("📝 Marginal Response Curves (Word)", data=export_response_curves_report(report_data, "docx"), file_name=f"{clean_sp_filename}_response_curves.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="exp_hub_docx_rc")
                    st.download_button("🖼️ Official MaxEnt Figures (Word)", data=export_official_response_curves_report(report_data, "docx"), file_name=f"{clean_sp_filename}_official_response_curves.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="exp_hub_docx_orc")
            with sec_t6:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📄 Occurrence Predictions (PDF)", data=export_occurrence_predictions_report(report_data, "pdf"), file_name=f"{clean_sp_filename}_occurrence_predictions.pdf", mime="application/pdf", key="exp_hub_pdf_occ")
                with c2:
                    st.download_button("📝 Occurrence Predictions (Word)", data=export_occurrence_predictions_report(report_data, "docx"), file_name=f"{clean_sp_filename}_occurrence_predictions.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="exp_hub_docx_occ")
            with sec_t7:
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📄 Execution Logs & Details (PDF)", data=export_execution_logs_report(report_data, "pdf"), file_name=f"{clean_sp_filename}_execution_logs_and_details.pdf", mime="application/pdf", key="exp_hub_pdf_log")
                with c2:
                    st.download_button("📝 Execution Logs & Details (Word)", data=export_execution_logs_report(report_data, "docx"), file_name=f"{clean_sp_filename}_execution_logs_and_details.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="exp_hub_docx_log")

    # =========================================================================
    # TAB 6: LIVE ENGINE LOGS & DETAILS
    # =========================================================================
    with tab6:
        st.subheader("Engine Execution Console & System Architecture")

        if is_comparison:
            log_t1, log_t2 = st.tabs(["🟢 Official MaxEnt Java Logs", "🔵 Native Python Engine Logs"])
            with log_t1:
                st.markdown(f"**Execution Time**: `{comparison_info['t_java']:.2f}s` | **Binary**: `maxent.jar v3.4.4`")
                if "log" in res_java and res_java["log"]:
                    st.code(res_java["log"], language="text")
                else:
                    st.info("Java engine output captured in session.")
            with log_t2:
                st.markdown(f"**Execution Time**: `{comparison_info['t_py']:.2f}s` | **Solver**: Dual L-BFGS")
                if "log" in res_py and res_py["log"]:
                    st.code(res_py["log"], language="text")
                else:
                    st.info("Python engine output captured in session.")
        else:
            st.markdown(f"**Selected Engine**: `{curr_res.get('engine', 'MaxEnt')}` | **Elapsed Time**: `{curr_res.get('elapsed_seconds', 0.0):.2f}s`")
            if "log" in curr_res and curr_res["log"]:
                st.code(curr_res["log"], language="text")

        st.markdown("#### 🗺️ Environmental Layer Metadata")
        if raster_dict:
            layer_records = []
            for k, arr in raster_dict.items():
                layer_records.append({
                    "Layer Name": k,
                    "Shape": f"{arr.shape[0]} × {arr.shape[1]}",
                    "Min Value": f"{np.nanmin(arr):.2f}",
                    "Mean Value": f"{np.nanmean(arr):.2f}",
                    "Max Value": f"{np.nanmax(arr):.2f}",
                    "Valid Cells (%)": f"{100.0 * np.sum(~np.isnan(arr)) / arr.size:.1f}%"
                })
            safe_render_df(pd.DataFrame(layer_records))

        # --- Export Execution Logs & Details (PDF & MS Word) ---
        st.markdown("---")
        l_exp_c1, l_exp_c2, l_exp_c3 = st.columns([1.8, 1, 1])
        with l_exp_c1:
            st.markdown("##### 📥 Export Execution Logs & Details")
            st.caption("Publication-ready report with software environment specs, hyperparameters, raster metadata, and engine console traces.")
        with l_exp_c2:
            log_pdf = export_execution_logs_report(report_data, doc_type="pdf")
            st.download_button(
                "📄 Download PDF",
                data=log_pdf,
                file_name=f"{clean_sp_filename}_execution_logs_and_details.pdf",
                mime="application/pdf",
                help="Download Execution Logs & System Details as publication-ready PDF",
                key="dl_tab_log_pdf"
            )
        with l_exp_c3:
            log_docx = export_execution_logs_report(report_data, doc_type="docx")
            st.download_button(
                "📝 Download Word (.docx)",
                data=log_docx,
                file_name=f"{clean_sp_filename}_execution_logs_and_details.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download Execution Logs & System Details as editable MS Word document",
                key="dl_tab_log_docx"
            )

    # =========================================================================
    # CITATION, REFERENCES & DATA SOURCES
    # =========================================================================
    st.markdown("---")
    with st.expander("📚 Citation, References & Data Sources", expanded=True):
        st.markdown("### 📖 How to Cite MaxEnt Studio Pro")
        st.markdown(
            "If you use **MaxEnt Studio Pro** for research, academic papers, ecological assessments, "
            "or geospatial modeling, please cite the software and the foundational literature:"
        )

        cite_tab1, cite_tab2, cite_tab3, cite_tab4 = st.tabs([
            "📋 Software Citation",
            "🧬 Scientific Foundations",
            "🗺️ Data & Spatial Sources",
            "💻 Computational Stack & Licenses"
        ])

        with cite_tab1:
            st.markdown("""
            **APA Format:**
            > Chandra, K. S. (2026). *MaxEnt Studio Pro: Modern Species Distribution & Ecological Niche Modeling Suite* (Version 1.0.0) [Computer software]. GitHub. https://github.com/kschandra6-Research/maxent_presence_model

            **BibTeX Entry:**
            ```bibtex
            @software{chandra2026maxent,
              author       = {Chandra, K. S.},
              title        = {{MaxEnt Studio Pro: Modern Species Distribution & Ecological Niche Modeling Suite}},
              year         = {2026},
              version      = {1.0.0},
              publisher    = {GitHub},
              url          = {https://github.com/kschandra6-Research/maxent_presence_model},
              keywords     = {maximum-entropy, species-distribution-modelling, continuous-boyce-index, spatial-ecology, geotiff}
            }
            ```
            """)

        with cite_tab2:
            st.markdown("""
            #### Primary Algorithmic Literature
            - **Canonical Maximum Entropy SDM:**  
              Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006). Maximum entropy modeling of species geographic distributions. *Ecological Modelling*, 190(3-4), 231–259. [doi:10.1016/j.ecolmodel.2005.03.026](https://doi.org/10.1016/j.ecolmodel.2005.03.026)
            - **Modeling Extensions & Evaluation:**  
              Phillips, S. J., & Dudík, M. (2008). Modeling of species distributions with Maxent: new extensions and a comprehensive evaluation. *Ecography*, 31(2), 161–175. [doi:10.1111/j.0906-7590.2008.5203.x](https://doi.org/10.1111/j.0906-7590.2008.5203.x)
            - **MaxEnt Open-Source Java Release:**  
              Phillips, S. J., Anderson, R. P., Dudík, M., Schapire, R. E., & Blair, M. E. (2017). Opening the black box: an open-source release of Maxent. *Ecography*, 40(7), 887–893. [doi:10.1111/ecog.03049](https://doi.org/10.1111/ecog.03049)
            - **Statistical Foundations for Ecologists:**  
              Elith, J., Phillips, S. J., Hastie, T., Dudík, M., Chee, Y. E., & Yates, C. J. (2011). A statistical explanation of MaxEnt for ecologists. *Diversity and Distributions*, 17(1), 43–57. [doi:10.1111/j.1472-4642.2010.00725.x](https://doi.org/10.1111/j.1472-4642.2010.00725.x)
            - **Continuous Boyce Index (CBI):**  
              Hirzel, A. H., Le Lay, G., Helfer, V., Randin, C., & Guisan, A. (2006). Evaluating the ability of habitat suitability models to predict species presences: The Continuous Boyce Index. *Ecological Modelling*, 199(2), 142–152. [doi:10.1016/j.ecolmodel.2006.05.017](https://doi.org/10.1016/j.ecolmodel.2006.05.017)
            - **Critique of Presence-Only ROC AUC:**  
              Lobo, J. M., Jiménez-Valverde, A., & Real, R. (2008). AUC: a misleading measure of the performance of predictive distribution models. *Global Ecology and Biogeography*, 17(2), 145–151. [doi:10.1111/j.1466-8238.2007.00358.x](https://doi.org/10.1111/j.1466-8238.2007.00358.x)
            - **Predictive Archaeological & Geographic Modeling:**  
              Conolly, J., & Lake, M. (2006). *Geographical Information Systems in Archaeology*. Cambridge University Press.  
              Verhagen, P., & Whitley, T. G. (2012). Integrating archaeological theory and predictive modelling. *Journal of Archaeological Method and Theory*, 19(1), 49–100.
            """)

        with cite_tab3:
            st.markdown("""
            #### Spatial Data & Basemap Tile Attributions
            - **WorldClim 2 Bioclimatic Rasters:**  
              Fick, S. E., & Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37(12), 4302–4315. [https://www.worldclim.org/](https://www.worldclim.org/)
            - **USGS 3D Elevation Program (3DEP):**  
              U.S. Geological Survey (USGS). Digital Elevation Models, Hydrography, and Shaded Relief. [https://www.usgs.gov/3dep](https://www.usgs.gov/3dep)
            - **GBIF (Global Biodiversity Information Facility):**  
              GBIF: The Global Biodiversity Information Facility. Occurrence records and museum specimens. [https://www.gbif.org/](https://www.gbif.org/)
            - **OpenStreetMap & OpenTopoMap Tiles (Keyless, Public Access):**  
              OpenStreetMap map tiles © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors. OpenTopoMap contours and hillshading © [OpenTopoMap](https://opentopomap.org). High-reliability tile delivery with zero API token restrictions.
            - **USGS The National Map (Public Domain):**  
              USGS National Boundaries, Topography, and Orthoimagery MapServer tiles (U.S. Department of the Interior).
            - **Esri ArcGIS Online Tile Services (Optional Authenticated):**  
              Esri World Imagery & Topographic base layers © Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, METI. Requires user API token for authenticated queries.
            """)

        with cite_tab4:
            st.markdown("""
            #### Core Computational Libraries & Licenses
            - **Python 3.10+**: PSF License.
            - **ReportLab**: Enterprise publication-grade PDF document rendering engine (BSD License).
            - **python-docx**: Microsoft Word OpenXML (.docx) programmatic authoring library (MIT License).
            - **Rasterio & GDAL**: Affine coordinate transformations, in-memory resampling, and GeoTIFF I/O (BSD License).
            - **SciPy**: L-BFGS-B convex optimization and dual entropy solver (BSD License).
            - **Scikit-Learn**: Machine learning utilities, cross-validation, and ROC metric evaluation (BSD License).
            - **Folium & Leaflet.js**: Client-side interactive web GIS mapping (MIT / BSD-2-Clause).
            - **Plotly & Matplotlib**: High-DPI scientific figures, interactive response curves, and diagnostic dashboards.
            - **Streamlit**: Reactive real-time web application architecture (Apache 2.0).
            - **Official MaxEnt (maxent.jar v3.4.4)**: American Museum of Natural History (AMNH) & Phillips et al. (MIT License).
            """)

    # --- FOOTER CREDITS ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #475569; font-size: 0.9rem; padding: 1rem 0; font-weight: 500;'>"
        "🌍 <b>MaxEnt Studio Pro</b> &nbsp;|&nbsp; <b>Concept & Design by K S Chandra 2026</b> &nbsp;<i>(Source: MaxEnt Java App)</i>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
