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
    create_interactive_map
)
from maxent.plotly_viz import (
    plot_plotly_roc,
    plot_plotly_variable_importance,
    plot_plotly_response_curve
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

    active_key = 'java' if 'java' in res_bundle else 'python'
    curr_res = res_bundle[active_key]

    suitability_grid = curr_res['suitability_grid']
    grid_extent = curr_res['extent']
    metrics = curr_res['metrics']
    var_importance = curr_res['variable_importance']
    sample_preds = curr_res['sample_predictions']
    roc_info = curr_res.get('roc_data')

    # --- TOP METRIC CARDS ---
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("ROC AUC", f"{metrics.get('auc', 0.0):.3f}", help="Area Under ROC Curve (>0.7 indicates useful model)")
    m_col2.metric("Regularized Gain", f"{metrics.get('training_gain', 0.0):.3f}", help="Log-likelihood improvement over uniform distribution")
    if "boyce_index" in metrics:
        m_col3.metric("Continuous Boyce", f"{metrics.get('boyce_index', 0.0):.3f}", help="Boyce index [-1, 1] for presence-only validation")
    else:
        m_col3.metric("Iterations", f"{metrics.get('iterations', 0)}")
    m_col4.metric("Optimal Cutoff", f"{metrics.get('threshold', 0.5):.3f}", help="Decision threshold maximizing sensitivity + specificity")
    m_col5.metric("Training Samples", f"{metrics.get('training_samples', len(presence_df))}")
    m_col6.metric("Compute Time", f"{curr_res.get('elapsed_seconds', 0.0):.2f}s")

    # --- MAIN TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Interactive Habitat Suitability Map",
        "📈 Model Diagnostics & ROC",
        "⭐ Variable Importance & Jackknife",
        "📉 Environmental Response Curves",
        "📋 Predictions & Export Suite",
        "🖥️ Live Engine Logs & Details"
    ])

    # =========================================================================
    # TAB 1: INTERACTIVE GIS MAP
    # =========================================================================
    with tab1:
        st.subheader("Geographic Occurrence & Predicted Habitat Suitability")

        # Map controls
        c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([1.1, 1.1, 1.1, 1.4])
        with c_ctrl1:
            cmap_name = st.selectbox("Suitability Color Palette", ["viridis", "turbo", "plasma", "YlOrRd", "Spectral_r", "RdYlGn_r"], index=0)
        with c_ctrl2:
            map_opacity = st.slider("Heatmap Layer Opacity", min_value=0.1, max_value=1.0, value=0.65, step=0.05)
        with c_ctrl3:
            show_binary = st.checkbox("Show Binary Habitat (Above Cutoff)", value=False)
        with c_ctrl4:
            map_layout = st.radio("Display Layout", ["Full-Width Web Map (Recommended)", "Side-by-Side (Web Map + 2D Raster)"], horizontal=True)

        display_grid = suitability_grid.copy() if suitability_grid is not None else None
        if show_binary and display_grid is not None:
            cutoff_val = metrics.get('threshold', 0.5)
            display_grid = np.where(display_grid >= cutoff_val, 1.0, 0.0)

        with st.spinner("Generating interactive GIS layers..."):
            folium_map = create_interactive_map(
                presence_df=presence_df,
                background_df=background_df,
                suitability_grid=display_grid,
                extent=grid_extent,
                colormap=cmap_name,
                opacity=map_opacity
            )

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

        if map_layout.startswith("Full-Width"):
            st.markdown("**Interactive GIS Web Map (Esri Satellite / Topography)**")
            safe_render_folium(folium_map, height=620)
            if fig_map is not None:
                with st.expander("🖼️ View 2D Spatial Raster Grid & Bounding Coordinates (Matplotlib)", expanded=False):
                    st.pyplot(fig_map)
                    plt.close(fig_map)
        else:
            col_folium, col_mpl = st.columns([1.3, 1])
            with col_folium:
                st.markdown("**Interactive GIS Web Map (Esri Satellite / Topography)**")
                safe_render_folium(folium_map, height=540)
            with col_mpl:
                st.markdown("**2D Spatial Raster Grid & Bounding Coordinates**")
                if fig_map is not None:
                    st.pyplot(fig_map)
                    plt.close(fig_map)

    # =========================================================================
    # TAB 2: DIAGNOSTICS & ROC
    # =========================================================================
    with tab2:
        st.subheader("Statistical Validation & Decision Thresholds")
        d_col1, d_col2 = st.columns([1.2, 1])

        with d_col1:
            if roc_info is not None and len(roc_info['fpr']) > 0:
                fig_roc = plot_plotly_roc(
                    fpr=roc_info['fpr'],
                    tpr=roc_info['tpr'],
                    auc_val=roc_info['auc'],
                    thresholds=roc_info.get('thresholds'),
                    opt_threshold=metrics.get('threshold')
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC Curve data not available.")

        with d_col2:
            st.markdown("#### 🎯 Decision Cutoff Thresholds")
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

    # =========================================================================
    # TAB 3: VARIABLE IMPORTANCE & JACKKNIFE
    # =========================================================================
    with tab3:
        st.subheader("Environmental Predictor Contributions & Jackknife Test")
        v_col1, v_col2 = st.columns([1.3, 1])

        with v_col1:
            if not var_importance.empty:
                fig_imp = plot_plotly_variable_importance(var_importance)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Variable importance calculation in progress.")

        with v_col2:
            st.markdown("#### 📊 Variable Contributions Table")
            safe_render_df(var_importance)

            # Show official Jackknife chart if available
            if "plots" in curr_res and any("jacknife" in p for p in curr_res["plots"]):
                for k, p in curr_res["plots"].items():
                    if "jacknife" in k:
                        st.image(p, caption="Official Jackknife Test of Variable Importance", use_container_width=True)

    # =========================================================================
    # TAB 4: RESPONSE CURVES
    # =========================================================================
    with tab4:
        st.subheader("Marginal Environmental Response Curves")
        st.markdown("Illustrates how predicted suitability changes as each environmental variable varies across its gradient, holding other variables constant.")

        resp_data = curr_res.get('response_curve_data', {})
        if resp_data:
            avail_vars = [k for k in resp_data.keys() if not k.endswith("_only")]
            if avail_vars:
                sel_var = st.selectbox("Select Environmental Predictor", avail_vars)
                df_var = resp_data[sel_var]
                fig_rc = plot_plotly_response_curve(
                    var_name=sel_var,
                    x_vals=df_var['x'].values,
                    y_vals=df_var['y'].values,
                    y_label=f"Suitability ({output_format.upper()})"
                )
                st.plotly_chart(fig_rc, use_container_width=True)

        elif "model" in curr_res:
            fig_resp = plot_response_curves(curr_res["model"])
            st.pyplot(fig_resp)
            plt.close(fig_resp)

        # Also show gallery of official MaxEnt response curve PNGs
        if "plots" in curr_res:
            st.markdown("---")
            st.markdown("#### 🖼️ Official MaxEnt Response Curve Figures")
            png_plots = [v for k, v in curr_res["plots"].items() if not any(x in k for x in ["roc", "omission", "jacknife", "overview"])]
            if png_plots:
                cols = st.columns(min(3, len(png_plots)))
                for idx, ppath in enumerate(png_plots):
                    with cols[idx % len(cols)]:
                        st.image(ppath, caption=os.path.basename(ppath), use_container_width=True)

    # =========================================================================
    # TAB 5: PREDICTIONS & EXPORT SUITE
    # =========================================================================
    with tab5:
        st.subheader("Occurrence Predictions & Export Suite")
        safe_render_df(sample_preds.head(100))

        st.markdown("---")
        st.subheader("📥 Export Habitat Suitability Models & Publications")
        e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)

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
            asc_bytes = export_ascii_grid_bytes(suitability_grid, grid_extent)
            st.download_button(
                "📥 ESRI ASCII (.asc)",
                data=asc_bytes,
                file_name="maxent_suitability_prediction.asc",
                mime="text/plain",
                help="ESRI ASCII Grid compatible with all legacy and modern GIS tools."
            )

        with e_col3:
            csv_bytes = sample_preds.to_csv(index=False).encode('utf-8')
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
            # Full ZIP Bundle
            with st.spinner("Compiling full ZIP package..."):
                html_map_str = folium_map._repr_html_()
                lambdas_p = curr_res.get('lambdas_path')
                zip_bytes = create_export_zip_bundle(
                    suitability_grid=suitability_grid,
                    extent=grid_extent,
                    predictions_df=sample_preds,
                    metrics=metrics,
                    html_map_str=html_map_str,
                    png_map_bytes=png_bytes,
                    lambdas_path=lambdas_p
                )
                st.download_button(
                    "📦 Full Bundle (.zip)",
                    data=zip_bytes,
                    file_name="maxent_complete_results.zip",
                    mime="application/zip",
                    help="Complete package containing GeoTIFF, ASCII grid, CSVs, map, and metadata."
                )

    # =========================================================================
    # TAB 6: LIVE ENGINE LOGS & DETAILS
    # =========================================================================
    with tab6:
        st.subheader("Engine Execution Console & System Architecture")
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
