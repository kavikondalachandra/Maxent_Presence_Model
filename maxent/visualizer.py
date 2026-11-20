"""
visualizer.py - Plotting and Spatial Mapping Suite for MaxEnt Models.

Provides:
- 4-panel Model Diagnostic Dashboard (ROC, Importance, Suitability, Boyce)
- Environmental Response Curves
- 2D Spatial Habitat Suitability Maps
- Interactive Leaflet/Folium Map with OpenStreetMap Base
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import folium
from folium.plugins import HeatMap


def plot_model_diagnostics(
    model: Any,
    X_presence_val: Union[np.ndarray, pd.DataFrame],
    X_background_val: Union[np.ndarray, pd.DataFrame],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate a 4-panel diagnostic dashboard for the fitted MaxEnt model.
    
    Panels:
    1. ROC Curve with AUC & Optimal Threshold
    2. Variable Importance (Permutation Importance)
    3. Suitability Distribution (Presence vs Background)
    4. Environmental Response Curves for Top 2 Variables
    
    Parameters
    ----------
    model : MaxEnt
        Fitted MaxEnt model instance.
    X_presence_val : array-like
        Presence data for evaluation.
    X_background_val : array-like
        Background data for evaluation.
    save_path : str, optional
        Path to save figure image (PNG/PDF).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    suit_p = model.predict(X_presence_val, output_type='cloglog')
    suit_b = model.predict(X_background_val, output_type='cloglog')
    
    y_true = np.concatenate([np.ones_like(suit_p), np.zeros_like(suit_b)])
    y_pred = np.concatenate([suit_p, suit_b])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    metrics = model.evaluate(X_presence_val, X_background_val)
    imp_df = model.get_feature_importance(X_presence_val, X_background_val)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#f8f9fa')

    # --- 1. ROC CURVE ---
    ax_roc = axes[0, 0]
    ax_roc.set_facecolor('#ffffff')
    ax_roc.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f"MaxEnt (AUC = {metrics['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], color='#7f7f7f', linestyle='--', lw=1.5, label='Random (AUC = 0.500)')
    
    # Mark optimal threshold
    best_th = metrics['threshold']
    idx = np.argmin(np.abs(thresholds - best_th))
    ax_roc.scatter(
        [fpr[idx]], [tpr[idx]],
        color='#d62728', s=90, zorder=5,
        label=f"Max TSS Cutoff ({best_th:.2f})"
    )
    
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax_roc.set_title('A. Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold', pad=10)
    ax_roc.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')
    ax_roc.grid(True, linestyle=':', alpha=0.6)

    # --- 2. VARIABLE IMPORTANCE ---
    ax_imp = axes[0, 1]
    ax_imp.set_facecolor('#ffffff')
    y_pos = np.arange(len(imp_df))
    bars = ax_imp.barh(
        y_pos, imp_df['Permutation_Importance'],
        color='#2ca02c', edgecolor='#1e7e34', alpha=0.85, height=0.55
    )
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(imp_df['Variable'], fontsize=10, fontweight='bold')
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel('Permutation Importance (%)', fontsize=11, fontweight='bold')
    ax_imp.set_title('B. Variable Permutation Importance', fontsize=13, fontweight='bold', pad=10)
    ax_imp.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    for bar in bars:
        w = bar.get_width()
        ax_imp.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                    va='center', ha='left', fontsize=9, fontweight='bold', color='#333333')
    ax_imp.set_xlim(0, max(imp_df['Permutation_Importance'].max() * 1.25, 20))

    # --- 3. SUITABILITY DISTRIBUTION HISTOGRAM ---
    ax_dist = axes[1, 0]
    ax_dist.set_facecolor('#ffffff')
    bins = np.linspace(0, 1, 30)
    ax_dist.hist(suit_b, bins=bins, density=True, alpha=0.5, color='#7f7f7f', label='Background / Landscape', edgecolor='white')
    ax_dist.hist(suit_p, bins=bins, density=True, alpha=0.7, color='#1f77b4', label='Presence Sites', edgecolor='white')
    ax_dist.axvline(best_th, color='#d62728', linestyle='--', lw=2.0, label=f"Decision Cutoff ({best_th:.2f})")
    ax_dist.set_xlabel('Predicted Habitat Suitability (Cloglog)', fontsize=11, fontweight='bold')
    ax_dist.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax_dist.set_title(f'C. Suitability Distribution (Boyce Index = {metrics["boyce_index"]:.3f})', fontsize=13, fontweight='bold', pad=10)
    ax_dist.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')
    ax_dist.grid(True, linestyle=':', alpha=0.6)

    # --- 4. TOP 2 RESPONSE CURVES ---
    ax_resp = axes[1, 1]
    ax_resp.set_facecolor('#ffffff')
    top_vars = imp_df['Variable'].tolist()[:2]
    colors = ['#ff7f0e', '#9467bd']
    
    for var_name, color in zip(top_vars, colors):
        x_vals, suit_vals = model.get_response_curve(var_name, n_points=100)
        # Normalize x to [0, 1] for dual plotting comparison
        x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
        ax_resp.plot(x_norm, suit_vals, label=f"{var_name} [{x_vals.min():.1f} to {x_vals.max():.1f}]", color=color, lw=2.5)

    ax_resp.set_xlabel('Normalized Environmental Gradient [0 = Min, 1 = Max]', fontsize=11, fontweight='bold')
    ax_resp.set_ylabel('Predicted Suitability (Cloglog)', fontsize=11, fontweight='bold')
    ax_resp.set_title('D. Marginal Response Curves (Top Drivers)', fontsize=13, fontweight='bold', pad=10)
    ax_resp.set_ylim([-0.02, 1.02])
    ax_resp.legend(loc='best', frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')
    ax_resp.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_response_curves(
    model: Any,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot individual marginal response curves for all environmental variables.
    """
    raw_vars = model.raw_feature_names_
    n_vars = len(raw_vars)
    ncols = min(3, n_vars)
    nrows = int(np.ceil(n_vars / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for idx, var in enumerate(raw_vars):
        ax = axes[idx]
        ax.set_facecolor('#ffffff')
        x_vals, suit = model.get_response_curve(var, n_points=120)
        ax.plot(x_vals, suit, color='#007acc', lw=2.5)
        ax.fill_between(x_vals, 0, suit, color='#007acc', alpha=0.15)
        ax.set_ylim([-0.02, 1.02])
        ax.set_title(f"Response: {var}", fontsize=11, fontweight='bold')
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel('Suitability', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_spatial_prediction(
    suitability_grid: np.ndarray,
    extent: Optional[List[float]] = None,
    presence_coords: Optional[pd.DataFrame] = None,
    threshold: Optional[float] = None,
    title: str = "Species Habitat Suitability Prediction (MaxEnt)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 2D geographic suitability raster map with presence overlay.
    
    Parameters
    ----------
    suitability_grid : np.ndarray of shape (H, W)
        Predicted suitability values.
    extent : list of float, optional
        [min_lon, max_lon, min_lat, max_lat].
    presence_coords : pd.DataFrame, optional
        DataFrame containing 'longitude' and 'latitude' (or 'grid_x', 'grid_y').
    threshold : float, optional
        Threshold line to display as a contour.
    title : str
        Figure title.
    save_path : str, optional
        Path to save plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f8f9fa')

    # Display suitability raster (origin='upper' ensures Row 0 = North aligns with geospatial extents)
    im = ax.imshow(
        suitability_grid,
        cmap='viridis',
        extent=extent,
        origin='upper',
        vmin=0.0,
        vmax=1.0
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Habitat Suitability (Cloglog)', fontsize=11, fontweight='bold')

    # Overlay threshold contour if provided
    if threshold is not None and not np.isnan(threshold):
        ax.contour(
            suitability_grid,
            levels=[threshold],
            colors=['#ff3333'],
            linewidths=[2.0],
            extent=extent,
            origin='upper'
        )

    # Overlay presence coordinates
    if presence_coords is not None:
        if 'longitude' in presence_coords and 'latitude' in presence_coords and extent is not None:
            ax.scatter(
                presence_coords['longitude'],
                presence_coords['latitude'],
                color='#ff2222',
                edgecolor='white',
                linewidth=0.8,
                s=40,
                alpha=0.9,
                label='Presence Occurrences'
            )
            ax.set_xlabel('Longitude (°)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Latitude (°)', fontsize=11, fontweight='bold')
        elif 'grid_x' in presence_coords and 'grid_y' in presence_coords:
            ax.scatter(
                presence_coords['grid_x'],
                presence_coords['grid_y'],
                color='#ff2222',
                edgecolor='white',
                linewidth=0.8,
                s=40,
                alpha=0.9,
                label='Presence Occurrences'
            )
            ax.set_xlabel('Grid X (pixels)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Grid Y (pixels)', fontsize=11, fontweight='bold')

        ax.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_spatial_difference(
    diff_grid: np.ndarray,
    extent: List[float],
    title: str = "Absolute Suitability Discrepancy (|Java - Python|)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Render 2D spatial heatmap of absolute prediction discrepancy between two engines.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f8f9fa')

    valid_diff = diff_grid[~np.isnan(diff_grid)]
    vmax = float(np.nanpercentile(valid_diff, 99)) if len(valid_diff) > 0 else 0.5
    vmax = max(vmax, 0.05)

    im = ax.imshow(
        diff_grid,
        cmap='magma',
        extent=extent,
        origin='upper',
        vmin=0.0,
        vmax=vmax
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Difference |Java - Python|', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Longitude (°)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_spatial_comparison(
    grid_java: np.ndarray,
    grid_py: np.ndarray,
    extent: List[float],
    title: str = "Dual-Engine Spatial Comparison: Official Java vs. Native Python",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Render side-by-side 3-panel figure showing Java prediction, Python prediction, and Difference.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('#f8f9fa')

    # 1. Java
    im0 = axes[0].imshow(grid_java, cmap='viridis', extent=extent, origin='upper', vmin=0.0, vmax=1.0)
    axes[0].set_title("A. Official MaxEnt (Java)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude (°)')
    axes[0].set_ylabel('Latitude (°)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Python
    im1 = axes[1].imshow(grid_py, cmap='viridis', extent=extent, origin='upper', vmin=0.0, vmax=1.0)
    axes[1].set_title("B. Native Python Engine", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude (°)')
    axes[1].set_ylabel('Latitude (°)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Difference
    diff = np.abs(grid_java - grid_py)
    v_diff = diff[~np.isnan(diff)]
    vmax = float(np.nanpercentile(v_diff, 99)) if len(v_diff) > 0 else 0.5
    vmax = max(vmax, 0.05)
    mean_val = float(np.nanmean(diff)) if len(v_diff) > 0 else 0.0
    im2 = axes[2].imshow(diff, cmap='magma', extent=extent, origin='upper', vmin=0.0, vmax=vmax)
    axes[2].set_title(f"C. Absolute Discrepancy (Mean: {mean_val:.4f})", fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Longitude (°)')
    axes[2].set_ylabel('Latitude (°)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def create_interactive_map(
    presence_df: pd.DataFrame,
    background_df: Optional[pd.DataFrame] = None,
    suitability_grid: Optional[np.ndarray] = None,
    extent: Optional[List[float]] = None,
    colormap: str = 'viridis',
    opacity: float = 0.65,
    out_html: Optional[str] = None,
    grid_title: str = 'MaxEnt Habitat Suitability Heatmap',
    secondary_grid: Optional[np.ndarray] = None,
    secondary_name: str = 'Native Python Suitability',
    diff_grid: Optional[np.ndarray] = None,
    diff_name: str = 'Spatial Discrepancy (|Java - Python|)',
    bg_point_color: str = '#888888',
    show_bg_points: bool = False,
    api_key: Optional[str] = None
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium map with reliable, keyless topographic and
    satellite base layers (free from 403 Forbidden errors and API key blocks).
    
    Includes:
    - High-reliability open basemaps: OpenStreetMap (Default), USGS Topographic, USGS Satellite, OpenTopoMap
    - Optional authenticated Esri ArcGIS Online layers if user provides an API key/token
    - Automatic coordinate validation, numeric conversion, and NaN filtering
    - Automatic bounds fitting (fit_bounds) to center around occurrence records
    - MaxEnt Predicted Habitat Suitability raster overlay (with dynamic colormap & opacity)
    - Optional secondary comparison raster overlay and difference overlay
    - Species Presence density heatmap
    - Interactive presence markers with coordinate popups
    - Configurable background sample sites with custom color
    - Fixed bottom citation & data sources attribution badge
    - Non-intrusive collapsed layer control so map canvas is never obscured or blocked
    """
    # 1. Clean and validate presence coordinates
    p_df = presence_df.copy()
    p_df['latitude'] = pd.to_numeric(p_df['latitude'], errors='coerce')
    p_df['longitude'] = pd.to_numeric(p_df['longitude'], errors='coerce')
    valid_p = p_df.dropna(subset=['latitude', 'longitude'])

    # Handle swapped coordinates (where lat is > 90 and lon is within [-90, 90])
    if len(valid_p) > 0:
        if valid_p['latitude'].abs().max() > 90.0 and valid_p['longitude'].abs().max() <= 90.0:
            valid_p['latitude'], valid_p['longitude'] = valid_p['longitude'].copy(), valid_p['latitude'].copy()

        # Clip / filter valid geographic coordinates
        valid_p = valid_p[
            (valid_p['latitude'] >= -90.0) & (valid_p['latitude'] <= 90.0) &
            (valid_p['longitude'] >= -180.0) & (valid_p['longitude'] <= 180.0)
        ]

    if len(valid_p) > 0:
        min_lat = float(valid_p['latitude'].min())
        max_lat = float(valid_p['latitude'].max())
        min_lon = float(valid_p['longitude'].min())
        max_lon = float(valid_p['longitude'].max())
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
    else:
        min_lat, max_lat, min_lon, max_lon = 20.0, 50.0, -125.0, -70.0
        center_lat, center_lon = 37.5, -95.0

    # Initialize map with Canvas optimization for smooth Leaflet rendering
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles=None,
        prefer_canvas=True
    )

    # 1. Keyless, high-reliability base layers (immune to 403 Forbidden / API Key restrictions)
    # Layer 1: OpenStreetMap Standard (Clean, high-performance general basemap - Primary Default)
    osm_url = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
    osm_attr = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'
    folium.TileLayer(
        tiles=osm_url,
        attr=osm_attr,
        name='OpenStreetMap (Default &bull; Keyless)',
        max_zoom=19,
        overlay=False,
        control=True
    ).add_to(m)

    # Layer 2: USGS Topographic Map (USGS National Map, Public Domain, zero API key)
    usgs_topo_url = 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}'
    usgs_topo_attr = 'USGS - The National Map: National Boundaries, 3DEP Elevation'
    folium.TileLayer(
        tiles=usgs_topo_url,
        attr=usgs_topo_attr,
        name='USGS Topographic Map (Public Domain)',
        max_zoom=16,
        overlay=False,
        control=True
    ).add_to(m)

    # Layer 3: USGS High-Resolution Satellite Imagery (Public Domain, zero API key)
    usgs_sat_url = 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}'
    usgs_sat_attr = 'USGS - The National Map: Orthoimagery'
    folium.TileLayer(
        tiles=usgs_sat_url,
        attr=usgs_sat_attr,
        name='USGS Satellite Imagery (Public Domain)',
        max_zoom=16,
        overlay=False,
        control=True
    ).add_to(m)

    # Layer 4: OpenTopoMap (Topographic contours and hillshading, keyless open access)
    opentopo_url = 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png'
    opentopo_attr = 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors, SRTM | Map style: &copy; <a href="https://opentopomap.org" target="_blank">OpenTopoMap</a>'
    folium.TileLayer(
        tiles=opentopo_url,
        attr=opentopo_attr,
        name='OpenTopoMap (Topographic Shading)',
        max_zoom=17,
        overlay=False,
        control=True
    ).add_to(m)

    # Layer 6 & 7: Esri ArcGIS Online layers (Optional; appends token if user provided API key)
    token_suffix = f"?token={api_key.strip()}" if (api_key and str(api_key).strip()) else ""
    esri_sat_url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}{token_suffix}'
    esri_sat_attr = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid'
    folium.TileLayer(
        tiles=esri_sat_url,
        attr=esri_sat_attr,
        name='Esri World Imagery' + (' (Key Authenticated)' if token_suffix else ' (Satellite)'),
        max_zoom=19,
        overlay=False,
        control=True
    ).add_to(m)

    esri_topo_url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}{token_suffix}'
    esri_topo_attr = 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI'
    folium.TileLayer(
        tiles=esri_topo_url,
        attr=esri_topo_attr,
        name='Esri Topographic' + (' (Key Authenticated)' if token_suffix else ''),
        max_zoom=19,
        overlay=False,
        control=True
    ).add_to(m)

    # Automatically fit viewport to bounds of presence records
    if len(valid_p) > 1 and (min_lat != max_lat or min_lon != max_lon):
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=[30, 30])

    # 2. Add Primary Suitability Raster Overlay (if provided)
    if suitability_grid is not None and extent is not None:
        grid_min_lon, grid_max_lon, grid_min_lat, grid_max_lat = extent
        valid_mask = ~np.isnan(suitability_grid)
        if np.any(valid_mask):
            norm_grid = np.clip(suitability_grid, 0.0, 1.0)
            try:
                cmap_fn = getattr(plt.cm, colormap, plt.cm.viridis)
            except Exception:
                cmap_fn = plt.cm.viridis
            rgba = cmap_fn(norm_grid)
            rgba[:, :, 3] = np.where(valid_mask, np.clip(norm_grid * opacity + 0.1, 0.0, opacity), 0.0)

            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[grid_min_lat, grid_min_lon], [grid_max_lat, grid_max_lon]],
                opacity=opacity,
                name=grid_title,
                show=True
            ).add_to(m)

    # 2b. Add Secondary Comparison Raster Overlay (if provided)
    if secondary_grid is not None and extent is not None:
        grid_min_lon, grid_max_lon, grid_min_lat, grid_max_lat = extent
        valid_mask_sec = ~np.isnan(secondary_grid)
        if np.any(valid_mask_sec):
            norm_grid_sec = np.clip(secondary_grid, 0.0, 1.0)
            try:
                cmap_fn = getattr(plt.cm, colormap, plt.cm.viridis)
            except Exception:
                cmap_fn = plt.cm.viridis
            rgba_sec = cmap_fn(norm_grid_sec)
            rgba_sec[:, :, 3] = np.where(valid_mask_sec, np.clip(norm_grid_sec * opacity + 0.1, 0.0, opacity), 0.0)

            folium.raster_layers.ImageOverlay(
                image=rgba_sec,
                bounds=[[grid_min_lat, grid_min_lon], [grid_max_lat, grid_max_lon]],
                opacity=opacity,
                name=secondary_name,
                show=False
            ).add_to(m)

    # 2c. Add Spatial Difference Overlay (if provided)
    if diff_grid is not None and extent is not None:
        grid_min_lon, grid_max_lon, grid_min_lat, grid_max_lat = extent
        valid_mask_d = ~np.isnan(diff_grid)
        if np.any(valid_mask_d):
            max_d = float(np.nanpercentile(diff_grid[valid_mask_d], 99)) if np.any(valid_mask_d) else 0.5
            max_d = max(max_d, 0.05)
            norm_d = np.clip(diff_grid / max_d, 0.0, 1.0)
            cmap_diff = plt.cm.magma
            rgba_d = cmap_diff(norm_d)
            rgba_d[:, :, 3] = np.where(valid_mask_d, np.clip(norm_d * opacity + 0.15, 0.0, opacity), 0.0)

            folium.raster_layers.ImageOverlay(
                image=rgba_d,
                bounds=[[grid_min_lat, grid_min_lon], [grid_max_lat, grid_max_lon]],
                opacity=opacity,
                name=diff_name,
                show=False
            ).add_to(m)

    # 3. Add presence points as HeatMap density layer
    if len(valid_p) > 0:
        heat_data = [[float(r['latitude']), float(r['longitude']), 1.0] for _, r in valid_p.iterrows()]
        HeatMap(
            heat_data,
            radius=16,
            blur=12,
            max_zoom=12,
            name="Presence Density Heatmap",
            show=False
        ).add_to(m)

        # 4. Add individual presence markers with tooltip & coordinates
        pres_group = folium.FeatureGroup(name="Presence Records", show=True)
        # Limit to 1,500 individual markers to prevent browser DOM lag
        display_p = valid_p.head(1500) if len(valid_p) > 1500 else valid_p
        for _, row in display_p.iterrows():
            folium.CircleMarker(
                location=[float(row['latitude']), float(row['longitude'])],
                radius=4.5,
                color='#d62728',
                fill=True,
                fill_color='#ff3333',
                fill_opacity=0.9,
                popup=f"<b>Species Presence Site</b><br>Lat: {row['latitude']:.4f}°<br>Lon: {row['longitude']:.4f}°",
                tooltip="Species Presence Record"
            ).add_to(pres_group)
        pres_group.add_to(m)

    # 5. Add background points feature group (with customizable color)
    if background_df is not None:
        b_df = background_df.copy()
        if 'latitude' in b_df.columns and 'longitude' in b_df.columns:
            b_df['latitude'] = pd.to_numeric(b_df['latitude'], errors='coerce')
            b_df['longitude'] = pd.to_numeric(b_df['longitude'], errors='coerce')
            valid_b = b_df.dropna(subset=['latitude', 'longitude'])
            if len(valid_b) > 0:
                bg_group = folium.FeatureGroup(name="Background Sample Sites", show=show_bg_points)
                sub_bg = valid_b.sample(min(300, len(valid_b)), random_state=42)
                for _, row in sub_bg.iterrows():
                    folium.CircleMarker(
                        location=[float(row['latitude']), float(row['longitude'])],
                        radius=2.5,
                        color='#333333',
                        fill=True,
                        fill_color=bg_point_color,
                        fill_opacity=0.6,
                        tooltip=f"Background Sample ({bg_point_color})"
                    ).add_to(bg_group)
                bg_group.add_to(m)

    # 6. Add fixed bottom citation and data sources badge (clean, non-blocking)
    citation_html = """
    <div id="map-citation-attribution" style="
        position: fixed;
        bottom: 12px;
        left: 12px;
        z-index: 9999;
        background: rgba(15, 23, 42, 0.90);
        color: #f8fafc;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 11px;
        line-height: 1.45;
        border: 1px solid #334155;
        max-width: 520px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        pointer-events: auto;
    ">
        <div style="font-weight:700; color:#38bdf8; margin-bottom:2px;">
            🌍 MaxEnt Studio Pro &bull; Species Distribution Model
        </div>
        <div style="font-size:10px; color:#cbd5e1;">
            <b>Software:</b> Chandra, K. S. (2026) | MaxEnt: Phillips et al. (2006, 2017) | CBI: Hirzel et al. (2006)<br>
            <b>Data:</b> WorldClim 2 (Fick &amp; Hijmans 2017) &bull; USGS 3DEP DEM &bull; GBIF Occurrences<br>
            <b>Tiles:</b> OpenStreetMap contributors &bull; USGS National Map (Public Domain, Keyless) &bull; OpenTopoMap
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(citation_html))

    # Set collapsed=True so the layer selection box does NOT block the map canvas!
    folium.LayerControl(collapsed=True, position='topright').add_to(m)
    if out_html:
        m.save(out_html)
    return m
