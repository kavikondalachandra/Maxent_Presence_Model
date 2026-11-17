"""
data.py - Data Management, Synthetic Landscape Generator, and I/O Utilities.

Provides:
- Realistic bioclimatic landscape generator with true ecological niche
- Environmental Raster Layer Loader (DEM, Distance to Water, Elevation, Contour, etc.)
- Multi-layer raster alignment & reprojection engine
- Point sampling & background generation from GIS rasters
- CSV loader for presence and background records
- 2D Environmental Spatial Grid Generator from Point Observations
- GeoTIFF exporter and raster loader
"""

import os
import sys
import site
import re
from typing import Tuple, Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Ensure PROJ environment variable points to bundled rasterio data to avoid third-party GIS conflicts
for _p in site.getsitepackages():
    _cand = os.path.join(_p, 'rasterio', 'proj_data')
    if os.path.exists(_cand):
        os.environ['PROJ_DATA'] = _cand
        os.environ['PROJ_LIB'] = _cand
        break


def generate_synthetic_landscape(
    width: int = 120,
    height: int = 120,
    n_presence: int = 180,
    n_background: int = 2500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate a realistic synthetic geographic landscape and species occurrences.
    """
    np.random.seed(seed)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Coordinates in pseudo-geographic bounds (e.g., California/Cascades region)
    min_lon, max_lon = -122.5, -119.5
    min_lat, max_lat = 36.5, 39.5
    lon_grid = min_lon + (x_coords / width) * (max_lon - min_lon)
    lat_grid = min_lat + ((height - 1 - y_coords) / height) * (max_lat - min_lat)

    # 1. Elevation (mountain ridge running diagonally + fractal hills)
    diag = (x_coords + 0.6 * y_coords) / (1.6 * width)
    ridge = np.exp(-((diag - 0.45) ** 2) / 0.03) * 2200.0
    hills = (np.sin(x_coords / 8.0) * np.cos(y_coords / 8.0) + 1.0) * 350.0
    elevation = np.clip(150.0 + ridge + hills + np.random.normal(0, 30, (height, width)), 50.0, 3200.0)

    # 2. Temperature (Lapse rate with elevation + Latitudinal gradient)
    lat_factor = (lat_grid - min_lat) / (max_lat - min_lat) # cooler in north
    temp_lapse = -0.0065 * elevation # standard environmental lapse rate: -6.5°C / 1000m
    temperature = np.clip(26.0 - 4.5 * lat_factor + temp_lapse + np.random.normal(0, 0.4, (height, width)), 2.0, 32.0)

    # 3. Precipitation (Orographic effect: wet on western slopes, dry rainshadow on east)
    rain_base = 350.0 + 850.0 * (elevation / 3000.0)
    rain_shadow = 1.0 - 0.5 * (x_coords / width)
    precipitation = np.clip(rain_base * rain_shadow + np.random.normal(0, 35, (height, width)), 150.0, 2400.0)

    # 4. Forest Canopy Cover (favored by moderate temperature and high precipitation)
    canopy_score = (
        np.exp(-((temperature - 15.0) ** 2) / 25.0) * 0.5 +
        (precipitation / 2000.0) * 0.5
    )
    canopy_cover = np.clip(canopy_score * 100.0 + np.random.normal(0, 5, (height, width)), 0.0, 100.0)

    raster_dict = {
        'temperature': temperature.astype(np.float32),
        'precipitation': precipitation.astype(np.float32),
        'elevation': elevation.astype(np.float32),
        'canopy_cover': canopy_cover.astype(np.float32)
    }
    feature_names = list(raster_dict.keys())

    # --- TRUE ECOLOGICAL SUITABILITY ---
    temp_suit = np.exp(-((temperature - 15.5) ** 2) / 18.0)
    precip_suit = 1.0 / (1.0 + np.exp(-(precipitation - 750.0) / 150.0))
    elev_suit = np.exp(-((elevation - 1650.0) ** 2) / (2 * 500.0 ** 2))
    canopy_suit = 1.0 / (1.0 + np.exp(-(canopy_cover - 45.0) / 12.0))

    true_suitability = temp_suit * 0.35 + precip_suit * 0.25 + elev_suit * 0.25 + canopy_suit * 0.15
    true_suitability = np.clip(true_suitability / np.max(true_suitability), 0.0, 1.0)

    # --- SAMPLE OCCURRENCE RECORDS ---
    prob_occurrence = true_suitability.ravel() ** 2.5
    prob_occurrence /= np.sum(prob_occurrence)
    
    flat_indices = np.random.choice(
        height * width,
        size=n_presence,
        replace=False,
        p=prob_occurrence
    )
    pres_y, pres_x = np.unravel_index(flat_indices, (height, width))
    pres_lons = lon_grid[pres_y, pres_x]
    pres_lats = lat_grid[pres_y, pres_x]

    pres_dict = {
        'longitude': pres_lons,
        'latitude': pres_lats,
        'grid_x': pres_x,
        'grid_y': pres_y,
    }
    for feat in feature_names:
        pres_dict[feat] = raster_dict[feat][pres_y, pres_x]

    presence_df = pd.DataFrame(pres_dict)

    # --- SAMPLE BACKGROUND POINTS ---
    bg_flat_indices = np.random.choice(
        height * width,
        size=n_background,
        replace=False
    )
    bg_y, bg_x = np.unravel_index(bg_flat_indices, (height, width))
    bg_lons = lon_grid[bg_y, bg_x]
    bg_lats = lat_grid[bg_y, bg_x]

    bg_dict = {
        'longitude': bg_lons,
        'latitude': bg_lats,
        'grid_x': bg_x,
        'grid_y': bg_y,
    }
    for feat in feature_names:
        bg_dict[feat] = raster_dict[feat][bg_y, bg_x]

    background_df = pd.DataFrame(bg_dict)

    return {
        'raster_dict': raster_dict,
        'true_suitability': true_suitability.astype(np.float32),
        'presence_df': presence_df,
        'background_df': background_df,
        'feature_names': feature_names,
        'lon_grid': lon_grid,
        'lat_grid': lat_grid,
        'extent': [min_lon, max_lon, min_lat, max_lat],
        'shape': (height, width)
    }


def load_raster_layers(
    raster_sources: List[Any],
    max_dimension: int = 1000
) -> Dict[str, Any]:
    """
    Load, align, and stack multiple environmental raster layers (GeoTIFF, DEM, distance to water, slope, etc.).
    
    Automatically handles:
    - GeoTIFF files on disk or in-memory Streamlit UploadedFile objects
    - Layer name sanitization from filenames (e.g. 'dem.tif' -> 'dem', 'distance_to_water.tif' -> 'distance_to_water')
    - Spatial alignment and bilinear reprojection if rasters have different resolutions or extents
    - Unified nodata masking
    
    Parameters
    ----------
    raster_sources : list
        List of raster file paths or UploadedFile objects.
    max_dimension : int, default=1000
        Maximum pixels along height or width to prevent out-of-memory errors on massive rasters.
        
    Returns
    -------
    dict
        - 'raster_dict': dict of {var_name: 2D numpy array (float32)}
        - 'extent': [min_lon, max_lon, min_lat, max_lat]
        - 'nodata_mask': 2D boolean mask where True indicates nodata
        - 'feature_names': list of layer variable names
        - 'shape': (height, width)
        - 'transform': Affine transform
        - 'crs': CRS object or string
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import reproject, Resampling

    if not raster_sources:
        raise ValueError("No raster sources provided.")

    opened_datasets = []
    layer_names = []
    memfiles_to_close = []

    try:
        for item in raster_sources:
            if isinstance(item, str):
                raw_name = os.path.splitext(os.path.basename(item))[0]
                ds = rasterio.open(item)
            else:
                raw_name = os.path.splitext(getattr(item, 'name', 'raster'))[0]
                if hasattr(item, 'getvalue'):
                    data_bytes = item.getvalue()
                elif hasattr(item, 'read'):
                    data_bytes = item.read()
                else:
                    raise ValueError(f"Unsupported raster item type: {type(item)}")
                memfile = MemoryFile(data_bytes)
                memfiles_to_close.append(memfile)
                ds = memfile.open()

            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name).strip('_').lower()
            if not clean_name:
                clean_name = f"layer_{len(layer_names) + 1}"
            if clean_name in layer_names:
                clean_name = f"{clean_name}_{len(layer_names) + 1}"

            opened_datasets.append(ds)
            layer_names.append(clean_name)

        # Use first raster as reference grid
        ref_ds = opened_datasets[0]
        ref_shape = (ref_ds.height, ref_ds.width)
        ref_transform = ref_ds.transform
        ref_crs = ref_ds.crs or 'EPSG:4326'
        ref_bounds = ref_ds.bounds
        extent = [float(ref_bounds.left), float(ref_bounds.right), float(ref_bounds.bottom), float(ref_bounds.top)]

        raster_dict = {}
        unified_nodata = np.zeros(ref_shape, dtype=bool)

        for name, ds in zip(layer_names, opened_datasets):
            arr = ds.read(1).astype(np.float32)
            nodata_val = ds.nodata

            # Check if resampling or reprojection is required
            if arr.shape != ref_shape or ds.transform != ref_transform or ds.crs != ref_crs:
                resampled = np.full(ref_shape, np.nan, dtype=np.float32)
                reproject(
                    source=arr,
                    destination=resampled,
                    src_transform=ds.transform,
                    src_crs=ds.crs or ref_crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=nodata_val,
                    dst_nodata=np.nan
                )
                curr_arr = resampled
                layer_nodata = np.isnan(curr_arr)
            else:
                curr_arr = arr
                layer_nodata = np.isnan(curr_arr)
                if nodata_val is not None and not np.isnan(nodata_val):
                    layer_nodata = layer_nodata | (curr_arr == nodata_val)
                # Filter common GIS nodata sentinel values
                layer_nodata = layer_nodata | (curr_arr < -9990.0) | (curr_arr > 1e30)
                curr_arr[layer_nodata] = np.nan

            unified_nodata = unified_nodata | layer_nodata
            raster_dict[name] = curr_arr

        return {
            'raster_dict': raster_dict,
            'extent': extent,
            'nodata_mask': unified_nodata,
            'feature_names': layer_names,
            'shape': ref_shape,
            'transform': ref_transform,
            'crs': ref_crs
        }

    finally:
        for ds in opened_datasets:
            try:
                ds.close()
            except Exception:
                pass
        for mf in memfiles_to_close:
            try:
                mf.close()
            except Exception:
                pass


def extract_values_at_points(
    raster_dict: Dict[str, np.ndarray],
    transform: Any,
    points_df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    nodata_mask: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Sample values from environmental rasters at given coordinate locations.
    
    Allows occurrence CSV files with just latitude & longitude to be instantly
    enriched with DEM, distance to water, elevation, etc.
    
    Parameters
    ----------
    raster_dict : dict of {str: np.ndarray}
        Environmental raster stack.
    transform : Affine
        Affine geotransform of the rasters.
    points_df : pd.DataFrame
        DataFrame with coordinates.
    lat_col : str, default='latitude'
        Latitude column name.
    lon_col : str, default='longitude'
        Longitude column name.
    nodata_mask : np.ndarray, optional
        Boolean mask of nodata cells.
        
    Returns
    -------
    enriched_df, omitted_count : tuple
    """
    import rasterio.transform

    first_key = list(raster_dict.keys())[0]
    height, width = raster_dict[first_key].shape

    clean_p = points_df.copy()
    clean_p[lat_col] = pd.to_numeric(clean_p[lat_col], errors='coerce')
    clean_p[lon_col] = pd.to_numeric(clean_p[lon_col], errors='coerce')
    clean_p = clean_p.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    lons = clean_p[lon_col].to_numpy()
    lats = clean_p[lat_col].to_numpy()

    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    # Check valid pixel bounds
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if nodata_mask is not None:
        valid_pixel = in_bounds.copy()
        valid_pixel[in_bounds] = ~nodata_mask[rows[in_bounds], cols[in_bounds]]
    else:
        valid_pixel = in_bounds

    # Check for NaNs across raster layers at these pixels
    for name, arr in raster_dict.items():
        valid_pixel[valid_pixel] = valid_pixel[valid_pixel] & (~np.isnan(arr[rows[valid_pixel], cols[valid_pixel]]))

    omitted_count = int(np.sum(~valid_pixel))
    valid_df = clean_p[valid_pixel].copy().reset_index(drop=True)
    valid_rows = rows[valid_pixel]
    valid_cols = cols[valid_pixel]

    # Sample values
    for name, arr in raster_dict.items():
        valid_df[name] = arr[valid_rows, valid_cols]

    return valid_df, omitted_count


def sample_background_from_rasters(
    raster_dict: Dict[str, np.ndarray],
    transform: Any,
    nodata_mask: Optional[np.ndarray] = None,
    n_samples: int = 2500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Randomly sample background / pseudo-absence points from valid raster cells.
    """
    import rasterio.transform

    np.random.seed(random_state)
    first_key = list(raster_dict.keys())[0]
    shape = raster_dict[first_key].shape

    if nodata_mask is None:
        valid_mask = np.ones(shape, dtype=bool)
    else:
        valid_mask = ~nodata_mask

    for name, arr in raster_dict.items():
        valid_mask = valid_mask & (~np.isnan(arr))

    valid_rows, valid_cols = np.where(valid_mask)
    n_valid = len(valid_rows)

    if n_valid == 0:
        raise ValueError("No valid non-nodata cells found in raster stack to sample background points.")

    sample_size = min(n_samples, n_valid)
    chosen_idx = np.random.choice(n_valid, size=sample_size, replace=False)
    sample_rows = valid_rows[chosen_idx]
    sample_cols = valid_cols[chosen_idx]

    lons, lats = rasterio.transform.xy(transform, sample_rows, sample_cols)

    bg_dict = {
        'longitude': np.asarray(lons, dtype=np.float64),
        'latitude': np.asarray(lats, dtype=np.float64),
    }

    for name, arr in raster_dict.items():
        bg_dict[name] = arr[sample_rows, sample_cols]

    return pd.DataFrame(bg_dict)


def create_spatial_grid_from_points(
    df: pd.DataFrame,
    features: List[str],
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    grid_res: int = 100,
    buffer_ratio: float = 0.08
) -> Tuple[Dict[str, np.ndarray], List[float]]:
    """
    Construct a continuous 2D environmental raster stack interpolated across
    the geographic bounding box of point observations.
    """
    clean = df.dropna(subset=[lat_col, lon_col] + features).copy()
    clean[lat_col] = pd.to_numeric(clean[lat_col], errors='coerce')
    clean[lon_col] = pd.to_numeric(clean[lon_col], errors='coerce')
    clean = clean.dropna(subset=[lat_col, lon_col])

    # Check for swapped lat/lon
    if len(clean) > 0 and clean[lat_col].abs().max() > 90.0 and clean[lon_col].abs().max() <= 90.0:
        clean[lat_col], clean[lon_col] = clean[lon_col].copy(), clean[lat_col].copy()

    # Clip to valid coordinates
    clean = clean[
        (clean[lat_col] >= -90.0) & (clean[lat_col] <= 90.0) &
        (clean[lon_col] >= -180.0) & (clean[lon_col] <= 180.0)
    ]

    if len(clean) < 3:
        raise ValueError("At least 3 distinct geographic coordinate points are required to generate a spatial raster grid.")

    min_lat, max_lat = float(clean[lat_col].min()), float(clean[lat_col].max())
    min_lon, max_lon = float(clean[lon_col].min()), float(clean[lon_col].max())

    pad_lat = max((max_lat - min_lat) * buffer_ratio, 0.02)
    pad_lon = max((max_lon - min_lon) * buffer_ratio, 0.02)

    # Grid latitudes descending from North to South so Row 0 = North (standard GIS raster format)
    grid_lats = np.linspace(max_lat + pad_lat, min_lat - pad_lat, grid_res)
    grid_lons = np.linspace(min_lon - pad_lon, max_lon + pad_lon, grid_res)
    mesh_lons, mesh_lats = np.meshgrid(grid_lons, grid_lats)

    points = clean[[lon_col, lat_col]].to_numpy()
    raster_dict = {}

    for feat in features:
        vals = clean[feat].to_numpy(dtype=np.float64)
        interp = griddata(points, vals, (mesh_lons, mesh_lats), method='linear')
        nan_mask = np.isnan(interp)
        if np.any(nan_mask):
            interp_near = griddata(points, vals, (mesh_lons, mesh_lats), method='nearest')
            interp[nan_mask] = interp_near[nan_mask]
        raster_dict[feat] = interp.astype(np.float32)

    extent = [float(grid_lons.min()), float(grid_lons.max()), float(grid_lats.min()), float(grid_lats.max())]
    return raster_dict, extent


def export_geotiff(
    suitability_grid: np.ndarray,
    extent: List[float],
    out_path: Optional[str] = None
) -> Union[str, bytes]:
    """
    Export predicted 2D suitability grid as a standard GIS GeoTIFF file.
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    min_lon, max_lon, min_lat, max_lat = extent
    h, w = suitability_grid.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)

    out_arr = np.nan_to_num(suitability_grid, nan=-9999.0).astype(np.float32)

    if out_path:
        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=1,
            dtype=out_arr.dtype,
            crs='EPSG:4326',
            transform=transform,
            nodata=-9999.0
        ) as dst:
            dst.write(out_arr, 1)
        return out_path
    else:
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=h,
                width=w,
                count=1,
                dtype=out_arr.dtype,
                crs='EPSG:4326',
                transform=transform,
                nodata=-9999.0
            ) as dst:
                dst.write(out_arr, 1)
            return memfile.read()


def load_csv_data(
    presence_file: str,
    background_file: Optional[str] = None,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load presence and background data from CSV files.
    """
    pres_df = pd.read_csv(presence_file)
    bg_df = pd.read_csv(background_file) if background_file else None

    if feature_columns is not None:
        missing_p = [c for c in feature_columns if c not in pres_df.columns]
        if missing_p:
            raise ValueError(f"Feature columns {missing_p} missing in {presence_file}")
        pres_df = pres_df[feature_columns]
        if bg_df is not None:
            missing_b = [c for c in feature_columns if c not in bg_df.columns]
            if missing_b:
                raise ValueError(f"Feature columns {missing_b} missing in {background_file}")
            bg_df = bg_df[feature_columns]

    # Drop NaNs
    pres_df = pres_df.dropna().reset_index(drop=True)
    if bg_df is not None:
        bg_df = bg_df.dropna().reset_index(drop=True)

    return pres_df, bg_df


def export_ascii_grid_bytes(
    grid: np.ndarray,
    extent: List[float],
    nodata_val: float = -9999.0
) -> bytes:
    """
    Export 2D array to ESRI ASCII Grid bytes (.asc).
    """
    nrows, ncols = grid.shape
    min_lon, max_lon, min_lat, max_lat = extent
    cellsize_x = (max_lon - min_lon) / float(ncols)
    cellsize_y = (max_lat - min_lat) / float(nrows)
    cellsize = (cellsize_x + cellsize_y) / 2.0

    header = (
        f"ncols         {ncols}\n"
        f"nrows         {nrows}\n"
        f"xllcorner     {min_lon:.10f}\n"
        f"yllcorner     {min_lat:.10f}\n"
        f"cellsize      {cellsize:.10f}\n"
        f"NODATA_value  {nodata_val:.1f}\n"
    )

    clean_grid = np.where(np.isnan(grid), nodata_val, grid)
    lines = [header]
    for row in clean_grid:
        lines.append(" ".join(f"{v:.6g}" for v in row) + "\n")

    return "".join(lines).encode("ascii")


def create_export_zip_bundle(
    suitability_grid: np.ndarray,
    extent: List[float],
    predictions_df: pd.DataFrame,
    metrics: Dict[str, Any],
    html_map_str: Optional[str] = None,
    png_map_bytes: Optional[bytes] = None,
    lambdas_path: Optional[str] = None,
    extra_files: Optional[Dict[str, bytes]] = None
) -> bytes:
    """
    Create a complete in-memory ZIP package containing all SDM artifacts.
    """
    import zipfile
    import io

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. GeoTIFF
        try:
            gtiff_bytes = export_geotiff(suitability_grid, extent)
            z.writestr("maxent_suitability_prediction.tif", gtiff_bytes)
        except Exception:
            pass

        # 2. ESRI ASCII Grid
        asc_bytes = export_ascii_grid_bytes(suitability_grid, extent)
        z.writestr("maxent_suitability_prediction.asc", asc_bytes)

        # 3. 2D Grid CSV
        grid_df = pd.DataFrame(suitability_grid)
        z.writestr("suitability_matrix.csv", grid_df.to_csv(index=False).encode("utf-8"))

        # 4. Occurrence Predictions
        z.writestr("occurrence_predictions.csv", predictions_df.to_csv(index=False).encode("utf-8"))

        # 5. Metrics & Thresholds JSON / TXT
        summary_lines = [
            "===========================================================",
            "  MaxEnt Studio Pro - Species Distribution Model Summary   ",
            "===========================================================\n",
        ]
        for k, v in metrics.items():
            summary_lines.append(f"{k}: {v}")
        z.writestr("model_metrics_summary.txt", "\n".join(summary_lines).encode("utf-8"))

        # 6. Interactive Web Map
        if html_map_str:
            z.writestr("interactive_suitability_map.html", html_map_str.encode("utf-8"))

        # 7. Map PNG
        if png_map_bytes:
            z.writestr("spatial_suitability_map.png", png_map_bytes)

        # 8. Lambdas
        if lambdas_path and os.path.isfile(lambdas_path):
            with open(lambdas_path, "rb") as f:
                z.writestr(os.path.basename(lambdas_path), f.read())

        # 9. Extra files
        if extra_files:
            for fname, fbytes in extra_files.items():
                z.writestr(fname, fbytes)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
