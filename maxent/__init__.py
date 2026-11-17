"""
MaxEnt Presence Prediction Model Package.

A modular, user-friendly Python implementation of Maximum Entropy Species Distribution Modeling.
"""

import os
import site

# Ensure PROJ environment points to bundled database to avoid third-party GIS conflicts
for _p in site.getsitepackages():
    _cand = os.path.join(_p, 'rasterio', 'proj_data')
    if os.path.exists(_cand):
        os.environ['PROJ_DATA'] = _cand
        os.environ['PROJ_LIB'] = _cand
        break

from .model import MaxEnt
from .java_engine import MaxEntJarRunner, find_maxent_jar, check_java_runtime
from .features import FeatureTransformer
from .metrics import evaluate_sdm, find_optimal_thresholds, calculate_continuous_boyce_index
from .data import (
    generate_synthetic_landscape,
    load_csv_data,
    load_raster_layers,
    extract_values_at_points,
    sample_background_from_rasters,
    create_spatial_grid_from_points,
    export_geotiff,
    export_ascii_grid_bytes,
    create_export_zip_bundle
)
from .visualizer import (
    plot_model_diagnostics,
    plot_response_curves,
    plot_spatial_prediction,
    create_interactive_map
)

__version__ = "1.0.0"

__all__ = [
    "MaxEnt",
    "MaxEntJarRunner",
    "find_maxent_jar",
    "check_java_runtime",
    "FeatureTransformer",
    "evaluate_sdm",
    "find_optimal_thresholds",
    "calculate_continuous_boyce_index",
    "generate_synthetic_landscape",
    "load_csv_data",
    "load_raster_layers",
    "extract_values_at_points",
    "sample_background_from_rasters",
    "create_spatial_grid_from_points",
    "export_geotiff",
    "plot_model_diagnostics",
    "plot_response_curves",
    "plot_spatial_prediction",
    "create_interactive_map",
]
