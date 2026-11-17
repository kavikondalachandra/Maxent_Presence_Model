"""
cli.py - Command Line Interface for Maximum Entropy Presence Prediction.

Usage Examples:
    # Run built-in demonstration
    python cli.py --demo

    # Train using uploaded environmental GIS rasters (DEM, distance to water, slope, etc.)
    python cli.py --presence sample_data/sample_species_occurrences_latlon.csv \
                  --raster-dir sample_data/ \
                  --output results_rasters/

    # Train using CSV tabular data
    python cli.py --presence sample_data/sample_presence.csv \
                  --background sample_data/sample_background.csv \
                  --features temperature,precipitation,elevation,canopy_cover \
                  --reg-mult 1.0 --output results_csv/
"""

import os
import sys
import site
import glob
import argparse
import numpy as np
import pandas as pd

# Fix PROJ_LIB to prevent third-party GIS conflict on Windows
for _p in site.getsitepackages():
    _cand = os.path.join(_p, 'rasterio', 'proj_data')
    if os.path.exists(_cand):
        os.environ['PROJ_DATA'] = _cand
        os.environ['PROJ_LIB'] = _cand
        break

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maxent import (
    MaxEnt,
    MaxEntJarRunner,
    find_maxent_jar,
    check_java_runtime,
    generate_synthetic_landscape,
    load_csv_data,
    load_raster_layers,
    extract_values_at_points,
    sample_background_from_rasters,
    create_spatial_grid_from_points,
    export_geotiff,
    plot_model_diagnostics,
    plot_response_curves,
    plot_spatial_prediction,
    create_interactive_map
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="User-Friendly Maximum Entropy (MaxEnt) Presence Prediction CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help="Run full end-to-end synthetic demonstration."
    )
    parser.add_argument(
        '--presence',
        type=str,
        default=None,
        help="Path to CSV file containing presence occurrence records."
    )
    parser.add_argument(
        '--background',
        type=str,
        default=None,
        help="Path to CSV file containing background/pseudo-absence points."
    )
    parser.add_argument(
        '--rasters',
        type=str,
        default=None,
        help="Comma-separated list of raster file paths (e.g. dem.tif,dist_water.tif,slope.tif)."
    )
    parser.add_argument(
        '--raster-dir',
        type=str,
        default=None,
        help="Path to folder containing environmental GeoTIFF layers (.tif, .asc, .bil)."
    )
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help="Comma-separated list of environmental feature column names."
    )
    parser.add_argument(
        '--reg-mult',
        type=float,
        default=1.0,
        help="Regularization multiplier (L1 penalty weight)."
    )
    parser.add_argument(
        '--feature-types',
        type=str,
        default="linear,quadratic,product,hinge",
        help="Comma-separated feature expansions: linear,quadratic,product,hinge,threshold"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="results",
        help="Directory to save output reports, maps, and figures."
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default="cloglog",
        choices=['cloglog', 'logistic', 'raw', 'cumulative'],
        help="Default suitability output format."
    )
    parser.add_argument(
        '--engine',
        type=str,
        default='python',
        choices=['python', 'java'],
        help="Modeling engine: 'python' (scikit-learn regularized L-BFGS solver) or 'java' (official AMNH maxent.jar)."
    )
    parser.add_argument(
        '--jar-path',
        type=str,
        default=None,
        help="Custom path to maxent.jar binary (if using --engine java)."
    )
    parser.add_argument(
        '--grid-res',
        type=int,
        default=100,
        help="Resolution (cells per side) for continuous spatial prediction raster if interpolating from points."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo or (args.presence is None):
        print("Running built-in demonstration...")
        from demo import run_demo
        run_demo()
        return

    os.makedirs(args.output, exist_ok=True)
    feat_types = [ft.strip().lower() for ft in args.feature_types.split(',')]

    print("=" * 70)
    print("      MAXIMUM ENTROPY (MaxEnt) PRESENCE PREDICTION")
    print("=" * 70)
    print(f"Loading presence data from:   {args.presence}")

    raw_pres = pd.read_csv(args.presence)

    # Detect coordinate columns
    lat_col = next((c for c in ['latitude', 'lat', 'Latitude', 'LAT', 'decimalLatitude', 'y'] if c in raw_pres.columns), None)
    lon_col = next((c for c in ['longitude', 'lon', 'long', 'Longitude', 'LON', 'decimalLongitude', 'x'] if c in raw_pres.columns), None)

    # -------------------------------------------------------------------------
    # CASE A: Environmental Rasters Provided (DEM, Distance to Water, etc.)
    # -------------------------------------------------------------------------
    raster_files = []
    if args.rasters:
        raster_files = [f.strip() for f in args.rasters.split(',')]
    elif args.raster_dir:
        raster_files = glob.glob(os.path.join(args.raster_dir, "*.tif")) + \
                       glob.glob(os.path.join(args.raster_dir, "*.asc")) + \
                       glob.glob(os.path.join(args.raster_dir, "*.bil"))

    if raster_files:
        print(f"\n[Mode: Environmental Rasters] Found {len(raster_files)} raster layers:")
        for rf in raster_files:
            print(f"  • {rf}")

        r_info = load_raster_layers(raster_files)
        raster_dict = r_info['raster_dict']
        extent = r_info['extent']
        feat_cols = r_info['feature_names']
        transform = r_info['transform']
        nodata_mask = r_info['nodata_mask']

        print(f"Raster grid dimensions: {r_info['shape'][0]} × {r_info['shape'][1]} cells")
        print(f"Geographic Extent:      [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}, {extent[3]:.3f}]")

        if not lat_col or not lon_col:
            raise ValueError("Latitude and Longitude columns are required to sample from environmental rasters.")

        # Sample values at occurrences
        pres_df, omitted = extract_values_at_points(raster_dict, transform, raw_pres, lat_col, lon_col, nodata_mask)
        print(f"Extracted raster values for {len(pres_df)} presence sites ({omitted} omitted outside raster bounds)")

        # Sample background from rasters
        bg_df = sample_background_from_rasters(raster_dict, transform, nodata_mask, n_samples=3000)
        print(f"Sampled {len(bg_df)} background points from raster extent")

    # -------------------------------------------------------------------------
    # CASE B: Tabular CSV Mode (Point Data with Automatic Spatial Gridding)
    # -------------------------------------------------------------------------
    else:
        raw_bg = pd.read_csv(args.background) if args.background else None
        feat_cols = [c.strip() for c in args.features.split(',')] if args.features else None

        if feat_cols is None:
            exclude = {'latitude', 'longitude', 'lat', 'lon', 'species', 'id', 'x', 'y'}
            feat_cols = [c for c in raw_pres.columns if c.lower() not in exclude and pd.api.types.is_numeric_dtype(raw_pres[c])]

        pres_df = raw_pres
        bg_df = raw_bg
        raster_dict = None
        extent = None

        if lat_col and lon_col:
            comb_spatial = pd.concat([raw_pres, raw_bg], ignore_index=True) if raw_bg is not None else raw_pres
            try:
                raster_dict, extent = create_spatial_grid_from_points(
                    df=comb_spatial,
                    features=feat_cols,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    grid_res=args.grid_res
                )
            except Exception as e:
                print(f"Note: Spatial grid interpolation skipped: {e}")

    print(f"\nEnvironmental Predictors: {', '.join(feat_cols)}")
    print(f"Presence records:        {len(pres_df)}")
    print(f"Background records:      {len(bg_df) if bg_df is not None else 0}")

    # -------------------------------------------------------------------------
    # MODEL FITTING
    # -------------------------------------------------------------------------
    if args.engine == 'java':
        print(f"\n[Engine: Official AMNH maxent.jar] Initializing Java execution...")
        j_status = check_java_runtime()
        if not j_status['available']:
            raise RuntimeError(f"Java runtime not found: {j_status['details']}")
        print(f"Java Detected: {j_status['details']}")

        jar = find_maxent_jar(args.jar_path)
        if not jar:
            raise FileNotFoundError("Could not locate maxent.jar. Please specify --jar-path.")
        print(f"MaxEnt JAR:    {jar}")

        if raster_dict is None or extent is None:
            raise ValueError("Official maxent.jar requires spatial raster layers or a spatial bounding grid.")

        runner = MaxEntJarRunner(jar_path=jar, memory_mb=2048)
        j_res = runner.run(
            raster_dict=raster_dict,
            extent=extent,
            presence_df=pres_df,
            lat_col=lat_col or 'latitude',
            lon_col=lon_col or 'longitude',
            species_name="Target_Species",
            output_format=args.output_format,
            reg_mult=args.reg_mult,
            feature_types=feat_types,
            jackknife=True,
            response_curves=True,
            temp_work_dir=os.path.join(args.output, "maxent_jar_workspace"),
            log_callback=print
        )
        print("\n" + "=" * 70)
        print("MaxEnt Java execution completed successfully!")
        print(f"AUC:             {j_res['metrics'].get('auc', 0.0):.4f}")
        print(f"Training Gain:   {j_res['metrics'].get('training_gain', 0.0):.4f}")
        print(f"Optimal Cutoff:  {j_res['metrics'].get('threshold', 0.5):.4f}")
        print(f"Elapsed Time:    {j_res['elapsed_seconds']:.2f}s")
        print("=" * 70)

        # Copy suitability raster, html report, and plots to output dir
        suitability_grid = j_res['suitability_grid']
        j_res['variable_importance'].to_csv(os.path.join(args.output, "variable_importance.csv"), index=False)
        print(f"[Saved] Variable Importance:  {os.path.join(args.output, 'variable_importance.csv')}")

        tif_path = os.path.join(args.output, "maxent_suitability_raster.tif")
        export_geotiff(suitability_grid, extent, out_path=tif_path)
        print(f"[Saved] GeoTIFF (.tif):       {tif_path}")

        html_path = os.path.join(args.output, "interactive_suitability_map.html")
        create_interactive_map(
            presence_df=pres_df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}),
            background_df=bg_df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}) if bg_df is not None and 'latitude' in bg_df.columns else None,
            suitability_grid=suitability_grid,
            extent=extent,
            out_html=html_path
        )
        print(f"[Saved] Interactive Web Map:  {html_path}")

        print(f"\nAll official Java outputs available at: {os.path.abspath(j_res['output_dir'])}")
        return

    # Python Engine Mode
    X_pres = pres_df[feat_cols].dropna()
    X_bg = bg_df[feat_cols].dropna() if bg_df is not None else None

    model = MaxEnt(
        reg_mult=args.reg_mult,
        feature_types=feat_types,
        default_output=args.output_format,
        verbose=True
    )

    print("\nFitting Native Python MaxEnt model...")
    model.fit(X_pres, X_bg, feature_names=feat_cols)
    print("Model optimization complete!\n")
    print(model.summary())

    # 1. Variable Importance Table
    imp_df = model.get_feature_importance(X_pres, X_bg)
    imp_path = os.path.join(args.output, "variable_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"\n[Saved] Variable Importance:  {imp_path}")
    print(imp_df.to_string(index=False))

    # 2. Marginal Response Curves
    resp_path = os.path.join(args.output, "response_curves.png")
    plot_response_curves(model, save_path=resp_path)
    print(f"[Saved] Response Curves:      {resp_path}")

    # 3. Diagnostics Plot
    if X_bg is not None:
        diag_path = os.path.join(args.output, "diagnostics.png")
        plot_model_diagnostics(model, X_pres, X_bg, save_path=diag_path)
        print(f"[Saved] Diagnostic Dashboard: {diag_path}")

    # 4. Continuous Spatial Habitat Suitability Raster
    if raster_dict is not None and extent is not None:
        print(f"\nPredicting 2D Spatial Habitat Suitability across raster landscape...")
        suitability_grid = model.predict_raster(raster_dict, output_type=args.output_format)

        # Save Map Image
        map_png = os.path.join(args.output, "spatial_suitability_map.png")
        plot_spatial_prediction(
            suitability_grid,
            extent=extent,
            presence_coords=pres_df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}),
            threshold=model.training_metrics_.get('threshold', 0.5),
            title=f"Predicted Species Habitat Suitability ({args.output_format.upper()})",
            save_path=map_png
        )
        print(f"[Saved] Spatial Raster Map:   {map_png}")

        # Save GeoTIFF
        tif_path = os.path.join(args.output, "maxent_suitability_raster.tif")
        export_geotiff(suitability_grid, extent, out_path=tif_path)
        print(f"[Saved] GeoTIFF (.tif):       {tif_path}")

        # Save Interactive Web Map
        html_path = os.path.join(args.output, "interactive_suitability_map.html")
        create_interactive_map(
            presence_df=pres_df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}),
            background_df=bg_df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}) if bg_df is not None and 'latitude' in bg_df.columns else None,
            suitability_grid=suitability_grid,
            extent=extent,
            out_html=html_path
        )
        print(f"[Saved] Interactive Web Map:  {html_path}")

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
