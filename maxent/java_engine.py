"""
java_engine.py - Official MaxEnt Java Binary (maxent.jar) Integration Engine.

Provides automated bridge, data prep, headless execution, live logging,
and result parsing for the official AMNH MaxEnt software (Phillips et al.).
"""

import os
import sys
import subprocess
import shutil
import re
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_MAXENT_JAR_LOCATIONS = [
    r"c:\Users\DuDu-PC\Downloads\maxent\maxent.jar",
    os.path.join(os.path.expanduser("~"), "Downloads", "maxent", "maxent.jar"),
    os.path.join(os.path.dirname(__file__), "..", "maxent.jar"),
]


def find_maxent_jar(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locate the official maxent.jar file on disk.
    Checks custom path, sibling/parent project directories, downloads, and known installation folders.
    """
    if custom_path and os.path.isfile(custom_path) and custom_path.lower().endswith(".jar"):
        return os.path.abspath(custom_path)

    app_root = Path(__file__).resolve().parent.parent
    parent_dir = app_root.parent

    candidate_locations = [
        # Sibling folders (e.g. maxent_studio_pro alongside maxent_presence_model)
        parent_dir / "maxent_studio_pro" / "maxent.jar",
        parent_dir / "maxent" / "maxent.jar",
        parent_dir / "maxent.jar",
        app_root / "maxent.jar",
        # Current working directory and its parents/siblings
        Path.cwd() / "maxent.jar",
        Path.cwd() / "maxent_studio_pro" / "maxent.jar",
        Path.cwd().parent / "maxent_studio_pro" / "maxent.jar",
        Path.cwd().parent / "maxent" / "maxent.jar",
        # Known drive locations
        Path(r"G:\Updated Profile KSC\Maxent Upgraded Application\maxent_studio_pro\maxent.jar"),
        Path(r"G:\Updated Profile KSC\Maxent Upgraded Application\maxent_presence_model\maxent.jar"),
        Path(r"c:\Users\DuDu-PC\Downloads\maxent\maxent.jar"),
        Path(os.path.expanduser("~")) / "Downloads" / "maxent" / "maxent.jar",
        Path(os.path.expanduser("~")) / "Downloads" / "maxent.jar",
    ]

    for cand in candidate_locations:
        if cand.is_file():
            return str(cand.resolve())

    # Shallow search in immediate workspace and parent folder
    for search_root in [app_root, parent_dir, Path.cwd(), Path.cwd().parent]:
        if search_root.exists():
            try:
                for p in search_root.glob("**/maxent.jar"):
                    if p.is_file():
                        return str(p.resolve())
            except Exception:
                pass

    return None


def check_java_runtime() -> Dict[str, Any]:
    """
    Check if Java is installed and query its version and 64-bit status.
    """
    try:
        res = subprocess.run(
            ["java", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        output = res.stderr or res.stdout
        # Version line typically: java version "26.0.2" ... or openjdk version "17.0.2"
        v_match = re.search(r'version\s+"([^"]+)"', output)
        version_str = v_match.group(1) if v_match else "Unknown"
        is_64bit = "64-Bit" in output or "x86_64" in output or "amd64" in output

        return {
            "available": True,
            "version": version_str,
            "is_64bit": is_64bit,
            "details": output.strip().splitlines()[0] if output else "Java detected",
            "raw": output
        }
    except FileNotFoundError:
        return {
            "available": False,
            "version": None,
            "is_64bit": False,
            "details": "Java is not found in system PATH.",
            "raw": ""
        }
    except Exception as e:
        return {
            "available": False,
            "version": None,
            "is_64bit": False,
            "details": f"Error querying Java: {str(e)}",
            "raw": str(e)
        }


def export_rasters_to_ascii_dir(
    raster_dict: Dict[str, np.ndarray],
    extent: List[float],
    out_dir: str,
    nodata_val: float = -9999.0
) -> List[str]:
    """
    Export in-memory 2D raster arrays to ESRI ASCII grid (.asc) files.
    Guarantees 100% identical headers (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value)
    so that maxent.jar will never crash due to extent/cell size discrepancies.
    """
    os.makedirs(out_dir, exist_ok=True)
    first_arr = next(iter(raster_dict.values()))
    nrows, ncols = first_arr.shape
    min_lon, max_lon, min_lat, max_lat = extent

    cellsize_x = (max_lon - min_lon) / float(ncols)
    cellsize_y = (max_lat - min_lat) / float(nrows)
    # MaxEnt requires square pixels; use average if tiny discrepancy
    cellsize = (cellsize_x + cellsize_y) / 2.0

    header = (
        f"ncols         {ncols}\n"
        f"nrows         {nrows}\n"
        f"xllcorner     {min_lon:.10f}\n"
        f"yllcorner     {min_lat:.10f}\n"
        f"cellsize      {cellsize:.10f}\n"
        f"NODATA_value  {nodata_val:.1f}\n"
    )

    created_paths = []
    for var_name, arr in raster_dict.items():
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', var_name).strip('_')
        asc_path = os.path.join(out_dir, f"{clean_name}.asc")
        
        # Replace NaNs with nodata sentinel
        data_clean = np.where(np.isnan(arr), nodata_val, arr)

        with open(asc_path, "w", encoding="ascii") as f:
            f.write(header)
            # Write rows
            for row in data_clean:
                row_str = " ".join(f"{v:.6g}" for v in row)
                f.write(row_str + "\n")

        created_paths.append(asc_path)

    return created_paths


def prepare_samples_csv(
    presence_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    species_name: str,
    out_csv_path: str
) -> pd.DataFrame:
    """
    Format presence occurrences into exact CSV structure required by maxent.jar:
    Header: species, longitude, latitude
    """
    clean_species = re.sub(r'[^\w\s-]', '', str(species_name)).strip().replace(" ", "_")
    if not clean_species:
        clean_species = "Target_Species"

    df_out = pd.DataFrame({
        "species": clean_species,
        "longitude": pd.to_numeric(presence_df[lon_col], errors="coerce"),
        "latitude": pd.to_numeric(presence_df[lat_col], errors="coerce")
    }).dropna()

    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)
    df_out.to_csv(out_csv_path, index=False)
    return df_out


def read_ascii_grid(asc_path: str) -> Tuple[np.ndarray, List[float], float]:
    """
    Parse an ESRI ASCII grid (.asc) file into a numpy array, bounding extent, and nodata value.
    """
    header_info = {}
    with open(asc_path, "r", encoding="ascii") as f:
        # Read header lines
        for _ in range(6):
            line = f.readline().strip()
            parts = line.split()
            if len(parts) >= 2:
                header_info[parts[0].lower()] = float(parts[1])

        ncols = int(header_info.get("ncols", 0))
        nrows = int(header_info.get("nrows", 0))
        xll = header_info.get("xllcorner", header_info.get("xllcenter", 0.0))
        yll = header_info.get("yllcorner", header_info.get("yllcenter", 0.0))
        cellsize = header_info.get("cellsize", 1.0)
        nodata = header_info.get("nodata_value", -9999.0)

        # Read grid data
        data_text = f.read()
        tokens = data_text.split()
        if len(tokens) != nrows * ncols:
            # Fallback line-by-line
            grid = np.array([float(x) for x in tokens]).reshape((nrows, ncols))
        else:
            grid = np.fromiter(tokens, dtype=np.float32).reshape((nrows, ncols))

    extent = [xll, xll + ncols * cellsize, yll, yll + nrows * cellsize]
    # Replace nodata with np.nan
    grid_masked = np.where(np.isclose(grid, nodata) | (grid <= -9990.0), np.nan, grid)
    return grid_masked, extent, nodata


class MaxEntJarRunner:
    """
    Orchestrates execution of official AMNH MaxEnt (maxent.jar) in batch mode.
    """

    def __init__(self, jar_path: Optional[str] = None, memory_mb: int = 2048):
        self.jar_path = find_maxent_jar(jar_path)
        if not self.jar_path or not os.path.isfile(self.jar_path):
            raise FileNotFoundError(
                f"Could not locate maxent.jar. Searched standard paths:\n"
                f"{chr(10).join(DEFAULT_MAXENT_JAR_LOCATIONS)}"
            )
        self.memory_mb = max(512, memory_mb)

    def run(
        self,
        raster_dict: Dict[str, np.ndarray],
        extent: List[float],
        presence_df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        species_name: str = "Target_Species",
        output_format: str = "cloglog",
        reg_mult: float = 1.0,
        feature_types: Optional[List[str]] = None,
        jackknife: bool = True,
        response_curves: bool = True,
        replicates: int = 1,
        replicate_type: str = "crossvalidate",
        random_seed: bool = True,
        background_points: int = 10000,
        temp_work_dir: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute official MaxEnt with modern Java runtime parameters.
        
        Returns parsed results dictionary containing:
        - suitability_grid: 2D numpy array
        - extent: bounding box
        - metrics: dictionary of model evaluation metrics (AUC, gain, thresholds)
        - variable_contributions: DataFrame with variable percent contribution & permutation importance
        - sample_predictions: DataFrame of predictions at presence sites
        - plots: dict of generated plot image paths (PNGs)
        - raw_output_dir: directory containing all outputs
        """
        # Create work directories
        base_dir = temp_work_dir or tempfile.mkdtemp(prefix="maxent_jar_run_")
        layers_dir = os.path.join(base_dir, "layers")
        out_dir = os.path.join(base_dir, "output")
        samples_csv = os.path.join(base_dir, "samples.csv")
        os.makedirs(layers_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        if log_callback:
            log_callback(f"📦 Exporting {len(raster_dict)} aligned environmental layers to ESRI ASCII grids...")

        # 1. Export Rasters to ASCII
        export_rasters_to_ascii_dir(raster_dict, extent, layers_dir)

        # 2. Prepare Samples CSV
        clean_species = re.sub(r'[^\w\s-]', '', str(species_name)).strip().replace(" ", "_")
        if not clean_species:
            clean_species = "Target_Species"
        prepare_samples_csv(presence_df, lat_col, lon_col, clean_species, samples_csv)

        # 3. Build Command Line Flags
        # Use modern -Xmx syntax (avoids deprecated -mx warning in Java 9+)
        cmd = [
            "java",
            f"-Xmx{self.memory_mb}m",
            "-jar", self.jar_path,
            f"environmentallayers={layers_dir}",
            f"samplesfile={samples_csv}",
            f"outputdirectory={out_dir}",
            "autorun",
            "novisible",
            "noaskoverwrite",
            "-r",
            f"outputformat={output_format}",
            f"betamultiplier={reg_mult:.4f}",
            f"maximumbackground={background_points}",
            "responsecurves" if response_curves else "noresponsecurves",
            "pictures",
            "jackknife" if jackknife else "nojackknife",
            "randomseed" if random_seed else "norandomseed",
            "writeplotdata",
            "threads=4"
        ]

        # Feature type constraints: in MaxEnt Java, boolean flags take prefixes 'no' or 'dont'
        # The auto-feature flag in maxent.jar is 'autofeature' (singular)
        if feature_types:
            f_set = {f.lower() for f in feature_types}
            cmd.append("noautofeature")
            cmd.append("linear" if "linear" in f_set else "nolinear")
            cmd.append("quadratic" if "quadratic" in f_set else "noquadratic")
            cmd.append("product" if "product" in f_set else "noproduct")
            cmd.append("threshold" if "threshold" in f_set else "nothreshold")
            cmd.append("hinge" if "hinge" in f_set else "nohinge")
        else:
            cmd.append("autofeature")

        # Replications (if > 1)
        if replicates > 1:
            cmd.append(f"replicates={replicates}")
            cmd.append(f"replicatetype={replicate_type}")

        if log_callback:
            log_callback(f"🚀 Launching Official MaxEnt (AMNH v3.4.4 Java Engine)...")
            log_callback(f"Command: {' '.join(cmd)}")

        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=base_dir
        )

        log_lines = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                line_str = line.strip()
                if line_str:
                    log_lines.append(line_str)
                    if log_callback:
                        log_callback(f"[MaxEnt.jar] {line_str}")

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode != 0:
            error_details = "\n".join(log_lines[-20:])
            raise RuntimeError(
                f"Official MaxEnt (maxent.jar) failed with exit code {process.returncode}.\n"
                f"Recent output:\n{error_details}"
            )

        if log_callback:
            log_callback(f"✅ MaxEnt Java Engine finished successfully in {elapsed:.2f}s!")
            log_callback(f"📊 Parsing output grids, statistics, and response plots...")

        # 4. Parse Results
        # Target species prediction grid (.asc)
        asc_pred_path = os.path.join(out_dir, f"{clean_species}.asc")
        if not os.path.isfile(asc_pred_path):
            # Check for any .asc file in output
            asc_files = list(Path(out_dir).glob("*.asc"))
            if asc_files:
                asc_pred_path = str(asc_files[0])
            else:
                raise FileNotFoundError(f"Prediction raster {clean_species}.asc was not generated by MaxEnt.")

        suitability_grid, pred_extent, nodata_val = read_ascii_grid(asc_pred_path)

        # Parse maxentResults.csv
        results_csv = os.path.join(out_dir, "maxentResults.csv")
        results_df = pd.read_csv(results_csv) if os.path.isfile(results_csv) else pd.DataFrame()

        metrics = {}
        var_importance = pd.DataFrame()
        thresholds = {}

        if not results_df.empty:
            row = results_df.iloc[0].to_dict()
            metrics = {
                "training_samples": row.get("#Training samples", len(presence_df)),
                "training_gain": row.get("Regularized training gain", 0.0),
                "unregularized_gain": row.get("Unregularized training gain", 0.0),
                "auc": row.get("Training AUC", 0.0),
                "iterations": row.get("Iterations", 0),
                "background_points": row.get("#Background points", background_points),
                "entropy": row.get("Entropy", 0.0),
                "prevalence": row.get("Prevalence (average probability of presence over background sites)", 0.5)
            }

            # Parse Variable Contributions & Permutation Importance
            vars_list = list(raster_dict.keys())
            var_records = []
            for v in vars_list:
                clean_v = re.sub(r'[^a-zA-Z0-9_]', '_', v).strip('_')
                contrib_key = f"{clean_v} contribution"
                perm_key = f"{clean_v} permutation importance"
                contrib = float(row.get(contrib_key, 0.0))
                perm = float(row.get(perm_key, 0.0))
                var_records.append({
                    "Variable": v,
                    "Percent_Contribution": contrib,
                    "Permutation_Importance": perm
                })
            var_importance = pd.DataFrame(var_records).sort_values("Percent_Contribution", ascending=False)

            # Thresholds
            th_cols = [
                ("Minimum training presence", "Minimum training presence Cloglog threshold"),
                ("10 percentile training presence", "10 percentile training presence Cloglog threshold"),
                ("Equal training sensitivity and specificity", "Equal training sensitivity and specificity Cloglog threshold"),
                ("Maximum training sensitivity plus specificity", "Maximum training sensitivity plus specificity Cloglog threshold"),
                ("Fixed cumulative value 10", "Fixed cumulative value 10 Cloglog threshold")
            ]
            for label, col_name in th_cols:
                if col_name in row:
                    thresholds[label] = float(row[col_name])

            # Select primary decision cutoff
            metrics["threshold"] = thresholds.get(
                "Maximum training sensitivity plus specificity",
                thresholds.get("10 percentile training presence", 0.5)
            )

        # Sample predictions
        sample_pred_csv = os.path.join(out_dir, f"{clean_species}_samplePredictions.csv")
        sample_preds_df = pd.read_csv(sample_pred_csv) if os.path.isfile(sample_pred_csv) else pd.DataFrame()

        # Omission curve and ROC data
        omission_csv = os.path.join(out_dir, f"{clean_species}_omission.csv")
        omission_df = pd.read_csv(omission_csv) if os.path.isfile(omission_csv) else pd.DataFrame()

        roc_data = None
        if not omission_df.empty and "Fractional area" in omission_df.columns and "Training omission" in omission_df.columns:
            # Sort by fractional area ascending
            sorted_om = omission_df.sort_values("Fractional area")
            fpr = sorted_om["Fractional area"].values
            tpr = 1.0 - sorted_om["Training omission"].values
            th_col = "Corresponding Cloglog value" if "Corresponding Cloglog value" in sorted_om.columns else sorted_om.columns[0]
            cutoffs = sorted_om[th_col].values
            roc_data = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": cutoffs,
                "auc": metrics.get("auc", 0.5)
            }

        # Collect plots and response curve .dat numeric tables
        plots_dir = os.path.join(out_dir, "plots")
        plots = {}
        response_curve_data = {}

        if os.path.isdir(plots_dir):
            for p in Path(plots_dir).glob("*.png"):
                if not p.name.endswith("_thumb.png"):
                    plots[p.stem] = str(p)
            for d in Path(plots_dir).glob("*.dat"):
                try:
                    df_dat = pd.read_csv(d)
                    var_key = d.stem.replace(f"{clean_species}_", "")
                    response_curve_data[var_key] = df_dat
                except Exception:
                    pass

        # Overview picture
        overview_png = os.path.join(out_dir, f"{clean_species}.png")
        if os.path.isfile(overview_png):
            plots["overview_map"] = overview_png

        # Lambdas file
        lambdas_path = os.path.join(out_dir, f"{clean_species}.lambdas")

        return {
            "engine": "Official MaxEnt (maxent.jar v3.4.4)",
            "species_name": clean_species,
            "suitability_grid": suitability_grid,
            "extent": pred_extent,
            "metrics": metrics,
            "thresholds": thresholds,
            "variable_importance": var_importance,
            "sample_predictions": sample_preds_df,
            "omission_curve": omission_df,
            "roc_data": roc_data,
            "response_curve_data": response_curve_data,
            "plots": plots,
            "html_report": os.path.join(out_dir, f"{clean_species}.html"),
            "lambdas_path": lambdas_path if os.path.isfile(lambdas_path) else None,
            "output_dir": out_dir,
            "elapsed_seconds": elapsed,
            "log": "\n".join(log_lines)
        }
