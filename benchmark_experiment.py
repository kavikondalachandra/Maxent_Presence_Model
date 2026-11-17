"""
benchmark_experiment.py - Comparative Benchmarking Suite for MaxEnt Models.

Compares:
1. Native Python MaxEnt Engine (Scikit-Learn compatible dual L-BFGS)
2. Official AMNH MaxEnt Engine (maxent.jar v3.4.4 headless batch execution)

Measures:
- Wall-clock Training & Prediction Runtime (s)
- Peak Memory Allocation (MB)
- Test ROC Area Under Curve (AUC)
- Continuous Boyce Index (CBI)
- True Skill Statistic (TSS)
- Inter-Engine Spatial Suitability Correlation (Spearman & Pearson)

Exports results to:
- benchmark_results.csv
- benchmark_table.md (Markdown for GitHub / papers)
- benchmark_table.tex (LaTeX tabular for SoftwareX / Elsevier)
"""

import os
import sys
import time
import tracemalloc
import tempfile
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

# Ensure maxent package can be imported from current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from maxent import (
    MaxEnt,
    generate_synthetic_landscape,
    calculate_continuous_boyce_index,
    find_optimal_thresholds
)
from maxent.java_engine import MaxEntJarRunner, find_maxent_jar, check_java_runtime


def run_benchmark_regime(
    regime_name: str,
    n_presence: int,
    n_background: int,
    grid_size: int = 100,
    reg_mult: float = 1.0,
    seed: int = 42,
    jar_path: str = None
) -> dict:
    """
    Execute a single benchmark regime for both Python and Java MaxEnt engines.
    """
    print(f"\n{'='*75}")
    print(f"Running Regime: {regime_name} (Presences={n_presence}, Background={n_background}, Grid={grid_size}x{grid_size})")
    print(f"{'='*75}")

    # 1. Generate Synthetic Landscape with ground-truth environmental response
    landscape = generate_synthetic_landscape(
        width=grid_size,
        height=grid_size,
        n_presence=n_presence,
        n_background=n_background,
        seed=seed
    )

    presence_df = landscape['presence_df']
    background_df = landscape['background_df']
    raster_dict = landscape['raster_dict']
    features = landscape['feature_names']
    extent = landscape['extent']

    # 75% Train / 25% Test split
    pres_train, pres_test = train_test_split(presence_df, test_size=0.25, random_state=seed)
    bg_train, bg_test = train_test_split(background_df, test_size=0.25, random_state=seed)

    X_pres_train = pres_train[features]
    X_pres_test = pres_test[features]
    X_bg_train = bg_train[features]
    X_bg_test = bg_test[features]

    results = {
        "regime": regime_name,
        "n_presence_train": len(pres_train),
        "n_presence_test": len(pres_test),
        "n_background": n_background,
        "grid_cells": grid_size * grid_size
    }

    # -------------------------------------------------------------
    # A. Native Python MaxEnt Engine
    # -------------------------------------------------------------
    print("\n[1/2] Benchmarking Native Python MaxEnt Engine...")
    tracemalloc.start()
    t0_py = time.perf_counter()

    py_model = MaxEnt(
        reg_mult=reg_mult,
        feature_types=['linear', 'quadratic', 'product', 'hinge'],
        hinge_knots=5,
        default_output='cloglog',
        random_state=seed,
        verbose=False
    )
    py_model.fit(X_pres_train, X_bg_train, feature_names=features)
    t1_py = time.perf_counter()

    # Spatial Landscape Prediction
    py_suitability_grid = py_model.predict_raster(raster_dict, output_type='cloglog')
    t2_py = time.perf_counter()

    current_mem, peak_mem_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    py_train_time = t1_py - t0_py
    py_spatial_time = t2_py - t1_py
    py_total_time = t2_py - t0_py
    py_peak_mb = peak_mem_py / (1024 * 1024)

    # Evaluation on Test Data
    py_test_eval = py_model.evaluate(X_pres_test, X_bg_test)

    print(f"  -> Python Total Time: {py_total_time:.3f}s (Fit: {py_train_time:.3f}s, Grid: {py_spatial_time:.3f}s)")
    print(f"  -> Python Peak RAM:   {py_peak_mb:.2f} MB")
    print(f"  -> Test AUC:          {py_test_eval['auc']:.4f}")
    print(f"  -> Test Boyce Index:  {py_test_eval['boyce_index']:.4f}")
    print(f"  -> Test TSS:          {py_test_eval['tss']:.4f}")

    results.update({
        "py_fit_sec": round(py_train_time, 3),
        "py_spatial_sec": round(py_spatial_time, 3),
        "py_total_sec": round(py_total_time, 3),
        "py_peak_mb": round(py_peak_mb, 2),
        "py_auc": round(py_test_eval['auc'], 4),
        "py_cbi": round(py_test_eval['boyce_index'], 4),
        "py_tss": round(py_test_eval['tss'], 4),
    })

    # -------------------------------------------------------------
    # B. Official AMNH Java MaxEnt Engine (maxent.jar)
    # -------------------------------------------------------------
    if jar_path and os.path.isfile(jar_path):
        print("\n[2/2] Benchmarking Official AMNH Java MaxEnt (maxent.jar)...")
        java_runner = MaxEntJarRunner(jar_path=jar_path, memory_mb=2048)
        
        tracemalloc.start()
        t0_java = time.perf_counter()
        
        temp_dir = tempfile.mkdtemp(prefix="maxent_bench_")
        try:
            java_out = java_runner.run(
                raster_dict=raster_dict,
                extent=extent,
                presence_df=pres_train,
                lat_col='latitude',
                lon_col='longitude',
                species_name="Benchmark_Taxon",
                reg_mult=reg_mult,
                feature_types=['linear', 'quadratic', 'product', 'hinge'],
                output_format='cloglog',
                background_points=min(n_background, 10000),
                random_seed=True,
                response_curves=False,
                jackknife=False,
                temp_work_dir=temp_dir
            )
            t1_java = time.perf_counter()
            _, peak_mem_java = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            java_total_time = t1_java - t0_java
            java_peak_mb = peak_mem_java / (1024 * 1024)

            java_grid = java_out['suitability_grid']
            java_metrics = java_out.get('metrics', {})
            java_auc = java_metrics.get('training_auc', np.nan)

            # Compute CBI and test AUC on Java predictions
            # Extract suitability from Java grid at test presence and test background points
            test_lats = pres_test['latitude'].to_numpy()
            test_lons = pres_test['longitude'].to_numpy()
            nrows, ncols = java_grid.shape
            min_lon, max_lon, min_lat, max_lat = extent

            def coord_to_pix(lats, lons):
                r = np.clip(np.round(((max_lat - lats) / (max_lat - min_lat)) * (nrows - 1)).astype(int), 0, nrows - 1)
                c = np.clip(np.round(((lons - min_lon) / (max_lon - min_lon)) * (ncols - 1)).astype(int), 0, ncols - 1)
                return r, c

            pr_rows, pr_cols = coord_to_pix(test_lats, test_lons)
            java_pres_test_scores = java_grid[pr_rows, pr_cols]

            bg_rows, bg_cols = coord_to_pix(bg_test['latitude'].to_numpy(), bg_test['longitude'].to_numpy())
            java_bg_test_scores = java_grid[bg_rows, bg_cols]

            # Remove any NaNs
            valid_p = ~np.isnan(java_pres_test_scores)
            valid_b = ~np.isnan(java_bg_test_scores)
            clean_jp = java_pres_test_scores[valid_p]
            clean_jb = java_bg_test_scores[valid_b]

            from sklearn.metrics import roc_auc_score
            y_true = np.concatenate([np.ones_like(clean_jp), np.zeros_like(clean_jb)])
            y_pred = np.concatenate([clean_jp, clean_jb])
            java_test_auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
            java_cbi = calculate_continuous_boyce_index(clean_jp, clean_jb)

            # Correlation between Python grid and Java grid
            valid_mask = (~np.isnan(py_suitability_grid)) & (~np.isnan(java_grid))
            spear_corr, _ = spearmanr(py_suitability_grid[valid_mask].ravel(), java_grid[valid_mask].ravel())
            pear_corr, _ = pearsonr(py_suitability_grid[valid_mask].ravel(), java_grid[valid_mask].ravel())

            print(f"  -> Java Total Time:  {java_total_time:.3f}s (Export + Process + Parse)")
            print(f"  -> Java Test AUC:    {java_test_auc:.4f}")
            print(f"  -> Java Test CBI:    {java_cbi:.4f}")
            print(f"  -> Spatial Agreement with Python Engine:")
            print(f"     * Spearman Rank Correlation: r_s = {spear_corr:.4f}")
            print(f"     * Pearson Linear Correlation: r  = {pear_corr:.4f}")

            results.update({
                "java_total_sec": round(java_total_time, 3),
                "java_auc": round(java_test_auc, 4),
                "java_cbi": round(java_cbi, 4),
                "inter_engine_spearman": round(spear_corr, 4),
                "inter_engine_pearson": round(pear_corr, 4)
            })

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print("\n[!] AMNH maxent.jar not found or specified; skipping Java run.")
        results.update({
            "java_total_sec": np.nan,
            "java_auc": np.nan,
            "java_cbi": np.nan,
            "inter_engine_spearman": np.nan,
            "inter_engine_pearson": np.nan
        })

    return results


def main():
    print("=" * 80)
    print("       MAXENT STUDIO PRO - BENCHMARK & COMPARATIVE EVALUATION")
    print("=" * 80)

    # Locate official maxent.jar
    jar_path = find_maxent_jar()
    print(f"Java Runtime Status: {check_java_runtime().get('details')}")
    print(f"AMNH maxent.jar:     {jar_path}")

    # Benchmark regimes
    regimes = [
        {"name": "Low Density",    "n_presence": 60,  "n_background": 1200, "grid_size": 80},
        {"name": "Medium Density", "n_presence": 180, "n_background": 3000, "grid_size": 120},
        {"name": "High Density",   "n_presence": 400, "n_background": 6000, "grid_size": 150}
    ]

    all_results = []
    for reg in regimes:
        res = run_benchmark_regime(
            regime_name=reg["name"],
            n_presence=reg["n_presence"],
            n_background=reg["n_background"],
            grid_size=reg["grid_size"],
            jar_path=jar_path
        )
        all_results.append(res)

    df_res = pd.DataFrame(all_results)
    csv_path = os.path.join(BASE_DIR, "benchmark_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n[+] Results saved to CSV: {csv_path}")

    # Build Markdown Table
    md_content = "# Empirical Benchmark: Native Python Engine vs. Official AMNH Java Engine\n\n"
    md_content += "| Regime | Presences (Tr/Te) | Background | Python Time (s) | Java Time (s) | Python AUC | Java AUC | Python CBI | Java CBI | Inter-Engine Agreement ($r_s$) |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for r in all_results:
        md_content += (
            f"| **{r['regime']}** | {r['n_presence_train']}/{r['n_presence_test']} | {r['n_background']} | "
            f"{r['py_total_sec']:.2f}s | {r['java_total_sec']:.2f}s | "
            f"{r['py_auc']:.4f} | {r['java_auc']:.4f} | "
            f"{r['py_cbi']:.4f} | {r['java_cbi']:.4f} | "
            f"**{r['inter_engine_spearman']:.4f}** |\n"
        )

    md_path = os.path.join(BASE_DIR, "benchmark_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Markdown table written to: {md_path}")

    # Build LaTeX Table for SoftwareX / Elsevier
    tex_content = r"""\begin{table*}[t]
\centering
\caption{Empirical comparative evaluation of the Native Python MaxEnt Engine vs. the Official AMNH Java Engine (\texttt{maxent.jar} v3.4.4) across three spatial sampling regimes. $r_s$ indicates the Spearman rank correlation of the continuous habitat suitability predictions across all landscape pixels.}
\label{tab:benchmark_comparison}
\begin{tabular}{lccccccccc}
\hline
\textbf{Regime} & \textbf{Presences} & \textbf{Background} & \multicolumn{2}{c}{\textbf{Total Runtime (s)}} & \multicolumn{2}{c}{\textbf{Test ROC AUC}} & \multicolumn{2}{c}{\textbf{Continuous Boyce Index}} & \textbf{Spatial Agreement} \\
\cline{4-5} \cline{6-7} \cline{8-9}
& \textbf{(Train/Test)} & \textbf{Points} & \textbf{Python} & \textbf{Java} & \textbf{Python} & \textbf{Java} & \textbf{Python} & \textbf{Java} & \textbf{Spearman $r_s$} \\
\hline
"""
    for r in all_results:
        tex_content += (
            f"{r['regime']} & {r['n_presence_train']}/{r['n_presence_test']} & {r['n_background']} & "
            f"{r['py_total_sec']:.2f} & {r['java_total_sec']:.2f} & "
            f"{r['py_auc']:.3f} & {r['java_auc']:.3f} & "
            f"{r['py_cbi']:.3f} & {r['java_cbi']:.3f} & "
            f"\\textbf{{{r['inter_engine_spearman']:.3f}}} \\\\\n"
        )
    tex_content += r"""\hline
\end{tabular}
\end{table*}
"""
    tex_path = os.path.join(BASE_DIR, "benchmark_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"[+] LaTeX table written to:    {tex_path}")

    print("\n" + md_content)


if __name__ == "__main__":
    main()
