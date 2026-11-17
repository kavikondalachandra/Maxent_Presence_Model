"""
demo.py - Comprehensive End-to-End Maximum Entropy Presence Prediction Demonstration.

This script demonstrates:
1. Generating a realistic bioclimatic landscape (Temperature, Precipitation, Elevation, Canopy Cover).
2. Simulating species occurrences with true ecological constraints.
3. Training the MaxEnt model with L1 regularization and hinge/product feature expansions.
4. Evaluating predictive performance on independent held-out test data (AUC, Boyce Index, TSS).
5. Computing variable permutation importance and marginal response curves.
6. Generating 2D spatial suitability maps and an interactive Leaflet/Folium web map.
7. Saving publication-ready figures and summary reports.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure maxent package can be imported from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maxent import (
    MaxEnt,
    generate_synthetic_landscape,
    plot_model_diagnostics,
    plot_response_curves,
    plot_spatial_prediction,
    create_interactive_map
)


def run_demo():
    print("=" * 70)
    print("      MAXIMUM ENTROPY (MaxEnt) PRESENCE PREDICTION MODEL")
    print("=" * 70)
    print("Step 1: Simulating realistic geographic landscape & species occurrences...")
    
    data = generate_synthetic_landscape(
        width=120,
        height=120,
        n_presence=200,
        n_background=3000,
        seed=42
    )

    presence_df = data['presence_df']
    background_df = data['background_df']
    raster_dict = data['raster_dict']
    features = data['feature_names']
    extent = data['extent']

    print(f"  • Landscape dimensions:     {data['shape'][0]} x {data['shape'][1]} cells")
    print(f"  • Environmental predictors: {', '.join(features)}")
    print(f"  • Total presence points:    {len(presence_df)}")
    print(f"  • Total background samples: {len(background_df)}")
    print()

    # Split presence and background into Train (75%) and Test (25%)
    pres_train, pres_test = train_test_split(presence_df, test_size=0.25, random_state=42)
    bg_train, bg_test = train_test_split(background_df, test_size=0.25, random_state=42)

    X_pres_train = pres_train[features]
    X_pres_test = pres_test[features]
    X_bg_train = bg_train[features]
    X_bg_test = bg_test[features]

    print("Step 2: Initializing and fitting MaxEnt model...")
    print("  • Feature types: Linear, Quadratic, Product, and Hinge")
    print("  • Regularization multiplier: 1.0 (L1 Lasso penalty)")
    
    model = MaxEnt(
        reg_mult=1.0,
        feature_types=['linear', 'quadratic', 'product', 'hinge'],
        hinge_knots=5,
        default_output='cloglog',
        random_state=42,
        verbose=False
    )

    model.fit(X_pres_train, X_bg_train, feature_names=features)
    print("  • Model optimization converged successfully!\n")

    # Display training summary
    print(model.summary())
    print()

    # Step 3: Independent Test Evaluation
    print("Step 3: Evaluating model on held-out test dataset...")
    test_metrics = model.evaluate(X_pres_test, X_bg_test)
    print("-" * 70)
    print("INDEPENDENT TEST EVALUATION RESULTS:")
    print(f"  • Test ROC AUC:                 {test_metrics['auc']:.4f}")
    print(f"  • Continuous Boyce Index (CBI): {test_metrics['boyce_index']:.4f}")
    print(f"  • Optimal Decision Threshold:   {test_metrics['threshold']:.4f}")
    print(f"  • True Skill Statistic (TSS):   {test_metrics['tss']:.4f}")
    print(f"  • Sensitivity (True Positive):  {test_metrics['sensitivity']:.4f}")
    print(f"  • Specificity (True Negative):  {test_metrics['specificity']:.4f}")
    print(f"  • F1 Score:                     {test_metrics['f1_score']:.4f}")
    print("-" * 70)
    print()

    # Step 4: Variable Importance
    print("Step 4: Computing Variable Importance (Permutation & Contribution)...")
    imp_df = model.get_feature_importance(X_pres_test, X_bg_test, n_repeats=5)
    print(imp_df.to_string(index=False))
    print()

    # Step 5: Spatial Prediction across entire landscape
    print("Step 5: Predicting spatial habitat suitability across the study region...")
    suitability_grid = model.predict_raster(raster_dict, output_type='cloglog')
    print(f"  • Suitability map generated: Min={np.nanmin(suitability_grid):.3f}, "
          f"Mean={np.nanmean(suitability_grid):.3f}, Max={np.nanmax(suitability_grid):.3f}")
    print()

    # Step 6: Generate and Save Visualizations
    out_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(out_dir, "outputs")
    os.makedirs(output_folder, exist_ok=True)

    print("Step 6: Generating publication-quality charts and interactive maps...")
    
    # 1. Diagnostic 4-panel figure
    diag_path = os.path.join(output_folder, "model_diagnostics.png")
    plot_model_diagnostics(model, X_pres_test, X_bg_test, save_path=diag_path)
    print(f"  [Saved] Diagnostic Dashboard: {diag_path}")

    # 2. Environmental Response Curves
    resp_path = os.path.join(output_folder, "response_curves.png")
    plot_response_curves(model, save_path=resp_path)
    print(f"  [Saved] Response Curves:      {resp_path}")

    # 3. Spatial Suitability Map
    map_path = os.path.join(output_folder, "spatial_suitability_map.png")
    plot_spatial_prediction(
        suitability_grid,
        extent=extent,
        presence_coords=presence_df,
        threshold=test_metrics['threshold'],
        title="Predicted Species Habitat Suitability & Occurrences",
        save_path=map_path
    )
    print(f"  [Saved] Spatial Map:          {map_path}")

    # 4. Interactive Folium HTML Map
    html_path = os.path.join(output_folder, "interactive_map.html")
    create_interactive_map(
        presence_df=presence_df,
        background_df=background_df,
        suitability_grid=suitability_grid,
        extent=extent,
        out_html=html_path
    )
    print(f"  [Saved] Interactive HTML Map: {html_path}")

    # 5. Export Predictions CSV
    pres_pred = pres_test.copy()
    pres_pred['predicted_suitability'] = model.predict(X_pres_test, output_type='cloglog')
    csv_path = os.path.join(output_folder, "test_presence_predictions.csv")
    pres_pred.to_csv(csv_path, index=False)
    print(f"  [Saved] Test Predictions CSV: {csv_path}")

    print("\n" + "=" * 70)
    print("DEMO EXECUTION COMPLETE!")
    print(f"All outputs have been successfully written to: {output_folder}")
    print("=" * 70)


if __name__ == '__main__':
    run_demo()
