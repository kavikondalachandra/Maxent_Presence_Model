"""
plotly_viz.py - Interactive Plotly Diagnostics and Response Curve Charts.
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_plotly_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_val: float,
    thresholds: Optional[np.ndarray] = None,
    opt_threshold: Optional[float] = None
) -> go.Figure:
    """
    Create an interactive Plotly ROC curve with hover details and optimal threshold marker.
    """
    fig = go.Figure()

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='#888888', dash='dash', width=1.5),
        name='Random Baseline (AUC = 0.500)',
        hoverinfo='skip'
    ))

    # ROC curve
    hover_text = []
    if thresholds is not None:
        for f, t, th in zip(fpr, tpr, thresholds):
            hover_text.append(f"FPR: {f:.3f}<br>TPR: {t:.3f}<br>Cutoff: {th:.3f}")
    else:
        for f, t in zip(fpr, tpr):
            hover_text.append(f"FPR: {f:.3f}<br>TPR: {t:.3f}")

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name=f"MaxEnt ROC (AUC = {auc_val:.3f})",
        text=hover_text,
        hoverinfo='text',
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))

    # Optimal threshold marker
    if opt_threshold is not None and thresholds is not None:
        idx = np.argmin(np.abs(thresholds - opt_threshold))
        fig.add_trace(go.Scatter(
            x=[fpr[idx]],
            y=[tpr[idx]],
            mode='markers',
            marker=dict(color='#d62728', size=11, symbol='circle', line=dict(color='white', width=2)),
            name=f"Optimal Cutoff ({opt_threshold:.3f})",
            hoverinfo='text',
            hovertext=f"Optimal Cutoff: {opt_threshold:.3f}<br>TPR (Sens): {tpr[idx]:.3f}<br>FPR (1-Spec): {fpr[idx]:.3f}"
        ))

    fig.update_layout(
        title=f"<b>Receiver Operating Characteristic (ROC) &mdash; AUC: {auc_val:.3f}</b>",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02], zeroline=False),
        yaxis=dict(title="True Positive Rate (Sensitivity)", range=[-0.02, 1.02], zeroline=False),
        template="plotly_white",
        legend=dict(x=0.55, y=0.1, bgcolor='rgba(255,255,255,0.85)'),
        margin=dict(l=50, r=40, t=50, b=50),
        height=450
    )
    return fig


def plot_plotly_variable_importance(imp_df: pd.DataFrame) -> go.Figure:
    """
    Interactive horizontal grouped bar chart showing Percent Contribution and Permutation Importance.
    """
    df = imp_df.copy()
    if "Variable" not in df.columns:
        return go.Figure()

    df = df.sort_values(
        by=["Percent_Contribution" if "Percent_Contribution" in df.columns else df.columns[1]],
        ascending=True
    )

    fig = go.Figure()

    if "Percent_Contribution" in df.columns:
        fig.add_trace(go.Bar(
            y=df["Variable"],
            x=df["Percent_Contribution"],
            name="Percent Contribution (%)",
            orientation='h',
            marker=dict(color='#2ca02c', line=dict(color='#1b611b', width=1)),
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}%<extra></extra>"
        ))

    if "Permutation_Importance" in df.columns:
        fig.add_trace(go.Bar(
            y=df["Variable"],
            x=df["Permutation_Importance"],
            name="Permutation Importance (%)",
            orientation='h',
            marker=dict(color='#1f77b4', line=dict(color='#12466b', width=1)),
            hovertemplate="<b>%{y}</b><br>Permutation Importance: %{x:.2f}%<extra></extra>"
        ))

    fig.update_layout(
        title="<b>Environmental Variable Importance & Contribution</b>",
        barmode='group',
        xaxis=dict(title="Importance (%)", zeroline=False),
        yaxis=dict(title="Environmental Predictor"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=50),
        height=400
    )
    return fig


def plot_plotly_response_curve(
    var_name: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    unit: str = "",
    y_label: str = "Suitability (Cloglog)"
) -> go.Figure:
    """
    Interactive marginal response curve plot for a single environmental variable.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name=var_name,
        hovertemplate=f"<b>{var_name}</b>: %{{x:.2f}} {unit}<br>{y_label}: %{{y:.3f}}<extra></extra>",
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.12)'
    ))

    fig.update_layout(
        title=f"<b>Response Curve: {var_name}</b>",
        xaxis=dict(title=f"{var_name} {unit}".strip(), zeroline=False),
        yaxis=dict(title=y_label, range=[-0.02, 1.02], zeroline=False),
        template="plotly_white",
        margin=dict(l=50, r=40, t=50, b=50),
        height=380
    )
    return fig
