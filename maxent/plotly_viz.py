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
    opt_threshold: Optional[float] = None,
    template: str = "plotly_white"
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

    bg_legend = 'rgba(255,255,255,0.85)' if 'white' in template else 'rgba(30,41,59,0.85)'
    fig.update_layout(
        title=f"<b>Receiver Operating Characteristic (ROC) &mdash; AUC: {auc_val:.3f}</b>",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02], zeroline=False),
        yaxis=dict(title="True Positive Rate (Sensitivity)", range=[-0.02, 1.02], zeroline=False),
        template=template,
        legend=dict(x=0.55, y=0.1, bgcolor=bg_legend),
        margin=dict(l=50, r=40, t=50, b=50),
        height=450
    )
    return fig


def plot_plotly_dual_roc(
    fpr_java: np.ndarray,
    tpr_java: np.ndarray,
    auc_java: float,
    fpr_py: np.ndarray,
    tpr_py: np.ndarray,
    auc_py: float,
    thresholds_java: Optional[np.ndarray] = None,
    thresholds_py: Optional[np.ndarray] = None,
    opt_th_java: Optional[float] = None,
    opt_th_py: Optional[float] = None,
    template: str = "plotly_white"
) -> go.Figure:
    """
    Create a dual interactive Plotly ROC curve overlaying Official MaxEnt (Java) and Native Python.
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

    # Official MaxEnt (Java)
    hover_java = []
    if thresholds_java is not None:
        for f, t, th in zip(fpr_java, tpr_java, thresholds_java):
            hover_java.append(f"Java &mdash; FPR: {f:.3f}<br>TPR: {t:.3f}<br>Cutoff: {th:.3f}")
    else:
        for f, t in zip(fpr_java, tpr_java):
            hover_java.append(f"Java &mdash; FPR: {f:.3f}<br>TPR: {t:.3f}")

    fig.add_trace(go.Scatter(
        x=fpr_java,
        y=tpr_java,
        mode='lines',
        line=dict(color='#0d9488', width=3),
        name=f"🟢 Official MaxEnt Java (AUC = {auc_java:.3f})",
        text=hover_java,
        hoverinfo='text',
        fill='tozeroy',
        fillcolor='rgba(13, 148, 136, 0.08)'
    ))

    # Native Python MaxEnt
    hover_py = []
    if thresholds_py is not None:
        for f, t, th in zip(fpr_py, tpr_py, thresholds_py):
            hover_py.append(f"Python &mdash; FPR: {f:.3f}<br>TPR: {t:.3f}<br>Cutoff: {th:.3f}")
    else:
        for f, t in zip(fpr_py, tpr_py):
            hover_py.append(f"Python &mdash; FPR: {f:.3f}<br>TPR: {t:.3f}")

    fig.add_trace(go.Scatter(
        x=fpr_py,
        y=tpr_py,
        mode='lines',
        line=dict(color='#3b82f6', width=3),
        name=f"🔵 Native Python Engine (AUC = {auc_py:.3f})",
        text=hover_py,
        hoverinfo='text',
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.08)'
    ))

    # Optimal Cutoff Java
    if opt_th_java is not None and thresholds_java is not None and len(thresholds_java) > 0:
        idx_j = int(np.argmin(np.abs(thresholds_java - opt_th_java)))
        fig.add_trace(go.Scatter(
            x=[fpr_java[idx_j]],
            y=[tpr_java[idx_j]],
            mode='markers',
            marker=dict(color='#0f766e', size=11, symbol='diamond', line=dict(color='white', width=2)),
            name=f"Java Cutoff ({opt_th_java:.3f})",
            hoverinfo='text',
            hovertext=f"Java Optimal Cutoff: {opt_th_java:.3f}<br>Sens: {tpr_java[idx_j]:.3f}<br>Spec: {1.0 - fpr_java[idx_j]:.3f}"
        ))

    # Optimal Cutoff Python
    if opt_th_py is not None and thresholds_py is not None and len(thresholds_py) > 0:
        idx_p = int(np.argmin(np.abs(thresholds_py - opt_th_py)))
        fig.add_trace(go.Scatter(
            x=[fpr_py[idx_p]],
            y=[tpr_py[idx_p]],
            mode='markers',
            marker=dict(color='#1d4ed8', size=11, symbol='circle', line=dict(color='white', width=2)),
            name=f"Python Cutoff ({opt_th_py:.3f})",
            hoverinfo='text',
            hovertext=f"Python Optimal Cutoff: {opt_th_py:.3f}<br>Sens: {tpr_py[idx_p]:.3f}<br>Spec: {1.0 - fpr_py[idx_p]:.3f}"
        ))

    bg_legend = 'rgba(255,255,255,0.7)' if 'white' in template else 'rgba(30,41,59,0.7)'
    fig.update_layout(
        title=f"<b>Comparative ROC Curves: Official Java (AUC: {auc_java:.3f}) vs. Native Python (AUC: {auc_py:.3f})</b>",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02], zeroline=False),
        yaxis=dict(title="True Positive Rate (Sensitivity)", range=[-0.02, 1.02], zeroline=False),
        template=template,
        legend=dict(x=0.45, y=0.15, bgcolor=bg_legend),
        margin=dict(l=50, r=40, t=50, b=50),
        height=470
    )
    return fig


def plot_plotly_variable_importance(
    imp_df: pd.DataFrame,
    template: str = "plotly_white"
) -> go.Figure:
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
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=50),
        height=400
    )
    return fig


def plot_plotly_grouped_variable_importance(
    imp_java: pd.DataFrame,
    imp_py: pd.DataFrame,
    template: str = "plotly_white"
) -> go.Figure:
    """
    Grouped horizontal bar chart comparing variable contributions between Official Java and Native Python.
    """
    df_j = imp_java.copy() if imp_java is not None else pd.DataFrame()
    df_p = imp_py.copy() if imp_py is not None else pd.DataFrame()

    vars_j = list(df_j.get("Variable", []))
    vars_p = list(df_p.get("Variable", []))
    all_vars = list(dict.fromkeys(vars_j + vars_p))
    if not all_vars:
        return go.Figure()

    j_map = dict(zip(df_j.get("Variable", []), df_j.get("Percent_Contribution", [0]*len(df_j))))
    p_map = dict(zip(df_p.get("Variable", []), df_p.get("Percent_Contribution", [0]*len(df_p))))

    # Sort variables by maximum contribution
    sorted_vars = sorted(all_vars, key=lambda v: max(j_map.get(v, 0.0), p_map.get(v, 0.0)), reverse=False)

    y_vars = sorted_vars
    x_j = [float(j_map.get(v, 0.0)) for v in y_vars]
    x_p = [float(p_map.get(v, 0.0)) for v in y_vars]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=y_vars,
        x=x_j,
        name="🟢 Official MaxEnt Java (%)",
        orientation='h',
        marker=dict(color='#0d9488', line=dict(color='#0f766e', width=1)),
        hovertemplate="<b>%{y}</b><br>Official Java: %{x:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        y=y_vars,
        x=x_p,
        name="🔵 Native Python Engine (%)",
        orientation='h',
        marker=dict(color='#3b82f6', line=dict(color='#1d4ed8', width=1)),
        hovertemplate="<b>%{y}</b><br>Native Python: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Comparative Variable Importance: Official Java vs. Native Python</b>",
        barmode='group',
        xaxis=dict(title="Percent Contribution (%)", zeroline=False),
        yaxis=dict(title="Environmental Predictor"),
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=50),
        height=420
    )
    return fig


def plot_plotly_response_curve(
    var_name: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    unit: str = "",
    y_label: str = "Suitability (Cloglog)",
    template: str = "plotly_white"
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
        template=template,
        margin=dict(l=50, r=40, t=50, b=50),
        height=380
    )
    return fig


def plot_plotly_dual_response_curve(
    var_name: str,
    x_vals_java: np.ndarray,
    y_vals_java: np.ndarray,
    x_vals_py: np.ndarray,
    y_vals_py: np.ndarray,
    unit: str = "",
    y_label: str = "Suitability",
    template: str = "plotly_white"
) -> go.Figure:
    """
    Interactive response curve comparison overlaying Official Java and Native Python.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals_java,
        y=y_vals_java,
        mode='lines',
        line=dict(color='#0d9488', width=3),
        name='🟢 Official MaxEnt (Java)',
        hovertemplate=f"<b>Official Java</b><br>{var_name}: %{{x:.2f}} {unit}<br>{y_label}: %{{y:.3f}}<extra></extra>",
        fill='tozeroy',
        fillcolor='rgba(13, 148, 136, 0.08)'
    ))

    fig.add_trace(go.Scatter(
        x=x_vals_py,
        y=y_vals_py,
        mode='lines',
        line=dict(color='#3b82f6', width=3, dash='dot'),
        name='🔵 Native Python Engine',
        hovertemplate=f"<b>Native Python</b><br>{var_name}: %{{x:.2f}} {unit}<br>{y_label}: %{{y:.3f}}<extra></extra>",
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.08)'
    ))

    fig.update_layout(
        title=f"<b>Comparative Response Curve: {var_name} (Java vs. Python)</b>",
        xaxis=dict(title=f"{var_name} {unit}".strip(), zeroline=False),
        yaxis=dict(title=y_label, range=[-0.02, 1.02], zeroline=False),
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=40, t=50, b=50),
        height=400
    )
    return fig
