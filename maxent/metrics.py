"""
metrics.py - Evaluation Metrics for Presence-Only and Presence-Background Models.

Includes:
- Receiver Operating Characteristic (ROC AUC)
- Continuous Boyce Index (CBI) - Gold standard for presence-only evaluation
- True Skill Statistic (TSS)
- Optimal threshold detection (Max TSS, Equal Sensitivity-Specificity, 10th Percentile)
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_continuous_boyce_index(
    suitability_presence: np.ndarray,
    suitability_background: np.ndarray,
    n_bins: int = 15,
    window_width: float = 0.2
) -> float:
    """
    Compute Continuous Boyce Index (Hirzel et al. 2006) for presence-only validation.
    
    Parameters
    ----------
    suitability_presence : np.ndarray
        Model predicted suitability at presence points (between 0 and 1).
    suitability_background : np.ndarray
        Model predicted suitability at background or landscape points (between 0 and 1).
    n_bins : int, default=15
        Number of overlapping moving windows.
    window_width : float, default=0.2
        Width of each moving suitability window.
        
    Returns
    -------
    cbi : float
        Spearman rank correlation coefficient [-1.0, 1.0].
        Values near +1.0 indicate excellent predictive ability.
        Values near 0 indicate random performance.
        Negative values indicate counter-predictive behavior.
    """
    suit_p = np.asarray(suitability_presence, dtype=np.float64)
    suit_b = np.asarray(suitability_background, dtype=np.float64)
    
    n_p = len(suit_p)
    n_b = len(suit_b)
    if n_p == 0 or n_b == 0:
        return np.nan

    min_val = min(suit_p.min(), suit_b.min())
    max_val = max(suit_p.max(), suit_b.max())
    if max_val - min_val <= 1e-6:
        return 0.0

    step = (max_val - min_val - window_width) / max(1, n_bins - 1)
    if step <= 0:
        step = (max_val - min_val) / n_bins
        window_width = step * 1.5

    f_ratios = []
    midpoints = []

    for i in range(n_bins):
        start = min_val + i * step
        end = start + window_width
        mid = (start + end) / 2.0
        
        # Fraction of presences in this window
        p_count = np.sum((suit_p >= start) & (suit_p <= end))
        p_frac = p_count / n_p

        # Fraction of background in this window
        b_count = np.sum((suit_b >= start) & (suit_b <= end))
        b_frac = b_count / n_b

        # Predicted-to-Expected ratio
        if b_frac > 0:
            pe_ratio = p_frac / b_frac
        else:
            pe_ratio = 0.0

        f_ratios.append(pe_ratio)
        midpoints.append(mid)

    # Filter out empty or static windows
    valid_idx = [i for i, r in enumerate(f_ratios) if r > 0]
    if len(valid_idx) < 3:
        corr, _ = spearmanr(midpoints, f_ratios)
        return float(corr) if not np.isnan(corr) else 0.0

    sub_mids = [midpoints[i] for i in valid_idx]
    sub_ratios = [f_ratios[i] for i in valid_idx]
    
    corr, _ = spearmanr(sub_mids, sub_ratios)
    return float(corr) if not np.isnan(corr) else 0.0


def find_optimal_thresholds(
    suitability_presence: np.ndarray,
    suitability_background: np.ndarray
) -> Dict[str, float]:
    """
    Find standard ecological threshold cutoffs for presence classification.
    
    Returns
    -------
    thresholds : dict
        - 'max_tss': Threshold maximizing True Skill Statistic
        - 'equal_sens_spec': Threshold where sensitivity equals specificity
        - 'p10': 10th percentile training presence suitability
        - 'min_presence': Minimum suitability among presence points
    """
    suit_p = np.asarray(suitability_presence, dtype=np.float64)
    suit_b = np.asarray(suitability_background, dtype=np.float64)
    
    y_true = np.concatenate([np.ones_like(suit_p), np.zeros_like(suit_b)])
    y_pred = np.concatenate([suit_p, suit_b])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # 1. Max TSS = TPR - FPR = Sensitivity + Specificity - 1
    tss_curve = tpr - fpr
    best_tss_idx = np.argmax(tss_curve)
    max_tss_th = float(thresholds[best_tss_idx])

    # 2. Equal Sensitivity and Specificity: TPR = 1 - FPR => TPR + FPR = 1
    # Minimize |TPR - (1 - FPR)|
    diff = np.abs(tpr - (1.0 - fpr))
    equal_idx = np.argmin(diff)
    equal_th = float(thresholds[equal_idx])

    # 3. 10th percentile presence threshold (10% training omission)
    p10_th = float(np.percentile(suit_p, 10))

    # 4. Minimum training presence threshold (0% training omission)
    min_pres_th = float(np.min(suit_p))

    return {
        'max_tss': np.clip(max_tss_th, 0.0, 1.0),
        'equal_sens_spec': np.clip(equal_th, 0.0, 1.0),
        'p10': np.clip(p10_th, 0.0, 1.0),
        'min_presence': np.clip(min_pres_th, 0.0, 1.0)
    }


def evaluate_sdm(
    suitability_presence: np.ndarray,
    suitability_background: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of species distribution model predictions.
    
    Parameters
    ----------
    suitability_presence : np.ndarray
        Model predictions at presence sites.
    suitability_background : np.ndarray
        Model predictions at background / pseudo-absence sites.
    threshold : float, optional
        Binary classification threshold. If None, max_tss threshold is used.
        
    Returns
    -------
    metrics : dict
        Collection of validation metrics:
        - 'auc': Area Under the ROC Curve
        - 'boyce_index': Continuous Boyce Index
        - 'threshold': Selected cutoff threshold
        - 'tss': True Skill Statistic at threshold
        - 'sensitivity': Sensitivity (TPR) at threshold
        - 'specificity': Specificity (TNR) at threshold
        - 'f1_score': F1-Score at threshold
    """
    suit_p = np.asarray(suitability_presence, dtype=np.float64)
    suit_b = np.asarray(suitability_background, dtype=np.float64)
    
    y_true = np.concatenate([np.ones_like(suit_p), np.zeros_like(suit_b)])
    y_pred = np.concatenate([suit_p, suit_b])
    
    # ROC AUC
    auc = float(roc_auc_score(y_true, y_pred))
    
    # Continuous Boyce Index
    cbi = float(calculate_continuous_boyce_index(suit_p, suit_b))
    
    # Thresholds
    threshold_dict = find_optimal_thresholds(suit_p, suit_b)
    if threshold is None:
        th = threshold_dict['max_tss']
    else:
        th = float(threshold)

    # Classification metrics at threshold
    tp = np.sum(suit_p >= th)
    fn = np.sum(suit_p < th)
    fp = np.sum(suit_b >= th)
    tn = np.sum(suit_b < th)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    tss = sensitivity + specificity - 1.0

    return {
        'auc': auc,
        'boyce_index': cbi,
        'threshold': th,
        'tss': tss,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'threshold_max_tss': threshold_dict['max_tss'],
        'threshold_equal_sens_spec': threshold_dict['equal_sens_spec'],
        'threshold_p10': threshold_dict['p10'],
        'threshold_min_presence': threshold_dict['min_presence']
    }
