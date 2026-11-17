"""
model.py - Maximum Entropy Presence Prediction Model (MaxEnt).

Implements the Maximum Entropy Species Distribution Model as described by:
- Phillips, Anderson & Schapire (2006): "Maximum entropy modeling of species geographic distributions"
- Phillips & Dudik (2008): "Modeling of species distributions with Maxent: new extensions and a comprehensive evaluation"
- Phillips et al. (2017): "Opening the black box: an open-source release of Maxent"
"""

import sys
from typing import List, Optional, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from .features import FeatureTransformer
from .metrics import evaluate_sdm, find_optimal_thresholds


class MaxEnt(BaseEstimator, ClassifierMixin):
    """
    Maximum Entropy Presence Prediction Model.
    
    Predicts species geographic distribution / habitat suitability from presence-only
    records and environmental background (or pseudo-absence) data.
    
    Parameters
    ----------
    reg_mult : float, default=1.0
        Regularization multiplier. Higher values impose heavier L1 penalty,
        producing simpler, smoother response curves and reducing overfitting.
        Lower values allow more complex, closely fitted responses.
    feature_types : list of str, optional
        Types of feature representations to include:
        ['linear', 'quadratic', 'product', 'hinge', 'threshold'].
        Default is ['linear', 'quadratic', 'product', 'hinge'].
    hinge_knots : int, default=5
        Number of internal knots for hinge features per variable.
    solver : str, default='lbfgs'
        Optimization algorithm: 'lbfgs' (direct dual convex optimization).
    max_iter : int, default=500
        Maximum number of iterations for the optimizer.
    tol : float, default=1e-6
        Convergence tolerance.
    default_output : str, default='cloglog'
        Default output format for predict():
        - 'cloglog' : Complementary log-log (recommended, estimated probability in [0, 1])
        - 'logistic': Logistic transformation in [0, 1]
        - 'raw'     : Probability density over the study area (sums to 1 over background)
        - 'cumulative': Percentile ranking of suitability in [0, 100]
    random_state : int, optional
        Seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress during optimization.
    """

    def __init__(
        self,
        reg_mult: float = 1.0,
        feature_types: Optional[List[str]] = None,
        hinge_knots: int = 5,
        solver: str = 'lbfgs',
        max_iter: int = 500,
        tol: float = 1e-6,
        default_output: str = 'cloglog',
        random_state: Optional[int] = 42,
        verbose: bool = False
    ):
        self.reg_mult = reg_mult
        self.feature_types = feature_types or ['linear', 'quadratic', 'product', 'hinge']
        self.hinge_knots = hinge_knots
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.default_output = default_output
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.transformer_: Optional[FeatureTransformer] = None
        self.weights_: Optional[np.ndarray] = None
        self.entropy_: float = 0.0
        self.log_z_: float = 0.0
        self.raw_bg_distribution_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.raw_feature_names_: List[str] = []
        self.n_presence_: int = 0
        self.n_background_: int = 0
        self.training_metrics_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    def fit(
        self,
        X_presence: Union[np.ndarray, pd.DataFrame],
        X_background: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'MaxEnt':
        """
        Fit the MaxEnt model on presence and background environmental data.
        
        Parameters
        ----------
        X_presence : np.ndarray or pd.DataFrame of shape (n_presences, n_features)
            Environmental covariates at observed presence sites.
        X_background : np.ndarray or pd.DataFrame of shape (n_background, n_features), optional
            Environmental covariates at random background / pseudo-absence sites.
            If None, background points are automatically sampled within the presence range.
        feature_names : list of str, optional
            Names of the environmental predictors.
            
        Returns
        -------
        self : MaxEnt
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Parse presence data
        if isinstance(X_presence, pd.DataFrame):
            self.raw_feature_names_ = list(X_presence.columns)
            X_p = X_presence.to_numpy(dtype=np.float64)
        else:
            X_p = np.asarray(X_presence, dtype=np.float64)
            if feature_names is not None:
                self.raw_feature_names_ = list(feature_names)
            else:
                self.raw_feature_names_ = [f"env_{i+1}" for i in range(X_p.shape[1])]

        self.n_presence_ = X_p.shape[0]
        n_features = X_p.shape[1]

        # 2. Parse or synthesize background data
        if X_background is None:
            if self.verbose:
                print("Warning: No background data provided. Generating 2,000 synthetic background samples.")
            p_min = np.nanmin(X_p, axis=0)
            p_max = np.nanmax(X_p, axis=0)
            p_range = np.maximum(p_max - p_min, 1e-4)
            # Uniform background across bounding box with 20% margin
            X_b = np.random.uniform(
                p_min - 0.2 * p_range,
                p_max + 0.2 * p_range,
                size=(2000, n_features)
            )
        elif isinstance(X_background, pd.DataFrame):
            X_b = X_background.to_numpy(dtype=np.float64)
        else:
            X_b = np.asarray(X_background, dtype=np.float64)

        self.n_background_ = X_b.shape[0]

        # 3. Fit feature transformer on combined landscape
        X_combined = np.vstack([X_p, X_b])
        self.transformer_ = FeatureTransformer(
            feature_types=self.feature_types,
            hinge_knots=self.hinge_knots
        )
        self.transformer_.fit(X_combined, feature_names=self.raw_feature_names_)
        self.feature_names_ = self.transformer_.feature_names

        # Transform presences and background
        Z_p = self.transformer_.transform(X_p)
        Z_b = self.transformer_.transform(X_b)
        n_expanded = Z_p.shape[1]

        # 4. Compute empirical presence feature expectations (bar_z_P)
        mean_p = np.mean(Z_p, axis=0)

        # 5. Regularization vector beta_k (Phillips et al. 2006)
        # beta_k = reg_mult * std_b / sqrt(N_pres)
        std_b = np.std(Z_b, axis=0)
        std_b[std_b < 1e-5] = 0.5
        beta = self.reg_mult * (std_b / np.sqrt(self.n_presence_))

        # 6. Optimize dual objective:
        # min_lambda - lambda @ mean_p + log( (1/n_b) sum_i exp(Z_b[i] @ lambda) ) + sum_k beta_k * |lambda_k|
        smooth_eps = 1e-6

        def objective_and_grad(lam):
            eta_b = Z_b @ lam
            max_eta = np.max(eta_b)
            exp_shifted = np.exp(eta_b - max_eta)
            sum_exp = np.sum(exp_shifted)
            log_z = max_eta + np.log(sum_exp) - np.log(self.n_background_)

            p_b = exp_shifted / sum_exp
            exp_feat = p_b @ Z_b

            # Smooth L1 approximation: sqrt(lambda^2 + eps^2)
            smooth_abs = np.sqrt(lam ** 2 + smooth_eps ** 2)
            smooth_grad = lam / smooth_abs

            loss = - np.dot(lam, mean_p) + log_z + np.sum(beta * smooth_abs)
            grad = - mean_p + exp_feat + (beta * smooth_grad)
            return loss, grad

        lam_init = np.zeros(n_expanded, dtype=np.float64)
        
        opt_res = minimize(
            fun=objective_and_grad,
            x0=lam_init,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
            }
        )

        weights = opt_res.x.copy()
        # Sparsify weights below threshold
        weights[np.abs(weights) < 1e-5] = 0.0
        self.weights_ = weights

        # 7. Compute partition function Z and background entropy H
        eta_b = Z_b @ self.weights_
        max_eta = np.max(eta_b)
        exp_eta = np.exp(eta_b - max_eta)
        sum_exp = np.sum(exp_eta)
        self.log_sum_exp_ = max_eta + np.log(sum_exp)
        self.log_z_ = self.log_sum_exp_ - np.log(self.n_background_)

        p_b = exp_eta / sum_exp
        # Entropy H = - sum p_i ln(p_i)
        safe_pb = np.clip(p_b, 1e-15, 1.0)
        self.entropy_ = - float(np.sum(safe_pb * np.log(safe_pb)))

        # Mark model as fitted before calculating predictions
        self.is_fitted_ = True

        # Cache raw background suitability for cumulative output transform
        self.raw_bg_distribution_ = np.sort(self._predict_raw(Z_b))

        # 8. Evaluate on training data
        train_p_suit = self.predict(X_p, output_type='cloglog')
        train_b_suit = self.predict(X_b, output_type='cloglog')
        self.training_metrics_ = evaluate_sdm(train_p_suit, train_b_suit)

        return self

    def _predict_raw(self, Z: np.ndarray) -> np.ndarray:
        """Compute raw probability density over study landscape."""
        eta = Z @ self.weights_
        log_prob = eta - self.log_sum_exp_
        return np.exp(np.clip(log_prob, -700, 700))

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        output_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict species presence suitability for given environmental conditions.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Environmental covariates to predict.
        output_type : str, optional
            'cloglog' (default), 'logistic', 'raw', or 'cumulative'.
            
        Returns
        -------
        suitability : np.ndarray
            Predicted values matching the requested output format.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict() can be called.")

        out_fmt = (output_type or self.default_output).lower()
        valid_fmts = {'cloglog', 'logistic', 'raw', 'cumulative'}
        if out_fmt not in valid_fmts:
            raise ValueError(f"Invalid output_type '{out_fmt}'. Choose from {valid_fmts}.")

        Z = self.transformer_.transform(X)
        eta = Z @ self.weights_

        if out_fmt == 'raw':
            return self._predict_raw(Z)

        elif out_fmt == 'cumulative':
            raw_vals = self._predict_raw(Z)
            # Percentile in background distribution
            percentiles = np.searchsorted(self.raw_bg_distribution_, raw_vals) / len(self.raw_bg_distribution_)
            return np.clip(percentiles * 100.0, 0.0, 100.0)

        elif out_fmt == 'logistic':
            # logistic = mu / (1 + mu) where mu = exp(eta - log_sum_exp + entropy)
            log_mu = eta - self.log_sum_exp_ + self.entropy_
            clipped_log_mu = np.clip(log_mu, -50, 50)
            return 1.0 / (1.0 + np.exp(-clipped_log_mu))

        elif out_fmt == 'cloglog':
            # cloglog = 1 - exp(- mu) where mu = exp(eta - log_sum_exp + entropy)
            log_mu = eta - self.log_sum_exp_ + self.entropy_
            clipped_log_mu = np.clip(log_mu, -50, 50)
            return 1.0 - np.exp(-np.exp(clipped_log_mu))

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (scikit-learn compatible).
        
        Returns
        -------
        probs : np.ndarray of shape (n_samples, 2)
            Col 0: Probability of absence (1 - suitability)
            Col 1: Probability of presence (suitability)
        """
        p_presence = self.predict(X, output_type='cloglog')
        p_absence = 1.0 - p_presence
        return np.column_stack([p_absence, p_presence])

    def evaluate(
        self,
        X_presence_test: Union[np.ndarray, pd.DataFrame],
        X_background_test: Union[np.ndarray, pd.DataFrame],
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test/validation data.
        
        Parameters
        ----------
        X_presence_test : np.ndarray or pd.DataFrame
            Presence sites held out for testing.
        X_background_test : np.ndarray or pd.DataFrame
            Background / absence sites held out for testing.
        threshold : float, optional
            Binary threshold. If None, optimal max TSS threshold is used.
            
        Returns
        -------
        metrics : dict
            AUC, Continuous Boyce Index, TSS, Sensitivity, Specificity, etc.
        """
        suit_p = self.predict(X_presence_test, output_type='cloglog')
        suit_b = self.predict(X_background_test, output_type='cloglog')
        return evaluate_sdm(suit_p, suit_b, threshold=threshold)

    def get_feature_importance(
        self,
        X_presence: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        X_background: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        n_repeats: int = 5
    ) -> pd.DataFrame:
        """
        Compute Permutation Feature Importance and Percent Contribution.
        
        Parameters
        ----------
        X_presence : array-like, optional
            Presence data to evaluate drop in AUC.
        X_background : array-like, optional
            Background data to evaluate drop in AUC.
        n_repeats : int, default=5
            Number of permutation runs per feature.
            
        Returns
        -------
        importance_df : pd.DataFrame
            Table sorted by permutation importance.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before computing feature importance.")

        # 1. Percent Contribution based on expanded weights
        raw_names = self.raw_feature_names_
        n_raw = len(raw_names)
        raw_contrib = np.zeros(n_raw)

        for w, name in zip(self.weights_, self.feature_names_):
            if abs(w) > 1e-6:
                for idx, r_name in enumerate(raw_names):
                    if r_name in name:
                        raw_contrib[idx] += abs(w)

        tot = np.sum(raw_contrib)
        pct_contrib = (raw_contrib / tot * 100.0) if tot > 0 else np.zeros(n_raw)

        # 2. Permutation Importance (Permuting each raw predictor)
        perm_importance = np.zeros(n_raw)
        
        if X_presence is not None and X_background is not None:
            if isinstance(X_presence, pd.DataFrame):
                X_p = X_presence.to_numpy(dtype=np.float64)
            else:
                X_p = np.asarray(X_presence, dtype=np.float64)
                
            if isinstance(X_background, pd.DataFrame):
                X_b = X_background.to_numpy(dtype=np.float64)
            else:
                X_b = np.asarray(X_background, dtype=np.float64)

            # Base AUC
            p_base = self.predict(X_p, output_type='cloglog')
            b_base = self.predict(X_b, output_type='cloglog')
            y_eval = np.concatenate([np.ones_like(p_base), np.zeros_like(b_base)])
            base_auc = roc_auc_score(y_eval, np.concatenate([p_base, b_base]))

            # Evaluate Permutation Importance by permuting variable j across combined pres/bg
            X_comb = np.vstack([X_p, X_b])
            n_p = len(X_p)

            for j in range(n_raw):
                auc_drops = []
                for _ in range(n_repeats):
                    X_comb_perm = X_comb.copy()
                    X_comb_perm[:, j] = np.random.permutation(X_comb_perm[:, j])
                    
                    p_pred = self.predict(X_comb_perm[:n_p], output_type='cloglog')
                    b_pred = self.predict(X_comb_perm[n_p:], output_type='cloglog')
                    perm_auc = roc_auc_score(y_eval, np.concatenate([p_pred, b_pred]))
                    auc_drops.append(max(0.0, base_auc - perm_auc))

                perm_importance[j] = np.mean(auc_drops)

            perm_tot = np.sum(perm_importance)
            if perm_tot > 0:
                perm_importance = (perm_importance / perm_tot) * 100.0

        df = pd.DataFrame({
            'Variable': raw_names,
            'Percent_Contribution': np.round(pct_contrib, 2),
            'Permutation_Importance': np.round(perm_importance, 2)
        }).sort_values(by=['Permutation_Importance', 'Percent_Contribution'], ascending=False).reset_index(drop=True)

        return df

    def get_response_curve(
        self,
        variable_name: str,
        n_points: int = 100,
        baseline: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate marginal response curve for an environmental predictor.
        
        Holds other variables constant at their mean (or median) and varies
        the selected variable across its training range.
        
        Parameters
        ----------
        variable_name : str
            Name of the environmental variable.
        n_points : int, default=100
            Number of points along the variable gradient.
        baseline : str, default='mean'
            'mean' or 'median' for holding other variables constant.
            
        Returns
        -------
        x_values : np.ndarray
            Sampled values along the environmental variable.
        suitability : np.ndarray
            Predicted presence suitability [0, 1].
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before computing response curves.")

        if variable_name not in self.raw_feature_names_:
            raise ValueError(f"Variable '{variable_name}' not found in {self.raw_feature_names_}")

        idx = self.raw_feature_names_.index(variable_name)
        v_min = self.transformer_.raw_min[idx]
        v_max = self.transformer_.raw_max[idx]
        
        x_vals = np.linspace(v_min, v_max, n_points)
        
        # Base vector
        base_vec = (self.transformer_.raw_min + self.transformer_.raw_max) / 2.0
        grid = np.tile(base_vec, (n_points, 1))
        grid[:, idx] = x_vals
        
        suit = self.predict(grid, output_type='cloglog')
        return x_vals, suit

    def predict_raster(
        self,
        raster_dict: Dict[str, np.ndarray],
        nodata_mask: Optional[np.ndarray] = None,
        output_type: str = 'cloglog'
    ) -> np.ndarray:
        """
        Generate spatial habitat suitability map across 2D raster grids.
        
        Parameters
        ----------
        raster_dict : dict of {str: np.ndarray}
            Dictionary mapping feature names to 2D numpy arrays of equal shape.
        nodata_mask : np.ndarray of bool, optional
            Boolean 2D mask where True indicates NoData/masked cells.
        output_type : str, default='cloglog'
            Output transformation ('cloglog', 'logistic', etc.)
            
        Returns
        -------
        suitability_raster : np.ndarray of shape (height, width)
            Spatial suitability predictions with NaNs at masked cells.
        """
        first_key = list(raster_dict.keys())[0]
        shape_2d = raster_dict[first_key].shape
        
        # Validate shapes and features
        for name in self.raw_feature_names_:
            if name not in raster_dict:
                raise ValueError(f"Required variable '{name}' missing from raster_dict.")
            if raster_dict[name].shape != shape_2d:
                raise ValueError(f"Raster '{name}' shape {raster_dict[name].shape} does not match {shape_2d}")

        # Flatten rasters
        flat_cols = []
        for name in self.raw_feature_names_:
            flat_cols.append(raster_dict[name].ravel())

        X_flat = np.column_stack(flat_cols)

        # Identify valid pixels (not NaN and not nodata)
        valid_pixels = ~np.isnan(X_flat).any(axis=1)
        if nodata_mask is not None:
            valid_pixels = valid_pixels & (~nodata_mask.ravel())

        suit_flat = np.full(X_flat.shape[0], np.nan, dtype=np.float32)
        if np.any(valid_pixels):
            X_valid = X_flat[valid_pixels]
            preds = self.predict(X_valid, output_type=output_type)
            suit_flat[valid_pixels] = preds.astype(np.float32)

        return suit_flat.reshape(shape_2d)

    def summary(self) -> str:
        """Return formatted ASCII summary of the model and training metrics."""
        if not self.is_fitted_:
            return "MaxEnt Model (Not fitted)"

        active_weights = np.sum(np.abs(self.weights_) > 1e-5)
        m = self.training_metrics_

        lines = [
            "=" * 65,
            "              MAXIMUM ENTROPY PRESENCE MODEL SUMMARY",
            "=" * 65,
            f"Presences (Training):       {self.n_presence_}",
            f"Background Points:          {self.n_background_}",
            f"Raw Predictors:             {len(self.raw_feature_names_)}",
            f"Expanded Features:          {len(self.feature_names_)}",
            f"Active Features (Non-zero): {active_weights}",
            f"Regularization Multiplier:  {self.reg_mult}",
            f"Background Entropy (H):     {self.entropy_:.4f}",
            "-" * 65,
            "TRAINING EVALUATION METRICS:",
            f"  • Training ROC AUC:        {m.get('auc', 0.0):.4f}",
            f"  • Continuous Boyce Index:  {m.get('boyce_index', 0.0):.4f}",
            f"  • Optimal Threshold (TSS): {m.get('threshold', 0.0):.4f}",
            f"  • True Skill Stat (TSS):   {m.get('tss', 0.0):.4f}",
            f"  • Sensitivity:             {m.get('sensitivity', 0.0):.4f}",
            f"  • Specificity:             {m.get('specificity', 0.0):.4f}",
            "=" * 65
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        state = "Fitted" if self.is_fitted_ else "Unfitted"
        return f"MaxEnt(reg_mult={self.reg_mult}, feature_types={self.feature_types}, state={state})"
