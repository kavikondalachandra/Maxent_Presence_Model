"""
features.py - Feature Transformation Engine for Maximum Entropy Modeling.

Implements feature expansions as formulated by Phillips, Anderson & Schapire (2006):
- Linear features
- Quadratic features
- Product / Interaction features
- Hinge (piecewise linear) features
- Threshold (step) features
- Categorical (dummy) features
"""

from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd


class FeatureTransformer:
    """
    Transforms raw environmental variables into the feature representations
    used in Maximum Entropy Presence Prediction modeling.
    
    All features are normalized to [0, 1] across the background and presence data
    to ensure balanced L1 regularization penalties across all dimensions.
    """
    def __init__(
        self,
        feature_types: Optional[List[str]] = None,
        hinge_knots: int = 5,
        threshold_knots: int = 5,
        clip_bounds: bool = True
    ):
        """
        Parameters
        ----------
        feature_types : list of str, optional
            Subset of ['linear', 'quadratic', 'product', 'hinge', 'threshold'].
            Defaults to ['linear', 'quadratic', 'product', 'hinge'].
        hinge_knots : int, default=5
            Number of knots to place along each variable for hinge features.
        threshold_knots : int, default=5
            Number of knots for step/threshold features.
        clip_bounds : bool, default=True
            Whether to clip out-of-bounds prediction values to training [min, max].
        """
        if feature_types is None:
            self.feature_types = ['linear', 'quadratic', 'product', 'hinge']
        else:
            valid_types = {'linear', 'quadratic', 'product', 'hinge', 'threshold'}
            for ft in feature_types:
                if ft.lower() not in valid_types:
                    raise ValueError(f"Unknown feature type '{ft}'. Valid choices: {valid_types}")
            self.feature_types = [ft.lower() for ft in feature_types]
            
        self.hinge_knots = max(2, int(hinge_knots))
        self.threshold_knots = max(2, int(threshold_knots))
        self.clip_bounds = clip_bounds
        
        # Fitted parameters
        self.is_fitted: bool = False
        self.raw_feature_names: List[str] = []
        self.feature_names: List[str] = []
        self.raw_min: np.ndarray = np.array([])
        self.raw_max: np.ndarray = np.array([])
        self.hinge_knot_dict: Dict[int, np.ndarray] = {}
        self.threshold_knot_dict: Dict[int, np.ndarray] = {}
        self.n_raw_features: int = 0
        self.n_expanded_features: int = 0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None) -> 'FeatureTransformer':
        """
        Fit the transformer on the training data (typically presences + background combined).
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame of shape (n_samples, n_raw_features)
            Environmental covariates.
        feature_names : list of str, optional
            Names of the raw environmental variables.
        """
        if isinstance(X, pd.DataFrame):
            self.raw_feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if feature_names is not None:
                self.raw_feature_names = list(feature_names)
            else:
                self.raw_feature_names = [f"var_{i+1}" for i in range(X_arr.shape[1])]

        self.n_raw_features = X_arr.shape[1]
        if len(self.raw_feature_names) != self.n_raw_features:
            raise ValueError(
                f"Length of feature_names ({len(self.raw_feature_names)}) "
                f"must match number of columns ({self.n_raw_features})"
            )

        # Compute min and max for normalization
        self.raw_min = np.nanmin(X_arr, axis=0)
        self.raw_max = np.nanmax(X_arr, axis=0)
        
        # Handle zero-variance columns
        zero_var_mask = (self.raw_max - self.raw_min) < 1e-12
        self.raw_max[zero_var_mask] = self.raw_min[zero_var_mask] + 1.0

        # Construct knots for hinge and threshold features
        self.hinge_knot_dict = {}
        self.threshold_knot_dict = {}
        
        for j in range(self.n_raw_features):
            col_vals = X_arr[:, j]
            valid_vals = col_vals[~np.isnan(col_vals)]
            
            if 'hinge' in self.feature_types:
                # Knots evenly spaced between min and max
                knots = np.linspace(self.raw_min[j], self.raw_max[j], self.hinge_knots + 2)[1:-1]
                self.hinge_knot_dict[j] = knots
                
            if 'threshold' in self.feature_types:
                # Knots based on quantiles or uniform spacing
                knots = np.percentile(valid_vals, np.linspace(10, 90, self.threshold_knots))
                self.threshold_knot_dict[j] = np.unique(knots)

        # Build feature names list
        self.feature_names = self._build_feature_names()
        self.n_expanded_features = len(self.feature_names)
        self.is_fitted = True
        return self

    def _build_feature_names(self) -> List[str]:
        """Construct descriptive names for all expanded features."""
        names = []
        
        # 1. Linear features
        if 'linear' in self.feature_types:
            for name in self.raw_feature_names:
                names.append(f"{name} (linear)")
                
        # 2. Quadratic features
        if 'quadratic' in self.feature_types:
            for name in self.raw_feature_names:
                names.append(f"{name}^2 (quadratic)")
                
        # 3. Product / Interaction features
        if 'product' in self.feature_types:
            for i in range(self.n_raw_features):
                for j in range(i + 1, self.n_raw_features):
                    names.append(f"{self.raw_feature_names[i]} * {self.raw_feature_names[j]} (product)")
                    
        # 4. Hinge features
        if 'hinge' in self.feature_types:
            for j in range(self.n_raw_features):
                name = self.raw_feature_names[j]
                knots = self.hinge_knot_dict[j]
                for k, knot in enumerate(knots):
                    names.append(f"{name} >= {knot:.3f} (hinge_fwd)")
                    names.append(f"{name} <= {knot:.3f} (hinge_rev)")
                    
        # 5. Threshold features
        if 'threshold' in self.feature_types:
            for j in range(self.n_raw_features):
                name = self.raw_feature_names[j]
                knots = self.threshold_knot_dict[j]
                for k, knot in enumerate(knots):
                    names.append(f"{name} > {knot:.3f} (threshold)")
                    
        return names

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform raw environmental variables into expanded [0, 1] normalized features.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame of shape (n_samples, n_raw_features)
            Raw covariates to transform.
            
        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_expanded_features)
            Transformed features.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureTransformer must be fitted before calling transform().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.n_raw_features:
            raise ValueError(
                f"Input has {X_arr.shape[1]} features, but transformer expected {self.n_raw_features}."
            )

        n_samples = X_arr.shape[0]
        
        # Optional clipping to training min/max bounds
        if self.clip_bounds:
            X_clipped = np.clip(X_arr, self.raw_min, self.raw_max)
        else:
            X_clipped = X_arr

        # Normalized base features into [0, 1]
        denom = self.raw_max - self.raw_min
        denom[denom == 0] = 1.0
        X_norm = (X_clipped - self.raw_min) / denom
        X_norm = np.clip(X_norm, 0.0, 1.0)

        feature_blocks = []

        # 1. Linear features
        if 'linear' in self.feature_types:
            feature_blocks.append(X_norm)

        # 2. Quadratic features
        if 'quadratic' in self.feature_types:
            feature_blocks.append(X_norm ** 2)

        # 3. Product / Interaction features
        if 'product' in self.feature_types:
            products = []
            for i in range(self.n_raw_features):
                for j in range(i + 1, self.n_raw_features):
                    products.append((X_norm[:, i] * X_norm[:, j]).reshape(-1, 1))
            if products:
                feature_blocks.append(np.hstack(products))

        # 4. Hinge features
        if 'hinge' in self.feature_types:
            hinges = []
            for j in range(self.n_raw_features):
                col_raw = X_clipped[:, j]
                knots = self.hinge_knot_dict[j]
                min_val = self.raw_min[j]
                max_val = self.raw_max[j]
                
                for knot in knots:
                    # Forward hinge: (x - knot) / (max - knot) for x >= knot
                    fwd_denom = max_val - knot
                    if fwd_denom < 1e-12:
                        fwd = np.zeros(n_samples)
                    else:
                        fwd = np.maximum(0.0, (col_raw - knot) / fwd_denom)
                    
                    # Reverse hinge: (knot - x) / (knot - min) for x <= knot
                    rev_denom = knot - min_val
                    if rev_denom < 1e-12:
                        rev = np.zeros(n_samples)
                    else:
                        rev = np.maximum(0.0, (knot - col_raw) / rev_denom)
                        
                    hinges.append(fwd.reshape(-1, 1))
                    hinges.append(rev.reshape(-1, 1))
            if hinges:
                feature_blocks.append(np.hstack(hinges))

        # 5. Threshold features
        if 'threshold' in self.feature_types:
            threshs = []
            for j in range(self.n_raw_features):
                col_raw = X_clipped[:, j]
                knots = self.threshold_knot_dict[j]
                for knot in knots:
                    threshs.append((col_raw > knot).astype(np.float64).reshape(-1, 1))
            if threshs:
                feature_blocks.append(np.hstack(threshs))

        Z = np.hstack(feature_blocks) if feature_blocks else np.empty((n_samples, 0))
        return Z

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit transformer and return transformed feature matrix."""
        return self.fit(X, feature_names=feature_names).transform(X)
