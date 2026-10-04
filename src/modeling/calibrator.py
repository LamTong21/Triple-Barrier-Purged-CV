"""
src/modeling/calibrator.py

Causal forward-chaining probability calibration.
Implements Bounded Platt Scaling via L-BFGS-B optimization to prevent logit saturation
and probability collapse across market regime shifts.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class TimeSeriesCalibrator(BaseEstimator, ClassifierMixin):
    """
    Forward-chaining probability calibrator.

    Supported methods:
    - 'sigmoid': Bounded Platt Scaling: a in [0.1, 2.5], b in [-1.5, 1.5].
      Restricted parameter bounds prevent flat slope degenerate probabilities
      and base rate drift under covariate shift.
    - 'none': Pass-through mode, directly evaluates base estimator raw probabilities.
    """

    def __init__(self, base_estimator, method: str = "sigmoid", calib_size: float = 0.3):
        self.base_estimator = base_estimator
        self.method = method
        self.calib_size = calib_size
        self.fitted_base_estimator_ = None
        self.calibrator_params_ = None
        self.classes_ = np.array([0, 1])

    def _extract_raw_scores(self, estimator, X: pd.DataFrame) -> np.ndarray:
        if hasattr(estimator, "decision_function"):
            return np.asarray(estimator.decision_function(X), dtype=np.float64)
        else:
            probs = estimator.predict_proba(X)[:, 1]
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            return np.log(probs / (1.0 - probs))

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        n_samples = len(X)

        if self.method == "none":
            self.fitted_base_estimator_ = clone(self.base_estimator)
            self.fitted_base_estimator_.fit(X, y)
            self.calibrator_params_ = None
            return self

        split_idx = int(n_samples * (1.0 - self.calib_size))
        assert split_idx > 50, "Base training window too small for robust estimation."

        # Causal temporal partition (Base Window -> Calibration Window)
        X_base, y_base = X.iloc[:split_idx], y[:split_idx]
        X_calib, y_calib = X.iloc[split_idx:], y[split_idx:]

        if len(np.unique(y_base)) < 2 or len(np.unique(y_calib)) < 2:
            self.fitted_base_estimator_ = clone(self.base_estimator)
            self.fitted_base_estimator_.fit(X, y)
            self.calibrator_params_ = None
            return self

        self.fitted_base_estimator_ = clone(self.base_estimator)
        self.fitted_base_estimator_.fit(X_base, y_base)

        raw_scores = self._extract_raw_scores(self.fitted_base_estimator_, X_calib)

        if self.method == "sigmoid":
            def bounded_platt_loss(params):
                a, b = params
                p = expit(a * raw_scores + b)
                p = np.clip(p, 1e-7, 1.0 - 1e-7)
                return -np.mean(y_calib * np.log(p) + (1.0 - y_calib) * np.log(1.0 - p))

            # Constrain slope a in [0.1, 2.5] and intercept b in [-1.5, 1.5]
            res = minimize(
                bounded_platt_loss,
                x0=[1.0, 0.0],
                bounds=[(0.1, 2.5), (-1.5, 1.5)],
                method="L-BFGS-B",
            )
            self.calibrator_params_ = res.x
        else:
            raise ValueError(f"Calibration method '{self.method}' is not supported.")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.method == "none" or self.calibrator_params_ is None:
            return self.fitted_base_estimator_.predict_proba(X)

        raw_scores = self._extract_raw_scores(self.fitted_base_estimator_, X)
        a, b = self.calibrator_params_
        p1 = expit(a * raw_scores + b)
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)