"""
tests/test_fractional_diff.py

Validates the Fractional Differentiation engine:
- Weights computation matches the binomial series expansion.
- Fractionally differenced series achieves stationarity (ADF p < 0.05).
- Information memory correlation with original prices is preserved.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from src.diagnostics.fractional import Layer5FractionalIntegration


def test_fractional_weights_expansion():
    """Validates binomial series weights: w_0 = 1, w_1 = -d, w_2 = d*(d-1)/2."""
    d = 0.4
    weights = Layer5FractionalIntegration.get_weights(d=d, size=5, threshold=1e-5)
    
    # weights are returned in reverse order for dot product convolution
    w = weights[::-1]
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], -d)
    assert np.isclose(w[2], (d * (d - 1.0)) / 2.0)


def test_optimal_d_recovers_stationarity():
    """
    Tests that fractional differencing transforms a non-stationary geometric
    random walk into a stationary series with optimal d in (0.0, 1.0).
    """
    np.random.seed(42)
    n = 600
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
    series = pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    # Initial prices should be non-stationary (ADF p-value > 0.05)
    raw_pvalue = adfuller(series, autolag="AIC")[1]
    assert raw_pvalue > 0.05, "Synthetic series should be non-stationary."

    res = Layer5FractionalIntegration.find_optimal_d(series, d_step=0.1)
    opt_d = res["optimal_d"]

    assert 0.0 < opt_d <= 1.0, f"Optimal d* must be in (0, 1], got {opt_d}"
    
    # Differentiate at optimal d*
    diffed = Layer5FractionalIntegration.fractionally_diff(series, d=opt_d).dropna()
    adf_p = adfuller(diffed, autolag="AIC")[1]
    assert adf_p < 0.05, f"Differenced series at d={opt_d} failed ADF test (p={adf_p:.4f})."
    assert res["memory_retention_corr"] > 0.3, "Memory correlation retention is too low."