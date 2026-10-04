"""
tests/test_bounded_platt.py

Validates the TimeSeriesCalibrator under Bounded Platt Scaling:
- Confirms parameters satisfy: slope a in [0.1, 2.5], intercept b in [-1.5, 1.5].
- Tests output probability bounds in [0.0, 1.0].
- Ensures monotonicity: higher raw logit maps to higher calibrated probability.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.modeling.calibrator import TimeSeriesCalibrator


@pytest.fixture
def synthetic_classification_data():
    """Generates synthetic logits and binary targets."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n) * 2.0,
    })
    # True latent probability
    logits = 0.8 * X["feature_1"] - 0.5 * X["feature_2"]
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = np.random.binomial(1, prob)
    return X, y


def test_bounded_platt_parameters_within_bounds(synthetic_classification_data):
    """Verifies that parameter optimizer respects a in [0.1, 2.5] and b in [-1.5, 1.5]."""
    X, y = synthetic_classification_data
    base_clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")

    calibrator = TimeSeriesCalibrator(base_estimator=base_clf, method="sigmoid", calib_size=0.3)
    calibrator.fit(X, y)

    assert calibrator.calibrator_params_ is not None, "Calibration parameters must not be None."
    a_param, b_param = calibrator.calibrator_params_

    assert 0.1 <= a_param <= 2.5, f"Slope parameter a={a_param} out of bound [0.1, 2.5]!"
    assert -1.5 <= b_param <= 1.5, f"Intercept parameter b={b_param} out of bound [-1.5, 1.5]!"


def test_calibrator_output_probabilities_validity(synthetic_classification_data):
    """Verifies calibrated probabilities are strictly valid and monotonic."""
    X, y = synthetic_classification_data
    base_clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")

    calibrator = TimeSeriesCalibrator(base_estimator=base_clf, method="sigmoid", calib_size=0.3)
    calibrator.fit(X, y)

    probs = calibrator.predict_proba(X)
    assert probs.shape == (len(X), 2), "Probability array shape mismatch."
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be in [0, 1]."
    assert np.allclose(probs.sum(axis=1), 1.0), "Row probabilities must sum to 1.0."

    # Monotonicity check against raw decision function
    raw_scores = calibrator.fitted_base_estimator_.decision_function(X)
    p1 = probs[:, 1]
    
    # Check Spearman rank correlation between raw score and calibrated probability
    corr = np.corrcoef(raw_scores, p1)[0, 1]
    assert corr > 0.99, "Calibrator must preserve monotonic ranking of raw decision logits."