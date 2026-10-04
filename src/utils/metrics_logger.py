"""
src/utils/metrics_logger.py

Comprehensive probability verification and classification metrics logger:
- Murphy (1973) Brier score decomposition: Uncertainty, Reliability, and Resolution.
- Classification reports and standardized 2x2 confusion matrix presentation.
"""

from typing import Dict, Any, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss


def evaluate_brier_decomposition(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 5,
) -> Dict[str, float]:
    """
    Decomposes the empirical Brier score into Uncertainty, Reliability, and Resolution (Murphy, 1973).

    Brier = Uncertainty + Reliability (Loss) - Resolution
    Where:
        Uncertainty = bar_y * (1 - bar_y)
        Reliability = (1 / N) * sum(n_k * (p_k - y_k)^2)
        Resolution  = (1 / N) * sum(n_k * (y_k - bar_y)^2)

    Parameters
    ----------
    y_true : Sequence[int]
        Binary ground truth labels (0 or 1).
    y_prob : Sequence[float]
        Predicted probability array.
    n_bins : int, default=5
        Number of bins for quantile partitioning.

    Returns
    -------
    Dict[str, float]
        Dictionary containing Base Rate, Uncertainty, Total Brier, Reliability, and Resolution.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    N = len(y_t)

    base_rate = float(np.mean(y_t))
    uncertainty = float(base_rate * (1.0 - base_rate))
    brier_score = float(np.mean((y_p - y_t) ** 2))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0

    print("\n" + "=" * 75)
    print("BRIER SCORE CALIBRATION DECOMPOSITION (MURPHY, 1973)")
    print(f"Base Rate: {base_rate:.4f} | Uncertainty: {uncertainty:.4f} | Total Brier: {brier_score:.4f}")
    print("=" * 75)
    print(f"{'Bin Range':<15} {'Mean Pred':<12} {'Empirical Win':<15} {'Count':<8} {'Reliability Bias':<15}")

    for k in range(n_bins):
        mask = bin_indices == k
        n_k = int(np.sum(mask))
        if n_k > 0:
            p_k = float(np.mean(y_p[mask]))
            y_k = float(np.mean(y_t[mask]))
            reliability += n_k * ((p_k - y_k) ** 2)
            resolution += n_k * ((y_k - base_rate) ** 2)
            print(f"[{bins[k]:.1f}-{bins[k+1]:.1f}]       {p_k:<12.4f} {y_k:<15.4f} {n_k:<8} {p_k - y_k:<+15.4f}")

    reliability /= N
    resolution /= N

    print("-" * 75)
    print(f"Reliability (Calibration Loss, lower is better) : {reliability:.6f}")
    print(f"Resolution (Information Discriminant, higher is better): {resolution:.6f}")
    status = "QUALIFIED" if brier_score < uncertainty else "UNQUALIFIED"
    print(f"Calibration Status: {status} (Brier Score < Uncertainty)")

    return {
        "base_rate": base_rate,
        "uncertainty": uncertainty,
        "brier_score": brier_score,
        "reliability": reliability,
        "resolution": resolution,
    }


def print_classification_and_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
) -> None:
    """
    Prints standard scikit-learn classification report, ROC-AUC, Brier score,
    and a formatted 2x2 confusion matrix.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground truth labels (0 or 1).
    y_pred : Sequence[int]
        Predicted binary decisions (0 or 1).
    y_prob : Optional[Sequence[float]], default=None
        Predicted probability array for class 1.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    print("\n" + "=" * 65)
    print("OUT-OF-SAMPLE CLASSIFICATION REPORT")
    print("=" * 65)
    print(
        classification_report(
            y_t,
            y_p,
            target_names=["Fail / Time-out (0)", "Take-Profit Hit (1)"],
            digits=4,
        )
    )

    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=float)
        auc_score = roc_auc_score(y_t, y_pr)
        brier = brier_score_loss(y_t, y_pr)
        print(f"ROC-AUC Score : {auc_score:.4f}")
        print(f"Brier Score   : {brier:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Loss/Time-out (0)", "Actual Win (1)"],
        columns=["Predicted Pass (0)", "Predicted Enter (1)"],
    )
    print(cm_df)