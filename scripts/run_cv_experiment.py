#!/usr/bin/env python3
"""
scripts/run_cv_experiment.py

Executes Purged & Embargo Time-Series Cross Validation (López de Prado standard).
Incorporates:
- 4 Orthogonal Clustered Features with Deg-2 Polynomial Expansion
- Bounded Platt Scaling Probability Calibration (L-BFGS-B)
- Pure Rolling Percentile Thresholding (Dynamic Regime Shift Invariance)
- Dynamic Rolling Z-Score Position Sizing
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modeling.splitters import TimeBasedPurgedTimeSeriesSplit
from src.modeling.calibrator import TimeSeriesCalibrator
from src.features.pipeline import EconometricsFeaturePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Purged & Embargo CV Execution Engine")
    parser.add_argument("--input_path", type=str, default="data/processed/df_ready_for_cv.csv", help="Input dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for predictions")
    parser.add_argument("--artifacts_dir", type=str, default="checkpoints", help="Directory to save trained pipelines")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of outer CV splits")
    parser.add_argument("--embargo_days", type=int, default=5, help="Embargo safe-buffer length in days")
    parser.add_argument("--max_train_size", type=int, default=750, help="Maximum rolling training bars")
    parser.add_argument("--buffer_days", type=int, default=150, help="Test buffer days to eliminate lag cold-start")
    parser.add_argument("--rolling_window", type=int, default=60, help="Rolling window for percentile thresholding")
    parser.add_argument("--enter_percentile", type=float, default=65.0, help="Percentile threshold (Top 35% enter)")
    parser.add_argument("--c_reg", type=float, default=0.03, help="L2 Regularization strength C for Logistic Regression")
    return parser.parse_args()


def evaluate_brier_decomposition(df_oos: pd.DataFrame):
    y_true = df_oos["true_target"].values
    y_prob = df_oos["prob_up"].values
    N = len(y_true)

    base_rate = np.mean(y_true)
    uncertainty = base_rate * (1.0 - base_rate)
    brier_score = np.mean((y_prob - y_true) ** 2)

    bins = np.linspace(0.0, 1.0, 6)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, 4)

    reliability = 0.0
    resolution = 0.0

    print("\n" + "=" * 70)
    print("BRIER SCORE CALIBRATION DECOMPOSITION (MURPHY, 1973)")
    print(f"Base Rate: {base_rate:.4f} | Uncertainty: {uncertainty:.4f} | Total Brier: {brier_score:.4f}")
    print("=" * 70)
    print(f"{'Bin':<12} {'Mean Pred':<12} {'Empirical Win':<15} {'Count':<8} {'Reliability Bias':<15}")

    for k in range(5):
        mask = (bin_indices == k)
        n_k = np.sum(mask)
        if n_k > 0:
            p_k = np.mean(y_prob[mask])
            y_k = np.mean(y_true[mask])
            reliability += n_k * ((p_k - y_k) ** 2)
            resolution += n_k * ((y_k - base_rate) ** 2)
            print(f"[{bins[k]:.1f}-{bins[k+1]:.1f}]     {p_k:<12.4f} {y_k:<15.4f} {n_k:<8} {p_k - y_k:<+15.4f}")

    reliability /= N
    resolution /= N

    print("-" * 70)
    print(f"Reliability (Calibration Loss, lower is better) : {reliability:.6f}")
    print(f"Resolution (Information Discriminant, higher is better): {resolution:.6f}")
    status = "QUALIFIED" if brier_score < uncertainty else "UNQUALIFIED"
    print(f"Calibration Status: {status} (Brier Score < Uncertainty)")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    # 1. Load Data
    print(f"[*] Loading dataset: {args.input_path}")
    df_ready = pd.read_csv(args.input_path)
    df_ready["time"] = pd.to_datetime(df_ready["time"])
    df_ready.set_index("time", inplace=True)
    df_ready.sort_index(inplace=True)
    df_ready["barrier_touch_time"] = pd.to_datetime(df_ready["barrier_touch_time"])

    # 2. Setup Purged & Embargo CV
    embargo_period = pd.Timedelta(days=args.embargo_days)
    outer_cv = TimeBasedPurgedTimeSeriesSplit(
        n_splits=args.n_splits,
        max_train_size=args.max_train_size,
        embargo_td=embargo_period
    )

    oos_predictions = []
    trained_models = []
    trained_pipelines = []
    selected_features_per_fold = []

    print("\n" + "=" * 80)
    print("EXECUTING PURGED & EMBARGO CROSS VALIDATION PIPELINE (MODELING 07.2)")
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df_ready, touch_time_col="barrier_touch_time")):
        test_start_time = df_ready.index[test_idx[0]]
        last_train_touch = df_ready["barrier_touch_time"].iloc[train_idx].max()
        actual_gap = test_start_time - last_train_touch

        print(f"\n>>> [FOLD {fold + 1}/{args.n_splits}]")
        print(f"    [Audit Timestamps] Test Start: {test_start_time} | Max Train Touch: {last_train_touch}")
        print(f"    [Safety Gap] Actual: {actual_gap} (Required > {embargo_period})")

        assert actual_gap > embargo_period, (
            f"Leakage detected at Fold {fold + 1}: Safety gap {actual_gap} is below threshold {embargo_period}!"
        )

        df_train = df_ready.iloc[train_idx].copy()
        df_test = df_ready.iloc[test_idx].copy()

        y_train = df_train["target_label"].astype(int)
        X_train_raw = df_train.drop(columns=["target_label"])

        # 3. Fit Clustered Feature Pipeline (Cap 4 Orthogonal Features)
        feat_pipe = EconometricsFeaturePipeline(
            target_horizon=5,
            max_final_features=4
        )
        feat_pipe.fit(X_train_raw, y_train)

        current_features = list(feat_pipe.selected_features_)
        selected_features_per_fold.append(current_features)
        trained_pipelines.append(feat_pipe)

        X_train_transformed = feat_pipe.transform(X_train_raw)
        common_train_idx = X_train_transformed.index.intersection(y_train.index)
        X_tr = X_train_transformed.loc[common_train_idx].reindex(columns=current_features, fill_value=np.nan)
        y_tr = y_train.loc[common_train_idx].values

        num_expanded = len(current_features) + (len(current_features) * (len(current_features) - 1)) // 2
        print(
            f"    [Train Samples] Size: {len(X_tr)} (Class 1: {np.sum(y_tr == 1)}, Class 0: {np.sum(y_tr == 0)}) | "
            f"Selected Base: {current_features} -> Deg-2 Features: {num_expanded} cols"
        )

        if len(np.unique(y_tr)) < 2:
            print(f"    [!] Fold {fold + 1} has only one class in Train. Skipping.")
            continue

        # 4. Base Estimator: Polynomial Features Degree 2 Interaction
        base_clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=args.c_reg,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ])

        # 5. Fit Bounded Platt Scaling Calibrator
        calibrated_model = TimeSeriesCalibrator(
            base_estimator=base_clf,
            method="sigmoid",
            calib_size=0.3
        )
        calibrated_model.fit(X_tr, y_tr)
        trained_models.append(calibrated_model)

        if calibrated_model.calibrator_params_ is not None:
            a_val, b_val = calibrated_model.calibrator_params_
            print(f"    [Calibration Audit] Bounded Platt: a (slope)={a_val:.4f}, b (intercept)={b_val:.4f}")

        # 6. OOS Inference with Safe Lookback Buffer
        buffer_start = max(0, test_idx[0] - args.buffer_days)
        df_test_buffer = df_ready.iloc[buffer_start : test_idx[-1] + 1].copy()

        X_test_buffer_raw = df_test_buffer.drop(columns=["target_label"], errors="ignore")
        X_test_buffer_trans = feat_pipe.transform(X_test_buffer_raw)

        X_test_pure = X_test_buffer_trans.loc[X_test_buffer_trans.index.isin(df_test.index)]
        common_test_idx = X_test_pure.index.intersection(df_test.index)

        X_te = X_test_pure.loc[common_test_idx].reindex(columns=current_features, fill_value=np.nan)
        y_te = df_test.loc[common_test_idx, "target_label"].astype(int).values

        # 7. Dynamic Rolling Percentile Thresholding
        train_probs = calibrated_model.predict_proba(X_tr)[:, 1]
        calibrated_test_probs = calibrated_model.predict_proba(X_te)[:, 1]

        train_probs_series = pd.Series(train_probs, index=common_train_idx).sort_index()
        test_probs_series = pd.Series(calibrated_test_probs, index=common_test_idx).sort_index()

        probs_extended = pd.concat([train_probs_series.iloc[-args.rolling_window:], test_probs_series])

        rolling_threshold_series = (
            probs_extended
            .rolling(window=args.rolling_window, min_periods=args.rolling_window // 2)
            .apply(lambda w: np.percentile(w, args.enter_percentile), raw=True)
        )
        oos_dyn_threshold = rolling_threshold_series.loc[common_test_idx]

        # 8. Binary Signals & Rolling Z-Score Position Sizing
        y_pred_binary = (test_probs_series >= oos_dyn_threshold).astype(int).values

        rolling_mean = probs_extended.rolling(window=args.rolling_window, min_periods=args.rolling_window // 2).mean().loc[common_test_idx]
        rolling_std = probs_extended.rolling(window=args.rolling_window, min_periods=args.rolling_window // 2).std(ddof=0).loc[common_test_idx] + 1e-8
        prob_zscores = ((test_probs_series - rolling_mean) / rolling_std).values

        bet_sizes = np.where(
            y_pred_binary == 1,
            np.clip(0.20 + (prob_zscores - 0.5) * (0.80 / 1.50), 0.20, 1.0),
            0.0
        )

        print(
            f"    [Threshold Audit] Rolling Avg: {oos_dyn_threshold.mean():.4f} "
            f"[{oos_dyn_threshold.min():.4f} - {oos_dyn_threshold.max():.4f}] | "
            f"OOS Entry Rate: {np.mean(y_pred_binary) * 100:.2f}%"
        )

        res_df = pd.DataFrame(
            {
                "fold": fold + 1,
                "prob_up": calibrated_test_probs,
                "dynamic_threshold": oos_dyn_threshold.values,
                "pred_signal": y_pred_binary,
                "bet_size": bet_sizes,
                "true_target": y_te,
                "barrier_return": df_test.loc[common_test_idx, "target_barrier_return"].values,
                "barrier_touch_time": df_test.loc[common_test_idx, "barrier_touch_time"].values,
            },
            index=common_test_idx,
        )
        oos_predictions.append(res_df)

    # 9. Aggregate Results and Dump Artifacts
    df_oos_all = pd.concat(oos_predictions, axis=0)
    pred_path = os.path.join(args.output_dir, "df_oos_predictions.csv")
    df_oos_all.to_csv(pred_path)
    print(f"\n[✓] OOS Predictions saved to {pred_path} ({len(df_oos_all)} samples)")

    # Save artifacts for diagnostics
    artifacts = {
        "models": trained_models,
        "pipelines": trained_pipelines,
        "features": selected_features_per_fold,
        "predictions": oos_predictions
    }
    art_path = os.path.join(args.artifacts_dir, "cv_artifacts.pkl")
    with open(art_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"[✓] Model artifacts saved to {art_path}")

    # 10. Print Comprehensive Classification Metrics
    print("\n" + "=" * 65)
    print("OOS SIGNAL DISTRIBUTION BY FOLD")
    print("=" * 65)
    df_oos_all["signal_label"] = df_oos_all["pred_signal"].map({0: "No Trade (0)", 1: "Enter Long (1)"})
    print((pd.crosstab(df_oos_all["fold"], df_oos_all["signal_label"], normalize="index") * 100).round(2))

    print("\n" + "=" * 65)
    print("OOS CLASSIFICATION REPORT")
    print("=" * 65)
    y_true = df_oos_all["true_target"].astype(int)
    y_pred = df_oos_all["pred_signal"].astype(int)
    y_prob = df_oos_all["prob_up"]
    print(classification_report(y_true, y_pred, target_names=["Loss/Time-out (0)", "Take-Profit Hit (1)"], digits=4))

    print(f"ROC-AUC Score : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Brier Score   : {brier_score_loss(y_true, y_prob):.4f}")

    evaluate_brier_decomposition(df_oos_all)


if __name__ == "__main__":
    main()