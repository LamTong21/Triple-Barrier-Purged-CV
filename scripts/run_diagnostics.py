#!/usr/bin/env python3
"""
scripts/run_diagnostics.py

Executes 4 comprehensive empirical diagnostic tests:
- Test 1: Bias-Variance Diagnostics (Train vs. OOS Generalization Gap)
- Test 2: Nonlinearity & Degree-2 Interaction Hypothesis Test
- Test 3: Brier Score Decomposition & Quantile Calibration Analysis (Murphy, 1973)
- Test 4: OOS Permutation Feature Importance across all Folds
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
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss, calibration_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modeling.splitters import TimeBasedPurgedTimeSeriesSplit


def parse_args():
    parser = argparse.ArgumentParser(description="Empirical Diagnostics & Hypothesis Testing")
    parser.add_argument("--data_path", type=str, default="data/processed/df_ready_for_cv.csv", help="Input dataset")
    parser.add_argument("--artifacts_path", type=str, default="checkpoints/cv_artifacts.pkl", help="Saved CV artifacts")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save diagnostic reports")
    parser.add_argument("--buffer_days", type=int, default=150, help="Test buffer days")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_ready = pd.read_csv(args.data_path)
    df_ready["time"] = pd.to_datetime(df_ready["time"])
    df_ready.set_index("time", inplace=True)
    df_ready.sort_index(inplace=True)
    df_ready["barrier_touch_time"] = pd.to_datetime(df_ready["barrier_touch_time"])

    with open(args.artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    trained_models = artifacts["models"]
    trained_pipelines = artifacts["pipelines"]
    selected_features_per_fold = artifacts["features"]
    oos_predictions = artifacts["predictions"]

    outer_cv = TimeBasedPurgedTimeSeriesSplit(
        n_splits=len(trained_models),
        max_train_size=750,
        embargo_td=pd.Timedelta(days=5)
    )

    # =========================================================================
    # TEST 1: BIAS - VARIANCE GENERALIZATION GAP
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: BIAS - VARIANCE GENERALIZATION GAP (TRAIN vs. OOS METRICS)")
    print("=" * 80)

    train_metrics = []
    test_metrics = []

    for fold, (train_idx, _) in enumerate(outer_cv.split(df_ready, touch_time_col="barrier_touch_time")):
        model = trained_models[fold]
        cur_feats = selected_features_per_fold[fold]
        res_fold = oos_predictions[fold]
        dynamic_thresh = res_fold["dynamic_threshold"].iloc[0]

        df_train = df_ready.iloc[train_idx].copy()
        y_tr = df_train["target_label"].astype(int)
        X_train_raw = df_train.drop(columns=["target_label"])

        feat_pipe = trained_pipelines[fold]
        X_tr_trans = feat_pipe.transform(X_train_raw)

        c_tr = X_tr_trans.index.intersection(y_tr.index)
        X_tr = X_tr_trans.loc[c_tr].reindex(columns=cur_feats, fill_value=np.nan)
        y_tr_eval = y_tr.loc[c_tr].values

        p_tr = model.predict_proba(X_tr)[:, 1]
        y_pred_tr = (p_tr >= dynamic_thresh).astype(int)

        train_metrics.append({
            "fold": fold + 1,
            "train_acc": accuracy_score(y_tr_eval, y_pred_tr),
            "train_auc": roc_auc_score(y_tr_eval, p_tr),
            "train_brier": brier_score_loss(y_tr_eval, p_tr),
            "train_loss": log_loss(y_tr_eval, p_tr),
        })

        test_metrics.append({
            "fold": fold + 1,
            "test_acc": accuracy_score(res_fold["true_target"], res_fold["pred_signal"]),
            "test_auc": roc_auc_score(res_fold["true_target"], res_fold["prob_up"]),
            "test_brier": brier_score_loss(res_fold["true_target"], res_fold["prob_up"]),
            "test_loss": log_loss(res_fold["true_target"], res_fold["prob_up"]),
        })

    df_diag = pd.DataFrame(train_metrics).merge(pd.DataFrame(test_metrics), on="fold")
    df_diag["acc_gap (Tr - Te)"] = df_diag["train_acc"] - df_diag["test_acc"]
    df_diag["auc_gap (Tr - Te)"] = df_diag["train_auc"] - df_diag["test_auc"]

    print(df_diag[[
        "fold", "train_acc", "test_acc", "acc_gap (Tr - Te)",
        "train_auc", "test_auc", "auc_gap (Tr - Te)"
    ]].to_string(index=False))

    print("-" * 80)
    print(f"[*] Mean Train AUC: {df_diag['train_auc'].mean():.4f} | Mean OOS Test AUC: {df_diag['test_auc'].mean():.4f} (Gap: {df_diag['auc_gap (Tr - Te)'].mean():+.4f})")
    print(f"[*] Mean Train ACC: {df_diag['train_acc'].mean() * 100:.2f}% | Mean OOS Test ACC: {df_diag['test_acc'].mean() * 100:.2f}% (Gap: {df_diag['acc_gap (Tr - Te)'].mean() * 100:+.2f}%)")

    # =========================================================================
    # TEST 2: NONLINEARITY & INTERACTION HYPOTHESIS TEST
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: NONLINEARITY & DEGREE-2 INTERACTION HYPOTHESIS TEST")
    print("=" * 80)

    interaction_aucs = []
    base_aucs = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df_ready, touch_time_col="barrier_touch_time")):
        cur_feats = selected_features_per_fold[fold]
        feat_pipe = trained_pipelines[fold]

        df_train = df_ready.iloc[train_idx].copy()
        y_tr = df_train["target_label"].astype(int)
        X_train_raw = df_train.drop(columns=["target_label"])

        X_tr_trans = feat_pipe.transform(X_train_raw)
        c_tr = X_tr_trans.index.intersection(y_tr.index)
        X_tr = X_tr_trans.loc[c_tr].reindex(columns=cur_feats, fill_value=np.nan)
        y_tr_eval = y_tr.loc[c_tr].values

        buffer_start = max(0, test_idx[0] - args.buffer_days)
        df_test_buffer = df_ready.iloc[buffer_start : test_idx[-1] + 1].copy()
        X_test_buffer_raw = df_test_buffer.drop(columns=["target_label"], errors="ignore")

        X_te_trans = feat_pipe.transform(X_test_buffer_raw)
        df_test = df_ready.iloc[test_idx].copy()
        X_te_pure = X_te_trans.loc[X_te_trans.index.isin(df_test.index)]

        c_te = X_te_pure.index.intersection(df_test.index)
        X_te = X_te_pure.loc[c_te].reindex(columns=cur_feats, fill_value=np.nan)
        y_te = df_test.loc[c_te, "target_label"].astype(int).values

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_tr_poly = poly.fit_transform(X_tr_imp)
        X_te_poly = poly.transform(X_te_imp)

        model_poly = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=0.05, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42))
        ])
        model_poly.fit(X_tr_poly, y_tr_eval)
        p_poly = model_poly.predict_proba(X_te_poly)[:, 1]
        p_base = trained_models[fold].predict_proba(X_te)[:, 1]

        base_aucs.append(roc_auc_score(y_te, p_base))
        interaction_aucs.append(roc_auc_score(y_te, p_poly))

    df_poly_res = pd.DataFrame({
        "Fold": range(1, len(base_aucs) + 1),
        "Linear Baseline AUC": base_aucs,
        "Interaction (Deg 2) AUC": interaction_aucs
    })
    df_poly_res["Delta AUC"] = df_poly_res["Interaction (Deg 2) AUC"] - df_poly_res["Linear Baseline AUC"]
    print(df_poly_res.to_string(index=False))
    print(f"[*] Mean Delta AUC from adding nonlinear interactions: {df_poly_res['Delta AUC'].mean():+.4f}")

    # =========================================================================
    # TEST 3: CALIBRATION ANALYSIS & QUANTILE BRIER DECOMPOSITION
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: CALIBRATION DIAGNOSIS & QUANTILE BRIER DECOMPOSITION")
    print("=" * 80)

    df_all_oos = pd.concat(oos_predictions).sort_index()
    y_true_all = df_all_oos["true_target"].astype(int).values
    p_pred_all = df_all_oos["prob_up"].values
    N = len(y_true_all)

    base_rate = float(np.mean(y_true_all))
    uncertainty = float(base_rate * (1.0 - base_rate))
    actual_brier = float(brier_score_loss(y_true_all, p_pred_all))

    quantiles = np.quantile(p_pred_all, np.linspace(0, 1, 6))
    quantiles[0] -= 1e-5
    bin_assignments = np.digitize(p_pred_all, quantiles) - 1

    reliability = 0.0
    resolution = 0.0
    calib_rows = []

    for k in range(5):
        mask = (bin_assignments == k)
        n_k = int(np.sum(mask))
        if n_k > 0:
            p_k = float(np.mean(p_pred_all[mask]))
            y_k = float(np.mean(y_true_all[mask]))
            reliability += n_k * ((p_k - y_k) ** 2)
            resolution += n_k * ((y_k - base_rate) ** 2)
            calib_rows.append({
                "Quantile Bin": f"Q{k+1} [{quantiles[k]:.3f} - {quantiles[k+1]:.3f}]",
                "Mean Pred": f"{p_k:.4f}",
                "Empirical Winrate": f"{y_k:.4f}",
                "Count": n_k,
                "Calibration Bias": f"{(p_k - y_k):+.4f}"
            })

    reliability /= N
    resolution /= N

    print(f"Base Rate (Prior)          : {base_rate:.4f}")
    print(f"Uncertainty (Baseline Var) : {uncertainty:.4f}")
    print(f"Empirical Brier Score      : {actual_brier:.4f}")
    print(f"  -> Reliability Loss (Miscalibration, -> 0): {reliability:.6f}")
    print(f"  -> Resolution (Predictive Partition, high): {resolution:.6f}")

    if actual_brier < uncertainty:
        print("  ✓ PASS: Brier Score < Uncertainty. Probability predictions offer empirical alpha.")
    else:
        print("  ✗ FAIL: Model degraded toward prior.")

    print("\nQuantile Calibration Breakdown:")
    print(pd.DataFrame(calib_rows).to_string(index=False))

    # =========================================================================
    # TEST 4: OOS PERMUTATION FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: OUT-OF-SAMPLE PERMUTATION FEATURE IMPORTANCE")
    print("=" * 80)

    fold_perm_drops = []
    np.random.seed(42)

    for fold, (train_indices, test_indices) in enumerate(outer_cv.split(df_ready, touch_time_col="barrier_touch_time")):
        model_eval = trained_models[fold]
        feats_eval = selected_features_per_fold[fold]
        feat_pipe = trained_pipelines[fold]

        buffer_start = max(0, test_indices[0] - args.buffer_days)
        df_test_buffer = df_ready.iloc[buffer_start : test_indices[-1] + 1].copy()
        X_test_buffer_raw = df_test_buffer.drop(columns=["target_label"], errors="ignore")

        X_test_buffer_trans = feat_pipe.transform(X_test_buffer_raw)
        df_test_fold = df_ready.iloc[test_indices].copy()
        X_test_pure = X_test_buffer_trans.loc[X_test_buffer_trans.index.isin(df_test_fold.index)]

        c_idx = X_test_pure.index.intersection(df_test_fold.index)
        X_te_raw = X_test_pure.loc[c_idx].reindex(columns=feats_eval, fill_value=np.nan)
        y_te_raw = df_test_fold.loc[c_idx, "target_label"].astype(int).values

        base_auc = roc_auc_score(y_te_raw, model_eval.predict_proba(X_te_raw)[:, 1])

        for col in feats_eval:
            X_perm = X_te_raw.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            p_perm = model_eval.predict_proba(X_perm)[:, 1]
            auc_perm = roc_auc_score(y_te_raw, p_perm)

            fold_perm_drops.append({
                "fold": fold + 1,
                "feature": col,
                "auc_drop": base_auc - auc_perm
            })

    df_perm_all = pd.DataFrame(fold_perm_drops)
    df_summary = (
        df_perm_all.groupby("feature")["auc_drop"]
        .agg(["mean", "std", "count"])
        .sort_values(by="mean", ascending=False)
        .rename(columns={"mean": "Mean OOS AUC Drop", "std": "Std Drop", "count": "Folds Present"})
    )
    print(df_summary.round(4).to_string())


if __name__ == "__main__":
    main()