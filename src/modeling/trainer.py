"""
src/modeling/trainer.py

Orchestrates the entire Purged & Embargo CV modeling execution:
- Coordinates feature pipelines, Deg-2 interaction polynomial estimators,
  Bounded Platt Scaling, and dynamic thresholding across all temporal folds.
- Collects and aggregates Out-Of-Sample (OOS) performance records.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.features.pipeline import EconometricsFeaturePipeline
from src.modeling.calibrator import TimeSeriesCalibrator
from src.modeling.splitters import TimeBasedPurgedTimeSeriesSplit
from src.modeling.thresholding import DynamicThresholdEngine


class PurgedCVTrainer:
    """
    Execution engine coordinating Purged & Embargo Cross-Validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 5,
        max_train_size: int = 750,
        buffer_days: int = 150,
        target_horizon: int = 5,
        rolling_window: int = 60,
        enter_percentile: float = 65.0,
        c_reg: float = 0.03,
    ):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.max_train_size = max_train_size
        self.buffer_days = buffer_days
        self.target_horizon = target_horizon
        self.rolling_window = rolling_window
        self.enter_percentile = enter_percentile
        self.c_reg = c_reg

        self.cv_splitter = TimeBasedPurgedTimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            embargo_td=pd.Timedelta(days=self.embargo_days),
        )
        self.threshold_engine = DynamicThresholdEngine(
            rolling_window=self.rolling_window,
            enter_percentile=self.enter_percentile,
        )

    def train_and_evaluate(self, df_ready: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, list]]:
        """
        Executes outer CV loop and returns aggregated OOS predictions and trained models.

        Parameters
        ----------
        df_ready : pd.DataFrame
            Preprocessed market dataset with target labels and barrier touch timestamps.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, list]]
            Aggregated OOS predictions DataFrame and dictionary of model artifacts.
        """
        oos_predictions = []
        trained_models = []
        trained_pipelines = []
        selected_features_per_fold = []

        for fold, (train_idx, test_idx) in enumerate(
            self.cv_splitter.split(df_ready, touch_time_col="barrier_touch_time")
        ):
            test_start_time = df_ready.index[test_idx[0]]
            last_train_touch = df_ready["barrier_touch_time"].iloc[train_idx].max()
            actual_gap = test_start_time - last_train_touch

            assert actual_gap > pd.Timedelta(days=self.embargo_days), (
                f"Leakage detected at Fold {fold + 1}: Safety gap {actual_gap} is below threshold!"
            )

            df_train = df_ready.iloc[train_idx].copy()
            df_test = df_ready.iloc[test_idx].copy()

            y_train = df_train["target_label"].astype(int)
            X_train_raw = df_train.drop(columns=["target_label"])

            # 1. Fit Feature Pipeline (Sparsity Cap = 4 Orthogonal Features)
            feat_pipe = EconometricsFeaturePipeline(
                target_horizon=self.target_horizon,
                max_final_features=4,
            )
            feat_pipe.fit(X_train_raw, y_train)

            current_features = list(feat_pipe.selected_features_)
            selected_features_per_fold.append(current_features)
            trained_pipelines.append(feat_pipe)

            # 2. Transform Train Set
            X_train_transformed = feat_pipe.transform(X_train_raw)
            common_train_idx = X_train_transformed.index.intersection(y_train.index)
            X_tr = X_train_transformed.loc[common_train_idx].reindex(columns=current_features, fill_value=np.nan)
            y_tr = y_train.loc[common_train_idx].values

            if len(np.unique(y_tr)) < 2:
                continue

            # 3. Base Estimator: Degree-2 Polynomial Interaction Pipeline
            base_clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=self.c_reg,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ])

            # 4. Calibrated Classifier via Bounded Platt Scaling
            calibrated_model = TimeSeriesCalibrator(
                base_estimator=base_clf,
                method="sigmoid",
                calib_size=0.3,
            )
            calibrated_model.fit(X_tr, y_tr)
            trained_models.append(calibrated_model)

            # 5. Transform Test Set with Lookback Buffer
            buffer_start = max(0, test_idx[0] - self.buffer_days)
            df_test_buffer = df_ready.iloc[buffer_start : test_idx[-1] + 1].copy()

            X_test_buffer_raw = df_test_buffer.drop(columns=["target_label"], errors="ignore")
            X_test_buffer_trans = feat_pipe.transform(X_test_buffer_raw)

            X_test_pure = X_test_buffer_trans.loc[X_test_buffer_trans.index.isin(df_test.index)]
            common_test_idx = X_test_pure.index.intersection(df_test.index)

            X_te = X_test_pure.loc[common_test_idx].reindex(columns=current_features, fill_value=np.nan)
            y_te = df_test.loc[common_test_idx, "target_label"].astype(int).values

            # 6. Predict Probabilities and Evaluate Dynamic Thresholds
            train_probs = calibrated_model.predict_proba(X_tr)[:, 1]
            calibrated_test_probs = calibrated_model.predict_proba(X_te)[:, 1]

            train_probs_series = pd.Series(train_probs, index=common_train_idx).sort_index()
            test_probs_series = pd.Series(calibrated_test_probs, index=common_test_idx).sort_index()

            oos_dyn_threshold, y_pred_binary, bet_sizes = (
                self.threshold_engine.compute_threshold_and_signals(
                    train_probs_series, test_probs_series
                )
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

        df_oos_all = pd.concat(oos_predictions, axis=0) if oos_predictions else pd.DataFrame()
        artifacts = {
            "models": trained_models,
            "pipelines": trained_pipelines,
            "features": selected_features_per_fold,
            "predictions": oos_predictions,
        }

        return df_oos_all, artifacts