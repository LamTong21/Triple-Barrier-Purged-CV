"""
src/features/pipeline.py

Scikit-learn compliant feature engineering pipeline incorporating:
- Point-in-time stateless calculation and diagnostic payload integration.
- Empirical feature blacklisting (purging noise features).
- 4 Orthogonal Cluster Feature Selection (Flow, Volatility, Momentum, Microstructure).
"""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_pipeline.audit import Layer1DataIntegrity
from src.data_pipeline.topology import Layer2MicrostructureTopologies
from src.diagnostics.scanner import Stage2SingleAssetDGPScanner
from src.features.technical_strategies import FractionalMemoryStrategy
from src.features.routers import StatelessFeatureRouter
from src.features.lag_transform import Layer14LagTransformEngine
from src.features.consensus_selection import Stage4ConsensusSelectionRouter


class StatefulStage3FeatureEngine(BaseEstimator, TransformerMixin):
    """
    Applies parametric features (Fractional Memory d*) and non-parametric rolling
    volatility percentile regimes learned from the training set.
    """

    def __init__(self, payload: dict):
        self.payload = payload
        self.opt_d = payload.get("optimal_d", 1.0)

    def fit(self, df_train: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = df.copy()

        if 0.0 < self.opt_d < 1.0:
            frac_strat = FractionalMemoryStrategy(optimal_d=self.opt_d)
            feat = pd.concat([feat, frac_strat.construct(df)], axis=1)

        r = df["log_return"].fillna(0.0)
        roll_vol = r.rolling(20).std(ddof=1)
        vol_regime_rank = roll_vol.rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 60 else 0.5,
            raw=False,
        )
        feat["regime_vol_rank_60"] = vol_regime_rank
        feat["regime_is_high_vol"] = (vol_regime_rank >= 0.70).astype(float)

        return feat


class Stage4SelectionEngine(BaseEstimator, TransformerMixin):
    """Encapsulates Consensus Feature Selection and lag transforms."""

    def __init__(self, payload: dict, n_blocks: int = 8, block_ratio: float = 0.75, consensus_thresh: float = 0.60):
        self.payload = payload
        self.n_blocks = n_blocks
        self.block_ratio = block_ratio
        self.consensus_thresh = consensus_thresh
        self.optimal_lags_: Optional[Dict[str, int]] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        router = Stage4ConsensusSelectionRouter(
            self.payload,
            n_blocks=self.n_blocks,
            block_ratio=self.block_ratio,
            consensus_thresh=self.consensus_thresh,
        )
        _, optimal_lags, final_features = router.execute(X_train, y_train)
        self.optimal_lags_ = optimal_lags
        self.selected_features_ = list(final_features)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.optimal_lags_ is None or self.selected_features_ is None:
            raise ValueError("Engine is not fitted.")

        X_trans = Layer14LagTransformEngine.apply_volatility_scaled_lags(X, self.optimal_lags_)
        if "regime_p_high_vol" in X.columns and "regime_p_high_vol" in self.selected_features_:
            X_trans["regime_p_high_vol"] = X["regime_p_high_vol"]

        cols = [c for c in self.selected_features_ if c in X_trans.columns]
        return X_trans[cols]


class EconometricsFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Master pipeline:
    1. Audits and constructs stateless features with a Point-in-time shift lock.
    2. Runs Stage 2 DGP diagnostics.
    3. Transforms stateful memory variables.
    4. Filters out blacklisted features.
    5. Enforces diversity across 4 orthogonal econometric clusters (Flow, Volatility, Momentum, Microstructure).
    """

    QUARANTINE_COLS = (
        "target_label",
        "target_barrier_return",
        "barrier_touch_time",
        "local_volatility",
        "split_factor",
        "dividend",
    )

    # Features exhibiting negative out-of-sample permutation drops are strictly blacklisted
    FEATURE_BLACKLIST = (
        "geo_lower_shadow_momentum",
        "geo_lower_shadow_zscaled_lag2",
        "regime_vol_rank_60_momentum",
        "regime_vol_rank_60_zscaled_lag1",
        "volume_zscaled_lag1",
        "volume_momentum",
        "cmf_slope_5_zscaled_lag2",
        "kinematic_squeeze_ratio_momentum",
        "sin_month_zscaled_lag1",
        "sin_month_momentum",
        "liq_amihud_pct_rank_zscaled_lag2",
        "obv_slope_5_zscaled_lag1",
        "wavelet_energy_ratio_momentum",
        "geo_upper_shadow_zscaled_lag2",
    )

    FEATURE_CLUSTERS = {
        "flow": ["cmf_slope_5", "obv_slope_5", "rvol_5"],
        "volatility": ["vol_parkinson", "hl_log_range", "kinematic_squeeze_ratio"],
        "momentum": ["ret_3", "body_direction_momentum"],
        "microstructure": ["skewness_20", "wavelet_energy_ratio", "liq_amihud_pct_rank", "geo_upper_shadow"],
    }

    def __init__(
        self,
        target_horizon: int = 5,
        default_lags: Optional[Sequence[int]] = None,
        max_final_features: int = 4,
    ):
        self.target_horizon = target_horizon
        self.default_lags = tuple(default_lags) if default_lags is not None else (1, 3, 5, 10, 20)
        self.max_final_features = max_final_features

        self.routing_payload_: Optional[dict] = None
        self.selected_features_: Optional[List[str]] = None
        self._s2_engine: Optional[Stage2SingleAssetDGPScanner] = None
        self._s3_engine: Optional[StatefulStage3FeatureEngine] = None
        self._s4_engine: Optional[Stage4SelectionEngine] = None
        self._stateless_router: Optional[StatelessFeatureRouter] = None

    def _sanitize_input(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=list(self.QUARANTINE_COLS), errors="ignore")

    def _generate_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_audited, _ = Layer1DataIntegrity.run_audit(df)
        df_topo = Layer2MicrostructureTopologies.compute(df_audited)
        return self._stateless_router.execute(df_topo)

    def _select_diversified_features(self, candidate_features: List[str]) -> List[str]:
        chosen = []
        chosen_clusters = set()

        for feat in candidate_features:
            for cluster_name, prefixes in self.FEATURE_CLUSTERS.items():
                if cluster_name not in chosen_clusters:
                    if any(feat.startswith(prefix) for prefix in prefixes):
                        chosen.append(feat)
                        chosen_clusters.add(cluster_name)
                        break
            if len(chosen) >= self.max_final_features:
                break

        if len(chosen) < self.max_final_features:
            for feat in candidate_features:
                if feat not in chosen:
                    chosen.append(feat)
                if len(chosen) >= self.max_final_features:
                    break

        return chosen

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            if "target_label" not in X.columns:
                raise ValueError("Must provide 'y' or include 'target_label' in input DataFrame.")
            y_target = X["target_label"].copy()
        else:
            y_target = y.copy()

        y_target = y_target.rename("target").dropna()
        X_clean = self._sanitize_input(X).loc[y_target.index]

        self._stateless_router = StatelessFeatureRouter(
            default_lags=list(self.default_lags),
            enforce_shift=True,
        )
        X_stateless = self._generate_raw_features(X_clean)

        self._s2_engine = Stage2SingleAssetDGPScanner()
        self.routing_payload_ = self._s2_engine.execute(X_stateless)

        self._s3_engine = StatefulStage3FeatureEngine(self.routing_payload_)
        self._s3_engine.fit(X_stateless)
        X_stateful = self._s3_engine.transform(X_stateless)

        common_idx = X_stateful.dropna().index.intersection(y_target.index)
        X_train_final = X_stateful.loc[common_idx]
        y_train_final = y_target.loc[common_idx]

        self._s4_engine = Stage4SelectionEngine(self.routing_payload_)
        self._s4_engine.fit(X_train_final, y_train_final)

        raw_selected = self._s4_engine.selected_features_
        filtered = [f for f in raw_selected if f not in self.FEATURE_BLACKLIST]
        if not filtered:
            filtered = raw_selected

        self.selected_features_ = self._select_diversified_features(filtered)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._s4_engine is None or self.selected_features_ is None:
            raise ValueError("Pipeline is not fitted. Call fit() first.")

        X_clean = self._sanitize_input(X)
        X_stateless = self._generate_raw_features(X_clean)
        X_stateful = self._s3_engine.transform(X_stateless)
        X_stage4 = self._s4_engine.transform(X_stateful)
        return X_stage4.reindex(columns=self.selected_features_, fill_value=np.nan)