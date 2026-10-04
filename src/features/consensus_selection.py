"""
src/features/consensus_selection.py

Consensus Feature Selection via Moving Block Subsampling, Hierarchical Spearman Clustering,
and Sparsity Capping (3-5 features).
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from statsmodels.tsa.stattools import grangercausalitytests

from src.features.lag_transform import Layer14LagTransformEngine


class Layer12RedundancyControl:
    """Multicollinearity control using Variance Inflation Factor (VIF) and Pearson filtering."""

    @staticmethod
    def linear_vif_prune(X: pd.DataFrame, threshold: float = 0.80, max_vif: float = 4.0) -> pd.DataFrame:
        if X.empty or X.shape[1] <= 1:
            return X.copy()

        X_num = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        X_num = X_num.bfill().ffill().fillna(0.0).astype(np.float64)

        variances = X_num.var(axis=0, ddof=0)
        X_curr = X_num[variances[variances > 1e-8].index].copy()

        if X_curr.shape[1] <= 1:
            return X_curr

        corr_matrix = X_curr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        X_curr = X_curr.drop(columns=to_drop)

        while X_curr.shape[1] > 1:
            cols = list(X_curr.columns)
            corr_mat = np.corrcoef(X_curr.values, rowvar=False)
            try:
                inv_corr = np.linalg.pinv(corr_mat)
                vifs = np.diag(inv_corr).tolist()
            except Exception:
                break

            max_vif_val = max(vifs)
            if max_vif_val > max_vif:
                drop_idx = int(np.argmax(vifs))
                X_curr = X_curr.drop(columns=[cols[drop_idx]])
            else:
                break

        return X_curr


class Layer13PredictiveDiagnostics:
    """Predictive causality screening via Granger Causality and Mutual Information."""

    @staticmethod
    def causality_screen(X: pd.DataFrame, y: pd.Series, max_lag: int = 2, p_threshold: float = 0.10) -> Tuple[List[str], Dict[str, int]]:
        valid_features = []
        optimal_lags = {}

        df_test = pd.concat([y, X], axis=1).dropna()
        if df_test.empty:
            return valid_features, optimal_lags

        y_col = df_test.columns[0]
        y_vals = df_test[y_col].values

        for col in X.columns:
            test_data = df_test[[y_col, col]]
            best_lag = 1
            is_causal = False

            # Granger Causality
            best_p = 1.0
            try:
                gc_res = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag + 1):
                    p_val = gc_res[lag][0]["ssr_ftest"][1]
                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag
                if best_p < p_threshold:
                    is_causal = True
            except Exception:
                pass

            # Lagged Mutual Information
            if not is_causal:
                best_mi = 0.0
                for lag in range(1, max_lag + 1):
                    x_lagged = df_test[col].shift(lag).fillna(0.0).values
                    mi_score = mutual_info_classif(x_lagged.reshape(-1, 1), y_vals, random_state=42)[0]
                    if mi_score > best_mi:
                        best_mi = mi_score
                        best_lag = lag
                if best_mi > 0.01:
                    is_causal = True

            if is_causal:
                valid_features.append(col)
                optimal_lags[col] = best_lag

        return valid_features, optimal_lags


class Stage4ConsensusSelectionRouter:
    """
    Subsamples multiple moving blocks on the training set, performs hierarchical
    clustering, computes mutual information medoids, and enforces a strict Sparsity Cap.
    """

    def __init__(
        self,
        payload: dict,
        n_blocks: int = 8,
        block_ratio: float = 0.75,
        consensus_thresh: float = 0.50,
        min_features: int = 3,
        max_features: int = 5,
        cluster_corr_threshold: float = 0.60,
    ):
        self.payload = payload
        self.n_blocks = n_blocks
        self.block_ratio = block_ratio
        self.consensus_thresh = consensus_thresh
        self.min_features = min_features
        self.max_features = max_features
        self.cluster_corr_threshold = cluster_corr_threshold

    def _cluster_and_select_best(self, X_sub: pd.DataFrame, y_sub: pd.Series) -> List[str]:
        if X_sub.shape[1] <= self.min_features:
            return list(X_sub.columns)

        corr_matrix = X_sub.corr(method="spearman").abs().fillna(0.0)
        np.fill_diagonal(corr_matrix.values, 1.0)

        dist_matrix = np.sqrt(np.clip(1.0 - corr_matrix.values, 0.0, 1.0))
        dist_sym = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_sym, 0.0)

        condensed_dist = squareform(dist_sym, checks=False)
        link = linkage(condensed_dist, method="average")

        dist_cut = np.sqrt(1.0 - self.cluster_corr_threshold)
        cluster_labels = fcluster(link, t=dist_cut, criterion="distance")

        mi_scores = mutual_info_classif(X_sub.values, y_sub.values, discrete_features=False, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_sub.columns)

        selected_features = []
        for c_id in np.unique(cluster_labels):
            cluster_cols = X_sub.columns[cluster_labels == c_id]
            best_col = mi_series.loc[cluster_cols].idxmax()
            selected_features.append(best_col)

        return selected_features

    def _run_single_filter_pipeline(self, X_sub: pd.DataFrame, y_sub: pd.Series) -> Tuple[List[str], Dict[str, int]]:
        X_pruned = Layer12RedundancyControl.linear_vif_prune(X_sub, threshold=0.80, max_vif=4.0)
        cluster_kept = self._cluster_and_select_best(X_pruned, y_sub)
        X_clustered = X_pruned[cluster_kept]

        try:
            valid_feats, lags = Layer13PredictiveDiagnostics.causality_screen(X_clustered, y_sub, max_lag=2, p_threshold=0.10)
        except Exception:
            valid_feats, lags = list(X_clustered.columns), {c: 1 for c in X_clustered.columns}

        if len(valid_feats) < self.min_features:
            valid_feats = list(X_clustered.columns)

        return valid_feats, lags

    def execute(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
        X_mat = X_train.copy()
        for c in X_mat.columns:
            X_mat[c] = pd.to_numeric(X_mat[c], errors="coerce")

        valid_cols = [c for c in X_mat.columns if X_mat[c].isna().sum() < len(X_mat) * 0.25]
        X_mat = X_mat[valid_cols].bfill().ffill().dropna(axis=1)

        common_idx = X_mat.index.intersection(y_train.dropna().index)
        X_mat = X_mat.loc[common_idx]
        y_mat = y_train.loc[common_idx]

        n_samples = len(X_mat)
        block_len = max(int(n_samples * self.block_ratio), 40)

        feature_votes = defaultdict(int)
        feature_lags_collected = defaultdict(list)

        max_start = n_samples - block_len
        step = max(1, max_start // max(1, self.n_blocks - 1)) if max_start > 0 else 1
        block_starts = sorted(list({min(i * step, max_start) for i in range(self.n_blocks)}))

        for start in block_starts:
            end = start + block_len
            X_b = X_mat.iloc[start:end]
            y_b = y_mat.iloc[start:end]

            if len(np.unique(y_b)) < 2:
                continue

            selected_b, lags_b = self._run_single_filter_pipeline(X_b, y_b)
            for feat in selected_b:
                feature_votes[feat] += 1
                if feat in lags_b:
                    feature_lags_collected[feat].append(lags_b[feat])

        total_valid_blocks = len(block_starts)
        vote_threshold = int(np.ceil(total_valid_blocks * self.consensus_thresh))
        sorted_by_votes = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        consensus_candidates = [f for f, count in sorted_by_votes if count >= vote_threshold]

        if len(consensus_candidates) < self.min_features:
            consensus_candidates = [f for f, _ in sorted_by_votes[: self.min_features]]

        final_candidate_df = X_mat[consensus_candidates]
        final_uncorrelated = self._cluster_and_select_best(final_candidate_df, y_mat)

        # ENFORCE STRICT SPARSITY CAP: TOP 3 TO 5 FEATURES
        final_features = sorted(final_uncorrelated, key=lambda x: feature_votes[x], reverse=True)[: self.max_features]

        final_lags = {}
        for f in final_features:
            if feature_lags_collected[f]:
                final_lags[f] = int(np.clip(np.median(feature_lags_collected[f]), 1, 2))
            else:
                final_lags[f] = 1

        X_transformed = Layer14LagTransformEngine.apply_volatility_scaled_lags(X_mat[final_features], final_lags)
        return X_transformed, final_lags, list(X_transformed.columns)