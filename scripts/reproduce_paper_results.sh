#!/usr/bin/env bash
# ==============================================================================
# scripts/reproduce_paper_results.sh
# 
# One-Click End-to-End Pipeline Reproducibility Script:
# 1. Download market data, audit bar geometry, and generate Triple Barrier targets
# 2. Execute Purged & Embargo CV with Bounded Platt Scaling & Rolling Percentile
# 3. Run comprehensive empirical diagnostics (Bias-Variance, Non-linearity, Brier)
# 4. Perform concurrent trade resolution and portfolio PnL audit
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo "========================================================================"
echo "  REPRODUCING RESEARCH PAPER RESULTS (MODELING 07.2)"
echo "========================================================================"

# Step 1: Data Preparation & Labeling
echo -e "\n[STEP 1/4] Running Data Download, Geometry Audit & Causal Labeling..."
python3 scripts/download_and_prepare.py \
    --portfolio_path "data/final_portfolio.csv" \
    --cache_dir "data/cache" \
    --output_dir "data/processed" \
    --target_horizon 5 \
    --pt 1.0 \
    --sl 1.0 \
    --vol_span 20 \
    --vol_floor 0.005

# Step 2: Purged & Embargo Cross-Validation Experiment
echo -e "\n[STEP 2/4] Running Purged & Embargo Cross-Validation..."
python3 scripts/run_cv_experiment.py \
    --input_path "data/processed/df_ready_for_cv.csv" \
    --output_dir "data/processed" \
    --artifacts_dir "checkpoints" \
    --n_splits 5 \
    --embargo_days 5 \
    --max_train_size 750 \
    --rolling_window 60 \
    --enter_percentile 65.0 \
    --c_reg 0.03

# Step 3: Comprehensive Model Diagnostics
echo -e "\n[STEP 3/4] Running Model Diagnostics (Bias-Variance, Permutation, Brier)..."
python3 scripts/run_diagnostics.py \
    --data_path "data/processed/df_ready_for_cv.csv" \
    --artifacts_path "checkpoints/cv_artifacts.pkl" \
    --buffer_days 150

# Step 4: Portfolio Backtest & PnL Audit
echo -e "\n[STEP 4/4] Auditing Portfolio Performance & Equity Drawdown..."
python3 scripts/run_portfolio_audit.py \
    --predictions_path "data/processed/df_oos_predictions.csv" \
    --output_dir "data/processed" \
    --fig_path "portfolio_performance.png" \
    --fee_rate 0.0015 \
    --slippage 0.0005 \
    --risk_free_rate 0.045

echo -e "\n========================================================================"
echo "  [✓] ALL EXPERIMENTS REPRODUCED SUCCESSFULLY!"
echo "  - Predictions Log : data/processed/df_oos_predictions.csv"
echo "  - Backtest Log    : data/processed/df_backtest_execution.csv"
echo "  - Equity Curve    : portfolio_performance.png"
echo "========================================================================"