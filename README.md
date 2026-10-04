# Optimizing Financial Return Dynamics Forecasting with Triple Barrier and Purged & Embargo Cross-Validation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Official implementation and empirical reproducibility suite for the research paper:
**"Tối Ưu Hóa Dự Báo Động Lực Học Lợi Nhuận Tài Chính Với Triple Barrier và Purged & Embargo Cross-Validation"**.

---

## 🔬 Abstract

Applying machine learning to financial time series introduces foundational econometric challenges: temporal look-ahead and autocorrelation leakage, probability calibration collapse under regime shifts, variance explosion driven by the curse of dimensionality, and structural payoff asymmetry between machine learning metrics and portfolio PnL.

This project implements an end-to-end framework integrating **Causal Triple Barrier Labeling**, **Purged & Embargoed Cross-Validation (López de Prado)**, **Clustered Consensus Feature Selection** with an orthogonal Sparsity Cap[cite: 1, 2], **Bounded Platt Scaling** calibration via L-BFGS-B[cite: 1, 2], and adaptive **Rolling Percentile Thresholding**[cite: 1, 2].

The methodology eliminates in-sample memorization (generalization gap $\vert{}Train - Test\vert{} \approx 0$) and achieves an out-of-sample (OOS) precision of **62.88%** across 1,070 evaluation sessions. However, auditing the portfolio with real-world execution frictions yields a net drawdown of **-19.68%**, caused by a distorted Win/Loss ratio of **0.37** imposed by the vertical time-out barrier. These findings empirically prove that a statistical classification edge cannot translate into cumulative positive drift without dynamic order and payoff management.

---

## 🏛️ System Architecture & Workflow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA AUDIT & CAUSAL TOPOLOGY                                                        │
│    Synthetic Total Return Index ──► Multi-Level L2 Depth ──► Causal EWMA Volatility    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
																						▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. CAUSAL TRIPLE BARRIER LABELING                                                      │
│    Upper Barrier (pt * σ) ──► Lower Barrier (sl * σ) ──► Vertical Barrier (h = 5 bars) │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
																						▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. PURGED & EMBARGO CROSS-VALIDATION                                                   │
│    Outer Time-Series Split ──► Purge Overlapping Spans ──► 5-Day Safety Embargo Buffer │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
																						▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. ORTHOGONAL CLUSTERED FEATURE SELECTION                                              │
│    VIF Filtering ──► Spearman Hierarchical Clustering ──► MI Medoids (Sparsity Cap: 4) │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
																						▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. CALIBRATION & DYNAMIC EXECUTION                                                     │
│    Bounded Platt Scaling ──► Rolling Percentile (W=60, Q=65%) ──► Z-Score Bet Sizing   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
																						▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 6. CAUSAL PORTFOLIO PnL AUDIT                                                          │
│    Concurrent Trade Resolution ──► 2-Way Frictions (0.40%) ──► Equity & Drawdown Curve │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## **📊 Empirical Evolution & Key Results**

### **1. Methodology Ablation (Capture Evolution)**

Through 6 structured iterations, the system systematically addressed each econometric failure mode:

| **Stage** | **Core Intervention** | **Observed Anomaly** | **OOS AUC** | **Precision (Class 1)** | **OOS Order Flow Allocation** |
| --- | --- | --- | --- | --- | --- |
| **Stage 1 (Baseline)** | Standard Purging, L2 Logistic, 2nd Derivatives | Acceleration dominates; negative Permutation drops | 0.4927 | ~50.8% | Out-of-sample divergence |
| **Stage 2 (L1 Penalty)** | Embargo added, strict L1 penalty (C=0.05) | *Resolution Collapse:* w0, all-or-nothing folds | 0.4611 | 50.2% | Folds 1, 2, 5: 0%; Folds 3, 4: 100% |
| **Stage 3 (Feature Clean)** | Purged all *_accel features, locked 5 raw vars | OOS AUC exceeded random walk; interaction test rose | 0.5218 | 55.67% | Fold 4 pinned at 100% via static threshold |
| **Stage 4 (Rolling)** | Shifted to Rolling Percentile (W=60) | Resolved Fold 4 saturation; Precision reached ~60% | 0.5218 | 59.94% | Base estimator logit saturation (T*=5.0) |
| **Stage 5 (Deg-2 Poly)** | Expanded 10 base features to 55 interactions | *Curse of Dimensionality:* severe overfitting (Gap=0.21) | 0.5123 | 62.68% | Fold 1 choked (14%) by absolute floor |
| **Stage 6 (Final Model)** | **4 Orthogonal Clusters (10 cols), Bounded Platt** | **Overfitting eliminated (Gap0), Peak Precision** | **0.5494** | **62.60%** | **Optimal balance: 32.7% – 37.8% across folds** |

### **2. Resolution of the Win Rate vs. PnL Paradox**

Even with a **62.88% Out-of-Sample Win Rate**, the strategy suffers a **-19.68%** net loss due to severe payoff asymmetry ($R=0.37$):

$$
\mathbb{E}[\text{Return}] = P_{\text{win}} \times \overline{\text{Gain}} - (1 - P_{\text{win}}) \times \overline{\text{Loss}} - \text{Cost} = -0.1385 \cdot \overline{\text{Loss}} - \text{Cost} < 0
$$

| **Partition** | **Trades** | **Win Rate** | **Profit Factor** | **Win / Loss Ratio** | **Cumulative Return** | **Sharpe Ratio** | **Max Drawdown** | **Avg Position Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Fold 1** | 40 | 57.50% | 0.20 | 0.15 | -12.36% | -1.94 | -13.73% | 0.37 |
| **Fold 2** | 51 | 62.75% | 0.83 | 0.49 | -1.22% | -2.23 | -3.27% | 0.28 |
| **Fold 3** | 53 | 64.15% | 1.00 | 0.56 | -0.05% | -1.82 | -2.43% | 0.22 |
| **Fold 4** | 45 | 66.67% | 0.62 | 0.31 | -6.23% | -1.56 | -7.66% | 0.40 |
| **Fold 5** | 40 | 62.50% | 0.91 | 0.54 | -1.01% | -0.90 | -6.57% | 0.35 |
| **Full Portfolio** | **229** | **62.88%** | **0.62** | **0.37** | **-19.68%** | **-1.46** | **-23.83%**
[cite: 1] | **0.32**
[cite: 1] |

**Why did this happen?**

1. **Vertical Barrier Premature Cutoff:** Liquidating trades at h=5 bars caps profitable trades (1.0), while trades lingering near stop-loss trigger full losses[cite: 1].
2. **Two-Way Structural Friction:** A round-trip transaction fee of $0.40\%$ ($2 \times 0.20\%$) degrades smaller winning margins during low-volatility regimes[cite: 1].

## **📂 Repository Organization**

```
quant-financial-dynamics-cv/
├── .github/workflows/tests.yml       # Continuous integration & test pipeline
├── configs/                          # Experiment configuration files
│   ├── base_config.yaml              # Environment, paths, and random seeds
│   ├── triple_barrier.yaml           # Labeling parameters (pt, sl, horizon)
│   ├── cv_config.yaml                # Purged CV, features, and model hyperparameters
│   └── execution_backtest.yaml       # Friction rates and trade resolution settings
├── data/
│   ├── final_portfolio.csv           # Asset portfolio weights
│   └── README.md                     # Data storage specifications
├── scripts/                          # Executable CLI experiment scripts
│   ├── download_and_prepare.py       # Data fetcher, geometry auditor, and labeler
│   ├── run_cv_experiment.py          # Outer Purged CV and calibration loop
│   ├── run_diagnostics.py            # Bias-Variance, Interaction, and Permutation tests
│   ├── run_portfolio_audit.py        # Causal trade resolution and PnL backtester
│   └── reproduce_paper_results.sh    # One-click master bash execution script
├── src/                              # Production source package
│   ├── data_pipeline/                # Ingestion, bar geometry audits, and meta-labeling
│   ├── diagnostics/                  # Econometric tests (Hurst, ARCH, BNS, BDS)
│   ├── features/                     # Feature builders, routers, and consensus selection
│   ├── modeling/                     # Purged splitters, calibrators, and thresholding
│   ├── backtest/                     # Concurrent trade resolver and risk metrics
│   └── utils/                        # Plotting utilities and Murphy decomposition
├── tests/                            # Pytest guardrails against temporal leakage
├── Makefile                          # Task automation commands
├── environment.yml                   # Conda virtual environment specification
└── requirements.txt                  # Python pip dependencies
```

## **⚡ Quickstart & Reproducibility**

### **1. Installation**

Clone the repository and set up the Conda environment:

```bash
git clone https://github.com/your-username/quant-financial-dynamics-cv.git 
cd quant-financial-dynamics-cv

# Create and activate environment
conda env create -f environment.yml
conda activate quant_cv

# Or install via pip
pip install -r requirements.txt
```

### **2. Running Automated Tests**

Run the unit tests to verify that no temporal leakage exists:

```bash
pytest -v tests/ --cov=src --cov-report=term-missing 
```

### **3. One-Click Reproduction**

To run the complete pipeline from raw data acquisition to PnL report generation:

```bash
bash scripts/reproduce_paper_results.sh
```

Or execute tasks individually using `make`:

```bash
make data       # Ingests history and applies causal Triple Barrier labeling
make train      # Executes Purged & Embargo CV with Bounded Platt Scaling
make diag       # Runs Bias-Variance, Interaction, and Permutation audits
make backtest   # Resolves overlapping trades and audits portfolio PnL
```

## **🛠️ Step-by-Step Module Usage**

### **Data Ingestion & Causal Labeling**

```bash
python scripts/download_and_prepare.py \
		--portfolio_path "data/final_portfolio.csv" \
		--target_horizon 5 \
		--pt 1.0 \
		--sl 1.0 \
		--vol_span 20
```

### **Model Training & Dynamic Sizing**

```bash
python scripts/run_cv_experiment.py \
		--input_path "data/processed/df_ready_for_cv.csv" \
		--n_splits 5 \
		--embargo_days 5 \
		--rolling_window 60 \
		--enter_percentile 65.0 \
		--c_reg 0.03 
```

### **Empirical Diagnostics**

```bash
python scripts/run_diagnostics.py \
		--data_path "data/processed/df_ready_for_cv.csv" \
		--artifacts_path "checkpoints/cv_artifacts.pkl" 
```

### **Financial Audit & Visualization**

```bash
python scripts/run_portfolio_audit.py \
		--predictions_path "data/processed/df_oos_predictions.csv" \
		--fee_rate 0.0015 \
		--slippage 0.0005 \
		--fig_path "portfolio_performance.png" 
```

## **🔮 Roadmap & Actionable Extensions**

- [] **Asymmetric Barriers:** Reconfigure Triple Barrier thresholds with a minimum Risk:Reward ratio of 1.5 (pt=1.5,sl=1.0) to raise the Win/Loss ratio above 0.80[cite: 1].
- **[] Dynamic Trailing Stops:** Replace static 5-bar timeouts with volatility-scaled trailing stops (e.g., Chandelier or ATR exits) to lock in intermediate gains before time-outs occur[cite: 1].
- **[] Linear Interaction Pruning:** Completely drop degree-2 polynomial interactions in favor of a strictly linear estimator across the 4 orthogonal clusters to prevent variance inflation[cite: 1].

## **📄 Citation**

```bash
@article{financial_dynamics_cv_2026,
		title   = {Tối Ưu Hóa Dự Báo Động Lực Học Lợi Nhuận Tài Chính Với Triple Barrier và Purged & Embargo Cross-Validation},
		author  = {Research Team},
		journal = {Quantitative Finance & Machine Learning Suite},
		year    = {2026}
} 
```

## **📜 License**

This project is licensed under the Apache License 2.0.