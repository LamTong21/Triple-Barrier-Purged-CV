.PHONY: help env install data train diag backtest reproduce test lint clean

PYTHON := python3
PIP := pip

help:
	@echo "Quantitative Financial Dynamics - Command Palette"
	@echo "==================================================="
	@echo "make env         : Create Conda environment (quant_cv)"
	@echo "make install     : Install all dependencies from requirements.txt"
	@echo "make data        : Download, audit bar geometry, and generate Triple Barrier labels"
	@echo "make train       : Run Purged & Embargo CV with Bounded Platt Scaling"
	@echo "make diag        : Run Bias-Variance, Non-linearity, and Calibration diagnostics"
	@echo "make backtest    : Resolve concurrent holding periods and audit Portfolio PnL"
	@echo "make reproduce   : Run end-to-end pipeline reproducing paper results"
	@echo "make test        : Run pytest test suite for leakage and calibration guards"
	@echo "make lint        : Lint codebase with Ruff"
	@echo "make clean       : Clean temporary files, caches, and checkpoints"

env:
	conda env create -f environment.yml

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/download_and_prepare.py \
		--portfolio_path "data/final_portfolio.csv" \
		--cache_dir "data/cache" \
		--output_dir "data/processed" \
		--target_horizon 5 \
		--pt 1.0 \
		--sl 1.0 \
		--vol_span 20 \
		--vol_floor 0.005

train:
	$(PYTHON) scripts/run_cv_experiment.py \
		--input_path "data/processed/df_ready_for_cv.csv" \
		--output_dir "data/processed" \
		--artifacts_dir "checkpoints" \
		--n_splits 5 \
		--embargo_days 5 \
		--max_train_size 750 \
		--rolling_window 60 \
		--enter_percentile 65.0 \
		--c_reg 0.03

diag:
	$(PYTHON) scripts/run_diagnostics.py \
		--data_path "data/processed/df_ready_for_cv.csv" \
		--artifacts_path "checkpoints/cv_artifacts.pkl" \
		--buffer_days 150

backtest:
	$(PYTHON) scripts/run_portfolio_audit.py \
		--predictions_path "data/processed/df_oos_predictions.csv" \
		--output_dir "data/processed" \
		--fig_path "portfolio_performance.png" \
		--fee_rate 0.0015 \
		--slippage 0.0005 \
		--risk_free_rate 0.045

reproduce:
	bash scripts/reproduce_paper_results.sh

test:
	pytest -v tests/ --cov=src --cov-report=term-missing

lint:
	ruff check . --select=E9,F63,F7,F82 --show-source
	ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf data/processed/*.csv checkpoints/*.pkl portfolio_performance.png