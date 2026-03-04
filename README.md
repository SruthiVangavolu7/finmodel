# ML Equity Signal Generator

A production-oriented machine learning research pipeline for predicting next-day equity return direction using engineered technical features and strict time-series evaluation controls. The project is designed to mirror quant research workflow: cached market data ingestion, leakage-safe feature generation, walk-forward out-of-fold prediction, strategy simulation with transaction costs, and diagnostics ready for portfolio-style iteration.

## Architecture

```text
yfinance + cache
      |
      v
Feature Engineering (shifted to avoid leakage)
      |
      v
Forward Label Construction
      |
      v
Walk-Forward CV (rolling train/test windows)
      |
      v
Model Training + OOF Predictions
      |
      v
Backtest (cost-aware long/short rules)
      |
      v
Metrics + Charts + CSV Artifacts in /results
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Visual Dashboard

After generating results, launch the local dashboard:

```bash
streamlit run app.py
```

This opens a browser view with KPI cards, all generated charts, and interactive tables for predictions, trades, and daily PnL.

## Methodology

This project uses walk-forward cross validation rather than random train/test splits. Each fold trains on a historical rolling window and tests on the next unseen block of dates, which preserves temporal order and prevents contamination from future data. Features are shifted by one day so signals at time `T` only use information available before `T`, reducing lookahead bias in both modeling and backtesting.

## Results

Running `python main.py` generates model diagnostics, backtest outputs, and plots under `results/`, including equity curve, drawdown, confusion matrix, monthly IC, feature importances, and fold timeline visualization.

## Limitations & Future Work

Current signals are built from price and volume technical features only. Future work can extend the feature set with earnings and fundamentals, NLP sentiment/alternative data, or explicit market regime detection to improve robustness across changing macro environments.
