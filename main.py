"""Entrypoint for the end-to-end ML equity signal generation pipeline."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

import config
from src.backtest import run_backtest
from src.data_loader import load_price_data
from src.features import compute_features, validate_no_lookahead
from src.labels import build_labels
from src.model import SignalModel
from src.pipeline import WalkForwardSplitter, combine_oof_predictions
from src.visualize import generate_all_plots


def setup_logging() -> None:
    """Configure process-wide logging format and level."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_model_dataset(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create combined ticker-level feature/label dataset aligned by date."""
    all_rows: list[pd.DataFrame] = []

    for ticker, df in price_data.items():
        feats = compute_features(
            df=df,
            feature_windows=config.FEATURE_WINDOWS,
            rsi_windows=config.RSI_WINDOWS,
            bollinger_window=config.BOLLINGER_WINDOW,
            bollinger_std_multiplier=config.BOLLINGER_STD_MULTIPLIER,
            volume_window=config.VOLUME_WINDOW,
            lag_returns=config.LAG_RETURNS,
        )
        validate_no_lookahead(
            features_df=feats,
            prices_df=df,
            label_horizon=config.LABEL_HORIZON,
            threshold=config.LOOKAHEAD_CORR_THRESHOLD,
        )

        labels = build_labels(
            df=df,
            label_horizon=config.LABEL_HORIZON,
            label_threshold=config.LABEL_THRESHOLD,
        )

        merged = feats.join(labels, how="inner")
        merged = merged.join(df[["Open", "Close"]], how="inner")
        merged["ticker"] = ticker
        merged["date"] = merged.index
        all_rows.append(merged)

    if not all_rows:
        raise ValueError("No ticker datasets available after feature/label construction.")

    dataset = pd.concat(all_rows, axis=0, ignore_index=True)
    dataset = dataset.replace([float("inf"), float("-inf")], pd.NA).dropna()
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    return dataset


def format_metric(metrics: pd.DataFrame, name: str, default: float = float("nan")) -> float:
    """Extract one aggregate metric by name from a metrics DataFrame."""
    row = metrics[(metrics["scope"] == "aggregate") & (metrics["metric"] == name)]
    if row.empty:
        return default
    return float(row["value"].iloc[0])


def format_backtest_metric(metrics: pd.DataFrame, name: str, default: float = float("nan")) -> float:
    """Extract one metric by name from backtest metrics DataFrame."""
    row = metrics[metrics["metric"] == name]
    if row.empty:
        return default
    try:
        return float(row["value"].iloc[0])
    except ValueError:
        return default


def main() -> None:
    """Run data loading, modeling, backtest simulation, and visualization."""
    setup_logging()

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading data...")
    price_data = load_price_data(
        tickers=config.TICKERS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        raw_data_dir=config.RAW_DATA_DIR,
        cache_max_age_days=config.CACHE_MAX_AGE_DAYS,
        price_columns=config.PRICE_COLUMNS,
        download_max_retries=config.DOWNLOAD_MAX_RETRIES,
        download_backoff_seconds=config.DOWNLOAD_BACKOFF_SECONDS,
        download_request_timeout=config.DOWNLOAD_REQUEST_TIMEOUT,
        download_pause_seconds=config.DOWNLOAD_PAUSE_SECONDS,
    )
    if not price_data:
        raise ValueError("No ticker data available. Cannot run pipeline.")

    print("[2/6] Engineering features...")
    print("[3/6] Constructing labels...")
    dataset = build_model_dataset(price_data)

    feature_cols = [
        col
        for col in dataset.columns
        if col
        not in {
            "date",
            "ticker",
            "label",
            "forward_return",
            "Open",
            "Close",
        }
    ]

    print(f"[4/6] Running walk-forward cross validation ({config.N_SPLITS} folds)...")
    splitter = WalkForwardSplitter(
        train_window=config.TRAIN_WINDOW,
        test_window=config.TEST_WINDOW,
        n_splits=config.N_SPLITS,
    )

    fold_ranges = splitter.fold_ranges(pd.DatetimeIndex(sorted(dataset["date"].unique())))
    for fr in fold_ranges:
        print(
            f"    Fold {fr.fold}/{config.N_SPLITS}: "
            f"Train {fr.train_start.date()} -> {fr.train_end.date()} | "
            f"Test {fr.test_start.date()} -> {fr.test_end.date()}"
        )

    model = SignalModel(
        model_type=config.MODEL_TYPE,
        random_seed=config.RANDOM_SEED,
        xgb_params=config.XGB_PARAMS,
        rf_params=config.RF_PARAMS,
        logistic_params=config.LOGISTIC_PARAMS,
    )

    model_outputs = model.run_walk_forward(
        data=dataset,
        feature_cols=feature_cols,
        splitter=splitter,
        classification_threshold=config.CLASSIFICATION_THRESHOLD,
    )
    predictions = combine_oof_predictions([model_outputs.predictions])

    print("[5/6] Simulating backtest...")
    if config.BENCHMARK_TICKER not in price_data:
        raise ValueError(f"Benchmark ticker {config.BENCHMARK_TICKER} not present in loaded data.")

    backtest = run_backtest(
        predictions=predictions,
        benchmark_prices=price_data[config.BENCHMARK_TICKER],
        long_threshold=config.LONG_THRESHOLD,
        short_threshold=config.SHORT_THRESHOLD,
        label_horizon=config.LABEL_HORIZON,
        transaction_cost_bps=config.TRANSACTION_COST_BPS,
        initial_capital=config.INITIAL_CAPITAL,
        risk_free_rate=config.RISK_FREE_RATE,
        trading_days_per_year=config.TRADING_DAYS_PER_YEAR,
        bps_denominator=config.BPS_DENOMINATOR,
    )

    metrics_out = pd.concat(
        [
            model_outputs.metrics,
            backtest.metrics.assign(scope="backtest", fold=pd.NA),
        ],
        ignore_index=True,
    )
    metrics_out.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    predictions.to_csv(config.RESULTS_DIR / "predictions_oof.csv", index=False)
    backtest.trades.to_csv(config.RESULTS_DIR / "trades.csv", index=False)
    backtest.daily_pnl.to_csv(config.RESULTS_DIR / "daily_pnl.csv", index=False)
    model_outputs.monthly_ic.rename("monthly_ic").to_csv(config.RESULTS_DIR / "monthly_ic.csv")
    if not model_outputs.feature_importance.empty:
        model_outputs.feature_importance.rename("importance").to_csv(config.RESULTS_DIR / "feature_importance.csv")

    print("[6/6] Generating charts...")
    plt.style.use(config.PLOT_STYLE)
    generate_all_plots(
        predictions=predictions,
        daily_pnl=backtest.daily_pnl,
        feature_importance=model_outputs.feature_importance,
        monthly_ic=model_outputs.monthly_ic,
        fold_ranges=model_outputs.fold_ranges,
        results_dir=config.RESULTS_DIR,
        dpi=config.PLOT_DPI,
        strategy_color=config.COLOR_STRATEGY,
        benchmark_color=config.COLOR_BENCHMARK,
        accent_color=config.COLOR_ACCENT,
        positive_color=config.COLOR_POSITIVE,
        negative_color=config.COLOR_NEGATIVE,
        figsize_equity=config.FIGSIZE_EQUITY,
        figsize_drawdown=config.FIGSIZE_DRAWDOWN,
        figsize_importance=config.FIGSIZE_IMPORTANCE,
        figsize_monthly_ic=config.FIGSIZE_MONTHLY_IC,
        figsize_confusion=config.FIGSIZE_CONFUSION,
        figsize_signal_dist=config.FIGSIZE_SIGNAL_DIST,
        figsize_walk_forward_base=config.FIGSIZE_WALK_FORWARD_BASE,
        walk_forward_height_per_fold=config.WALK_FORWARD_HEIGHT_PER_FOLD,
        walk_forward_ticks=config.WALK_FORWARD_TICKS,
        confusion_cmap=config.CONFUSION_CMAP,
        signal_hist_bins=config.HIST_BINS,
        signal_hist_negative_color=config.COLOR_BENCHMARK,
        signal_hist_positive_color=config.COLOR_STRATEGY,
    )

    auc = format_metric(model_outputs.metrics, "auc_roc")
    ic = format_metric(model_outputs.metrics, "ic")
    sharpe = format_backtest_metric(backtest.metrics, "annualized_sharpe")
    total_return = format_backtest_metric(backtest.metrics, "total_return")
    max_dd = format_backtest_metric(backtest.metrics, "max_drawdown")
    win_rate = format_backtest_metric(backtest.metrics, "win_rate")
    num_trades = int(format_backtest_metric(backtest.metrics, "num_trades", 0.0))

    print()
    print("=" * config.SUMMARY_WIDTH)
    print(" MODEL PERFORMANCE SUMMARY")
    print("=" * config.SUMMARY_WIDTH)
    print(f" AUC-ROC:        {auc:0.3f}")
    print(f" Mean IC:        {ic:0.3f}")
    print(f" Sharpe Ratio:   {sharpe:0.2f}")
    print(f" Total Return:   {total_return * 100:0.1f}%")
    print(f" Max Drawdown:   {max_dd * 100:0.1f}%")
    print(f" Win Rate:       {win_rate * 100:0.1f}%")
    print(f" Num Trades:     {num_trades}")
    print("=" * config.SUMMARY_WIDTH)
    print(f"Results saved to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
