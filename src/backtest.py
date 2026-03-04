"""Backtesting utilities for out-of-fold ML signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest artifacts and performance statistics."""

    daily_pnl: pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.DataFrame


def _compute_drawdown_stats(equity_curve: pd.Series) -> tuple[float, int, pd.Timestamp | None, pd.Timestamp | None]:
    """Return max drawdown magnitude, max duration, and peak-to-trough dates."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0

    if drawdown.empty:
        return 0.0, 0, None, None

    trough_date = drawdown.idxmin()
    max_drawdown = float(drawdown.min())
    peak_date = equity_curve.loc[:trough_date].idxmax() if trough_date is not None else None

    duration = 0
    max_duration = 0
    for value in drawdown:
        if value < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    return max_drawdown, max_duration, peak_date, trough_date


def run_backtest(
    predictions: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    long_threshold: float,
    short_threshold: float,
    label_horizon: int,
    transaction_cost_bps: float,
    initial_capital: float,
    risk_free_rate: float,
    trading_days_per_year: int,
    bps_denominator: float,
) -> BacktestResult:
    """Simulate long/short trades from OOF probabilities and compute performance metrics."""
    preds = predictions.copy()
    preds["date"] = pd.to_datetime(preds["date"])

    preds["signal"] = 0
    preds.loc[preds["pred_prob"] > long_threshold, "signal"] = 1
    preds.loc[preds["pred_prob"] < short_threshold, "signal"] = -1

    preds = preds.sort_values(["ticker", "date"]).reset_index(drop=True)

    cost_per_trade = 2.0 * (transaction_cost_bps / bps_denominator)
    trade_rows: list[dict] = []

    for ticker, grp in preds.groupby("ticker", sort=False):
        g = grp.sort_values("date").reset_index(drop=True)

        # Entry is next open; exit is open after holding horizon days.
        entry_open = g["Open"].shift(-1)
        exit_open = g["Open"].shift(-(1 + label_horizon))
        raw_ret = (exit_open / entry_open) - 1.0

        g["entry_date"] = g["date"].shift(-1)
        g["exit_date"] = g["date"].shift(-(1 + label_horizon))
        g["trade_return_raw"] = raw_ret
        g["trade_return_net"] = g["signal"] * raw_ret - cost_per_trade

        tradable = g[(g["signal"] != 0) & g["trade_return_net"].notna()].copy()
        for _, row in tradable.iterrows():
            trade_rows.append(
                {
                    "ticker": ticker,
                    "signal_date": row["date"],
                    "entry_date": row["entry_date"],
                    "exit_date": row["exit_date"],
                    "position": int(row["signal"]),
                    "pred_prob": float(row["pred_prob"]),
                    "raw_return": float(row["trade_return_raw"]),
                    "net_return": float(row["trade_return_net"]),
                }
            )

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        daily_index = pd.DatetimeIndex(sorted(preds["date"].unique()))
        daily = pd.DataFrame(index=daily_index)
        daily["strategy_return"] = 0.0
    else:
        daily = (
            trades.groupby("signal_date")["net_return"]
            .mean()
            .rename("strategy_return")
            .to_frame()
            .sort_index()
        )

    all_dates = pd.DatetimeIndex(sorted(preds["date"].unique()))
    daily = daily.reindex(all_dates).fillna(0.0)

    daily["strategy_equity"] = initial_capital * (1.0 + daily["strategy_return"]).cumprod()

    benchmark = benchmark_prices.copy().sort_index()
    benchmark = benchmark.reindex(all_dates).ffill().dropna()
    daily = daily.loc[benchmark.index].copy()
    benchmark_returns = benchmark["Close"].pct_change().fillna(0.0)
    daily["benchmark_return"] = benchmark_returns
    daily["benchmark_equity"] = initial_capital * (1.0 + daily["benchmark_return"]).cumprod()

    max_dd, dd_duration, dd_peak, dd_trough = _compute_drawdown_stats(daily["strategy_equity"])
    total_return = daily["strategy_equity"].iloc[-1] / initial_capital - 1.0

    num_days = max(len(daily), 1)
    ann_return = (1.0 + total_return) ** (trading_days_per_year / num_days) - 1.0

    strategy_excess = daily["strategy_return"] - (risk_free_rate / trading_days_per_year)
    ann_vol = daily["strategy_return"].std(ddof=0) * np.sqrt(trading_days_per_year)
    sharpe = float(strategy_excess.mean() / daily["strategy_return"].std(ddof=0) * np.sqrt(trading_days_per_year)) if ann_vol > 0 else np.nan
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else np.nan

    win_rate = float((trades["net_return"] > 0).mean()) if not trades.empty else np.nan
    num_trades = int(len(trades))

    metrics_rows = [
        {"metric": "total_return", "value": float(total_return)},
        {"metric": "annualized_return", "value": float(ann_return)},
        {"metric": "annualized_sharpe", "value": float(sharpe)},
        {"metric": "max_drawdown", "value": float(max_dd)},
        {"metric": "drawdown_duration_days", "value": float(dd_duration)},
        {"metric": "win_rate", "value": float(win_rate) if not np.isnan(win_rate) else np.nan},
        {"metric": "num_trades", "value": float(num_trades)},
        {"metric": "calmar_ratio", "value": float(calmar) if not np.isnan(calmar) else np.nan},
        {"metric": "max_drawdown_peak_date", "value": dd_peak.isoformat() if dd_peak is not None else ""},
        {"metric": "max_drawdown_trough_date", "value": dd_trough.isoformat() if dd_trough is not None else ""},
    ]

    daily_out = daily.reset_index().rename(columns={"index": "date"})
    return BacktestResult(daily_pnl=daily_out, trades=trades, metrics=pd.DataFrame(metrics_rows))
