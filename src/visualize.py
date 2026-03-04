"""Visualization utilities for signal model diagnostics and performance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.pipeline import FoldRange


def _save_fig(path: Path, dpi: int) -> None:
    """Apply tight layout, save figure, and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_equity_curve(
    daily_pnl: pd.DataFrame,
    results_dir: Path,
    dpi: int,
    strategy_color: str,
    benchmark_color: str,
    figsize: tuple[float, float],
) -> None:
    """Plot strategy and benchmark equity curves with max drawdown shading."""
    df = daily_pnl.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    equity = df["strategy_equity"]
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax() if not drawdown.empty else None

    plt.figure(figsize=figsize)
    plt.plot(df.index, df["strategy_equity"], label="Strategy", color=strategy_color, linewidth=2)
    plt.plot(df.index, df["benchmark_equity"], label="SPY Benchmark", color=benchmark_color, linewidth=2)

    if peak is not None and trough is not None:
        plt.axvspan(peak, trough, color="#FCA5A5", alpha=0.25, label="Max Drawdown Period")

    plt.title("Equity Curve: Strategy vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    _save_fig(results_dir / "equity_curve.png", dpi)


def plot_drawdown(
    daily_pnl: pd.DataFrame,
    results_dir: Path,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    """Plot rolling drawdown as a filled area chart."""
    df = daily_pnl.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    equity = df["strategy_equity"]
    drawdown = equity / equity.cummax() - 1.0

    plt.figure(figsize=figsize)
    plt.fill_between(drawdown.index, drawdown.values, 0, color="#93C5FD", alpha=0.8)
    plt.axhline(0, color="#6B7280", linewidth=1)
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    _save_fig(results_dir / "drawdown.png", dpi)


def plot_feature_importance(
    feature_importance: pd.Series,
    results_dir: Path,
    dpi: int,
    accent_color: str,
    figsize: tuple[float, float],
) -> None:
    """Plot top 20 average feature importances across folds."""
    if feature_importance.empty:
        return

    top = feature_importance.head(20).sort_values(ascending=True)
    plt.figure(figsize=figsize)
    plt.barh(top.index, top.values, color=accent_color)
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    _save_fig(results_dir / "feature_importance.png", dpi)


def plot_monthly_ic(
    monthly_ic: pd.Series,
    results_dir: Path,
    dpi: int,
    positive_color: str,
    negative_color: str,
    figsize: tuple[float, float],
) -> None:
    """Plot monthly IC bars colored by sign."""
    if monthly_ic.empty:
        return

    vals = monthly_ic.dropna()
    colors = [positive_color if v >= 0 else negative_color for v in vals.values]

    plt.figure(figsize=figsize)
    plt.bar(vals.index, vals.values, color=colors, width=20)
    plt.axhline(0, color="#111827", linewidth=1)
    plt.title("Monthly Information Coefficient (IC)")
    plt.xlabel("Month")
    plt.ylabel("Spearman IC")
    _save_fig(results_dir / "monthly_ic.png", dpi)


def plot_confusion_matrix(
    predictions: pd.DataFrame,
    results_dir: Path,
    dpi: int,
    figsize: tuple[float, float],
    cmap: str,
) -> None:
    """Plot aggregate confusion matrix for predicted vs actual labels."""
    cm = confusion_matrix(predictions["label"], predictions["pred_label"])
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    _save_fig(results_dir / "confusion_matrix.png", dpi)


def plot_signal_distribution(
    predictions: pd.DataFrame,
    results_dir: Path,
    dpi: int,
    figsize: tuple[float, float],
    bins: int,
    negative_color: str,
    positive_color: str,
) -> None:
    """Plot histogram of predicted probabilities segmented by realized label."""
    plt.figure(figsize=figsize)
    pos = predictions.loc[predictions["label"] == 1, "pred_prob"]
    neg = predictions.loc[predictions["label"] == 0, "pred_prob"]
    plt.hist(neg, bins=bins, alpha=0.6, label="Actual Down (0)", color=negative_color, density=True)
    plt.hist(pos, bins=bins, alpha=0.6, label="Actual Up (1)", color=positive_color, density=True)
    plt.title("Predicted Probability Distribution by Actual Outcome")
    plt.xlabel("Predicted Probability of Up Move")
    plt.ylabel("Density")
    plt.legend()
    _save_fig(results_dir / "signal_distribution.png", dpi)


def plot_walk_forward_splits(
    fold_ranges: list[FoldRange],
    results_dir: Path,
    dpi: int,
    base_figsize: tuple[float, float],
    fold_height: float,
    num_ticks: int,
) -> None:
    """Plot timeline diagram of walk-forward train/test fold windows."""
    if not fold_ranges:
        return

    plt.figure(figsize=(base_figsize[0], base_figsize[1] + fold_height * len(fold_ranges)))
    ax = plt.gca()

    for i, fr in enumerate(fold_ranges, start=1):
        y = len(fold_ranges) - i
        train_start = fr.train_start.toordinal()
        train_len = fr.train_end.toordinal() - fr.train_start.toordinal()
        test_start = fr.test_start.toordinal()
        test_len = fr.test_end.toordinal() - fr.test_start.toordinal()

        ax.broken_barh([(train_start, max(train_len, 1))], (y - 0.35, 0.3), facecolors="#60A5FA")
        ax.broken_barh([(test_start, max(test_len, 1))], (y + 0.05, 0.3), facecolors="#F59E0B")

    y_ticks = [len(fold_ranges) - i for i in range(1, len(fold_ranges) + 1)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"Fold {fr.fold}" for fr in fold_ranges])

    min_date = min(fr.train_start for fr in fold_ranges)
    max_date = max(fr.test_end for fr in fold_ranges)
    ax.set_xlim(min_date.toordinal(), max_date.toordinal())

    tick_dates = pd.date_range(min_date, max_date, periods=num_ticks)
    ax.set_xticks([d.toordinal() for d in tick_dates])
    ax.set_xticklabels([d.strftime("%Y-%m") for d in tick_dates], rotation=30, ha="right")

    ax.set_title("Walk-Forward Train/Test Splits")
    ax.set_xlabel("Date")
    ax.grid(False)
    _save_fig(results_dir / "walk_forward_splits.png", dpi)


def generate_all_plots(
    predictions: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    feature_importance: pd.Series,
    monthly_ic: pd.Series,
    fold_ranges: list[FoldRange],
    results_dir: Path,
    dpi: int,
    strategy_color: str,
    benchmark_color: str,
    accent_color: str,
    positive_color: str,
    negative_color: str,
    figsize_equity: tuple[float, float],
    figsize_drawdown: tuple[float, float],
    figsize_importance: tuple[float, float],
    figsize_monthly_ic: tuple[float, float],
    figsize_confusion: tuple[float, float],
    figsize_signal_dist: tuple[float, float],
    figsize_walk_forward_base: tuple[float, float],
    walk_forward_height_per_fold: float,
    walk_forward_ticks: int,
    confusion_cmap: str,
    signal_hist_bins: int,
    signal_hist_negative_color: str,
    signal_hist_positive_color: str,
) -> None:
    """Generate and save all required project visualizations."""
    plot_equity_curve(daily_pnl, results_dir, dpi, strategy_color, benchmark_color, figsize_equity)
    plot_drawdown(daily_pnl, results_dir, dpi, figsize_drawdown)
    plot_feature_importance(feature_importance, results_dir, dpi, accent_color, figsize_importance)
    plot_monthly_ic(monthly_ic, results_dir, dpi, positive_color, negative_color, figsize_monthly_ic)
    plot_confusion_matrix(predictions, results_dir, dpi, figsize_confusion, confusion_cmap)
    plot_signal_distribution(
        predictions,
        results_dir,
        dpi,
        figsize_signal_dist,
        signal_hist_bins,
        signal_hist_negative_color,
        signal_hist_positive_color,
    )
    plot_walk_forward_splits(
        fold_ranges,
        results_dir,
        dpi,
        figsize_walk_forward_base,
        walk_forward_height_per_fold,
        walk_forward_ticks,
    )
