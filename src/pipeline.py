"""Walk-forward splitting utilities for rigorous time-series evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldRange:
    """Metadata for one walk-forward train/test split."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSplitter:
    """Generate non-overlapping walk-forward train/test folds for time-series."""

    def __init__(self, train_window: int, test_window: int, n_splits: int):
        """Initialize splitter with rolling window lengths and number of folds."""
        self.train_window = train_window
        self.test_window = test_window
        self.n_splits = n_splits

    def split(self, dates: pd.DatetimeIndex):
        """Yield train/test positional indices on a sorted unique DatetimeIndex."""
        unique_dates = pd.DatetimeIndex(sorted(pd.Index(dates).unique()))
        total = len(unique_dates)
        required = self.train_window + self.test_window + (self.n_splits - 1) * self.test_window
        if total < required:
            raise ValueError(
                f"Insufficient samples for {self.n_splits} splits. Have {total} dates, need at least {required}."
            )

        for fold in range(self.n_splits):
            train_start = fold * self.test_window
            train_end = train_start + self.train_window
            test_end = train_end + self.test_window

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(train_end, test_end)

            yield train_idx, test_idx

    def fold_ranges(self, dates: pd.DatetimeIndex) -> list[FoldRange]:
        """Return date ranges for each fold for logging and visualization."""
        unique_dates = pd.DatetimeIndex(sorted(pd.Index(dates).unique()))
        ranges: list[FoldRange] = []
        for fold_num, (train_idx, test_idx) in enumerate(self.split(unique_dates), start=1):
            fold_range = FoldRange(
                fold=fold_num,
                train_start=unique_dates[train_idx[0]],
                train_end=unique_dates[train_idx[-1]],
                test_start=unique_dates[test_idx[0]],
                test_end=unique_dates[test_idx[-1]],
            )
            LOGGER.info(
                "Fold %d/%d: Train %s -> %s | Test %s -> %s",
                fold_num,
                self.n_splits,
                fold_range.train_start.date(),
                fold_range.train_end.date(),
                fold_range.test_start.date(),
                fold_range.test_end.date(),
            )
            ranges.append(fold_range)
        return ranges


def combine_oof_predictions(folds: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine out-of-fold prediction frames into one time-ordered DataFrame."""
    if not folds:
        return pd.DataFrame()
    combined = pd.concat(folds, axis=0, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    return combined
