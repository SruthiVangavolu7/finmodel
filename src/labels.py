"""Label construction utilities for next-horizon return prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(df: pd.DataFrame, label_horizon: int, label_threshold: float) -> pd.DataFrame:
    """Construct forward log returns and binary classification labels."""
    labels = pd.DataFrame(index=df.index)
    forward_return = np.log(df["Close"].shift(-label_horizon) / df["Close"])
    labels["forward_return"] = forward_return
    labels["label"] = (forward_return > label_threshold).astype(int)
    labels = labels.iloc[:-label_horizon] if label_horizon > 0 else labels
    return labels.dropna()
