"""Feature engineering for equity signal modeling."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing method."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV)."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def compute_features(
    df: pd.DataFrame,
    feature_windows: list[int],
    rsi_windows: list[int],
    bollinger_window: int,
    bollinger_std_multiplier: float,
    volume_window: int,
    lag_returns: list[int],
) -> pd.DataFrame:
    """Generate technical features for a single ticker without lookahead leakage."""
    features = pd.DataFrame(index=df.index)
    close = df["Close"]
    volume = df["Volume"]

    # Returns and momentum
    log_ret = np.log(close / close.shift(1))
    features["log_return_1d"] = log_ret

    for window in feature_windows:
        rolling_ret = np.log(close / close.shift(window))
        features[f"rolling_log_return_{window}d"] = rolling_ret
        features[f"momentum_sign_{window}d"] = np.sign(rolling_ret)
        features[f"volatility_{window}d"] = log_ret.rolling(window).std()

    short_vol = features.get("volatility_5d")
    long_vol = features.get("volatility_20d")
    if short_vol is not None and long_vol is not None:
        features["vol_ratio_5d_20d"] = short_vol / long_vol.replace(0.0, np.nan)

    # Moving averages and trend
    for window in feature_windows:
        sma = close.rolling(window).mean()
        features[f"sma_{window}d"] = sma
        features[f"price_rel_sma_{window}d"] = (close - sma) / sma.replace(0.0, np.nan)

    if "sma_5d" in features.columns and "sma_20d" in features.columns:
        features["sma_ratio_5d_20d"] = features["sma_5d"] / features["sma_20d"].replace(0.0, np.nan)
    if "sma_10d" in features.columns and "sma_50d" in features.columns:
        features["sma_ratio_10d_50d"] = features["sma_10d"] / features["sma_50d"].replace(0.0, np.nan)

    # RSI (Wilder)
    for window in rsi_windows:
        features[f"rsi_{window}d"] = _wilder_rsi(close, window)

    # Bollinger bands
    bb_mid = close.rolling(bollinger_window).mean()
    bb_std = close.rolling(bollinger_window).std()
    bb_upper = bb_mid + bollinger_std_multiplier * bb_std
    bb_lower = bb_mid - bollinger_std_multiplier * bb_std
    band_width = (bb_upper - bb_lower)
    features["bollinger_pct_b"] = (close - bb_lower) / band_width.replace(0.0, np.nan)
    features["bollinger_bandwidth"] = band_width / bb_mid.replace(0.0, np.nan)

    # Volume features
    vol_mean = volume.rolling(volume_window).mean()
    features[f"volume_rel_{volume_window}d"] = volume / vol_mean.replace(0.0, np.nan)
    features["log_volume_change_1d"] = np.log(volume / volume.shift(1))

    obv = _compute_obv(close, volume)
    obv_mean = obv.rolling(volume_window).mean()
    features[f"obv_norm_{volume_window}d"] = obv / obv_mean.replace(0.0, np.nan)

    # Return lags
    for lag in lag_returns:
        features[f"log_return_lag_{lag}d"] = log_ret.shift(lag)

    # Shift all features to ensure model only uses data available before prediction time.
    features = features.shift(1)  # prevents lookahead by removing same-day information leakage

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def validate_no_lookahead(
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    label_horizon: int,
    threshold: float,
) -> None:
    """Warn if any feature is suspiciously correlated with same-day forward returns."""
    forward_return = np.log(prices_df["Close"].shift(-label_horizon) / prices_df["Close"])
    aligned = features_df.join(forward_return.rename("forward_return"), how="inner").dropna()
    if aligned.empty:
        return

    correlations = aligned.drop(columns=["forward_return"]).corrwith(aligned["forward_return"])
    suspect = correlations[correlations.abs() > threshold].sort_values(key=np.abs, ascending=False)
    if not suspect.empty:
        warnings.warn(
            "Potential lookahead leakage: high absolute correlation with same-day forward return found "
            f"for features: {suspect.to_dict()}",
            RuntimeWarning,
            stacklevel=2,
        )
