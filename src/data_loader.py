"""Data loading utilities with on-disk caching for OHLCV downloads."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import yfinance as yf


LOGGER = logging.getLogger(__name__)


def _is_cache_stale(file_path: Path, max_age_days: int) -> bool:
    """Return True if cached file is older than the configured max age."""
    if not file_path.exists():
        return True
    modified_at = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    return modified_at < datetime.now(timezone.utc) - timedelta(days=max_age_days)


def _download_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    price_columns: list[str],
) -> pd.DataFrame:
    """Download adjusted OHLCV data from yfinance for one ticker."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance can return a MultiIndex for columns in some versions.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    missing_cols = [col for col in price_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Ticker {ticker} missing columns: {missing_cols}")

    data = data[price_columns].copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data.dropna(subset=price_columns)
    return data


def _extract_ticker_frame(
    data: pd.DataFrame,
    ticker: str,
    price_columns: list[str],
) -> pd.DataFrame:
    """Extract one ticker frame from yfinance output (single or multi-index columns)."""
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(1):
            return pd.DataFrame()
        one = data.xs(ticker, axis=1, level=1, drop_level=True).copy()
    else:
        one = data.copy()

    missing_cols = [col for col in price_columns if col not in one.columns]
    if missing_cols:
        return pd.DataFrame()

    one = one[price_columns].copy()
    one.index = pd.to_datetime(one.index)
    one = one.sort_index().dropna(subset=price_columns)
    return one


def _download_batch_with_retry(
    tickers: list[str],
    start_date: str,
    end_date: str,
    price_columns: list[str],
    max_retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Dict[str, pd.DataFrame]:
    """Download multiple tickers with retry/backoff and return per-ticker frames."""
    if not tickers:
        return {}

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                actions=False,
                timeout=timeout,
                group_by="column",
                threads=False,
            )
            output: Dict[str, pd.DataFrame] = {}
            for ticker in tickers:
                one = _extract_ticker_frame(data, ticker, price_columns)
                if not one.empty:
                    output[ticker] = one
            return output
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            sleep_s = backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "Batch download attempt %d/%d failed: %s. Retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    return {}


def _load_from_cache(file_path: Path, price_columns: list[str]) -> pd.DataFrame:
    """Load ticker data from CSV cache into a cleaned DataFrame."""
    data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    data = data.sort_index()
    data = data.dropna(subset=price_columns)
    data = data[price_columns]
    return data


def load_price_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    raw_data_dir: Path,
    cache_max_age_days: int,
    price_columns: list[str],
    download_max_retries: int,
    download_backoff_seconds: float,
    download_request_timeout: int,
    download_pause_seconds: float,
) -> Dict[str, pd.DataFrame]:
    """Load adjusted OHLCV data for each ticker with stale-cache refresh logic."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    output: Dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    tickers_list = list(tickers)

    for ticker in tickers_list:
        cache_file = raw_data_dir / f"{ticker}.csv"
        try:
            if cache_file.exists() and not _is_cache_stale(cache_file, cache_max_age_days):
                LOGGER.info("Loading %s from cache: %s", ticker, cache_file)
                df = _load_from_cache(cache_file, price_columns)
                if df.empty:
                    LOGGER.warning("Ticker %s has empty cleaned cache. Scheduling refetch.", ticker)
                    missing.append(ticker)
                    continue
                output[ticker] = df
            else:
                missing.append(ticker)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to load cache for %s: %s. Scheduling refetch.", ticker, exc)
            missing.append(ticker)

    if missing:
        LOGGER.info("Batch fetching %d uncached/stale tickers from yfinance: %s", len(missing), missing)
        try:
            batch_data = _download_batch_with_retry(
                tickers=missing,
                start_date=start_date,
                end_date=end_date,
                price_columns=price_columns,
                max_retries=download_max_retries,
                backoff_seconds=download_backoff_seconds,
                timeout=download_request_timeout,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Batch fetch failed after retries: %s", exc)
            batch_data = {}

        for ticker in missing:
            cache_file = raw_data_dir / f"{ticker}.csv"
            if ticker in batch_data and not batch_data[ticker].empty:
                df = batch_data[ticker]
                df.to_csv(cache_file, index_label="Date")
                output[ticker] = df
                continue

            # Fallback: single-ticker retries for symbols missing from batch response.
            LOGGER.info("Falling back to single-ticker fetch for %s", ticker)
            loaded = False
            for attempt in range(1, download_max_retries + 1):
                try:
                    df = _download_ticker(ticker, start_date, end_date, price_columns)
                    df.to_csv(cache_file, index_label="Date")
                    output[ticker] = df
                    loaded = True
                    break
                except Exception as exc:  # pylint: disable=broad-except
                    sleep_s = download_backoff_seconds * (2 ** (attempt - 1))
                    LOGGER.warning(
                        "Single fetch failed for %s attempt %d/%d: %s. Retrying in %.1fs",
                        ticker,
                        attempt,
                        download_max_retries,
                        exc,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
            if not loaded:
                LOGGER.warning("Failed to load ticker %s after all retries. Skipping.", ticker)

            time.sleep(download_pause_seconds)

    return output
