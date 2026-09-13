"""Feature engineering: turn raw OHLCV into model inputs.

Raw prices are non-stationary (level drifts over time), so models tend to
choke on them. We transform to features that capture structure:

- Returns         : stationarity, % changes (log returns preferred)
- Momentum        : direction over lags/windows (donchian-style)
- Volatility      : risk regime — rolling std/e.g. realized vol
- Rolling stats   : means/mins/maxs/z-scores over windows
- Calendar        : day-of-week / hour-of-day seasonality

Every feature is computed with *trailing* information only (no lookahead),
so it is safe to feed a backtest or live model. All shift-1 style guards
are handled here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default feature windows (in bars)
MOMENTUM_WINDOWS = [3, 5, 10, 20, 50]
VOL_WINDOWS = [5, 10, 20]
ZSCORE_WINDOWS = [10, 20, 50]


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns (stationary, symmetric). First value is NaN."""
    return np.log(close).diff()


def simple_returns(close: pd.Series) -> pd.Series:
    """Arithmetic returns."""
    return close.pct_change()


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Price momentum over a window: close/close.shift(window) - 1."""
    return close / close.shift(window) - 1


def rolling_vol(log_ret: pd.Series, window: int, annualize: int | None = None) -> pd.Series:
    """Realized volatility: rolling std of log returns (annualizable)."""
    vol = log_ret.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(annualize)
    return vol


def rolling_zscore(close: pd.Series, window: int) -> pd.Series:
    """How far current price is from its rolling mean (in rolling std units)."""
    mu = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - mu) / sd


def rolling_min(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).min()


def rolling_max(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).max()


def add_features(
    df: pd.DataFrame,
    momentum_windows: list[int] = MOMENTUM_WINDOWS,
    vol_windows: list[int] = VOL_WINDOWS,
    zscore_windows: list[int] = ZSCORE_WINDOWS,
) -> pd.DataFrame:
    """Enrich an OHLCV frame with a standard feature set.

    Returns a copy with a ``log_return`` column plus prefixed feature
    columns. All features are trailing-only (no lookahead), so the result
    is safe for model training and backtesting.
    """
    out = df.copy()
    close = out["close"]

    out["log_return"] = log_returns(close)
    out["simple_return"] = simple_returns(close)

    for w in momentum_windows:
        out[f"mom_{w}"] = momentum(close, w)

    for w in vol_windows:
        out[f"vol_{w}"] = rolling_vol(out["log_return"], w)

    for w in zscore_windows:
        out[f"zscore_{w}"] = rolling_zscore(close, w)
        out[f"min_{w}"] = rolling_min(close, w)
        out[f"max_{w}"] = rolling_max(close, w)

    # Calendar features (crypto trades seasonally)
    if isinstance(out.index, pd.DatetimeIndex):
        out["day_of_week"] = out.index.dayofweek
        if out.index.freq is not None and pd.Timedelta(out.index.freq) < pd.Timedelta("1D"):
            out["hour_of_day"] = out.index.hour

    # Volume features: relative volume vs rolling average
    if "volume" in out.columns:
        out["vol_ratio"] = out["volume"] / out["volume"].rolling(20).mean()

    return out


# Columns that hold raw/non-stationary levels we don't want as model features
RAW_LEVEL_COLS = ["open", "high", "low", "close", "volume"]
