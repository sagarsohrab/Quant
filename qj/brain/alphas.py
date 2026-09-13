"""WorldQuant BRAIN-style alpha expressions implemented in pandas.

WorldQuant BRAIN (alpha.factors building environment) lets researchers
compose *expressions* from a fixed operator vocabulary (``rank``, ``delay``,
``adv``, ``zscore``, ``product``, ``ts_*`` rolling stats, ...) and run them
over a huge cross-section of assets to produce per-bar scores. Submitting an
alpha requires it to beat thresholds on Sharpe, turnover, drawdown, etc.

This module re-implements a small library of those expression *ideas* as
plain pandas functions that take a single-asset OHLCV ``pd.DataFrame``
(``open/high/low/close/volume`` + DatetimeIndex) and return a ``pd.Series``
of per-bar "scores".

Two honest caveats up front:

- BRAIN operators operate on a *cross-section* (many assets) using things
  like ``rank(...)`` (cross-sectional percentile) and ``adv20``
  (20-day average *dollar* volume). Here we only have one asset, so
  ``rank`` is applied *over time* as a rolling rank and "adv" is a rolling
  average of raw volume. This is a faithful single-instrument reduction,
  not a drop-in BRAIN alpha.
- Nothing here is guaranteed (or even likely) to pass WorldQuant's bar. The
  purpose is a functional, honest trainer you can iterate on.

Every alpha is defined as ``alpha_N(df) -> pd.Series`` aligned to ``df.index``
and only uses trailing information (rolling windows / shifts >= 1), so
results are free of lookahead bias and safe to feed a backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ignore warnings for NaN slices inside rolling/rank on bootstrap windows.
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="qj.brain.alphas")


def _check(df: pd.DataFrame) -> None:
    """Coerce/validate a single-asset OHLCV frame to the columns we need."""
    if df.index is None or not isinstance(df.index, pd.Index):
        raise ValueError("alpha functions require an index (DatetimeIndex ideal)")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"alpha functions require a '{col}' column")


def _close(df: pd.DataFrame) -> pd.Series:
    return df["close"]


def _volume(df: pd.DataFrame) -> pd.Series:
    return df["volume"]


def rank(series: pd.Series) -> pd.Series:
    """Cross-sectional-style rank as a *rolling* rank over a short window.

    In BRAIN, ``rank()`` is computed across the universe at each bar. With a
    single asset we estimate that per-bar cross-sectional percentile is not
    directly available, so for expression-only alphas we treat the series as
    a rolling rank (percentile rank of the current observation within the
    trailing `window` observations), normalised to 0..1.
    """
    return series.rolling(len(series), min_periods=2).rank(pct=True)


def delay(series: pd.Series, d: int) -> pd.Series:
    """``delay(x, d)`` — value of ``x`` d bars ago (no lookahead)."""
    return series.shift(d)


def adv(df: pd.DataFrame, window: int) -> pd.Series:
    """``adv20``-style average volume proxy = rolling mean of volume."""
    return _volume(df).rolling(window).mean()


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of a series over trailing ``window`` bars."""
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """``ts_rank`` — time-series percentile rank over trailing window."""
    return series.rolling(window).rank(pct=True)


def ts_min(series: pd.Series, window: int) -> pd.Series:
    """Rolling minimum over trailing ``window`` bars."""
    return series.rolling(window).min()


def ts_max(series: pd.Series, window: int) -> pd.Series:
    """Rolling maximum over trailing ``window`` bars."""
    return series.rolling(window).max()


def ts_mean(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over trailing ``window`` bars."""
    return series.rolling(window).mean()


def stddev(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation over trailing ``window`` bars."""
    return series.rolling(window).std()


def alpha_1(df: pd.DataFrame) -> pd.Series:
    """Long momentum: rank of close 1 bar ago relative to 5 bars ago.

    Mirror of ``rank(delay(close,1)/delay(close,5))``. Positive raw score when
    price is trending up over the short run.
    """
    _check(df)
    c = _close(df)
    return rank(delay(c, 1) / delay(c, 5))


def alpha_2(df: pd.DataFrame) -> pd.Series:
    """Absolute momentum: close / close 10 bars ago - 1.

    Mirror of ``rank((close - delay(close,10))/delay(close,10))`` style
    absolute-momentum operator. Higher when price rose over the window.
    """
    _check(df)
    c = _close(df)
    return rank(c / delay(c, 10) - 1)


def alpha_3(df: pd.DataFrame) -> pd.Series:
    """Short-term mean reversion: negative z-score of close.

    Mirror of ``-zscore(close,10)``. Bet that prices pulled far above recent
    mean will revert (short) and vice-versa.
    """
    _check(df)
    return -zscore(_close(df), 10)


def alpha_4(df: pd.DataFrame) -> pd.Series:
    """Direction-of-trend momentum: sign of 20-bar momentum, z-scored.

    Mirror of ``zscore(close,20)`` combined with `sign(close-delay(close,20))`.
    """
    _check(df)
    c = _close(df)
    return zscore(c, 20) * np.sign(c - delay(c, 20))


def alpha_5(df: pd.DataFrame) -> pd.Series:
    """Volatility-scaled raw momentum: mom / vol20.

    Mirror of the classic `close/delay(close,20) - 1` scaled by 1/vol20 so
    calm assets are weighted up (risk-parity flavour). Uses rank on the ratio.
    """
    _check(df)
    c = _close(df)
    mom = c / delay(c, 20) - 1
    v = stddev(np.log(c), 20)
    return rank(mom / v)


def alpha_6(df: pd.DataFrame) -> pd.Series:
    """Displaced mean-reversion: 20-bar z-score of 10-bar delayed close.

    Mirror of ``rank(zscore(delay(close,10),20))`` — z-score the delayed close
    so we mean-revert *where* price was 10 bars ago (displacement).
    """
    _check(df)
    c = _close(df)
    return zscore(delay(c, 10), 20)


def alpha_7(df: pd.DataFrame) -> pd.Series:
    """Displacement + zscore combo: momentum of the 5/10 delay pair.

    Mirror of ``rank(delay(close,5)-delay(close,10))`` combined with z-score.
    Captures slope of the recent price path.
    """
    _check(df)
    c = _close(df)
    slope = delay(c, 5) - delay(c, 10)
    return rank(slope) * zscore(slope, 10)


def alpha_8(df: pd.DataFrame) -> pd.Series:
    """High-low range momentum (volatility proxy), z-scored.

    Mirror of ``zscore((high+low)/2, 10)`` style but uses the range trend:
    (high+low)/2 relative to a 20-bar trailing average.
    """
    _check(df)
    x = (df["high"] + df["low"]) / 2
    return zscore(x - ts_mean(x, 20), 20)


def alpha_9(df: pd.DataFrame) -> pd.Series:
    """Volume-sponsored momentum.

    Mirror of ``rank(volume / adv20 * close / delay(close, 10))``. Momentum
    weighted by relative volume — a classic BRAIN expression that blends
    return and participation.
    """
    _check(df)
    c = _close(df)
    rel_vol = _volume(df) / adv(df, 20)
    return rank(rel_vol * (c / delay(c, 10)))


def alpha_10(df: pd.DataFrame) -> pd.Series:
    """Mean-reversion with volatility-neutraliser: -zscore(vol_ratio*close).

    Uses change in volume (volume / rolling-mean) times displaced close.
    """
    _check(df)
    c = _close(df)
    vratio = _volume(df) / ts_mean(_volume(df), 10)
    return -zscore(c * vratio, 20)


def alpha_11(df: pd.DataFrame) -> pd.Series:
    """Rolling-correlation-style price/volume rank spread.

    Approximates BRAIN correlation operators: ranks z-scored close against
    z-scored volume trend, taking the difference (a spread bet).
    """
    _check(df)
    c = _close(df)
    p = zscore(c, 10)
    v = zscore(_volume(df), 10)
    return rank(p) - rank(v)


def alpha_12(df: pd.DataFrame) -> pd.Series:
    """Breakout / Donchian: position within the trailing 50-bar range.

    Mirror of ``rank(close - ts_min(close,50)) / (ts_max - ts_min)``. Values
    near 1 mean fresh highs (trend), near 0 fresh lows.
    """
    _check(df)
    c = _close(df)
    lo = ts_min(c, 50)
    hi = ts_max(c, 50)
    rng = hi - lo
    return (c - lo) / rng


# Registry: name -> expression function. Exposed so callers can iterate over
# the whole library uniformly. Call each with ``ALPHAS[name](df)``.
ALPHAS: dict[str, object] = {
    "alpha_1": alpha_1,
    "alpha_2": alpha_2,
    "alpha_3": alpha_3,
    "alpha_4": alpha_4,
    "alpha_5": alpha_5,
    "alpha_6": alpha_6,
    "alpha_7": alpha_7,
    "alpha_8": alpha_8,
    "alpha_9": alpha_9,
    "alpha_10": alpha_10,
    "alpha_11": alpha_11,
    "alpha_12": alpha_12,
}


def get_alpha(name: str):
    """Fetch an alpha by name, raising KeyError if absent."""
    if name not in ALPHAS:
        raise KeyError(f"Unknown alpha {name!r}; available: {sorted(ALPHAS)}")
    return ALPHAS[name]
