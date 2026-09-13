"""BRAIN-style alpha simulation and evaluation metrics.

The BRAIN "simulation" is a paper-account pass over your historical universe:
at each bar an alpha produces a raw score per asset; those scores are turned
into a long/flat (or long/short) position held into the *next* bar, and the
realised forward returns accrue to an equity curve. The platform then reports
Sharpe, turnover, max drawdown, and out-of-sample stability.

This module re-implements that loop for a *single asset* with a target
long/flat backtest consistent with the repo's ``qj/backtest/engine.py``
naming/conventions:

- ``compute_signal``   : run an alpha function over an OHLCV frame -> scores.
- ``forward_returns``  : realised per-bar P&L from holding a score-signed
  position with no lookahead (score at bar t is traded at bar t+1).
- ``sharpe_style``     : annualised-ish Sharpe from per-bar returns.
- ``simulate_alpha``   : full report: Sharpe, turnover, coverage, n_bars, and
  an in-sample / out-of-sample Sharpe split.

Caveat, stated plainly: real BRAIN walks forward over a huge multi-asset
universe and applies turnover/decay filters; our IS/OOS split is a two-half
time split of a single symbol. It is a *proxy* for out-of-sample behaviour,
not a submission guarantee. Alphas here are likely far from WorldQuant's bar
— treat this as a training harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qj.backtest.engine import performance_metrics
from qj.brain import neutralize as _nz


def compute_signal(alpha_func, df: pd.DataFrame) -> pd.Series:
    """Run ``alpha_func(df)`` to produce a per-bar score Series.

    The result is aligned to ``df.index``. The alpha function may return any
    raw (un-normalized) score; we z-normalize it so simulated position sizing
    behaves sensibly (comparable across alphas).
    """
    raw = alpha_func(df)
    if not isinstance(raw, pd.Series):
        raise TypeError(f"alpha_func must return a pd.Series, got {type(raw).__name__}")
    if len(raw.index) != len(df.index):
        # Reindex to guarantee alignment even if alpha drops/injects points.
        raw = raw.reindex(df.index)
    return _nz.neutralize_series(raw)


def forward_returns(scores: pd.Series, close: pd.Series, horizon: int = 1) -> pd.Series:
    """Realised per-bar P&L from a score without lookahead.

    Score at bar ``t`` sets a position ``sign(score)`` held for ``horizon``
    bars; the payoff is the ``horizon``-bar forward return starting at ``t+1``.
    By using ``scores.shift(1)`` we never peek at the bar we're about to earn.

    Returns a Series aligned to ``close.index`` with the per-bar return.
    """
    s = scores.reindex(close.index).astype(float)
    if horizon == 1:
        fwd = close.pct_change().fillna(0.0)
    else:
        fwd = (close.shift(-horizon) / close - 1).fillna(0.0)

    pos = np.sign(s.shift(1)).fillna(0.0)  # position decided bar t-1, held into t
    pnl = pos * fwd
    return pnl.reindex(close.index)


def sharpe_style(returns: pd.Series, periods_per_year: float = 24) -> float:
    """Annualised-ish Sharpe from per-bar forward returns.

    ``periods_per_year`` defaults to 24 (a rough hourly->yearly multiple for
    crypto bars), so the value is comparable to the repo's
    ``performance_metrics`` annualisation convention. Returns NaN when there is
    no volatility or insufficient data.
    """
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return float("nan")
    sd = r.std()
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def turnover(scores: pd.Series) -> float:
    """Mean absolute change in the normalized score (a proxy for trade churn).

    BRAIN reports turnover as how much the position changes between bars;
    with a score-signed position this maps to the mean absolute score diff.
    Higher turnover = more trades = more drag. Returns 0 for constant/empty.
    """
    s = scores.dropna()
    if len(s) < 2:
        return 0.0
    return float(s.diff().abs().mean())


def _half_sharpe(scores: pd.Series, close: pd.Series, horizon: int) -> tuple[float, int]:
    pnl = forward_returns(scores, close, horizon=horizon)
    return sharpe_style(pnl), int(pnl.dropna().shape[0])


def simulate_alpha(
    alpha_func,
    df: pd.DataFrame,
    horizon: int = 1,
    periods_per_year: float = 24,
) -> dict:
    """Run a full BRAIN-style simulation report for one alpha.

    Metrics returned (all finite numeric where computable):

    - ``sharpe``        : Sharpe on the full sample.
    - ``turnover``      : mean absolute change in normalized score.
    - ``n_bars``        : number of bars with a position (non-null pnl).
    - ``coverage``      : fraction of bars with a non-null score.
    - ``is_sharpe``     : Sharpe on the first half of the data (in-sample).
    - ``oos_sharpe``    : Sharpe on the second half (out-of-sample proxy).
    - ``drawdown``      : max drawdown of the net long/flat curve if finances
      permit computation (reuses ``qj.backtest.engine``).
    - ``annual_return`` : from the long/flat equity curve (reuses the engine).

    ``horizon`` is the forward-return horizon in bars.
    """
    scores = compute_signal(alpha_func, df)
    close = df["close"]
    pnl = forward_returns(scores, close, horizon=horizon)

    sharpe = sharpe_style(pnl, periods_per_year)
    n_bars = int(pnl.dropna().shape[0])
    coverage = float(scores.notna().mean())

    # IS / OOS by a two-half time split (see module docstring caveat).
    half = len(scores) // 2
    first = scores.iloc[:half]
    second = scores.iloc[half:]
    is_sharpe, oos_sharpe = (
        sharpe_style(forward_returns(first, close.iloc[:half], horizon), periods_per_year),
        sharpe_style(forward_returns(second, close.iloc[half:], horizon), periods_per_year),
    )

    # Optional long/flat backtest metrics via the shared engine. Build a
    # position series (sign of score, shifted by engine) and reuse metrics.
    drawdown = float("nan")
    annual_return = float("nan")
    try:
        pos = np.sign(scores.fillna(0.0))
        bt = _nz_engine_backtest(close, pos)
        if bt is not None and len(bt) > 0:
            pm = performance_metrics(bt, periods_per_year=periods_per_year)
            drawdown = pm["max_drawdown"]
            annual_return = pm["annual_return"]
    except Exception:
        # Evaluation metrics should never break a report; degrade gracefully.
        drawdown = float("nan")
        annual_return = float("nan")

    return {
        "sharpe": sharpe,
        "turnover": turnover(scores),
        "n_bars": n_bars,
        "coverage": coverage,
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "drawdown": drawdown,
        "annual_return": annual_return,
    }


def _nz_engine_backtest(close: pd.Series, pos: pd.Series):
    """Build the backtest frame the shared engine expects (net long/flat)."""
    import numpy as np

    ret = close.pct_change().fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)
    gross = pos_lag * ret
    frame = pd.DataFrame(
        {"return": ret, "position": pos_lag, "net": gross},
        index=close.index,
    )
    frame["net_cum"] = (1 + gross).cumprod()
    frame["gross"] = gross
    frame["cost"] = np.zeros(len(frame))
    frame["gross_cum"] = frame["net_cum"]
    return frame
