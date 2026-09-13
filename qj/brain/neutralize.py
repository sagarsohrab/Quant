"""Neutralization helpers for alpha scores (BRAIN-style residualization).

In WorldQuant BRAIN, *neutralization* is a mandatory pre-submission step that
removes exposure to common factors *across the cross-section* of assets you
submit on. The three standard factors are:

- ``AMM`` (American Market)   — broad US-equity beta/theme
- ``SUBIND`` (Sub-Industry)   — industry-membership drift
- ``SECTOR``                  — sector-level drift

Conceptually each alpha score at a given bar is regressed onto these factor
exposures and you keep only the *residual*. This leaves the alpha's idiosyncratic
signal intact while stripping out the factor tilt, which (a) makes the alpha
"additive" in a portfolio and (b) keeps you from accidentally double-paying for
a factor you already run elsewhere.

Because this module operates on a *single-instrument* time series (the repo's
universe is one symbol), we model those cross-sectional operations with their
single-series analogues:

- ``cross_sectional_rank``  -> fractional rank over the whole series (time).
- ``neutralize_market``     -> residualize an alpha against a "market" alpha
  (e.g. a buy-and-hold proxied by a rank of close) using OLS, isolated to the
  current bar's window.
- ``demean``                -> subtract the demeaned mean of the cross-section;
  for a single series this is simply subtract the mean over time.

These are deliberately dependency-light (numpy + pandas), and are faithful
reductions of the BRAIN concept rather than a full multi-factor engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """Cross-sectional fractional rank, normalised to [0, 1].

    In BRAIN this ranks an asset within its universe at a single bar. With a
    single series we rank observations over the whole series. Non-null values
    map to evenly spaced fractions; NaN is preserved (rolled forward by the
    caller if desired).
    """
    valid = series.dropna()
    if valid.empty:
        return series * np.nan
    rank = valid.rank(method="average", pct=True)
    return rank.reindex(series.index)


def neutralize_market(
    alpha: pd.Series,
    market_alpha: pd.Series,
) -> pd.Series:
    """Residualize ``alpha`` against a ``market_alpha`` factor via OLS.

    Returns ``alpha - (b0 + b1 * market_alpha)`` estimated on the non-null
    intersection of the two series. This is the single-factor analogue of
    stripping AMM exposure: the returned series is orthogonal (in the least
    squares sense) to the market factor, so it has near-zero beta to it.

    If fewer than 2 overlapping non-null points exist, the market could not be
    fit and the *unmodified* alpha is returned (with the caveat documented).
    """
    a = alpha.dropna()
    m = market_alpha.dropna()
    idx = a.index.intersection(m.index)
    if len(idx) < 2:
        return alpha.copy()

    x = np.asarray(m.loc[idx], dtype=float)
    y = np.asarray(a.loc[idx], dtype=float)

    # OLS: y = b0 + b1*x   (2 x n least squares)
    X = np.column_stack([np.ones_like(x), x])
    try:
        coef, *_rest = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return alpha.copy()
    b0, b1 = coef

    resid = alpha - (b0 + b1 * market_alpha)
    return resid


def demean(series: pd.Series) -> pd.Series:
    """Subtract the cross-sectional mean.

    For a single series this demeans over the whole (non-null) series — the
    direct analogue of BRAIN's ``demean()`` operator applied across the
    universe at one bar. Result has zero mean over non-null entries.
    """
    return series - series.mean()


def neutralize_series(raw_score: pd.Series) -> pd.Series:
    """Convenience: fully neutralize a raw alpha with demeaning + z-scoring.

    Applies the two-step "clean and neutralized" transform typical of BRAIN
    submission: demean (remove any constant tilt) then standardize to unit
    scale so scores are comparable across alphas. Returns NaN-preserving.
    """
    d = demean(raw_score)
    sd = d.std()
    if not np.isfinite(sd) or sd == 0:
        return d.copy()
    return d / sd
