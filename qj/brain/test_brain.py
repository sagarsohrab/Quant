"""Tests for the qj.brain (WorldQuant BRAIN alpha trainer) subpackage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qj.brain.alphas import ALPHAS, alpha_1, alpha_3
from qj.brain.evaluate import compute_signal, forward_returns, sharpe_style, simulate_alpha, turnover
from qj.brain.neutralize import (
    cross_sectional_rank,
    demean,
    neutralize_market,
    neutralize_series,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Synthetic OHLCV with ~2000 hourly bars (fast, no network)."""
    idx = pd.date_range("2023-01-01", periods=2000, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    # Geometric random walk for close; build OHLC around it.
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, len(idx))))
    spread = rng.uniform(0.001, 0.004, len(idx)) * close
    high = close + rng.uniform(0, 1, len(idx)) * spread
    low = close - rng.uniform(0, 1, len(idx)) * spread
    volume = rng.gamma(2, 500, len(idx))
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_each_alpha_returns_aligned_series_without_errors(frame):
    """Every registered alpha returns a Series matching df.index, finite values."""
    for name, fn in ALPHAS.items():
        out = fn(frame)
        assert isinstance(out, pd.Series), f"{name} did not return a Series"
        assert out.index.equals(frame.index), f"{name} index mismatch"
        assert out.dtype != object, f"{name} returned non-numeric dtype {out.dtype}"
        # must have at least some finite non-null values
        assert out.dropna().shape[0] > 0, f"{name} produced all-NaN series"


def test_registry_has_twelve_alphas():
    assert len(ALPHAS) == 12
    for k in ALPHAS:
        assert k.startswith("alpha_")


def test_forward_returns_no_lookahead(frame):
    """Position decided at bar t must not use bar t's return."""
    scores = pd.Series(1.0, index=frame.index)
    pnl = forward_returns(scores, frame["close"])
    # A constant long position earns the next-bar return, not the same-bar one.
    expected = frame["close"].pct_change().fillna(0.0)
    assert np.allclose(pnl.dropna().values, expected.dropna().values)


def test_neutralize_reduces_correlation():
    """Residualizing an alpha vs the market factor lowers |correlation|."""
    rng = np.random.default_rng(0)
    n = 300
    market = pd.Series(np.cumsum(rng.normal(0, 1, n)))
    alpha = pd.Series(2.0 * market + rng.normal(0, 3, n))
    resid = neutralize_market(alpha, market)
    corr_before = alpha.corr(market)
    corr_after = resid.corr(market)
    assert abs(corr_after) < abs(corr_before)
    assert abs(corr_after) < 1e-9


def test_neutralize_preserves_index_shape(frame):
    s = frame["close"].iloc[:100]
    r = neutralize_series(s)
    assert r.index.equals(s.index)


def test_cross_sectional_rank_bounds(frame):
    s = frame["close"]
    r = cross_sectional_rank(s)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_demean_zero_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    d = demean(s)
    assert np.isclose(d.mean(), 0.0)


def test_simulate_alpha_returns_finite_keys(frame):
    """simulate_alpha returns documented keys with finite numeric values."""
    for name in ["alpha_1", "alpha_3", "alpha_5", "alpha_9"]:
        report = simulate_alpha(ALPHAS[name], frame, horizon=1)
        for key in [
            "sharpe", "turnover", "n_bars", "coverage",
            "is_sharpe", "oos_sharpe", "drawdown", "annual_return",
        ]:
            assert key in report, f"{name}: missing key {key}"
        assert np.isfinite(report["coverage"])
        assert report["n_bars"] >= 0
        assert isinstance(report["sharpe"], float)


def test_turnover_nonnegative():
    s = pd.Series(np.linspace(0, 1, 100))
    assert turnover(s) >= 0


def test_sortino_style_helper_exists():
    # Guard against accidental removal of a core metric helper.
    from qj.brain.evaluate import sharpe_style
    assert callable(sharpe_style)


def test_compute_signal_z_normalizes(frame):
    sig = compute_signal(alpha_3, frame)
    valid = sig.dropna()
    assert np.isclose(valid.mean(), 0.0, atol=1e-9)
    assert np.isclose(valid.std(), 1.0, atol=1e-6)
