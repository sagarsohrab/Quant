"""Tests for the backtesting engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qj.backtest.engine import backtest_positions, performance_metrics, signal_to_position


def _close(n=100, drift=0.0002):
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(3).normal(drift, 0.02, n)
    close = 100 * np.exp(np.cumsum(rng))
    return pd.Series(close, index=idx)


def test_always_long_positions():
    prob = pd.Series([0.6] * 100, index=_close().index)
    pos = signal_to_position(prob, threshold=0.5)
    assert (pos == 1.0).all()


def test_flat_to_zero_position():
    prob = pd.Series([0.5] * 100, index=_close().index)
    pos = signal_to_position(prob, threshold=0.5)
    assert (pos == 0.0).all()


def test_short_geometry():
    prob = pd.Series([0.3, 0.3, 0.7, 0.5], index=pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC"))
    pos = signal_to_position(prob, threshold=0.55, allow_short=True)
    assert (pos == pd.Series([-1.0, -1.0, 1.0, 0.0], index=prob.index)).all()


def test_no_lookahead_position_shift():
    close = _close(50)
    # A "predict 100% up tomorrow" signal at bar 0 means we earn bar 0's
    # return only from bar 1 onward — never the return that decided it.
    prob = pd.Series([1.0] + [0.5] * 49, index=close.index)
    pos = signal_to_position(prob, threshold=0.5)
    bt = backtest_positions(close, pos, commission=0.0)
    # On the first bar the (lagged) position should be 0 (we only act next bar)
    assert bt["position"].iloc[0] == 0.0


def test_metrics_shape():
    close = _close(200)
    prob = (np.sin(np.arange(200)) + 2) / 4  # 0.25..0.75
    prob = pd.Series(prob, index=close.index)
    pos = signal_to_position(prob, threshold=0.5)
    bt = backtest_positions(close, pos, commission=0.001)
    m = performance_metrics(bt)
    assert "sharpe" in m and "max_drawdown" in m and "total_return" in m
    assert bt["net_cum"].iloc[-1] > 0
