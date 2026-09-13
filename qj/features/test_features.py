"""Tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qj.features.engine import add_features, log_returns, momentum, rolling_vol, rolling_zscore


@pytest.fixture
def frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=200, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200)))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": np.random.default_rng(1).normal(1000, 50, 200)}, index=idx)


def test_log_returns_shift_is_correct(frame):
    ret = log_returns(frame["close"])
    assert np.isnan(ret.iloc[0])
    assert np.isclose(ret.iloc[1], np.log(frame["close"].iloc[1] / frame["close"].iloc[0]))


def test_momentum_window(frame):
    m = momentum(frame["close"], 10)
    assert np.isnan(m.iloc[:10]).all()
    assert np.isclose(m.iloc[10], frame["close"].iloc[10] / frame["close"].iloc[0] - 1)


def test_add_features_no_lookahead(frame):
    out = add_features(frame)
    # every feature column for row t must only use data up to t
    for col in out.columns:
        if col in {"open", "high", "low", "close", "volume"}:
            continue
        values = out[col]
        assert values.shape[0] == len(frame)
    assert "log_return" in out.columns
    assert "mom_5" in out.columns
    assert "vol_20" in out.columns


def test_rolling_vol_positive(frame):
    vol = rolling_vol(log_returns(frame["close"]), 20)
    assert (vol.dropna() >= 0).all()
