"""Tests for forecasting models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qj.features.engine import add_features
from qj.models.forecast import make_clf, make_dataset, make_target
from qj.models.walkforward import walk_forward_evaluate


@pytest.fixture
def frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=1200, freq="1h", tz="UTC")
    # a synthetic series with mild momentum structure so the model can learn
    rng = np.random.default_rng(7)
    rets = 0.02 * np.array([(1 if rng.random() < 0.55 else -1) for _ in range(1200)])
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": rng.normal(1000, 100, 1200)},
        index=idx,
    )


def test_make_target_no_lookahead():
    close = pd.Series([1.0, 1.1, 1.2, 1.0, 0.9], dtype=float)
    y = make_target(close, horizon=1)
    # At t=0, future close (1.1) > 1.0 => 1
    # At t=1, future (1.2) > 1.1 => 1 ; t=3 future(0.9) not > 1.0 => 0; etc.
    expected = [1.0, 1.0, 0.0, 0.0, np.nan]
    vals = y.tolist()
    for got, exp in zip(vals, expected):
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert got == exp


def test_make_dataset_drops_trivial_rows(frame):
    feats = add_features(frame)
    X, y = make_dataset(feats, horizon=1)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] >= 5
    assert not X.isna().any().any()


def test_walk_forward_runs(frame):
    feats = add_features(frame)
    cols = [c for c in feats.columns if c in {"log_return", "mom_5", "vol_10", "zscore_10"}]
    res = walk_forward_evaluate(feats, cols, horizon=1, n_splits=3, train_pct=0.5)
    assert "acc" in res
    assert 0 <= res["acc"] <= 1
    assert res["n_splits"] >= 1
