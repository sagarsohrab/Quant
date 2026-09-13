"""Tests for the live forecasting layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qj.features.engine import add_features
from qj.live.forecaster import LiveForecaster
from qj.models.forecast import make_clf, make_target


def _candles(n=200):
    idx = pd.date_range("2025-01-01", freq="1h", periods=n, tz="UTC")
    rng = np.random.default_rng(5).normal(0.001, 0.02, n)
    close = 100 * np.exp(np.cumsum(rng))
    rows = [
        {"timestamp": t, "open": float(c), "high": float(c * 1.01),
         "low": float(c * 0.99), "close": float(c), "volume": 1000.0}
        for t, c in zip(idx, close)
    ]
    return rows


def test_live_forecaster_predicts_after_warmup():
    rows = _candles(300)
    df = pd.DataFrame(rows).set_index("timestamp")
    feats = add_features(df)
    y = make_target(df["close"], 1)
    X = feats[["log_return", "mom_5", "vol_10"]].iloc[:-1]
    yy = y.iloc[:-1]
    clf = make_clf(random_state=1)
    clf.fit(X, yy)

    fc = LiveForecaster(clf, feature_cols=["log_return", "mom_5", "vol_10"], warmup=30)
    for r in rows[:300]:
        fc.add_candle(r)
    pred = fc.predict()
    assert pred is not None
    assert "prob_up" in pred
    assert 0 <= pred["prob_up"] <= 1
    assert pred["direction"] in {"LONG", "FLAT"}


def test_live_forecaster_none_until_warmup():
    rows = _candles(10)
    fc = LiveForecaster(make_clf(random_state=1), warmup=50)
    for r in rows[:10]:
        fc.add_candle(r)
    assert fc.predict() is None
