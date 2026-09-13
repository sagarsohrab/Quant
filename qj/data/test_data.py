"""Tests for the market data layer."""
from __future__ import annotations

import pandas as pd
import pytest

from qj.data.base import OHLCV_COLS, BaseDataProvider
from qj.data.binance import BinanceProvider
from qj.data.store import ParquetStore


def _dummy_frame(n=100, start="2025-01-01", freq="1h"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = pd.Series(range(n), index=idx, dtype=float)
    df = pd.DataFrame(
        {
            "open": rng,
            "high": rng + 1,
            "low": rng - 1,
            "close": rng,
            "volume": rng * 10,
            "timestamp": idx,  # note: provider would pass numeric ms
        }
    )
    return df.set_index("timestamp")


def test_normalize_lowercases_and_sorts():
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
            "OPEN": [1.0, 2, 3, 4, 5],
            "HIGH": [2.0, 3, 4, 5, 6],
            "LOW": [0.5, 1, 2, 3, 4],
            "CLOSE": [1.1, 2.1, 3.1, 4.1, 5.1],
            "VOLUME": [100.0] * 5,
        }
    )
    out = BaseDataProvider._normalize(df, "1d")
    assert set(OHLCV_COLS) <= set(out.columns)
    assert out.index.name == "timestamp"
    assert out.index.tz is not None


def test_binance_rejects_bad_freq():
    with pytest.raises(ValueError):
        BinanceProvider(freq="9x", symbol="BTCUSDT")


def test_store_roundtrip(tmp_path):
    store = ParquetStore(tmp_path)
    from qj.data.base import MarketDataFrame

    frame = MarketDataFrame(
        df=_dummy_frame(), symbol="BTCUSDT", source="binance", freq="1h"
    )
    path = store.save(frame)
    assert path.exists()
    loaded = store.load("binance", "BTCUSDT", "1h")
    pd.testing.assert_frame_equal(loaded.df, frame.df)
