"""Market data abstractions (source-agnostic).

The quant pipeline should not care whether prices come from Binance,
Yahoo, or a local file. This module defines the interface and storage
contract, so you can swap data providers without touching downstream
feature/model/backtest code.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd

# Canonical schema every provider returns / we store. Renaming to these
# column names keeps the rest of the stack stable regardless of source.
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class MarketDataFrame:
    """A normalized OHLCV frame with a time zone-aware index."""

    df: pd.DataFrame
    symbol: str
    source: str
    freq: str  # e.g. "1m", "1h", "1d"

    @property
    def ohlcv(self) -> pd.DataFrame:
        return self.df[OHLCV_COLS]


class BaseDataProvider(abc.ABC):
    """Interface every market data provider must implement."""

    def __init__(self, freq: str = "1h", symbol: str = "BTCUSDT") -> None:
        self.freq = freq
        self.symbol = symbol

    @abc.abstractmethod
    def fetch_history(
        self, start: str, end: str, limit: int | None = None
    ) -> MarketDataFrame:
        """Fetch OHLCV history between start and end (ISO timestamps)."""

    @abc.abstractmethod
    def fetch_latest(self) -> MarketDataFrame:
        """Fetch the most recent candles (for live scenarios)."""

    @staticmethod
    def _normalize(
        df: pd.DataFrame, freq: str, ts_unit: str = "s"
    ) -> pd.DataFrame:
        """Ensure a uniform, numeric, time-indexed OHLCV frame.

        Input is expected to have a parsed timestamp and OHLCV columns
        (lowercased). The returned frame is sorted, numeric, and indexed
        by a time zone-aware DatetimeIndex named "timestamp".

        ``ts_unit`` is the unit of the raw numeric timestamp (e.g. "s", "ms",
        "ns"). Providers pass the unit their API returns.
        """
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]  # normalize case
        keep = [c for c in OHLCV_COLS if c in df.columns]
        # A "time" or "timestamp" column must exist and is used as the index.
        if "timestamp" not in df.columns and "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        keep.append("timestamp")
        df = df[keep]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=ts_unit, utc=True)
        for c in OHLCV_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.set_index("timestamp").sort_index()
        # Some providers label volume by base vs quote currency; keep volume.
        return df
