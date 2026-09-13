"""Binance market data provider.

Uses the public Binance REST API (no API key required for public market
data). Handles pagination (klines are capped at 1000 rows/request) and
interval translation.

Docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
"""

from __future__ import annotations

import time
import zoneinfo
from datetime import datetime, timedelta

import pandas as pd
import requests

from qj.data.base import BaseDataProvider, MarketDataFrame, OHLCV_COLS

# Binance interval string -> pandas frequency alias
INTERVAL_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "D",
    "3d": "3D",
    "1w": "W",
    "1M": "ME",
}

# Max rows per klines call per Binance
KLINES_LIMIT = 1000


class BinanceProvider(BaseDataProvider):
    """Fetch OHLCV history from Binance's public REST API."""

    BASE = "https://api.binance.com"
    KLINES_ENDPOINT = "/api/v3/klines"

    def __init__(
        self, freq: str = "1h", symbol: str = "BTCUSDT", timeout: int = 10
    ) -> None:
        super().__init__(freq=freq, symbol=symbol)
        if freq not in INTERVAL_MAP:
            raise ValueError(
                f"Unsupported freq {freq!r}; choose from {sorted(INTERVAL_MAP)}"
            )
        self.timeout = timeout
        self.session = requests.Session()

    def _klines(
        self,
        start_ms: int,
        end_ms: int | None = None,
        limit: int = KLINES_LIMIT,
    ) -> list[list]:
        params = {
            "symbol": self.symbol,
            "interval": self.freq,
            "startTime": start_ms,
            "limit": limit,
        }
        if end_ms is not None:
            params["endTime"] = end_ms
        resp = self.session.get(self.BASE + self.KLINES_ENDPOINT, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_history(
        self, start: str, end: str, limit: int | None = None
    ) -> MarketDataFrame:
        """Fetch candles between ISO timestamps, auto-paginating."""
        freq_td = pd.Timedelta(self.freq)
        start_dt = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC")
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        rows: list[list] = []
        cursor = start_ms
        # Binance returns 0-based open time; exclusive-ish end handling
        while cursor < end_ms:
            batch = self._klines(cursor, end_ms=end_ms)
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            if len(batch) < KLINES_LIMIT:
                break
            cursor = last_open + int(freq_td.total_seconds() * 1000)
            time.sleep(0.1)  # be gentle with the free API

        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base",
                "taker_quote", "ignore",
            ],
        )
        df = df[OHLCV_COLS + ["timestamp"]]
        normalized = self._normalize(df, self.freq, ts_unit="ms")
        # clip to requested end
        normalized = normalized[normalized.index < end_dt]
        if limit:
            normalized = normalized.iloc[-limit:]
        return MarketDataFrame(
            df=normalized, symbol=self.symbol, source="binance", freq=self.freq
        )

    def fetch_latest(self) -> MarketDataFrame:
        """Fetch the most recent candle."""
        end_ms = int(time.time() * 1000)
        freq_td = pd.Timedelta(self.freq)
        start_ms = end_ms - int(freq_td.total_seconds() * 1000 * 2)
        df = self._klines(start_ms, end_ms=end_ms, limit=1)
        rows = [r for r in df][-1:] if df else []
        out = pd.DataFrame(
            rows,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base",
                "taker_quote", "ignore",
            ],
        )
        out = out[OHLCV_COLS + ["timestamp"]]
        normalized = self._normalize(out, self.freq, ts_unit="ms")
        return MarketDataFrame(
            df=normalized, symbol=self.symbol, source="binance", freq=self.freq
        )
