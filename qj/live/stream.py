"""Binance WebSocket streaming client.

Subscribe to live candle/trade streams over WebSocket and yield normalized
events. Unlike the REST provider, this pushes data to us as it happens —
the foundation for a live quant system.

Docs: https://binance-docs.github.io/apidocs/spot/en/#web-socket-streams
Streams: <symbol>@kline_<interval>  and  <symbol>@trade
"""

from __future__ import annotations

import json

import pandas as pd
import websockets

KLINE_STREAM = "{}@kline_{}"

# JSON field map for kline payloads
KLINE_KEYS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_base",
    "taker_quote", "ignore",
]


def build_stream_url(symbol: str, stream: str) -> str:
    return f"wss://stream.binance.com:9443/ws/{stream}"


async def stream_klines(symbol: str, freq: str = "1m", callback=None):
    """Yield (dataframe-row, raw-payload) for each finished kline.

    If ``callback`` is provided it is called with the parsed row instead of
    yielding. The row is a normalized candle (Open/High/Low/Close/Volume
    with a UTC timestamp index) matching the OHLCV schema.
    """
    stream = KLINE_STREAM.format(symbol.lower(), freq)
    url = build_stream_url(symbol, stream)
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            payload = json.loads(msg)
            if payload.get("e") != "kline":
                continue
            k = payload["k"]
            row = {
                "timestamp": pd.to_datetime(k["t"], unit="ms", utc=True),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
            }
            if callback:
                callback(row)
            else:
                yield row


async def torch_trade_print(symbol: str, freq: str = "1m", max_events: int = 5) -> None:
    """Short demo: connect, print a few live candles, then stop."""
    async for row in stream_klines(symbol, freq):
        print(f"{row['timestamp']:%H:%M:%S}  O={row['open']:.2f} H={row['high']:.2f} "
              f"L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:.1f}")
        max_events -= 1
        if max_events <= 0:
            break
