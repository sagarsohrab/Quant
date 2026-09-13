"""Command-line interface for the quant journey.

Examples:
    uv run qj fetch BTCUSDT 1h --start 2025-01-01 --end 2025-01-31
    uv run qj fetch BTCUSDT 1d --last 365
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

from qj.data import BinanceProvider, ParquetStore


def _fetch(args: argparse.Namespace) -> int:
    provider = BinanceProvider(freq=args.freq, symbol=args.symbol)
    store = ParquetStore("data")

    if args.last:
        end = datetime.now(timezone.utc).isoformat()
        freq_map = {"1m": 1, "1h": 60, "1d": 1440, "4h": 240}
        minute_span = freq_map.get(args.freq, 60) * int(args.last)
        start = (datetime.now(timezone.utc) - timedelta(minutes=minute_span)).isoformat()
    else:
        start, end = args.start, args.end

    mdf = provider.fetch_history(start, end)
    path = store.save(mdf)
    _print_summary(mdf, path)
    return 0


def _print_summary(mdf, path) -> None:
    df = mdf.df
    print(f"source={mdf.source} symbol={mdf.symbol} freq={mdf.freq}")
    print(f"saved -> {path}")
    print(f"rows: {len(df)}  range: {df.index[0]:%Y-%m-%d %H:%M} .. {df.index[-1]:%Y-%m-%d %H:%M}")
    print(f"last close: {df['close'].iloc[-1]:,.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qj", description="Quant Journey CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="Download OHLCV history to Parquet cache")
    fetch.add_argument("symbol", nargs="?", default="BTCUSDT")
    fetch.add_argument("freq", nargs="?", default="1h")
    grp = fetch.add_mutually_exclusive_group()
    grp.add_argument("--start", help="ISO start timestamp")
    grp.add_argument("--last", type=int, help="fetch last N candles")
    fetch.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    fetch.set_defaults(func=_fetch)

    live = sub.add_parser("live", help="Run the live WebSocket forecasting loop")
    live.add_argument("symbol", nargs="?", default="BTCUSDT")
    live.add_argument("freq", nargs="?", default="1h")
    live.set_defaults(func=_live)

    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) == "fetch" and not getattr(args, "start", None) and not getattr(args, "last", None):
        args.start = "2024-01-01"  # sane default
    return args.func(args)


def _live(args: argparse.Namespace) -> int:
    """Train a quick model on cached history, then stream live forecasts."""
    from qj.data import ParquetStore
    from qj.features import add_features
    from qj.live import LiveForecaster, stream_klines
    from qj.models import default_feature_cols, train_on_frame
    import asyncio

    store = ParquetStore("data")
    if not store.has("binance", args.symbol, args.freq):
        print(f"No cached {args.symbol} {args.freq} — fetching 500 bars first…")
        provider = BinanceProvider(freq=args.freq, symbol=args.symbol)
        mdf = provider.fetch_history("2025-01-01", datetime.now(timezone.utc).isoformat(), limit=500)
        store.save(mdf)
    mdf = store.load("binance", args.symbol, args.freq)

    print("Training model on cached history…")
    model = train_on_frame(mdf.df, horizon=1, random_state=42)
    fc = LiveForecaster(model)

    async def _run() -> None:
        print(f"Streaming live {args.symbol} {args.freq} (Ctrl+C to stop)…")
        async for row in stream_klines(args.symbol, args.freq):
            fc.add_candle(row)
            pred = fc.predict()
            if pred:
                print(f"{pred['timestamp']:%H:%M:%S} close={pred['close']:.2f} "
                      f"prob_up={pred['prob_up']:.3f} -> {pred['direction']}")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
