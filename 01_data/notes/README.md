# Layer 1: Data — foundations

Raw market data is the bedrock. Everything downstream (features, models,
backtests, live trading) consumes it, so we make the data layer clean and
*source-agnostic*.

## Code map
- `qj/data/base.py` — the abstraction: `MarketDataFrame` + `BaseDataProvider`
- `qj/data/binance.py` — a Binance implementation (REST, auto-paginated)
- `qj/data/store.py` — Parquet storage + disk cache

## Why this design?
1. **Source-agnostic interface**: `BaseDataProvider` defines `fetch_history`
   and `fetch_latest`. The rest of the stack only ever sees a normalized
   `MarketDataFrame` (columns: open/high/low/close/volume + UTC time index).
   Swap Binance for Yahoo or a CSV and *nothing* downstream changes.
2. **Parquet for storage**: columnar, compressed, keeps dates/dtypes. Fast
   to read across the stack, standard in quant/data engineering.
3. **Pagination handled**: Binance caps klines at 1000 rows/call; the
   provider loops until it has the full range — so fetching "3 years of 1m",
   which would be many pages, just works.

## Key concepts you'll build on
- **OHLCV**: Open, High, Low, Close, Volume — the canonical candle shape.
- **Time index**: all analysis keys on a UTC `DatetimeIndex`; timezone-naive
  data is a common source of silent bugs.
- **Frequency**: 1m/1h/1d denote bar length. Higher freq = more data +
  noise; this choice drives everything downstream.

## Try
```
uv run qj fetch BTCUSDT 1h --last 500
```
Then load the cached Parquet with `ParquetStore('data').load(...)`.
