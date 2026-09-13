# Quant Journey

From raw market data → forecasting models → live systems → ML infrastructure → LLM architecture internals. One walking path, each layer builds on the last.

## The Stack (bottom-up)

| Layer | Folder | What you'll master |
|-------|--------|--------------------|
| **1. Data** | `01_data/` | Sourcing OHLCV (Binance first), source-agnostic design, Parquet storage |
| **2. Features** | `02_features/` | Returns, momentum, volatility, rolling stats — building model inputs |
| **3. Models** | `03_models/` | Baselines → gradient boosting (LightGBM/XGBoost) forecasting |
| **4. Backtest** | `04_backtest/` | Trade simulation, equity curve, Sharpe / max-drawdown / win-rate |
| **5. Live** | `05_live/` | WebSocket streaming, live forecast loop, event handling |
| **6. ML ops** | `06_mlops/` | Training pipelines, model registry, drift monitoring |
| **7. LLM arch** | `07_llm_arch/` | Tokenizers, transformers from scratch, inference optimization |

Shared code lives in `qj/`. Data/model artifacts go in `data/` and `models/`.

## Quickstart

```bash
uv sync                      # install deps into venv
uv run qj fetch BTCUSDT 1h   # download & cache history
uv run qj live BTCUSDT 1h    # run the live forecast loop
uv run --extra llm python -m qj.llm.train_demo   # watch an LLM learn from scratch
uv run --extra ml python -m pytest qj/ -q        # run the full test suite
```

## Progress Tracker

- [x] Layer 1 — Data fetching + storage (notes: `01_data/notes/`)
- [x] Layer 2 — Feature engineering (notes: `02_features/notes/`)
- [x] Layer 3 — Forecasting models (notes: `03_models/notes/`)
- [x] Layer 4 — Backtesting engine (notes: `04_backtest/notes/`)
- [x] Layer 5 — Live streaming + forecasting (notes: `05_live/notes/`)
- [x] Layer 6 — ML ops: registry + drift (notes: `06_mlops/notes/`)
- [x] Layer 7 — LLM architecture: MiniGPT from scratch (notes: `07_llm_arch/notes/`)

Every layer has a `notes/README.md` explaining the concepts and pointing to
the exact code that implements them. When you want the full explain-mode
walkthrough (intuition + formulas + code), just ask and we go layer by layer.

