# quant-journey

A full-stack systematic research pipeline: raw market data → feature engineering → forecasting models → backtest → live system → ML operations → LLM architecture. Built as a single walking path from first principles, engineered the way a production quant desk would build it — with no-lookahead discipline, honest evaluation, and drift monitoring throughout.

## One-line pitch

*An end-to-end, reproducible systematic-research stack that fetches and caches market data, engineers features, trains walk-forward forecasting models, backtests and simulates live signal generation, watches for feature drift in production, and re-implements a transformer decoder from scratch — evaluated honestly against buy-and-hold, not against a fantasy track record.*

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │                   DATA                       │
                │  qj/data/  (binance.py, store.py, base.py)  │
                │  source-agnostic OHLCV fetch + Parquet cache │
                └──────────────┬──────────────────────────────┘
                               │
                ┌──────────────▼──────────────────────────────┐
                │                 FEATURES                     │
                │  qj/features/  (engine.py)                  │
                │  returns, momentum, vol, z-score, extremes  │
                └──────────────┬──────────────────────────────┘
                               │
                ┌──────────────▼──────────────────────────────┐
                │                 MODELS                       │
                │  qj/models/  (forecast.py, walkforward.py)  │
                │  baselines → gradient boosting; walk-forward │
                │  IS/OOS with expanding windows (no lookahead)│
                └──────────────┬──────────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │            BACKTEST                  │
            │  qj/backtest/  (engine.py)          │
            │  position sizing, costs, Sharpe,    │
            │  max drawdown, win rate              │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │               LIVE                   │
            │  qj/live/  (stream.py, forecaster.py)│
            │  WebSocket klines + rolling forecast │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │              MLOPS                   │
            │  qj/mlops/  (drift.py, registry.py) │
            │  feature drift (KS test) + versioned │
            │  model registry / persistence        │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │               LLM                    │
            │  qj/llm/  (minigpt.py)              │
            │  attention, transformer decoder,    │
            │  from-scratch GPT-style trainer      │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │              BRAIN                   │
            │  qj/brain/  (alphas.py, neutralize.py,│
            │  evaluate.py)                        │
            │  alpha simulation, market neutral    │
            └─────────────────────────────────────┘
```

## Each layer

| Layer | Where | What it does |
|-------|-------|--------------|
| **Data** | `qj/data/` | `binance.py` fetches OHLCV from Binance; `store.py` caches to Parquet; `base.py` defines a source-agnostic `MarketDataFrame`, so the pipeline isn't bolted to one exchange. |
| **Features** | `qj/features/engine.py` | Log/simple returns, momentum, rolling volatility, z-scores, rolling min/max — deterministic, documented model inputs. |
| **Models** | `qj/models/forecast.py`, `walkforward.py` | Gradient-boosting classifiers; `walk_forward_predictions`/`walk_forward_evaluate` do expanding-window IS/OOS evaluation so results never peek over the validation boundary. |
| **Backtest** | `qj/backtest/engine.py` | Turns signals into long/flat positions at the *next* bar (`signal_to_position`), applies costs, and reports equity, Sharpe, max drawdown, win rate. |
| **Live** | `qj/live/stream.py`, `forecaster.py` | WebSocket kline streaming and a rolling forecast loop that rebuilds features from the live window — the research loop, ready to run against a real feed. |
| **MLOps** | `qj/mlops/drift.py`, `registry.py` | Kolmogorov–Smirnov feature-drift detection between training and live windows, plus a versioned model registry/persistence; `qj/live/forecaster.py` reports drift against the model's training distribution. |
| **LLM** | `qj/llm/minigpt.py` | A self-attention transformer decoder (multi-head attention, MLP, learned positional embeddings, training loop) implemented from scratch — the same architecture family used for time-series forecasting, built component-by-component. |
| **Brain** | `qj/brain/alphas.py`, `neutralize.py`, `evaluate.py` | BRAIN-style alpha simulation: score at bar *t* traded at bar *t+1* (no lookahead), plus cross-sectional neutralization (`neutralize_market`) and realized forward-return evaluation. |

## Results — honest evaluation, not a track record

Raw momentum on BTC hourly reaches roughly **~52% directional accuracy**, which after transaction costs **does not cleanly beat buy-and-hold**. We report this plainly. A credible quant process measures results with the same rigor it builds signals with, and this is evidence of just that:

- **No-lookahead discipline:** signals at bar *t* are only traded at bar *t+1* (in both backtest and live forecaster).
- **Walk-forward IS/OOS:** expanding-window evaluation with a rolling validation band — never a single naive split.
- **Transaction costs:** the backtest subtracts costs, so headline accuracy is not mistaken for edge.
- **Drift monitoring:** the live path compares the live feature distribution to the model's training distribution (KS test) and flags when the environment has moved.
- **Neutralization:** market exposure is neutralized cross-sectionally before evaluation.

The point of this repo is the **process**, not a backtest number. Disciplined data flow, honest metrics, and production-style machinery are what transfer to a real research desk.

## Getting started

```bash
# 1. Install dependencies (uv)
uv sync

# 2. Full test suite
uv run --extra ml python -m pytest qj/ -q

# 3. Fetch & cache market history
uv run qj fetch BTCUSDT 1h

# 4. Run the live forecast loop (WebSocket + drift check)
uv run qj live BTCUSDT 1h

# 5. Watch an LLM learn from scratch (optional)
uv run --extra llm python -m qj.llm.train_demo
```

## Badges

_These render once the repo is pushed; replace `USERNAME/REPONAME`._

![CI](https://github.com/USERNAME/REPONAME/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)

## Repo layout

```
qj/              core Python package (data, features, models, backtest, live, mlops, llm, brain)
01_data/ … 07_llm_arch/   annotated learning layers (each with notes/)
docs/RECRUITER_README.md  this recruiter-facing overview
data/           (gitignored) cached Parquet market data
```
