# Layer 5: Live — from backtest to production stream

A live quant system isn't a backtest; it consumes data *as it happens* and
makes decisions at runtime. This is where engineering (events, sockets,
loops) meets the model.

## Code map
- `qj/live/stream.py` — Binance WebSocket streaming client
- `qj/live/forecaster.py` — `LiveForecaster`: rolling window → prediction

## Two data models (REST vs WebSocket)
| REST (`qj/data`) | WebSocket (`qj/live`) |
|------------------|------------------------|
| You pull history | Server pushes new data |
| Good for backfill | Good for real-time |
| Request/response | Persistent connection |

Binance's `wss://stream.binance.com:9443/ws/<symbol>@kline_<interval>`
pushes a candle update as soon as it closes. We consume it as an async
iterator.

## LiveForecaster
Wraps the running candle history in a rolling `deque`. On each new closed
candle it:
1. rebuilds features (same `add_features` as training — feature contract
   must match exactly),
2. scores the latest bar with the trained model,
3. emits `prob_up` and a LONG/FLAT decision.

This mirrors exactly what the backtest did, but in real time.

## Try
```
uv run qj live BTCUSDT 1h
```
(This trains a quick model on cached history, warns if none, then prints a
live forecast per bar. Ctrl+C to stop.)

## Why this matters for "everything from tech to LLM"
The same pattern — stream → transform → model → act — underpins live LLM
serving, fraud detection, and every streaming ML system. Getting this
loop right (feature parity with training, no lookahead, robust reconnect)
is transferable engineering skill.
